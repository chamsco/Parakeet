"""Evaluate a whole series of flow checkpoints and report the trajectory.

    python scripts/flow_trajectory.py --run runs/flow_v2 --limit 8 --steps 4

A single checkpoint's number cannot distinguish "learning" from "stuck", and this project has twice
spent hours on a run whose progress was invisible (round 28 had no intermediate checkpoints at all).
This walks every saved checkpoint in step order and evaluates the three quantities that matter for the
flow, in the order they must come right:

1. **length ratio** — generated seconds against reference seconds.  A flow that predicts a near-zero
   length cannot be intelligible no matter what else is right (measured: 0.01 before the round-32 fix,
   1.67 after);
2. **log-mel cosine** — the acoustic proxy, computable only once (1) is roughly right;
3. **WER with a control** on real speech, withheld if the control fails.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def step_of(path: Path) -> int:
    match = re.search(r"step(\d+)", path.name)
    if match:
        return int(match.group(1))
    return 10**9  # `flow_last.pt`: the final weights, sorts last


def sampled_latent_correlation(checkpoint: Path, config: str, cache: Path, items: int = 6):
    """Per-dim correlation between the flow's *sampled* latents and the teacher's, for the same text.

    Round 37 calibrated what this number must be: a decoded latent is read perfectly at rho >= 0.75 and
    fails at 0.60.  Correlating samples is therefore a *predictive* metric for the flow -- it says whether
    intelligibility is reachable before any audio exists, and unlike the mel-envelope proxy it cannot be
    satisfied by noise.  Runs in a subprocess so a checkpoint that cannot be sampled does not take the
    report down with it.
    """
    code = f'''
import sys
sys.path.insert(0, r"{ROOT}")
import torch
from pathlib import Path
from parakeet.config import load_config
from parakeet.data.dataset import LatentShardDataset
from parakeet.models import build_model
from parakeet.models.flow import consistency_sample, unfold_time
from parakeet.train.common import apply_checkpoint_geometry, load_latent_norm_from_cache

cfg = load_config(r"{config}")
payload = torch.load(r"{checkpoint}", map_location="cpu", weights_only=False)
state = (payload.get("ema") or {{}}).get("shadow") or payload["model"]
applied = apply_checkpoint_geometry(cfg, state)
model = build_model(cfg)
current = model.state_dict()
model.load_state_dict({{k: v for k, v in state.items()
                       if k in current and tuple(current[k].shape) == tuple(v.shape)}}, strict=False)
load_latent_norm_from_cache(model, r"{cache}")
model.eval()
dataset = LatentShardDataset(r"{cache}")
scores = []
with torch.no_grad():
    for index in range(min({items}, len(dataset))):
        item = dataset[index]
        ids = item["ids"][None]
        mask = item.get("text_mask")
        mask = mask[None] if mask is not None else None
        voice = item.get("voice")
        voice = voice.reshape(1) if voice is not None else None
        memory, memory_mask, cond = model.conditions(ids, mask, voice=voice)
        frames = int(model.predict_latent_frames(model.text(ids, mask), mask, cond).item())
        tc = model.compressed_frames(frames)
        x1c = consistency_sample(model.vf, memory, memory_mask,
                                 (1, cfg.flow.latent_dim * cfg.flow.compress, tc),
                                 steps=4, device=torch.device("cpu"), cfg_scale=cfg.flow.cfg_scale)
        sampled = unfold_time(x1c, cfg.flow.compress, t_out=frames)
        target = item["latent"][None]
        length = min(sampled.shape[-1], target.shape[-1])
        per_dim = []
        for dim in range(sampled.shape[1]):
            x, y = sampled[0, dim, :length], target[0, dim, :length]
            if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
                continue
            per_dim.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
        if per_dim:
            scores.append(sum(per_dim) / len(per_dim))
print("RESULT", sum(scores) / max(1, len(scores)) if scores else "nan")
'''
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=900, cwd=str(ROOT))
    for line in (done.stdout or "").splitlines():
        if line.startswith("RESULT"):
            try:
                value = float(line.split()[1])
                return value if value == value else None
            except (IndexError, ValueError):
                return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Flow checkpoint trajectory")
    ap.add_argument("--run", default="runs/flow_v2")
    ap.add_argument("--config", default="configs/parakeet_flow.yaml")
    ap.add_argument("--corpus", default="data/gutenberg_corpus/corpus")
    ap.add_argument("--manifest", default="val.jsonl")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4, help="sampler NFE")
    ap.add_argument("--reference", action="store_true",
                        help="condition on a partner mel from the same voice: the flow is trained that way")
    ap.add_argument("--out", default=None)
    ap.add_argument("--latent-cache", default=None,
                    help="cache whose teacher latents to correlate the samples against; rho >= 0.75 is the"
                         " measured threshold for intelligibility (round 37)")
    args = ap.parse_args()

    checkpoints = sorted(Path(args.run).glob("flow_*.pt"), key=step_of)
    if not checkpoints:
        print(f"no checkpoints in {args.run}")
        return 2
    out = Path(args.out) if args.out else Path(args.run) / "trajectory.json"

    rows = []
    for checkpoint in checkpoints:
        report = Path(args.run) / f"trajectory_step{step_of(checkpoint)}.json"
        command = [
            sys.executable, "scripts/real_eval.py",
            "--checkpoint", str(checkpoint),
            "--config", args.config,
            "--corpus", args.corpus,
            "--manifest", args.manifest,
            "--prose-only",
            "--limit", str(args.limit),
            "--steps", str(args.steps),
                        *(["--reference"] if args.reference else []),
            "--out", str(report.parent / f"eval_{report.stem}"),
        ]
        done = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
        payload_path = report.parent / f"eval_{report.stem}" / "report.json"
        if done.returncode != 0 or not payload_path.exists():
            print(f"  step {step_of(checkpoint):>6}: evaluation failed")
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        latent_correlation = None
        if args.latent_cache:
            # the calibrated metric: rho >= 0.75 is what intelligibility needs (round 37), and unlike the
            # mel proxy it cannot be satisfied by noise
            latent_correlation = sampled_latent_correlation(
                checkpoint, args.config, Path(args.latent_cache)
            )
        row = {
            "step": step_of(checkpoint),
            "length_ratio": payload["synthesis"].get("length_ratio"),
            "log_mel_cosine": payload["synthesis"].get("log_mel_cosine_vs_reference"),
            "x_realtime": payload["synthesis"].get("x_realtime"),
            "wer": payload["wer"]["student"],
            "wer_control": payload["wer"]["teacher"],
            "dnsmos": payload["naturalness"]["student"],
            "sampled_latent_correlation": latent_correlation,
        }
        rows.append(row)
        print(f"  step {row['step']:>6}: length {row['length_ratio']:.2f} | "
              f"cosine {row['log_mel_cosine'] if row['log_mel_cosine'] is None else round(row['log_mel_cosine'], 4)} | "
              f"latent rho {latent_correlation if latent_correlation is None else round(latent_correlation, 3)} "
              f"(target 0.75) | WER {row['wer']:.3f} (control {row['wer_control']:.3f}) | "
              f"DNSMOS {row['dnsmos']:.3f} | {row['x_realtime']:.0f}x real time")

    if not rows:
        return 2
    best = min(rows, key=lambda r: (r["wer"] if r["wer"] is not None else 9.9))
    summary = {
        "run": args.run,
        "sampler_nfe": args.steps,
        "utterances": args.limit,
        "checkpoints": len(rows),
        "rows": rows,
        "best_by_wer": best,
        "verdict": (
            "still at chance on every checkpoint: the flow's length is right, its acoustic proxy is "
            "well above zero, and it is still not intelligible"
            if all((r["wer"] or 1.0) >= 0.9 for r in rows)
            else "at least one checkpoint is intelligible -- see the rows"
        ),
    }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  {summary['verdict']}")
    print(f"trajectory -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
