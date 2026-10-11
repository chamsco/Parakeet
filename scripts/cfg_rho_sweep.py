"""Does classifier-free guidance buy anything on a weakly-conditioned flow?

The measured situation: the velocity field's own objective is only 2-7 % text-dependent and its samples correlate
~0 with the right latent.  The field already *has* the text at every step and CFG (config default 1.5) is
implemented, so the question is narrow and standard: **if the text direction is merely under-weighted,
extrapolating along it should raise the correlation with the target** -- and if it does not, the text direction is
not a direction that points at the target at any gain.

Inference-only, so it is cheap: one checkpoint load, then sampling at several guidance weights.

    python scripts/cfg_rho_sweep.py --run runs/ab_fresh --checkpoint flow_step2000.pt \
        --cache runs/expanded_v3 [--items 64] [--steps 4] [--weights 1,1.5,2,4,8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from parakeet.config import load_config
from parakeet.data.dataset import LatentShardDataset
from parakeet.models import build_model
from parakeet.models.flow import consistency_sample
from parakeet.models.parakeet import unfold_time
from parakeet.train.common import apply_checkpoint_geometry, derive_n_voices_from_cache, load_latent_norm_from_cache


def frame_rho(sampled: torch.Tensor, target: torch.Tensor) -> float:
    """Mean per-dimension correlation between a sampled frame latent and its target."""
    length = min(sampled.shape[-1], target.shape[-1])
    scores = []
    for dim in range(sampled.shape[0]):
        x, y = sampled[dim, :length].float(), target[dim, :length].float()
        if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
            continue
        scores.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
    return sum(scores) / max(1, len(scores))


def main() -> int:
    ap = argparse.ArgumentParser(description="rho against classifier-free guidance weight")
    ap.add_argument("--run", required=True, help="directory holding the checkpoint")
    ap.add_argument("--checkpoint", default="flow_last.pt")
    ap.add_argument("--config", default="configs/parakeet_flow_plan.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--items", type=int, default=64)
    ap.add_argument("--probe-items", type=int, default=12)
    ap.add_argument("--steps", type=int, default=4, help="NFE")
    ap.add_argument("--weights", default="1,1.5,2,4,8")
    args = ap.parse_args()

    path = Path(args.run) / args.checkpoint
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path}")

    cfg = load_config(args.config)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    apply_checkpoint_geometry(cfg, state)
    cfg.n_voices = derive_n_voices_from_cache(args.cache) or 1
    model = build_model(cfg)
    current = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items()
                           if k in current and tuple(current[k].shape) == tuple(v.shape)}, strict=False)
    load_latent_norm_from_cache(model, args.cache)
    model.eval()

    dataset = LatentShardDataset(args.cache, indices=list(range(args.items)))
    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    print(f"checkpoint {path} (step {payload.get('step')}) | NFE {args.steps} | {len(dataset)} items")
    print(f"{'cfg':>6s} {'frame rho':>11s} {'peak':>8s}")

    rows = []
    with torch.no_grad():
        for weight in weights:
            frames_rho, peaks = [], []
            for index in range(min(args.probe_items, len(dataset))):
                item = dataset[index]
                ids = item["ids"][None]
                mask = item.get("text_mask")
                mask = mask[None] if mask is not None else None
                voice = item.get("voice")
                voice = voice.reshape(1) if voice is not None else None
                memory, memory_mask, cond = model.conditions(ids, mask, voice=voice)
                n_frames = int(model.predict_latent_frames(model.text(ids, mask), mask, cond).item())
                tc = model.compressed_frames(n_frames)
                generator = torch.Generator().manual_seed(0)
                x0 = torch.randn(1, cfg.flow.latent_dim * cfg.flow.compress, tc, generator=generator)
                x1c = consistency_sample(
                    model.vf, memory, memory_mask,
                    (1, cfg.flow.latent_dim * cfg.flow.compress, tc),
                    steps=args.steps, device=torch.device("cpu"), cfg_scale=weight, x0=x0,
                )
                sampled = unfold_time(x1c, cfg.flow.compress, t_out=n_frames)
                frames_rho.append(frame_rho(sampled[0], item["latent"]))
                peaks.append(float(sampled.abs().max()))
            frame_mean = sum(frames_rho) / len(frames_rho)
            rows.append({"cfg": weight, "frame_rho": frame_mean, "peak": max(peaks)})
            print(f"{weight:6.2f} {frame_mean:+11.4f} {max(peaks):8.2f}", flush=True)

    Path("runs/cfg_rho_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    best = max(rows, key=lambda r: r["frame_rho"])
    print(f"\nbest frame rho {best['frame_rho']:+.4f} at cfg {best['cfg']} "
          f"(target 0.75 to be intelligible)")
    if best["frame_rho"] <= 0.05:
        print("  -> guidance does not point the samples at the target at any weight: the direction the field "
              "learned from the text is not the direction that identifies this utterance")
    else:
        print("  -> guidance moves the samples toward the target; sweep further before concluding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
