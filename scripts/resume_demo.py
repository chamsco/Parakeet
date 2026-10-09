"""Resume demo: does an interrupted run continue, or restart with warm weights?

    python scripts/resume_demo.py --quick     # ~40 s
    python scripts/resume_demo.py             # ~2 min

Round 13 found that ``train.py --resume`` restored **only the model**: the optimizer moments, the
EMA (which is the Reflow teacher and the better final weights), the discriminator, the LR schedule
position, the RNG and the batch order were all discarded.  A resumed run therefore restarted the
cosine schedule from its warmup peak and reshuffled its data -- the kind of defect that costs GPU
days and is invisible until the final quality is worse than it should be.

This script proves the fix on the *real* data path (fixture corpus -> latent cache ->
``LatentShardBatchSource``), not just on synthetic batches:

* run A: train N steps uninterrupted;
* run B: train N/2 steps, save, then resume in a **fresh model/optimizer/EMA/source** for the rest;
* the two must agree **bit-exactly**, and the LR must continue rather than restart.

Exactness is the only convincing evidence here: a "close enough" comparison would also pass if the
LR schedule restarted, because the warmup peak is not that far from the cosine value at step 60.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.data.dataset import LatentShardBatchSource, LatentShardDataset  # noqa: E402
from parakeet.data.features import cache_teacher_corpus, fit_latent_normalizer  # noqa: E402
from parakeet.data.teacher import build_backend, synthesize_corpus  # noqa: E402
from parakeet.models import build_model  # noqa: E402
from parakeet.train.stages import run_stage  # noqa: E402

PROMPTS = [
    "the quick brown fox jumps over the lazy dog",
    "a resumed run must continue the same trajectory",
    "checkpoints carry the optimizer and the schedule",
]


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def small_cfg(base, n_voices: int):
    cfg = copy.deepcopy(base)
    cfg.n_voices = max(1, n_voices)
    cfg.autoencoder.encoder_dims = [32, 48, 64]
    cfg.autoencoder.encoder_blocks = [1, 2, 2]
    cfg.autoencoder.decoder_dim = 64
    cfg.autoencoder.decoder_blocks = 2
    cfg.text.dim = 64
    cfg.text.n_layers = 2
    cfg.text.n_heads = 4
    cfg.flow.text_dim = 64
    cfg.flow.cond_dim = 64
    cfg.flow.dim = 64
    cfg.speaker.channels = [16, 24]
    cfg.speaker.emb_dim = 32
    cfg.speaker.style_dim = 32
    cfg.speaker.n_query = 4
    cfg.duration.hidden = 64
    cfg.train.lr = 1e-3
    cfg.train.warmup_steps = 10
    cfg.train.batch_size = 4
    cfg.train.log_every = 1
    return cfg.validate()


def build_cache(cfg, out: Path) -> Path:
    corpus = out / "corpus"
    synthesize_corpus(
        [t for t in PROMPTS for _ in range(2)],
        corpus,
        mix={"stub_low": 0.6, "stub_high": 0.4},
        voices={"stub_low": ["v0", "v1"], "stub_high": ["v0", "v1"]},
        backends={"stub_low": build_backend("stub_low"), "stub_high": build_backend("stub_high")},
    )
    import soundfile as sf

    records = [
        json.loads(l) for l in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    wavs = [torch.from_numpy(sf.read(str(corpus / r["wav_path"]), dtype="float32")[0]) for r in records]
    model = build_model(cfg)

    class _Source:
        def __init__(self, waves, batch_size, seed=0):
            self.waves, self.batch_size = waves, batch_size
            self.generator = torch.Generator().manual_seed(seed)

        def __call__(self):
            idx = torch.randint(len(self.waves), (self.batch_size,), generator=self.generator)
            picked = [self.waves[int(i)] for i in idx]
            width = max(p.numel() for p in picked)
            batch = torch.zeros(len(picked), width)
            for i, p in enumerate(picked):
                batch[i, : p.numel()] = p
            return {"wav": batch}

    source = _Source(wavs, cfg.train.batch_size, seed=cfg.train.seed)
    run_stage("autoencoder", cfg, model=model, batches=source, max_steps=40, out_dir=str(out / "ae"))
    fit_latent_normalizer(model.latent_norm, model.autoencoder, source, cfg, max_batches=4)
    return cache_teacher_corpus(corpus, out / "cache", cfg, model.autoencoder)


def train(
    cfg, cache_dir: Path, steps: int, run_dir: Path, resume_from: Optional[str] = None,
    save_every: int = 0, interrupt_after: Optional[int] = None,
):
    """One training run; returns (weights, loss curve, lr curve).

    ``interrupt_after`` simulates a crash *at* that step (after the periodic checkpoint for it has
    been written), which is what resume is actually for.  Note that ``steps`` -- the total budget --
    must be the same for the interrupted and the uninterrupted run: the cosine schedule is
    parameterised by the total, so a different budget is a different schedule, not a resume.
    """
    run_cfg = copy.deepcopy(cfg)
    run_cfg.train.save_every = save_every
    dataset = LatentShardDataset(cache_dir)
    source = LatentShardBatchSource(dataset, batch_size=run_cfg.train.batch_size, seed=0)
    model = build_model(run_cfg)
    losses: List[float] = []
    lrs: List[float] = []
    steps_seen: List[int] = []

    class _Crash(Exception):
        pass

    def record(logs):
        # the step-0 provenance banner has no 'lr'; only real training steps are recorded
        if "lr" not in logs or "loss" not in logs or logs.get("loss") != logs.get("loss"):
            return
        losses.append(float(logs["loss"]))
        lrs.append(float(logs["lr"]))
        steps_seen.append(int(logs["step"]))
        if interrupt_after is not None and int(logs["step"]) > interrupt_after:
            raise _Crash(f"simulated crash after step {interrupt_after}")

    try:
        run_stage(
            "distill-text", run_cfg, model=model, batches=source, max_steps=steps,
            out_dir=str(run_dir), log_fn=record, resume_from=resume_from,
        )
    except _Crash as exc:
        print(f"  !! {exc}")
    weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return weights, {"loss": losses, "lr": lrs, "step": steps_seen}


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify checkpoint/resume on the real data path")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--out", default="runs/resume_demo")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps = 40

    base = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    imports = load_config(args.config)
    cfg = small_cfg(imports, n_voices=2)
    _banner(f"build a fixture corpus + latent cache ({args.steps} steps per run)")
    cache = build_cache(cfg, out)
    print(f"  -> cache {cache} ({len(LatentShardDataset(cache))} items)")

    # ---------------- run A: uninterrupted ----------------
    _banner("run A: uninterrupted")
    torch.manual_seed(cfg.train.seed)
    weights_a, curve_a = train(cfg, cache, args.steps, out / "full")
    print(f"  -> {len(curve_a['loss'])} logged points | first loss {curve_a['loss'][0]:.4f} | "
          f"last loss {curve_a['loss'][-1]:.4f} | lr {curve_a['lr'][0]:.2e} -> "
          f"{curve_a['lr'][-1]:.2e}")

    # ---------------- run B: crash at half, resume in a fresh everything ----------------
    half = max(1, args.steps // 2)
    _banner(f"run B: crash after step {half}, then resume to {args.steps} from a fresh model")
    torch.manual_seed(cfg.train.seed)
    _weights_b1, curve_b1 = train(
        cfg, cache, args.steps, out / "part1", save_every=half, interrupt_after=half
    )
    ckpt = out / "part1" / f"distill-text_step{half}.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    components = sorted(k for k in payload if k != "model")
    print(f"  -> checkpoint {ckpt.name} contains: {components}")
    torch.manual_seed(12345)  # deliberately different process state before resuming
    weights_b, curve_b2 = train(
        cfg, cache, args.steps, out / "part2", resume_from=str(ckpt), save_every=0
    )

    # ---------------- compare ----------------
    _banner("comparison")
    worst = max(
        float((weights_a[k] - weights_b[k]).abs().max()) for k in weights_a
    )
    # the resumed run logs only steps half+1..steps, so line the curves up by step
    after = {s: l for s, l in zip(curve_b2["step"], curve_b2["loss"])}
    after_lr = {s: l for s, l in zip(curve_b2["step"], curve_b2["lr"])}
    before = {s: l for s, l in zip(curve_a["step"], curve_a["loss"])}
    before_lr = {s: l for s, l in zip(curve_a["step"], curve_a["lr"])}
    shared = sorted(set(after) & set(before))
    loss_gap = max(abs(after[s] - before[s]) for s in shared) if shared else float("inf")
    lr_gap = max(abs(after_lr[s] - before_lr[s]) for s in shared) if shared else float("inf")
    jump = abs(curve_b2["loss"][0] - curve_b1["loss"][-1]) if curve_b1["loss"] else float("inf")

    print(f"  max parameter difference between the two runs: {worst:.3e}")
    print(f"  max loss difference over {len(shared)} shared steps: {loss_gap:.3e}")
    print(f"  max lr difference over {len(shared)} shared steps:   {lr_gap:.3e}")
    print(f"  loss across the interruption (last before -> first after): "
          f"{curve_b1['loss'][-1]:.4f} -> {curve_b2['loss'][0]:.4f}")

    # Control that isolates the resume: the two runs must be identical *before* the interruption.
    # If they are not, the demo's own seeding is at fault and any statement about resume is void.
    pre_a = {s: l for s, l in zip(curve_a["step"], curve_a["loss"]) if s <= half}
    pre_b = {s: l for s, l in zip(curve_b1["step"], curve_b1["loss"])}
    shared_pre = sorted(set(pre_a) & set(pre_b))
    pre_gap = max(abs(pre_a[s] - pre_b[s]) for s in shared_pre) if shared_pre else float("inf")
    print(f"  before the interruption ({len(shared_pre)} steps): max loss difference {pre_gap:.3e}")

    checks = {
        "checkpoint_has_all_components": all(
            k in components for k in ("optimizer", "ema", "rng", "step", "config")
        ),
        "pre_interruption_runs_match": pre_gap < 1e-9,
        "resume_is_bit_exact": worst == 0.0,
        "losses_match_the_uninterrupted_run": loss_gap < 1e-9,
        "lr_continues_instead_of_restarting": lr_gap < 1e-12,
        "no_loss_jump_at_the_interruption": jump < 0.05,
    }
    report = {
        "config": args.config,
        "steps": args.steps,
        "checkpoint_components": components,
        "max_parameter_difference": worst,
        "max_loss_difference": loss_gap,
        "max_lr_difference": lr_gap,
        "pre_interruption_loss_difference": pre_gap,
        "loss_at_interruption": [curve_b1["loss"][-1], curve_b2["loss"][0]],
        "curves": curve_a,
        "checks": checks,
        "seconds_total": time.perf_counter() - t_start,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _banner("RESULT")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nreport -> {out/'report.json'}")
    print("RESUME DEMO " + ("PASSED" if all(checks.values()) else "FAILED"))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
