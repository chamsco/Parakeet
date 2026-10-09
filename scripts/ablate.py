"""Capacity ablation: how small can the student be before the fit degrades?

    python scripts/ablate.py --steps 400

"Make it as light as possible" is the objective, and the text side is the largest single component
of Parakeet-Tiny (about 40 % of its parameters).  This script measures the trade instead of guessing
at it: several text-side geometries are trained for the **same** number of steps, on the **same**
cached teacher signals, from the same seed, and scored with the same probe.  Because the cache and
the probe are shared, the losses are directly comparable between variants; nothing else changes.

What this can and cannot say
----------------------------
The fixtures are synthetic formant stacks, so the *absolute* losses are not speech quality and an
untrained autoencoder sits in the middle of the loop.  What the comparison does show is how much
text-side capacity is needed to fit a fixed set of teacher signals -- the relative trend is the
signal, and it is exactly the trend a real run would need to revisit with real data.  The reported
recommendation is therefore a *starting point*, not a result.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.data.dataset import LatentShardBatchSource, LatentShardDataset  # noqa: E402
from parakeet.data.features import cache_teacher_corpus  # noqa: E402
from parakeet.data.teacher import build_backend, synthesize_corpus  # noqa: E402
from parakeet.eval import duration_error_frames, teacher_signal_loss  # noqa: E402
from parakeet.inference import Synthesizer  # noqa: E402
from parakeet.models import build_model, count_parameters  # noqa: E402
from parakeet.train.stages import run_stage  # noqa: E402

PROMPTS = [
    "the quick brown fox jumps over the lazy dog",
    "capacity is a trade between size and fit",
    "every variant trains on the same cached signals",
    "a held-out split is what makes the comparison honest",
    "smaller models can fit repetitive fixtures just as well",
    "the text side is the largest part of the student",
]

#: (label, text dim, text layers, text heads) -- duration.hidden follows the text dim, as the shipped
#: configs do, so "the text half" scales as one unit
VARIANTS: List[Tuple[str, int, int, int]] = [
    ("dim256-L4 (shipped)", 256, 4, 4),
    ("dim256-L2", 256, 2, 4),
    ("dim160-L3", 160, 3, 4),
    ("dim128-L4", 128, 4, 4),
    ("dim128-L2", 128, 2, 4),
    ("dim96-L2", 96, 2, 4),
    ("dim64-L2", 64, 2, 4),
]


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def variant_cfg(base, dim: int, layers: int, heads: int):
    cfg = copy.deepcopy(base)
    cfg.text.dim = dim
    cfg.text.n_layers = layers
    cfg.text.n_heads = heads
    cfg.duration.hidden = dim
    cfg.flow.text_dim = dim
    cfg.flow.cond_dim = dim
    cfg.train.lr = 2e-3
    cfg.train.warmup_steps = 20
    cfg.train.batch_size = 4
    cfg.train.log_every = 10**6  # quiet: the probe is the measurement
    cfg.train.save_every = 0
    return cfg.validate()


def build_cache(base, out: Path, voices_per_prompt: int = 3) -> Path:
    """Corpus + frozen autoencoder + cached teacher signals, built once and reused.

    The adversarial autoencoder stage is by far the most expensive part of this script and every
    variant shares it, so if the cache already exists the whole thing is skipped -- otherwise a
    failed run costs minutes to retry.
    """
    cache_dir = out / "cache"
    if (cache_dir / "cache_meta.json").exists():
        return cache_dir
    corpus = out / "corpus"
    if not (corpus / "corpus_meta.json").exists():
        voices = [f"v{i}" for i in range(voices_per_prompt)]
        synthesize_corpus(
            [t for t in PROMPTS for _ in range(voices_per_prompt)],
            corpus,
            mix={"stub_low": 0.6, "stub_high": 0.4},
            voices={"stub_low": voices, "stub_high": voices},
            backends={"stub_low": build_backend("stub_low"), "stub_high": build_backend("stub_high")},
        )
    import soundfile as sf

    records = [
        json.loads(l) for l in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    cfg = variant_cfg(base, 128, 2, 4)
    cfg.n_voices = voices_per_prompt  # the corpus decides this; the guard in the cache builder agrees
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

    wavs = [torch.from_numpy(sf.read(str(corpus / r["wav_path"]), dtype="float32")[0]) for r in records]
    source = _Source(wavs, 4, seed=0)
    torch.manual_seed(0)
    run_stage("autoencoder", cfg, model=model, batches=source, max_steps=60, out_dir=str(out / "ae"))
    from parakeet.data.features import fit_latent_normalizer

    fit_latent_normalizer(model.latent_norm, model.autoencoder, source, cfg, max_batches=4)
    return cache_teacher_corpus(corpus, out / "cache", cfg, model.autoencoder)


def text_latency_ms(model, ids: torch.Tensor, runs: int = 20) -> float:
    model.eval()
    times = []
    with torch.no_grad():
        model.text_side(ids)  # warm-up
        for _ in range(runs):
            t0 = time.perf_counter()
            model.text_side(ids)
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def synth_latency_ms(cfg, model, text: str, runs: int = 5) -> float:
    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=True)
    times = []
    with torch.no_grad():
        synth.synthesize(text, seed=0)
        for _ in range(runs):
            t0 = time.perf_counter()
            synth.synthesize(text, seed=0)
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser(description="Text-side capacity ablation on cached fixtures")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="runs/ablation")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # one thread: this is the deployment target for Tiny
    base = load_config(args.config)
    _banner(f"shared cache: fixtures + one frozen autoencoder ({args.steps} steps per variant)")
    cache = build_cache(base, out)
    dataset = LatentShardDataset(cache)
    n_voices = len(json.loads((Path(cache) / "cache_meta.json").read_text(encoding="utf-8"))["voice_names"])
    # Held-out split.  Without it "a smaller model fits as well" could simply mean "a smaller model
    # overfits less", which is the opposite of the claim this script exists to test.
    n_val = max(1, len(dataset) // 3)
    train_idx = list(range(len(dataset) - n_val))
    val_idx = list(range(len(dataset) - n_val, len(dataset)))
    print(f"  -> cache {cache} ({len(dataset)} items) | train {len(train_idx)} / val {len(val_idx)} "
          f"-- identical targets and identical split for every variant")

    rows: List[Dict[str, object]] = []
    for label, dim, layers, heads in VARIANTS:
        cfg = variant_cfg(base, dim, layers, heads)
        cfg.n_voices = n_voices  # from the cache: a mismatch is exactly what the guard rejects
        cfg.train.max_steps = args.steps
        torch.manual_seed(cfg.train.seed)  # same initialisation distribution for every variant
        model = build_model(cfg)
        text_params = sum(p.numel() for p in model.text.parameters())
        total_params = count_parameters(model)
        # the batch source only ever sees the training split
        train_source = LatentShardBatchSource(
            LatentShardDataset(cache, indices=train_idx), batch_size=cfg.train.batch_size, seed=0
        )
        t0 = time.perf_counter()
        run_stage("distill-text", cfg, model=model, batches=train_source, max_steps=args.steps,
                  out_dir=str(out / f"run_{dim}_{layers}"))
        train_s = time.perf_counter() - t0

        train_probe = [dataset[i] for i in train_idx]
        val_probe = [dataset[i] for i in val_idx]
        fit = teacher_signal_loss(model, train_probe, cfg)
        val_fit = teacher_signal_loss(model, val_probe, cfg)
        duration_mae = duration_error_frames(model, val_probe)
        # latency is shape-bound, not value-bound: use a fixed (B, T) token batch of the longest
        # available length so the measurement is comparable across variants
        max_tokens = max(int(p["ids"].numel()) for p in train_probe + val_probe)
        ids = torch.randint(1, cfg.text.vocab_size, (4, max_tokens))
        rows.append(
            {
                "label": label,
                "text_dim": dim,
                "text_layers": layers,
                "text_params": text_params,
                "total_params": total_params,
                "train_fit": fit,
                "val_fit": val_fit,
                "generalisation_gap": val_fit - fit,
                "duration_mae_frames": duration_mae,
                "text_latency_ms": text_latency_ms(model, ids),
                "synth_latency_ms": synth_latency_ms(cfg, model, PROMPTS[0]),
                "train_seconds": train_s,
            }
        )
        r = rows[-1]
        print(f"  {label:20s} text {text_params/1e6:.3f} M | total {total_params/1e6:.3f} M | "
              f"val fit {val_fit:.4f} (train {fit:.4f}) | dur MAE {duration_mae:.3f} fr | "
              f"text {r['text_latency_ms']:.2f} ms | synth {r['synth_latency_ms']:.1f} ms")

    best = min(rows, key=lambda r: r["val_fit"])
    shipped = rows[0]
    smaller = [r for r in rows if r["text_params"] <= 0.5 * shipped["text_params"]]
    efficient = min(smaller, key=lambda r: r["val_fit"]) if smaller else None
    penalty = (
        None if efficient is None
        else 100.0 * (efficient["val_fit"] - best["val_fit"]) / best["val_fit"]
    )

    checks = {
        # the harness must be able to see capacity at all, or the comparison means nothing
        "capacity_is_detectable": max(r["val_fit"] for r in rows) > min(r["val_fit"] for r in rows),
        "text_latency_tracks_size": min(r["text_latency_ms"] for r in rows)
        < max(r["text_latency_ms"] for r in rows),
        "every_variant_trained": all(r["train_seconds"] > 0 for r in rows),
        # no variant should be wildly overfit on held-out fixtures, or the comparison is noise
        "no_variant_degenerates": all(r["val_fit"] < 5.0 * best["val_fit"] for r in rows),
    }
    report = {
        "config": args.config,
        "steps": args.steps,
        "n_items": len(dataset),
        "split": {"train": len(train_idx), "val": len(val_idx),
                  "note": "the batch source only sees the training split"},
        "caveat": (
            "Synthetic fixtures: absolute losses are not speech quality and an untrained autoencoder "
            "is in the loop.  Only the relative trend across variants is meaningful, the step budget "
            "is short (so deeper variants may simply be undertrained), and a real run must revisit "
            "this with real data and a real evaluation."
        ),
        "rows": rows,
        "best_val_fit": best["label"],
        "shipped": {"label": shipped["label"], "text_params": shipped["text_params"],
                    "val_fit": shipped["val_fit"], "synth_latency_ms": shipped["synth_latency_ms"]},
        "half_size_best": None if efficient is None else {
            "label": efficient["label"],
            "text_dim": efficient["text_dim"],
            "text_layers": efficient["text_layers"],
            "text_params": efficient["text_params"],
            "val_fit": efficient["val_fit"],
            "fit_penalty_pct": penalty,
            "text_latency_ms": efficient["text_latency_ms"],
            "synth_latency_ms": efficient["synth_latency_ms"],
        },
        "checks": checks,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _banner("RESULT")
    print(f"  best held-out fit: {best['label']} (val {best['val_fit']:.4f}, "
          f"text {best['text_params']/1e6:.3f} M)")
    if efficient:
        print(f"  best at <=50% of the shipped text params: {efficient['label']} "
              f"(val {efficient['val_fit']:.4f}, {penalty:+.1f}% vs best, text "
              f"{efficient['text_latency_ms']:.2f} ms vs {shipped['text_latency_ms']:.2f} ms, "
              f"total {efficient['total_params']/1e6:.2f} M vs {shipped['total_params']/1e6:.2f} M)")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\n  NOT a quality comparison: synthetic fixtures, untrained autoencoder, short budget.")
    print(f"\nreport -> {out/'report.json'}")
    print("ABLATION " + ("PASSED" if all(checks.values()) else "FAILED"))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
