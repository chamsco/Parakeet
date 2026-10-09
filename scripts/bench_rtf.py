"""Benchmark real-time factor on one CPU thread (the headline deployment metric).

    python scripts/bench_rtf.py --config configs/parakeet_tiny.yaml --steps 2

Reports RTF and "x real time" as Paradee and the public CPU benchmarks do, so the numbers are
directly comparable: Paradee 25.0x (17.8x ONNX) on one CPU thread; Supertonic-3 RTF 0.313 at
5 steps / 0.165 at 2 steps; Kokoro-82M RTF ~0.47.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.eval.metrics import measure_rtf  # noqa: E402
from parakeet.inference import Synthesizer, quantize_weights_, size_report  # noqa: E402
from parakeet.models import build_model, count_parameters  # noqa: E402

TEXTS = [
    "Parakeet is a small and fast text to speech model.",
    "The quick brown fox jumps over the lazy dog, twice, for good measure.",
    "Real time factor is the only number that matters on a laptop.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--steps", type=int, default=None, help="flow sampler NFE")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--int8", action="store_true", help="weight-only int8 before timing")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps:
        cfg.flow.distilled_nfe = args.steps
    steps = args.steps or cfg.flow.distilled_nfe

    if args.checkpoint:
        synth = Synthesizer.from_checkpoint(args.checkpoint, device="cpu")
        cfg = synth.cfg
        model = synth.model
    else:
        model = build_model(cfg)
        synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=True)
    if args.int8:
        quantize_weights_(model, bits=8, per_channel=True)

    sizes = size_report(model)
    print(f"{cfg.name}: {count_parameters(model)/1e6:.3f}M params | fp32 {sizes['fp32_mb']:.2f} MB "
          f"| int8 {sizes['int8_mb']:.2f} MB | NFE {steps}")

    rows = []
    for text in TEXTS:
        res = measure_rtf(
            lambda t=text: synth.synthesize(t, steps=steps, seed=0),
            sample_rate=cfg.audio.sample_rate,
            warmup=1,
            runs=args.runs,
            threads=args.threads,
        )
        rows.append({"text": text, **res.__dict__})
        print(f"  RTF {res.rtf:7.3f} | {res.seconds_per_audio_second:6.2f}x real time | "
              f"{res.audio_seconds:5.2f}s audio | {res.samples_per_second:9.0f} samples/s | {text[:42]!r}")

    summary = {
        "config": args.config,
        "variant": cfg.variant,
        "params": count_parameters(model),
        "nfe": steps,
        "threads": args.threads,
        "int8": bool(args.int8),
        "mean_rtf": sum(r["rtf"] for r in rows) / len(rows),
        "mean_x_realtime": sum(r["seconds_per_audio_second"] for r in rows) / len(rows),
        "size_report": sizes,
        "runs": rows,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    print(f"MEAN RTF {summary['mean_rtf']:.3f} = {summary['mean_x_realtime']:.2f}x real time on {args.threads} thread(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
