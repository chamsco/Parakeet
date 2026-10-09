"""Profile where a full Tiny synthesis actually spends its time.

    python scripts/profile_pipeline.py --config configs/parakeet_tiny.yaml

"Why is the model still 20x real time when the decoder alone is 90x?"  This answers it.  It is
easy to optimise the wrong component: for the Tiny model the vocoder is only ~40 % of the budget,
with the text side taking a comparable share, so exporting the vocoder to int8 alone cannot be the
whole story.  Run this before optimising anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.inference import Synthesizer, phase_lock  # noqa: E402
from parakeet.models import build_model  # noqa: E402

TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "parakeet is a small and fast text to speech model",
    "real time factor is the number that matters on a laptop",
]


def timeit(fn: Callable[[], object], runs: int = 5) -> Tuple[float, object]:
    out = fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        out = fn()
    return (time.perf_counter() - t0) / runs * 1000.0, out


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile the Tiny synthesis pipeline")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    model = build_model(cfg).eval()
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("ema", {}).get("shadow", payload["model"]), strict=False)
        print(f"loaded {args.checkpoint}")
    else:
        print("WARNING: random weights -- timings are representative, output is not")

    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=False)
    rows: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for text in TEXTS:
            ids, mask, _ = synth.prepare_text(text)
            t_side, side = timeit(lambda: model.text_side(ids, mask), args.runs)
            durations = side["log_duration"].exp().round().clamp_min(1).long()
            t_latent, lat_out = timeit(
                lambda: model.decoder_latent_from_tokens(
                    side["latent_token"], durations, side["f0"], side["energy"]
                ),
                args.runs,
            )
            latent, _ = lat_out  # (latent, frame_mask)
            t_decoder, wav = timeit(lambda: model.autoencoder.decode(latent), args.runs)
            t_lock, _ = timeit(
                lambda: phase_lock(wav, sample_rate=cfg.audio.sample_rate,
                                   n_fft=cfg.audio.n_fft, hop_length=cfg.audio.hop_length),
                args.runs,
            )
            t_full, wav_full = timeit(
                lambda: synth.synthesize(text, seed=0), max(1, args.runs // 2)
            )

            def shipped() -> torch.Tensor:
                """The configuration we actually deploy: synthesis + phase-lock filter."""
                wav = synth.synthesize(text, seed=0)
                return phase_lock(
                    wav,
                    sample_rate=cfg.audio.sample_rate,
                    n_fft=cfg.audio.n_fft,
                    hop_length=cfg.audio.hop_length,
                )

            t_full_locked, _ = timeit(shipped, max(1, args.runs // 2))
            audio_s = wav_full.shape[-1] / cfg.audio.sample_rate
            rows[text] = {
                "audio_seconds": audio_s,
                "latent_frames": float(durations.sum()),
                "text_side_ms": t_side,
                "latent_build_ms": t_latent,
                "decoder_ms": t_decoder,
                "phase_lock_ms": t_lock,
                "full_ms": t_full,
                "full_with_filter_ms": t_full_locked,
                "full_rtf": (t_full / 1000.0) / max(audio_s, 1e-9),
                "full_x_realtime": audio_s / max(t_full / 1000.0, 1e-9),
                "shipped_x_realtime": audio_s / max(t_full_locked / 1000.0, 1e-9),
            }

    keys = ["text_side_ms", "latent_build_ms", "decoder_ms", "phase_lock_ms"]
    means = {k: sum(r[k] for r in rows.values()) / len(rows) for k in keys}
    full = sum(r["full_ms"] for r in rows.values()) / len(rows)
    full_shipped = sum(r["full_with_filter_ms"] for r in rows.values()) / len(rows)
    audio = sum(r["audio_seconds"] for r in rows.values()) / len(rows)
    accounted = sum(means.values())

    print(f"\n{cfg.name} | {args.threads} thread(s) | mean audio {audio:.3f}s | "
          f"{cfg.flow.distilled_nfe} NFE | {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    print(f"{'component':<26}{'ms':>8}{'% of shipped':>14}{'x realtime':>12}")
    for k in keys:
        ms = means[k]
        print(f"{k:<26}{ms:8.2f}{100*ms/full_shipped:13.1f}%{(audio/(ms/1000.0)) if ms else 0:12.1f}")
    overhead = max(0.0, full - accounted)
    print(f"{'python/dispatch overhead':<26}{overhead:8.2f}{100*overhead/full_shipped:13.1f}%")
    print(f"{'FULL (no filter)':<26}{full:8.2f}{100*full/full_shipped:13.1f}%{audio/(full/1000.0):12.1f}")
    print(f"{'SHIPPED (with filter)':<26}{full_shipped:8.2f}{100.0:13.1f}%{audio/(full_shipped/1000.0):12.1f}")

    bottleneck = max(means, key=means.get)
    report = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "threads": args.threads,
        "nfe": cfg.flow.distilled_nfe,
        "params": sum(p.numel() for p in model.parameters()),
        "mean_component_ms": means,
        "mean_full_ms": full,
        "mean_full_with_filter_ms": full_shipped,
        "mean_audio_seconds": audio,
        "full_rtf": (full / 1000.0) / max(audio, 1e-9),
        "full_x_realtime": audio / max(full / 1000.0, 1e-9),
        "shipped_rtf": (full_shipped / 1000.0) / max(audio, 1e-9),
        "shipped_x_realtime": audio / max(full_shipped / 1000.0, 1e-9),
        "accounted_fraction": accounted / max(full, 1e-9),
        "bottleneck": bottleneck,
        "per_text": rows,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.out}")
    print(f"\nBOTTLENECK: {bottleneck} ({100*means[bottleneck]/full:.0f}% of full); "
          f"accounted {100*accounted/full:.0f}% of the full path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
