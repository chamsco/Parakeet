"""What does the teacher's own alignment add over the unaligned fallback?

    python scripts/alignment_value.py --corpus data/speechify_corpus/corpus --manifest train.jsonl

Round 26 verified that Speechify's word timings reproduce the audio duration (0.8 % median error).
That makes them *correct*; it does not make them *informative* relative to the fallback the pipeline used
before.  This measures the difference directly: for the same utterances, compare the teacher's aligned
per-character frame counts against the energy-weighted fallback that `extract_signals` produces when no
alignment is given.

If the two agree closely the alignment changes little and a training A/B is not worth its hours; if the
aligned targets vary where the fallback is smooth, the alignment is carrying real phonetic structure --
which is exactly what the duration head is supposed to learn.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.data.features import extract_signals  # noqa: E402
from parakeet.data.text import TextTokenizer  # noqa: E402


def pearson(a: List[float], b: List[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ta, tb = torch.tensor(a), torch.tensor(b)
    ta = ta - ta.mean()
    tb = tb - tb.mean()
    denom = float(ta.norm() * tb.norm())
    return float((ta * tb).sum() / denom) if denom > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="Aligned vs fallback duration targets")
    ap.add_argument("--corpus", default="data/speechify_corpus/corpus")
    ap.add_argument("--manifest", default="train.jsonl")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--token-sample-rate", type=int, default=24000,
                    help="the rate the aligned frame counts were computed at (the *student's*, not the "
                         "teacher's: comparing them against 48 kHz frames made the alignment look 50%% "
                         "wrong when it is 0.8%% right)")
    ap.add_argument("--out", default="runs/alignment_value.json")
    args = ap.parse_args()

    import soundfile as sf

    corpus = Path(args.corpus)
    cfg = load_config(args.config)
    tokenizer = TextTokenizer(mode=cfg.text.mode)
    records = [
        json.loads(l)
        for l in (corpus / args.manifest).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ][: args.limit]

    rows = []
    for record in records:
        if not record.get("token_frames"):
            continue
        wav, sr = sf.read(str(corpus / record["wav_path"]), dtype="float32")
        wav_t = torch.from_numpy(wav).reshape(1, -1)
        if sr != cfg.audio.sample_rate:
            # the real pipeline resamples to the student rate before extracting anything; comparing a
            # 48 kHz fallback against 24 kHz aligned frames made the fallback look 100% wrong
            new_len = int(wav_t.shape[-1] * cfg.audio.sample_rate / sr)
            wav_t = torch.nn.functional.interpolate(
                wav_t[:, None, :], size=new_len, mode="linear", align_corners=False
            )[:, 0, :]
        ids = tokenizer.encode(record["text"], add_special=False)
        aligned = [float(f) for f in record["token_frames"]][: int(ids.numel())]
        signal = extract_signals(wav_t, cfg, ids)
        fallback = [float(d) for d in signal.durations]
        n = min(len(aligned), len(fallback))
        if n < 4:
            continue
        rows.append(
            {
                "utt_id": record["utt_id"],
                "tokens": n,
                "correlation": pearson(aligned[:n], fallback[:n]),
                "aligned_total": sum(aligned[:n]),
                "fallback_total": sum(fallback[:n]),
                "audio_frames": float(
                    args.token_sample_rate * record["duration_s"] / cfg.audio.hop_length
                ),
            }
        )

    if not rows:
        print("no aligned records found")
        return 2

    correlations = [r["correlation"] for r in rows if r["correlation"] == r["correlation"]]
    length_errors = [
        abs(r["aligned_total"] - r["audio_frames"]) / max(1.0, r["audio_frames"]) for r in rows
    ]
    fallback_length_errors = [
        abs(r["fallback_total"] - r["audio_frames"]) / max(1.0, r["audio_frames"]) for r in rows
    ]
    checks = {
        # the fallback's *total* is exact by construction (it splits the available frames), so the
        # meaningful question is whether it puts them in the same places the teacher does
        "the_alignment_changes_the_per_token_structure": (
            bool(correlations) and statistics.median(correlations) < 0.5
        ),
        "both_targets_cover_the_audio": all(
            error < 0.05 for error in length_errors + fallback_length_errors
        ),
        "the_two_agree_on_the_total_length": all(
            abs(r["aligned_total"] - r["fallback_total"]) / max(1.0, r["aligned_total"]) < 0.05
            for r in rows
        ),
    }
    payload = {
        "corpus": corpus.as_posix(),
        "manifest": args.manifest,
        "utterances": len(rows),
        "correlation_aligned_vs_fallback": {
            "median": statistics.median(correlations) if correlations else None,
            "min": min(correlations) if correlations else None,
            "max": max(correlations) if correlations else None,
        },
        "length_error_vs_audio": {
            "aligned_median": statistics.median(length_errors),
            "fallback_median": statistics.median(fallback_length_errors),
        },
        "examples": rows[:4],
        "checks": checks,
        "note": (
            "The fallback's total length is exact by construction -- it splits the frames the audio "
            "has -- so the informative measurement is the correlation: a low one means the teacher "
            "puts the duration on different tokens than an even/energy-weighted split does, which is "
            "what the duration head is supposed to learn.  A high correlation would mean the fallback "
            "already carries the same information and a training A/B would be hard to justify."
        ),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"  {len(rows)} utterances compared")
    print(f"  correlation(aligned, fallback): median "
          f"{payload['correlation_aligned_vs_fallback']['median']:.3f} "
          f"(min {payload['correlation_aligned_vs_fallback']['min']:.3f})")
    print(f"  length error vs audio: aligned {statistics.median(length_errors):.4f} | "
          f"fallback {statistics.median(fallback_length_errors):.4f}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"report -> {args.out}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
