"""Check an expansion corpus before the merge spends CPU on it.

Run this on the streaming corpus while it is still being built, or on the finished one before merging. It answers the
questions that would otherwise be discovered after a twenty-minute merge, or worse, after a training run: are the wavs
there, are the ids unique, is any prompt duplicated, does any held-out prompt appear, and is the duration distribution
sane.

It exits non-zero if something would corrupt the merged cache, and prints warnings for things that are merely worth
knowing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"utt_id", "text", "voice", "wav_path", "duration_s", "teacher"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a corpus before merging it")
    ap.add_argument("--corpus", default="data/kokoro_v4")
    ap.add_argument("--holdout", default="data/holdout_v1/manifest.jsonl")
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    args = ap.parse_args()

    corpus = ROOT / args.corpus
    manifest = corpus / "manifest.jsonl"
    if not manifest.exists():
        print(f"no manifest at {manifest}")
        return 1

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"  records: {len(rows)}")
    if not rows:
        return 1

    problems: list[str] = []

    ids = Counter(row["utt_id"] for row in rows)
    duplicates = [key for key, count in ids.items() if count > 1]
    print(f"  duplicate utt_ids: {len(duplicates)}")
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate utt_ids, e.g. {duplicates[:3]}")

    missing = [r["utt_id"] for r in rows if not (corpus / r["wav_path"]).exists()]
    print(f"  missing wav files: {len(missing)}")
    if missing:
        problems.append(f"{len(missing)} missing wavs, e.g. {missing[:3]}")

    incomplete = [r["utt_id"] for r in rows if not set(r) >= REQUIRED]
    print(f"  records missing a required field: {len(incomplete)}")
    if incomplete:
        problems.append(f"{len(incomplete)} incomplete records")

    durations = [float(r["duration_s"]) for r in rows]
    print(f"  audio: {sum(durations) / 3600:.2f} h   median {statistics.median(durations):.2f} s   "
          f"min {min(durations):.2f}   max {max(durations):.2f}")
    out_of_range = [r["utt_id"] for r, d in zip(rows, durations) if not args.min_seconds <= d <= args.max_seconds]
    if out_of_range:
        print(f"  outside {args.min_seconds}-{args.max_seconds} s: {len(out_of_range)} (warning, not a failure)")

    texts = Counter(row["text"] for row in rows)
    repeated = sum(1 for count in texts.values() if count > 1)
    print(f"  distinct texts: {len(texts)}   texts appearing more than once: {repeated}")

    voices = Counter(row["voice"] for row in rows)
    rates = sorted({row.get("sample_rate") for row in rows})
    print(f"  voices: {len(voices)}   sample rates: {rates}")
    if len(rates) > 1:
        problems.append(f"more than one sample rate: {rates}")

    if (ROOT / args.holdout).exists():
        holdout = {json.loads(line)["text"].strip().lower() for line in
                   (ROOT / args.holdout).read_text(encoding="utf-8").splitlines() if line.strip()}
        overlap = holdout & {row["text"].strip().lower() for row in rows}
        print(f"  overlap with {len(holdout)} held-out prompts: {len(overlap)}")
        if overlap:
            problems.append(f"{len(overlap)} held-out prompts appear in the corpus")

    if problems:
        print("\nWOULD CORRUPT THE MERGE:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("\nclean: safe to merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
