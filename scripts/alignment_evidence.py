"""Verify a teacher's alignment and the corpus quality it produced.

    python scripts/alignment_evidence.py --corpus data/speechify_corpus/corpus

Speechify's word timings are the first real alignment in this project -- every duration target before
them came from the unaligned fallback.  They are only useful if they are *consistent with the audio*:
sum(frame counts) x hop / 24 kHz must reproduce the utterance duration.  This checks that directly
rather than trusting the API, and reports the DNSMOS distribution the curation gates measured, so the
teacher's quality and its alignment are both on the record.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Alignment consistency and corpus quality evidence")
    ap.add_argument("--corpus", default="data/speechify_corpus/corpus")
    ap.add_argument("--manifest", default="curated/kept.jsonl")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--hop-length", type=int, default=256)
    ap.add_argument("--out", default="runs/alignment_evidence.json")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    manifest = corpus / args.manifest
    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not records:
        print(f"no records in {manifest}")
        return 2

    seconds_per_frame = args.hop_length / args.sample_rate
    aligned = [r for r in records if r.get("token_frames")]
    errors: List[float] = []
    totals: List[int] = []
    for record in aligned:
        predicted = sum(int(f) for f in record["token_frames"]) * seconds_per_frame
        actual = float(record["duration_s"])
        if actual <= 0:
            continue
        errors.append(abs(predicted - actual) / actual)
        totals.append(len(record["token_frames"]))

    mos = [
        float(r["quality"]["mos"])
        for r in records
        if isinstance(r.get("quality"), dict) and r["quality"].get("mos") is not None
    ]
    teachers: Dict[str, int] = {}
    for record in records:
        teachers[str(record.get("teacher"))] = teachers.get(str(record.get("teacher")), 0) + 1

    def pct(values: List[float], p: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))]

    checks = {
        "the_corpus_carries_an_alignment": len(aligned) > 0,
        "every_aligned_record_has_one_frame_per_character": all(
            len(r["token_frames"]) == len(r["text"]) for r in aligned
        ),
        # the timings must reproduce the audio duration, or the duration targets are worse than the
        # fallback they replaced
        "the_alignment_reproduces_the_audio_duration": bool(
            errors and statistics.median(errors) < 0.10
        ),
        "every_utterance_kept_a_dnsmos_reading": len(mos) == len(records),
    }
    payload = {
        "corpus": corpus.as_posix(),
        "manifest": args.manifest,
        "records": len(records),
        "teachers": teachers,
        "alignment": {
            "aligned_records": len(aligned),
            "coverage": len(aligned) / max(1, len(records)),
            "characters_per_record_median": statistics.median(totals) if totals else None,
            "median_relative_duration_error": statistics.median(errors) if errors else None,
            "p90_relative_duration_error": pct(errors, 90) if errors else None,
            "seconds_per_frame": seconds_per_frame,
        },
        "dnsmos": {
            "mean": statistics.mean(mos) if mos else None,
            "median": statistics.median(mos) if mos else None,
            "p10": pct(mos, 10) if mos else None,
            "p90": pct(mos, 90) if mos else None,
        },
        "checks": checks,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"  {len(records)} records | teachers {teachers}")
    if aligned:
        print(f"  alignment: {len(aligned)}/{len(records)} "
              f"({100 * len(aligned) / len(records):.0f}%) | median duration error "
              f"{statistics.median(errors):.3f} | p90 {pct(errors, 90):.3f}")
    else:
        print("  alignment: none (this teacher provides no timings)")
    if mos:
        print(f"  DNSMOS: mean {statistics.mean(mos):.2f} | median {statistics.median(mos):.2f} "
              f"| p10 {pct(mos, 10):.2f} | p90 {pct(mos, 90):.2f}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"report -> {args.out}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
