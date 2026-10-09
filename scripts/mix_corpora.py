"""Combine curated corpora from several teachers into one mixture corpus.

    python scripts/mix_corpora.py \
        --corpus data/gutenberg_corpus/corpus --corpus data/speechify_corpus/corpus \
        --manifest train.jsonl --out data/mixed_corpus/train \
        --weights kokoro=0.5,speechify=0.5

"The mix training of both" was the original request, and until now there was no way to express it
against real teachers: `synthesize_corpus` realises a mixture by *choosing* one teacher per prompt, so a
corpus is always single-teacher, and `cache_teacher_corpus` reads the mixture from `corpus_meta.json`.
This joins several finished corpora into one manifest, preserving each record's teacher, licence and --
when the teacher provided them -- its `token_frames` alignment.

Waveforms are referenced by **relative path** from the combined directory rather than copied: an hour of
48 kHz audio is hundreds of megabytes, and the cache builder already resolves records against a base
directory.  Nothing downstream needs to know the files live elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def main() -> int:
    ap = argparse.ArgumentParser(description="Join teacher corpora into one mixture manifest")
    ap.add_argument("--corpus", action="append", required=True,
                    help="a corpus directory, optionally PATH:MANIFEST (repeatable); its "
                         "corpus_meta.json supplies the licences")
    ap.add_argument("--manifest", default="curated/kept.jsonl",
                    help="manifest inside each corpus, e.g. train.jsonl")
    ap.add_argument("--out", required=True, help="output directory for the combined corpus")
    ap.add_argument("--weights", default=None,
                    help="teacher=weight pairs (default: share of utterances)")
    ap.add_argument("--limit-per-teacher", type=int, default=None,
                    help="cap how many utterances each teacher contributes (balance the mixture)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    explicit: Dict[str, float] = {}
    if args.weights:
        for item in args.weights.split(","):
            name, _, value = item.partition("=")
            explicit[name.strip()] = float(value)

    records: List[dict] = []
    spec_by_teacher: Dict[str, dict] = {}
    kept_by_teacher: Dict[str, int] = {}
    seconds_by_teacher: Dict[str, float] = {}

    for entry in args.corpus:
        # `PATH:MANIFEST` lets corpora with different layouts join one mixture -- the generated ones
        # write train.jsonl/val.jsonl while an ingested corpus writes curated/kept.jsonl.  The split is
        # decided by *checking the filesystem* rather than by pattern-matching colons: on Windows a
        # plain path is already `C:\...`, so looking for a colon splits the drive letter off.
        left, separator, right = entry.rpartition(":")
        if separator and right.strip() and Path(left).is_dir():
            root_text, manifest_name = left, right
        else:
            root_text, manifest_name = entry, args.manifest
        root = Path(root_text)
        manifest = root / manifest_name
        if not manifest.exists():
            print(f"  !! {manifest} missing; skipping {root}")
            continue
        meta_path = root / "corpus_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        spec_by_teacher.update(meta.get("teachers") or {})
        rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.limit_per_teacher:
            rows = rows[: args.limit_per_teacher]
        for row in rows:
            source = (root / row["wav_path"]).resolve()
            relative = Path(os.path.relpath(source, out.resolve())).as_posix()
            records.append({**row, "wav_path": relative, "corpus_root": root.as_posix()})
            teacher = str(row.get("teacher") or "unknown")
            kept_by_teacher[teacher] = kept_by_teacher.get(teacher, 0) + 1
            seconds_by_teacher[teacher] = seconds_by_teacher.get(teacher, 0.0) + float(
                row.get("duration_s") or 0.0
            )

    if not records:
        print("no records; nothing to mix")
        return 2

    # interleave teachers so every shard of a cache holds the full mixture (the same reasoning as
    # `synthesize_corpus`: shard-local balance avoids long stretches of gradient from one teacher)
    by_teacher: Dict[str, List[dict]] = {}
    for row in records:
        by_teacher.setdefault(str(row.get("teacher") or "unknown"), []).append(row)
    interleaved: List[dict] = []
    index = 0
    while any(len(v) > index for v in by_teacher.values()):
        for rows in by_teacher.values():
            if len(rows) > index:
                interleaved.append(rows[index])
        index += 1

    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in interleaved) + "\n", encoding="utf-8"
    )
    mix = explicit or {t: float(count) for t, count in kept_by_teacher.items()}
    meta = {
        "mix": mix,
        "teachers": spec_by_teacher,
        "n_utterances": len(interleaved),
        "sources": [Path(c).as_posix() for c in args.corpus],
        "source_manifest": args.manifest,
        "utterances_by_teacher": kept_by_teacher,
        "minutes_by_teacher": {t: round(s / 60, 2) for t, s in seconds_by_teacher.items()},
    }
    (out / "corpus_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    aligned = sum(1 for r in interleaved if r.get("token_frames"))
    print(f"mixed {len(interleaved)} utterances {dict(kept_by_teacher)}")
    print(f"  audio: {sum(seconds_by_teacher.values()) / 60:.1f} min | "
          f"teachers with an alignment: {aligned}/{len(interleaved)}")
    print(f"  mixture weights: {mix}")
    print(f"  -> {out/'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
