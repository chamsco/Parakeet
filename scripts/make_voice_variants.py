"""Render a punctuation variant of every Kokoro-rendered record, with that record's OWN voice.

The generator assigns voices round-robin by index, which cannot reproduce a manifest's per-record voice when the
corpus mixes teachers (the base corpus has 12 voices across Kokoro and Speechify) -- and Speechify items cannot be
re-synthesised at all under the scoped permission recorded in docs/LEGAL.md.  So this calls the Kokoro teacher
directly and passes the record's own voice, which is the only way to guarantee the pair is same-voice.

Only Kokoro records are touched; the rest are skipped and reported.  The output is a manifest in the same format, so
`make_teacher_corpus --cache-only` and everything downstream accept it unchanged.

    python scripts/make_voice_variants.py --manifest data/mixed_corpus/train_v2/manifest.jsonl \
        --out data/rendition_base_v1 [--limit 700]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf

from make_rendition_pairs import variants  # noqa: E402
from parakeet.data.teacher import build_backend, check_teacher  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Same-voice punctuation variants of a manifest's Kokoro records")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="kokoro")
    ap.add_argument("--limit", type=int, default=0, help="0 = every Kokoro record")
    ap.add_argument("--variants-per-record", type=int, default=1)
    args = ap.parse_args()

    check_teacher(args.teacher)
    teacher = build_backend(args.teacher)
    records = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
               if line.strip()]

    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    skipped = written = identical = 0

    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            if str(record.get("teacher", "")) != args.teacher:
                skipped += 1
                continue
            if args.limit and written >= args.limit:
                break
            text = str(record.get("text", "")).strip()
            voice = str(record.get("voice", "")) or None
            options = variants(text)
            if not options:
                identical += 1
                continue
            variant = options[index % len(options)]
            audio, rate = teacher.synthesize(variant, voice=voice)
            utt_id = f"variant_{written:07d}"
            path = out / "wav" / f"{utt_id}.wav"
            sf.write(str(path), audio, rate)
            new = dict(record)
            new.update({
                "utt_id": utt_id,
                "text": variant,
                "wav_path": str(Path("wav") / f"{utt_id}.wav"),
                "duration_s": float(len(audio) / rate),
                "variant_of": record.get("utt_id"),
                "variant_kind": "punctuation",
            })
            handle.write(json.dumps(new) + "\n")
            handle.flush()
            written += 1
            if written % 100 == 0:
                print(f"  {written} variants rendered", flush=True)

    teacher.close() if hasattr(teacher, "close") else None
    print(f"\nwrote {written} same-voice variants -> {manifest_path}")
    print(f"  skipped {skipped} records from other teachers (not re-synthesised: scoped permission)")
    print(f"  {identical} records had no usable variant and were left out rather than duplicated")
    print("  each variant carries variant_of so the pair can be reconstructed from the manifest alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
