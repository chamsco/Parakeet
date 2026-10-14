"""Build a punctuation variant of every prompt in a corpus manifest, keeping the prompt ORDER.

The variant list is emitted in the manifest's own order and with the same length, which is what makes the voices
line up: the corpus generator assigns voices round-robin by prompt INDEX, so variant j and original j both land on
voice (j mod 11) as long as the lists correspond. No voice bookkeeping is needed and the generator is untouched.

Used with scripts/onmanifold_shared_decode.py, this gives the whole corpus the treatment the 110-pair experiment
validated: average each original waveform with its variant, re-encode, and train the text side on the
rendition-independent target -- which decodes at WER 0.368 against this project's 1.000.

    python scripts/make_corpus_variants.py --manifest data/mixed_corpus/train_v2/manifest.jsonl \
        --out data/rendition_pairs/base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make_rendition_pairs import variants  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Punctuation variants of a corpus manifest's prompts, same order")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="prefix for <out>_originals.txt and <out>_variants.txt")
    ap.add_argument("--limit", type=int, default=0, help="0 = every record")
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if args.limit:
        records = records[:args.limit]

    originals, changed, unchanged = [], [], 0
    for record in records:
        text = str(record.get("text", "")).strip()
        originals.append(text)
        options = variants(text)
        # keep ONE variant per record so positions correspond; where punctuation cannot change, reuse the original
        # (the pair is then identical and contributes no rendition difference -- recorded, not hidden)
        chosen = options[0] if options else text
        if chosen == text:
            unchanged += 1
        changed.append(chosen)

    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}_originals.txt").write_text("\n".join(originals) + "\n", encoding="utf-8")
    Path(f"{prefix}_variants.txt").write_text("\n".join(changed) + "\n", encoding="utf-8")

    print(f"{len(records)} records -> {prefix}_originals.txt / {prefix}_variants.txt")
    print(f"  {len(records) - unchanged} differ, {unchanged} identical (reused; those pairs carry no variant)")
    print(f"  voices line up because both lists are in manifest order and the generator assigns by INDEX")
    for i in range(min(2, len(records))):
        print(f"    A: {originals[i][:74]}")
        print(f"    B: {changed[i][:74]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
