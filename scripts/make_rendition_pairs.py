"""Build near-identical-text pairs to measure the rendition reliability with a real sample size.

The ceiling that decides whether a deterministic text->latent route has headroom is a RELIABILITY: how much two
renditions of the same words by the same voice agree.  The project has four such pairs, whose 95% CI spans
[-0.90, +0.99] -- i.e. no measurement at all.

Repeat generations are not available: Kokoro is deterministic, and the corpus holds one take per (text, voice).  But
a conservative stand-in can be manufactured in minutes.  Take a sentence and change only its final punctuation or
insert a comma: the WORDS are identical, so the phonetic content is identical and a recogniser reads both the same,
while the rendering differs.  Same text for our purposes, different rendition for the model's.

That makes it a LOWER bound on the reliability, deliberately: punctuation changes prosody systematically rather than
independently, so the rendition difference here is at least as large as between two independent generations, which
pushes the measured agreement DOWN and the implied ceiling DOWN with it.  If even this conservative estimate leaves
the recogniser's 0.60 bar reachable, the headroom question is settled in the affirmative.

    python scripts/make_rendition_pairs.py --prompts data/prompts/prompts.txt --out data/rendition_pairs \
        [--pairs 120]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

#: ordered so the replacements are attempted in turn; every one keeps the words identical
def variants(sentence: str) -> list[str]:
    out = []
    stripped = sentence.strip()
    if not stripped:
        return out
    # final punctuation swaps
    if stripped.endswith("."):
        out.append(stripped[:-1] + "!")
        out.append(stripped[:-1] + "?")
    elif stripped.endswith("!"):
        out.append(stripped[:-1] + ".")
    elif stripped.endswith("?"):
        out.append(stripped[:-1] + ".")
    # a comma inserted before the last clause if the sentence is long enough to have one
    if len(stripped) > 60:
        match = re.search(r"\b(and|but|which|while|because|when|that)\b", stripped[20:])
        if match:
            position = 20 + match.start()
            out.append((stripped[:position] + "," + stripped[position:]).replace(" ,", ","))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Near-identical-text pairs for a rendition-reliability estimate")
    ap.add_argument("--prompts", default="data/prompts/prompts.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=120)
    args = ap.parse_args()

    sentences = [line.strip() for line in Path(args.prompts).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    rows: list[tuple[str, str]] = []
    for sentence in sentences:
        for variant in variants(sentence):
            rows.append((sentence, variant))
            break            # one variant per sentence: the pairs must be independent of each other
        if len(rows) >= args.pairs:
            break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    originals = out / "originals.txt"
    variants_file = out / "variants.txt"
    originals.write_text("\n".join(a for a, _ in rows) + "\n", encoding="utf-8")
    variants_file.write_text("\n".join(b for _, b in rows) + "\n", encoding="utf-8")

    changed = sum(1 for a, b in rows if a != b)
    print(f"{len(rows)} pairs -> {originals} and {variants_file} ({changed} actually differ)")
    print("  words identical in every pair; only punctuation differs")
    for a, b in rows[:3]:
        print(f"    A: {a[:76]}")
        print(f"    B: {b[:76]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
