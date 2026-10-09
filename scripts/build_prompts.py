"""Build a large, licence-clean prompt list from public-domain text.

    python scripts/build_prompts.py --target 500

The round-24 hold-out made the binding problem measurable: with 47 distinct training sentences the
text side is at chance (WER 1.000) on unseen prompts, whatever the corpus duration.  Duration is not
the constraint -- *text diversity* is.  Project Gutenberg texts are public domain, so sentences taken
from them can be synthesised and used for training with no licence question at all (unlike the MiniMax
teacher, which this project refuses to train on).

Outputs ``prompts.txt`` (one sentence per line) and ``prompts.json`` with the provenance of every
source, because a corpus whose licence cannot be traced is not usable.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Dict, List

#: public-domain novels with long, varied, modern-English prose
SOURCES: List[Dict[str, str]] = [
    {
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "licence": "public domain (Project Gutenberg)",
    },
    {
        "title": "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "url": "https://www.gutenberg.org/files/1661/1661-0.txt",
        "licence": "public domain (Project Gutenberg)",
    },
    {
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "url": "https://www.gutenberg.org/files/11/11-0.txt",
        "licence": "public domain (Project Gutenberg)",
    },
    {
        "title": "The Time Machine",
        "author": "H. G. Wells",
        "url": "https://www.gutenberg.org/files/35/35-0.txt",
        "licence": "public domain (Project Gutenberg)",
    },
    {
        "title": "Frankenstein",
        "author": "Mary Shelley",
        "url": "https://www.gutenberg.org/files/84/84-0.txt",
        "licence": "public domain (Project Gutenberg)",
    },
]

START_MARKER = "*** START OF THE PROJECT GUTENBERG"
END_MARKER = "*** END OF THE PROJECT GUTENBERG"
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
ALLOWED_RE = re.compile(r"^[A-Za-z][A-Za-z ,;:'\-()]*[.!?]$")
#: headings and titles are read aloud differently from how they are written ("CHAPTER IV" is spoken
#: "chapter four"), so a WER computed against the written form can never be low.  Round 27 had a
#: recogniser *control* fail on a hold-out for exactly this reason, which inflated every WER over the
#: split until the headings were filtered out of the prompt list.
ROMAN_RE = re.compile(r"\b(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\b\.?$")
HEADING_HEADS = {"CHAPTER", "PART", "BOOK", "ACT", "SCENE", "PREFACE", "CONTENTS", "ADVENTURE"}


def is_usable_prompt(line: str, min_words: int, max_words: int) -> bool:
    words = line.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if any(w.isupper() and len(w) > 1 and w != "I" for w in words):
        return False
    if words[0].strip(".,;:!?").upper() in HEADING_HEADS:
        return False
    if ROMAN_RE.search(line):
        return False
    if sum(1 for w in words if w[:1].isupper()) / len(words) > 0.5:
        return False
    return True


def strip_gutenberg(text: str) -> str:
    start = text.find(START_MARKER)
    if start != -1:
        text = text[text.find("\n", start) + 1 :]
    end = text.find(END_MARKER)
    if end != -1:
        text = text[:end]
    return text


def sentences(text: str, min_words: int, max_words: int) -> List[str]:
    out = []
    for raw in SENTENCE_RE.split(text):
        line = " ".join(raw.split())
        if not line or len(line) > 400:
            continue
        line = line.replace("_", " ").replace("--", ", ").strip()
        if not ALLOWED_RE.match(line):
            continue
        words = line.split()
        if not (min_words <= len(words) <= max_words):
            continue
        if any(len(w) > 16 for w in words):  # proper nouns and archaisms synth poorly
            continue
        if not is_usable_prompt(line, min_words, max_words):
            continue
        out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Public-domain prompt list for corpus synthesis")
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--min-words", type=int, default=8)
    ap.add_argument("--max-words", type=int, default=22)
    ap.add_argument("--out", default="data/prompts")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows: List[Dict[str, str]] = []
    provenance = []
    for source in SOURCES:
        try:
            with urllib.request.urlopen(source["url"], timeout=60) as response:
                text = response.read().decode("utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {source['title']}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        body = strip_gutenberg(text)
        found = sentences(body, args.min_words, args.max_words)
        kept = 0
        for line in found:
            key = re.sub(r"[^a-z]", "", line.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": line, "source": source["title"], "author": source["author"]})
            kept += 1
        provenance.append({**source, "sentences_kept": kept, "chars": len(body)})
        print(f"  {source['title']}: {kept} usable sentences")

    # interleave sources so the split is not one author, then take the target
    by_source: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    interleaved: List[Dict[str, str]] = []
    index = 0
    while any(len(v) > index for v in by_source.values()):
        for values in by_source.values():
            if len(values) > index:
                interleaved.append(values[index])
        index += 1
    selected = interleaved[: args.target]
    if len(selected) < args.target:
        print(f"  note: only {len(selected)} sentences available (target {args.target})")

    (out / "prompts.txt").write_text(
        "\n".join(row["text"] for row in selected) + "\n", encoding="utf-8"
    )
    (out / "prompts.json").write_text(
        json.dumps(
            {
                "count": len(selected),
                "filters": {"min_words": args.min_words, "max_words": args.max_words},
                "sources": provenance,
                "licence_note": (
                    "all sources are public domain (Project Gutenberg), so sentences may be "
                    "synthesised and used for training without a licence question"
                ),
                "sentences": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(selected)} prompts -> {out/'prompts.txt'}")
    from collections import Counter

    print("  by source:", dict(Counter(row["source"] for row in selected)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
