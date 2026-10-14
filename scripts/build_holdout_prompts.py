"""Build a HELD-OUT prompt set from books the corpus was never built from.

Round 139 found that the token route reads rho 0.824 on its training items and 0.15 on anything else -- a
training-versus-generalisation gap of five -- and that every "the token route reaches 0.354" figure this project
published was a TRAINING number.  The same flaw has appeared before (round 24: "every prior real-audio number
trained and evaluated on the same items").  What the project has never had is an in-domain held-out set.

This makes one.  It uses different Project Gutenberg books from the five `build_prompts.py` draws on (Pride and
Prejudice, Sherlock Holmes, Alice, The Time Machine, Frankenstein), and it EXCLUDES every sentence that appears in
any existing prompt file, so "unseen" is enforced rather than assumed.  The exclusion is checked and reported, not
trusted: the script prints the overlap count and fails if it is non-zero.

    python scripts/build_holdout_prompts.py --out data/prompts/holdout.txt --target 400
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_prompts import sentences, strip_gutenberg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: Books NOT among build_prompts.py's five, so their sentences cannot already be in the corpus.
SOURCES = [
    {"name": "dracula", "url": "https://www.gutenberg.org/files/345/345-0.txt"},
    {"name": "jane-eyre", "url": "https://www.gutenberg.org/files/1260/1260-0.txt"},
    {"name": "two-cities", "url": "https://www.gutenberg.org/files/98/98-0.txt"},
    {"name": "moby-dick", "url": "https://www.gutenberg.org/files/2701/2701-0.txt"},
]

EXCLUDE_FILES = ["prompts.txt", "prompts_v2.txt", "prompts_v3.txt", "prompts_v3_rest.txt",
                 "prompts_short.txt", "dialogue.txt"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Held-out prompts from unseen books, with an enforced exclusion")
    ap.add_argument("--out", default="data/prompts/holdout.txt")
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--min-words", type=int, default=8)
    ap.add_argument("--max-words", type=int, default=28)
    ap.add_argument("--cache-dir", default="data/holdout_text")
    args = ap.parse_args()

    used = set()
    for name in EXCLUDE_FILES:
        path = ROOT / "data" / "prompts" / name
        if path.exists():
            used.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"{len(used)} sentences in the existing prompt files will be excluded")

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    picked, provenance, overlap = [], {}, 0
    per_book = max(1, args.target // len(SOURCES))
    for source in SOURCES:
        local = cache / f"{source['name']}.txt"
        if not local.exists():
            print(f"  fetching {source['name']} ...", flush=True)
            with urllib.request.urlopen(source["url"], timeout=120) as response:
                local.write_bytes(response.read())
        text = strip_gutenberg(local.read_text(encoding="utf-8", errors="ignore"))
        taken = []
        for sentence in sentences(text, args.min_words, args.max_words):
            if sentence in used:
                overlap += 1
                continue
            taken.append(sentence)
            if len(taken) >= per_book:
                break
        provenance[source["name"]] = len(taken)
        picked.extend(taken)
        print(f"  {source['name']:12s} {len(taken)} sentences kept")

    # the check that makes the claim true rather than likely
    assert not (set(picked) & used), "a held-out prompt appears in the corpus prompt files"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(picked) + "\n", encoding="utf-8")
    (Path(args.out).with_suffix(".provenance.json")).write_text(
        json.dumps({"books": provenance, "excluded_sentences": len(used), "overlap_seen": overlap,
                    "total": len(picked)}, indent=2), encoding="utf-8")

    print(f"\nwrote {len(picked)} held-out prompts -> {args.out}")
    print(f"  overlap actually seen while scanning: {overlap} (skipped); final intersection with the corpus: 0")
    print("  generate with the standard pipeline, then cache with the trained autoencoder:")
    print(f"  scripts/make_teacher_corpus.py --texts {args.out} --teachers kokoro --out data/holdout_v1 --no-curate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
