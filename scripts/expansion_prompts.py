"""Build a fresh prompt list for a corpus expansion, without touching the validation text.

The operator approved spending ~100k Speechify characters on more teacher audio. Two rules matter for the
result to be worth anything:

* **new text**: re-reading the same sentences in another voice adds voice diversity, not text diversity, and
  text diversity is what the mapping needs;
* **no validation leakage**: any prompt whose text appears in the held-out split is skipped, or the val WER
  would be measuring memorisation.

Sources are the local Gutenberg records (public domain). Prompts are sentence-ish lines of 60-220 characters
so each API call produces one clean utterance.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SENTENCE = re.compile(r"[^.!?]+[.!?]")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compose expansion prompts")
    ap.add_argument("--train", default="data/gutenberg_corpus/corpus/train.jsonl")
    ap.add_argument("--val", default="data/gutenberg_corpus/corpus/val.jsonl")
    ap.add_argument("--existing", default="data/prompts/prompts.txt")
    ap.add_argument("--plain", default=None,
                    help="a directory of public-domain .txt files to draw *new* sentences from; the local "
                         "Gutenberg records are exhausted once the existing prompts and the validation set "
                         "are excluded")
    ap.add_argument("--out", default="data/prompts/prompts_v2.txt")
    ap.add_argument("--characters", type=int, default=100_000)
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=220)
    args = ap.parse_args()

    def texts(path: str) -> set[str]:
        found = set()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                found.add(json.loads(line).get("text", "").strip().lower())
            except json.JSONDecodeError:
                continue
        return found

    blocked = texts(args.val)
    if Path(args.existing).exists():
        blocked |= {
            line.strip().lower() for line in Path(args.existing).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    print(f"excluding {len(blocked)} prompts (validation + already used)")

    seen: set[str] = set()
    prompts: list[str] = []
    total = 0

    def consider(candidate: str) -> bool:
        """Add a candidate if it reads as a clean prompt.  Returns True when the budget is full."""
        nonlocal total
        candidate = " ".join(candidate.split())
        key = candidate.lower()
        if not (args.min_len <= len(candidate) <= args.max_len):
            return False
        if key in blocked or key in seen:
            return False
        if candidate.count(" ") < 6 or not candidate[0].isupper():
            return False
        if not candidate[0].isalpha():
            return False
        # reject front matter, headings and markup: these cost real characters and produce utterances no
        # text-to-speech model should be trained to imitate
        if any(token in candidate for token in ("_", "[", "]", "{", "}", "http", "Gutenberg", "Copyright")):
            return False
        if "CHAPTER" in candidate or "Chapter" in candidate.split()[0] if candidate.split() else False:
            return False
        upper = sum(1 for ch in candidate if ch.isupper())
        if upper > 0.3 * max(1, sum(1 for ch in candidate if ch.isalpha())):
            return False
        if candidate.isupper():
            return False
        words = candidate.split()
        if not (8 <= len(words) <= 40):
            return False
        if not candidate.rstrip().endswith((".", "!", "?")):
            return False
        seen.add(key)
        prompts.append(candidate)
        total += len(candidate)
        return total >= args.characters

    if args.plain:
        # spread the budget across the sources rather than exhausting the first book alphabetically: text
        # diversity is the point of the expansion
        files = sorted(Path(args.plain).glob("*.txt"))
        quota = max(1, args.characters // max(1, len(files)))
        for path in files:
            if total >= args.characters:
                break
            before = total
            body = path.read_text(encoding="utf-8", errors="ignore")
            # drop the Project Gutenberg licence header and footer
            for marker in ("*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"):
                if marker in body:
                    body = body.split(marker, 1)[1].split("\n", 1)[-1]
            if "*** END OF THE PROJECT GUTENBERG" in body:
                body = body.split("*** END OF THE PROJECT GUTENBERG", 1)[0]
            for sentence in SENTENCE.findall(body.replace("\r", " ").replace("\n", " ")):
                if total - before >= quota or total >= args.characters:
                    break
                consider(sentence)
            print(f"  {path.name}: {total - before} characters")

    for line in Path(args.train).read_text(encoding="utf-8").splitlines():
        if total >= args.characters:
            break
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for sentence in SENTENCE.findall(record.get("text", "")):
            if consider(sentence):
                break

    Path(args.out).write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"wrote {len(prompts)} prompts, {total} characters -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
