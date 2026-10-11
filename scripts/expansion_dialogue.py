"""Build a *conversational* prompt set from public-domain plays.

This exists because of one untested hypothesis. Every corpus element this project has is **narration** -- one
speaker, one register, smooth prosody, no turn-taking -- and narration is an unusually *easy* distribution to model
unconditionally, which makes the text look worthless to a flow that only has to fit the marginal. Spontaneous
conversation is the opposite case: the acoustics depend strongly on what is said and when.

Plays give turn-taking cheaply and permissively.  The generator assigns voices round-robin by prompt index, so
emitting the turns **in order** makes consecutive prompts alternate between two voices -- which is the acoustic
pattern being tested, without touching the generator.

    python scripts/expansion_dialogue.py --texts data/dialogue_text --out data/prompts/dialogue.txt \
        --characters 120000 [--existing data/prompts/_used.txt] [--per-play 30000]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Gutenberg plays put the speaker on its own line in caps: "GWENDOLEN." / "FIRST GENTLEMAN."
SPEAKER = re.compile(r"^([A-Z][A-Z0-9 .'\-]{1,30})\.\s*$")
# Stage directions, which are not spoken.
STAGE = re.compile(r"^[\[\(].*[\]\)]\s*$")
# Front and back matter that is not play text.
BOILERPLATE = re.compile(
    r"gutenberg|project gutenberg|copyright|licence|license|transcriber|www\.|http|"
    r"^\s*(act|scene)\b|dramatis personae",
    re.IGNORECASE,
)


def turns_from_play(path: Path) -> list[str]:
    """Extract spoken turns, in order, from one play's text."""
    current: list[str] = []
    speaker = None
    out: list[str] = []

    def flush() -> None:
        if speaker is None or not current:
            return
        text = " ".join(" ".join(current).split())
        if text:
            out.append(text)
        current.clear()

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = SPEAKER.match(line)
        if match:
            flush()
            speaker = match.group(1)
            continue
        if STAGE.match(line) or BOILERPLATE.search(line):
            continue
        if speaker is None:
            continue          # still in front matter
        current.append(line)
    flush()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Conversational prompts from public-domain plays")
    ap.add_argument("--texts", default="data/dialogue_text")
    ap.add_argument("--out", default="data/prompts/dialogue.txt")
    ap.add_argument("--characters", type=int, default=120_000)
    ap.add_argument("--per-play", type=int, default=0, help="0 = split the budget evenly")
    ap.add_argument("--min-chars", type=int, default=25, help="shorter turns are not worth synthesising")
    ap.add_argument("--max-chars", type=int, default=220)
    ap.add_argument("--existing", default="", help="prompt file whose lines to exclude")
    args = ap.parse_args()

    plays = sorted(Path(args.texts).glob("*.txt"))
    if not plays:
        raise SystemExit(f"no .txt under {args.texts}")

    used = set()
    if args.existing and Path(args.existing).exists():
        used = {line.strip() for line in Path(args.existing).read_text(encoding="utf-8").splitlines()
                if line.strip()}

    quota = args.per_play or max(1, args.characters // len(plays))
    chosen: list[str] = []
    for play in plays:
        candidates = [t for t in turns_from_play(play)
                      if args.min_chars <= len(t) <= args.max_chars and t not in used]
        # keep the turns in their original order: that ordering is what makes consecutive prompts alternate
        # between two voices, which is the acoustic pattern this set exists to test
        taken, total = [], 0
        for turn in candidates:
            if total + len(turn) > quota:
                break
            taken.append(turn)
            total += len(turn)
        print(f"  {play.name}: {len(candidates)} usable turns, took {len(taken)} ({total} chars)")
        chosen.extend(taken)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print(f"\nwrote {len(chosen)} dialogue turns, {sum(len(t) for t in chosen)} characters -> {args.out}")
    print("  generate with TWO voices so consecutive prompts alternate speakers:")
    print(f"  scripts/make_teacher_corpus.py --texts {args.out} --teachers kokoro --out data/dialogue_v1 --no-curate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
