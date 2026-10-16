"""Ask earshot what our audio actually sounds like.

The question this answers is the operator's: is the student's output audible speech or straight noise? earshot hears
through CLAP and an AudioSet tagger, and its own README is explicit that it cannot judge speech CONTENT - so this is not
a substitute for the word error rate, which says 1.0000. What it can say is what the sound IS: which AudioSet classes it
lands in, how well a description fits, and whether it reads as a real recording or as code-synthesized.

Everything runs in one process on purpose. Each earshot CLI call is a fresh process that loads ~1.4 GB of models, which
is 20-40 s on a laptop and much worse here; the README recommends exactly this for more than a few files.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import earshot  # noqa: E402


def main() -> int:
    files = sys.argv[1:] or [
        "runs/say_demo.wav",
        "runs/demo_listen/generated_0.wav",
        "runs/demo_listen/reference_0.wav",
    ]
    for name in files:
        path = Path(name)
        if not path.exists():
            print(f"  missing: {name}")
            continue
        print(f"\n=== {name} ===")
        try:
            report = earshot.describe(str(path))
            text = report if isinstance(report, str) else report.get("summary") or str(report)
            print("  describe:", str(text)[:600])
        except Exception as error:  # noqa: BLE001
            print(f"  describe failed: {type(error).__name__}: {error}")
        for query, alts in (
            ("a person speaking a sentence", ["a buzzing electrical hum", "white noise", "music"]),
        ):
            try:
                match = earshot.match(str(path), query, alts=alts)
                print("  match:", str(match)[:400])
            except Exception as error:  # noqa: BLE001
                print(f"  match failed: {type(error).__name__}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
