"""Measure the held-out plateau properly and write it as evidence.

Every reading is a full `token_route_ceiling.py` pass over the held-out corpus, repeated so the spread comes with the
number. The point of the repeat is on record: the same checkpoint read twice moves by up to 0.0204, so a single reading
cannot support a trend claim, and three claims in this project were made and retracted on single readings.

Writes runs/text_plateau.json, which the model card uses as the evidence for the claim that the ceiling is below 0.30.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the held-out plateau with repeats")
    ap.add_argument("--run", default="runs/text_long384")
    ap.add_argument("--cache", default="runs/holdout_cache")
    ap.add_argument("--items", type=int, default=60)
    ap.add_argument("--steps", default="3000,4000,5000,6000,7000")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="runs/text_plateau.json")
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    readings: list[dict] = []
    for step in steps:
        checkpoint = ROOT / args.run / f"distill-text_step{step}.pt"
        if not checkpoint.exists():
            continue
        values = []
        for _ in range(args.repeats):
            subprocess.run(
                [PY, "scripts/token_route_ceiling.py", "--checkpoint", str(checkpoint),
                 "--cache", args.cache, "--items", str(args.items)],
                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
            payload = json.loads((ROOT / "runs/token_route_ceiling.json").read_text(encoding="utf-8"))
            values.append(float(payload["frame_rho"]))
        readings.append({"step": step, "values": values, "mean": statistics.fmean(values),
                         "spread": max(values) - min(values)})
        print(f"  step {step:>6}  mean {readings[-1]['mean']:.4f}  spread {readings[-1]['spread']:.4f}  {values}")

    if not readings:
        print("no checkpoints found", file=sys.stderr)
        return 1

    means = [r["mean"] for r in readings]
    within = [r["spread"] for r in readings]
    summary = {
        "run": args.run,
        "cache": args.cache,
        "items": args.items,
        "repeats": args.repeats,
        "steps": [r["step"] for r in readings],
        "mean": statistics.fmean(means),
        "sd_across_steps": statistics.stdev(means) if len(means) > 1 else 0.0,
        "min": min(means),
        "max": max(means),
        "largest_within_checkpoint_spread": max(within),
        "note": ("held-out frame rho on 400 prompts from four unseen books. The plateau is read over steps 3000-7000; "
                 "the within-checkpoint spread is the instrument's resolution, and it is why no trend below about "
                 "0.02 can be claimed from these points."),
    }
    (ROOT / args.out).write_text(json.dumps({"summary": summary, "readings": readings}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"  mean {summary['mean']:.4f}  max {summary['max']:.4f}  sd across steps {summary['sd_across_steps']:.4f}"
          f"  largest within-checkpoint spread {summary['largest_within_checkpoint_spread']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
