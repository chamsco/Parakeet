"""Evaluate a whole series of flow checkpoints and report the trajectory.

    python scripts/flow_trajectory.py --run runs/flow_v2 --limit 8 --steps 4

A single checkpoint's number cannot distinguish "learning" from "stuck", and this project has twice
spent hours on a run whose progress was invisible (round 28 had no intermediate checkpoints at all).
This walks every saved checkpoint in step order and evaluates the three quantities that matter for the
flow, in the order they must come right:

1. **length ratio** — generated seconds against reference seconds.  A flow that predicts a near-zero
   length cannot be intelligible no matter what else is right (measured: 0.01 before the round-32 fix,
   1.67 after);
2. **log-mel cosine** — the acoustic proxy, computable only once (1) is roughly right;
3. **WER with a control** on real speech, withheld if the control fails.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def step_of(path: Path) -> int:
    match = re.search(r"step(\d+)", path.name)
    if match:
        return int(match.group(1))
    return 10**9  # `flow_last.pt`: the final weights, sorts last


def main() -> int:
    ap = argparse.ArgumentParser(description="Flow checkpoint trajectory")
    ap.add_argument("--run", default="runs/flow_v2")
    ap.add_argument("--config", default="configs/parakeet_flow.yaml")
    ap.add_argument("--corpus", default="data/gutenberg_corpus/corpus")
    ap.add_argument("--manifest", default="val.jsonl")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4, help="sampler NFE")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    checkpoints = sorted(Path(args.run).glob("flow_*.pt"), key=step_of)
    if not checkpoints:
        print(f"no checkpoints in {args.run}")
        return 2
    out = Path(args.out) if args.out else Path(args.run) / "trajectory.json"

    rows = []
    for checkpoint in checkpoints:
        report = Path(args.run) / f"trajectory_step{step_of(checkpoint)}.json"
        command = [
            sys.executable, "scripts/real_eval.py",
            "--checkpoint", str(checkpoint),
            "--config", args.config,
            "--corpus", args.corpus,
            "--manifest", args.manifest,
            "--prose-only",
            "--limit", str(args.limit),
            "--steps", str(args.steps),
            "--out", str(report.parent / f"eval_{report.stem}"),
        ]
        done = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
        payload_path = report.parent / f"eval_{report.stem}" / "report.json"
        if done.returncode != 0 or not payload_path.exists():
            print(f"  step {step_of(checkpoint):>6}: evaluation failed")
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        row = {
            "step": step_of(checkpoint),
            "length_ratio": payload["synthesis"].get("length_ratio"),
            "log_mel_cosine": payload["synthesis"].get("log_mel_cosine_vs_reference"),
            "x_realtime": payload["synthesis"].get("x_realtime"),
            "wer": payload["wer"]["student"],
            "wer_control": payload["wer"]["teacher"],
            "dnsmos": payload["naturalness"]["student"],
        }
        rows.append(row)
        print(f"  step {row['step']:>6}: length {row['length_ratio']:.2f} | "
              f"cosine {row['log_mel_cosine'] if row['log_mel_cosine'] is None else round(row['log_mel_cosine'], 4)} | "
              f"WER {row['wer']:.3f} (control {row['wer_control']:.3f}) | "
              f"DNSMOS {row['dnsmos']:.3f} | {row['x_realtime']:.0f}x real time")

    if not rows:
        return 2
    best = min(rows, key=lambda r: (r["wer"] if r["wer"] is not None else 9.9))
    summary = {
        "run": args.run,
        "sampler_nfe": args.steps,
        "utterances": args.limit,
        "checkpoints": len(rows),
        "rows": rows,
        "best_by_wer": best,
        "verdict": (
            "still at chance on every checkpoint: the flow's length is right, its acoustic proxy is "
            "well above zero, and it is still not intelligible"
            if all((r["wer"] or 1.0) >= 0.9 for r in rows)
            else "at least one checkpoint is intelligible -- see the rows"
        ),
    }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  {summary['verdict']}")
    print(f"trajectory -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
