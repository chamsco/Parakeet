"""Compare the domain A/B's two arms against the prediction that was registered before the runs.

Pre-registered (scripts/run_domain_ab.ps1, written before any result): arm D (dialogue) flow-term mean exceeds arm N
(narration)'s, both averaged over four gate points; expected effect ~1-2 points on a 1-2% base; and a null does NOT
refute the premise, because the in-loop gate's per-checkpoint noise is +-3-4 points and the running arms use
repeats=2.

The script reports the spread of each arm's own points as well, because that spread IS the instrument's noise: if it
is the same size as the difference, the test cannot resolve the effect and that is a statement about the
measurement, not about the hypothesis.

    python scripts/analyse_domain_ab.py [--runs runs/dom_narration runs/dom_dialogue]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

PATTERN = re.compile(r"\[gate\] step (\d+): FLOW-TERM text dependence ([+-][0-9.]+)%")


def points(log: Path) -> list[tuple[int, float]]:
    if not log.exists():
        return []
    out = []
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = PATTERN.search(line)
        if match:
            out.append((int(match.group(1)), float(match.group(2))))
    return out


def summarise(name: str, log: Path) -> dict:
    found = points(log)
    if not found:
        print(f"{name:16s} no gate points yet ({log})")
        return {"arm": name, "points": []}
    values = [value for _, value in found]
    print(f"{name:16s} {len(found)} points at {[step for step, _ in found]}")
    print(f"{'':16s} values {[f'{v:+.1f}' for v in values]}")
    print(f"{'':16s} mean {statistics.mean(values):+.2f}  "
          f"spread {max(values) - min(values):.1f}  sd {statistics.stdev(values) if len(values) > 1 else 0:.2f}")
    return {"arm": name, "points": [{"step": step, "flow_share": value} for step, value in found],
            "mean": statistics.mean(values), "min": min(values), "max": max(values)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyse the domain A/B against its pre-registered prediction")
    ap.add_argument("--runs", nargs=2, default=["runs/dom_narration.log", "runs/dom_dialogue.log"],
                    help="the two arm LOG FILES (run_domain_ab.ps1 redirects each arm's stdout to a file, and "
                         "that is where the gate lines are -- not <rundir>/train.log, which does not exist)")
    ap.add_argument("--out", default="runs/domain_ab_analysis.json")
    args = ap.parse_args()

    narration = summarise("narration (N)", Path(args.runs[0]))
    dialogue = summarise("dialogue (D)", Path(args.runs[1]))

    report: dict = {"narration": narration, "dialogue": dialogue,
                    "pre_registered": "arm D's flow-term mean exceeds arm N's across four gate points"}
    print()
    if narration.get("points") and dialogue.get("points"):
        difference = dialogue["mean"] - narration["mean"]
        noise_n = narration["max"] - narration["min"]
        noise_d = dialogue["max"] - dialogue["min"]
        report["difference"] = difference
        report["narration_spread"] = noise_n
        report["dialogue_spread"] = noise_d
        print(f"difference (D - N): {difference:+.2f} points")
        print(f"each arm's own spread: N {noise_n:.1f}, D {noise_d:.1f} points "
              f"(this is the instrument's noise, and the running arms use repeats=2)")
        if abs(difference) < max(noise_n, noise_d) / 2:
            verdict = ("UNRESOLVED: the difference is smaller than the arms' own spread. A null here is a "
                       "statement about the instrument, not about the premise -- the in-loop gate ran repeats=2 "
                       "against a +-3-4 point noise floor, and the predicted effect is ~1-2 points.")
        elif difference > 0:
            verdict = ("CONSISTENT WITH THE PREDICTION: dialogue's flow term is higher than narration's.")
        else:
            verdict = ("OPPOSITE TO THE PREDICTION: narration's flow term is higher. Worth taking seriously, "
                       "because it would say a harder marginal does not translate into more text use.")
        report["verdict"] = verdict
        print(f"\nVERDICT: {verdict}")
    elif narration.get("points") or dialogue.get("points"):
        only = narration if narration.get("points") else dialogue
        report["verdict"] = "PARTIAL: only one arm has gate points; no comparison is possible yet."
        print("PARTIAL: only one arm has gate points, so there is nothing to compare yet. The completed arm's own")
        print("spread is still useful -- it is the noise the other arm will be read against.")
    else:
        report["verdict"] = "NO DATA: neither arm has produced a gate point."
        print("NO DATA yet.")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
