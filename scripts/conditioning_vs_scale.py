"""How does the flow's text dependence scale with corpus size?

Round 64 established that the velocity field's own objective is only ~2 % text-dependent at 200 utterances,
while the 2-utterance regime reached rho 0.25 -- so the field *can* learn the conditioning when the marginal
has little to offer. This measures the curve between those points: train the same architecture for the same
number of steps on 4, 16, 64 and 200 utterances and report the FLOW TERM's text dependence at each.

The prediction is that it falls as the corpus grows, because a bigger corpus makes the marginal distribution
a cheaper thing to model. If it does, the number it falls to says how much scale a conditioning-driven run
would need; if it does not, the round-64 explanation is wrong and something else is limiting the field.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = [4, 16, 64, 200]
STEPS = 1000


def run_one(items: int) -> dict:
    out = f"runs/flow_scale_{items}"
    log = Path(f"runs/flow_scale_{items}.log")
    print(f"\n=== {items} utterances, {STEPS} steps ===", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [sys.executable, "-u", "scripts/overfit_flow.py",
             "--config", "configs/parakeet_flow_plan.yaml",
             "--items", str(items), "--steps", str(STEPS), "--batch-size", "8",
             "--crop-frames", "160", "--lr", "0.0005", "--save-every", str(STEPS),
             "--out", out],
            cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, timeout=3600, check=False,
        )
    tail = log.read_text(encoding="utf-8", errors="ignore")[-600:]
    print(tail[-400:], flush=True)
    return {"items": items, "out": out}


def measure(items: int) -> dict:
    """Run text_dependence.py and pull the flow-term share out of its table.

    Measured on the items the run was trained on (at least 8): a 4-utterance model scored on 32 unseen
    utterances would report a dependence dominated by its own error rather than by the conditioning.
    """
    result = subprocess.run(
        [sys.executable, "scripts/text_dependence.py",
         "--checkpoint", f"runs/flow_scale_{items}/flow_last.pt",
         "--config", "configs/parakeet_flow_plan.yaml", "--items", str(max(8, items))],
        cwd=str(ROOT), capture_output=True, text=True, timeout=1800, check=False,
    )
    text = (result.stdout or "") + (result.stderr or "")
    rows = {}
    for label, key in (("total", "total"), ("flow", "flow"), ("plan", "plan")):
        match = re.search(rf"^{label}.*?([+-]\d+\.\d)%", text, re.MULTILINE)
        if match:
            rows[key] = float(match.group(1))
    match = re.search(r"velocity field's OWN text dependence: ([+-]\d+\.\d)%", text)
    if match:
        rows["flow_own"] = float(match.group(1))
    return rows


results = []
for size in SIZES:
    run_one(size)
    rows = measure(size)
    print(f"  {size:4d} utterances -> flow-term text dependence {rows.get('flow_own')}% "
          f"(total {rows.get('total')}%, plan {rows.get('plan')}%)", flush=True)
    results.append({"items": size, **rows})

Path("runs/flow_scale_curve.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\n=== the curve ===")
for row in results:
    print(f"  {row['items']:4d} utterances: flow {row.get('flow_own')}%  total {row.get('total')}%  "
          f"plan {row.get('plan')}%")
print("\n  If the flow's share falls with corpus size, the marginal's cheapness is what starves the")
print("  conditioning, and the number it falls to says how much scale a conditioning-driven run needs.")
