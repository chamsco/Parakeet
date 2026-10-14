"""Dev: print the folded residuals from the two saved budget reports."""

import json

for name in ("variance_narration", "variance_dialogue"):
    data = json.load(open(f"runs/{name}.json", encoding="utf-8"))
    folded = data["folded"]
    print(f"{name:20s} dims {folded['dims']:4d} | per-voice R2 {folded['per_voice_r2']:.3f} "
          f"| plus AR1 {folded['plus_ar1_r2']:.3f} | residual for text {folded['residual_for_text']:.3f}")
