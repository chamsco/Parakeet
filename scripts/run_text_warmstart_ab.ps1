# The pre-registered text-warm-start A/B, both arms on identical data and geometry.
#
# Arm A (fresh) and arm B (text encoder warm-started from the token route) run back to back on the same
# cache, the same item count and the same 512x6 text geometry, so the only difference is the warm start.
# The readout is the flow term's own text dependence at steps 2000 and 4000 -- the gate every conditioning
# change in this project is judged by.
$env:PYTHONIOENCODING = "utf-8"
$common = @(
  "--config", "configs\parakeet_flow_plan.yaml",
  "--cache", "runs\expanded_v2", "--items", "1978",
  "--steps", "4000", "--batch-size", "16", "--crop-frames", "160", "--lr", "0.0005",
  "--set", "text.dim=512", "--set", "text.n_layers=6",
  "--set", "flow.text_dim=512", "--set", "flow.cond_dim=512",
  "--save-every", "2000"
)

Write-Host "[A/B] arm A (fresh, 512x6) -- baseline"
Remove-Item -Recurse -Force runs\ab_fresh -EA SilentlyContinue
& .\.venv\Scripts\python.exe -u scripts\overfit_flow.py @common --out runs\ab_fresh > runs\ab_fresh.log 2>&1
Write-Host "[A/B] arm A done"

Write-Host "[A/B] arm B (text encoder warm-started from the token route)"
Remove-Item -Recurse -Force runs\ab_warmtext -EA SilentlyContinue
& .\.venv\Scripts\python.exe -u scripts\overfit_flow.py @common `
    --warm-start runs\text_continue\distill-text_step5000.pt --out runs\ab_warmtext > runs\ab_warmtext.log 2>&1
Write-Host "[A/B] arm B done"

foreach ($arm in @("ab_fresh", "ab_warmtext")) {
  Write-Host "`n=== $arm ==="
  Get-Content "runs\$arm.log" | Select-String -Pattern "warm start|step\s+2000|step\s+4000" |
      Select-Object -Last 3 | Select-Object -ExpandProperty Line
}
