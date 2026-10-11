# The pre-registered text-warm-start A/B, both arms on identical data and geometry.
#
# Arm A (fresh) and arm B (text encoder warm-started from the token route) run back to back on the same corpus,
# the same item count and the same 512x6 text geometry, so the only difference is the warm start.  The readout
# is the flow term's own text dependence -- the gate every conditioning change in this project is judged by --
# sampled every 500 steps so four points per arm show the shape, plus validation rho at the end.
#
# Why this and not more corpus: round 93 read the gate at step 2000 on 4533 items (2.3x the previous run) and
# the field's own dependence went DOWN (2.4% against 4.0%) while sampled rho stayed at ~0.  More data is not the
# lever at this scale, and the token route's text encoder is trained to regress text -> acoustics at rho 0.37,
# which is the one mechanism in this project that has demonstrably learned the mapping.
$env:PYTHONIOENCODING = "utf-8"
$common = @(
  "--config", "configs\parakeet_flow_plan.yaml",
  "--cache", "runs\expanded_v3", "--items", "4501", "--holdout", "32",
  "--steps", "2000", "--batch-size", "16", "--crop-frames", "160", "--lr", "0.0005",
  "--set", "text.dim=512", "--set", "text.n_layers=6",
  "--set", "flow.text_dim=512", "--set", "flow.cond_dim=512",
  "--gate-every", "500", "--save-every", "2000"
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
  Get-Content "runs\$arm.log" -EA SilentlyContinue |
      Select-String -Pattern "warm start|\[gate\]|holdout" | Select-Object -ExpandProperty Line
}
