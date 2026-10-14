# Domain A/B: does the flow use the text more on conversational data than on narration?
#
# Round 121 measured the premise directly: the dialogue corpus's marginal (per-voice mean + AR(1) smoothness)
# explains 62.6% of the target's variance against narration's 78.7%, so the dialogue set leaves 37.4% for the text
# against 21.3% -- about 1.75x more conditional signal.  If the field captures a similar FRACTION of what is
# available, its share of the flow objective should rise by roughly the same factor on dialogue data.
#
# PRE-REGISTERED READOUT, written before the runs:
#   - arm D (dialogue) flow-term mean > arm N (narration) flow-term mean, both averaged over four gate points;
#   - the expected effect is ~1.75x on a base of 1-2%, i.e. ~1.5-2 points, against a gate noise floor of +-3-4
#     points per checkpoint.  FOUR points per arm reduce that to roughly +-1.5-2 on the mean, so this test is AT
#     THE EDGE of what the instrument can resolve.  A null therefore does NOT refute the premise; it means the
#     instrument cannot see an effect of this size, and that will be stated rather than spun.
#   - equal item counts (1400 each) so corpus SIZE is not a variable; the only difference is the domain.
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$common = @(
  "--config", "configs\parakeet_flow_plan.yaml",
  "--items", "1400", "--holdout", "32",
  "--steps", "2000", "--batch-size", "16", "--crop-frames", "160", "--lr", "0.0005",
  "--set", "text.dim=512", "--set", "text.n_layers=6",
  "--set", "flow.text_dim=512", "--set", "flow.cond_dim=512",
  "--gate-every", "500", "--save-every", "2000"
)

Write-Host "[domain] arm N -- narration (runs/v3_fixed)"
Remove-Item -Recurse -Force runs\dom_narration -EA SilentlyContinue
& $py -u scripts\overfit_flow.py @common --cache runs\v3_fixed --out runs\dom_narration > runs\dom_narration.log 2>&1
Write-Host "[domain] arm N done (exit $LASTEXITCODE)"

Write-Host "[domain] arm D -- dialogue (runs/dialogue_v1_cache)"
Remove-Item -Recurse -Force runs\dom_dialogue -EA SilentlyContinue
& $py -u scripts\overfit_flow.py @common --cache runs\dialogue_v1_cache --out runs\dom_dialogue > runs\dom_dialogue.log 2>&1
Write-Host "[domain] arm D done (exit $LASTEXITCODE)"

Write-Host "`n=== results ==="
foreach ($arm in @("dom_narration", "dom_dialogue")) {
  Write-Host "--- $arm ---"
  Get-Content "runs\$arm.log" -EA SilentlyContinue |
      Select-String -Pattern "\[gate\]|after 2000|holdout" | Select-Object -ExpandProperty Line
}
