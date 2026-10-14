# The matched shared-target experiment: identical items, target the only variable.
#
# Round 136 measured the rendition-independent target at WER 0.368 (0.455 on this subset) against the models'
# 1.000.  Both arms here train on the SAME 304 items -- verified identical in ids, voice and durations -- with the
# same geometry, the same warm start and the same steps, so the only difference is the target.
#
# The readout is WER on decoded samples, measured on the HELD-OUT corpus (data/holdout_v1), because round 139
# showed this model memorises: the token route reads rho 0.82 on its training items and 0.15 anywhere else.  A
# pilot on 304 items will memorise both targets; what it can still show is whether the shared target produces
# better SAMPLES, which is the mechanical question and the one the whole route depends on.
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$common = @(
  "--config", "configs\parakeet_tiny.yaml", "--stage", "distill-text",
  "--warm-start", "runs\text_continue\distill-text_step5000.pt",
  "--steps", "1500", "--batch-size", "16", "--device", "cpu",
  "--set", "text.dim=512", "--set", "text.n_layers=6",
  "--set", "flow.text_dim=512", "--set", "flow.cond_dim=512",
  "--set", "train.save_every=500"
)

Write-Host "[targets] arm O -- ORIGINAL targets (control)"
Remove-Item -Recurse -Force runs\arm_original -EA SilentlyContinue
& $py -u scripts\train.py @common --cache runs\target_original --out runs\arm_original > runs\arm_original.log 2>&1
Write-Host "[targets] arm O done (exit $LASTEXITCODE)"

Write-Host "[targets] arm S -- SHARED (rendition-independent) targets"
Remove-Item -Recurse -Force runs\arm_shared -EA SilentlyContinue
& $py -u scripts\train.py @common --cache runs\target_shared --out runs\arm_shared > runs\arm_shared.log 2>&1
Write-Host "[targets] arm S done (exit $LASTEXITCODE)"

Write-Host "`n=== both arms finished; checkpoints: ==="
Get-ChildItem runs\arm_original\*.pt, runs\arm_shared\*.pt -EA SilentlyContinue | Select-Object FullName
