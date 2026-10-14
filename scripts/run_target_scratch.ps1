# The decisive version of the target experiment: train BOTH arms FROM SCRATCH.
#
# Round 144 tested the rendition-independent target by fine-tuning for 500 steps from a model trained on the
# ORIGINAL targets.  That is unfavourable: the model must first unlearn rendition-specific detail the original
# target rewarded, and 500 steps is not enough to do it.  The result was a tie on WER and a six-fold rho penalty.
#
# This runs the fair version.  Both arms start from the same randomly-initialised text side, train the same number
# of steps on the same 304 items with the same geometry, and differ only in the cache -- so the target is still the
# only variable, without the warm-start handicap.  Same readouts: rho against held-out targets (not saturated) and
# WER on held-out samples (saturated, but recorded).
#
# Budget: ~1.5 s/step on 304 items, so 3000 steps is roughly 75 minutes per arm.  Both arms and the evaluation are
# a few hours, which is why this is a background job that outlives any single round.
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$common = @(
  "--config", "configs\parakeet_tiny.yaml", "--stage", "distill-text",
  "--steps", "3000", "--batch-size", "16", "--device", "cpu",
  "--set", "text.dim=512", "--set", "text.n_layers=6",
  "--set", "flow.text_dim=512", "--set", "flow.cond_dim=512",
  "--set", "train.save_every=1000"
)

foreach ($arm in @(@{n="scratch_original"; c="runs\target_original"}, @{n="scratch_shared"; c="runs\target_shared"})) {
  Write-Host "[scratch] $($arm.n)  <-  $($arm.c)"
  Remove-Item -Recurse -Force "runs\$($arm.n)" -EA SilentlyContinue
  & $py -u scripts\train.py @common --cache $arm.c --out "runs\$($arm.n)" > "runs\$($arm.n).log" 2>&1
  Write-Host "[scratch] $($arm.n) done (exit $LASTEXITCODE)"
}

Write-Host "`n[scratch] both arms finished.  Readout next:"
Write-Host "  rho  : scripts/token_route_ceiling.py --checkpoint runs/scratch_X/distill-text_step3000.pt --cache runs/holdout_cache --items 40"
Write-Host "  WER  : scripts/real_eval.py --config configs/parakeet_tiny.yaml --checkpoint runs/scratch_X/... --cache runs/holdout_cache --limit 24"
