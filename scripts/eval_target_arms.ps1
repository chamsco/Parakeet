# Evaluate the matched target experiment on the HELD-OUT corpus.
#
# Round 139 showed this model memorises (the token route reads rho 0.82 in-sample and 0.15 out), so an arm's own
# training loss says nothing about whether it produces better speech.  Everything here is measured on data no run
# has seen: 400 prompts from four Gutenberg books the corpus was never built from, with the exclusion of every
# sentence already in the corpus asserted to be empty.
#
# Three models, same held-out cache, same settings:
#   baseline   runs/text_continue/distill-text_step5000.pt   the token route before this experiment
#   arm O      runs/arm_original/...                        trained on the ORIGINAL targets
#   arm S      runs/arm_shared/...                          trained on the rendition-independent targets
#
# WER with the recogniser's own control is the readout, because it is the metric that applies and the one this
# project's models score 1.000 on.  dnsmos is reported alongside as a check, not as a decision.
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$limit = 8

# the held-out cache must be built WITH the autoencoder: without --ae-checkpoint the latents come from a random
# encoder, which is the trap that produced a garbage dialogue cache a few rounds ago.
if (-not (Test-Path runs\holdout_cache\index.json)) {
  Write-Host "[eval] caching the held-out corpus with the trained autoencoder"
  & $py -u scripts\make_teacher_corpus.py --corpus data\holdout_v1 --cache-only `
      --cache-out runs\holdout_cache --ae-checkpoint runs\ae_scaled\adversarial\autoencoder_last.pt `
      --latent-rate 3 > runs\holdout_cache.log 2>&1
  Write-Host "[eval] cache exit $LASTEXITCODE"
} else {
  Write-Host "[eval] held-out cache already present"
}

$models = @(
  @{ name = "baseline_step5000"; ckpt = "runs\text_continue\distill-text_step5000.pt" },
  @{ name = "armO_original";     ckpt = "" },
  @{ name = "armS_shared";       ckpt = "" }
)
# pick each arm's LAST checkpoint by STEP NUMBER.  Sorting by name picks distill-text_step500.pt over
# distill-text_step1000.pt, because "1" sorts before "5" -- the first run of this script evaluated step 500 while
# step 1000 existed.
foreach ($arm in @(@{k="armO_original";d="runs\arm_original"}, @{k="armS_shared";d="runs\arm_shared"})) {
  $last = Get-ChildItem "$($arm.d)\*.pt" -EA SilentlyContinue |
          Sort-Object { [int]([regex]::Match($_.Name, 'step(\d+)').Groups[1].Value) } |
          Select-Object -Last 1
  if ($last) {
    ($models | Where-Object { $_.name -eq $arm.k }).ckpt = $last.FullName
  }
}

foreach ($model in $models) {
  if (-not $model.ckpt) { Write-Host "[eval] $($model.name): no checkpoint yet"; continue }
  Write-Host "`n[eval] $($model.name)  <-  $($model.ckpt)"
  & $py -u scripts\real_eval.py --config configs\parakeet_tiny.yaml --checkpoint $model.ckpt `
      --cache runs\holdout_cache --limit $limit --out "runs\eval_$($model.name).json" 2>&1 |
      Select-String -Pattern "student WER|teacher WER|dnsmos|RESULT|PASS|FAIL" | Select-Object -ExpandProperty Line
}

Write-Host "`n=== summary: WER on data no run has seen ==="
Get-ChildItem runs\eval_*.json -EA SilentlyContinue | ForEach-Object {
  $dir = $_.FullName
  $report = Join-Path $dir "report.json"
  if (Test-Path $report) {
    $data = Get-Content $report -Raw | ConvertFrom-Json
    Write-Host ("  {0,-22} WER {1}  (control {2})  dnsmos {3}" -f $data.wer.recogniser, $data.wer.student, $data.wer.teacher, $data.naturalness.student)
  }
}
