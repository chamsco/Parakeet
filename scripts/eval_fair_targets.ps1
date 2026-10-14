# The FAIR evaluation of the shared-target arms.
#
# Round 145 established that the first evaluation was invalid: it scored both arms by rho against the ORIGINAL
# held-out renditions, but arm S was trained on the rendition-INDEPENDENT signal (std 0.54 against ~1.0), so a lower
# rho is what predicting a different signal looks like, not evidence of failure.  And WER was 1.000 for both arms --
# saturated, unable to separate two unintelligible models.
#
# This fixes both halves.  Punctuation variants of the 400 HELD-OUT prompts give a shared target for data no run has
# seen, so each arm can be scored against the signal it was actually trained to predict -- and the samples are still
# scored by WER with the recogniser's control, which is the outcome that matters.
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path runs\holdout_shared_cache\index.json)) {
  if ((Get-ChildItem data\holdout_variants_v1\wav -Filter *.wav -EA SilentlyContinue).Count -lt 400) {
    Write-Host "[fair] held-out variants still generating: $((Get-ChildItem data\holdout_variants_v1\wav -Filter *.wav -EA SilentlyContinue).Count)/400"
    exit 1
  }
  Write-Host "[fair] building the held-out SHARED target cache"
  & $py -u scripts\make_shared_target_cache.py --base runs\holdout_cache `
      --base-manifest data\holdout_v1\manifest.jsonl `
      --variants data\holdout_variants_v1\manifest.jsonl --out runs\holdout_shared_cache
  Write-Host "[fair] cache exit $LASTEXITCODE"
}

Write-Host "`n=== rho against the signal each arm was trained to predict ==="
Write-Host "    arm O and the baseline were trained on ORIGINAL targets -> score against runs\holdout_cache"
Write-Host "    arm S was trained on SHARED targets                     -> score against runs\holdout_shared_cache"
foreach ($m in @(
  @{ n = "baseline_step5000";          ck = "runs\text_continue\distill-text_step5000.pt"; cache = "runs\holdout_cache" },
  @{ n = "armO_step1000_originals";    ck = "runs\arm_original\distill-text_step1000.pt";  cache = "runs\holdout_cache" },
  @{ n = "armS_step1000_sharedTARGET"; ck = "runs\arm_shared\distill-text_step1000.pt";     cache = "runs\holdout_shared_cache" },
  @{ n = "armO_vs_shared_target";      ck = "runs\arm_original\distill-text_step1000.pt";   cache = "runs\holdout_shared_cache" }
)) {
  if (-not (Test-Path $m.ck)) { Write-Host "  $($m.n): no checkpoint"; continue }
  $out = & $py scripts\token_route_ceiling.py --checkpoint $m.ck --cache $m.cache --items 40 2>&1
  $line = ($out | Where-Object { $_ -match "FRAMES" }) -join " "
  Write-Host ("  {0,-26} {1}" -f $m.n, ($line -replace '\s+', ' '))
}

Write-Host "`n=== WER on held-out samples (recorded even though it is saturated in this regime) ==="
foreach ($arm in @(@{n="armO"; ck="runs\arm_original\distill-text_step1000.pt"},
                   @{n="armS"; ck="runs\arm_shared\distill-text_step1000.pt"})) {
  if (-not (Test-Path $arm.ck)) { continue }
  & $py -u scripts\real_eval.py --config configs\parakeet_tiny.yaml --checkpoint $arm.ck `
      --cache runs\holdout_cache --limit 24 --out "runs\fair_eval_$($arm.n).json" > $null 2>&1
  $report = "runs\fair_eval_$($arm.n).json\report.json"
  if (Test-Path $report) {
    $d = Get-Content $report -Raw | ConvertFrom-Json
    Write-Host ("  {0,-6} WER {1}  control {2}" -f $arm.n, [math]::Round($d.wer.student, 3), [math]::Round($d.wer.teacher, 3))
  }
}
