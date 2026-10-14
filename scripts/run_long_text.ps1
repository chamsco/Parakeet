# Long text-side run on the full cache, at the capacity the ladder showed is not the binding constraint.
#
# Why this replaces the wide arm. The ladder's narrow arm was still climbing when it stopped: held-out frame rho went
# 0.093 at step 1000, 0.178 at 2000, 0.215 at 3000, on 60 held-out items, against 0.199 for a 22.6M warm-started
# model read on the same items. A model that is still climbing at fixed data is under-trained, so "capacity helps"
# cannot be asked at a fixed 3000 steps without confounding it with under-training, and the wide arm cost 1.9 s/step
# against the narrow arm's 0.2. Steps are the lever the evidence points at, and steps are what CPU can buy.
#
# The run uses the FULL 4533-item cache rather than the 1207 base, dim 384 / 6 layers, 20000 steps, a checkpoint every
# 1000 so the curve is readable while it trains. scripts/monitor_long_text.ps1 reads held-out rho from each new
# checkpoint as it appears.
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$out = "runs\text_long384"
New-Item -ItemType Directory -Force $out | Out-Null

Write-Host "[long384] training 20000 steps, dim 384 / 6 layers, cache runs\v3_fixed (4533 items)"
& $py -u scripts\train.py --config configs\parakeet_tiny.yaml --stage distill-text `
    --cache runs\v3_fixed --steps 20000 --batch-size 16 --device cpu `
    --set text.dim=384 --set text.n_layers=6 `
    --set flow.text_dim=384 --set flow.cond_dim=384 `
    --set train.save_every=1000 --out $out > "$out\train.log" 2>&1
Write-Host "[long384] training finished with exit $LASTEXITCODE"

# final WER with the recogniser's control, on the last checkpoint, 24 held-out utterances
$last = Get-ChildItem "$out\*.pt" -EA SilentlyContinue |
        Where-Object { $_.Name -match 'step(\d+)' } |
        Sort-Object { [int]([regex]::Match($_.Name, 'step(\d+)').Groups[1].Value) } |
        Select-Object -Last 1
if ($last) {
  Write-Host "[long384] WER on $($last.Name)"
  & $py -u scripts\real_eval.py --config configs\parakeet_tiny.yaml --checkpoint $last.FullName `
      --cache runs\holdout_cache --limit 24 --out "runs\eval_long384.json" > "$out\eval.log" 2>&1
  $report = "runs\eval_long384.json\report.json"
  if (Test-Path $report) {
    $d = Get-Content $report -Raw | ConvertFrom-Json
    "long384 {0}  WER {1:N4}  control {2:N4}  dnsmos {3:N3}" -f $last.Name, $d.wer.student, $d.wer.teacher, $d.naturalness.student |
      Tee-Object -FilePath "$out\report.txt" -Append | Write-Host
  }
}
