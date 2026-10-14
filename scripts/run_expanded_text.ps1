# The data-lever experiment: the SAME configuration on the enlarged corpus, compared at matched steps.
#
# Registered before the result, because the comparison is free. The current long run (runs/text_long384) uses
# runs/v3_fixed: 4533 records over 3744 distinct prompts, dim 384 with 6 layers, batch 16. This run uses
# runs/v4_fixed, the same config and the same step count, so the only variable is text diversity, which the expansion
# takes from 3744 to about 8100 distinct prompts.
#
# What is already measured at matched steps, all on the same 60 held-out items:
#   step 1000   ladder (418 texts, 9.62M)  0.0932     long run (3744 texts, ~30M)  0.0884
#   step 2000   ladder                     0.1783     long run                      0.1741
# Two independent settings land within 0.004 of each other at both points, which is the measurement's own noise, so
# text count and capacity have not helped YET. A larger pool normally pays off later rather than at matched update
# counts, because it delays memorisation, which is why the comparison has to run past 2000 steps to mean anything.
#
# PREDICTION, registered now:
#   - if the expanded run beats the current run by more than 0.01 at matched steps of 4000 or beyond, text diversity
#     is the lever and the next spend is more prompts;
#   - if the two curves are within 0.01 of each other all the way out, text diversity is NOT the lever at this scale
#     either, and the objective's stop rule applies: report the ceiling instead of growing further.
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$out = "runs\text_v4"
New-Item -ItemType Directory -Force $out | Out-Null

Write-Host "[text_v4] training 10000 steps on the expanded cache, dim 384 / 6 layers"
& $py -u scripts\train.py --config configs\parakeet_tiny.yaml --stage distill-text `
    --cache runs\v4_fixed --steps 10000 --batch-size 16 --device cpu `
    --set text.dim=384 --set text.n_layers=6 `
    --set flow.text_dim=384 --set flow.cond_dim=384 `
    --set train.save_every=1000 --out $out > "$out\train.log" 2>&1
Write-Host "[text_v4] training finished with exit $LASTEXITCODE"

$last = Get-ChildItem "$out\*.pt" -EA SilentlyContinue |
        Where-Object { $_.Name -match 'step(\d+)' } |
        Sort-Object { [int]([regex]::Match($_.Name, 'step(\d+)').Groups[1].Value) } |
        Select-Object -Last 1
if ($last) {
  Write-Host "[text_v4] WER on $($last.Name)"
  & $py -u scripts\real_eval.py --config configs\parakeet_tiny.yaml --checkpoint $last.FullName `
      --cache runs\holdout_cache --limit 24 --out "runs\eval_text_v4.json" > "$out\eval.log" 2>&1
  $report = "runs\eval_text_v4.json\report.json"
  if (Test-Path $report) {
    $d = Get-Content $report -Raw | ConvertFrom-Json
    "text_v4 {0}  WER {1:N4}  control {2:N4}  dnsmos {3:N3}" -f $last.Name, $d.wer.student, $d.wer.teacher, $d.naturalness.student |
      Tee-Object -FilePath "$out\report.txt" -Append | Write-Host
  }
}
