# Conclude milestone 3: the corpus comparison, read with the power already calculated.
#
# The comparison model is runs/text_long384, the same configuration on the base cache: 4533 items over 3744 distinct
# prompts, held-out rho plateau 0.2223 with sd 0.0201 over steps 3000-7000 (section 137), WER 1.0000 on 24 held-out
# utterances. The new one is runs/text_v4, the same configuration on the merged cache, around 8890 items over about
# 8100 distinct prompts.
#
# FIVE readings per checkpoint, not three, because the power was calculated in advance: with three the standard error of
# a difference of means is about 0.0096 and a +0.02 difference is 2.1 standard errors, marginal; with five it is 2.7.
#
# The prediction is in scripts/run_milestone3.ps1: above +0.02 at matched steps of 3000 or beyond means text diversity
# does matter after all; inside the spread means it does not, and milestones 2 and 3 close with a negative result rather
# than an open one.
#
# WHERE THE COMPARISON IS FAIR, because the two runs have different step budgets and therefore different schedules.
# The comparison model ran 20000 steps with warmup 1000, so its cosine runs from step 1000 to 20000. The new one runs
# 10000 with warmup clamped to 500, so its cosine runs from 500 to 10000. The fraction of peak learning rate at matched
# steps is therefore:
#
#   step 1000   base 1.00   new 0.99
#   step 2000   base 0.99   new 0.95
#   step 3000   base 0.97   new 0.84
#   step 4000   base 0.95   new 0.71
#   step 5000   base 0.90   new 0.54
#
# So steps 1000 to 3000 are a fair comparison and steps 4000 and 5000 handicap the new arm by 24 and 36 points of peak
# learning rate. The verdict should be read on 1000 to 3000, and any win the new arm shows there is conservative. The
# endpoints are not comparable at all: different budgets, so a difference there mixes text with training compute.
#
# AND THE COMPARISON IS TEXT ONLY, verified rather than assumed. The base cache carries 21 voices; the expansion's 11
# Kokoro voices are all among them, adding ZERO new voices. So the merged corpus increases distinct prompts from 3744 to
# about 8100 and holds voices constant, which is what makes this a text-diversity test rather than a general "more data"
# test. Had it added voices, the result would have mixed two changes.
param(
  [string]$Base = "runs/text_long384",
  [string]$New = "runs/text_v4",
  [int]$Repeats = 5,
  [string]$Out = "runs/milestone3_conclusion.txt"
)
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"

function Log($line) { $line | Tee-Object -FilePath $Out -Append | Write-Host }

Log "=== milestone 3 conclusion: $Base against $New, $Repeats readings per checkpoint ==="
Log "base held-out plateau, from section 137: mean 0.2223, sd across steps 0.0201, max 0.2315"

Log ""
Log "--- held-out rho, matched steps ---"
.\scripts\compare_runs.ps1 -Runs "$Base,$New" -Steps 1000,2000,3000,4000,5000 -Repeats $Repeats 2>&1 |
  Where-Object { $_ -notmatch 'InvalidOperation|Line \||~~~|CategoryInfo|FullyQualified|^\s*\+' } |
  Tee-Object -FilePath $Out -Append | Out-Null
Get-Content $Out | Select-Object -Last 12 | Write-Host

Log ""
Log "--- in-sample against held-out for the new model, which is the under-fitting check ---"
$last = Get-ChildItem "$New\distill-text_step*.pt" -EA SilentlyContinue |
        Sort-Object { [int]([regex]::Match($_.Name, 'step(\d+)').Groups[1].Value) } | Select-Object -Last 1
if ($last) {
  & $py scripts\token_route_ceiling.py --checkpoint $last.FullName --cache runs\holdout_cache --items 60 > $null 2>&1
  $held = (Get-Content runs\token_route_ceiling.json -Raw | ConvertFrom-Json).frame_rho
  & $py scripts\token_route_ceiling.py --checkpoint $last.FullName --cache runs\v4_fixed --items 60 > $null 2>&1
  $insample = (Get-Content runs\token_route_ceiling.json -Raw | ConvertFrom-Json).frame_rho
  Log ("  {0}: in-sample {1:N4}  held-out {2:N4}  gap {3:N2}x" -f $last.Name, $insample, $held, ($insample / [math]::Max(0.0001, $held)))
}

Log ""
Log "--- held-out WER, milestone 3's second readout ---"
if ($last) {
  & $py -u scripts\real_eval.py --config configs\parakeet_tiny.yaml --checkpoint $last.FullName `
      --cache runs\holdout_cache --limit 24 --out runs\eval_text_v4 > $null 2>&1
  $report = "runs\eval_text_v4\report.json"
  if (Test-Path $report) {
    $d = Get-Content $report -Raw | ConvertFrom-Json
    Log ("  WER {0:N4}  control {1:N4}  dnsmos {2:N3}" -f $d.wer.student, $d.wer.teacher, $d.naturalness.student)
  } else { Log "  no eval report" }
}

Log ""
Log "--- and the same readout on the comparison model, so the two are quoted together ---"
if (Test-Path "runs\text_long384\distill-text_step7000.pt") {
  & $py -u scripts\real_eval.py --config configs\parakeet_tiny.yaml `
      --checkpoint runs\text_long384\distill-text_step7000.pt `
      --cache runs\holdout_cache --limit 24 --out runs\eval_long384_step7000 > $null 2>&1
  $report = "runs\eval_long384_step7000\report.json"
  if (Test-Path $report) {
    $d = Get-Content $report -Raw | ConvertFrom-Json
    Log ("  WER {0:N4}  control {1:N4}  dnsmos {2:N3}" -f $d.wer.student, $d.wer.teacher, $d.naturalness.student)
  }
}
Log ""
Log "written to $Out"
