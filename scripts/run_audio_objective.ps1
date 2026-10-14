# Continue the through-decoder objective, the only one in this project with a WER trend pointing down.
#
# THE MEASURED TREND, from docs/evidence/objective_ab.json (runs/real_audio_long and siblings):
#
#   latent L1 on cached latents (distill-text), 800 steps          WER 1.648   DNSMOS 1.633   mel cosine 0.9427
#   audio mel+spectral through the frozen decoder, 600 steps       WER 1.000   DNSMOS 1.586   mel cosine 0.9447
#   the same objective, 2400 steps                                 WER 0.667   DNSMOS 1.583   mel cosine 0.9486
#   teacher control                                                 WER 0.000                  DNSMOS 2.613
#
# 0.667 is the best model-level WER anywhere in this project's evidence, and it is two-thirds of words wrong, so this is
# a direction rather than a solution. Two things make it worth CPU now:
#
#   * the run's own record shows lr 1.02e-05 at step 2400, so its schedule was fully annealed: the endpoint was already
#     the end of that schedule, and "does the gain keep coming?" was never asked past it;
#   * every other objective in this project reads 1.000 or worse, including the one the corpus comparison uses.
#
# THE CONFIGURATION IS THE SAME ONE THAT PRODUCED 0.667, from runs/real_audio_long/run.json: stage distill-audio, variant
# tiny, text 256x4, batch 16, lr 2e-4, audio_mel 1, audio_spectral 3, audio_aux 0.1, and latent_rate 3 which the code
# derives from the cache rather than from the config. Only the step budget changes, from 2400 to 12000, so the schedule
# anneals over the longer run.
#
# PREDICTION, registered before the result: if WER keeps falling the way it fell from 600 to 2400 steps, 12000 steps
# should put it below 0.5 and the objective question is answered in the affirmative. If it settles near 0.6, the
# through-decoder direction is exhausted at this data scale too, and the two remaining statements about this machine are
# that the latent-L1 text side plateaus at rho 0.22 and the through-decoder side plateaus near two-thirds wrong.
#
# WER is read at three checkpoints, not all twelve, because each real_eval pass costs about a minute of contention.
param(
  [int]$Steps = 12000,
  [string]$Out = "runs\text_audio_long"
)
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force $Out | Out-Null

Write-Host "[audio_long] distill-audio, $Steps steps, cache runs\v3_fixed (latent_rate derived = 3)"
& $py -u scripts\train.py --config configs\parakeet_tiny.yaml --stage distill-audio `
    --cache runs\v3_fixed --steps $Steps --batch-size 16 --device cpu `
    --set train.save_every=1000 --out $Out > "$Out\train.log" 2>&1
Write-Host "[audio_long] training exit $LASTEXITCODE"

$checkpoints = Get-ChildItem "$Out\*.pt" -EA SilentlyContinue |
               Where-Object { $_.Name -match 'step(\d+)' } |
               Sort-Object { [int]([regex]::Match($_.Name, 'step(\d+)').Groups[1].Value) }
if (-not $checkpoints) { Write-Host "[audio_long] no checkpoints"; exit 1 }

# held-out rho at every checkpoint, so the text-side fit is comparable with the plateau already measured
New-Item -ItemType Directory -Force runs\text_audio_long | Out-Null
foreach ($ck in $checkpoints) {
  & $py scripts\token_route_ceiling.py --checkpoint $ck.FullName --cache runs\holdout_cache --items 60 > $null 2>&1
  $json = Get-Content runs\token_route_ceiling.json -Raw | ConvertFrom-Json
  ("{0,-30} heldout_frame_rho {1,8:N4}  heldout_token_rho {2,8:N4}" -f $ck.Name, $json.frame_rho, $json.token_rho) |
    Tee-Object -FilePath "$Out\report.txt" -Append | Write-Host
}

Write-Host "[audio_long] WER at three checkpoints"
foreach ($step in @(2000, 6000, 12000)) {
  $ck = Join-Path $Out "distill-audio_step$step.pt"
  if (-not (Test-Path $ck)) { continue }
  & $py -u scripts\real_eval.py --config configs\parakeet_tiny.yaml --checkpoint $ck `
      --cache runs\holdout_cache --limit 24 --out "runs\eval_audio_long_step$step" > $null 2>&1
  $report = "runs\eval_audio_long_step$step\report.json"
  if (Test-Path $report) {
    $d = Get-Content $report -Raw | ConvertFrom-Json
    ("audio_long step {0,6}  WER {1,7:N4}  control {2,7:N4}  dnsmos {3,6:N3}" -f $step, $d.wer.student, $d.wer.teacher, $d.naturalness.student) |
      Tee-Object -FilePath "$Out\report.txt" -Append | Write-Host
  }
}
