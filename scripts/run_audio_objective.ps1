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
# CORRECTION TO THE COMPARISON, found by reading both runs' recorded metadata rather than assuming they matched. That
# 0.667 was measured on a SMALL corpus: runs/real_audio_long/run.json records n_shards 1 and only three voices
# (af_bella, af_heart, af_sky), and its evaluation ran on kept.jsonl with SIX utterances. This run trains on
# runs/v3_fixed: 4533 items, 11 voices, 3 shards, scored on 24 held-out utterances from four unseen books. Fewer voices
# on one shard is a much easier problem, and a six-utterance WER is noisy to the point that a single word moves it by
# about 0.17. So 0.667 is not the number to beat here and the two runs are not comparable.
#
# THE RIGHT BASELINE is the same cache and the same holdout. runs/text_long384, the latent L1 objective on
# runs/v3_fixed, reads WER 1.0000 on 24 held-out utterances, and its held-out rho plateaus at 0.2223.
#
# PREDICTION, restated against that baseline: this run reads BELOW 1.0000 at 4000 steps if training through the decoder
# beats regressing cached latents on the same data, and at or above 1.0000 if the objective's advantage on tiny data
# does not survive 4533 items across 11 voices. A WER near 0.6 here would be a far stronger result than the 0.667 it
# replaces, and a WER of 1.0 would mean the objective difference was a small-corpus effect.
#
# WER is read at two checkpoints, not all four. Measured cost: 3.60 s/step, so the budget is set by wall clock, and a
# 4000-step run is about four hours alone and longer under contention.
param(
  [int]$Steps = 4000,
  [string]$Out = "runs\text_audio_long"
)
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force $Out | Out-Null

Write-Host "[audio_long] distill-audio, $Steps steps, cache runs\v3_fixed (latent_rate derived = 3)"
# warmup is pinned to the measured run's 1000. configs/parakeet_tiny.yaml sets 200, which is 5 % of a 4000-step
# schedule against the 42 % that 1000 was of the measured 2400, so leaving it would confound the extension with a
# different schedule shape. Found by diffing this run's run.json against runs/real_audio_long/run.json field by field.
& $py -u scripts\train.py --config configs\parakeet_tiny.yaml --stage distill-audio `
    --cache runs\v3_fixed --steps $Steps --batch-size 16 --device cpu `
    --set train.warmup_steps=1000 `
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

Write-Host "[audio_long] WER at two checkpoints"
foreach ($step in @(2000, 4000)) {
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
