# The contrast A/B, second attempt. The first attempt was two identical runs, and this file says why.
#
# WHAT THE FIRST ATTEMPT GOT WRONG. This launched a "contrast arm" with
#   --set train.loss.signal_latent_contrast=1.0 --set train.loss.audio_aux=1.0
# and both flags were silently inert. `run_stage` built its text criterion as `TextSideDistillLoss()` with no
# weights, and only `text_audio_step`, the AUDIO stage, read `cfg.train.loss.signal_latent_contrast`. So a
# `distill-text` run accepted the key and ignored it, and the two arms differed only by RNG. The instance now at
# runs/text_contrast is therefore a seed REPLICATE of runs/text_long384, not a contrast arm, and it is kept for the
# one thing it can still give: the training-seed spread of the whole curve. Its step-1000 held-out rho was 0.0854
# against the other arm's 0.0908, a spread of 0.0054 on identical settings.
#
# WHAT CHANGED. `parakeet/train/stages.py` now has `build_text_criterion(cfg)`, the text stage calls it, and
# `tests/test_latent_contrast.py` pins both that the default stays 0.0 and that a config value of 1.0 reaches the
# criterion. So the flag is live for this stage for the first time.
#
# WHAT THAT MEANS FOR THE INTERVENTION. Round 30 measured the mean-invariant term at +59 % train and +66 %
# validation per-dimension correlation, with WER unchanged at 1.000, and it did that on the AUDIO stage, which is
# where the key was reachable. The term has never been applied to the stage that actually trains the text side. That
# is a genuinely untested intervention, not a re-run of a finished experiment.
#
# PREDICTION, registered before the result: if this arm's held-out rho is more than 0.02 above runs/text_long384 at
# matched steps, the mean-invariant term belongs in the token route's config; if it is inside the 0.0054 seed spread,
# the term does not help this stage and the objective's remaining lever is steps alone.
param(
  [string]$Out = "runs\text_contrast_ab",
  [int]$Steps = 10000
)
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force $Out | Out-Null

Write-Host "[contrast_ab] $Steps steps, dim 384 / 6, cache runs\v3_fixed, latent_contrast 1.0"
& $py -u scripts\train.py --config configs\parakeet_tiny.yaml --stage distill-text `
    --cache runs\v3_fixed --steps $Steps --batch-size 16 --device cpu `
    --set text.dim=384 --set text.n_layers=6 `
    --set flow.text_dim=384 --set flow.cond_dim=384 `
    --set train.loss.signal_latent_contrast=1.0 `
    --set train.save_every=1000 --out $Out > "$Out\train.log" 2>&1
Write-Host "[contrast_ab] finished with exit $LASTEXITCODE"
