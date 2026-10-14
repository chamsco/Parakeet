# Arm B of the text-side A/B: the mean-invariant latent term, which this run does not have.
#
# The diagnosis and the fix both already existed and I re-derived them from the audio before finding them. Round 30
# measured the text side predicting the AVERAGE latent (flattened cosine 0.805 against a per-dimension correlation of
# only 0.126), concluded that MSE between raw values rewards exactly that, and added `signal_latent_contrast`, a
# centred MSE that is blind to the mean and cannot be satisfied by collapsing to it. This round's real_eval on arm A
# shows the same failure in the waveform: voiced fraction 1.000 against the teacher's 0.565, median F0 93.75 Hz, which
# is exactly the 24000/256 hop rate, with a log-mel cosine of 0.9192. A frame-rate buzz with the right gross spectrum
# is a mean prediction.
#
# The recorded result of the fix, 2400 steps warm-started with signal_latent_contrast=1.0 and audio_aux=1.0:
#   per-dimension correlation  train 0.126 -> 0.200 (+59 %)   validation 0.107 -> 0.178 (+66 %)
#   WER, unseen Kokoro prompts 1.000 -> 1.000
# So the term moved the quantity it was built for and intelligibility did not follow. That is arm B's purpose here:
# not to reach WER 0 in one run, but to test whether the +66 % survives in THIS configuration at matched steps, on
# the 400-prompt held-out set, with the same instrument as arm A.
#
# `configs/parakeet_tiny.yaml` does not set the flag, so arm A trains with `latent_contrast = 0.0`. Arm B is arm A
# plus the two flags round 30 used. Registered prediction: if arm B's held-out rho is more than 0.02 above arm A's at
# matched steps, the mean-invariance term is worth keeping in the token route's config; if the gap is inside the
# 0.004 measurement noise, the +59 % that round 30 measured does not survive at this scale and the term should stay
# out of the config rather than be carried along on the strength of an earlier configuration.
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$out = "runs\text_contrast"
New-Item -ItemType Directory -Force $out | Out-Null

Write-Host "[contrast] arm B: 20000 steps, dim 384 / 6, cache runs\v3_fixed, latent_contrast 1.0 and audio_aux 1.0"
& $py -u scripts\train.py --config configs\parakeet_tiny.yaml --stage distill-text `
    --cache runs\v3_fixed --steps 20000 --batch-size 16 --device cpu `
    --set text.dim=384 --set text.n_layers=6 `
    --set flow.text_dim=384 --set flow.cond_dim=384 `
    --set train.loss.signal_latent_contrast=1.0 --set train.loss.audio_aux=1.0 `
    --set train.save_every=1000 --out $out > "$out\train.log" 2>&1
Write-Host "[contrast] finished with exit $LASTEXITCODE"
