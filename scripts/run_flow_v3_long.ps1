# The last untested dimension: training time.
#
# Eight conditioning interventions are closed (gain, pooled token, plan, residual plan, logit-normal t, a 4.2x
# corpus, a pretrained text encoder, classifier-free guidance), and none moved the flow term out of the 2-7% band
# or sampled rho above 0.09.  What no run has tested is simply MORE STEPS: every flow experiment here has been
# 2000-4000 steps, while SupertonicTTS trains its 44M module on 945 h with a schedule to match, and the token
# route's own history showed the mapping appearing only after several thousand steps.
#
# So this continues arm A's step-2000 checkpoint for 8000 more steps on the full corpus, with the gate every 1000
# so the trend has eight points.
#
# PRE-REGISTERED READOUT (written before the run):
#   rho > 0.20 by step 8000  -> training time is the lever; the route is plausible with more compute
#   rho 0.05-0.20            -> trending but far from the 0.75 bar; state the slope and extrapolate
#   rho <= 0.05, flat        -> the route is closed at this model size, whatever the step budget
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path runs\ab_fresh\flow_step2000.pt)) { Write-Host "no warm-start checkpoint"; exit 1 }
if (-not (Test-Path runs\expanded_v3\index.json)) { Write-Host "no expanded_v3 cache"; exit 1 }

& .\.venv\Scripts\python.exe -u scripts\overfit_flow.py `
    --config configs\parakeet_flow_plan.yaml `
    --cache runs\expanded_v3 --items 4501 --holdout 32 `
    --steps 8000 --batch-size 16 --crop-frames 160 --lr 0.0003 `
    --set text.dim=512 --set text.n_layers=6 --set flow.text_dim=512 --set flow.cond_dim=512 `
    --warm-start runs\ab_fresh\flow_step2000.pt `
    --gate-every 1000 --save-every 2000 --out runs\flow_v3_long
