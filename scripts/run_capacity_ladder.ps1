# Capacity ladder on the text side, read on the in-domain held-out set.
#
# Why this and not another long run. The thread's measurements say:
#   - the model fits in sample (rho 0.82) and generalises at about 0.15 on unseen text, so the wall is generalisation;
#   - corpus size was nearly flat across 304 and 1207 items (0.139 against 0.151);
#   - capacity is the one lever with a measured effect: 4.6M to 22.6M moved held-out fit 0.118 to 0.190.
# So this run isolates capacity at fixed data and fixed steps, and nothing else. Two arms, same 1207 base items, same
# 3000 steps, same warm start (none), differing only in text width and depth.
#
# The readout is held-out frame rho against the 400-prompt corpus built from four books no training run used
# (data/holdout_v1 -> runs/holdout_cache), plus WER with the recogniser's control at the end. Pre-registered:
#   - if the wide arm beats the narrow arm by more than the 0.004 within-session measurement sd, capacity buys
#     generalisation here and the next run should push it further on the full 4533-item cache;
#   - if the two arms land within noise of each other, capacity is exhausted at this data scale and the remaining
#     lever is corpus size, which the driver then starts.
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$log = "runs\ladder\report.txt"
New-Item -ItemType Directory -Force runs\ladder | Out-Null

$arms = @(
  @{ name = "narrow_d256_l4"; dim = 256; layers = 4 },
  @{ name = "wide_d768_l8";   dim = 768; layers = 8 }
)

foreach ($arm in $arms) {
  $out = "runs\ladder\$($arm.name)"
  Write-Host "[ladder] training $($arm.name)  dim=$($arm.dim) layers=$($arm.layers)"
  Remove-Item -Recurse -Force $out -EA SilentlyContinue
  & $py -u scripts\train.py --config configs\parakeet_tiny.yaml --stage distill-text `
      --cache runs\mixed_v2\latent_cache --steps 3000 --batch-size 16 --device cpu `
      --set "text.dim=$($arm.dim)" --set "text.n_layers=$($arm.layers)" `
      --set "flow.text_dim=$($arm.dim)" --set "flow.cond_dim=$($arm.dim)" `
      --set "train.save_every=1000" --out $out > "$out.log" 2>&1
  Write-Host "[ladder] $($arm.name) exit $LASTEXITCODE"

  # held-out rho at every checkpoint, so the slope is visible and not just the endpoint
  Get-ChildItem "$out\*.pt" | Sort-Object { [int]([regex]::Match($_.Name, 'step(\d+)').Groups[1].Value) } | ForEach-Object {
    & $py scripts\token_route_ceiling.py --checkpoint $_.FullName --cache runs\holdout_cache --items 60 > $null 2>&1
    $json = Get-Content runs\token_route_ceiling.json -Raw | ConvertFrom-Json
    $line = "{0,-18} {1,-26} heldout_frame_rho {2,8:N4}" -f $arm.name, $_.Name, $json.frame_rho
    $line | Tee-Object -FilePath $log -Append | Write-Host
  }

  # WER on 24 held-out utterances with the recogniser's control
  $last = Get-ChildItem "$out\*.pt" | Sort-Object { [int]([regex]::Match($_.Name, 'step(\d+)').Groups[1].Value) } | Select-Object -Last 1
  & $py -u scripts\real_eval.py --config configs\parakeet_tiny.yaml --checkpoint $last.FullName `
      --cache runs\holdout_cache --limit 24 --out "runs\ladder\eval_$($arm.name).json" > "$out.eval.log" 2>&1
  $report = "runs\ladder\eval_$($arm.name).json\report.json"
  if (Test-Path $report) {
    $d = Get-Content $report -Raw | ConvertFrom-Json
    $wer = "{0,-18} WER {1,7:N4}  control {2,7:N4}  dnsmos {3,6:N3}" -f $arm.name, $d.wer.student, $d.wer.teacher, $d.naturalness.student
    $wer | Tee-Object -FilePath $log -Append | Write-Host
  }
}

Write-Host "`n[ladder] done. report at $log"
Get-Content $log
