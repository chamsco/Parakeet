# Read held-out frame rho from every new checkpoint of the long run, while it trains.
#
# The point is incremental evidence instead of one endpoint. A long run whose curve is readable at 1000-step
# intervals can be stopped the moment it flattens, which is the discipline this project uses everywhere else.
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$out = "runs\text_long384"
$report = "$out\report.txt"
$done = @{}
if (Test-Path $report) {
  Get-Content $report | ForEach-Object { if ($_ -match 'step(\d+)\.pt\s+heldout_frame_rho') { $done[$Matches[1]] = $true } }
}

while ($true) {
  $training = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
              Where-Object { $_.CommandLine -match 'text_long384' -and $_.CommandLine -match 'train.py' }
  $checkpoints = Get-ChildItem "$out\*.pt" -EA SilentlyContinue |
                 Where-Object { $_.Name -match 'step(\d+)' } |
                 Sort-Object { [int]([regex]::Match($_.Name, 'step(\d+)').Groups[1].Value) }

  foreach ($ck in $checkpoints) {
    $step = [regex]::Match($ck.Name, 'step(\d+)').Groups[1].Value
    if ($done.ContainsKey($step)) { continue }
    & $py scripts\token_route_ceiling.py --checkpoint $ck.FullName --cache runs\holdout_cache --items 60 > $null 2>&1
    $json = Get-Content runs\token_route_ceiling.json -Raw | ConvertFrom-Json
    $line = "{0,-30} heldout_frame_rho {1,8:N4}  heldout_token_rho {2,8:N4}" -f $ck.Name, $json.frame_rho, $json.token_rho
    $line | Tee-Object -FilePath $report -Append | Write-Host
    $done[$step] = $true
  }

  if (-not $training) {
    Write-Host "[monitor] training process is gone, stopping"
    break
  }
  Start-Sleep -Seconds 120
}
