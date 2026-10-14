# Read held-out frame rho from every new checkpoint of a text-side run, while it trains.
#
# Parameterised so one script serves the current long run and the expanded-corpus run that follows it. The point is
# incremental evidence instead of one endpoint: a curve readable at 1000-step intervals can be stopped the moment it
# flattens, which is the discipline this project uses everywhere else.
param(
  [Parameter(Mandatory = $true)][string]$RunDir,
  [Parameter(Mandatory = $true)][string]$Cache,
  [int]$Items = 60
)
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$report = Join-Path $RunDir "report.txt"
New-Item -ItemType Directory -Force $RunDir | Out-Null

$done = @{}
if (Test-Path $report) {
  Get-Content $report | ForEach-Object { if ($_ -match 'step(\d+)\.pt\s+heldout_frame_rho') { $done[$Matches[1]] = $true } }
}

while ($true) {
  $training = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
              Where-Object { $_.CommandLine -match [regex]::Escape($RunDir.Replace('\','\')) -and $_.CommandLine -match 'train.py' }
  $checkpoints = Get-ChildItem (Join-Path $RunDir "*.pt") -EA SilentlyContinue |
                 Where-Object { $_.Name -match 'step(\d+)' } |
                 Sort-Object { [int]([regex]::Match($_.Name, 'step(\d+)').Groups[1].Value) }

  foreach ($ck in $checkpoints) {
    $step = [regex]::Match($ck.Name, 'step(\d+)').Groups[1].Value
    if ($done.ContainsKey($step)) { continue }
    & $py scripts\token_route_ceiling.py --checkpoint $ck.FullName --cache $Cache --items $Items > $null 2>&1
    $json = Get-Content runs\token_route_ceiling.json -Raw | ConvertFrom-Json
    $line = "{0,-30} heldout_frame_rho {1,8:N4}  heldout_token_rho {2,8:N4}" -f $ck.Name, $json.frame_rho, $json.token_rho
    $line | Tee-Object -FilePath $report -Append | Write-Host
    $done[$step] = $true
  }

  if (-not $training) {
    Write-Host "[monitor $RunDir] training process is gone, stopping"
    break
  }
  Start-Sleep -Seconds 120
}
