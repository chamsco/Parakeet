# Read held-out frame rho from every new checkpoint of a text-side run, while it trains.
#
# Parameterised so one script serves the current long run and the expanded-corpus run that follows it. The point is
# incremental evidence instead of one endpoint: a curve readable at 1000-step intervals can be stopped the moment it
# flattens, which is the discipline this project uses everywhere else.
#
# REPEATS DEFAULTS TO THREE, and that is a correction rather than a nicety. Every reading in this project's older logs
# is a single one, and five separate times a single reading has disagreed with a repeated mean by more than the effect
# being claimed: the same checkpoint moves by up to 0.0204, ten readings of five checkpoints gave sd 0.0201, the step-4000
# point read 0.3548 against a five-reading mean of 0.3612, and a baseline step-4000 reading came in at 0.1920 against
# 0.2211. A single reading is therefore not a cheap measurement, it is a coin flip, so the reading written to the log now
# carries its own spread beside it.
param(
  [Parameter(Mandatory = $true)][string]$RunDir,
  [Parameter(Mandatory = $true)][string]$Cache,
  [int]$Items = 60,
  [int]$Repeats = 3
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
    $frames = @()
    $tokens = @()
    foreach ($pass in 1..[Math]::Max(1, $Repeats)) {
      & $py scripts\token_route_ceiling.py --checkpoint $ck.FullName --cache $Cache --items $Items > $null 2>&1
      $json = Get-Content runs\token_route_ceiling.json -Raw | ConvertFrom-Json
      $frames += [double]$json.frame_rho
      $tokens += [double]$json.token_rho
    }
    $frameMean = ($frames | Measure-Object -Average).Average
    $frameSpread = ($frames | Measure-Object -Maximum).Maximum - ($frames | Measure-Object -Minimum).Minimum
    $tokenMean = ($tokens | Measure-Object -Average).Average
    $line = "{0,-30} heldout_frame_rho {1,8:N4}  heldout_token_rho {2,8:N4}  spread {3,7:N4}  n={4}" -f `
            $ck.Name, $frameMean, $tokenMean, $frameSpread, $frames.Count
    $line | Tee-Object -FilePath $report -Append | Write-Host
    $done[$step] = $true
  }

  if (-not $training) {
    Write-Host "[monitor $RunDir] training process is gone, stopping"
    break
  }
  Start-Sleep -Seconds 120
}
