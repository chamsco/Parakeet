# Compare two text-side runs at matched steps, with the readout repeated and its spread reported.
#
# Why this exists. The held-out rho readout has a resolution of about +-0.02: the SAME checkpoint, measured twice back
# to back, moved by up to 0.0204, and ten readings of five checkpoints gave sd 0.0201 (section 137 of the verification
# log). Single readings therefore cannot resolve the differences this project needs, and two experiments were designed
# and read with a 0.02 threshold against a 0.02 noise floor before that was measured.
#
# So: one checkpoint, several readings, and a reported spread. The comparison between runs is only called when the
# difference clears the combined spread of both.
#
# Usage:
#   pwsh scripts\compare_runs.ps1 -Runs "runs/text_long384,runs/text_contrast_ab" -Steps 1000,2000,3000 -Repeats 3
param(
  [Parameter(Mandatory = $true)][string]$Runs,        # comma-separated run directories
  [Parameter(Mandatory = $true)][int[]]$Steps,
  [int]$Repeats = 3,
  [string]$Cache = "runs\holdout_cache",
  [int]$Items = 60
)
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$runList = $Runs -split "," | ForEach-Object { $_.Trim() }

Write-Host "repeats per checkpoint: $Repeats     held-out items: $Items"
Write-Host ("{0,-24} {1,6}  {2,9} {3,9} {4,9}  {5}" -f "run", "step", "mean", "min", "max", "readings")
$results = @{}
foreach ($run in $runList) {
  foreach ($step in $Steps) {
    $ck = Join-Path $run "distill-text_step$step.pt"
    if (-not (Test-Path $ck)) { continue }
    $values = @()
    foreach ($i in 1..$Repeats) {
      & $py scripts\token_route_ceiling.py --checkpoint $ck --cache $Cache --items $Items > $null 2>&1
      $json = Get-Content runs\token_route_ceiling.json -Raw | ConvertFrom-Json
      $values += [double]$json.frame_rho
    }
    $mean = ($values | Measure-Object -Average).Average
    $min = ($values | Measure-Object -Minimum).Minimum
    $max = ($values | Measure-Object -Maximum).Maximum
    $results["$run|$step"] = [pscustomobject]@{ Run = $run; Step = $step; Mean = $mean; Min = $min; Max = $max }
    Write-Host ("{0,-24} {1,6}  {2,9:N4} {3,9:N4} {4,9:N4}  {5}" -f `
      (Split-Path $run -Leaf), $step, $mean, $min, $max, (($values | ForEach-Object { "{0:N4}" -f $_ }) -join " "))
  }
}

if ($runList.Count -ge 2) {
  Write-Host "`nmatched-step differences:"
  foreach ($step in $Steps) {
    $a = $results["$($runList[0])|$step"]; $b = $results["$($runList[1])|$step"]
    if (-not $a -or -not $b) { continue }
    $spread = [math]::Max($a.Max - $a.Min, $b.Max - $b.Min)
    $diff = $b.Mean - $a.Mean
    $verdict = if ([math]::Abs($diff) -gt $spread) { "SEPARATED" } else { "inside the spread" }
    Write-Host ("  step {0,6}  {1:N4} vs {2:N4}   diff {3:+0.0000;-0.0000}   largest spread {4:N4}   -> {5}" -f `
      $step, $a.Mean, $b.Mean, $diff, $spread, $verdict)
  }
}
