# Shard a teacher synthesis across N processes, then merge the shards into one corpus directory.
#
# WHY THIS EXISTS. `make_teacher_corpus.py` is strictly serial - it has no --jobs flag and no worker path - and it runs
# at about 20 prompts per minute, which sets the growth loop's wall clock: about three hours of synthesis per round, and
# the two-scenario estimate for reaching the recogniser's 0.60 puts that at 24 to 42 hours for two or three more
# doublings. The machine has 16 logical cores and the synthesis uses about one. Sharding the prompt list and running N
# teachers side by side multiplies the one lever this project has measured moving anything, without touching the model.
#
# The merge is the part that needed care rather than the splitting: each shard's manifest is written with wav_path
# RELATIVE TO ITS OWN DIRECTORY, so the merged manifest has to rewrite those paths to shardN/... or every downstream
# reader - the cache builder, the validator, `expand_corpus.py` - silently resolves them against the wrong directory.
# That is the same class of defect as the five this project has already found in its measurement path, so the merge is
# verified rather than trusted: the merged manifest is parsed, every wav_path is resolved to a real file, and a missing
# one fails the script.
param(
  [Parameter(Mandatory = $true)][string]$Texts,
  [Parameter(Mandatory = $true)][string]$Out,
  [int]$Shards = 8,
  [string]$Teachers = "kokoro",
  [switch]$KeepShards
)
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Texts)) { Write-Host "[shard] no prompts at $Texts"; exit 1 }
New-Item -ItemType Directory -Force $Out | Out-Null

# --- 1. split round-robin, so every shard sees the same book and voice mix rather than a contiguous block of one book
$lines = Get-Content $Texts | Where-Object { $_.Trim().Length -gt 0 }
$buckets = @()
for ($i = 0; $i -lt $Shards; $i++) { $buckets += ,(New-Object System.Collections.ArrayList) }
for ($i = 0; $i -lt $lines.Count; $i++) { [void]$buckets[$i % $Shards].Add($lines[$i]) }
Write-Host ("[shard] {0} prompts across {1} shards: {2}" -f $lines.Count, $Shards, (($buckets | ForEach-Object { $_.Count }) -join ', '))

$procs = @()
for ($i = 0; $i -lt $Shards; $i++) {
  $dir = Join-Path $Out "shard$i"
  New-Item -ItemType Directory -Force $dir | Out-Null
  $promptFile = Join-Path $dir "prompts.txt"
  Set-Content -Path $promptFile -Value $buckets[$i] -Encoding utf8
  if ($buckets[$i].Count -eq 0) { continue }
  $procs += Start-Process -FilePath $py -PassThru -NoNewWindow -ArgumentList @(
    "-u", "scripts\make_teacher_corpus.py", "--texts", $promptFile, "--teachers", $Teachers,
    "--out", $dir, "--no-curate"
  ) -RedirectStandardOutput (Join-Path $dir "synth.out") -RedirectStandardError (Join-Path $dir "synth.err")
}
Write-Host "[shard] $($procs.Count) synthesis processes started"
foreach ($p in $procs) { $p.WaitForExit(); if ($p.ExitCode -ne 0) { Write-Host "[shard] a shard exited $($p.ExitCode)"; } }

# --- 2. merge: concatenate manifests, rewriting each wav_path so it resolves against the merged directory
$merged = Join-Path $Out "manifest.jsonl"
if (Test-Path $merged) { Remove-Item $merged -Force }
$count = 0
$missing = 0
$dupes = 0
$seen = @{}
for ($i = 0; $i -lt $Shards; $i++) {
  $dir = Join-Path $Out "shard$i"
  $mf = Join-Path $dir "manifest.jsonl"
  if (-not (Test-Path $mf)) { Write-Host "[shard] shard$i wrote no manifest"; continue }
  foreach ($line in Get-Content $mf) {
    if (-not $line.Trim()) { continue }
    $rec = $line | ConvertFrom-Json
    # EVERY SHARD NUMBERS ITS UTTERANCES FROM ZERO, so the ids collide across shards. The first version of this script
    # rewrote the paths and not the ids, and the project's own validator refused the merge with "4 duplicate utt_ids" -
    # which is why the merge is validated by that script rather than by this one's own checks.
    $rec.utt_id = "s$i`_" + $rec.utt_id
    if ($seen.ContainsKey($rec.utt_id)) { $dupes++ }
    $seen[$rec.utt_id] = $true
    $rec.wav_path = "shard$i/" + ($rec.wav_path -replace '\\', '/')
    $resolved = Join-Path $Out $rec.wav_path
    if (-not (Test-Path $resolved)) { $missing++ }
    ($rec | ConvertTo-Json -Compress -Depth 8) | Add-Content -Path $merged -Encoding utf8
    $count++
  }
}
Write-Host "[shard] merged $count records; unresolvable wav paths: $missing; duplicate ids: $dupes"
if ($missing -gt 0 -or $dupes -gt 0) { Write-Host "[shard] FAILING: the merge is not clean"; exit 1 }

# --- 3. verify with the project's own validator rather than trusting this script's merge
& $py scripts\validate_corpus.py --corpus $Out 2>&1 | Select-Object -Last 8
if ($LASTEXITCODE -ne 0) { Write-Host "[shard] the validator rejected the merged corpus"; exit 1 }

if (-not $KeepShards) { Write-Host "[shard] shards kept: the validator and the cache builder read the merged manifest, and deleting them would break the relative paths" }
Write-Host "[shard] done: $Out"
