# Milestone 3, automated: wait for the expansion, merge it, then train the same configuration on it.
#
# The chain is mechanical but it has a wall-clock gap between the synthesis finishing and a run starting, so it is
# scripted rather than waited on. It does not start until the streaming manifest stops growing, because a half-written
# corpus would be merged silently.
#
# BASELINE AND PREDICTION, registered before the result.
#
# The baseline is arm A, runs/text_long384, the same configuration on runs/v3_fixed (4533 records over 3744 distinct
# prompts). Its held-out rho, measured with the repeated readout, is a plateau of about 0.2167 with sd 0.0201 across
# steps 3000 to 7000 (section 137). The measured resolution of the instrument is about +-0.02, so this experiment can
# only detect a difference larger than that.
#
# The enlargement roughly doubles the distinct prompts to about 8100. Two earlier data points measured NO effect from
# text count at matched steps: 418 texts against 3744 texts agreed to within 0.005 at steps 1000, 2000 and 3000
# (section 133), and 304 against 1207 items were similarly flat. So the prediction is that this changes nothing
# either, and the informative outcome is the difference measured with repeats:
#   - a mean difference above +0.02 at matched steps of 3000 or beyond means text diversity does matter after all, and
#     the earlier flat results were too small a range;
#   - a difference inside the spread means the objective's remaining lever is not corpus size, which closes milestone 2
#     and 3 with a negative result rather than an unfinished one.
param(
  [string]$Corpus = "data\kokoro_v4",
  [string]$Merged = "runs\v4_fixed",
  [int]$Steps = 10000,
  [int]$StableForSeconds = 180
)
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"

Write-Host "[m3] waiting for $Corpus to stop growing"
$last = -1
$stable = 0
while ($true) {
  $count = (Get-ChildItem "$Corpus\wav" -Filter *.wav -EA SilentlyContinue | Measure-Object).Count
  if ($count -eq $last) { $stable += 30 } else { $stable = 0; $last = $count }
  Write-Host ("[m3] {0} wavs, stable for {1}s" -f $count, $stable)
  if ($stable -ge $StableForSeconds) { break }
  Start-Sleep -Seconds 30
}

Write-Host "[m3] merging $Corpus onto runs\v3_fixed -> $Merged"
& $py -u scripts\expand_corpus.py --corpus $Corpus --cache "runs\kokoro_v4_cache" --base "runs\v3_fixed" --out $Merged
Write-Host "[m3] merge exit $LASTEXITCODE"
if (-not (Test-Path "$Merged\cache_meta.json")) { Write-Host "[m3] no merged cache, stopping"; exit 1 }

$meta = Get-Content "$Merged\cache_meta.json" -Raw | ConvertFrom-Json
$voiceNorm = if ($meta.voice_norm) { ($meta.voice_norm.PSObject.Properties | Measure-Object).Count } else { 0 }
Write-Host "[m3] merged cache: $(($meta.shards | Measure-Object).Count) shards, $voiceNorm voices normalised"

Write-Host "[m3] training the same configuration on the merged cache"
New-Item -ItemType Directory -Force runs\text_v4 | Out-Null
& $py -u scripts\train.py --config configs\parakeet_tiny.yaml --stage distill-text `
    --cache $Merged --steps $Steps --batch-size 16 --device cpu `
    --set text.dim=384 --set text.n_layers=6 `
    --set flow.text_dim=384 --set flow.cond_dim=384 `
    --set train.save_every=1000 --out runs\text_v4 > runs\text_v4\train.log 2>&1
Write-Host "[m3] training exit $LASTEXITCODE"
Write-Host "[m3] compare with: pwsh scripts\compare_runs.ps1 -Runs 'runs/text_long384,runs/text_v4' -Steps 1000,2000,3000,4000,5000 -Repeats 3"
