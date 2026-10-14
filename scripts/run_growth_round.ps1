# One growth round: synthesise the prepared prompts, cache them, merge onto the current largest cache, and train.
#
# WHETHER to run this is decided by the objective's stop rule, not by convenience: a rising held-out rho means keep
# growing, a flat one means stop and report the ceiling. This script exists so the positive branch is one command rather
# than five remembered ones, and it is not launched until a read says to launch it.
#
# It validates the corpus BEFORE the merge, which is the check that would otherwise be discovered after twenty minutes
# of encoding, and it stops if the merged cache is not written. merge_caches now also refuses a merge whose halves were
# encoded by different autoencoders, so an inconsistent merge cannot proceed silently.
#
# The configuration matches runs/text_v4 exactly, so the two are comparable at matched steps and the same
# cumulative-exposure arithmetic applies: 10000 steps with warmup clamped to 500.
param(
  [string]$Texts = "data\prompts\prompts_v5.txt",
  [string]$Corpus = "data\kokoro_v5",
  [string]$Cache = "runs\kokoro_v5_cache",
  [string]$Base = "runs\v4_fixed",
  [string]$Out = "runs\v5_fixed",
  [string]$RunDir = "runs\text_v5",
  [int]$Steps = 10000,
  [int]$Shards = 6
)
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"

# SHARDED SYNTHESIS. The serial teacher runs at about 20 prompts per minute on one of this machine's 16 cores, which
# made a round cost three hours of synthesis; sharding it six ways measured 58.3 utterances per minute, a 2.9x. The
# sharded corpus is verified to cache correctly - the rewritten wav_paths resolve against the merged directory and
# make_teacher_corpus reads it with exit 0 - because a merge that only looks right is the defect class this project
# keeps finding.
Write-Host "[grow] synthesising $Texts -> $Corpus in $Shards shards"
.\scripts\shard_synthesis.ps1 -Texts $Texts -Out $Corpus -Shards $Shards -Teachers kokoro
Write-Host "[grow] synthesis exit $LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { Write-Host "[grow] synthesis or its merge failed, stopping"; exit 1 }

Write-Host "[grow] validating before the merge"
& $py scripts\validate_corpus.py --corpus $Corpus
if ($LASTEXITCODE -ne 0) { Write-Host "[grow] validation failed, stopping before the merge"; exit 1 }

Write-Host "[grow] caching and merging onto $Base -> $Out"
& $py -u scripts\expand_corpus.py --corpus $Corpus --cache $Cache --base $Base --out $Out
if (-not (Test-Path "$Out\cache_meta.json")) { Write-Host "[grow] no merged cache, stopping"; exit 1 }

$meta = Get-Content "$Out\cache_meta.json" -Raw | ConvertFrom-Json
Write-Host ("[grow] merged cache: n_shards {0}  voices {1}  voice_norm {2}  fingerprint {3}" -f `
  $meta.n_shards, ($meta.voice_names | Measure-Object).Count, ($meta.voice_norm | Measure-Object).Count, `
  $meta.autoencoder_fingerprint)

Write-Host "[grow] training $Steps steps on $Out"
New-Item -ItemType Directory -Force $RunDir | Out-Null
& $py -u scripts\train.py --config configs\parakeet_tiny.yaml --stage distill-text `
    --cache $Out --steps $Steps --batch-size 16 --device cpu `
    --set text.dim=384 --set text.n_layers=6 `
    --set flow.text_dim=384 --set flow.cond_dim=384 `
    --set train.save_every=1000 --out $RunDir > "$RunDir\train.log" 2>&1
Write-Host "[grow] training exit $LASTEXITCODE"
Write-Host "[grow] to watch it: pwsh scripts\monitor_text_run.ps1 -RunDir $RunDir -Cache runs\holdout_cache"
Write-Host "[grow] to score it: pwsh scripts\compare_runs.ps1 -Runs 'runs/text_v4,$RunDir' -Steps 2000,3000,4000 -Repeats 5"
