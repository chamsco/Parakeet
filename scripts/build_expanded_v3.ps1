# Finish the corpus: rebuild the second half's manifest, cache it, merge, then train the flow.
#
# The second half was generated before the round-82 durability fix, so its manifest was never written -- the
# same state the first half was in.  The rebuild is exact because the generator is deterministic: voice =
# voices[i % len(voices)] and files are named {teacher}_{index:07d}.wav.  Verified before writing this:
# kokoro_v3b's indices are contiguous from 0, and prompts_v3.txt[725] == prompts_v3_rest.txt[0], so index i in
# that directory is prompts_v3_rest.txt[i].
#
# Every step here has failed at least once in this project, which is why they are in this order:
#   manifest -> cache (--latent-rate 3, or the merge refuses) -> merge onto the EXISTING v3a (whose voices are
#   already remapped and whose normaliser covers the base corpus) -> train with the gate on.
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"

function Step($name, $block) {
    Write-Host "`n=== $name ==="
    & $block
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $name (exit $LASTEXITCODE)"; exit 1 }
}

if (-not (Test-Path data\kokoro_v3b\manifest.jsonl)) {
    Step "rebuild the second half's manifest" {
        & $py scripts\rebuild_manifest.py --corpus data\kokoro_v3b `
            --texts data\prompts\prompts_v3_rest.txt --teacher kokoro
    }
} else {
    Write-Host "manifest already present -- skipping the rebuild"
}

if (-not (Test-Path runs\kokoro_v3b_cache\index.json)) {
    Step "cache the second half" {
        & $py -u scripts\make_teacher_corpus.py --corpus data\kokoro_v3b --cache-only `
            --cache-out runs\kokoro_v3b_cache --latent-rate 3
    }
} else {
    Write-Host "cache already present -- skipping"
}

if (-not (Test-Path runs\expanded_v3\index.json)) {
    Step "chained merge onto the existing expanded corpus" {
        & $py -u scripts\expand_corpus.py --corpus data\kokoro_v3b --cache runs\kokoro_v3b_cache `
            --base runs\expanded_v3a --out runs\expanded_v3 --skip-cache
    }
} else {
    Write-Host "expanded_v3 already present -- skipping"
}

Step "train the flow with the conditioning gate" {
    & pwsh -NoProfile -File scripts\run_flow_v3.ps1
}
