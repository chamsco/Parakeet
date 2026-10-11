# Launch the flow run on the full expanded corpus, with the in-loop conditioning gate.
#
# Round 66's guard refuses --items larger than the cache, and round 76's holdout crash does not reproduce on
# this cache (verified on 1932 items: "holdout: 32 utterances never trained on"), so the run carries a real
# validation split as well as the gate.
#
# The gate is the number this spend exists to move: the flow term's own text dependence, printed every 2000
# steps.  Round 65 measured it at 1.9% -> 7.4% across a 50x corpus range, so the question is whether ~11 h of
# audio rather than 1.9 h moves it further.
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path runs\expanded_v3\index.json)) {
    Write-Host "runs\expanded_v3 does not exist yet -- run the chained merge first"
    exit 1
}

$total = & .\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '.'); from parakeet.data.dataset import LatentShardDataset; print(len(LatentShardDataset('runs/expanded_v3')))"
$total = [int]$total
if ($total -le 100) { Write-Host "expanded_v3 has only $total items -- refusing to start"; exit 1 }

$holdout = 32
$train = $total - $holdout
Write-Host "corpus: $total items | training on $train, holding out $holdout"

& .\.venv\Scripts\python.exe -u scripts\overfit_flow.py `
    --config configs\parakeet_flow_plan.yaml `
    --cache runs\expanded_v3 --items $train --holdout $holdout `
    --steps 12000 --batch-size 16 --crop-frames 160 --lr 0.0005 `
    --set text.dim=512 --set text.n_layers=6 --set flow.text_dim=512 --set flow.cond_dim=512 `
    --gate-every 2000 --save-every 2000 --out runs\flow_v3
