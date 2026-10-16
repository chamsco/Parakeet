"""Sweep batch size on the T4 to find out whether the GPU can beat the CPU at all.

The three-way benchmark showed the step is not I/O-bound (a local cache copy made it slower) and that batch 512 is over
the 14.56 GiB card. That leaves batch size as the one untested variable, and it matters because the step cost is
dominated by fixed per-step work - the frozen autoencoder decode that the f0 and energy losses require - which a larger
batch amortises.

100 steps per batch size, local disk, same configuration as every other run in this project, so the only number that
changes is s/step. If the step time is flat in batch size, the GPU gives nothing here and the honest conclusion is to
use Kaggle for uptime rather than throughput.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/chamsco/Parakeet.git"
STEPS = int(os.environ.get("PARAKEET_STEPS", "100"))
BATCHES = [int(b) for b in os.environ.get("PARAKEET_BATCHES", "16,32,64,128").split(",")]


def sh(command: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    print(f"$ {command}", flush=True)
    result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
    tail = (result.stdout or "")[-600:]
    if tail:
        print(tail, flush=True)
    if result.returncode != 0:
        print("--- stderr tail ---", flush=True)
        print((result.stderr or "")[-600:], flush=True)
    return result


def train(cache: str, batch: int, cwd: Path) -> float | None:
    command = (
        f"PYTHONPATH={cwd} {sys.executable} -u scripts/train.py --config configs/parakeet_tiny.yaml "
        f"--stage distill-text --cache {cache} --steps {STEPS} --batch-size {batch} --device cuda "
        f"--set text.dim=384 --set text.n_layers=6 --set flow.text_dim=384 --set flow.cond_dim=384 "
        f"--out /kaggle/working/sweep_{batch}"
    )
    began = time.time()
    result = sh(command, cwd=cwd)
    if result.returncode != 0:
        return None
    per_step = (time.time() - began) / STEPS
    items_per_second = batch / per_step
    print(f"RESULT batch={batch}: {per_step:.3f} s/step  = {items_per_second:.1f} items/s", flush=True)
    return per_step


def main() -> int:
    sh(f"git clone --depth 1 {REPO} /tmp/parakeet")
    repo = Path("/tmp/parakeet")
    sh(f"{sys.executable} -m pip install --quiet soundfile")

    caches = sorted(glob.glob("/kaggle/input/**/cache_meta.json", recursive=True))
    if not caches:
        print("no mounted cache", flush=True)
        return 1
    remote = str(Path(caches[0]).parent)
    local = Path("/tmp/cache_local")
    if not local.exists():
        print("[sweep] copying the cache to local disk ...", flush=True)
        shutil.copytree(remote, local)
    print(f"[sweep] cache at {local}", flush=True)

    import torch

    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
    print(f"VRAM {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB", flush=True)

    results = {}
    for batch in BATCHES:
        try:
            results[batch] = train(str(local), batch, repo)
        except Exception as error:  # noqa: BLE001
            print(f"  batch {batch} raised {type(error).__name__}", flush=True)
            results[batch] = None

    print("\n[sweep] summary, 100 steps each", flush=True)
    print(f"  {'batch':>6} {'s/step':>8} {'items/s':>9}", flush=True)
    for batch, value in sorted(results.items()):
        if value is None:
            print(f"  {batch:>6} {'FAILED':>8}", flush=True)
        else:
            print(f"  {batch:>6} {value:>8.3f} {batch / value:>9.1f}", flush=True)
    print("  CPU reference: 0.52-0.93 s/step at batch 16 = 17-31 items/s", flush=True)
    return 0


def _exit(code: int) -> None:
    try:
        get_ipython  # noqa: B018
    except NameError:
        sys.exit(code)


if __name__ == "__main__":
    _exit(main())
