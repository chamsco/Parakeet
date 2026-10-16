"""Measure what actually limits the T4, instead of assuming.

The first GPU run gave 0.404-0.448 s/step at batch 16 reading the cache from /kaggle/input, a read-only network mount.
That is 1.3-2x the CPU, not the order of magnitude I predicted, so the interesting question is which of the two suspects
is responsible. Three short runs answer it, 100 steps each:

  1. network mount, batch 16   - reproduces the original number
  2. local disk,    batch 16   - isolates I/O
  3. local disk,    batch 512  - adds batching

100 steps is enough to time and cheap enough to run three times. AMP is deliberately NOT tested: the trainer has no
autocast path, adding one would change numerics and break comparability with every fp32 result this project has, and that
is a decision to make on evidence rather than in passing.
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


def sh(command: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    print(f"$ {command}", flush=True)
    result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
    tail = (result.stdout or "")[-1200:]
    if tail:
        print(tail, flush=True)
    if result.returncode != 0:
        print("--- stderr ---", flush=True)
        print((result.stderr or "")[-1500:], flush=True)
    return result


def train(cache: str, batch: int, out: str, cwd: Path) -> float | None:
    command = (
        f"PYTHONPATH={cwd} {sys.executable} -u scripts/train.py --config configs/parakeet_tiny.yaml "
        f"--stage distill-text --cache {cache} --steps {STEPS} --batch-size {batch} --device cuda "
        f"--set text.dim=384 --set text.n_layers=6 --set flow.text_dim=384 --set flow.cond_dim=384 "
        f"--out {out}"
    )
    began = time.time()
    result = sh(command, cwd=cwd)
    elapsed = time.time() - began
    if result.returncode != 0:
        return None
    per_step = elapsed / STEPS
    print(f"RESULT batch={batch} cache={cache}: {elapsed:.1f}s / {STEPS} steps = {per_step:.3f} s/step", flush=True)
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

    print("[bench] copying the cache to local disk ...", flush=True)
    began = time.time()
    local = Path("/tmp/cache_local")
    if local.exists():
        shutil.rmtree(local)
    shutil.copytree(remote, local)
    print(f"[bench] copied in {time.time() - began:.1f}s", flush=True)

    results = {}
    results["network_batch16"] = train(remote, 16, "/kaggle/working/bench_net16", repo)
    results["local_batch16"] = train(str(local), 16, "/kaggle/working/bench_loc16", repo)
    results["local_batch512"] = train(str(local), 512, "/kaggle/working/bench_loc512", repo)

    print("\n[bench] summary, 100 steps each", flush=True)
    baseline = results.get("network_batch16")
    for name, value in results.items():
        if value is None:
            print(f"  {name:18s} FAILED", flush=True)
            continue
        speedup = f"{baseline / value:.2f}x vs the network run" if baseline else ""
        print(f"  {name:18s} {value:.3f} s/step   {speedup}", flush=True)
    print("  CPU reference: 0.52 to 0.93 s/step at the same configuration", flush=True)
    return 0


def _exit(code: int) -> None:
    try:
        get_ipython  # noqa: B018
    except NameError:
        sys.exit(code)


if __name__ == "__main__":
    _exit(main())
