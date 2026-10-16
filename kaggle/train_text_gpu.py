"""Run the text stage on a Kaggle GPU and report the step time.

This is the measurement that decides whether the accelerator is worth the plumbing: the same stage, the same cache, the
same configuration as the CPU runs, so the only number that changes is seconds per step. It runs a short budget on
purpose - 300 steps is enough to time it and to see the loss move - because the question is speed, not quality.

The cache is mounted read-only from the dataset uploaded for this purpose; checkpoints go to /kaggle/working so they can
be pulled back. `--device cuda` needs no porting: resolve_device already returns it.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/chamsco/Parakeet.git"
STEPS = int(os.environ.get("PARAKEET_STEPS", "300"))


def sh(command: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {command}", flush=True)
    result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
    tail = (result.stdout or "")[-2000:]
    if tail:
        print(tail, flush=True)
    if result.returncode != 0 and check:
        print((result.stderr or "")[-2000:], flush=True)
    return result


def main() -> int:
    started = time.time()
    # Clone into /tmp, not /kaggle/working: whatever is in /kaggle/working is what `kernels output` downloads, and a
    # repository clone there turns a 20 KB log into thousands of files. That mistake also extracted a second copy of the
    # repo into this project's own package directory on the way back.
    sh(f"git clone --depth 1 {REPO} /tmp/parakeet")
    repo = Path("/tmp/parakeet")
    sh(f"{sys.executable} -m pip install --quiet soundfile", check=False)

    # Kaggle mounted this dataset at /kaggle/input/datasets/<owner>/<slug>/ - three levels down. Two hand-written globs
    # guessed one and two levels and both found nothing, and the log's own tree dump is what gave the path. So this is a
    # recursive glob now: it does not care how deep the mount is, which is the property the two guesses lacked.
    caches = sorted(glob.glob("/kaggle/input/**/cache_meta.json", recursive=True))
    print("caches found:", caches, flush=True)
    if not caches:
        print("nothing mounted at either depth; /kaggle/input tree:", flush=True)
        for path in sorted(glob.glob("/kaggle/input/**", recursive=True))[:25]:
            print("   ", path, flush=True)
        return 1
    cache = str(Path(caches[0]).parent)
    print("using cache:", cache, flush=True)
    print("cache contents:", sorted(p.name for p in Path(cache).glob("*"))[:12], flush=True)

    import torch

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0), flush=True)

    command = (
        f"PYTHONPATH={repo} {sys.executable} -u scripts/train.py --config configs/parakeet_tiny.yaml "
        f"--stage distill-text --cache {cache} --steps {STEPS} --batch-size 16 --device cuda "
        f"--set text.dim=384 --set text.n_layers=6 --set flow.text_dim=384 --set flow.cond_dim=384 "
        f"--out /kaggle/working/text_gpu"
    )
    began = time.time()
    result = sh(command, cwd=repo, check=False)
    elapsed = time.time() - began
    print(f"training exit {result.returncode} in {elapsed:.1f}s", flush=True)
    if result.returncode != 0:
        # a failed run is only useful if it says why, and the trainer's argparse errors go to stderr
        print("--- stderr tail ---", flush=True)
        print((result.stderr or "")[-3000:], flush=True)
        sh(f"{sys.executable} scripts/train.py --help", cwd=repo, check=False)
    else:
        print(f"RESULT: {STEPS} steps on GPU in {elapsed:.1f}s -> {elapsed / STEPS:.3f} s/step", flush=True)
        print("for comparison, the CPU measured 0.52 to 0.93 s/step at the same configuration", flush=True)
    print(f"total wall in the kernel: {time.time() - started:.1f}s", flush=True)
    return 0


def _exit(code: int) -> None:
    """Exit with a status in a script, but do nothing inside a notebook.

    Kaggle presents a script kernel as notebook cells. `sys.exit()` there raises SystemExit, which IPython reports as an
    error - "To exit: use 'exit', 'quit', or Ctrl-D" - even when the code finished successfully. That happened on the
    first operator run of this kernel: the training had completed and the exit call made it look like a crash, which is
    the worst kind of misleading because the work was already done.
    """
    try:
        get_ipython  # noqa: B018 - defined only inside IPython
    except NameError:
        sys.exit(code)


if __name__ == "__main__":
    _exit(main())
