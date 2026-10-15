"""Smoke test for a Kaggle accelerator session, GPU first.

The TPU smoke kernel came back with `torch 2.11.0+cpu`, no torch_xla and no TPU environment variables, so `enableTpu` on
an API push did not attach a TPU. This asks the neighbouring question, which may matter more: Kaggle's newer kernel
metadata carries an `accelerator` field, and a free GPU needs no porting at all - Parakeet already resolves `cuda`.
"""

from __future__ import annotations

import os
import platform
import sys


def main() -> int:
    print("python", sys.version.split()[0], "|", platform.platform())
    for key, value in sorted(os.environ.items()):
        if any(word in key.upper() for word in ("TPU", "GPU", "CUDA", "ACCELERATOR", "NVIDIA")):
            print(f"  env {key}={value[:60]}")
    try:
        import torch

        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device:", torch.cuda.get_device_name(0))
            a = torch.ones(2048, 2048, device="cuda")
            print(f"matmul on the GPU: {(a @ a).sum().item():.0f}")
            print("RESULT: a GPU is available to this kernel")
            return 0
        print("RESULT: no GPU attached (torch is", torch.__version__, ")")
    except Exception as error:
        print("torch import failed:", type(error).__name__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
