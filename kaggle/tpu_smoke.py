"""Smoke test for a Kaggle TPU session.

The cheapest question that can be asked before uploading 10.7 GB of cached latents: does Kaggle still grant a TPU, and
does its PyTorch image carry torch_xla? The pasted help page describes TensorFlow 2.1 and nightly torch-xla builds, so
the current state is worth measuring rather than assuming.

It also prints the facts the port depends on: the XLA core count, the torch and torch_xla versions, and whether a tensor
can be created and multiplied on the XLA device.
"""

from __future__ import annotations

import os
import platform
import sys


def main() -> int:
    print("python", sys.version.split()[0], "|", platform.platform())
    print("torch: ", end="")
    try:
        import torch

        print(torch.__version__)
    except Exception as error:  # pragma: no cover
        print(f"ABSENT ({error})")
        return 1

    for key in ("TPU_NAME", "COLAB_TPU_ADDR", "KAGGLE_TPU", "XRT_TPU_CONFIG"):
        if os.environ.get(key):
            print(f"env {key} is set")

    print("torch_xla import: ", end="")
    try:
        import torch_xla

        print("OK", getattr(torch_xla, "__version__", "unknown"))
    except Exception as error:
        print(f"ABSENT ({error})")
        print("RESULT: no torch_xla in this image - the port cannot run here")
        return 0

    import torch_xla.core.xla_model as xm

    devices = xm.get_xla_supported_devices()
    print(f"XLA devices: {len(devices)} -> {devices[:8]}")
    device = xm.xla_device()
    print("xla_device():", device)
    a = torch.ones(1024, 1024, device=device)
    b = (a @ a).sum().item()
    print(f"matmul on the XLA device: {b:.0f}")
    print("RESULT: a TPU is available to this notebook")


if __name__ == "__main__":
    sys.exit(main())
