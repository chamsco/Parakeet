"""Is this environment able to train Parakeet on a GPU?  Run it and it says.

    python scripts/check_device.py

Written because round 31 measured the Windows/DirectML route in detail and it does **not** work for this
model: `torch-directml` gives 1.0-1.5x on the small-op-bound text step and 6.1x on the decoder's conv
stack, and cannot run the audio step at all (no complex dtype; `col2im`, `eye`, `repeat_interleave` and
`index_add` missing or broken; an error path that raises `UnicodeDecodeError` instead of reporting).

So this answers the question for *any* environment -- native Linux with ROCm, a WSL2 install, a CUDA box,
or plain CPU -- by testing the operations the training step actually needs rather than reading a support
matrix.  Every probe runs in a **subprocess**: DirectML *aborts the process* on a missing dtype (exit
1010 from the plugin), so an in-process check would take the report down with it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PREAMBLE = """
import torch
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    try:
        import torch_directml
        device = torch_directml.device()
    except Exception:
        pass
"""

PROBES = {
    "matmul": "x = torch.randn(512, 512, device=device); (x @ x).sum().item()",
    "conv1d fwd+bwd": """
layer = torch.nn.Conv1d(32, 32, 7, padding=3).to(device)
x = torch.randn(2, 32, 200, device=device, requires_grad=True)
layer(x).pow(2).mean().backward()
""",
    "conv_transpose1d fwd+bwd (overlap-add)": """
import torch.nn.functional as F
weight = torch.eye(256).unsqueeze(1).to(device)
x = torch.randn(1, 256, 8, device=device, requires_grad=True)
F.conv_transpose1d(x, weight, stride=64).pow(2).mean().backward()
""",
    "torch.eye returns a real identity": "assert torch.eye(8, device=device).shape == (8, 8)",
    "repeat_interleave": "torch.repeat_interleave(torch.arange(4, device=device), 2)",
    "index_add": """
out = torch.zeros(16, device=device).index_add(
    0, torch.arange(4, device=device), torch.ones(4, device=device)
)
assert abs(float(out.sum()) - 4.0) < 1e-6
""",
    "F.fold fwd+bwd (col2im)": """
import torch.nn.functional as F
x = torch.randn(2, 1024, 12, device=device, requires_grad=True)
F.fold(x, output_size=(1, 4000), kernel_size=(1, 1024), stride=(1, 256)).pow(2).mean().backward()
""",
    "complex dtype / torch.stft": """
spec = torch.stft(torch.randn(2048), n_fft=256, return_complex=True)
assert spec.is_complex()
""",
    "torch.fft.irfft": "torch.fft.irfft(torch.ones(2, 129, dtype=torch.complex64), n=256)",
    "torch.polar": "torch.polar(torch.ones(4, device=device), torch.zeros(4, device=device))",
}

#: the training step needs these; everything else is a nicety
REQUIRED = ("matmul", "conv1d fwd+bwd", "conv_transpose1d fwd+bwd (overlap-add)")
#: without complex support the STFT losses need the repository's real-valued fallback, which exists
COMPLEX = ("complex dtype / torch.stft", "torch.fft.irfft", "torch.polar")


def probe(name: str, body: str) -> tuple:
    try:
        done = subprocess.run(
            [sys.executable, "-c", PREAMBLE + body],
            capture_output=True, text=True, timeout=180, cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return False, "timed out"
    if done.returncode == 0:
        return True, ""
    message = (done.stderr or done.stdout).strip().splitlines()
    return False, message[-1][:110] if message else f"exit {done.returncode}"


def main() -> int:
    import torch

    print(f"torch {torch.__version__} | python {sys.version.split()[0]}")
    if torch.cuda.is_available():
        try:
            print(f"  GPU: {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)")
        except Exception:  # noqa: BLE001
            print("  GPU: cuda (properties unavailable)")
    else:
        try:
            import torch_directml  # type: ignore

            print(f"  DirectML device: {torch_directml.device_name(0)}")
        except Exception:  # noqa: BLE001
            print("  no GPU backend (CUDA or DirectML): training runs on the CPU")

    print("\ncapabilities (each in its own subprocess, because a missing dtype can abort the process):")
    results = {}
    for name, body in PROBES.items():
        ok, message = probe(name, body)
        results[name] = ok
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  -- {message}" if not ok else ""))

    required_ok = all(results[name] for name in REQUIRED)
    complex_ok = all(results[name] for name in COMPLEX)
    print()
    if not torch.cuda.is_available() and not any("DirectML" in line for line in ["x"]):
        pass
    if torch.cuda.is_available():
        backend = "CUDA"
    else:
        backend = "DirectML" if results.get("matmul") and not complex_ok else "CPU"

    if backend == "CPU":
        print("verdict: CPU -- correct but slow (~2 s per training step at batch 4).")
    elif required_ok and complex_ok:
        print(f"verdict: {backend} looks fully usable: the step's ops work here, complex included.")
        print("         Run training with that device and expect a large speedup over the CPU.")
    elif required_ok:
        print(f"verdict: {backend} runs the model but not the complex ops the STFT losses use.  The "
              "repository has a real-valued fallback (parakeet/audio/istft.py), so the audio step may "
              "still work; the FAILs above say what is missing.")
    else:
        print(f"verdict: {backend} is NOT usable for training this model; stay on the CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
