#!/usr/bin/env bash
# What is actually installed in this WSL distro? Run from Windows with:
#   wsl -d Ubuntu -- bash /mnt/c/Users/chamy/Documents/code/Parakeet/scripts/wsl_gpu_check.sh
# Read-only: it inspects, it does not install anything.
echo "=== distro ==="
. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"
uname -r

echo "=== python ==="
command -v python3 && python3 -V || echo "no python3"
for mod in torch numpy soundfile; do
    python3 -c "import $mod; print('$mod', getattr($mod, '__version__', 'ok'))" 2>/dev/null \
        || echo "$mod MISSING"
done

echo "=== rocm ==="
ls -d /opt/rocm* 2>/dev/null || echo "no /opt/rocm"
command -v rocminfo >/dev/null && rocminfo 2>/dev/null | grep -m2 -E "Name|gfx" || echo "no rocminfo"
command -v rocm-smi >/dev/null && rocm-smi --showproductname 2>/dev/null | head -6 || echo "no rocm-smi"

echo "=== devices ==="
ls -l /dev/dxg /dev/kfd 2>/dev/null || echo "no /dev/dxg or /dev/kfd"

echo "=== torch sees the gpu? ==="
python3 - <<'PY' 2>&1 | tail -5
try:
    import torch
    print("torch", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("arch:", torch.cuda.get_device_capability(0))
except Exception as exc:  # noqa: BLE001
    print("torch import/query failed:", exc)
PY

echo "=== raw glibc gpu visibility (if rocm is installed) ==="
ls /opt/rocm*/bin/rocminfo >/dev/null 2>&1 && /opt/rocm*/bin/rocminfo 2>/dev/null | grep -c "Agent" \
    || echo "n/a"
