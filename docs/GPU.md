# Running Parakeet on a GPU (and why Windows/DirectML is not the route)

This project trained on the CPU for its first thirty rounds. Round 28 measured that the model was
**undertrained** (WER 1.000 on its own training prompts, ~5 passes over the corpus in 1600 steps), so
compute is the binding constraint on every experiment. This machine has an **AMD Radeon RX 6950 XT**
(RDNA2, gfx1030, 16 GB), and round 31 measured what it can actually do.

## The measurement, in short

| workload | CPU | DirectML (`torch-directml`) | speedup |
|---|---|---|---|
| `distill-text` step, batch 8 | 91 ms | 91 ms | 1.0× |
| `distill-text` step, batch 16 | 162 ms | 107 ms | 1.5× |
| decoder conv stack, 4×900 frames | 67 ms | 11 ms | **6.1×** |
| full `distill-audio` step | 478 ms | **cannot complete** | — |

DirectML is fast where the work is big and regular and no faster where it is small ops and per-op
overhead, and it cannot run the audio step at all: no complex dtype (`torch.polar`, `torch.fft.irfft`,
`torch.stft`), `F.fold`'s backward (`aten::col2im`) unimplemented, **`torch.eye` returns an empty
tensor**, `repeat_interleave` and `index_add` unimplemented, and — worst — a plugin error path that
raises `UnicodeDecodeError` instead of reporting the real failure. Full detail:
`docs/evidence/gpu_directml_investigation.json`.

**Conclusion: DirectML is not the unlock.** For real GPU training of a gfx1030 card, use ROCm on Linux.

## Before anything: check the environment

```bash
python scripts/check_device.py
```

It probes each operation the training step needs in its own subprocess (a missing dtype *aborts the
process* on DirectML, so an in-process check takes the report down with it) and ends with a verdict.
Run it as the first thing in any new environment — WSL, native Linux, a cloud box.

## Option A — native Linux with ROCm (the supported route for this card)

ROCm officially supports gfx1030 (RX 6900/6950 XT) on native Linux. That gives complete op coverage,
including the complex dtypes the STFT losses use.

```bash
# Ubuntu 22.04/24.04, as root or with sudo
sudo apt update && sudo apt install -y python3-pip python3-venv
# AMD's installer: https://repo.radeon.com/amdgpu-install/latest/ubuntu/ (match your release)
sudo apt install -y ./amdgpu-install_*.deb
sudo amdgpu-install --usecase=rocm --no-opengl
sudo usermod -aG render,video "$USER"        # then log out and back in
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
python scripts/check_device.py                # expect every probe [ok] and "fully usable"
```

The repository's own dependencies are `torch`, `numpy`, `pyyaml` (plus `soundfile` for audio I/O), so
a ROCm venv is a small install. **Do not** install `torch-directml` there.

## Option B — WSL2 with ROCm (worth trying; check the support matrix first)

AMD's *official* ROCm-on-WSL support currently lists RDNA3 (RX 7000 series). The 6950 XT is RDNA2, so
this may or may not work — which is exactly why `scripts/check_device.py` exists: run it and believe it
rather than the matrix.

```powershell
# ADMIN PowerShell
wsl --install -d Ubuntu-24.04
wsl --update
```

```bash
# inside Ubuntu
sudo apt update && sudo apt install -y python3-pip python3-venv
# AMD's "ROCm on WSL" guide: install the WSL-specific amdgpu-install package, then
sudo amdgpu-install --usecase=wsl,rocm --no-opengl
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
rocminfo | head -20                          # does it see gfx1030?
python scripts/check_device.py
```

If `check_device.py` reports the required probes `[ok]` and the complex ones too, training can move
there unchanged: the pipeline is CPU-only code plus torch, and `text_audio_step` now takes a
`loss_device`, so a backend that cannot hold complex tensors can still place the STFT losses itself.

## Training commands (both environments)

```bash
# build a cache from a corpus (resamples, extracts teacher signals, writes per-token latents)
python scripts/real_train_demo.py --corpus data/mixed_corpus/train_v2 --manifest manifest.jsonl \
    --latent-rate 3 --reuse-ae --out runs/mixed_v2

# the stage that trains the text side through the frozen decoder
python scripts/train.py --stage distill-audio --cache runs/mixed_v2/latent_cache \
    --warm-start runs/mixed_v2/distill-text_last.pt --steps 6000 \
    --set autoencoder.latent_rate=3 \
    --set train.loss.audio_aux=1.0 --set train.loss.signal_latent_contrast=1.0 \
    --out runs/long_v2
```

## While the GPU question is open

The CPU is not idle: a 6000-step run is ~4 hours at batch 4 (~2 s/step), which is enough for several
experiments per day, and the acceptance criterion is now the **fit** number (WER on the *training*
prompts), which needs no GPU to read. The demo pack (`scripts/demo_pack.py`) renders listenable audio
from any checkpoint so progress is judged by ear as well as by proxy.
