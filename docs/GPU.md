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

## Option B — WSL2 with ROCm: CHECKED, and it cannot work for this card

**Verified against AMD's own support matrices (ROCm 7.2).** Both the
[Linux matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)
and the
[WSL matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)
list the same supported Radeon hardware, and it is **RDNA3/RDNA4 only**: RX 9070/9060 series, RX
7900/7800/7700 series, PRO W7900/W7800/W7700. **There is no RX 6000-series entry on either list.**

The card in this machine is a **Radeon RX 6950 XT = gfx1030 = RDNA2**, so it is outside ROCm 7.2's support
by matrix — on native Linux as well as WSL. This was checked on the machine, not assumed:

```
$ wsl -d Ubuntu -- bash scripts/wsl_gpu_check.sh
=== distro ===   Ubuntu 26.04 LTS, kernel 6.18.33.1-microsoft-standard-WSL2
=== python ===   Python 3.14.4; torch MISSING; numpy MISSING; soundfile MISSING
=== rocm ===     no /opt/rocm, no rocminfo, no rocm-smi
=== devices ===  /dev/dxg present (WSL GPU passthrough works), /dev/kfd absent (no ROCm driver)
=== torch ===    No module named 'torch'
```

Two things this does and does not say. **`/dev/dxg` exists**, so WSL *can* see the GPU — the platform is
capable. But ROCm is not installed, `/dev/kfd` is absent, and even with a perfect install the matrix above
excludes gfx1030. `scripts/wsl_gpu_check.sh` re-runs that inspection in one command.

### Routes that can actually reach gfx1030

| route | status |
|---|---|
| ROCm 6.x on **native Linux** | gfx1030 was supported in the ROCm 5.x/6.x era; the 7.x Radeon matrices are RDNA3+. Needs a Linux install, not WSL. |
| **Windows + HIP SDK**, community ROCm PyTorch wheels | the promising one: no OS change. See the [RDNA2 (gfx1030–1036) Windows ROCm torch installer work](https://github.com/unslothai/unsloth/pull/7277) and [the request to ship wheels for every supported card](https://github.com/unslothai/unsloth/issues/11815). |
| DirectML | already measured here: text step 1.0–1.5×, conv stack 6.1×, **the audio step cannot complete** — see the "measurement, in short" section. |
| CPU | works today, at the rates in the table below. |

Whatever the route, `scripts/check_device.py` is the acceptance test: run it and believe it rather than the
matrix.

```bash
# inside Ubuntu, if ROCm is ever installed there
sudo apt update && sudo apt install -y python3-pip python3-venv
sudo amdgpu-install --usecase=wsl,rocm --no-opengl
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
rocminfo | head -20                          # does it see gfx1030?
python scripts/check_device.py
```

If `check_device.py` reports the required probes `[ok]` and the complex ones too, training can move there
unchanged: the pipeline is CPU-only code plus torch, and `text_audio_step` takes a `loss_device`, so a
backend that cannot hold complex tensors can still place the STFT losses itself.

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

## What to run once it works, and what to judge it on

The CPU measurements bound the problem. Per step, measured on this machine:

| stage | batch | seconds/step (CPU) | audio per step | notes |
|---|---|---|---|---|
| `distill-text` (text → cached latents) | 16 | ~0.3 (small), ~0.7 (dim 512) | — | the cheapest signal |
| `distill-audio` (text → decoder → waveform) | 4 | ~0.5 | 9.1 s | decodes audio |
| `flow` (velocity field) | 8 | ~2.0 | 73 s | never decodes audio |

A few thousand steps is what fits in an evening on the CPU. The papers train for **orders of magnitude
more**, and round 40 measured what that costs: the flow's loss falls normally while the correlation
between its samples and the teacher's latents stays at zero — the *marginal* solution, learned first,
with the conditioned solution not yet reached. So the GPU run should be sized in the tens of thousands of
steps, with checkpoints every few thousand, and judged on the metrics this project has calibrated:

1. **sampled-latent ρ** (`scripts/flow_trajectory.py --latent-cache runs/mixed_v2/latent_cache`) — the
   recogniser reads a decoded latent perfectly at **ρ ≥ 0.75** and fails at 0.60 (round 37). This is the
   first number to watch, and unlike the mel proxy it cannot be satisfied by noise (round 39: ρ ≈ 0 with a
   mel cosine of 0.90).
2. **speech-likeness** (`real_eval.py` prints it for student and teacher) — voiced fraction 0.25–0.9, pitch
   70–350 Hz, spectral flatness < 0.65. Both trained paths currently fail it: the Tiny path with a
   0.96-voiced buzz, the flow with 0.07-voiced noise.
3. **WER with its control** — withheld automatically if the recogniser cannot read real speech.
4. **fit before generalisation** — `real_eval.py --corpus data/mixed_corpus/train_v2` on the *training*
   prompts. Every model so far is at WER 1.000 on its own training text, so that is the gate that matters
   first.

Suggested first GPU runs, in order:

```bash
# 1. does the flow learn the mapping at all?  Two utterances, a few thousand steps -- cheap and decisive.
python scripts/overfit_flow.py --items 2 --steps 5000 --batch-size 2 --lr 1e-3 --out runs/gpu_overfit

# 2. the real run: flow on the mixture, checkpoints every 2000
python scripts/train.py --config configs/parakeet_flow.yaml --stage flow \
    --cache runs/mixed_v2/latent_cache \
    --warm-start runs/ae_scaled/adversarial/autoencoder_last.pt \
    --steps 40000 --batch-size 16 --set train.save_every=2000 --device cuda --out runs/flow_gpu

# 3. judge it as it goes
python scripts/flow_trajectory.py --run runs/flow_gpu --limit 8 --steps 4 \
    --latent-cache runs/mixed_v2/latent_cache
```

Everything the flow needs is in the cache and the checkpoint carries the fitted latent-normaliser
statistics path (`load_latent_norm_from_cache`), so a GPU run needs no data regeneration. If the flow
crosses ρ ≥ 0.75, the next steps are the `reflow` stage for few-step inference (NFE 2–4) and the int8
export path — both already implemented.


The CPU is not idle: a 6000-step run is ~4 hours at batch 4 (~2 s/step), which is enough for several
experiments per day, and the acceptance criterion is now the **fit** number (WER on the *training*
prompts), which needs no GPU to read. The demo pack (`scripts/demo_pack.py`) renders listenable audio
from any checkpoint so progress is judged by ear as well as by proxy.

## While the GPU question is open
