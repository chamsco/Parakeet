# Parakeet

A tiny, lightning-fast, high-realism TTS model built by distilling a **mixture of teachers**
into one small student.

Teachers: **[Canopy Labs' Orpheus](https://github.com/canopylabs/orpheus-tts)** (Llama-3.2-3B + SNAC
24 kHz, expressive, tag-controllable) and **MiniMax speech-2.8-turbo** (high-fidelity, closed API).
A third, fully permissive teacher — **Kokoro-82M** (Apache-2.0) — is the safe default that keeps the
whole recipe reproducible without licence risk.

> **Status:** working research framework, CPU-verifiable end to end. No trained checkpoint yet —
> the numbers below are *architecture*, not quality claims. `scripts/smoke_test.py` trains every
> stage for a couple of steps and synthesises audio on CPU so the pipeline is provably functional.

---

## The idea in one table

You cannot merge an autoregressive discrete-codec LM (Orpheus) with a closed audio-only API
(MiniMax) at the token level — their representations are incompatible and MiniMax exposes no
tokens at all. So Parakeet unifies them **at the audio level**: a small continuous-latent
autoencoder re-encodes every teacher's waveform into one shared 24-dim latent grid, and the
student is trained against that. Teacher identity disappears; only acoustics remain.

| | What we take from it | Why |
|---|---|---|
| **[Paradee](https://www.alphaxiv.org/abs/2610.06817)** (Kokoro → 8M) | Two halves trained *separately* against a frozen cached teacher corpus; **low spectral weight** (3, not 45); adversarial decoder; weight-only int8; phase-lock filter | Its central empirical result: **loss balance dominates decoder size** (spectral-only 2.98 → weight 45 → 3.02 → weight 10 → 4.32 → weight 3 → 4.39 UTMOS). Direct feature supervision of the text half beats end-to-end distillation (4.39 vs 3.78). |
| **[SupertonicTTS](https://www.alphaxiv.org/abs/2503.23108)** (44M) | 24-dim continuous latent, ConvNeXt blocks, character-level text with implicit cross-attention alignment (no G2P), temporal compression `Kc=6`, causal streaming decoder, `Ke=4` context-sharing batch expansion | Proven 44M budget (0.5M duration + 18.5M text-to-latent + 25M latent-to-speech), RTF 0.02 on a 4090, trained on only 945 h. Its NFE-4 WER blow-up (11.43 vs 2.64 at NFE 32) is exactly why we distil the *sampler*. |
| **[PilotTTS](https://www.alphaxiv.org/abs/2605.27258)** (200k h, Qwen3-0.6B) | Factorised conditioning: frozen **CAM++ identity embedding + Q-Former style tokens**, cross-sample paired training; disciplined data pipeline; teacher-generated data for scarce capabilities | Lets one student hold both teachers' strengths — MiniMax-grade prosody *and* Orpheus-grade expressivity — without entangling identity with style. |
| **Orpheus** | Its inline paralinguistic tags (`<laugh>`, `<sigh>`, …) become a shared control vocabulary; its audio is a fidelity/expressivity target | Expressive control is Orpheus's differentiator, and tags are cheap supervision. |
| **MiniMax speech-2.8-turbo** | 19 interjection tags as a **labelling taxonomy**; pronunciation/word timestamps as alignment supervision | See [docs/LEGAL.md](docs/LEGAL.md): **its terms bar using the Voice API to develop foundation models**, so its audio is refused by default. |

## Architecture at a glance

```
text ──► TextEncoder (char-level, no G2P) ──┬──► DurationPredictor ──► frames
                                            │
reference mel ──► CAM++ identity  ──┐       └──► Flow-matching VF Estimator (ConvNeXt + cross-attn)
                ──► Q-Former style ─┴──► conditioning memory          │  NFE 2-4 after Reflow
                                                                     ▼
                                        latent (24-dim, 93.75 Hz; Kc=6 folded for the sampler)
                                                                     ▼
                              Causal ConvNeXt decoder ──► iSTFT head ──► waveform
                                                                     ▼
                                            phase-lock filter (2–8 kHz) ──► int8 export
```

Two variants share every block:

| | Parakeet-Tiny | Parakeet-Small |
|---|---|---|
| params (measured) | **9.62M** | **45.04M** |
| voices | 1 (learned constant style; Paradee replaces the style input) | zero-shot cloning (CAM++ + Q-Former) |
| inference | no sampler — the text side predicts duration/F0/energy/latent features directly | flow matching, NFE 2 with Reflow distillation |
| int8 (weight-only + fp16 scales) | 12.0 MB | 56.3 MB |
| measured RTF, 1 CPU thread, fp32 | **0.044 → 22.6× real time** | 1.92 on a *randomly initialised* 0.04 s output (overhead-dominated; not a valid number until trained) |
| intended use | laptop / on-device | GPU server or a fast CPU with a few steps |

## Quickstart (CPU, no data, no GPU)

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"     # Windows
python scripts/smoke_test.py --steps 2          # trains every stage, synthesises, quantises
python -m pytest -q                            # 79 tests
python scripts/bench_rtf.py --config configs/parakeet_tiny.yaml --steps 2
```

`scripts/smoke_test.py` is the proof-of-life: it exercises all five training stages, the
streaming decoder (which it checks is *numerically identical* to offline decoding), the
phase-lock filter, int8 quantisation, and RTF. See [docs/05-VERIFICATION.md](docs/05-VERIFICATION.md).

## Documentation

| | |
|---|---|
| [docs/00-DESIGN.md](docs/00-DESIGN.md) | The full reasoning: teacher analysis, how the mixture actually works, the five recipe stages, budgets |
| [docs/01-ARCHITECTURE.md](docs/01-ARCHITECTURE.md) | Module-by-module spec with dimensions and measured parameter counts |
| [docs/02-TRAINING.md](docs/02-TRAINING.md) | Stage-by-stage hyperparameters, losses, the spectral-weight anneal, compute estimate |
| [docs/03-DATA.md](docs/03-DATA.md) | Teacher corpus + latent cache + the PilotTTS-style filtering pipeline (with thresholds) |
| [docs/04-EVALUATION.md](docs/04-EVALUATION.md) | Metrics, targets to beat, comparison table |
| [docs/LEGAL.md](docs/LEGAL.md) | **Read this before using teacher audio.** Licence/ToS analysis and the safe paths |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Phases, milestones, compute budget, what "done" means |
| [docs/research/](docs/research/) | Source-grounded research notes on all three papers and both teachers |

## Repository map

```
parakeet/
  audio/      mel filterbank, autocorrelation F0, streaming/offline OLA iSTFT (pure torch)
  models/     blocks, speech autoencoder, text/duration/speaker encoders, flow-matching VF, assemblies
  train/      losses (MRSTFT, MPD/MSD, phase, distillation) + the five staged training loops
  data/       text normalisation/tags, teacher backends + licence gate, feature cache, datasets
  inference/  streaming synthesizer, phase-lock filter, int8 quantisation
  eval/       RTF, MCD, spectral convergence, phase coherence, optional UTMOS/WER/SECS
configs/      parakeet_tiny.yaml (9.6M) | parakeet_small.yaml (45M) | parakeet_small_44k.yaml
scripts/      smoke_test.py | train.py | make_teacher_corpus.py | bench_rtf.py
tests/        79 tests: config, audio DSP, models, losses, all five stages, inference, data
```

## Honest limitations

* **No trained weights yet.** Every quality number in the papers (UTMOS 4.41, WER 5.7 %, RTF 0.02
  on a 4090) is a target, not a Parakeet result.
* **MiniMax distillation is legally blocked by default.** The framework supports it; the licence
  gate refuses it. Default mixture is Orpheus 60 / Kokoro 40.
* **Synthetic-only corpora are a known trap** — Canopy itself warns that training on synthetic
  audio degrades codec utilisation and maps everything onto the same tokens. Teacher data is for
  capability/render distillation, mixed with real speech for the autoencoder and speaker encoder.
* The `SpeakerConditioner` ships with a small randomly-initialised ECAPA stand-in; real runs must
  load frozen CAM++ (see `SpeakerConfig.checkpoint`).
* The flow sampler is not yet streaming (chunked *decoding* is); TTFA for Small is currently
  dominated by the whole latent being produced first.

Licence: Apache-2.0 (this repository). Teacher weights and audio carry their own terms.
