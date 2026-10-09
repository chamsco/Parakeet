# Parakeet

A tiny, lightning-fast, high-realism TTS model built by distilling a **mixture of teachers**
into one small student.

Teachers: **[Canopy Labs' Orpheus](https://github.com/canopylabs/orpheus-tts)** (Llama-3.2-3B + SNAC
24 kHz, expressive, tag-controllable) and **MiniMax speech-2.8-turbo** (high-fidelity, closed API).
A third, fully permissive teacher — **Kokoro-82M** (Apache-2.0) — is the safe default that keeps the
whole recipe reproducible without licence risk.

> **Status:** working research framework, CPU-verifiable end to end. No trained checkpoint yet —
> the numbers below are *architecture*, not quality claims. Two artefacts carry the verification:
> `scripts/smoke_test.py` trains every stage and synthesises audio (the pipeline **runs**), and
> `scripts/learn_demo.py` shows the stages actually **learn and compose** on structured synthetic
> speech, with before/after metrics and pass/fail criteria.

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
| measured RTF, 1 CPU thread, fp32 | **0.044–0.052 → 19–23× real time** across runs | 1.92 on a *randomly initialised* 0.04 s output (overhead-dominated; not a valid number until trained) |
| intended use | laptop / on-device | GPU server or a fast CPU with a few steps |

## Measured so far (one CPU thread, no GPU)

| | Number | Where |
|---|---|---|
| Full Tiny synthesis, PyTorch fp32 | ~21 ms for 0.57 s audio → **19–27× real time** | `smoke_test.py`, `profile_pipeline.py` |
| **Full Tiny pipeline as int8 ONNX** | **5.2 ms → 103× real time** (4.0× faster than PyTorch); **7.3 ms → 74×** with the phase-lock filter shipped (was 71× before the per-call constants were cached) | `export_onnx.py --pipeline` |
| Model size, int8 ONNX | **36.7 → 9.9 MB** (text side 18.8 → 5.2, vocoder 18.0 → 4.7) | `export_onnx.py --pipeline` |
| int8 fidelity vs PyTorch | waveform cosine **0.992–0.9998**, mel L1 0.010–0.027 (untrained weights, seeded run, so reproducible) | `export_onnx.py --pipeline` |
| Where the time goes | decoder 35 %, text side 30 %, phase lock 10 %, python/dispatch 12 % — the vocoder is *not* the bottleneck; the removable part of the dispatch overhead was the phase-lock grids (0.386 ms → 0.4 µs cached) | `profile_pipeline.py` |
| Streaming decoder | **exactly equal** to offline decoding (5.6e-09) | `test_inference.py` |
| **Time to first audio** | 21.9 s utterance: one-shot **979 ms → streaming 125 ms (7.81×)**, and ~flat in length; blockwise output 50–100× closer to one-shot than an independent draw. Cost is stated: 2.81× total compute at block 16, 1.07× at block 128 | `streaming_demo.py` |
| Does it learn? | AE recon **36 %** better, text side **91 %**, text→audio **26 %** vs untrained, duration MAE **3 ms**, generated/target log-mel cosine **0.954** | `learn_demo.py` |
| **Does the teacher mixture work?** | predicted F0 moves **87 → 200 Hz** monotonically across a mixture sweep between two synthetic teachers (92/215 Hz, span 112.6 Hz), and the fit to the low teacher degrades in the same order | `mixture_demo.py` |
| Is NFE 2 viable? | reflowed 2-step agrees with the NFE-32 reference **better than the teacher's own NFE-16** (1.51 vs 1.85) and 29 % better than a naive 2-step cut; **9.7× wall-clock** at NFE 2 | `reflow_demo.py` |
| **Does multi-voice work?** | for three fixture voices the student predicts **81.7 / 88.8 / 148.6 Hz** (fixtures at 80.8 / 95 / 152) and fits the per-voice targets **7x better** than a voice-blind control | `voice_demo.py` |
| **Does speaker/style conditioning reach the model?** | with the pre-fix cached path the identity/style encoders get **exactly zero gradient**; wired, the reference changes the conditioning by 0.2254 (1.74 vs null) and a same-text/different-reference check shows conditioning is live | `recipe_dry_run.py --stage flow` |
| **Does the documented pipeline actually run?** | yes — prompts -> synthesis -> **P1 curation** -> cache from the curated manifest -> training -> synthesis, 10/10 checks on both the Tiny and Small paths, offline | `recipe_dry_run.py` |
| **Does the repository actually contain the project?** | verified by cloning the pushed repo and running its tests there — an unanchored .gitignore rule (data/) had kept the entire parakeet/data package out of GitHub for ten commits; a hygiene test now fails if any package file is untracked | 	est_provenance_and_hygiene.py |
| **Is the real teacher path correct?** | it had never been executed (weights/API needed) and **was wrong**: Orpheus codebooks were grouped contiguously instead of the published `{0}`/`{1,4}`/`{2,3,5,6}`, which decodes to noise. Now verified element-wise against a reimplementation of the published decoder, with a positive control | `test_teacher_backends.py` |
| **Does the phase-lock filter lock?** | yes, and its delay grid was under-configured: 64 → 256 (now the default) triples the coherence added to glottal-locked speech while *reducing* what it adds to white noise. **But the speech-vs-noise gap narrows, so the statistic cannot show speech-specific locking** — and the perceptual claim stays a citation, since UTMOS is unavailable | `phase_lock_ab.py` |
| **Does resume actually resume?** | yes — a crashed-then-resumed run is **bit-identical** to an uninterrupted one (parameters, losses and LR all differ by 0.000e+00). Checkpoints carry optimizer, EMA, discriminator, RNG and batch order, and the LR schedule continues instead of restarting | `resume_demo.py` |
| **Is every advertised number backed by evidence?** | yes, mechanically: the [model card](docs/MODEL_CARD.md) is *generated* from the demos' report files (15/15 claims verified), a claim with no report renders as unmeasured, and CI regenerates the whole evidence bundle and fails if the card is stale | `model_card.py` |
| **How light can it be?** | measured frontier: a **dim128-L4** text side (1.008 M vs 3.851 M) gives up **2.3 %** of held-out fixture fit for **35 % fewer total parameters** (6.23 M vs 9.62 M) and 2.6× the text-side speed. Offered as `configs/parakeet_tiny_lite.yaml`, not as the new default — the fixtures are repetitive | `ablate.py` |
| **Real teacher speech?** | **yes** — Kokoro-82M (Apache-2.0) via sherpa-onnx runs at **0.53× real time on this CPU**; 16 real 24 kHz utterances went through corpus → curation → cache, and real speech exposed three fixture-hidden defects (pitch voicing threshold, 0 Hz targets from the unaligned split, and the published curation gates rejecting 7/16) | `real_corpus_demo.py` |
| **Trained on real speech?** | **yes** — 21 curated Kokoro utterances (74.7 s): autoencoder reconstruction **−31.1 %**, text side **−79.8 %**, text→audio log-mel cosine **0.940** vs the real reference. Baseline metrics with controls: **DNSMOS 1.77 vs the teacher's 2.61**, **WER 1.00 vs the teacher's 0.00** (recogniser ase.en) — the student is **not yet intelligible**, and now we can say so with numbers | `real_train_demo.py`, `real_eval.py` |
| **Why is it unintelligible?** | localised: the **autoencoder** is the bottleneck, not the text side. Its round-trip on real audio is **uncorrelated with its input** (waveform cosine +0.000, SNR −0.10 dB) while the mel proxy looked merely poor — and the per-token→frame seam costs only 0.005 mel cosine. One real defect found en route: durations were the last prosody target still in raw log space, collapsing to **0.29×**; normalised, they predict **0.94×** | `real_diagnose.py` |
| **Is the autoencoder fixed?** | **yes** — the adversarial step costs **24× more** than a reconstruction step (5.25 s vs 0.22 s), so the budget went reconstruction-first (2000 + 200 steps): round-trip WER **1.000 → 0.153** against a 0.000 teacher, mel L1 1.386 → **0.506**. That exposed the *next* bottleneck: the per-token→frame seam costs 0.167 → **0.870** WER for 0.016 of mel cosine | `ae_ablation.py`, `ae_train.py` |
| **Is the seam fixed?** | **partly, and the A/B says so.** `distill-decoder` could not even run on a real cache (no waveform in the shards), so it now trains on the token-expanded distribution its docstring always promised: seam WER **0.889 → 0.722** with the working paths undamaged — but the gap to the frame path (0.167) stays open, because the per-token latent is an *average* over ~6 frames. Also puts `prosody_proj` in the graph, after which removing it *hurts* (0.907 vs 0.722) | `seam_ab.py` |
| **Was the seam information?** | **yes, 77 % of it.** An oracle sweep (no training) taking the teacher's own frame latent at 1/2/3/6/12 sub-latents per token: WER **0.722 → 0.204 → 0.093** (frame-latent ceiling 0.167). So latent_rate is now a config field, and end to end (same AE, same 800 steps) the student improves on both axes: **WER 2.204 → 1.648**, **DNSMOS 1.43 → 1.63**, still 116× real time. It remains unintelligible, and the oracle path does not — so what is left is **latent prediction**, not the seam | `seam_rate.py`, `latent_rate_ab.py` |
| **Does the objective matter?** | **yes, more than the architecture.** A new `distill-audio` stage trains the text side **through the frozen decoder** against the teacher's audio (with the cached signals as a small aux term to pin length). Same AE, same corpus, rate 3: WER **1.648** (latent L1) → **1.000** (600 steps) → **0.667** (2400 steps), mel cosine 0.9427 → 0.9486, still 120× real time. DNSMOS stays flat (1.63 → 1.58) — recorded, not hidden | `objective_ab.py` |
| **Did the data scale?** | yes: **19.4 min** of Kokoro speech (236 utterances, 59 prompts) with a **prompt-disjoint** split — 129 train (10.9 min, 47 prompts) / 31 val (2.7 min, **12 unseen prompts**). Two borrowed thresholds broke on contact: the PilotTTS `min_dnsmos 3.5` keeps **1 of 160** utterances (Kokoro scores OVRL 2.86 mean; P.808 ~3.9), now calibrated to 2.0 from the measured distribution, and the 5 kHz `narrowband` gate rejects 32 % | `scale_corpus.py` |
| **Does more data help?** | on a **prompt-disjoint hold-out**, 6× more audio improves both proxies (DNSMOS 1.441 → **1.537**, mel cosine 0.9391 → **0.9466**) — but **WER on unseen prompts stays 1.000 in both arms**. So round 23's 1.648 → 0.667 was measured on the corpus's own texts and was partly fitting. The hold-out exists to say that out loud | `data_scale_ab.py` |
| **Is it a data problem?** | the round-24 hold-out said the model had seen only **47 distinct sentences**, so round 25 replaces them with **600 public-domain sentences** from five novels (provenance recorded) → after curation **185 training prompts / 50 unseen val prompts**, 22.2 min of audio. Curation's third borrowed threshold bites here: the 3 s minimum rejects **25 %** of ordinary prose. Phoneme input (the next lever if text alone is not enough) is **verified available**: `espeakng-loader` + `phonemizer` give correct IPA for unseen words over a 39-symbol vocabulary | `build_prompts.py` |
| **A second real teacher?** | **yes — Speechify** (written permission for demonstration/quantization, recorded with the exact scope). It brings the project's **first real alignment**: word-level speech_marks → per-character duration targets, verified to reproduce the audio duration to **0.8 % median error** (964/964 utterances). 2.2 h generated in 43 min at DNSMOS **3.33** vs Kokoro's 2.86; re-curating with a *measured* 4 kHz bandwidth floor kept **964/1350** where the borrowed 5 kHz floor kept 383. Mixture: **1169 train utterances / 109.5 min / 2 teachers, 74 % aligned** | `mix_corpora.py`, `alignment_evidence.py` |
| **Does the mixture fix generalisation?** | **no.** 1169 utterances / 109.5 min / 2 teachers / 74 % aligned / two genders / three locales, trained 1600 steps with zero divergence — and on unseen prompts the student is at **WER 1.000** for both teachers (controls 0.086 and 0.096). With rounds 24–25 that is a **three-part negative result**: corpus duration, text diversity, and teacher count/quality/alignment are each insufficient. Three more sample-rate bugs surfaced and were fixed on the way (a 48 kHz teacher read as 16 kHz by the recogniser, declared 24 kHz in the evaluator, and in the waveform source) | `real_eval.py`, `test_sample_rate_hygiene.py` |
| **Is it generalisation or fitting?** | **fitting.** Phoneme input (the last data-side suspect, now wired) scores WER 1.000/1.026 vs characters' 1.000/1.000 — but the run also showed the student is at **WER 1.000 on its own training prompts**. It never fit the data it trained on, so rounds 24–27 were comparing **undertrained** models: 1600 steps × batch 4 is ~5.5 passes over 1169 utterances. A 6000-step run is in flight and the fit number is what it is judged on first | `tokenizer_ab.py` |
| **Can it use recordings?** | now yes. Eight takes (the same paragraph in **8 voices**, 39–48 s each) all failed curation on the 30 s cap, and arrived with no transcripts: the pipeline gained a **silence segmenter** and an **ingest** path (ASR word timestamps, with the text provenance recorded as a *measurement*, not a teacher transcript). 38/57 segments kept after the calibrated bandwidth floor → **mixture v2: 1207 utterances / 112.7 min / 8 Speechify voices** | `segment.py`, `ingest_audio.py` |
| **What did the text side actually learn?** | a fit diagnosis: it matches the latent's **mean** (cosine 0.805) with almost **no per-token structure** (per-dim correlation **0.126**) and under-predicts length (**0.77×** train, 0.58× val). So the model learned the average latent — which is what the mel loss and cosine reward — not which latent each token needs. That is why the mel proxy looks healthy at WER 1.0 | `fit_diagnosis.py` |
| **Why?** | read the objective: the whole cached-signal bundle is **~1 % of the loss** and the token-latent term inside it is **~0.7 %**, so everything optimises the rendered mel envelope — where matching the *average* latent already works. Added a deliberately **mean-invariant** contrast term (blind to the mean, so it can only be satisfied by matching the variation) and redirected the CPU to it | `losses.py`, `test_latent_contrast.py` |
| **Is it a data problem?** | **no, not by itself.** Two independent levers measured on prompt-disjoint hold-outs: 6× the audio moved DNSMOS 1.44 → 1.54 and WER 1.000 → 1.000; 4× the distinct text (47 → **185 public-domain prompts**, provenance recorded) moved DNSMOS 1.486 → **1.600**, mel cosine 0.9484 → **0.9525** and WER 1.000 → **0.997**. Both proxies improve; intelligibility on unseen sentences does not. The constraint is the text side's inductive bias/capacity, and **phoneme input is verified available** for the next attempt | `text_diversity_ab.py` |
| **A second real teacher?** | **yes — Speechify** (written permission for demonstration/quantization, recorded with the exact scope). It brings the project's **first real alignment**: word-level speech_marks → per-character duration targets, verified to reproduce the audio duration to **0.8 % median error** (964/964 utterances). 2.2 h generated in 43 min at DNSMOS **3.33** vs Kokoro's 2.86; re-curating with a *measured* 4 kHz bandwidth floor kept **964/1350** where the borrowed 5 kHz floor kept 383. Mixture: **1169 train utterances / 109.5 min / 2 teachers, 74 % aligned** | `mix_corpora.py`, `alignment_evidence.py` |
| **Does the mixture fix generalisation?** | **no.** 1169 utterances / 109.5 min / 2 teachers / 74 % aligned / two genders / three locales, trained 1600 steps with zero divergence — and on unseen prompts the student is at **WER 1.000** for both teachers (controls 0.086 and 0.096). With rounds 24–25 that is a **three-part negative result**: corpus duration, text diversity, and teacher count/quality/alignment are each insufficient. Three more sample-rate bugs surfaced and were fixed on the way (a 48 kHz teacher read as 16 kHz by the recogniser, declared 24 kHz in the evaluator, and in the waveform source) | `real_eval.py`, `test_sample_rate_hygiene.py` |
| **Is it generalisation or fitting?** | **fitting.** Phoneme input (the last data-side suspect, now wired) scores WER 1.000/1.026 vs characters' 1.000/1.000 — but the run also showed the student is at **WER 1.000 on its own training prompts**. It never fit the data it trained on, so rounds 24–27 were comparing **undertrained** models: 1600 steps × batch 4 is ~5.5 passes over 1169 utterances. A 6000-step run is in flight and the fit number is what it is judged on first | `tokenizer_ab.py` |
| **Can it use recordings?** | now yes. Eight takes (the same paragraph in **8 voices**, 39–48 s each) all failed curation on the 30 s cap, and arrived with no transcripts: the pipeline gained a **silence segmenter** and an **ingest** path (ASR word timestamps, with the text provenance recorded as a *measurement*, not a teacher transcript). 38/57 segments kept after the calibrated bandwidth floor → **mixture v2: 1207 utterances / 112.7 min / 8 Speechify voices** | `segment.py`, `ingest_audio.py` |
| **What did the text side actually learn?** | a fit diagnosis: it matches the latent's **mean** (cosine 0.805) with almost **no per-token structure** (per-dim correlation **0.126**) and under-predicts length (**0.77×** train, 0.58× val). So the model learned the average latent — which is what the mel loss and cosine reward — not which latent each token needs. That is why the mel proxy looks healthy at WER 1.0 | `fit_diagnosis.py` |
| **Why?** | read the objective: the whole cached-signal bundle is **~1 % of the loss** and the token-latent term inside it is **~0.7 %**, so everything optimises the rendered mel envelope — where matching the *average* latent already works. Added a deliberately **mean-invariant** contrast term (blind to the mean, so it can only be satisfied by matching the variation) and redirected the CPU to it | `losses.py`, `test_latent_contrast.py` |
| Does the whole recipe run? | **yes, offline** — prompts → corpus → cache → distillation → synthesis, 6/6 checks pass with dependency-free fixture teachers | `recipe_dry_run.py` |
| Int8 weights (simulated) | 12.0 MB Tiny / 56.3 MB Small | `smoke_test.py` |

Full protocol, caveats and negative results: [docs/05-VERIFICATION.md](docs/05-VERIFICATION.md).

## Quickstart (CPU, no data, no GPU)

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"     # Windows
python scripts/smoke_test.py --steps 2          # trains every stage, synthesises, quantises
python scripts/learn_demo.py --quick            # proves the stages learn (before/after metrics)
python scripts/reflow_demo.py --quick           # validates NFE-2 sampling after Reflow
python scripts/streaming_demo.py --quick        # blockwise streaming + TTFA vs one-shot
python scripts/mixture_demo.py --quick          # the teacher mixture steering the student
python scripts/voice_demo.py --quick            # multi-voice conditioning vs a control
python scripts/phase_lock_ab.py --quick          # phase-lock filter: lock, fidelity cost, latency
python scripts/resume_demo.py --quick            # crash + resume is byte-for-byte exact
python scripts/recipe_dry_run.py --stage flow    # Small flow path + paired references
python scripts/recipe_dry_run.py --quick        # the WHOLE recipe offline, no teachers needed
python scripts/export_onnx.py                   # int8 ONNX vocoder + PyTorch/ONNX benchmark
python -m pytest -q                            # 272 tests
python scripts/bench_rtf.py --config configs/parakeet_tiny.yaml --steps 2
```

`scripts/smoke_test.py` is the proof-of-life: it exercises all five training stages, the
streaming decoder (which it checks is *numerically identical* to offline decoding), the
phase-lock filter, int8 quantisation, and RTF.  `scripts/learn_demo.py` goes further and measures
whether the autoencoder reconstructs better, whether the distilled text side fits its cached
teacher signals, and whether text→audio beats an untrained text side — on synthetic utterances with
exact token boundaries, so no corpus or GPU is needed.  See
[docs/05-VERIFICATION.md](docs/05-VERIFICATION.md).

## Documentation

| | |
|---|---|
| [docs/00-DESIGN.md](docs/00-DESIGN.md) | The full reasoning: teacher analysis, how the mixture actually works, the five recipe stages, budgets |
| [docs/01-ARCHITECTURE.md](docs/01-ARCHITECTURE.md) | Module-by-module spec with dimensions and measured parameter counts |
| [docs/02-TRAINING.md](docs/02-TRAINING.md) | Stage-by-stage hyperparameters, losses, the spectral-weight anneal, compute estimate |
| [docs/03-DATA.md](docs/03-DATA.md) | Teacher corpus + latent cache + the PilotTTS-style filtering pipeline (with thresholds) |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Generated model card: intended use, out-of-scope, teacher licences, measured results, limitations |
| [docs/04-EVALUATION.md](docs/04-EVALUATION.md) | Metrics, targets to beat, comparison table |
| [docs/LEGAL.md](docs/LEGAL.md) | **Read this before using teacher audio.** Licence/ToS analysis and the safe paths |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Phases, milestones, compute budget, what "done" means |
| [docs/research/](docs/research/) | Source-grounded research notes on all three papers and both teachers |

## Repository map

```
parakeet/
  audio/      mel filterbank, autocorrelation F0 + target scaling, streaming/offline OLA iSTFT
  models/     blocks, speech autoencoder, text/duration/speaker encoders, flow-matching VF, assemblies
  train/      losses (MRSTFT, MPD/MSD, phase, distillation) + the five staged training loops
  data/       text/tags, teacher backends + licence gate, curation filters, feature cache,
              datasets, structured synthetic fixtures for CPU experiments
  inference/  streaming synthesizer, phase-lock filter, int8 + ONNX int8 (QDQ) export and runtime
  eval/       RTF, MCD, spectral convergence, phase coherence, optional UTMOS/WER/SECS, learning probes
configs/      parakeet_tiny.yaml (9.6M) | parakeet_tiny_lite.yaml (6.2M, fixture-derived) |
scripts/      smoke_test.py | learn_demo.py | reflow_demo.py | streaming_demo.py |
              mixture_demo.py | voice_demo.py | recipe_dry_run.py | export_onnx.py | profile_pipeline.py |
              train.py | make_teacher_corpus.py | bench_rtf.py
tests/        272 tests: config, audio DSP, models, losses, all five stages, inference,
              streaming, mixture weighting, data/corpus paths, curation, ONNX/int8, learning
.github/      ci.yml -- suite + smoke test + fast demos on every push; benchmarks on demand
```

## Honest limitations

* **No trained weights yet.** Every quality number in the papers (UTMOS 4.41, WER 5.7 %, RTF 0.02
* **The CLI entry points were bypassing the library.** Round 10 found 	rain.py building its batch source without cross-sample pairing, make_teacher_corpus.py building the cache with no teacher mixture, and curate_manifest (the entire documented P1 pipeline) never called at all.  Fixed structurally: the decisions moved into make_batch_source / cache_teacher_corpus, which are unit-tested.  Wiring curation also exposed two gate defects: digital silence passed the gates, and silence_ratio scored 0.0 for it.
* **The published repository was missing the data package.**  An unanchored .gitignore rule (data/, meant for the corpus directory) also matched the Python package parakeet/data/, so 7 of 33 source files were absent from GitHub for ten commits — everything passed locally, and a fresh clone could not import.  Anchored to /data/, and now guarded by a test that every package file is tracked, verified by cloning and running the tests *from the clone*.
* **The real teacher backends had never been executed — and Orpheus was wrong.**  No demo, test or CI job could run them (a 3B checkpoint, an 82M model, a paid API), so the one path that turns a real teacher's output into training audio was unverified.  Its SNAC codebook-to-level mapping grouped the 7 codes contiguously instead of the published {0}/{1,4}/{2,3,5,6}: every Orpheus corpus would have decoded to noise.  Fixed, cross-checked against two independent copies of the published decoder, and pinned element-wise by contract tests with injected stubs (plus a positive control showing the old grouping fails).
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
