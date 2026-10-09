# Roadmap

Ordered so that every phase ends with something measurable, and the legal question never blocks
the engineering.

## P0 — Framework (done, this repo)

* [x] Autoencoder: mel → 24-dim latent → causal iSTFT waveform, streaming-exact
* [x] Tiny (9.62 M) and Small (45.04 M) assemblies with measured parameter budgets
* [x] Flow matching with `Kc=6`, `Ke=4` context sharing, CFG dropout, `null_memory`
* [x] Reflow / few-step distillation path with EMA teacher
* [x] Losses: log-mel, multi-resolution STFT, MPD+MSD adversarial, phase-linearity, distillation
* [x] Teacher backends with a licence gate; corpus builder; latent shard cache
* [x] Streaming synthesizer, phase-lock filter (two lock references), int8 quantisation
* [x] 105 tests + a CPU smoke test that trains **every** stage and synthesises audio
* [x] **Learning demo** (`scripts/learn_demo.py`): provably learns end to end on synthetic speech —
      AE reconstruction 36 % better, text side 91 % better, text→audio 26 % better than an
      untrained text side, duration MAE 3 ms, generated log-mel cosine 0.954 vs target
* [x] Measured: Tiny RTF **0.044 on one CPU thread (22.6× real time)**, int8 12.0 MB
* [x] Reference-free curation filters implemented and tested (`parakeet/data/curate.py`)

## P1 — Real representation (next; no teacher needed)

* [x] Curation pipeline code (gates, WER agreement, punctuation gaps, ratio tails, keep-rejects)
* [ ] Run it over 100–300 h of *real* speech and freeze the manifest
      ([03-DATA.md](03-DATA.md)); inject DNSMOS and a real ASR ensemble
* [ ] Train the autoencoder (with the speaker encoder) on it; publish the mel-reconstruction
      baseline and the latent statistics
* [ ] Gate: `phase_coherence` of decoded real speech within noise floor of the source; latent
      mean/var stable; no dead latent dimensions
* [ ] **Exit criterion:** AE can reconstruct unseen real speech at a quality where a second
      listener cannot identify the reconstruction in an ABX test — this caps everything downstream

## P2 — Tiny, fully distilled (first real checkpoint)

* [ ] Build a 2–5 k-prompt teacher corpus (Orpheus 60 / Kokoro 40), cache latents + signals.
* [ ] Train text side (Paradee's 8 k-step schedule) and decoder with the 45→10→3 anneal.
* [ ] Join halves, quantise to int8, enable the phase lock.
* [ ] Measure and publish: UTMOS, WER, SECS, RTF (1 thread, fp32 and int8-ONNX), int8 size.
* [ ] Replace the README's target row with real numbers.
* [ ] **Exit criterion:** ≥ 4.30 UTMOS and ≥ 20× real time on one CPU thread at ≤ 15 MB — i.e.
      Paradee-class on a corpus we built ourselves.

## P3 — Small + few-step sampling

* [x] **Reflow validated on CPU** (`scripts/reflow_demo.py`): the reflowed 2-step sampler agrees with
      the NFE-32 reference better than the teacher's own NFE-16 discretisation (latent MSE 1.506 vs
      1.848) and beats a naive 2-step cut by 29 % (2.125); in audio space it is 32 % closer; 16×
      fewer VF passes gives a 9.7× wall-clock speed-up
* [ ] Scale the teacher corpus to 50–200 h, add tag/emotion parallel data (1–10 h) with 50/50
      mixed-prompt SFT
* [ ] Train flow matching (Ke 4, CFG 3), then Reflow to NFE 2–4
* [ ] Load frozen CAM++ and validate zero-shot cloning (SECS) and style factorisation
* [ ] **Exit criterion:** NFE ≤ 4 at WER ≤ 3 % on held-out text — no NFE-collapse.  Note the
      caveat the CPU experiment surfaced: at small training budgets the *model* (not the sampler) is
      the bottleneck (the NFE-32 endpoint is further from the data than a 2-step cut), so this
      criterion is only meaningful with real data and a real budget

## P4 — Release engineering

* [x] ONNX export of the decoder compute with dynamic batch/time axes, int8 QDQ quantisation with a
      calibration set, a runtime wrapper that keeps streaming iSTFT, and a **measured** PyTorch vs
      ONNX-fp32 vs ONNX-int8 benchmark (`scripts/export_onnx.py`) — the real-time story has to be
      judged on ONNX, not PyTorch
* [x] **Export the text side too** (it is ~31 % of the budget).  Full int8 pipeline on one CPU
      thread: **5.4 ms for 0.57 s → 106× real time** (3.9× vs PyTorch), **9.9 MB** total, waveform
      cosine ≥ 0.992 against the PyTorch pipeline.  `OnnxTinyPipeline` composes both halves
* [ ] A streaming sampler for Small (currently only decoding streams)
* [ ] Attack the 13 % python/dispatch overhead measured by `scripts/profile_pipeline.py`
* [ ] A/B the two phase-lock references (`ramp` vs `smooth`) against UTMOS and phase coherence (the
      filter is 10 % of the shipped path, so this trades quality against latency)
* [ ] Model card: teachers used, licence obligations, intended/misuse cases, deep-synthesis
      marking, provenance hashes
* [ ] CI job that runs `smoke_test.py` + `learn_demo.py` + `pytest` and fails on RTF, streaming
      parity, or learning-metric regression

## P5 — Optional quality work

* [ ] Multi-voice Tiny (`n_voices > 1`) and per-voice constant styles.
* [ ] Multilingual text (the char-level encoder makes this an alphabet problem, not a G2P problem).
* [ ] Consistency distillation as an alternative to Reflow; compare NFE 1 feasibility.
* [ ] Soft/distributional teacher targets to counteract the synthetic-data failure mode.
* [ ] Prosody transfer experiments (swap style tokens between speakers of the same language).

## Blocked / needs a decision

* **MiniMax as a teacher** — blocked on written permission ([LEGAL.md](LEGAL.md)). The code path
  exists; the gate refuses. Everything above is achievable without it (Orpheus + Kokoro), which is
  why that is the default mixture.
* **Public release of Orpheus-derived weights** — needs legal review of the Llama-3.2 chain.

## Compute budget summary

| Phase | Compute | Wall clock (1× 24 GB GPU, assuming ~1 GPU-day per 30 k steps at our sizes) |
|---|---|---|
| P1 curation | CPU only, ~100 h audio | 1–3 days CPU |
| P1 AE training | ~2–4 GPU-days | 3–5 days |
| P2 Tiny | < 2 GPU-days | 2 days |
| P3 Small | 5–10 GPU-days | 1–2 weeks |
| P4 release | < 1 GPU-day | 2–3 days |

Total to a released pair of checkpoints: **~2–3 GPU-weeks**, dominated by the autoencoder and the
Small flow model. Re-measure steps/second early in P2 and correct this table — it is an estimate,
not a measurement.
