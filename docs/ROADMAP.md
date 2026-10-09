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
* [x] **Streaming sampler for Small** (`ParakeetFlow.synthesize_stream`): blockwise ODE
      integration over the vector field's finite 18-frame context, plus a chunked phase-lock filter.
      Measured: TTFA 979 ms → 125 ms at 21.9 s of audio (**7.81×**, and roughly flat in length),
      blockwise output 50–100× closer to the one-shot result than an independent draw, interior
      streaming audio matching the offline filter at cosine 0.9994.  Cost is explicit: 2.81× total
      compute at block 16, 1.07× at block 128
* [ ] Attack the 13 % python/dispatch overhead measured by `scripts/profile_pipeline.py`
* [x] A/B the two phase-lock references (`ramp` vs `smooth`) in `scripts/phase_lock_ab.py`. The
      delay grid was **under-configured**: 64 points over [0, 1/60 s] is 260 µs, more than two
      periods of phase error at 8 kHz. At matched strength 64 → 256 triples the coherence the filter
      adds to glottal-locked speech while *halving* what it adds to white noise, so 256 is now the
      default in both the offline and streaming filters. Two negative findings are reported rather
      than buried: `smooth` is inert on both the within-frame and the temporal statistic, and the
      filter adds coherence to noise too, so the metric cannot demonstrate speech-specific locking.
      Perceptual benefit remains unmeasured (no UTMOS here), and the fixtures cannot serve as the
      test signal at all (they randomise harmonic phase and score like noise)
* [ ] Re-run the A/B against UTMOS on real audio, and reconsider the band and `strength` with a
      perceptual metric rather than a phase statistic
* [x] CI: `.github/workflows/ci.yml` runs the suite plus the smoke test and three fast demos on
      every push (Python 3.11 + 3.13); timing-sensitive benchmarks are a `workflow_dispatch` job
* [x] End-to-end recipe dry run with dependency-free fixture teachers
      (`scripts/recipe_dry_run.py`) — it immediately found two bugs in the documented cache path
      that no per-part demo could see
* [x] **Real teacher backends contract-tested with stubs** (no weights, network or API key, so they run in CI) -- which found that the Orpheus SNAC codebook-to-level mapping had been wrong all along: contiguous grouping instead of the published `{0}`/`{1,4}`/`{2,3,5,6}`, decoding every corpus to noise.  Verified against two independent copies of the published decoder.
* [x] **Resumable checkpoints**: save_checkpoint now writes optimizer, EMA, discriminator, step, RNG (torch/python/numpy) and the batch source's order state, and un_stage(resume_from=...) repositions the LR schedule -- previously --resume restored only the model, so a resumed run restarted the schedule and reshuffled its data.  Verified bit-exact against an uninterrupted run on the real cache path (esume_demo.py, 120 steps).
* [x] **Repository hygiene + provenance**: every package file must be tracked by git (an unanchored data/ ignore rule had hidden the whole data pipeline for ten commits — verified by cloning and running the tests from the clone); un_stage writes un.json with git revision, config hash, versions and the trainable/frozen report; no-dead-public-API test.
* [x] **Entry-point wiring**: `make_batch_source` and `cache_teacher_corpus` moved the batch-source
      and cache decisions out of the scripts, which had been bypassing cross-sample pairing, the
      reference cap and the teacher mixture entirely; `curate_manifest` (the whole P1 pipeline) was
      dead code and now runs by default. Wiring it exposed two real gate defects: digital silence
      passed (no absolute level gate) and `silence_ratio` scored 0.0 for it.
* [ ] Model card: teachers used, licence obligations, intended/misuse cases, deep-synthesis
      marking, provenance hashes

## P5 — Optional quality work

* [x] **Multi-voice Tiny**: the manifest's per-record voice reaches the model (cache → collation →
      loss), the voice embedding conditions the *whole* text side (not just the latent feature), and
      a fixture demo shows the student reproducing three voices' relative pitch (81.7 / 88.8 /
      148.6 Hz against fixtures at 80.8 / 95 / 152) with a 7× better per-voice fit than a
      voice-blind control.  The work also fixed three bugs in the F0 **target** pipeline: an
      unbounded fixture pitch sweep, a formant-biased default estimator (autocorrelation reported
      168 Hz for an 81 Hz voice), and per-token aggregation that averaged in unvoiced zeros
* [x] **Speaker/style conditioning from the cache**: `collate` was dropping `log_mel`, so the Small
      model trained from a cache with `ref_mel=None` — zero speaker embedding, no style tokens, and
      no way to learn identity or style. Now the reference is padded/masked into the batch,
      `pair_references=True` implements PilotTTS cross-sample pairing (a *different* utterance of the
      same voice, plus a different-voice negative), and the style term is a **separation** loss
      rather than a same-speaker consistency loss that leaked identity into the style channel.
      Verified with a control (the identity/style encoders get exactly zero gradient without a
      reference) and by `recipe_dry_run.py --stage flow`.
* [ ] Per-voice constant styles for Tiny 2.0; speaker-encoder fine-tuning on real data
* [ ] Consistency distillation as an alternative to Reflow; compare NFE 1 feasibility
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
