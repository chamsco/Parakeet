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
* [x] **Round 19, failure localisation** (`scripts/real_diagnose.py`): the **autoencoder is the bottleneck** -- its round-trip on real audio is *uncorrelated with its input* (waveform cosine +0.000, SNR -0.10 dB, WER 1.000) while the mel proxy looked merely poor (log-mel L1 1.83).  The per-token->frame seam costs 0.005 mel cosine, so no text-side machinery is implicated.  One real defect found en route: durations were the last prosody target still regressed in raw log space and collapsed to **0.29x**; normalised to the measured corpus statistics (mean 1.728, std 0.390) they predict **0.94x**.
* [x] **Autoencoder: a real training run** (round 20) — an adversarial step costs **24×** a reconstruction step (5.25 s vs 0.22 s, measured from the same seed), so the budget goes reconstruction-first through `run_stage`: 2000 recon + 200 adversarial steps took round-trip WER **1.000 → 0.153** against a 0.000 teacher, mel L1 1.386 → 0.506, 0 divergent steps.  `run_stage` gained a non-finite guard (a hand-rolled loop without warmup diverged to NaN with no symptom until an all-NaN report), and `whisper_wer_with_control` runs the ASR control first and retries a transient failure.
* [x] **Token-expansion seam, part 1** (round 21): `distill-decoder` could not run on a real cache at all -- the shards stored features but no waveform, so `autoencoder_step` raised `KeyError: 'wav'` and the stage had only ever run under `--dry-run`.  The cache now stores the target waveform, `train.py` derives `n_voices` from it, `--warm-start` and `--set` exist, and dry runs exercise the token-expanded path.  A/B (same warm start, 300 steps): seam WER **0.889 -> 0.722**, working paths undamaged, and `prosody_proj` finally trained (removing it now hurts: 0.907 vs 0.722).  The gap to the frame path (0.167) is **not** closed.
* [x] **Token-expansion seam, part 2** (round 22): the seam is mostly **information**.  An oracle sweep over sub-latents per token built from the teacher's own frame latent gives WER 0.722 (rate 1) -> 0.204 (rate 2) -> **0.093** (rate 3) against 0.167 for the frame latent -- 77% of the seam is the 3.9-dimensions-per-frame bottleneck.  `latent_rate` is now a config field (cache + head width + a *shared* `subtoken_spans` geometry, pinned by a test), and end to end the student improves on both axes: WER 2.204 -> 1.648, DNSMOS 1.43 -> 1.63, at 116x real time.
* [x] **The objective, changed** (round 23): a new `distill-audio` stage trains the text side *through the frozen decoder* against the teacher's audio (mel + multi-resolution STFT), with the cached signals as a small auxiliary term because rounding durations to frames is not differentiable.  Same autoencoder, same corpus, rate 3: WER **1.648** (latent L1) -> **1.000** (600 steps) -> **0.667** (2400 steps), mel cosine 0.9427 -> 0.9486, 120x real time.  DNSMOS is flat (1.633 -> 1.583), recorded as a check rather than omitted.
* [x] **Scaled corpus with a real hold-out** (round 24): 59 prompts x 4 voices = 236 utterances, **19.4 min**, prompt-disjoint split (129 train / 31 val over 12 *unseen* prompts), because every prior real-audio number trained and evaluated on the same 21 utterances.  Two borrowed thresholds broke on contact: `min_dnsmos 3.5` keeps **1 of 160** utterances (the same audio scores P.808 3.83-4.04, so the constant is on the wrong scale) and is now calibrated to 2.0 from the measured distribution; the 5 kHz `narrowband` gate rejects 32%.  Also: the scaled run diverged at step 1756 and exposed an incomplete guard -- `run_stage` now checks the *gradients*, not just the loss (NaN grads were being written with a finite loss).
* [x] **Held-out evaluation** (round 24): the scaled autoencoder *generalises* (round-trip WER **0.289** against the teacher's own **0.281** on unseen prompts, mel 0.4597 -- better than 0.4987 on 21 utterances).  The **text side does not**: with 21 vs 129 utterances, WER on 12 unseen prompts is **1.000 in both arms** while DNSMOS (1.441 -> 1.537) and mel cosine (0.9391 -> 0.9466) improve.  Round 23's 1.648 -> 0.667 was therefore measured on the corpus's own texts and was partly fitting.
* [x] **Text diversity** (round 25): the round-24 hold-out named the binding constraint -- the model had seen only **47 distinct sentences**.  `build_prompts.py` takes 600 sentences from five public-domain novels (provenance recorded, 4 810 usable after filtering); 450 prompts x 2 voices = 900 utterances (62.6 min of audio, RTF 0.49) curate down to **304 train utterances over 185 prompts** and 79 val utterances over 50 unseen prompts.  Curation's third borrowed threshold bit: the 3 s CosyVoice minimum rejects 25% of ordinary prose.
* [x] **Phoneme input verified available** -- the next lever if more text is not enough.  The obvious route (misaki -> spacy -> blis) does not build here, but `espeakng-loader` + `phonemizer` produce correct IPA locally, including unseen words (quixotic -> kwɪksɑːɾik) over a 39-symbol vocabulary.
* [x] **Text diversity is not sufficient either** (round 25 result): on the same 50 unseen prompts, 47 -> 185 distinct prompts moved DNSMOS 1.486 -> **1.600**, mel cosine 0.9484 -> **0.9525** and WER 1.000 -> **0.997**.  With round 24 that rules out *both* corpus duration and text diversity as the binding constraint; what remains is the text side's inductive bias and capacity.
* [x] **Second real teacher + the first real alignment** (round 26): Speechify (simba-3.2) under written permission **scoped to demonstration/quantization** -- recorded in the spec and docs/LEGAL.md, with the key in a gitignored secret and a hygiene test that scans tracked files for key-shaped strings.  It returns word-level speech_marks, which become per-character duration targets verified to **0.8% median error** (964/964 utterances) -- the alignment every previous duration target lacked.  2.2 h of audio in 43 min, DNSMOS **3.33** vs Kokoro 2.86; the borrowed 5 kHz bandwidth floor kept 383/1350, a *measured* 4 kHz floor keeps **964**.  \mix_corpora.py\ joins teachers into one mixture: **1169 train utterances / 109.5 min / 74% aligned**.
* [x] **Metric bug fixed**: WER depended on the teacher's sample rate (faster-whisper assumes 16 kHz; 24 kHz audio was read 1.5x fast, 48 kHz 3x).  Speechify's *teacher* WER was 1.488 before the fix and 0.268 after; round 25's conclusion was re-checked under the fixed metric (0.994 vs 0.997) and holds.
* [x] **Mixture trained and evaluated** (round 27): 1169 utterances, 109.5 min, 2 teachers at 0.5/0.5, 74% with the teacher's own alignment, 1600 steps with zero divergence.  On unseen prompts, prose only, with valid controls: student WER **1.000** for both teachers (controls 0.086 Speechify / 0.096 Kokoro).  Three more sample-rate bugs found and fixed: the recogniser assumed 16 kHz, the evaluator declared the config rate for a 48 kHz reference, and `WaveformCorpusSource` fed raw 48 kHz audio to a 24 kHz model.
* [x] **Phoneme input** (round 28): espeak-ng G2P wired in (g2p.py), the IPA inventory **derived** from the phonemiser (the hand-written list missed ː, which appears 1865 times in 600 prompts, and the script ɡ), word timings converted to phoneme frames preserving each word's total, vocabulary fits the embedding.  Result: **WER 1.000/1.026 vs characters' 1.000/1.000** -- phonemes do not fix it either.
* [x] **IT IS FITTING, NOT GENERALISATION** (round 28, the important finding): asked to synthesise its **own training prompts**, the student is still at **WER 1.000** (controls 0.082).  1600 steps x batch 4 = ~5.5 passes over 1169 utterances, with the audio objective still ~1.4 against the autoencoder's own 0.46.  Rounds 24-27 therefore compared *undertrained* models along axes that could not have mattered yet.  Fix the training budget before another A/B.
* [x] **Recording ingestion** (round 29): eight user-provided Speechify takes (same paragraph, **8 voices**, 39-48 s each) all failed curation on the 30 s cap and arrived without transcripts.  Added a silence segmenter (parakeet/data/segment.py) that packs ASR word timings into window-sized utterances without losing a word, and scripts/ingest_audio.py (transcribe -> segment -> curate -> manifest), recording that the text is an ASR *measurement* rather than a teacher transcript.  38/57 segments kept with the calibrated 4 kHz floor; **mixture v2 = 1207 utterances / 112.7 min / 8 Speechify voices**.
* [x] **Mixer accepts mixed layouts**: `--corpus PATH:MANIFEST` so generated corpora (train.jsonl) and ingested ones (curated/kept.jsonl) join one mixture.
* [x] **Fit diagnosis** (round 30): the text side matches the latent's MEAN (cosine 0.805) with almost no per-token structure (**per-dim correlation 0.126**) and under-predicts length (0.77x train / 0.58x val) -- a duration bias, not just noise.  Every metric is worse on validation and none is good on train: **underfitting, not memorisation**.  The mel loss and the cosine reward the mean, which is why the proxy looked healthy at WER 1.0.
* [x] **The objective explains the diagnosis** (round 30): the cached-signal bundle is ~1% of the loss and the token-latent term ~**0.7%**, so the rendered mel envelope dominates -- and matching the average latent already reproduces the envelope.  Added `signal_latent_contrast`: centred MSE (mean-invariant), blind to the mean and unable to cheat by collapsing to it (a constant prediction costs exactly the target variance).  Reachable via `--set` so it can be A/B'd.
* [x] **The objective fix, tested** (round 30): the mean-invariant term raised the per-dimension correlation 0.126 -> **0.200** (val 0.107 -> 0.178), cut F0 error 31% and energy 26%, and halved `aux_latent` (0.618 -> 0.340) -- and WER stayed at 1.000.  The quantity moved; intelligibility did not.  At ~0.2 correlation, a one-shot regression from a small character encoder has a ceiling no weighting reaches.
* [x] **The Radeon, measured** (round 31): `torch-directml` in a Python 3.11 venv sees the RX 6950 XT.  Text step **1.0-1.5x**, decoder conv stack **6.1x**, and the full audio step **cannot complete** -- DirectML has no complex dtype and is missing/broken on `col2im`, `eye`, `repeat_interleave` and `index_add`, with an error path that raises `UnicodeDecodeError` instead of reporting.  **DirectML is not the unlock**; real GPU training of gfx1030 needs ROCm on Linux/WSL2.  Kept from the attempt: a real-valued iSTFT/magnitude path (tested equal to `torch.istft`/`torch.stft`), an overlap-add without `F.fold`, and a device-safe index in `align_token_frames`.
* [x] **The flow route, on real data** (round 32): `configs/parakeet_flow.yaml` (flow model + **Tiny autoencoder geometry**, since the Small config's 384-wide AE would pair a decoder with latents from a different encoder), `--warm-start` now skips shape-mismatched keys instead of failing, `real_eval.py` reports the **length ratio** and survives a near-zero-length sample, `--steps` selects NFE.  Measured: flow **2.02 s/step at batch 8 (~73 s of audio)** vs the token path's 9.1 s -- ~8x more audio per unit of compute.  **Correction:** round 31 claimed the flow had no synthesis path; it does (ParakeetFlow.synthesize), it had simply never been trained on real data.
* [x] **ROOT CAUSE FOUND** (round 36): the cache stores **normalised** latents (it_latent_normalizer -> 
ormalize) and the fitted statistics were **never persisted**, so denormalize was the identity (flow checkpoints: n=0) or used **stale** values (Tiny checkpoints: var 1.90 against the cache's true 0.437).  Measured on one cached latent through the same decoder: **encoder(audio) -> WER 0.000, the cached latent -> 1.000, the cached latent with the recovered inverse -> 0.000** (per-dim correlation 1.000, affine residual 0.0000).  Every run trained and synthesised in the wrong latent space.  Fixed: the statistics are persisted in cache_meta.json, scripts/repair_latent_norm.py recovers them for existing caches, load_latent_norm_from_cache() installs them in **training and evaluation**, and tests pin the invariant (an unfitted normaliser does NOT undo a fitted transform -- exactly the bug).
* [x] **Acceptance criterion** (round 35): speechlikeness() -- voiced fraction, median F0, spectral flatness.  The teacher passes; both trained paths fail (Tiny = 0.96 voiced tonal buzz, flow = 0.07 voiced noise).  WER 1.0 could not distinguish wrong words from not-speech.
* [ ] **IN FLIGHT: the capacity question** (round 36).  A pure distill-text run (no decoder in the loop) with the fixes reaches train per-dim correlation **0.285** at step 3000 but only **0.118** on validation -- better than the audio-stage models (0.145-0.200) yet far from usable.  Running the same stage with **text.dim 512 / 6 layers (22.6M trainable against 4.6M)** for a direct capacity comparison at equal steps.
* [ ] **Watch the flow run** (4000 steps, checkpoint every 400): the first thing that must come right is the **length ratio** (0.008 untrained), then WER at NFE 4.
* [ ] **Ask the operator about WSL2/ROCm** -- the prerequisite for using the GPU for real; meanwhile training stays on the CPU, where the binding constraint is step count.
* [ ] **NEXT: move Tiny's acoustic stage onto the flow decoder** (the paper-faithful path, already implemented for Small and able to read the same cache).  The token side keeps durations/prosody; a flow-matching decoder produces the frame latents instead of a one-shot 72-dim regression.
 (run in flight: 2400 steps, `audio_aux=1.0`, `signal_latent_contrast=1.0`) -- if the per-dimension correlation climbs, continue and then measure WER; if it does not, move Tiny's acoustic stage onto the flow decoder the Small path implements.
* [ ] **The structural fix this points at**: a one-shot regression of 72 latent values per token from a small character encoder is being asked to learn speech acoustics from text.  The papers run a **flow-matching / AR decoder** over acoustic tokens -- which this repository's Small path implements and the Tiny path skipped.  Test more optimisation first (6000-step run in flight), then move Tiny's acoustic stage onto the flow decoder.
* [ ] **Train until it fits** (in flight: 6000 steps, warm-started) -- the acceptance criterion is train-prompt WER well below 1.0, then generalisation, then the A/Bs can be revisited.
 (rounds 24-27): duration, text diversity, teacher count, teacher quality and real alignment are each insufficient.  What remains is the text side's **inductive bias and capacity** -- next: phoneme input (capability verified) and/or a larger text side, A/B'd on the same 50-prompt hold-out.
* [ ] **Phoneme input** (next, capability verified): character input has to relearn English spelling-to-sound from a few hundred sentences, while the papers feed phonemes.  `espeakng-loader` + `phonemizer` give IPA locally over a 39-symbol vocabulary; the change touches the tokenizer, the cache targets and the ONNX export, so it needs its own A/B on the same hold-out.
 -- not renderer capacity, corpus size, latent rate or the objective.  Next levers: a much larger and more varied corpus (synthesis is RTF 0.52, so this is cheap) and a text side with a stronger inductive bias across unseen sentences.
* [ ] **Push the audio objective further** -- WER 0.667 is still not usable; the trend says more steps help, and the next levers are the auxiliary weight, the decoder's own capacity (its round-trip ceiling is 0.167), and the corpus (21 utterances).
 -- the student is still unintelligible while the oracle token path is not, and the text side fits its latent L1 well (0.655) while rendering wrong audio.  Change the objective, not the architecture: train the text side *through the decoder* with a mel/audio loss instead of an L1 on cached latents.
 -- the residual is structural: the cached per-token latent is an *average* over ~6 frames, so raise the effective token rate (sub-token latents) or add a refinement stage that maps token latents to frame latents (the flow/consistency sampler's role in the Small model).
 — the new measured bottleneck, exposed only once the autoencoder worked: per-token latents expanded as inference does give WER 0.870 where the frame-level latent gives 0.167, for just 0.016 of mel cosine.  `prosody_proj` being untrained was hypothesised and **falsified** (0.926 without it, i.e. slightly worse).  Likely cause: the token latent is an *average* over ~6 frames, and `distill-decoder` trains on the clean frame latent while its own docstring promises the token-expanded distribution.
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
* [x] Attack the python/dispatch overhead: cProfile found the phase-lock rebuilding its per-call constants (grid construction 0.386 ms -> 0.4 us cached); a controlled A/B measured 7.101 -> 6.458 ms and the shipped report went 70.6 -> 74.0x real time
* [x] **Capacity ablation** (scripts/ablate.py, fixture-derived): a dim128-L4 text side (1.008 M) is within 2.3 % of the shipped dim256-L4 held-out fit for 35 % fewer total parameters and 2.6x the text-side speed; offered as `configs/parakeet_tiny_lite.yaml`.  The first version had no held-out split and would have recommended the *worst* variant on unseen data
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
* [x] Model card, **generated from evidence** (`scripts/model_card.py` -> `docs/MODEL_CARD.md`): 15 claims, each citing a report file and JSON path; missing evidence renders as unmeasured and a bad value as FAILING; the cited evidence is committed under `docs/evidence/` with a SHA-256 manifest so a fresh clone can verify it; the licence table comes from the teacher specs.  CI (workflow_dispatch) re-runs every demo, refreshes the bundle and fails on `git diff --exit-code`, so the card cannot rot
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
