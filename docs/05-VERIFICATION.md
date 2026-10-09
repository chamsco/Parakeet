# Verification: what is actually proven by this repository

Everything below was produced by running the code in this repo on a Windows machine with an
AMD Ryzen 7 7800X3D (8 cores), 31 GB RAM, **no GPU**, Python 3.13.14, torch 2.9.1+cpu.

Reproduce with:

```bash
python scripts/smoke_test.py --steps 2 --out runs/smoke     # trains every stage, synthesises
python scripts/learn_demo.py                                # proves the stages learn (~10 min CPU)
python scripts/reflow_demo.py                               # validates NFE-2 sampling (~5 min CPU)
python scripts/streaming_demo.py                            # blockwise streaming + TTFA (~7 min CPU)
python scripts/export_onnx.py --pipeline                    # int8 ONNX + runtime benchmark
python scripts/profile_pipeline.py                          # where does the time actually go
python -m pytest -q                                         # 204 tests
python scripts/bench_rtf.py --config configs/parakeet_tiny.yaml
```

## 1. Test suite

```
204 passed
```

Coverage by area:

| File | What it pins down |
|---|---|
| `test_config.py` | defaults valid, YAML round-trip, typo rejection, cross-field validation (`latent_dim`, `cond_dim`) |
| `test_audio.py` | OLA iSTFT matches `torch.istft` to **1e-4**; streaming OLA == offline to **1e-5**; mel shape; F0 on a 200 Hz synthetic tone within **6 Hz**; F0-bin round-trip |
| `test_models.py` | Tiny stays **8–12 M**, Small lands **40–52 M** on the shipped configs; `fold/unfold` invertible; rectified-flow endpoints; `Ke=3` really triples the estimator batch (forward-hook measured); samplers restore `train()` mode |
| `test_losses.py` | MRSTFT ≈ 0 for identical signals and differentiable otherwise; MPD/MSD generator+discriminator steps backprop; phase-linearity loss prefers locked over random phase by construction; annealer 45→10→3; multi-teacher weights normalise and never drop a teacher entirely |
| `test_train_stages.py` | **all five stages** run, produce finite losses and checkpoints; `distill-decoder` freezes the encoder; `distill-text` freezes the autoencoder; 20 steps of `distill-text` measurably **reduce** loss |
| `test_inference.py` | Tiny and Small synthesize finite audio; phase lock preserves loudness (<5 %) and spectral envelope (cosine >0.99); phase lock **increases** coherence on a randomised-phase voiced signal; **streaming decoder == offline decoder to <1e-4**; int8 shrinks the model; 4-bit without per-channel is refused; wav round-trip |
| `test_data_and_text.py` | tag-aware normalisation (numbers→words, tags preserved), tokeniser round-trip, vocab fits embedding capacity; MiniMax is **refused by default**; corpus builder writes a manifest and interleaves both teachers; latent shard cache end-to-end + collate; synthetic batch source key sets |
| `test_curate.py` | every curation gate fires on a constructed failure (too short/long, clipped, silent, low SNR, narrowband, low MOS, ASR disagreement); **all** reasons reported, not just the first; SNR correctly reported as *unevaluable* without a noise floor; WER and punctuation-gap maths; reject records are never dropped |
| `test_learning.py` | synthetic fixture is structurally exact (frame counts, peak, F0 declination); latent normaliser fits and inverts; token targets are exact and normalised; decoder-latent alignment; supplied-latent path leaves the encoder gradient-free; **stage freezing does not leak between stages**; and the headline: 40 CPU steps measurably improve both the representation and the distilled text side |
| `test_onnx.py` | ONNX decoder matches PyTorch to **<1e-4**; text side matches to **<1e-4 across token lengths 5/8/17** (the legacy exporter's baked-in length would fail this); dynamic time axis across 7/23/41 frames; int8 files are smaller and run on CPU; dtypes are validated at the wrapper boundary; external weight sidecars are counted in size; full int8 pipeline tracks PyTorch (cosine >0.95) and is smaller in total (skips if `onnx`/`onnxruntime`/`onnxscript` absent) |
| `test_resume.py` | **an interrupted-then-resumed run is bit-identical to an uninterrupted one** (max parameter difference exactly 0); checkpoints carry model + optimizer + EMA + discriminator + step + RNG; the RNG stream is restored (and can be opted out); the LR schedule continues instead of restarting; batch-order state round-trips; the step resumed from is reported; a missing checkpoint raises |
| `test_teacher_backends.py` | the three **real** teacher backends (the only code that had never been executed) verified with injected stubs: the Orpheus codebook-to-SNAC-level mapping asserted element-wise against a reimplementation of the published decoder — with a positive control proving the old contiguous grouping fails it — plus level shapes, partial-frame dropping, token filtering, prompt wrapping and sampling settings; Kokoro chunk concatenation, empty output and durations fallback; MiniMax request payload/headers, hex WAV decode, missing-audio error and the licence gate |
| `test_provenance_and_hygiene.py` | **every `.py` under `parakeet/` is tracked by git** (the unanchored `data/` ignore rule hid the whole data package for ten commits), plus scripts/CI/configs ship; no public name in the package is referenced nowhere (with an explicit allowlist escape hatch); `run_stage` writes `run.json` with git revision, config SHA-256, versions, and the trainable/frozen report, and merges caller provenance; `SpeakerConfig.freeze` really freezes the identity encoder while the Q-Former adapts; a saved checkpoint round-trips from both a raw encoder state dict and a full-model state dict, and a mismatched one raises |
| `test_pipeline_wiring.py` | `make_batch_source` pairs references for the flow stage only (and honours the config cap, and falls back to synthetic batches); `cache_teacher_corpus` takes the mixture from `corpus_meta.json` (the CLI used to pass none), prefers a curated `kept.jsonl` including in `curated/`, and errors without a manifest; the P1 gates discriminate, **reject digital silence even with duration/bandwidth/SNR relaxed** (`min_rms_dbfs`), and score `silence_ratio` 1.0 for it; a curated manifest round-trips into a valid cache |
| `test_conditioning.py` | the cache batch carries a padded, masked reference (and legacy caches without `log_mel` still collate); pairing never uses the target utterance and takes the positive from the same voice and the negative from a different one; `max_ref_frames` truncates like PilotTTS's 15 s cap; **with no reference the identity/style encoders receive exactly zero gradient** (the control for the pre-fix cached path) while with one they receive gradient and the separation term is active; the separation loss pushes different speakers apart and the consistency term is off by default |
| `test_voice.py` | the fixture voices really are multi-voice (measured monotone pitch); the voice embedding conditions **every** head (duration/F0/energy/latent — the first version only modulated the latent); the default voice is index 0; the cache records per-voice indices and rejects a corpus with more voices than the model has; **cached F0 targets follow the voice pitch** (fails if the unbounded fixture sweep, the formant-biased estimator, or the unvoiced-zero averaging regresses); YIN is the default and is accurate; per-token aggregation ignores unvoiced zeros |
| `test_streaming.py` | the reported vector-field context is confirmed **empirically** (perturb a frame, the response stops exactly at `3 × depth`); `layer_scale_init` leaves the VF per-frame at init (documented, not hidden); a single block equals the one-shot sampler exactly; multi-block stays closer to one-shot than an independent draw; blocks cover every frame in order and concatenate to the batch result; chunked phase lock matches offline (cosine >0.99) and buffers a full `n_fft`; streaming synthesis yields several chunks whose total length matches the one-shot path, with and without the filter |

## 2. Learning demo (measured)

`scripts/learn_demo.py` trains the real Tiny architecture (9.616 M) on 16 structured synthetic
utterances (**15.0 s** of audio, two duration layouts, 8 character tokens each) for 250 autoencoder
steps, 400 text-side steps and 120 decoder steps — **569 s on CPU**, no GPU, no data, no network.

| Measurement | Before | After | Criteria | Result |
|---|---|---|---|---|
| autoencoder reconstruction (log-mel L1, independent of the training loss) | 2.5980 | **1.6637** | < 0.90 × | **PASS** (36.0 % better) |
| Tiny text side on cached teacher signals | 5.1649 | **0.4673** | < 0.50 × | **PASS** (91.0 % better) |
| end-to-end text → audio (log-mel L1 vs target) | 2.1526 | **1.5890** | < 0.90 × | **PASS** (26.2 % better) |
| per-token duration MAE | — | **0.30 frames (3 ms)** | ≤ 2.0 frames | **PASS** |

The end-to-end baseline uses the *same trained rendering stack* with an untrained text side, so the
26 % improvement isolates the distilled halves rather than the autoencoder.

Audio-level sanity checks on the synthesised sample (not part of the pass criteria, but the numbers
that make the above believable):

| | duration | RMS | peak | log-mel cosine vs target |
|---|---|---|---|---|
| target | 0.853 s | 0.0850 | 0.300 | 1.000 |
| **generated** | 0.821 s (ratio **0.963**) | 0.0435 | 0.399 | **0.954** |
| generated + phase lock | 0.821 s | 0.0435 | 0.400 | — |
| untrained text-side baseline | **0.064 s** | 0.0817 | 0.445 | — |

Two things worth reading from this table: the generated audio has a **0.954 spectral cosine** with
the target and a duration within 4 %, while the untrained baseline collapses to 0.064 s of nonsense
— i.e. the duration and content behaviour is genuinely learned, not an artefact of the decoder.  The
phase lock leaves RMS and peak untouched (as designed: it is a phase-only, magnitude-preserving
filter) and raises 2–8 kHz phase coherence from 0.136 to 0.178.

Listen to `runs/learn_demo/{target,generated,generated_phase_locked,baseline_untrained_text}.wav`.
Full numbers: `runs/learn_demo/report.json`.

## 3. ONNX export + int8 (measured)

`scripts/export_onnx.py` has two modes. `--pipeline` exports **both halves** — the Tiny text side
(attention-based, ~34 % of the budget) and the decoder compute (~40 %) — with dynamic axes,
quantises both to int8, composes them with the cheap torch stages, and verifies equivalence against
the pure-PyTorch path. Without `--pipeline` it benchmarks the decoder alone.

### Full Tiny pipeline, one CPU thread, mean 0.572 s audio

| pipeline | latency | real time | vs PyTorch |
|---|---|---|---|
| PyTorch fp32 | 21.20 ms | 27.0× | 1.00× |
| ONNX fp32 | 11.86 ms | 48.3× | 1.79× |
| **ONNX int8** | **5.42 ms** | **105.7×** | **3.91×** |
| **ONNX int8 + phase lock (shipped)** | **8.11 ms** | **70.6×** | **2.61×** |

Total model: **36.73 MB → 9.91 MB** (text side 18.76 → 5.22 MB, vocoder 17.97 → 4.68 MB).
Equivalence against PyTorch on the same weights: **waveform cosine 0.992–0.9998**, mel L1
0.010–0.027 — i.e. int8 changes the audio by about 1 % of full scale. (The spread is across runs
with *different random initialisations*: each run builds a fresh untrained model, so this is a
lower bound — a trained model quantises more gracefully than a random one.)

### Decoder alone

```
  PyTorch fp32          10.72 ms     94.52x real time
  ONNX fp32              8.72 ms     17.97 MB   1.23x vs PyTorch
  ONNX int8 (QDQ)        4.24 ms      4.68 MB   2.53x vs PyTorch
  int8 size reduction 3.84x | int8 speedup vs ONNX fp32 2.06x
  int8 deviation: |dlog_mag|max 0.0154, |dphase|max 0.0149 rad
```

This CPU is a Zen 4 (AVX-512 VNNI), which is why int8 *Conv* beats fp32 here — hence measuring
rather than assuming. Numerical parity of the fp32 graphs with PyTorch is asserted in
`tests/test_onnx.py` (decoder `<1e-4`, text side `<1e-4` across token lengths 5/8/17), along with
int8 size/run checks, dynamic-time-axis behaviour and pipeline equivalence. Those tests skip
cleanly when `onnx`/`onnxruntime`/`onnxscript` are absent.

### Two traps found while building this, both now covered

* **The legacy TorchScript exporter silently bakes the dummy sequence length into
  `nn.MultiheadAttention`'s reshapes.** The text side exported "successfully" with `dynamo=False`
  and then failed at inference on any sentence of a different length — a deploy-time landmine.
  `export_text_side_onnx` now defaults to the `torch.export`-based exporter (`dynamo=True`), and
  the parity test deliberately uses token lengths different from the traced one.
* **The dynamo exporter writes weights to an external `.onnx.data` sidecar**, so the graph file
  alone looked like 0.47 MB for a 3.85 M-parameter model. `onnx_artifact_bytes` sums the sidecar and
  every size reported here goes through it (`test_onnx_artifact_bytes_counts_external_data`).
  We also pass `external_data=False` for a single-file artifact.
* ONNX Runtime **casts** a float input to int64 embedding indices instead of erroring, so the
  runtime wrappers now validate dtype and raise (`test_text_side_onnx_rejects_wrong_dtype`,
  `test_vocoder_rejects_non_float_latent`).

## 4. Pipeline profile (measured)

`scripts/profile_pipeline.py` profiles a full Tiny synthesis, one CPU thread, mean of three
sentences (**0.612 s** of audio, 4 NFE, 9.62 M params). Shares are of the shipped configuration
(synthesis + phase-lock filter), which is the number that matters:

| component | ms | % of shipped | standalone × real time |
|---|---|---|---|
| decoder + iSTFT | 7.94 | **35.6 %** | 77× |
| text side (encoder + duration/F0/energy/latent heads) | 6.80 | **30.5 %** | 90× |
| phase-lock filter | 2.23 | 10.0 % | 275× |
| latent construction (align + prosody + de-normalise) | 0.29 | 1.3 % | 2075× |
| python/dispatch overhead | 2.98 | 13.4 % | — |
| FULL (no filter) | 20.24 | 90.8 % | 30.2× |
| **SHIPPED (with filter)** | **22.29** | 100 % | **27.4×** |

**Finding: the vocoder is not the bottleneck.** It is ~36 % of the budget with the text side taking
~30 %, so int8-ONNX-ing both halves (as above) is what moves the needle: 27× → 71× for the shipped
configuration. Remaining levers, in order: the 13 % python/dispatch overhead, then the phase-lock
filter's 10 %.

## 5. Few-step sampling (Reflow) validation (measured)

This is the experiment behind the "lightning fast" claim for Parakeet-Small: SupertonicTTS needs
NFE 32, and cutting steps naively collapses quality (their WER 2.64 → 11.43 at NFE 4). Our recipe
adds a Reflow stage to make NFE 2 viable. `scripts/reflow_demo.py` tests that on CPU with no
corpus — 45.04 M model, 200 autoencoder fixture steps, 300 flow steps, 150 Reflow steps, **~5 min**:

```
  latent MSE vs teacher@32:  teacher@16 1.8476 | naive@2 2.1248 | reflow@2 1.5056
  audio log-mel L1 vs decoded reference: naive@2 0.3128 | reflow@2 0.2130
  timing: NFE 32 = 275 ms | NFE 2 = 28 ms  ->  9.7x wall-clock (16x fewer VF passes)
```

* **The reflowed 2-step sampler agrees with the NFE-32 reference better than the teacher's own
  NFE-16 discretisation** (1.506 vs 1.848) and 29 % better than a naive 2-step cut (2.125). In audio
  space, comparing both samplers against the *reference decoded through the same autoencoder* (so
  the AE's error is common-mode), Reflow is 32 % closer (0.213 vs 0.313).
* 16× fewer velocity-field passes gives **9.7× wall-clock** speed-up at this sequence length — the
  gap is the once-per-sample conditioning work, which is why the number is not 16×.

**And the honest negative result, which is the more useful half:** this experiment validates the
*sampler*, not the *model*. With 15 s of synthetic audio the flow is nowhere near converged — the
NFE-32 endpoint sits **2.24** from the data latent while a naive 2-step cut sits at **0.79**, i.e. at
this training budget *sampling longer makes things worse*, and Reflow faithfully reproduces a
teacher that is still wrong. Likewise the mel-vs-target metric is saturated: the autoencoder's own
round-trip error (1.669) is **21× larger** than the difference between the two samplers (0.081), so
it cannot discriminate at this fixture quality. Both are reported as diagnostics in
`runs/reflow_demo/report.json`, not as pass criteria.

## 6. Streaming (blockwise) sampling (measured)

The one-shot sampler must integrate the whole latent before any audio exists, so time-to-first-audio
grows with the text.  Parakeet's vector field mixes time with finite-support depthwise convolutions
rather than a transformer, so the ODE can be integrated block by block with a bounded window.
`scripts/streaming_demo.py` (45 M model, 200 flow steps, **~7 min** CPU) checks that this is both
faithful and faster.

**Vector-field temporal context: 18 compressed frames each side (~1.15 s of audio)**, derived from
the block structure and confirmed empirically (perturb one frame; the response reaches exactly
`3 × depth` frames, and nothing beyond).

### Agreement with the one-shot sampler (identical noise and conditioning, 200 compressed frames = 13 blocks)

| variant | latent MSE | cosine vs one-shot | independent draw |
|---|---|---|---|
| `interpolated` (production) | **0.0006** | **0.9998** | 0.8277 |
| `final` context | 0.0018 | 0.9994 | 0.8277 |
| `no_lookahead` | 0.0344 | 0.9894 | 0.8277 |

Audio space: mel L1 0.0016 for blockwise vs 0.0879 for an independent draw; waveform cosine 1.0000.
So blockwise output is **~50–100× closer to the one-shot result than an independent sample from the
same conditioning** — the approximation stays far inside the model's own sampling variability.

**Positive control.** Agreement being "exact" is only meaningful if the metric could detect error.
Opening the layer scales of a copy of the vector field (see the next bullet for why) raises the
response beyond the perturbed frame from ~1e-6 to 0.266, and then `no_lookahead` degrades by **29×**
(0.0577 vs 0.0020) — the measurement is sensitive, and dropping the right-hand context is the
approximation that actually costs quality.

**A finding worth recording: `layer_scale_init=1e-6` makes the vector field a per-frame function at
initialization.** A freshly built VF has a response of ~1e-6 beyond the perturbed frame — every
ConvNeXt branch is scaled to nothing — so early in training blockwise sampling is *exactly* the
one-shot sampler, for a trivial reason. Temporal coupling (and therefore the block approximation
error) only becomes measurable once the branch opens. The demo measures both states rather than
reporting the flattering one.

### Time to first audio (NFE 2, one CPU thread)

| audio length | one-shot | streaming | speed-up | chunks |
|---|---|---|---|---|
| 1.37 s | 124 ms | 112 ms | 1.10× | 2 |
| 5.46 s | 292 ms | 134 ms | 2.18× | 6 |
| **21.85 s** | **979 ms** | **125 ms** | **7.81×** | 22 |

TTFA is ~125 ms almost independently of length (the sampling window is
`context + block + lookahead` frames regardless of text length) while the one-shot sampler grows
linearly. This is the interaction claim that matters for a voice assistant.

### The cost, stated plainly: streaming trades total compute for latency

| block | TTFA vs one-shot | total compute vs one-shot |
|---|---|---|
| 16 frames | **7.28×** | 2.81× |
| 32 | 5.96× | 1.74× |
| 64 | 4.17× | 1.30× |
| 128 | 2.55× | 1.07× |

Each block re-integrates its context and lookahead band, so smaller blocks give lower latency at a
higher total cost. That is a deployment knob, not a free win, and the demo prints both columns.

### Chunked phase-lock filter

Streaming audio still gets the same post-processing: an `n_fft` look-ahead makes the interior of the
stream identical to the offline filter (cosine **0.9994**, max |diff| 6.7e-3, coherence 0.200 vs
0.196). A library bug surfaced here — `torch.stft(center=True)` cannot handle signals shorter than
`n_fft`, which is exactly what a final streamed chunk looks like — so `phase_lock` now pads, filters
and trims.

## 7. The teacher mixture is a mechanism, not a config value (measured)

This section exists because the plumbing was **missing**: `MultiTeacherMixer` was implemented,
exported and unit-tested, but nothing in the cache, the collation or the training stages ever read
it. The objective's central claim — *mix training* of several teachers — was a documented config
value. It is now wired end to end and demonstrated:

```
teacher corpus (manifest: teacher per record)
  -> build_latent_cache(teacher_weights=...)   stores teacher_weight, teacher_index per sample
  -> collate()                                 carries them into the batch
  -> tiny_text_step / flow_step                 pass them to the loss
  -> TextSideDistillLoss / flow_loss            per-sample reduction, weights rescaled to mean 1
```

Two synthetic teachers (**92 Hz** and **215 Hz**) provide the same text; the Tiny text side is
trained from an **identical initialisation**, on identical data, for identical steps, with only the
per-sample mixture weight changing (`scripts/mixture_demo.py`):

| high-teacher share | predicted F0 | low-pitch probe loss (5.94 at init) |
|---|---|---|
| 0.00 | 87 Hz | 2.04 |
| 0.25 | 97 Hz | 2.16 |
| 0.50 | 136 Hz | 2.26 |
| 0.75 | 170 Hz | 2.39 |
| **1.00** | **200 Hz** | **2.55** |

Two independent signals, not one restated: the student's predicted F0 tracks the mixture share
monotonically across a **112.6 Hz span**, *and* the fit on the low-pitch teacher's own samples
degrades in the same order — fitting one teacher costs fit on the other, which is what a weighted
objective is supposed to do.

`tests/test_mixture.py` pins every link in the chain individually: raw shares (not per-item
normalised, which would make every weight 1.0), the quality multiplier and the 0.05 floor,
mean-1 renormalisation being loss-scale invariant, masked per-sample reductions dividing by the
valid count rather than the padded count **including the channel dimension** (getting that wrong
inflated the latent term 24× — caught by `test_pipeline_learns_on_structured_synthetic_speech`
regressing, and now pinned by `test_per_sample_masking_counts_channels`), the weight reaching both
`tiny_text_step` and `flow_step`, weight repetition under `Ke>1` context expansion, and the cache
preserving teacher provenance.

## 8. End-to-end recipe dry run (measured)

Every other demo tests a part. `scripts/recipe_dry_run.py` runs the recipe **in the order the
documentation claims**, offline, with two synthetic fixture teachers (so no network, no API key, no
3B checkpoint):

```
prompts -> synthesize_corpus -> manifest.jsonl -> build_latent_cache -> LatentShardDataset
        -> run_stage("distill-text") -> Synthesizer.synthesize
```

It found **two real bugs in the documented training path that no per-part demo could see**:

* the cache stored **frame-level** F0 and energy while the text side predicts **per token**, so
  `train.py --stage distill-text --cache …` crashed with a 45-token prediction against a 315-frame
  target.  Every demo used the in-memory target builder, which was already correct;
* the cache tokenised **with** BOS/EOS while inference tokenises without, so a model trained from a
  cache would learn two tokens that inference never supplies.

Both are fixed (`aggregate_to_tokens`, `add_special=False`) and pinned by
`test_cache_token_shapes_are_consistent` / `test_cache_ids_match_inference_tokenisation`.

Result (full run, 6 prompts, 120 autoencoder steps, 250 distillation steps, ~4 min):

| check | outcome |
|---|---|
| corpus uses both fixture teachers | PASS (mixture 0.6/0.4, interleaved) |
| autoencoder improved | PASS |
| cache records the mixture | PASS (`teacher_names`, `teacher_weights` in `cache_meta.json`) |
| teacher weights reach the batch | PASS (`[0.6, 0.6, 0.4, 0.6]` in one batch — both values present) |
| distillation improved | PASS (**4.8993 → 1.0418**, 79 % better) |
| synthesis produces audio | PASS (3.32 s, phase coherence 2–8 kHz 0.175) |

The lesson generalises: every bug found in rounds 6–7 — the mixture never reaching the loss, the 24×
mask bug, these two — lived in the *seams between* components, which is exactly what a per-part demo
suite cannot see, and why this end-to-end run earns its four minutes.

## 9. Multi-voice conditioning (measured)

The same class of bug as the mixture, found the same way: `n_voices` and `voice_embed` existed and
the manifest had carried a `voice` per record since round 1, but **nothing between them produced a
voice tensor**, so every sample trained as voice 0. Fixing the plumbing exposed a second, *design*
gap: the voice embedding only modulated the latent feature, leaving the duration/F0/energy heads
voice-blind, so voices could not differ in pitch even once the index arrived. The voice now
conditions the whole text side.

`scripts/voice_demo.py` trains the Tiny text side twice from an identical initialisation — once with
the voice index reaching the model, once with it forced to 0 (the pre-fix behaviour) — on a corpus
where **the same texts are rendered once per fixture voice** (pitch multipliers 0.85 / 1.0 / 1.6):

| | voice 0 (low) | voice 1 (mid) | voice 2 (high) | fit |
|---|---|---|---|---|
| fixture pitch | 80.8 Hz | 95.0 Hz | 152.0 Hz | — |
| **with conditioning** | **81.7 Hz** | **88.8 Hz** | **148.6 Hz** | loss **0.3726** |
| control (voice forced to 0) | 128.0 Hz | 106.9 Hz | 0.0 Hz | loss 2.6021 |

The conditioned student reproduces the fixtures' relative pitch almost exactly (81.7/88.8/148.6 vs
80.8/95/152) and fits the per-voice targets **7× better** than the control, which — given identical
inputs with contradictory targets and no way to disambiguate — drifts to nonsense (0 Hz for the
high voice).

### Three more bugs the voice work surfaced in the F0 *target* pipeline

Pitch is the target, so it had to be right; it was not.

1. **The fixture's pitch sweep was unbounded** — 2 % decline per *token* over a 43-character
   sentence swept down to 0.16× the base pitch (24 Hz for the high voice), below any tracker's
   range. The fallback target then looked scrambled in a way that first appeared to be a modelling
   failure. Fixed to a bounded 15 %.
2. **The default pitch estimator was formant-biased autocorrelation**, which on a full utterance
   reported **168 Hz for an 81 Hz voice** and marked only 52 % of frames voiced (YIN: 74 Hz, 100 %).
   The estimator is now YIN (difference function, CMND, absolute threshold on the first *local
   minimum*, parabolic refinement on the difference function) with a wider analysis window. Two
   implementation details mattered: stopping at the first threshold *crossing* rather than the first
   local minimum reported 226 Hz for a pure 200 Hz tone.
3. **Per-token aggregation averaged in the unvoiced zeros**, so a half-voiced token was labelled
   with half its true pitch. F0 now averages over voiced frames only (a fully unvoiced span stays 0).

All three are pinned by `test_cached_f0_targets_follow_the_voice_pitch` (the cached targets must
increase with the voice's pitch multiplier — it fails if any of the three regresses),
`test_yin_is_used_by_default_and_is_no_worse_than_autocorrelation`, and
`test_aggregate_to_tokens_ignores_unvoiced_zeros`.

## 10. The CLI entry points were bypassing all of it (measured)

Rounds 6-9 wired the mixture, multi-voice conditioning and cross-sample pairing into the *library*.
Round 10 checked the two entry points a user actually runs, and found the features unreachable from
both:

* **`scripts/train.py` built `LatentShardBatchSource` without `pair_references`**, so the documented
  `--stage flow` command trained on each utterance's own mel — no cross-sample pairing whatsoever —
  and with no reference-length cap. The batch-source decision now lives in
  `parakeet.data.dataset.make_batch_source`, which is shared, tested and defaults to pairing for the
  flow stage (`--no-pair-references` opts out, `--max-ref-frames` overrides the 1500-frame cap);
* **`make_teacher_corpus.py --cache-only` called `build_latent_cache` with no mixture at all**, so
  every teacher weight became 1.0 and "mix training" reverted to a decoration on the documented
  path. The cache is now built by `parakeet.data.features.cache_teacher_corpus`, which reads the
  mixture from the corpus's own provenance (`corpus_meta.json`), prefers the curated `kept.jsonl`
  (also in `curated/`) and passes the corpus root as the record base;
* **`curate_manifest` — the entire documented P1 quality pipeline — was dead code.** No entry point
  called it. Curation now runs by default in the corpus builder (`--no-curate` skips it) and in the
  dry run.

### Two real defects the wiring exposed in the P1 gates

Wiring curation is what made them observable:

1. **Digital silence passed the gates.** With the duration and bandwidth gates relaxed, a file with
   `rms = -160 dBFS` was **kept** — there was no absolute level check, and `min_snr_db` was skipped
   because SNR is unevaluable for silence. Fixed with a `min_rms_dbfs` gate ([proposed], -45 dBFS);
   the test asserts silence is rejected *even under otherwise-relaxed gates*.
2. **`silence_ratio` reported 0.0 for digital silence.** The rule is "frame energy below peak − 40
   dB", which is degenerate when the peak is zero, so a silent file scored as *not* silent. Fixed:
   a signal with no discernible peak is 100 % silence.

Both matter because a teacher API that fails silently produces exactly such files, and the curation
stage is the only thing standing between them and the training set.

Also reported deliberately: the **published** thresholds (CosyVoice: ≥ 3 s; ≥ 5 kHz bandwidth)
reject **100 %** of the synthetic fixtures, which are ~0.3 s band-limited stacks. The dry run prints
both the published and the fixture-appropriate results and uses the latter, because the mismatch is
a property of the fixture rather than a defect in the gates — but hiding it would have been a lie
about what the fixture can demonstrate.

`recipe_dry_run.py` now covers the whole documented chain — prompts → synthesis → **curation** →
cache from the curated manifest → training → synthesis — passing 10/10 checks on both the Tiny and
the Small/flow stage.

## 11. The published repository was missing the data package (found by cloning, not inspecting)

The most serious defect in this project so far, and the one no test in the working tree could see.

`.gitignore` carried an **unanchored `data/`** rule, meant for the corpus directory at the repo root.
It also matches the Python package `parakeet/data/`, so for the first ten commits the tokenizer,
teacher backends, feature cache, datasets and curation pipeline were never committed:
**7 of 33 package source files were absent from GitHub**. Everything worked locally, every test
passed locally, `git push` succeeded every time — and a fresh clone could not even
`import parakeet.data`.

It surfaced only because a routine audit tried to recover a deleted helper from git and found the
file was not in git at all. The rule is now root-anchored (`/data/`, `/runs/`, `/checkpoints/`).

Verified the only way that counts — by **cloning the pushed repository** and running it there:

```
$ git clone https://github.com/chamsco/Parakeet.git /tmp/clone
$ cd /tmp/clone && python -c "import parakeet; print(parakeet.__file__)"
/tmp/clone/parakeet/__init__.py
$ python -m pytest tests/test_provenance_and_hygiene.py tests/test_conditioning.py \
      tests/test_pipeline_wiring.py
31 passed
```

`tests/test_provenance_and_hygiene.py` makes this class of failure impossible to repeat silently: it
asserts every `.py` under `parakeet/` is tracked by git (plus scripts, CI config and configs), and it
runs in CI, where the checkout is a real git repository.

### Closing out the "declared but unwired" theme

A systematic audit of every public name and config field (not opportunistic discovery) found the
remainder:

* **`write_run_metadata` was dead code** — no run had ever recorded its provenance. `run_stage` now
  writes `<out_dir>/run.json` with the git revision and dirty flag, a config SHA-256, python/torch
  versions, the stage, and the **trainable/frozen parameter report** (the diagnostic that would have
  caught the round-2 bug where `distill-decoder` silently trained nothing). `train.py` adds the
  corpus provenance: teacher mixture, voices, and a SHA-256 of the cache index.
* **`SpeakerConfig.checkpoint` / `SpeakerConfig.freeze` were never read.** The documented production
  path — frozen CAM++ identity with an adapting Q-Former — did not exist; every run trained the
  randomly-initialised stand-in. Implemented, with a prefix-tolerant loader (raw encoder state dict
  or full-model checkpoint) that raises on zero overlap instead of silently training a random encoder.
* **`use_teacher_durations` was accepted and ignored**; it now selects the teacher's durations when
  the manifest carries `token_frames`, and the uniform fallback otherwise (whose comment claimed
  "energy valleys" while splitting uniformly, and computed an energy vector it never used).
* **`consistency_distillation_loss`** duplicated an inline MSE in `reflow_step`; the named loss is
  now the one used.
* **14 superseded helpers deleted and 7 dead config knobs removed** (listed in the commit). One
  deletion was an error I caught in the same command: `sha1_of_array` *is* used for corpus
  provenance and was restored immediately.
* **`test_no_dead_public_api_in_the_package`** now fails if any public name becomes unreferenced
  again, with an explicit allowlist escape hatch, so the finding cannot silently return.

## 12. Speaker/style conditioning from a latent cache (measured)

The third instance of the same class of bug, this time in the **flagship** path. `collate` dropped
`log_mel` entirely, so the Small/flow model trained from a latent cache received `ref_mel=None`:
a frozen zero speaker embedding, no style tokens, and therefore **no way to learn identity or
style** — the zero-shot capability the whole Small design exists for. Every demo that exercised
conditioning built its own `ref_mel` by hand, so nothing noticed. Two related gaps:

* **Cross-sample pairing was never implemented.** PilotTTS conditions on a *different* utterance of
  the same speaker; training on the target's own mel teaches the conditioner to copy the answer.
  `LatentShardBatchSource(pair_references=True)` now picks a positive reference from the same voice
  group and a negative reference from a different voice group, never the target itself.
* **The style objective had the wrong sign.** The only implemented term was a same-speaker
  *consistency* loss, which pulls two same-speaker style sets together and so invites speaker
  identity to leak into the style channel — the opposite of what pairing is for. The default is now
  a **separation** term against a different speaker's style; the consistency term is kept but off
  (`style_consistency_pair: 0.0`).

`tests/test_conditioning.py` pins each link and, importantly, a **control**: with the pre-fix
behaviour (no reference) the three modules that can only learn identity/style *from a reference*
(the ECAPA speaker encoder, the mel memory encoder, the Q-Former) receive **exactly zero gradient**,
while with references they receive non-zero gradient and the separation term is active.

`scripts/recipe_dry_run.py --stage flow` runs the Small path end to end, offline:

| check | outcome |
|---|---|
| corpus uses both fixture teachers | PASS |
| autoencoder improved | PASS |
| cache records the mixture | PASS |
| teacher weights reach the batch | PASS |
| **references reach the flow stage** | PASS (`ref_mel` padded and masked; negative reference present) |
| student improved | PASS (flow loss 39.76 → 39.03) |
| **style separation is active** | PASS (0.9827) |
| synthesis produces audio | PASS (3.34 s) |
| **reference changes the conditioning** | PASS (conditioning differs by 0.2254 between two references, 1.74 against the null fallback) |

The acoustic comparison is deliberately **not** the control, and the dry run prints why: the flow
estimator's residual branches start at `layer_scale_init = 1e-6`, so an untrained model's output is
dominated by `x0` and two different references give near-identical audio (cosine 0.999980). That is
the same degeneracy found in round 5; measuring it would have been a fake control, so the structural
measurement is the meaningful one at this stage.

## 13. The real teacher path had never been executed — and was wrong (measured)

Every other measurement in this file uses the synthetic fixtures or the stub teachers, because the
real ones need a 3B checkpoint, an 82M model or a paid API. That meant the **only** code that turns a
real teacher's output into training audio had never run a single line, in any demo, test or CI job.
It was wrong.

`OrpheusBackend` maps the 7 SNAC codebooks per super-frame onto the three SNAC levels by grouping
them **contiguously** — `{0}`, `{1, 2}`, `{3, 4, 5, 6}`. The published Orpheus decoder uses
`{0}`, `{1, 4}`, `{2, 3, 5, 6}`. Contiguous grouping passes every shape check, looks entirely
plausible, and decodes to noise: **every corpus built from Orpheus would have been garbage**, and
nothing in the repository could have noticed.

The mapping was verified against two independent copies of the published decoder before changing it
([CrispTTS `decoder.py`](https://github.com/CrispStrobe/CrispTTS/blob/main/decoder.py), which
matches the canopyai Orpheus-TTS decoder — see
[DeepWiki: audio decoding](https://deepwiki.com/canopyai/Orpheus-TTS/2.2-audio-decoding)); both
assign `codes_1 = {i+1, i+4}` and `codes_2 = {i+2, i+3, i+5, i+6}`, flattened frame-major, with
`int32` codes. The implementation now matches, and the codes are cast to `int32` as the reference
does rather than the `int64` this repo happened to produce.

`tests/test_teacher_backends.py` (13 tests) pins the contract of all three real backends with
injected stubs — no weights, no network, no API key, so it runs in CI:

* **Orpheus**: the level mapping is asserted **element-wise against a reimplementation of the
  published algorithm** on a token stream with distinguishable codes, plus level shapes (1/2/4 codes
  per super-frame), partial-super-frame dropping, out-of-range token filtering, the chat wrapper
  tokens, and the sampling settings. A **positive control** asserts that the old contiguous grouping
  *cannot* satisfy the mapping, so the test provably has teeth.
* **Kokoro**: chunks are concatenated (not stacked or dropped), the voice and speed reach the
  pipeline, empty output yields an empty array rather than an exception, and `durations()` passes
  teacher timings through while degrading to `None` when the installed pipeline has no such option.
* **MiniMax** (legally gated, but the code still has to be correct): the request payload, URL and
  `Authorization` header are asserted, the hex-encoded WAV response is decoded to `[-1, 1]` float at
  the right rate, a response with no audio raises with the body in the message, and the backend stays
  refused without an explicit acknowledgement.

The lesson is the same one as round 11, one level up: *unexecuted code is unverified code, and its
looks are not evidence.* The fixtures made ten demos possible and, in doing so, hid the only path
that matters for the real corpus.

## 14. Phase-lock A/B: what the filter does, and what the statistic cannot show (measured)

The phase-locking filter is the only *quality* intervention here that is not a trained model. It
rewrites the phase of synthesized speech between 2 and 8 kHz, leaves the magnitudes alone, and
reportedly moves UTMOS 4.39 → 4.41; it costs ~10 % of the shipped CPU pipeline.
`scripts/phase_lock_ab.py` separates three questions and refuses to blur them.

**First, the test signal had to be fixed.** The obvious choice — the synthetic corpus fixtures — is
invalid: `render_token` assigns every harmonic a **random** phase, so the fixtures score 0.149 on the
phase-concentration statistic against **0.144 for white noise**. The filter targets a glottal pulse
train, whose phase is linear in frequency, so the A/B uses a partially glottal-locked stack (70 %
linear phase + 30 % jitter + a noise floor) alongside white noise as a control.

**Second, the statistic has a floor.** `phase_coherence` maximises over a delay grid, so white noise
scores ~0.14 rather than 0 — and *any* linear-phase imposition raises it. That makes "coherence went
up" a much weaker claim than it looks.

Results (6 signals × 3 s, one CPU thread):

| configuration | coherence | noise (control) | speech−noise gap | mel L1 cost | cost |
|---|---|---|---|---|---|
| baseline | 0.2456 | 0.1435 | +0.1023 | — | — |
| ramp 0.7, n_tau=64 *(old default)* | 0.2514 | 0.1788 | **+0.0727** | 0.007 | 1.5 ms/audio-s |
| ramp 0.7, n_tau=256 | 0.2576 | 0.1643 | +0.0935 | 0.014 | 1.6 ms/audio-s |
| ramp 0.7, n_tau=1024 | 0.2608 | 0.1648 | +0.0962 | 0.014 | 2.3 ms/audio-s |
| ramp 1.0, n_tau=64 | 0.2600 | 0.1945 | +0.0658 | 0.018 | 1.5 ms/audio-s |
| ramp 1.0, n_tau=1024 | 0.2667 | 0.1755 | +0.0915 | 0.023 | 2.4 ms/audio-s |
| smooth 0.7 | 0.2433 | 0.1430 | +0.1003 | 0.001 | 1.3 ms/audio-s |

Three findings, one of them negative:

1. **The delay grid was too coarse, and that was a real defect in the shipped configuration.** The
   filter searches delays τ on a grid of `n_tau` points; resolution bounds the achievable lock
   across the band (64 points over [0, 1/60 s] is 260 µs — more than two periods of phase error at
   8 kHz). At matched strength, 64 → 256 roughly **triples** the coherence added to glottal-locked
   speech (+0.0058 → +0.0120) while **reducing** what it adds to white noise (+0.0353 → +0.0208):
   less of the effect is the filter's own arithmetic. The default is now **256** in both the offline
   and the streaming filter, and a test asserts they stay in sync. Cost: 1.5 → 1.6 ms per audio
   second.
2. **`smooth` mode is inert on this statistic.** It changes within-frame coherence by −0.004…−0.0004
   and temporal coherence by less than 0.002, at a tiny fidelity cost. Either Paradee's variant
   operates on something neither statistic captures, or this implementation of it does not reproduce
   the paper's mechanism. We report the measurement rather than the intention.
3. **The filter adds coherence to white noise too, so the speech-vs-noise gap *narrows*** — from
   +0.1023 to +0.0727 at the old default, and even at the best configuration it only returns to
   +0.0915. **The within-frame concentration statistic therefore cannot demonstrate speech-specific
   locking**; it measures that a linear phase was imposed, not that the signal had one. `buzz` in
   `eval/metrics.py` is `1 − coherence` and inherits exactly this weakness.

What is deliberately **not** claimed: that any of this sounds better. UTMOS is unavailable in this
environment, so the paper's 4.39 → 4.41 stays a citation. The honest summary is that the filter
provably imposes the linear phase it is designed to impose, that its resolution was under-configured
until now, and that its *benefit* is unmeasured.

## 15. Checkpoint/resume: a resumed run continues, byte for byte (measured)

`train.py --resume` used to restore **only the model**. The optimizer moments, the EMA (which is the
Reflow teacher *and* the better-quality final weights), the discriminator, the LR schedule position,
the RNG and the batch order were all discarded — so a resumed run silently restarted the cosine
schedule from its warmup peak and reshuffled its data. Nothing in the repo could notice: the loss
after a resume looks plausible either way. On a 50k-step GPU run you would find out as worse final
quality.

Now `save_checkpoint` writes model + optimizer + EMA + discriminator + step + RNG (torch, python,
numpy) + the batch source's order/RNG state, `load_checkpoint` restores all of it, and
`run_stage(resume_from=...)` repositions the LR schedule. `train.py --resume` passes it through to
`run_stage` instead of loading the model itself.

Verified on the **real** data path (fixture corpus → latent cache → `LatentShardBatchSource`), with a
simulated crash rather than a clean stop (`scripts/resume_demo.py`, 120 steps, ~3 min):

| quantity | difference between an uninterrupted run and a crashed-then-resumed run |
|---|---|
| final parameters | **0.000e+00** |
| loss at each shared step | **0.000e+00** |
| learning rate at each shared step | **0.000e+00** |
| loss across the interruption | 0.6337 → 0.6337 (no jump) |

The checkpoint contains `config`, `ema`, `extra`, `optimizer`, `rng`, `step` (and the model), and the
CLI path was checked separately: `train.py --dry-run --steps 2` then `--resume … --steps 4` prints
`resumed from … at step 2 (lr 6.000e-07)` and writes a `run.json` recording the stage, git revision
and config hash.

**One semantic caveat, found by this demo's own control.** The cosine schedule is parameterised by
the *total* step budget, so resuming with a different `--steps` gives a different schedule — and a
different trajectory — by construction. The demo's first version compared a 40-step run against a
20-step run and "detected" a divergence of 8.8e-3 that was entirely its own doing. A control
(`pre_interruption_runs_match`) now isolates that, so a real resume regression cannot hide behind it.

## 16. Smoke test output (measured)

```
parakeet-tiny [tiny] sr=24000 mel=80@93.8Hz latent=24 compress=1/6 voice=constant
  autoencoder            5.033 M
  text                   3.851 M
  duration               0.657 M
  latent_head            0.072 M
  prosody_proj           0.002 M
  TOTAL                  9.616 M
  fp32 38.47 MB | fp16 19.24 MB | int8+fp16 scales 12.02 MB
  [autoencoder    ] loss=17.1408  2.36s  mel=3.0956 spectral=4.6580 spec_weight=3.0000
                                          spec_sc=0.9669 spec_log_mag=3.6912 adv=0.0136
                                          feat_match=0.0111 phase_lock=0.7048 disc=1.9993
  [distill-text   ] loss=5.4429   0.23s  duration=1.2847 f0=1.0165 energy=1.0455 latent=2.0962
  [distill-decoder] loss=212.6493 1.33s  mel=3.0926 spectral=4.6552 spec_weight=45.0000
  audio 13568 samples = 0.565s @ 24000 Hz
  phase coherence 2-8k: 0.1414 -> 0.1820 (higher = less buzz)
  RTF(1 thread) 0.044  (22.62x real time)
  streaming vs offline decoder max|diff| = 5.59e-09
  int8 weights -> 12.02 MB (fp32 38.47 MB)

parakeet-small [small] sr=24000 mel=80@93.8Hz latent=24 compress=1/6 voice=reference
  autoencoder 14.114 M | text 5.431 M | speaker 6.076 M | cond_proj 0.066 M
  vf 19.154 M | length_predictor 0.197 M | TOTAL 45.038 M
  fp32 180.17 MB | fp16 90.09 MB | int8+fp16 scales 56.31 MB
  [autoencoder] loss=18.1232 1.98s
  [flow       ] loss=19.7684 1.36s  flow=2.4362 length=17.3322
  [reflow     ] loss=2.0511  0.97s  reflow=2.0511 x1_std=1.1516
  streaming vs offline decoder max|diff| = 0.00e+00
  int8 weights -> 56.31 MB (fp32 180.17 MB)

SMOKE TEST PASSED
```

### What these numbers do and do not mean

**Do mean:** the architecture is the size we claim, every training objective is numerically
well-behaved and differentiable, the streaming path is exact, int8 saves ~3×, and the Tiny
architecture generates 0.565 s of audio in 25–30 ms of single-thread CPU time **in fp32** —
0.044–0.052 RTF, i.e. **19–23× real time** across runs on a busy desktop CPU, which is Paradee's
reported ballpark (25.0× for a trained 8.07 M model, 17.8× via ONNX). Loss weights of 3.0 and 45.0
in the log are the spectral anneal working as designed.

**Do not mean:** any quality claim. The models are **randomly initialised**; the loss values are
one-step values on random data and will be different (and meaningful) once trained. Small's RTF
number is not reported above because its output was 0.04 s long and therefore dominated by fixed
overhead — it is not a valid throughput measurement until the model predicts sane durations.  The
learning demo (§2) trains properly but on 15 seconds of *synthetic* audio, so it demonstrates that
the machinery learns, not that the model is good.

## 17. Deliberate engineering checks worth calling out

* **Streaming == offline, bit-for-bit (5.6e-09).** Getting this right required a specific fix:
  prefilling the latent with zeros is *not* equivalent to the offline path, because offline zero
  padding happens at every causal convolution's input, whereas a latent prefill only feeds the
  first layer — and deeper layers respond non-zero to a zero input (biases, layer norm, GELU).
  `StreamingVocoder` therefore caches each block's own input history
  (`ConvNeXtBlock.forward_with_history`, `CausalConv1d.forward_no_pad`). This is the kind of bug
  that would otherwise only show up as "the first half-second of streamed audio sounds slightly
  different".
* **Phase lock cannot be exactly re-analysable.** A phase-modified STFT is generally an
  *inconsistent* representation, so re-analysing the output changes magnitudes somewhat even
  though only phase was edited. The test therefore asserts loudness (<5 %) and spectral-envelope
  (cosine >0.99) preservation, and the *purpose* — increased phase coherence on buzzy speech — is
  tested separately. An earlier version of this test asserting exact magnitude parity was simply
  wrong and was replaced.
* **Licence gating is tested, not just documented.** `test_minimax_is_refused_by_default` fails if
  someone removes the gate.
* **A stage was silently training nothing (found by designing the learning test).** `run_stage`
  applied each stage's freeze pattern *on top of* whatever the previous stage left frozen. Since
  `distill-text` freezes the whole autoencoder, a following `distill-decoder` run — the stage whose
  entire job is training the decoder — inherited a frozen decoder and reported a finite loss while
  changing no weights. `run_stage` now resets trainability per stage, and
  `test_stage_freezing_does_not_leak_between_stages` asserts the decoder weights actually change.
* **A "four-term" loss was really a one-term loss (found by the learning test).** Regressing F0 as
  quantised bin indices (0–256) and energy in raw dBFS made those two terms ~140 of the 141 total,
  so the duration and latent heads got almost no gradient. Both prosody targets are now normalised
  to `[0, 1]` (`f0_to_normalized`, `energy_to_normalized`), the loss starts at ~3 and falls by more
  than half in 40 CPU steps. Paradee regresses `F0/100` for the same reason.
* **SNR is reported as unevaluable when it is.** The percentile SNR estimate is meaningless on
  continuous speech with no pauses; `estimate_snr_db` returns `None` and the gate is skipped rather
  than rejecting good continuous speech on a bogus 0 dB reading.
* **A `(B, 1, N)` waveform is accepted everywhere.** Two callers produced that shape in one round;
  `MelSpectrogram.stft` now squeezes the singleton channel and raises a clear error for anything
  else, rather than surfacing a cryptic `torch.stft` message.

## 18. Environment notes

* CPU torch was installed from the PyTorch CPU index (no CUDA on this machine), in a dedicated
  Python 3.13 venv; the system Python 3.14 also has torch wheels available (2.14.1).
* `soundfile` is required for corpus I/O; `transformers`/`snac`/`kokoro` (teachers),
  `faster-whisper`, `funasr`, `onnxruntime` are optional and every code path that needs them
  reports unavailability instead of faking a metric.
* The repo's `runs/`, `data/`, `*.pt`, `*.wav` outputs are git-ignored; nothing large is committed.
