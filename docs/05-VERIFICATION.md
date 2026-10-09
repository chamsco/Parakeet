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
python -m pytest -q                                         # 272 tests
python scripts/bench_rtf.py --config configs/parakeet_tiny.yaml
```

## 1. Test suite

```
272 passed
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
| `test_speechify_backend.py` | the hosted teacher against a **stubbed transport** (a suite that spends the operator's characters is a bug in itself): request shape, base64/WAV decoding and int16 scaling, character accounting, retry on a transient failure, **no retry on a 401**, the licence distinction in the spec, and that the key never reaches a corpus file.  `test_mix_corpora.py` covers the mixture: relative paths that resolve without copying, interleaving so no shard is one teacher, preserved `token_frames`, and weights that default to the utterance share.  `test_provenance_and_hygiene.py` gained the last line of defence — **no key-shaped string in any tracked file** |
| `test_train_stages.py` (extended once more) | `distill-audio` really trains **through** the decoder: the text-side heads and `prosody_proj` move, the autoencoder does not, and the logs carry the audio objective and the auxiliary term |
| `test_speechify_backend.py` | the hosted teacher against a **stubbed transport** (a suite that spends the operator's characters is a bug in itself): request shape, base64/WAV decoding and int16 scaling, character accounting, retry on a transient failure, **no retry on a 401**, the licence distinction in the spec, and that the key never reaches a corpus file.  `test_mix_corpora.py` covers the mixture: relative paths that resolve without copying, interleaving so no shard is one teacher, preserved `token_frames`, and weights that default to the utterance share.  `test_provenance_and_hygiene.py` gained the last line of defence — **no key-shaped string in any tracked file** |
| `test_train_stages.py` (extended once more) | `distill-audio` really trains **through** the decoder: the text-side heads and `prosody_proj` move, the autoencoder does not, and the logs carry the audio objective and the auxiliary term |
| `test_train_stages.py` (extended, again) | the decoder stage really consumes the **token-expanded** distribution when the flag is on (and the log records which was used, so a silent revert is visible); and `--set`/warm-start/frame-latent paths still behave |
| `test_train_stages.py` (extended) | a reconstruction-only phase really is reconstruction-only (`adversarial` weight 0 ⇒ no `disc` loss and no discriminator step, while the generator still trains) — that option is 24× cheaper per step and is what fixed the autoencoder; and a non-finite loss is **skipped, counted, reported with its first step, written to `divergence.json`, and leaves the parameters finite**, so divergence is a finding instead of an all-NaN report |
| `test_duration_scale.py` (extended) | the sub-token geometry is **shared**: a sub-vector index must land in the frame span its target was averaged over, so targets and expansion cannot drift apart; degenerate tokens (fewer frames than the rate) stay in bounds; and `latent_rate` widens the head without changing the frame count |
| `test_duration_scale.py` (extended) | the sub-token geometry is **shared**: a sub-vector index must land in the frame span its target was averaged over, so targets and expansion cannot drift apart; degenerate tokens (fewer frames than the rate) stay in bounds; and `latent_rate` widens the head without changing the frame count |
| `test_duration_scale.py` | durations round-trip through their normalisation exactly; zero on the normalised scale means the measured corpus mean (6 frames), not 1; a one-standard-deviation head output stretches the sequence by ~1.48× — not the ~2.7× that `log_duration.exp()` would give, so the trap is caught even though both produce audio; the loss term is ~zero against the normalised target and >0.5 against a raw-log one; and the diagnosis evidence carries its validity checks and a named bottleneck |
| `test_real_audio.py` | the voicing threshold admits more frames as it rises **while the pitch estimate stays put** (the real-speech signature of a threshold that was too strict); an unvoiced token inherits the nearest voiced pitch instead of 0 Hz; the unaligned split neither collapses nor ignores the energy; the Kokoro runtime dispatch lands on an importable backend and rejects an unknown voice with the available list; the DNSMOS wrapper reads `ovrl_mos` rather than defaulting to 0.0 (the bug the teacher control caught); the real-audio, real-training and real-evaluation evidence all record their provenance, their controls and their caveats |
| `test_speechify_backend.py` | the hosted teacher against a **stubbed transport** (a suite that spends the operator's characters is a bug in itself): request shape, base64/WAV decoding and int16 scaling, character accounting, retry on a transient failure, **no retry on a 401**, the licence distinction in the spec, and that the key never reaches a corpus file.  `test_mix_corpora.py` covers the mixture: relative paths that resolve without copying, interleaving so no shard is one teacher, preserved `token_frames`, and weights that default to the utterance share.  `test_provenance_and_hygiene.py` gained the last line of defence — **no key-shaped string in any tracked file** |
| `test_train_stages.py` (extended once more) | `distill-audio` really trains **through** the decoder: the text-side heads and `prosody_proj` move, the autoencoder does not, and the logs carry the audio objective and the auxiliary term |
| `test_speechify_backend.py` | the hosted teacher against a **stubbed transport** (a suite that spends the operator's characters is a bug in itself): request shape, base64/WAV decoding and int16 scaling, character accounting, retry on a transient failure, **no retry on a 401**, the licence distinction in the spec, and that the key never reaches a corpus file.  `test_mix_corpora.py` covers the mixture: relative paths that resolve without copying, interleaving so no shard is one teacher, preserved `token_frames`, and weights that default to the utterance share.  `test_provenance_and_hygiene.py` gained the last line of defence — **no key-shaped string in any tracked file** |
| `test_train_stages.py` (extended once more) | `distill-audio` really trains **through** the decoder: the text-side heads and `prosody_proj` move, the autoencoder does not, and the logs carry the audio objective and the auxiliary term |
| `test_train_stages.py` (extended, again) | the decoder stage really consumes the **token-expanded** distribution when the flag is on (and the log records which was used, so a silent revert is visible); and `--set`/warm-start/frame-latent paths still behave |
| `test_train_stages.py` (extended) | a reconstruction-only phase really is reconstruction-only (`adversarial` weight 0 ⇒ no `disc` loss and no discriminator step, while the generator still trains) — that option is 24× cheaper per step and is what fixed the autoencoder; and a non-finite loss is **skipped, counted, reported with its first step, written to `divergence.json`, and leaves the parameters finite**, so divergence is a finding instead of an all-NaN report |
| `test_duration_scale.py` (extended) | the sub-token geometry is **shared**: a sub-vector index must land in the frame span its target was averaged over, so targets and expansion cannot drift apart; degenerate tokens (fewer frames than the rate) stay in bounds; and `latent_rate` widens the head without changing the frame count |
| `test_duration_scale.py` (extended) | the sub-token geometry is **shared**: a sub-vector index must land in the frame span its target was averaged over, so targets and expansion cannot drift apart; degenerate tokens (fewer frames than the rate) stay in bounds; and `latent_rate` widens the head without changing the frame count |
| `test_duration_scale.py` | durations round-trip through their normalisation exactly; zero on the normalised scale means the measured corpus mean (6 frames), not 1; a one-standard-deviation head output stretches the sequence by ~1.48× — not the ~2.7× that `log_duration.exp()` would give, so the trap is caught even though both produce audio; the loss term is ~zero against the normalised target and >0.5 against a raw-log one; and the diagnosis evidence carries its validity checks and a named bottleneck |
| `test_real_audio.py` | the voicing threshold admits more frames as it rises **while the pitch estimate stays put** (the real-speech signature of a threshold that was too strict); an unvoiced token inherits the nearest voiced pitch instead of 0 Hz; the unaligned split neither collapses nor ignores the energy; the Kokoro runtime dispatch lands on an importable backend and rejects an unknown voice with the available list; the real-audio evidence records its provenance and its duration caveat |
| `test_ablation.py` | the offered lite config really is smaller and really trains and synthesizes; it **matches the geometry the ablation recommended** (read from the committed evidence, so it holds in a clone); and the evidence records its own limitations (a caveat, a held-out split of ≥ 3 items, both train and val fits) — a study without its caveat is an overclaim |
| `test_model_card.py` | every claim cites **committed** evidence whose SHA-256 matches the manifest; a missing report comes back `unmeasured` and a corrupted value comes back `fail` (positive controls on the check itself); the rendered card states its limitations (no checkpoint, out-of-scope uses, no UTMOS) and its licence table is generated from the teacher specs |
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

## 16. The model card is generated from evidence, and fails without it (measured)

A model card is usually prose that drifts: numbers copied from an old run, a limitation quietly
dropped, a claim whose measurement stopped being produced. `scripts/model_card.py` removes the prose
from the loop — `docs/MODEL_CARD.md` is **generated**, and every advertised number carries the report
file and JSON path it came from.

| feature | how it works |
|---|---|
| every claim is tied to evidence | 15 claims, each with a report path and a JSON field |
| missing evidence is visible | a claim with no report renders as **unmeasured**, never deleted |
| bad numbers are visible | a measured value outside its threshold renders as **FAILING**, named |
| a fresh clone can verify | the reports live in gitignored `runs/`, so the cited evidence is **committed** in `docs/evidence/` with a SHA-256 manifest |
| the card cannot rot | `--refresh-evidence` regenerates the bundle from fresh runs; CI re-runs every demo, refreshes, and fails on `git diff --exit-code` |

Verified: **15/15 claims pass**, each citing `docs/evidence/<claim>.json`; a simulated clone
containing only `docs/`, `parakeet/` and `configs/` (no `runs/`) still verifies all 15 from the
committed bundle; and two **positive controls** prove the check has teeth — against an empty report
root every claim comes back `unmeasured`, and a deliberately corrupted value comes back `fail`.

The card also states what the tables cannot: no trained checkpoint exists, the fixtures are not
speech, UTMOS is unavailable (so naturalness remains a citation), and the teacher/licence table is
generated from `parakeet.data.teacher.TEACHERS` — the same specs the licence gate uses — rather than
being retyped. MiniMax appears there as **restricted and non-trainable**, and the parameter counts
(9.616 M / 45.038 M) are computed by building the models, not copied.

CI runs the whole evidence chain as a `workflow_dispatch` job: every demo, then
`--refresh-evidence`, `--check` and a diff against the committed card. If a demo stops writing its
report or a measurement moves, the job fails — so "every number in the card is reproducible from the
commands printed in the card" is a gate, not a promise.

## 17. The python/dispatch overhead: what was actually removable (measured)

The profile had been reporting ~12-15 % of the shipped latency as unaccounted "python/dispatch", so
this round profiled the real path with `cProfile` instead of guessing. The finding was specific:
`phase_lock` rebuilt its **per-call constants** — the Hann window, the band frequency grid, the delay
grid, and a `(n_tau, band_bins)` complex search matrix built with `torch.exp` — on every call, and
`torch.exp`/`torch.polar` on those grids showed up as ~0.4 ms per call of pure setup. They do not
depend on the audio at all.

Fixed by caching them per `(device, dtype, geometry, method)`, with a bounded cache:

| measurement | result |
|---|---|
| grid construction, cold | **0.386 ms** |
| cached lookup | **0.4 µs** |
| fixed saving per synthesis call (audio-independent) | **~0.39 ms** |
| controlled interleaved A/B, full int8 pipeline | 7.101 → **6.458 ms (-9.1 %)** |
| shipped int8 pipeline, benchmark report | 8.11 → **7.25 ms (70.6 → 74.0× real time)** |
| phase-lock output with and without the cache | **bit-identical** |

Two notes on honesty here. First, my initial estimate was "12 %", derived from reasoning about the
profile rather than measuring the cache — the controlled A/B then said 9.1 % and the direct
measurement of the grid cost said 0.386 ms/call (~5 %). The three numbers bracket the truth, and all
three are reported rather than the most flattering one. Second, the cache key originally used
`str(device)` and `str(dtype)`, i.e. it allocated two strings per call inside the very code meant to
remove overhead; `torch.device` and `torch.dtype` are hashable as-is.

The rest of the unaccounted time is **not** removable python: it is ONNX Runtime's `run` call
(2.0 ms/call for both sessions, including the compute itself). What remains above that is
`torch.polar` (0.4 ms), the iSTFT overlap-add (0.3 ms) and the latent build (0.1 ms) — all real work.

### A reproducibility defect found while regenerating the benchmark

The regenerated report showed `waveform cosine 0.9148` where the README cited 0.992–0.9998. The
cause was not a regression: `export_onnx.py --pipeline` built an **unseeded** random model, so the
int8-vs-PyTorch fidelity figure moved from run to run and the cited number could not be reproduced.
`scripts/export_onnx.py` now seeds before `build_model` (a checkpoint makes the seed irrelevant), and
the seeded report reads `mel L1 0.0174, cosine 0.9981`.

## 18. Capacity ablation: how light can the student be? (measured, fixture-derived)

"Make it as light as possible" is the objective and the text side is the largest component of
Parakeet-Tiny, so `scripts/ablate.py` measures the frontier instead of guessing. Seven text-side
geometries train for the **same 400 steps**, from the **same seed**, on the **same cached teacher
signals**, and are scored on a **held-out split** (12 train / 6 validation items):

| text dim / layers | text params | total params | held-out fit | train fit | text latency | 1-thread synth |
|---|---|---|---|---|---|---|
| **dim256-L4 (shipped)** | 3.851 M | 9.616 M | **0.3860** | 0.1509 | 20.9 ms | 53.5 ms |
| dim256-L2 | 2.272 M | 8.037 M | 0.4440 | 0.1500 | 13.5 ms | 51.3 ms |
| dim160-L3 | 1.238 M | 6.560 M | 0.4596 | 0.1439 | 8.5 ms | 47.9 ms |
| **dim128-L4** | **1.008 M** | **6.228 M** | **0.3949** (+2.3 %) | 0.1580 | **7.9 ms** | 47.9 ms |
| dim128-L2 | 0.612 M | 5.831 M | 0.4480 | 0.1522 | 4.6 ms | 47.1 ms |
| dim96-L2 | 0.360 M | 5.500 M | 0.3976 (+3.0 %) | 0.2037 | 3.6 ms | 45.8 ms |
| dim64-L2 | 0.175 M | 5.257 M | 0.4452 | 0.2151 | 2.4 ms | 46.1 ms |

Three things this shows, and one it deliberately does not:

1. **The shipped text side looks ~4× oversized.** `dim128-L4` gives up 2.3 % of held-out fit for
   **26 % of the text parameters**, **35 % fewer total parameters** (9.616 → 6.228 M) and 2.6× the
   text-side speed. `dim96-L2` gets within 3.0 % on an 11× smaller text side.
2. **Depth beats width at matched budget** (dim128-L4 0.3949 vs dim128-L2 0.4480; dim256-L4 0.3860 vs
   dim256-L2 0.4440), while the smallest variants plateau — 0.175 M, 0.360 M and 0.612 M are
   indistinguishable from each other, which is also the noise floor of a 6-item validation split.
3. **The held-out split changed the answer.** Scored on the *training* items, `dim128-L2` looked best
   (0.0823 vs the shipped 0.0805) and `dim128-L4` looked worst (0.1249). On held-out items the order
   inverts. The first version of this script had no split and would have recommended a configuration
   that is the worst of the seven on unseen data — a reminder that "fits as well" and "generalises as
   well" are different claims. `LatentShardDataset` gained an `indices` argument for exactly this.
4. **Not shown: that the lite geometry is right for real speech.** The fixtures are repetitive
   synthetic stacks, the autoencoder is untrained, and 400 steps is a short budget. So
   `configs/parakeet_tiny_lite.yaml` is offered as a **documented alternative** — with the table above
   in its header — not as the new default. A real corpus plausibly needs more capacity; the point is
   that the current geometry is not *justified* by measurement, and there is now a measured candidate
   to compare against.

A side effect worth noting: the `n_voices` guard added in round 8 rejected this script twice while it
was being developed (a 3-voice corpus with a 2-voice config), i.e. the cheap check keeps paying.

## 19. Real teacher speech, at last — and the assumption that blocked it was false (measured)

For sixteen rounds every measurement here used synthetic fixtures, on the stated basis that the
teachers needed a GPU and 6 GB of weights.  Round 17 tested that assumption instead of repeating it:
**PyPI and HuggingFace were both reachable all along**, and **Kokoro-82M is Apache-2.0 and runs
faster than real time on this CPU**.

The official `kokoro` pip package still cannot install here — it needs `misaki[en]` → `spacy` →
`blis`, which has no wheel for this Python and no Rust toolchain to build from source.  But
`sherpa-onnx` ships prebuilt wheels, bundles its own espeak-ng G2P, and runs the *same* weights, so
`SherpaKokoroBackend` was added: same teacher, same licence, different runtime.  A name outside the
bundle raises with the available list rather than synthesising with the wrong speaker.

`scripts/real_corpus_demo.py` then ran the documented chain on real 24 kHz speech — teacher → corpus
→ curation → latent cache — and **real speech immediately found three things that fixtures could
not**:

1. **The pitch tracker's voicing threshold was far too strict for speech.** At the old 0.25,
   real utterances were only 0.42 voiced; at 0.50 they are 0.63, and the median F0 does not move
   (93–198 Hz either way), which is the signature of admitting genuinely-voiced frames rather than
   noise.  Above ~0.70 the median starts drifting (142 Hz on one voice), so the useful range is
   0.45–0.55.  The threshold is now per method: 0.50 for YIN's CMND (its local-minimum requirement
   makes it stricter than the paper's 0.10–0.15) and 0.30 for the legacy autocorrelation peak.
   Fixtures could never show this — a formant stack is periodic everywhere, so every threshold
   looked fine.
2. **The unaligned duration fallback produced 0 Hz pitch targets.** With no aligner the frames were
   split *uniformly*, which on real speech puts token spans entirely inside pauses; measured, whole
   utterances came back with per-token F0 of 0 Hz (60 Hz in normalised space — the lowest pitch in
   the range, taught for letters next to a pause).  The fallback is now an energy-weighted split
   **blended with uniform** (pure equal-energy collapses on a mostly-silent signal — measured
   `[56, 1, 1, 1, …]`), and a token whose span is entirely unvoiced inherits the nearest voiced
   token's pitch instead of 0.  Per-token targets went from **0–186 Hz** to **100–200 Hz**.
3. **The published curation gates, run on real speech for the first time.** Kept **9/16**: three
   rejected `too_short` (the CosyVoice ≥ 3 s rule, which real short prompts fail) and four
   `narrowband` (bandwidth99 < 5 kHz).  Kept-set statistics are now real: rms −25 dB,
   bandwidth99 **5617 Hz**, and SNR **45 dB** — evaluable at all, where on fixtures the estimator
   correctly refused with `snr_unevaluable`.

All ten checks pass, and the evidence is committed as `docs/evidence/real_audio.json` with the
teacher's licence, runtime, per-utterance pitch statistics and the explicit duration caveat.  The
corpus itself is not committed (`data/` is ignored; the bundle is 320 MB).

**What this does not show:** the model quality.  No autoencoder training happens in this script, the
corpus is 16 prompts, and the duration targets are the unaligned fallback.  What it establishes is
that the data path works on real audio — and that the fixtures had been hiding three real defects
precisely because they are not speech.

## 20. Training and evaluating on real speech — the first real numbers (measured)

Round 17 got real audio through the data path. This round **trains on it** and measures what the
papers measure. Corpus: 21 curated Kokoro utterances, **74.7 s** of real 24 kHz speech.

| stage | before | after | change |
|---|---|---|---|
| autoencoder, real speech (300 steps, 24 min) | recon log-mel L1 2.0121 | **1.3864** | **−31.1 %** |
| text side, real teacher signals (400 steps, 18 s) | teacher-signal loss 3.9990 | **0.8060** | **−79.8 %** |
| text → audio vs the real reference (21 utts) | — | log-mel cosine **0.9404** | — |

Then `scripts/real_eval.py` measures the trained checkpoint with a naturalness metric, an ASR, and —
critically — **the teacher as a control**:

| metric | student | teacher (Kokoro) | control's purpose |
|---|---|---|---|
| DNSMOS P.835 overall | **1.77** | **2.61** | a naturalness metric that does not separate them is broken |
| WER (`faster-whisper base.en`) | **1.00** | **0.00** | the recogniser must transcribe the teacher's audio correctly |
| RTF, one thread | **0.019 (52× real time)** | — | the shipped path on real speech |

**The honest reading: the student is not yet intelligible.** WER 1.00 means `base.en` could not
recover a single intended word from the generated audio, and DNSMOS 1.77 against a 2.61 teacher is a
wide gap. This is what a 75-second corpus and a 25-minute CPU budget buy, and it is the first time
this project could say so with real numbers instead of proxies. The controls are what make those
numbers trustworthy: the ASR scores the *same text spoken by Kokoro* at 0.00 WER, and the
naturalness metric rates Kokoro well above the student.

Two things had to be fixed to get here, both found by the controls rather than by inspection:

* **`utmos` is not installable here** — it needs `fairseq`, whose sdist does not build on this
  Python. `SpeechMOS` (DNSMOS P.835) installs and is the metric the curation gate already references
  (`min_dnsmos` 3.5, PilotTTS), so UTMOS is reported **unavailable with its reason** and DNSMOS
  carries the naturalness evidence.
* **The DNSMOS wrapper read the wrong key names.** `dnsmos.run` returns `ovrl_mos`, `sig_mos`,
  `bak_mos`, `p808_mos`; the wrapper looked for `mos_ovrl` and defaulted to 0.0 — scoring *real Kokoro
  speech* at zero. Only the teacher control exposed it, because a metric that rates a real teacher at
  zero is obviously broken. Fixed, pinned by a monkeypatched regression test, and the sub-scores now
  travel in the metric's `detail` so a low number can be diagnosed.

Not claimed: that this is a usable model. The corpus is one voice set, the duration targets are the
unaligned fallback, and the WER recogniser is `base.en` rather than the papers' `large-v3`, which
makes the WER an upper bound. What changed is that these are now *measurements*.

## 21. Where the text → audio chain breaks: the autoencoder, not the seam (measured)

Round 18 ended with WER 1.00 and no idea why. "The model is bad" is not a diagnosis, so
`scripts/real_diagnose.py` walks the chain with **teacher inputs** at every seam and scores each path
with the same recogniser:

| path | what it isolates | WER | DNSMOS | waveform cosine vs reference |
|---|---|---|---|---|
| reference (Kokoro) | the ceiling | **0.000** | 2.61 | 1.000 |
| 1. autoencoder round-trip | encode real audio → decode it | **1.000** | 1.24 | **+0.000** |
| 2. teacher frame latent | cached latent → decoder | 1.000 | 1.24 | +0.000 |
| 3. teacher token expanded | per-token latents expanded exactly as inference does, with the teacher's own durations/F0/energy | 1.000 | 1.35 | +0.001 |
| 4. student | the real thing | 1.000 | 1.32 | +0.001 |

**The autoencoder is the bottleneck and nothing downstream can exceed it.** Its output is
*uncorrelated* with its input: waveform cosine **+0.000**, SNR **−0.10 dB**, while the mel distance
looks merely poor (log-mel L1 1.83). That combination is the important part — the mel proxy this
project used for sixteen rounds cannot distinguish "slightly degraded" from "unrelated audio", which
is why WER and the waveform-level fidelity measurement were needed. 300 steps at 4.8 s/step bought a
31 % mel improvement and an autoencoder that still produces something else.

The **seam is fine**, which is worth stating because it was the hypothesis: expanding per-token
latents exactly as inference does costs 0.005 of mel cosine against the frame-level path, and all
three teacher-input paths score the same as the round-trip. None of the text-side machinery is
implicated.

### The one real defect the diagnosis did find: durations were regressed in raw log space

Durations were the **only** prosody target still un-normalised, while F0 and energy had been mapped to
O(1) since round 2 — the same fix had simply never been applied to the third signal. The measured
consequence was severe: with targets near log(6) = 1.79 and a head initialised near zero, the
mean-absolute-error gradient on the head's *weights* is divided by the token count, so after 400
steps the head had learned only a constant **1.75 frames per token** — a **0.29× duration collapse** —
and that single term was 1.2 of the 2.6 total loss.

Normalised to the measured corpus statistics (log-duration mean 1.728, std 0.390 over 1141 real
tokens), the same head, same data:

| | before | after |
|---|---|---|
| predicted total frames vs reference | 0.29× | **0.94×** (298 vs 317) |
| text-side teacher-signal loss | 0.806 | **0.505** |

That fix introduces a trap worth a regression test: **every** consumer of `log_duration` must
de-normalise, and calling `.exp()` on the normalised output is off by `exp(1.728) ≈ 5.6`. My own
diagnostic script and `profile_pipeline.py` had that bug within minutes of the change, which is why
`normalized_to_durations()` exists as the single conversion point and why a test checks the *length
ratio* that a one-standard-deviation change must produce.

A reproducibility note, because it should not be quietly dropped: an earlier run of the same
diagnostic printed DNSMOS 1.88 for the autoencoder round-trip where two subsequent runs print 1.24.
The AE weights in both checkpoints are identical to 1.3e−05, DNSMOS is deterministic on a fixed input
(three repeats in-process, plus 1.241 from re-scoring the saved wavs independently), and the two later
runs agree to three decimals. The intervening state could not be reconstructed, so the numbers above
are from the reproducible runs — and the diagnostic now reports its own validity checks separately
from system health, so a future drift is visible instead of being averaged away.

## 22. Fixing the autoencoder, and the next bottleneck it exposed (measured)

Round 19's diagnosis said the autoencoder was the bottleneck; this round asked what the budget was
being spent on. `scripts/ae_ablation.py` runs the same step from the same seed with and without the
MPD/MSD discriminator: **5.25 s versus 0.22 s per step — a factor of 24**. The shipped recipe spent
300 steps on the sharpening term from a random initialisation, i.e. it bought discriminator progress
before the autoencoder could encode anything. Codec training normally does the opposite, so
`scripts/ae_train.py` runs two phases through `run_stage` (keeping warmup, EMA, schedule, provenance
and resume):

| | shipped recipe | phased (2000 recon + 200 adversarial) |
|---|---|---|
| mel L1 | 1.3864 | **0.5058** |
| **autoencoder round-trip WER** | **1.000** | **0.153** |
| teacher WER (the control) | 0.000 | 0.000 |
| DNSMOS | 1.24 | 1.65 |
| divergent steps | — | **0** |

The autoencoder round-trip is now **intelligible** (WER 0.153 against a 0.000 teacher) where round 19
measured 1.000. Time: 445 s of reconstruction (0.22 s/step) plus 966 s of adversarial (4.83 s/step).
The text side was then retrained on the improved autoencoder: reconstruction L1 2.0139 → **0.4987**.

Two things went wrong on the way and both are now guarded:

* **The first attempt diverged to NaN.** It hand-rolled its own training loop and dropped the warmup
  schedule, going mel 2.06 → 0.50 and then to NaN between steps 1200 and 1800 — with no visible
  symptom until an all-NaN report at the end. That is the "reimplemented the helper" mistake again
  (the same one as calling `build_latent_cache` instead of `cache_teacher_corpus`), so the trainer now
  uses `run_stage`, and **`run_stage` itself gained a non-finite guard**: a divergent update is
  skipped, the first offending step is recorded, `divergence.json` is written, and the warning is in
  the returned logs. The pipeline had no such guard either.
* **The teacher WER read 0.926 once, then 0.000 on the same clips.** The audio tensors were verified
  byte-identical (no in-place mutation anywhere in the mel/encode/decode/phase-coherence/DNSMOS path)
  and DNSMOS was verified deterministic, so the bad number came from the recogniser — a transient
  int8 CTranslate2 failure. The control is what caught it, so `whisper_wer_with_control()` now runs the
  control **first**, retries once, and reports the WER as *unavailable with the failure as the reason*
  if the control still fails.

### The fixed autoencoder exposed the next bottleneck, which is the point

With the autoencoder working, the same diagnostic becomes informative instead of saturated at WER 1.0
everywhere:

| path | round 19 | now |
|---|---|---|
| reference (Kokoro) | 0.000 | 0.000 |
| 1. autoencoder round-trip | 1.000 | **0.167** |
| 2. teacher frame latent | 1.000 | **0.167** |
| 3. teacher token expanded | 1.000 | **0.870** |
| 4. student (text in) | 1.000 | ~1.0–1.7 |

**The seam is now the bottleneck**: expanding per-token latents exactly as inference does, using the
*teacher's own* durations/F0/energy, costs a 5× WER increase (0.167 → 0.870) for only 0.016 of mel
cosine (0.9941 → 0.9782). The proxy understates the damage again, in the same direction as round 19 —
which is why WER with a teacher control is the metric that decides here.

Two hypotheses were tested and one was killed:

* **`prosody_proj` is never trained.** It is called only from `latent_from_tokens` (inference), while
  no stage calls that function: the autoencoder stage encodes and decodes directly, `distill-text`
  only compares token-level predictions, and `distill-decoder` consumes the *cached frame* latent. So
  an untrained random projection is added to the latent at synthesis time — a textbook "declared but
  unwired" defect. Path **3b** removes it, and the measurement **falsifies** the hypothesis: 0.926
  versus 0.870, slightly *worse* without it. The control stays in the diagnostic so the negative
  result is reproducible rather than folklore.
* **The remaining loss looks like the design, not a wiring bug**: the cached per-token latent is an
  *average* over the ~6 frames of that token, so re-expansion cannot recover within-token detail — and
  `distill-decoder`'s docstring says it trains "on exactly the latent distribution the text side will
  produce at synthesis time" while `run_stage` hands it `batch["latent"]`, the clean frame-level
  latent. Intent and implementation disagree, which is the next thing to fix and measure.

A measurement note: the student's WER ranges from 1.04 to 1.72 across otherwise identical runs, while
every working path is stable to the digit. That is not pipeline nondeterminism — it is what WER looks
like when the audio is near-random and the recogniser is guessing.

## 23. The token → frame seam: what the fix achieved, and what it did not (measured)

Round 20 left the seam as the bottleneck: expanding per-token latents as inference does scored WER
0.870 where the frame-level latent scored 0.167. `distill-decoder` exists for exactly this, and its
docstring promised it — "the decoder then trains on exactly the latent distribution the text side will
produce at synthesis time" — but the stage consumed the cached *frame* latent **and could not run on a
real cache at all**: the shards stored features but no waveform, so `autoencoder_step` raised
`KeyError: 'wav'` and the stage had only ever executed under `--dry-run`. Three things had to be fixed
before the A/B was even expressible:

* **the cache now stores the target waveform** (aligned to the latent frames, ~4 bytes/sample), and an
  old cache fails with an actionable message instead of a bare `KeyError`;
* **`train.py` derives `n_voices` from the cache** — a mismatch cannot be loaded even with
  `strict=False`, and `--resume` on a 3-voice checkpoint against `n_voices: 1` failed for this reason;
* **`--warm-start` was added**: `--resume` restores the previous stage's optimizer state, which does
  not match a different stage, so "fine-tune a new stage from existing weights" needed its own flag.
  Also `--set key.path=value`, so an A/B is a run-time switch instead of two configs that drift apart.

`scripts/seam_ab.py` then runs both arms from one warm start, same cache, same 300 steps (the
discriminator held at zero: it is orthogonal to *which distribution* the decoder sees and costs 24×
per step):

| path | frame latent (flag off) | token-expanded (flag on) |
|---|---|---|
| 1. autoencoder round-trip | 0.167 | **0.167** (undamaged) |
| 2. teacher frame latent | 0.167 | 0.167 |
| 3. teacher token expanded | **0.889** | **0.722** |
| 3b. token expanded, no `prosody_proj` | 0.796 | 0.907 |
| 4. student (text in) | 2.796 | 1.037 |

**The documented mechanism works and is not sufficient.** Training the decoder on the distribution it
actually meets improves the seam by 19 % relative (0.889 → 0.722) without touching the paths that
already worked — and the student improves with it — but it does not close the gap to the frame path
(0.167). The A/B report records that as a check
(`gap_to_the_frame_path_remains_and_is_recorded`) so the headline cannot quietly become "seam fixed".

A side-effect worth naming, because it confirms the round-20 reading: with the flag on, the decoder's
input is built by `decoder_latent_from_tokens`, which puts **`prosody_proj` in the graph** — and once
trained it *helps*: removing it now costs 0.907 versus 0.722, the reverse of the untrained case. The
projection was not the seam's cause, but it was a real defect, and this stage now trains it.

The residual is structural: the cached per-token latent is an **average** over that token's ~6 frames,
so no decoder can recover within-token detail that averaging removed. Two candidate fixes, recorded for
the next round: raise the effective token rate (predict sub-token latents rather than one average per
text token), or add a refinement stage that consumes token latents and predicts frame latents — the
role the flow/consistency sampler plays in the Small model.

## 24. The seam is mostly *information*: a token-rate sweep, and the rate change end to end (measured)

Round 21 showed that training the decoder on the distribution inference feeds it helps the seam
(0.889 → 0.722) but does not close it. The suspected cause was arithmetic: one latent per text token
is 24 numbers per ~6 frames — **3.9 dimensions per frame against the encoder's 24** — so no decoder
recovers detail the averaging removed.

`scripts/seam_rate.py` tests that directly and **without training**: for K sub-latents per token it
fills each sub-span with the mean of the teacher's own frame latent over that sub-span and decodes.
K=1 reproduces the pipeline; K = frames-per-token approaches the frame latent. This is an *oracle* for
a text side that predicts K latents perfectly, which is exactly the quantity needed to decide whether
to change the architecture.

| rate | dims/frame | WER | log-mel cosine | DNSMOS |
|---|---|---|---|---|
| 1 (shipped) | 3.9 | 0.722 | 0.9783 | 1.55 |
| 2 | 7.8 | 0.204 | 0.9886 | 1.47 |
| **3** | 11.7 | **0.093** | 0.9911 | 1.50 |
| 6 | 23.3 | 0.296 | 0.9931 | 1.68 |
| 12 | 46.7 | 0.167 | 0.9938 | 1.66 |

**77 % of the seam is information loss**, and doubling the token rate already recovers most of it. The
curve plateaus (and the WER bounces around at this sample size) beyond rate 3 — the decoder was trained
on rate-1 inputs, so higher rates are off-distribution. The check is therefore stated as a plateau, not
as "more is always better", which the data does not support.

Two things surfaced while measuring this, both fixed: the decoder's output can leave
`[-1, 1]` (DNSMOS refused the array), and the polyphase resampler **overshoots even after a clamp**, so
`dnsmos_score` now clamps after resampling and reports the clipped fraction in its detail.

### The change, and the honest result

`latent_rate` is now a config field: the cache averages sub-token targets over sub-spans and the text
side's latent head widens to `latent_rate × latent_dim`. The geometry comes from one function
(`subtoken_spans`) used by **both** the target builder and the expansion, because splitting the span in
two places is how this kind of change silently misaligns — a test pins that contract.

Both arms: same autoencoder, same corpus, same 800 steps.

| | rate 1 | rate 3 |
|---|---|---|
| student WER (`base.en`) | 2.204 | **1.648** |
| student DNSMOS | 1.430 | **1.633** |
| log-mel cosine vs reference | 0.9393 | 0.9427 |
| speed | 135× real time | 116× real time |
| teacher WER / DNSMOS (controls) | 0.000 / 2.613 | 0.000 / 2.613 |

The oracle gain **does** transfer: the student improves on both axes at no meaningful cost in speed
(the model is 32 KB wider, and both arms run two orders of magnitude faster than real time). And the
student is still unintelligible while the oracle path is not — which localises what remains: **the gap
is latent prediction, not the seam and not the autoencoder.** The text side fits its objective well
(loss 0.655) while the audio it renders is wrong, the signature of a latent-space L1 that does not
track what the decoder needs. The next change is therefore to the **objective**, not the architecture:
train the text side through the decoder with a mel/audio loss instead of an L1 on cached latents.

## 25. Changing the objective: train the text side *through* the decoder (measured)

Round 22 localised what was left: the text side fitted its latent L1 well (0.655) while the audio it
rendered was unintelligible, and the oracle token path was fine — so the objective, not the
architecture, was wrong. A small error in latent space is not a small error in the decoder's output,
and nothing in the loss ever looked at the decoder's output.

`distill-audio` is a new stage that closes the loop: predict the token signals, build the decoder input
with the **same call synthesis makes**, decode, and compare the resulting audio to the teacher's with
the mel + multi-resolution STFT losses. The autoencoder stays frozen as the fixed renderer, so the
gradients reach the text side, the duration head, and `prosody_proj` — for the first time in a path
that is trained rather than merely executed. A test pins that contract (heads move, autoencoder does
not). The cached-signal objective stays as a small auxiliary term (`audio_aux` 0.1), because rounding
durations to frames is not differentiable and something has to pin the utterance length; the logs
record it (`rendered_frames` 415 vs `target_frames` 405, so it works).

Three arms, same autoencoder, same corpus, `latent_rate` 3 throughout:

| arm | student WER | student DNSMOS | log-mel cosine | speed |
|---|---|---|---|---|
| latent L1 (`distill-text`, 800 steps) | 1.648 | 1.633 | 0.9427 | 116× |
| through the decoder, 600 steps | 1.000 | 1.586 | 0.9447 | 115× |
| **through the decoder, 2400 steps** | **0.667** | 1.583 | **0.9486** | 120× |
| teacher (controls) | 0.000 | 2.613 | 1.000 | — |

The objective change works and **the gain keeps coming with steps** (2.5× lower WER than the latent L1
at 2400 steps, with a better mel proxy and no speed cost). The naturalness metric is the honest
exception: it does not move (1.633 → 1.583), which the A/B records as a check
(`naturalness_trade_off_is_recorded`) rather than leaving out.

Not yet a usable model: WER 0.667 still means the recogniser misses about as many words as it finds.
What changed is that the loss now measures what the decoder produces — the prerequisite for every later
gain — and the pipeline finally has a stage that optimises the thing the project is judged on.

Two bugs surfaced on the way, both in code added this round: `latent_from_tokens` summed durations over
the whole *batch* instead of per item (5200 frames against the prosody path's 351 — a single-item test
with equal durations could not catch it, so the test now uses a batch with uneven durations), and
`real_eval.py` inferred `n_voices` from a neighbouring cache directory, which failed for any run whose
cache lived elsewhere; it now reads the width from the checkpoint itself.

## 26. Scaling the corpus, and two thresholds that a real corpus broke (measured)

Round 23 fixed the objective, so the binding constraint became data: every real-audio measurement in
this project had trained and evaluated on the **same 21 utterances**. `scripts/scale_corpus.py`
synthesises a larger Kokoro corpus (59 prompts × 4 voices = 236 utterances, **19.4 min** of audio in
10 min at RTF 0.52) and splits it **by prompt**, so validation text is unseen rather than merely held
out by filename:

| | utterances | minutes | prompts |
|---|---|---|---|
| train | 129 | 10.9 | 47 |
| val | 31 | 2.7 | 12 (unseen text) |

All 11 voices in this bundle are `af_*`, so the corpus is single-gender — which limits what multi-voice
conditioning can be shown to do, and is recorded in the corpus report rather than implied away.

### What the hold-out revealed: the earlier WER gains were partly fitting

The scaled autoencoder generalises: trained on 129 utterances over 47 prompts, its round-trip on the
validation utterances is **WER 0.289 against the teacher's own 0.281** — indistinguishable from the
recogniser's noise floor on that split, and a better mel fit than the small-corpus run (0.4597 vs
0.4987). The renderer is no longer the limit.

The text side is a different story, and it is the important one:

| evaluated on **12 unseen prompts** | 21-utterance model | scaled model (129 utts / 47 prompts) |
|---|---|---|
| student WER | 1.000 | **1.000** |
| student DNSMOS | 1.441 | **1.537** |
| log-mel cosine | 0.9391 | **0.9466** |
| speed | 125× real time | 143× real time |
| teacher WER / DNSMOS (controls) | 0.000 / 2.655 | 0.000 / 2.655 |

Six times more audio improves both proxies and nothing else moves. **Round 23's headline (WER 1.648 →
0.667) was measured on the corpus's own texts**, so it was partly fitting; with a prompt-disjoint
hold-out the student is still at chance on new sentences. The A/B records this as
`not_yet_generalising: true` in its body and states it in the conclusion, because a hold-out exists to
change the story when the story was flattering.

This reframes the next step: the remaining problem is not renderer capacity, corpus size, latent rate,
or the objective — it is that ~10 minutes of multi-voice synthetic speech does not teach a 9.6 M model
to read *new* text. The levers that follow from this measurement are a much larger and more varied
corpus (the RTF 0.52 synthesiser makes that cheap), and a text side with a stronger inductive bias for
generalising across unseen sentences.

### The gate that would have deleted the whole corpus

`CurateConfig.min_dnsmos = 3.5` was a borrowed constant ("PilotTTS: deficient if MOS ≤ 3.5") and, like
several things in this project, it had never met a real corpus: no caller injected a predictor, so the
perceptual stage was silently skipped and only the signal gates ran. Now that `speechmos` is installed
it is the default, and the first measurement is unambiguous:

| statistic (160 curated Kokoro utterances, DNSMOS P.835) | value |
|---|---|
| mean / median | 2.86 / 3.21 |
| p05 / p10 / p25 | 1.88 / 1.94 / 2.19 |
| the same audio's **P.808** sub-score | 3.83–4.04 |
| kept by `min_dnsmos = 3.5` | **1 of 160** |
| kept by the calibrated `min_dnsmos = 2.0` | 138 of 160 (86 %) |

The borrowed 3.5 would have discarded a corpus whose ASR WER is **0.000**. The same audio scores ~3.9
on P.808, so the constant was almost certainly on that scale rather than the P.835 *overall* scale —
which is why the default is now **calibrated from this distribution** (2.0, dropping the worst 14 %)
with the measurement in the code comment, and why the test asserts the calibrated value rather than
trusting any published number.

The signal gates have the same flavour of problem: **76 of 236** utterances (32 %) were rejected as
`narrowband` (bandwidth99 < 5 kHz), a CosyVoice-derived threshold meeting a different synthesiser. That
is a real cost — ~5 minutes of audio — and the honest reading is that the gate is doing something
(selecting the wider-band half) while the *number* is not calibrated for this teacher.

### Divergence at scale, and the guard's second iteration

The first scaled autoencoder run **diverged at step 1756** and never recovered: 2245 of 4000 steps were
skipped, and because only the *loss* was checked, NaN gradients were still written into the parameters,
so every later loss was NaN. Round 20's guard turned an all-NaN report into a diagnosis with a step
number; round 24 completes it — `run_stage` now checks the **gradients** too (`_optimise` returns False
if any gradient is non-finite, so the update is skipped) — and the run caps clip length
(`--max-seconds`), because a padded batch containing a 20 s utterance is both heterogeneous and, most
likely, the trigger. A test injects a NaN gradient with a finite loss and asserts the step is skipped
and no parameter is left non-finite.

## 27. Text diversity: 185 public-domain prompts instead of 47 (measured)

Round 24's hold-out named the binding constraint: the model had only ever seen **47 distinct
sentences**, so more *audio* (21 → 129 utterances) moved the proxies but left WER at 1.000 on unseen
prompts. Duration was never the constraint — *text* was.

`scripts/build_prompts.py` takes 600 sentences from five public-domain novels (Austen, Doyle, Carroll,
Wells, Shelley — 4 810 usable sentences after length/character filtering, interleaved so no split is one
author) and records the provenance of every source, because a corpus whose licence cannot be traced is
not usable. Synthesis is cheap (RTF 0.49), so 450 prompts × 2 voices = 900 utterances and **62.6
minutes** of audio took 31 minutes to generate.

Curation kept **383 of 900**:

| reason | count | what it means |
|---|---|---|
| `narrowband` | 333 (37 %) | bandwidth99 < 5 kHz — the CosyVoice constant meeting a different synthesiser |
| `too_short` | 227 (25 %) | < 3 s — the CosyVoice minimum meeting *natural prose*, which is mostly short sentences |
| `low_mos` | 0 | everything that survived cleared the calibrated 2.0 DNSMOS floor |

The `too_short` count is a third borrowed threshold biting on real text: an ordinary novel loses a
quarter of its sentences to a 3 s floor. The surviving corpus is what matters:

| | utterances | minutes | prompts |
|---|---|---|---|
| train | 304 | 22.2 | **185** |
| val | 79 | 6.1 | 50 (unseen) |

4× the distinct training text and 2× the audio of round 24, on the same kind of prompt-disjoint
hold-out — so the comparison isolates text diversity. **And it is not enough either:**

| evaluated on the same 24 utterances from 50 unseen prompts | 47-prompt model | 185-prompt model |
|---|---|---|
| student WER | 1.000 | **0.997** |
| student DNSMOS | 1.486 | **1.600** |
| log-mel cosine | 0.9484 | **0.9525** |
| speed | 142× real time | 128× real time |
| teacher WER / DNSMOS (controls) | 0.168 / 3.257 | 0.168 / 3.257 |

More distinct text improves both proxies again and leaves WER at chance.  Together with round 24 that is
a useful negative result with two parts: **corpus duration is not the constraint (round 24), and neither
is text diversity alone (round 25)** — 47 → 185 prompts moved DNSMOS 1.49 → 1.60 and WER 1.000 → 0.997.
The remaining suspects are the text side's inductive bias (character input relearning English
spelling-to-sound from a few hundred sentences) and its capacity, and the next round has a verified
lever for the first of those.  Note also that the teacher's own WER on this hold-out is 0.168, not
0.000 — novel prose is genuinely harder for `base.en` than the hand-written prompts were, which is
itself a reason to keep the control attached to every number.

### The other lever, verified available: phoneme input

If more text is not enough, the remaining suspect is the text side's *inductive bias*: character input
has to relearn English spelling-to-sound from a few hundred sentences, where the papers feed phonemes.
That depends on a local phonemiser, and the obvious route (`misaki` → spacy → blis) does not build here
— so it was checked rather than assumed: `espeakng-loader` + `phonemizer` produce correct IPA locally,
including for words the corpus never contains (`quixotic → kwɪksɑːɾɪk`,
`perspicacious → pɚspɪkeɪʃəs`) over a 39-symbol vocabulary. Recorded so the next round starts from a
verified capability instead of a hope.

## 28. A second real teacher: Speechify, and the first real alignment (measured)

The objective asked for "mix training of both" teachers. Until now that was expressible only against
fixtures: `synthesize_corpus` realises a mixture by *choosing* one teacher per prompt, so every corpus
was single-teacher, and the one hosted teacher wired up (MiniMax) is refused by default for licence
reasons. The operator obtained written permission from **Speechify** for this task, stated to cover
demonstration/quantization, so a second real teacher is now in the pipeline.

### The permission is narrower than "train and publish", and that is recorded

`SPEECHIFY`'s spec says `allows_training=True` with the wording *"stated to cover demonstration and
quantization purposes"*, and `docs/LEGAL.md` marks the boundary explicitly: distilling a student is
demonstration, **publishing derivative weights is a broader act that this permission does not
establish**. The licence string travels on every corpus record and appears in the model card's teacher
table. The key lives in `.secrets/speechify.key` (gitignored) or `SPEECHIFY_API_KEY`; a hygiene test
scans every tracked file for key-shaped strings, and the backend test asserts the key never reaches a
corpus file.

### The service brings word-level timings, which the project never had

`speech_marks` returns, per word, the character offsets **and millisecond timings**. That is the
alignment every duration target in this project has lacked — rounds 19 and 22 both measured the
unaligned fallback as a real quality limit. `marks_to_token_frames()` turns the marks into one frame
count per character (the char tokenizer emits exactly `len(text)` tokens, so a mismatch would mis-assign
every duration), and the corpus builder writes `token_frames` into the record; the cache picks it up
through the existing `use_teacher_durations` path. Verified rather than assumed:

| check (964 curated Speechify utterances) | result |
|---|---|
| records carrying an alignment | **964 / 964** |
| one frame count per character | every record |
| **median relative error between the timings and the audio duration** | **0.008** (p90 0.019) |

### A fourth borrowed threshold, and this one cost 60 % of a paid corpus

Curation kept **383 of 1350** Speechify utterances. The reason was `narrowband` (965), not quality:
the CosyVoice `min_bandwidth_hz = 5 kHz` floor was calibrated on a 24 kHz synthesiser, and at 48 kHz a
5 kHz bandwidth is **0.21 of Nyquist** — right at the edge. Speechify's DNSMOS is *higher* than
Kokoro's (mean 3.33 vs 2.86), so the gate was discarding the better teacher. Re-curating the same
audio with a measured 4 kHz floor keeps **964 of 1350**; the choice is now a CLI parameter
(`--min-bandwidth-hz`) with the measurement recorded, rather than a constant from another system.

### Two teachers, mixed, with a prompt-disjoint hold-out

`scripts/mix_corpora.py` joins finished corpora into one mixture manifest — referencing waveforms by
relative path instead of copying them — and writes the `corpus_meta.json` that
`cache_teacher_corpus` reads for the mixture weights. That is what makes mix training real:

| | utterances | minutes | prompts | alignment |
|---|---|---|---|---|
| Kokoro (English, all `af_*`) | 304 | 22.2 | 185 | — |
| Speechify (3 voices, 2 locales, mixed gender) | 865 | 87.3 | 399 | **yes** |
| **mixture (train)** | **1169** | **109.5** | | 865/1169 (74 %) |
| mixture (val, 50 unseen prompts) | 178 | 16.2 | 50 | |

### A metric bug: WER depended on the teacher's sample rate

Evaluating Speechify audio reported a **teacher** WER of 1.488 — impossible for a service scoring
DNSMOS 3.33. The cause: faster-whisper treats a numpy array as **16 kHz**, so 24 kHz audio is read 1.5×
too fast and 48 kHz audio 3× too fast. Kokoro's audio survived that (teacher WER 0.000), which is
precisely why the bug hid for ten rounds. `whisper_wer` now resamples from the declared rate to 16 kHz:

| | before | after |
|---|---|---|
| Speechify teacher WER | 1.488 | **0.268** (control valid) |
| Kokoro teacher WER on the round-25 hold-out | 0.168 | 0.140 |
| student WER on that hold-out | 0.997 | **0.994** |

The last row is the one that matters: round 25's conclusion (*the student is at chance on unseen
prompts*) was **re-checked under the fixed metric rather than assumed**, and it holds. Every earlier
number was measured with a 1.5× speed-up fed to the recogniser; the ones that were fine were fine by
luck of the robustness range, which is a reason to fix the metric and re-run rather than to trust it.

## 29. The two-teacher mixture, and what the alignment actually adds (measured)

Round 26 wired in a second real teacher and verified its word timings against the audio. Round 27
trains on the mixture and asks the second question: **does the alignment change what the duration head
is taught?**

### What the alignment adds

`scripts/alignment_value.py` compares, on the same utterances, the teacher's aligned per-character
frame counts against the energy-weighted fallback that `extract_signals` produces without an alignment:

| | |
|---|---|
| median correlation between aligned and fallback per-token durations | **0.148** |
| length error vs the audio, aligned | 0.83 % |
| length error vs the audio, fallback | 0.07 % |

The fallback's *total* is exact **by construction** — it splits the frames the audio actually has — so
"which is closer in length" is not the question. The informative number is the correlation: **0.148**
means the teacher puts duration on *different tokens* than a smooth split does, which is exactly the
phonetic structure a duration head should learn. That justifies an A/B rather than assuming one.

Two of my own measurement bugs had to be fixed to get there, and both were caught because round 26
independently verified the alignment at 0.8 % while this script first reported 50 %, then 100 %:
`token_frames` are computed at the **student's** 24 kHz, so comparing them against 48 kHz frames made
the alignment look half-length; and `extract_signals` was being run on the 48 kHz waveform, so *its*
fallback looked twice as long as the 24 kHz reference. The pipeline resamples before extracting
anything, and the script now does too. Cross-checking against an independent measurement is what found
both.

## 30. The two-teacher mixture: what it changed, and what it did not (measured)

The mixture trained for 1600 steps with **zero divergent steps**: 1169 utterances, 109.5 minutes, two
teachers at 0.5/0.5 (Kokoro 304 + Speechify 865), 74 % of them carrying the teacher's own alignment, and
a voice table spanning two genders and three locales.

Getting a *valid* number out of it took three more fixes, each found by a control or a cross-check
rather than by inspection:

* **the recogniser control failed at 0.510** on the mixed hold-out, and the first suspect was wrong: the
  prompt list had picked up **chapter headings** from the source novels ("A Caucus-Race and a Long Tale
  CHAPTER IV." is spoken "…chapter four"), whose written form can never match a transcript. Hence
  `is_prose_like()` plus a `--prose-only` evaluation flag, and a filter in `build_prompts.py` so future
  corpora never contain them.
* **the control still failed at 0.925 on Speechify** once headings were filtered: `real_eval.py`
  declared the *config* rate (24 kHz) for a 48 kHz teacher's reference audio, so the control was
  time-stretched 2×. Resampling the reference took the control to **0.086**.
* **`WaveformCorpusSource` had the same mis-rating in the training path**: it fed raw 48 kHz teacher
  audio to an autoencoder whose mel filterbank assumes 24 kHz. `build_latent_cache` always resampled;
  that path did not. It now takes a `sample_rate`, and `ae_train.py` passes the student's.

With valid controls, on unseen prompts (prose only, 24 utterances each):

| hold-out | student WER | teacher WER (control) | student DNSMOS | mel cosine | speed |
|---|---|---|---|---|---|
| Kokoro, 50 unseen prompts | **1.000** | 0.096 | 1.567 | 0.9521 | 100× |
| Speechify, 50 unseen prompts | **1.000** | 0.086 | 1.542 | 0.9203 | 76× |

**A second real teacher, its own alignment, five times the audio and two genders do not move
intelligibility on unseen text.** With rounds 24 and 25 this is a three-part negative result — corpus
duration, text diversity, and teacher count/quality/alignment are each insufficient — which leaves the
text side's inductive bias and capacity as the remaining suspects rather than the data. The capability
to test the first of those (phoneme input) was verified in round 25.

## 31. Phoneme input: the last suspect, made testable (measured)

Rounds 24–27 ruled out corpus duration, text diversity, teacher count, teacher quality and real
alignment, leaving the text side's **inductive bias**: a character model has to relearn English
spelling-to-sound from a few hundred sentences, where the papers feed phonemes. Round 25 verified a
local phonemiser is available; this round wires it in and makes the comparison runnable.

### The vocabulary had to be derived, not written

`_IPA_CHARS` was hand-written and **missing seven symbols espeak actually emits** — including the length
mark `ː`, which appears **1865 times** in 600 corpus prompts, and the script `ɡ`. A phoneme model built
on that list would have sent `UNK` for nearly every long vowel. The inventory is now derived from the
phonemiser's own output over the corpus (44 symbols), and a test asserts that encoding a sample produces
**zero** `UNK`.

### What was added

* `parakeet/data/g2p.py`: `text_to_phonemes()` (cached espeak-ng via `espeakng-loader` + `phonemizer`,
  the route that actually builds here) and `phoneme_frames_from_char_frames()`, which converts the
  teacher's **word** timings into per-phoneme frame counts while preserving each word's measured total —
  the within-word split is a documented approximation, the word boundaries are the teacher's own.
* `TextTokenizer(mode="phoneme")` phonemises inside `prepare()`, so the cache builder and synthesis
  cannot disagree about what the model reads, and degrades to characters (loudly) when no phonemiser is
  present rather than emitting `UNK`.
* `build_latent_cache` converts the alignment to phoneme tokens and reconciles the two axes.

Verified before training rather than after:

| check on the phoneme cache (1169 utterances, 2 teachers) | result |
|---|---|
| `UNK` tokens across the whole cache | **0** |
| items whose duration axis matches the token axis | **1169 / 1169** |
| tokens per utterance | 28 / 67 / 132 (min / median / max) |
| phoneme vocabulary fits the configured embedding | 106 ≤ 128 |

### The A/B, and the measurement that reframed it

Both arms: the same mixture cache, the same autoencoder, 1600 steps, `latent_rate` 3, evaluated on the
same 24 utterances from 50 unseen prompts per hold-out (prose only, controls valid):

| arm | hold-out | student WER | teacher WER | student DNSMOS | mel cosine |
|---|---|---|---|---|---|
| char | Kokoro | 1.000 | 0.096 | 1.567 | 0.9521 |
| char | Speechify | 1.000 | 0.086 | 1.542 | 0.9203 |
| phoneme | Kokoro | 1.000 | 0.096 | 1.448 | 0.9521 |
| phoneme | Speechify | 1.026 | 0.086 | 1.417 | 0.9189 |

Phonemes do not improve intelligibility and cost a little on the proxies, so the inductive-bias
hypothesis is not supported either — a fourth negative result.

**But the same run reported something more important.** Asked to synthesise its **own training prompts**,
the student is still at WER 1.000 (char and phoneme alike, controls 0.082):

| evaluation set | student WER | teacher WER (control) |
|---|---|---|
| 24 utterances from 50 **unseen** prompts | 1.000 | 0.086–0.096 |
| 24 utterances from the **training** prompts | **1.000** | 0.082 |

The model is not failing to *generalise*; it never fit the data it trained on. That reframes rounds
24–27: they compared variants of an **undertrained** model along axes (corpus duration, text diversity,
teacher count, tokeniser) that could not have mattered yet. The arithmetic agrees — 1600 steps at batch
4 is 6 400 samples over 1169 utterances, about **5.5 passes** — and the audio objective is still ~1.4
against the autoencoder's own reconstruction of ~0.46.

So the project's own framing, mine included, was wrong for four rounds, and the fix is not another A/B
but **training until the model fits before comparing anything**. A long run (6000 steps, warm-started)
is now in flight, and the fit number is what it will be judged on first.

## 32. Ingesting recordings you were handed (measured)

Eight Speechify takes arrived as files rather than through the API: the same Shakespeare paragraph in
**eight voices**, 24 kHz mono, 39–48 s each, 5.6 minutes total. Two problems, both structural:

* **Every one failed curation on `max_duration_s`** (30 s). The gate *rejects* long audio; nothing in
  the pipeline could *segment* it, so any real recording — a chapter, a podcast, a studio take — was
  unusable. `parakeet/data/segment.py` now packs ASR word timings into utterances inside the duration
  window, preferring the largest silences and splitting on the cap when a stretch has no usable pause,
  without losing or duplicating a word (asserted).
* **Recordings arrive without transcripts**, and TTS training needs (audio, text) pairs.
  `scripts/ingest_audio.py` transcribes with `word_timestamps=True`, segments, curates, and writes the
  manifest — recording that the text is an **ASR measurement**, not the teacher's own transcript, so a
  later reader is not misled about which it is.

| stage | result |
|---|---|
| ingested from 8 files | 57 segments, 5.0 min, **8 voices** |
| after curation with the CosyVoice 5 kHz floor | 21 kept (the same borrowed threshold round 26 calibrated) |
| after curation with the measured 4 kHz floor | **38 kept** (3.2 min, all 8 voices; the rest `narrowband`) |
| mixture v2 (Kokoro + Speechify + ingested) | **1207 utterances, 112.7 min, 8 Speechify voices** |

The lesson generalises past these files: a corpus builder that only accepts one-utterance teacher
output cannot use a recording, and a teacher whose licence is fine may still hand you audio with no
text. A test pins the segmenter's contract and the mixer's ability to join corpora with different
manifest layouts (`PATH:MANIFEST`) — the generated corpora write `train.jsonl`, an ingested one writes
`curated/kept.jsonl`.

## 33. What the text side actually learned: a fit diagnosis (measured)

Round 28's WER-1.000-on-training-text finding said the model never fit. WER says *that* it did not; this
says *which* output is wrong, which is what chooses the fix. `scripts/fit_diagnosis.py` runs the trained
text side over the cache and compares each of its four outputs against the teacher's:

| split | duration MAE | duration ratio | F0 MAE | energy MAE | latent cosine | **per-dim correlation** |
|---|---|---|---|---|---|---|
| train (the corpus's own utterances) | 2.31 frames | **0.770** | 0.159 | 0.179 | 0.805 | **0.126** |
| validation (tail of the cache) | 3.71 frames | 0.583 | 0.166 | 0.218 | 0.772 | 0.107 |

Three readings, all actionable:

1. **The latent is right in the mean and wrong in the structure.** A flattened cosine of 0.81 looks
   healthy, but the per-dimension correlation is **0.126** — the predicted 72-dimensional token latent
   is essentially uncorrelated with the target dimension by dimension. The model learned *the average
   latent* (which dominates both the cosine and the mel loss), not *which latent this token needs*. That
   is the signature the whole project has been chasing: mel cosine 0.95 against the reference while the
   speech is unintelligible.
2. **The duration head is biased, not merely noisy:** it predicts 0.77× the true length on training text
   and 0.58× on unseen text, and at ~6 frames per token a 2.31-frame MAE is a ~38 % relative error.
3. **This is underfitting, not memorisation:** every metric is worse on validation, and none is good on
   train. There is no fit to overfit from yet.

The fix this points at is not another data A/B. A one-shot regression of 72 latent values per token from
a small character encoder is being asked to learn the acoustics of speech from text — and the papers do
not do that: they run a **flow-matching or autoregressive decoder** over acoustic tokens, which is what
this repository's Small path implements and the Tiny path deliberately skipped. The 6000-step run in
flight tests the cheaper hypothesis (more optimisation) first; if the per-dimension correlation does not
climb, the structural one is next.

## 34. Why the latent was only ever learned in the mean: the objective, measured (round 30)

The fit diagnosis said the text side predicts the teacher's *average* token latent (flattened cosine
0.805) with a per-dimension correlation of 0.126. Reading the objective explains it arithmetically.
`text_audio_step` computes

```
total = audio_mel * l_mel + audio_spectral * l_spec + audio_aux * aux_loss
```

with `audio_mel = 1.0`, `audio_spectral = 3.0`, `audio_aux = 0.1`, and `aux_loss` is the *sum* of the
duration, F0, energy and latent terms. From the run's own logs (`audio_mel` 1.64, `audio_spectral`
1.64, `aux` 1.59, `aux_latent` 0.62) the whole cached-signal bundle is **~1 % of the objective**, and
the token-latent term inside it is **~0.7 %**. Everything else optimises the *rendered* mel and
spectral envelope — through a frozen decoder, where matching the average latent already reproduces the
envelope. The model learns exactly what it is paid for.

So the loss gained the term that is missing, and it is deliberately **mean-invariant**:

```python
pred_c   = pred   - pred.mean(dim=1, keepdim=True)     # across tokens, per dimension
target_c = target - target.mean(dim=1, keepdim=True)
l_latent_contrast = per_sample_mse(pred_c, target_c, mask)
```

It is blind to the mean, so it can only be reduced by matching the *variation* the diagnosis showed was
missing, and centring (not standardising) is deliberate: standardising by the prediction's own spread
could be minimised by shrinking toward a constant, which is the failure being fixed. Three algebraic
properties are pinned by `tests/test_latent_contrast.py` rather than a slow "it goes down" test:

| property | result |
|---|---|
| perfect prediction | contrast **0** |
| prediction shifted by a constant per dimension (i.e. any error in the mean) | contrast **0** — while the raw MSE moves by 24.5 |
| constant prediction (collapse to the mean) | contrast **= the target's variance** (0.718 vs 0.718) |

It is reachable from the command line (`--set train.loss.signal_latent_contrast=1.0`, alongside
`--set train.loss.audio_aux=1.0`), because a weight that only a code change can reach cannot be A/B'd.

### The decision to redirect the CPU, and why

The 6000-step run launched in round 28 was killed ~40 minutes in, at 143 CPU-minutes. It re-optimised
the objective whose composition is measured above — the latent term at ~0.7 % — so the most likely
outcome was a better mel envelope and an unchanged per-dimension correlation. The same CPU now trains
the corrected objective for 2400 steps, and it will be judged on the **fit diagnosis** (does the
per-dimension correlation climb?) rather than on WER, because WER is far less sensitive to exactly the
defect being fixed.

## 35. Smoke test output (measured)

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

## 36. Deliberate engineering checks worth calling out

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

## 37. Environment notes

* CPU torch was installed from the PyTorch CPU index (no CUDA on this machine), in a dedicated
  Python 3.13 venv; the system Python 3.14 also has torch wheels available (2.14.1).
* `soundfile` is required for corpus I/O; `transformers`/`snac`/`kokoro` (teachers),
  `faster-whisper`, `funasr`, `onnxruntime` are optional and every code path that needs them
  reports unavailability instead of faking a metric.
* The repo's `runs/`, `data/`, `*.pt`, `*.wav` outputs are git-ignored; nothing large is committed.
