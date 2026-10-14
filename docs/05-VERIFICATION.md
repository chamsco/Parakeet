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

### The result: the term worked, and it was not enough

2400 steps on the same mixture, warm-started, with `audio_aux = 1.0` and `signal_latent_contrast = 1.0`:

| metric | before | after |
|---|---|---|
| per-dimension correlation, train | 0.126 | **0.200** (+59 %) |
| per-dimension correlation, validation | 0.107 | **0.178** (+66 %) |
| flattened cosine, train | 0.805 | 0.839 |
| F0 MAE (train) | 0.159 | **0.110** (−31 %) |
| energy MAE (train) | 0.179 | **0.133** (−26 %) |
| duration MAE (train) | 2.312 | 2.217 |
| WER, training prompts | 1.000 | 0.994 |
| WER, unseen Kokoro prompts | 1.000 | **1.000** |

The training logs agree with the diagnosis: `aux_latent` 0.618 → 0.340 and `aux_duration` 0.693 → 0.441.

**The term moved exactly the quantity it was built for, and intelligibility did not follow.** The
correlation is still only ~0.2, so the honest reading is that a one-shot regression from a small
character encoder to 72-dimensional token latents has a *ceiling* well below what the decoder needs —
one that no re-weighting of the objective reaches. That is what the papers avoid by running a
flow-matching or autoregressive decoder over the acoustic tokens, which this repository's Small path
already implements. The next move is that architecture, not another weight.

## 36. The Radeon: measured, and why it is not the unlock (round 31)

This machine has an **AMD Radeon RX 6950 XT** (RDNA2, gfx1030). Since round 28 established that the
model was *undertrained*, compute is the binding constraint on every experiment, so this was worth
measuring rather than assuming. On Windows the route into PyTorch is `torch-directml`, which needs
Python ≤ 3.12: a second venv (`.venv-dml`, Python 3.11) was built with `torch-directml 0.2.5` and
`torch 2.4.1`. The device works — `torch_directml.device_name(0)` reports *AMD Radeon RX 6950 XT* and
20 2048² matmuls take 0.134 s.

### What the GPU is actually worth here

| workload | CPU | DirectML | speedup |
|---|---|---|---|
| `distill-text` step, batch 8 | 91 ms | 91 ms | **1.0×** |
| `distill-text` step, batch 16 | 162 ms | 107 ms | **1.5×** |
| decoder conv stack, 4×900 frames | 67 ms | **11 ms** | **6.1×** |
| full `distill-audio` step | 478 ms | *cannot complete* | — |

The GPU is fast exactly where the work is big and regular (the conv stack, 6.1×) and no faster where
it is small ops and per-op overhead (the text side, 1.0–1.5×). The audio step's CPU breakdown shows
where the time goes: text side 23 ms, **decode 158 ms**, mel loss 7 ms, spectral loss 34 ms.

### Why the audio step still cannot train on it

Each of these was found by measurement, not by reading documentation, and they compound:

1. **No complex dtype.** `torch.polar` / `torch.fft.irfft` / `torch.stft` are unavailable
   (*"Invalid or unsupported data type ComplexFloat"*), and the autoencoder's output head plus both
   STFT losses depend on them.
2. **`F.fold`'s backward is `aten::col2im`**, unimplemented; the plugin's CPU fallback leaves the
   autograd graph crossing devices, and the backward then dies *inside the plugin's own error handler*.
3. **`torch.eye` is broken**: it falls back and returns an **empty tensor** `(0,)`.
4. **`repeat_interleave` is unimplemented**, so an integer index ends up on the CPU while the features
   are on the GPU.
5. **`index_add` falls back to the CPU.**
6. **The plugin's error path raises `UnicodeDecodeError`** while decoding a Windows error string, so
   real failures surface as decode errors instead of messages — which is what made 2–5 slow to find.

### What the repository gained anyway

The changes are correct and stay (all CPU behaviour verified unchanged — 291 tests pass, the smoke test
passes, synthesis numerically identical):

* a **real-valued iSTFT / inverse-DFT and magnitude-spectrogram path**, selected only where the device
  has no complex dtype, asserted equal to `torch.fft.irfft` and `torch.istft` (five equivalence tests,
  plus the streaming path);
* the **overlap-add rewritten as a transposed convolution** with an identity kernel, removing the
  `F.fold`/`col2im` dependency while keeping streaming numerically identical to offline;
* **`align_token_frames` builds its index on the CPU** and moves it, so the gather — and the gradient —
  stays on the model's device;
* `text_audio_step` takes a `loss_device`, for a backend that needs the losses placed elsewhere.

### The honest conclusion

**DirectML is not the unlock.** Realistic gains are 1.0–1.5× on the small-op-bound parts and 6.1× on
the conv stack, while the ops the audio stage needs are missing or broken and the failure reporting is
hostile. For real GPU training on a gfx1030 card the viable route is **ROCm on Linux (native or
WSL2)**, which officially supports this card and has complete op coverage. DirectML remains a
reasonable route for *inference* — which is not this project's bottleneck, since synthesis already runs
at 80–126× real time on the CPU.

None of this changes the scientific state: the model is undertrained, and the next work is training it
properly, on CPU, or on the GPU once a ROCm path exists.

## 37. The flow route, made usable on real data (round 32)

Round 30's A/B closed with a specific reading: a one-shot regression from a small character encoder to
72-dimensional token latents has a ceiling (per-dimension correlation 0.20 while the autoencoder round
trip transcribes at **WER 0.0**), and the papers avoid that ceiling by **sampling** the acoustic latents
with a flow-matching velocity field. This round puts that route on real data.

### A correction to round 31

Round 31 stated that the flow model "was trained by the smoke test but no synthesis path existed for
it". **That was wrong**: `ParakeetFlow.synthesize` exists (it samples the velocity field, unfolds the
compressed frames and decodes), and `Synthesizer.synthesize` already dispatches to it for non-Tiny
variants. What is true is narrower and still important — the flow variant had only ever been trained on
**synthetic** batches by the smoke-test recipe, never on a real corpus, and `real_eval.py` could not
report its numbers. Both are fixed here.

### A config trap worth naming

The flow cannot simply use `parakeet_small.yaml`: its autoencoder has `decoder_dim 384` / `encoder_dims
[128, 192, 256]`, while the cached latents were produced by the **Tiny** autoencoder (256 / [64, 96,
128]). Training the flow there would pair a decoder with latents from an encoder it does not share — the
warm start failed with exactly that shape mismatch rather than silently training on the wrong latents.
`configs/parakeet_flow.yaml` therefore takes `variant: small` (the flow model) with the **Tiny
autoencoder geometry**, so the existing caches and the trained autoencoder are directly reusable.

`--warm-start` also stopped failing on a single stale key: it now skips keys whose shape differs and
reports them (the autoencoder checkpoint predates the corpus growing from 4 voices to 12, and one stale
embedding should not cost a run).

### What the flow costs, measured

| | per step | audio per step | audio per second of compute |
|---|---|---|---|
| `distill-audio` (token latents, decodes audio) | 2.25 s @ batch 4 | 9.1 s | ~4 s |
| `flow` (velocity field, no decoding) | **2.02 s @ batch 8** | **73 s** | **~36 s** |

The flow sees roughly **eight times more audio per unit of compute** because its objective never
decodes a waveform — which is what makes a real flow run affordable on this CPU.

### The evaluation had to be fixed for it

An early flow checkpoint predicts a near-zero length, and then nothing overlaps a reference:
`real_eval.py` died inside `statistics.mean` on an empty list. It now reports `n/a` and, crucially, the
**length ratio** — for a flow sampler that is the first thing that must be right (untrained: **0.008**,
0.2 s generated against 21.3 s of reference). `--steps` selects the sampler's NFE, because the default
(`cfg.flow.nfe = 32`) runs at **0.34× real time** on the CPU while a distilled sampler is meant for 2-4;
SupertonicTTS reports WER 11.43 at NFE 4 against 2.64 at NFE 32, so the NFE belongs in every report.

### The decision, and the run

The 6000-step token-side run was stopped ~20 minutes in and the CPU redirected to a **4000-step flow run
on the 12-voice mixture** (`runs/flow_v2`, checkpoint every 400 steps so intermediate fit is
inspectable). The reasoning is measured, not aesthetic: the token path's ceiling is quantified, the
acoustic path is proven (round-trip WER 0.0), and the flow is both the papers' route and ~8× cheaper per
second of audio.

### Verification: what the two fixes changed, measured

The first checkpoint of the *pre-fix* run generated 0.3 s for a 35.8 s reference (length ratio 0.01).
Its length head read 0.35 in log frames against a 6.75 target, unchanged between step 12 and step 400 —
it was not learning at all. Training *only* that head at lr 1e-2 for 60 steps took the loss from 40.975
to **0.006** (0.35 → 6.81), which ruled the head out and pointed at arithmetic: AdamW moves a parameter
by roughly the learning rate per step regardless of gradient size, so a bias starting 6.75 away needs
~32 000 steps at lr 2e-4 to arrive. Every run in this project has been 1 600–6 000 steps. **That is the
duration collapse measured in round 30** — the duration head was never given the chance, not badly
trained. `DurationConfig.log_length_init = 6.5` fixes the start, and `train.py` now scales warmup to 5 %
of the run instead of consuming a third of it (the pre-fix flow run trained at an effective lr near
1e-5).

Re-run with both fixes, same corpus, batch 8, evaluated at step 400 (10 % of the run, NFE 4):

| at step 400 | before the fix | after |
|---|---|---|
| length ratio (generated / reference) | **0.01** | **1.67** |
| log-mel cosine against the reference | n/a — nothing overlapped | **0.794** |
| speed | — | RTF 0.03 (**30× real time** at NFE 4) |
| student WER (control 0.087) | 1.000 | 1.000 |
| student DNSMOS | 0.879 | 1.412 (14.6 % of samples clipped) |

Full-length audio, a computable acoustic proxy, and 30× real time at NFE 4 after 400 steps: the flow
trajectory has a gradient to follow, which the token path never showed. It is still not intelligible, so
the run continues to 4 000 steps and the trajectory (cosine, WER, clipping) is what gets reported next.

### The trajectory, and a proxy that misleads a sampler

`scripts/flow_trajectory.py` walks every saved checkpoint in step order and reports the three quantities
in the order they must come right (length ratio, acoustic proxy, WER with a control). At NFE 4:

| step | length ratio | log-mel cosine | student WER | control | DNSMOS | speed |
|---|---|---|---|---|---|---|
| 400 | 1.67 | 0.794 | 1.000 | 0.087 | 1.412 | 29× real time |
| 800 | 1.73 | 0.752 | 1.000 | 0.087 | 1.386 | 32× real time |

Full-length audio and ~30× real time, still at chance at 20 % of the run — and the acoustic proxy is
**flat**. That is worth stating plainly: **log-mel cosine against the reference is a poor progress
signal for a generative sampler.** The deterministic Tiny path copies the reference's timing and
phase-aligned envelope, which is why it scored 0.95 while being unintelligible; a flow samples a
*plausible* rendition from noise, so matching a different speaker's exact mel trajectory is neither
expected nor required. WER and DNSMOS are the metrics that mean something here, and both are still at
their floor — the run continues to 4 000 steps.

## 39. The head calibration did not pay off, and the throughput changed (round 34)

Round 33 measured the Tiny path's latent head starting with an output standard deviation of 0.25–0.29
against a target of ~1.03 (3.6× too small) and argued that AdamW's ~lr-per-step travel meant ~18 000
steps to close it. The fix was applied, and a 3 000-step run made with **every** fix in place
(`latent_rate` inferred, `audio_aux 1.0`, `signal_latent_contrast 1.0`, `length_init 6.5`, warmup
aligned to the run). Result, on the corpus's **own training prompts**, with a valid control:

| run | steps | per-dim correlation (train) | latent cosine | WER, training prompts | control |
|---|---|---|---|---|---|
| zero-shot (round 30 baseline) | 1 600 | 0.126 | 0.805 | 1.000 | 0.082 |
| contrast objective (round 30b) | 2 400 | **0.200** | 0.839 | 0.994 | 0.082 |
| **calibrated** (round 34) | 3 000 | 0.145 | 0.828 | **1.000** | 0.082 |

**The calibration did not move the quantity it was built for.** The output *scale* was fixed exactly
(it starts at 1.031 instead of 0.287) and the run's rendered/target frame ratio improved to **1.027**
from 0.77 — but the per-dimension correlation, which measures whether the right latent is produced for
each token, is unchanged within noise, and the model still cannot render its own training text. Fixing
where the head *starts* is necessary and insufficient: the structure is the expensive part, and a
one-shot regression does not learn it in 3 000 steps.

Two incidental findings worth keeping:

* **the audio stage is ~4.5× faster than earlier rounds measured**: 500 steps per ~4 minutes
  (**0.5 s/step** at batch 4) against the 2.25 s/step the round-30 runs recorded, and
  `nonfinite_steps: 0` confirms the speedup is not skipped updates. The likely cause is that earlier
  runs shared the CPU with my own concurrent evaluations — a reminder that a timing measured under
  contention is not a property of the code.
* the training log's rendered/target ratio (1.027) and the fit diagnosis's duration ratio (0.73, first
  24 cache items) disagree because they measure different subsets. Both are reported rather than one.

### What it means for the plan

At 0.5 s/step, 20 000 steps is ~3 hours instead of ~12, so "undertrained" is finally testable at scale.
But the evidence points elsewhere now: two different fixes to the one-shot regression each improved
their own targeted internal and neither moved the correlation past 0.2, while the flow — which
*samples* the frame latents instead of regressing them — already produces full-length audio at 30× real
time. The flow run was therefore resumed from step 800 to **6 000 steps**, and the Tiny regression path
is not where the next three hours go.

## 40. Neither path produces speech, and the proxies could not say so (round 35)

Every model in this project reaches WER 1.000 while its envelope proxy looks healthy — the Tiny path at
a log-mel cosine of 0.95, the flow at 0.90. WER alone cannot distinguish *wrong words*, which more
training can fix, from *not speech*, which it cannot. Measured for both, against the teacher's own audio:

| audio | voiced fraction | median F0 | spectral flatness | speech-like |
|---|---|---|---|---|
| **teacher** (control) | 0.67–0.72 | 193–202 Hz | 0.285–0.300 | **yes** |
| Tiny regression (calibrated, 3 000 steps) | **0.96** | 209 Hz (or 491 Hz) | 0.318 | **no** |
| flow (step 1 600, NFE 4) | **0.07** | 96 Hz | **0.547** | **no** |

Two different failures, both non-speech: the Tiny path emits a **near-silent tonal buzz** (98 % of frames
"voiced" at one pitch — 0.012 amplitude against the teacher's 0.49 in one sample, pinned at the pitch
tracker's 500 Hz ceiling in another), and the flow emits **loud unvoiced noise** at the right length and
(after level matching) the right envelope. Neither is speech, and the mel proxy rewarded both. That is
the missing acceptance criterion, and it is now a first-class metric:

* `parakeet.eval.metrics.speechlikeness()` reports voiced fraction, median F0 and spectral flatness with a
  `speech_like` verdict that rejects both a buzz and noise — a buzz has a pitch pinned at one frequency,
  noise is flat. `real_eval.py` prints it for the student **and the teacher** and records it in the
  report; `tests/test_speechlikeness.py` pins the three cases (harmonic tone, white noise, ceiling buzz).

### Two more measurements from the same round

* **the level was distorting every number.** The flow's raw waveforms peaked at up to **55× full scale**
  (the autoencoder's magnitude head exponentiates, `exp(log_mag)`, clamped only at 8) while the
  references it is compared against are peak-normalised — so WER and DNSMOS were reading clipping as much
  as the model. `Synthesizer.synthesize` now peak-normalises (recording the gain) and `real_eval` reports
  the correction. With level matching, the flow's mel cosine rises from 0.75–0.79 to **0.90**.
* **the sampled latents are in-distribution**, so the gain is the decoder's, not the sampler's: std
  0.788–0.791 against the teacher's 1.083, absmax 3.3–3.5 against 5.48. Two candidate levers were ruled
  out by measurement rather than argument: guidance (`cfg_scale` 1.0/1.5/3.0 leaves the latent std at
  0.788/0.791/0.789) and reference conditioning (supplying a partner mel of the same voice does change
  the waveform — verified — and changes the result not at all: WER 1.000 either way).

### Where this leaves the objective

The acoustic path is proven perfect (round-trip WER **0.0**) and the text→acoustic stage is now known to
produce *non-speech* by two different routes with two different signatures. That is a far more useful
statement than "WER 1.0": it says what to look at next (voicing and formant structure, not more steps of
envelope matching) and it stops the project being fooled by its own proxies again.

## 41. The root cause: every run decoded in the wrong latent space (round 36)

Round 35 established that neither path produces speech (a buzz, and noise) while the autoencoder round
trip transcribes at **WER 0.0**. This round found why, and it is a single convention bug with a measured
**WER 1.000 → 0.000** effect.

### What the cache holds, and what the decoder was given

`real_train_demo.py` fits a `LatentNormalizer` on the corpus and passes it to the cache builder, which
stores `normalize(latent)` — so **every cached latent is normalised**. Undoing that is the job of
`LatentNormalizer.denormalize`, and the code does call it (`decoder_latent_from_tokens` and
`ParakeetFlow.synthesize` both do). What was missing is the **statistics**: the fitted normaliser lived
for one process and was never saved, so anything loading a checkpoint later got `mean 0 / var 1` — the
identity — or stale numbers from an older fit.

Recovered from the data (encode the corpus with the trained autoencoder and take per-dimension
mean/variance), the transform is exact:

| | |
|---|---|
| per-dimension correlation(raw encoder output, cached latent) | **1.000** |
| residual after `raw = cached * sqrt(var + eps) + mean` | **0.0000** |
| per-dimension mean | −0.99 … +1.05 |
| per-dimension variance | 0.109 … 1.341 |

and decoding the same utterance three ways through the *same* frozen decoder:

| latent fed to the decoder | speech-like | WER |
|---|---|---|
| `encoder(teacher audio)` | yes | **0.000** |
| the cached latent, as stored | yes (voiced 0.64) | **1.000** |
| the cached latent with the inverse applied | yes | **0.000** |

### Which models it corrupted, measured per checkpoint

| checkpoint | `latent_norm.n` | mean abs | var mean | decoded space |
|---|---|---|---|---|
| autoencoder | 0 | 0.0000 | 1.0000 | identity (unused) |
| **flow** step 400/800/1600 | **0** | 0.0000 | 1.0000 | **normalised ✗** |
| Tiny `mixed_v2`/`calibrated_v2`/`contrast_audio` | 4 | 1.43 | 1.90 | **stale ✗** |
| Tiny `gutenberg_audio` | 4 | 0.807 | 1.408 | stale ✗ |
| the cache's true statistics | — | −0.08 (range ±1) | **0.437** | correct |

So the flow decoded normalised latents (identity), and the Tiny path — which *did* have non-zero
statistics — used values from an **earlier fit** whose variance (1.90) is four times the cache's true 0.437.
Both paths trained and synthesised against a mis-scaled, mis-shifted decoder input, and the models
compensated as best they could (a near-silent buzz, and noise).

### The fix

* `build_latent_cache` now **persists** the normaliser statistics in `cache_meta.json`;
* `scripts/repair_latent_norm.py` recovers them for caches built before the fix (which is exactly the
  measurement above) and merges them in;
* `train.common.load_latent_norm_from_cache()` installs them into a model, and **both training and
  evaluation** call it — a cache without statistics now says so loudly instead of decoding silently;
* `tests/test_latent_convention.py` pins the invariant (a fitted normaliser round-trips, an unfitted one
  does **not** undo a fitted transform), the persistence, the missing-statistics case and the
  dimension-mismatch refusal.

### Honest status of the correction

A 5 000-step Tiny run was restarted in the corrected space with every earlier fix (calibration, contrast
objective, aligned schedule). At step 1 000–1 500 it is **still not speech-like** (voiced 0.20, flatness
0.52, WER 1.000) at 59× real time. The convention fix is necessary and its effect is unambiguous on a
*real* latent (WER 0.000), but the models have to be retrained to benefit, and that is running.

## 42. How good must the latent be? (round 37) — and a correction to my own gate

The fit diagnosis reports a per-dimension correlation between predicted and teacher latents (0.13–0.38 so
far) and WER 1.000, but nothing connected the two. This manufactures latents with a *known* correlation
against a real one and decodes them through the same frozen decoder:

    latent = rho * standardised(real) + sqrt(1 - rho^2) * noise      (per dimension)

| rho | 1.00 | 0.90 | **0.75** | **0.60** | 0.45 | 0.30 | 0.15 |
|---|---|---|---|---|---|---|---|
| WER | **0.000** | **0.000** | **0.000** | **1.000** | 1.000 | 1.000 | 1.000 |

**The recogniser reads the sentence perfectly at rho ≥ 0.75 and fails at 0.60.** That is the first
quantified target this project has had for the text side, and it reframes the work.

The small text side (4.6M trainable), trained *purely* on the regression with no decoder in the loop,
runs out of slope well before it:

| step | train correlation | validation correlation | duration ratio | F0 MAE |
|---|---|---|---|---|
| 3 000 | 0.285 | 0.118 | 0.767 | 0.108 |
| 6 000 | 0.343 | 0.212 | 0.802 | 0.100 |
| 9 000 | 0.375 | 0.245 | 0.814 | 0.099 |
| 12 000 | **0.387** | **0.251** | 0.817 | 0.098 |

The increments per 3 000 steps are +0.058, +0.032, +0.012: it is **saturating at ~0.39 train / 0.25
validation — 52 % and 33 % of the target**. "More steps" is therefore ruled out as the answer, which is
what makes capacity the decisive question rather than an optional one.

### My speech-likeness gate was rejecting good audio

The same experiment exposed a mistake in round 35's gate: at rho = 1.00 — the *real* latent, WER
**0.000** — the gate said **not speech**, because it required spectral flatness < 0.5 and the
autoencoder's decoded output sits at **0.51**. The thresholds are now calibrated on measured anchors:

| anchor | voiced | flatness | must be |
|---|---|---|---|
| the teacher's own audio | 0.67 | 0.30 | speech |
| **the autoencoder round trip of a real latent** (WER 0.000) | 0.50 | **0.51** | **speech** |
| the Tiny path's buzz | 0.96 | 0.32 | not speech |
| the flow's noise | 0.07 | 0.55 | not speech |

with `speech_like = 0.25 ≤ voiced ≤ 0.9 and 70 ≤ F0 ≤ 350 and flatness < 0.65`. The round-35 conclusion
survives — both trained paths are still rejected, now for voicing rather than flatness — and the real
anchor is a **test** (`test_the_real_autoencoder_round_trip_passes_the_gate`), so a regression in either
the gate or the latent convention fails the suite instead of quietly changing a verdict.

### In flight

A 12 000-step `distill-text` run with **text.dim 512 / 6 layers (22.6M trainable against 4.6M)**, same
stage and steps as the 0.375 run, to separate capacity from step budget. Its first checkpoint had not
landed by the end of the round.

## 43. Is ρ ≥ 0.75 attainable? What the target actually contains (round 38)

Before blaming capacity for the 0.39 plateau, the *target* has to be checked: text does not determine
acoustics uniquely. The operator's eight takes are ideal material — the same paragraph read by eight
voices — so `scripts/cross_voice_variation.py` encodes them with the trained autoencoder and, for pairs of
voices carrying the same text, measures the per-dimension correlation between their latents:

| comparison | per-dim correlation |
|---|---|
| **same text, different voice** | **+0.172** (n=52, range −0.13…+0.53) |
| different text, same voice (control) | +0.028 (n=53) |

The control confirms the encoder's latents are content-driven (unrelated text ⇒ ≈0), and the first row
says most of the per-*frame* latent structure is **voice and rendition, not text**. Two consequences:

* a single text→latent predictor serving *many* voices is being asked for a one-to-many mapping, and the
  measured cross-voice agreement (0.17) is far below the 0.75 that intelligibility needs. That is an
  argument for the architecture the papers use — predict the text-determined part, **sample** the rest —
  rather than for a bigger regressor;
* the *within*-voice ceiling is **not** measured here, and the honest reading is bounded by that: the
  corpus contains no repeated renditions by one voice (Kokoro 1 voice, Speechify 3 voices, the takes 8
  voices of one paragraph), so "how well could a fixed-voice model do" remains open. What is measured is
  that *cross-voice* generalisation of the fine structure is ~0.17.

## 44. Capacity: it helps, and it is not the ceiling (round 38)

Same stage (`distill-text`), same corpus, same step count — only the text side differs:

| text side | trainable | train corr | val corr | duration ratio | F0 MAE | latent cosine |
|---|---|---|---|---|---|---|
| dim 256, 4 layers | 4.6M | 0.285 | 0.118 | 0.767 | 0.108 | 0.849 |
| **dim 512, 6 layers** | **22.6M** | **0.321** | **0.190** | 0.764 | 0.113 | 0.853 |

Capacity helps — **+13 % on training correlation and +61 % on validation**, so it buys generalisation more
than fitting — and it is nowhere near the 0.75 the threshold experiment measured. Combined with the
cross-voice result (two voices reading the *same text* agree at only 0.172), the reading is coherent: the
per-frame latent target carries a great deal of voice and rendition detail, so a deterministic text
regressor is fitting a one-to-many mapping, and a wider regressor does not change that. That is what the
papers' split exists for: predict what the text determines, **sample** the rest.

The `ParakeetFlow` run was therefore restarted **in the corrected latent space** (6 000 steps, checkpoint
every 1 000) — its earlier runs used the identity normaliser, so this is the first flow training that
decodes where the decoder expects. Verified before spending the hours: the flow's normaliser reads
`var 1.0000` after the warm start and `0.4368` after loading the cache's statistics, matching the cache
exactly.

### One more shape mismatch, now closed as a class

`fit_diagnosis.py` crashed loading the 22.6M checkpoint because it built the model from the config
(`text.dim 256`) — the **sixth** time a script has hit this class (`n_voices`, `latent_rate`, `text.dim`,
`text.n_layers`). `train.common.apply_checkpoint_geometry()` now reads all four off the checkpoint's own
tensors and configures the config before building, and it prints what it applied.

## 45. The flow's samples are uncorrelated with the target — and the new metric says so (round 39)

Round 37 calibrated a predictive metric (decoded latents are read perfectly at ρ ≥ 0.75, fail at 0.60).
Applying it to the flow's *sampled* latents — the per-dimension correlation against the teacher's latents
for the same text — gives the sharpest result of the project so far. The old (identity-space) flow:

| checkpoint | length ratio | **log-mel cosine** | **sampled latent ρ** | WER (control) |
|---|---|---|---|---|
| step 400 | 1.41 | 0.9018 | **0.004** | 1.000 (0.276) |
| step 800 | 1.46 | 0.9024 | **−0.006** | 1.000 (0.079) |
| step 1 600 | 1.52 | 0.8932 | **0.019** | 1.000 (0.079) |

**ρ ≈ 0 with a mel cosine of 0.90.** The flow learned the *marginal* distribution — which is why its
latents looked in-distribution (std 0.79 against the teacher's 1.08) and its envelope plausible — and
nothing at all about which latent belongs to which text. It has been sampling from the prior.

That is exactly the failure the audio proxy cannot see and the calibrated metric catches: ρ ≈ 0 predicts
the measured WER of 1.000, while the mel cosine said 0.90. From here the flow's progress is measured by
**ρ**, with 0.75 as the target, and `flow_trajectory.py` reports it for every checkpoint via
`--latent-cache`.

## 46. The flow's conditioning is *weak*, not broken — measured (round 39)

ρ ≈ 0 for the flow's samples could mean "not trained yet" or "never wired". Measurements separate them:

| probe | result |
|---|---|
| **untrained** model, velocity for two *different texts*, same noise and timestep | ρ = **+0.9968** |
| trained (step 1 000) model, same probe | ρ = **+0.9994** |
| trained model, conditioned vs `drop_cond=True` | ρ = **+0.9981** |
| sampled latents for two different texts (same seed) | ρ = **+0.998…+0.999** |
| cross-attention contribution vs the conv branch, at initialisation | 0.063 vs 0.604 → **ratio 0.10** |

An **untrained** network proving insensitive to its conditioning rules out "needs more training" as the
explanation for the *wiring*, and the last row says why: the cross-attention enters at about **a tenth of
the local conv branch's strength**, so ~90 % of the velocity is text-independent from the start. The model
is not ignoring the text — its own prior is drowning it out, which is also what the cross-voice
measurement (0.172) says about the target: most frame-latent detail is *not* determined by the text.

That makes the next change concrete rather than speculative: **strengthen the conditioning path** — the
pooled text/style embedding concatenated onto the estimator's input alongside cross-attention, and a cross
branch scaled comparably to the conv branch — rather than adding steps.

## 47. The flow optimises its objective without learning the mapping (round 40)

With the latent convention fixed, the conditioning strengthened and a calibrated metric in hand, the flow
was pushed through the sharpest tests available on a CPU:

| measurement | result | reading |
|---|---|---|
| flow-matching loss, fresh → 500 → 1 000 steps | 2.005 → 1.477 → **0.890** | it *is* learning |
| sampled-latent ρ at those steps | 0.002 / 0.006 | uncorrelated with the target |
| ρ vs sampler NFE (4 / 16 / 32 / 64) | 0.0021 / 0.0021 / 0.0021 / **0.0022** | the sampler is converged, not the limitation |
| overfit **12 utterances**, 400 steps (loss 1.96 → 0.75) | ρ = **−0.003** | memorising is not happening either |
| overfit 12 utterances with cross-gain **60** | ρ = **−0.005** | conditioning *strength* is not the blocker |
| overfit **2 utterances**, 600 steps, correct schedule (loss 2.0 → 0.45) | ρ = **−0.013** | the sharpest test: still no mapping |

The loss falls steadily in every configuration while the correlation between what the model samples and
what the teacher produced stays at zero. That is the signature of learning the **marginal** velocity
field — the data's average flow — which reduces the objective substantially and carries no information
about *which* latent belongs to *which* text. Sampling from it gives plausible-looking, wrong audio,
which is precisely what rounds 35–39 measured from the outside.

Two incidental things this round, both worth keeping:

* **`overfit_flow.py`** — a reusable harness that trains on a handful of cache items and reports ρ before
  and after, so "can this architecture learn the mapping at all?" is a ten-minute question instead of a
  multi-hour one;
* the harness's *first* run was invalid and said so quietly: calling `run_stage` directly skipped the
  `--steps` schedule alignment from round 32, so the learning rate sat at ~0 for the whole run (`lr 0.0`
  at step 250 in its own log). Fixed in the harness; every number above is from the corrected run.

### What this means for the objective

The diagnostics are calibrated (ρ target, speech-likeness gate), several real defects are fixed (latent
convention, length initialisation, conditioning strength, schedule alignment), and what remains is
**training at a scale this CPU cannot reach** — a few thousand steps against the orders of magnitude the
papers use. That is the case for the operator's WSL2/ROCm work, and the code is ready for it:
`resolve_device` already selects CUDA/ROCm, the training path is plain fp32 with no CUDA-specific AMP
assumptions, and the per-utterance host round-trip that existed for DirectML is now conditional.

## 48. Correction: the architecture *can* learn the mapping — it needs the steps (round 41)

Round 40 concluded from the overfit tests that the flow optimises its objective without learning the
mapping, resting on the sharpest test (2 utterances, 600 steps). Extending that same test to **3 000
steps** with a higher learning rate changes the conclusion:

| 2 utterances | flow loss | sampled-latent ρ |
|---|---|---|
| 600 steps, lr 2e-4 | 2.00 → 0.455 | **−0.013** |
| **3 000 steps, lr 1e-3** | 2.00 → **0.064** | **+0.2495** |

The conditioned solution *does* emerge — the model first fits the marginal velocity field, which dominates
the early loss, and only later encodes *which* latent belongs to *which* text. Round 40's reading ("it does
not learn, even when memorising") was an artefact of the step budget, not of the architecture.

Two negative checks in this round make that trustworthy, and both are worth recording:

* **the ρ measurement is not an alignment artefact**: forcing the sample to the target's length gives
  ρ = −0.021, and a shift search over ±60 frames gives +0.030 at the best offset, against −0.000 as
  measured. A time shift is not hiding a correct mapping;
* **conditioning strength is not the lever**: a cross-attention gain of 60 produced ρ = −0.005 at 400
  steps, no better than the default.

### What this changes about the plan

The requirement is now quantified rather than asserted: **2 utterances reach ρ 0.25 after 3 000 steps**
against the **0.75** intelligibility needs, so the full corpus (1 207 utterances, 12 voices) needs a
training scale beyond what this CPU can reach — which is the case for the operator's ROCm work.
`docs/GPU.md` carries the commands and the four criteria to judge it by (ρ ≥ 0.75 first, then
speech-likeness, then WER with its control, then fit on the training prompts).

## 49. Aligned crops: implemented, tested — and not yet shown to help (round 42)

Training the flow on whole utterances spends each step on one long sequence, while what has to be learned
— which latent belongs to which text — needs *variety* per step. The standard answer is random crops, and
this project can do it properly because every cache item carries per-token frame counts, so a frame window
maps to a token span: `crop_item_to_tokens()` cuts **text and acoustics together** (keeping the whole
sentence while cutting the audio would teach the model to predict an arbitrary part of an utterance from
all of it). Six tests pin the invariant (cropped tokens cover the cropped frames within one token), the
per-token slicing (`latent_token` is `(T, features)` — the wrong axis would cut the features), the
in-bounds clamp, and the degenerate cases.

Measured on the only task small enough to run on this CPU — overfitting two utterances:

| 3 000 steps, lr 1e-3, 2 utterances | wall time | flow loss | sampled-latent ρ |
|---|---|---|---|
| whole utterances | ~30 min | 0.064 | **+0.2495** |
| 160-frame aligned crops | **10.2 min** | 0.194 | **+0.0548** |

So crops are **3× cheaper per step** and **worse per step** on this task: with two items a whole utterance
is the more informative sample and a 160-frame window is not. The efficiency argument for crops is about
*variety across a large corpus*, which this test cannot show. Both numbers are recorded rather than the
convenient one — the feature stays (tested, off by default, and the right thing for a full-corpus run),
with no claim the measurement does not support.

## 50. "Light and fast", measured (round 42)

The objective asks for a small, lightning-fast model. Those two properties can be measured even while the
speech quality is unresolved, and they are:

| Tiny (9.63M params) | storage | synthesis speed | output |
|---|---|---|---|
| fp32 | 38.5 MB (36.8 MB state dict) | **42× real time** (74 ms for 2.9 s) | — |
| int8 dynamic (`quantize_dynamic_int8`) | **20.3 MB** (55 %) | **85× real time** (34 ms) | waveform cosine **0.9952** vs fp32 |
| int8 weights + fp16 scales (export grid) | 12.0 MB (`size_report`'s own accounting) | — | — |

Two honest notes: `quantize_weights_` is **simulated** quantisation (it snaps fp32 values to the int8 grid
for quality studies, which is why a naive state-dict size check shows no saving — real storage needs
`quantize_dynamic_int8`, as measured above); and every number here describes a model whose **speech is not
yet intelligible**, so this demonstrates the deployment path, not a usable system.

**Speed reference for the flow path**: the sampler runs at **27–37× real time** at NFE 4 on this CPU
(`flow_trajectory`'s `x_realtime`), which satisfies "lightning fast" — the missing piece is quality, not
latency.

## 51. Optimisation levers tested — and one piece of de-risking that worked (round 43)

The flow learns the marginal velocity field first and the conditioning later (round 41), so levers that
speed up the *conditional* part are worth testing, and CPU-sized tests exist for them.

**1. Warm-starting the text encoder — no effect.** The text encoder (dim 256, 4 layers) is shared between
the Tiny variant and the flow, so the text side already trained to regress latents (`runs/text_only`,
12 000 steps) can be loaded into the flow, giving its cross-attention a meaningful memory from step 0:

| 2 utterances, 3 000 steps, lr 1e-3 | flow loss | sampled-latent ρ |
|---|---|---|
| baseline (random text encoder) | 0.064 | **+0.2495** |
| text encoder warm-started | 0.065 | **+0.2466** |

Identical within noise. On a two-item task the objective is dominated by fitting the two latent sequences
rather than by the quality of the text representation, so this test *cannot* show a benefit — and claiming
one would be dishonest.

**2. Logit-normal timestep sampling — lower loss, same mapping.** Uniform t spends most samples near the
noise end where the conditional information is weakest; SD3's logit-normal concentrates t around 0.5.
`sample_timesteps` already implemented the mode, but `flow_loss` hard-coded `"uniform"` so it had never
been used; it is now wired through `FlowConfig.t_sampling` (default unchanged).

| 2 utterances, 3 000 steps, lr 1e-3 | flow loss | sampled-latent ρ |
|---|---|---|
| uniform t | 0.064 | **+0.2495** |
| logit-normal t | **0.023** | +0.2304 |
| text encoder warm-started | 0.065 | +0.2466 |

Logit-normal cuts the flow-matching loss by **~3×** and does not improve the mapping at all. That is the
round's most useful lesson: **the loss and the conditional mapping are decoupled**, because the loss is
dominated by the marginal velocity field, which carries no text information. Every "the loss is falling"
observation in rounds 40–43 is therefore *not* evidence of progress; ρ is the only number that is.

### The only lever that worked was steps

Across four configurations at identical step counts (2 utterances, 3 000 steps, lr 1e-3) ρ stays in
0.23–0.25. The one thing that moved it was **more steps** — 600 steps gave −0.013 and 3 000 gave +0.25
(round 41) — while cropping, the conditioning gain, warm-starting the text encoder and logit-normal
timesteps all leave it where it is. The requirement is compute, and `docs/GPU.md` is the response to it.

**3. De-risking that worked: the flow stage verifiably resumes.** A long GPU run must survive
interruption, so this was tested rather than assumed — 40 steps with checkpoints every 20, then
`--resume flow_step20.pt` to step 80:

```
[stage flow] resumed from runs/resume_check/flow_step20.pt at step 20 (lr 8.400e-05)
final log: step 80, resumed_from 20.0, nonfinite_steps 0
checkpoints: flow_step20.pt, flow_step40.pt, flow_step80.pt
```

Optimizer, EMA, schedule position and the step counter all continue correctly.

## 52. The flow's first train/holdout measurement (round 44)

Every flow number until now came either from memorising two utterances or from the end of a short run on
the whole corpus, so nothing said whether ρ *generalises*. `overfit_flow.py --holdout N` trains on
`[0, items)` and scores a range that is never trained on, and `rho_curve.py` walks a run's checkpoints to
produce ρ against steps for both.

First datapoint (200 trained utterances, 32 held out, 160-frame aligned crops, batch 16, lr 5e-4):

| step | ρ trained | ρ held-out |
|---|---|---|
| 2 250 | **−0.0405** | +0.0264 |
| 4 500 | **+0.0098** | **+0.0381** |
| 6 750 | **+0.0616** | −0.0144 |

ρ_trained rises steadily — **+0.05 per 2 250 steps, close to linear** — while ρ_held-out stays at zero
within noise (12 probe items put the noise at roughly ±0.03, so +0.026 / +0.038 / −0.014 is one flat line).
Read together: the flow is beginning to **fit the training utterances without generalising to unseen
text**. That is the phenomenon the cross-voice measurement predicted (0.172 agreement for the same text
across voices): most frame-latent detail is not determined by the text, so fitting specific utterances is
easier than predicting new ones.

### What the curve implies for CPU-only training

Reading ρ_trained as linear, 0.75 would need on the order of **150 000 steps ≈ 38 hours** of continuous CPU
at the measured ~1.1 steps/s with crops. But the number that matters is ρ_held-out, and it is not moving —
so the honest position is that **more time on this data scale buys memorisation, not generalisation**. Two
things could change that, and both are cheap to test relative to a 40-hour run:

* **more data per step** — 200 utterances can be memorised; 1 175 cannot, which is the standard remedy for
  exactly this shape of curve;
* **the target itself** — if the frame-latent detail is not text-determined (the 0.172 cross-voice result),
  then the flow can only learn the text-determined part, and ρ would saturate well below 0.75 no matter how
  long it trains.

The 9 000-step point and then a long full-corpus run are what separate those.

## 53. The long run, and how to operate it (round 46)

Since the decision was CPU-only and the curve said *more data per step* is the cheap remedy for
memorising without generalising, the run that is now training uses the **whole corpus** rather than 200
utterances:

```powershell
.\.venv\Scripts\python.exe -u scripts\overfit_flow.py --items 1175 --holdout 32 `
    --steps 200000 --batch-size 16 --crop-frames 160 --lr 0.0005 --save-every 2500 `
    --out runs\flow_long > runs\flow_long\train.log 2>&1
```

* 1 175 training utterances, **32 held out** (ρ is reported for both, so generalisation is measured, not
  assumed);
* 160-frame aligned crops, batch 16 — the throughput setting: ~1.1 steps/s, so 200 000 steps is roughly
  **50 hours** on this machine;
* checkpoints every 2 500 steps (~40 min), which is the unit to read progress in.

Monitoring, in one command each:

```powershell
Get-Content runs\flow_long\train.log -Tail 5           # loss per 50 steps, and the schedule position
python scripts\rho_curve.py --run runs\flow_long --items 1175 --holdout 32   # rho vs steps, both splits
```

The first log lines, from before the run started:

```
before training: sampled latent rho on 24 of the 1175 training utterances = -0.0027
holdout: 32 utterances never trained on (rho before = +0.0002)
schedule: 200000 steps with 1000 warmup at lr 0.0005
{'step': 0, 'trainable_params': 16132310.0, ...}
{'flow': 2.0281, 'length': 1.8263, 'loss': 3.8544, 'lr': 0.0, 'step': 50}
{'flow': 1.4192, 'length': 0.2452, 'loss': 1.6643, 'lr': 0.0, 'step': 100}
{'flow': 0.8990, 'length': 0.0001, 'loss': 0.8991, 'lr': 0.0001, 'step': 250}
```

Note the length term collapsing to ~1e-4 within 250 steps — the bias initialisation from round 32 doing
its job — while the flow term falls the way the marginal solution falls. **ρ, not the loss, is the
progress metric** (round 43): the loss falling here is expected and proves nothing until ρ moves.

The run is `python -u` on purpose: without `-u`, Python buffers stdout when it is redirected to a file, so
a multi-day run would show an empty log until it exited. `--save-every 2500` replaces the harness's
quarter-of-the-run default, which for a 200 000-step run would have been a single 50-hour checkpoint.



## 53.1. Smoke test output (measured)

> Renumbered from a duplicate `## 53.` found by an audit in round 74, along with the orphaned
> `## 62. Environment notes` at the end of this file. Both are the same failure mode as the roadmap's: an
> anchor-based shell replacement that matched nothing, so the renumbering reported in several rounds never
> happened -- the heading it targeted says `62`, not `80`. Section numbers 35 and 38 have never existed; that
> is a historical gap, left as is rather than renumber twenty-five sections and break the cross-references
> that cite them.

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

## 54. Coarse-to-fine plan conditioning — the papers' split, finally implemented (round 47)

The measurements had converged on one diagnosis: the flow fits the *marginal* velocity field while its
**text** conditioning stays unused (an untrained field was already indifferent to which text it was given,
round 39; ρ_trained climbs while ρ_held-out stays at zero, round 46). Raw characters are a thin signal for
frame-level acoustics, and every paper in the objective splits the problem for exactly that reason:
predict the text-determined content, *sample* the rest.

`cfg.flow.use_plan` implements it:

* `ParakeetFlow.plan_head` predicts a **token-level acoustic plan** from the text memory;
* `plan_proj` puts that plan into the conditioning memory, one token per text token, so cross-attention
  reads a predicted acoustic state rather than only characters;
* the plan is **supervised directly** against the cached token latents (`latent_token`) with weight
  `flow.plan_weight`, because an unstructured extra conditioning token is precisely what round 39 measured
  the field learning to skip.

It is **off by default**, so the baseline is bit-identical, and it is +78k parameters (16.21M trainable
against 16.13M). Four tests pin it: off-by-default adds no parameters and no memory tokens; the plan widens
the memory by exactly one token per text token and is not masked out; gradients reach the plan head; and the
plan loss falls ≥30 % over 40 steps on a fixed batch.

**The plan learns the coarse acoustics almost immediately** — the smoke run's own log, 2 utterances:

```
{'flow': 1.5133, 'length': 0.0014, 'plan': 0.2139, 'loss': 1.5147, 'lr': 0.0009, 'step': 50}
{'flow': 0.9168, 'length': 0.0000, 'plan': 0.0074, 'loss': 0.9168, 'lr': 0.0006, 'step': 100}
{'flow': 0.6808, 'length': 0.0000, 'plan': 0.0018, 'loss': 0.6808, 'lr': 0.0001, 'step': 150}
```

100× reduction in 150 steps, against ρ ≈ 0 for the flow term over thousands. So the plan is a *learnable*
coarse signal where raw text was not — which is why the long run was switched to it: 20 minutes of the
plain run were traded for an architecture the measurements support, and its log is kept as the baseline
(`docs/evidence/flow_long_plain_baseline.log`).

Careful about what this does *not* show: the plan **learning** is not evidence that the flow's ρ improves.
That A/B is still unmeasured, and until it is, the honest claim is "the coarse signal is learnable", not
"this fixes the mapping".

### The A/B, measured (same task, same budget)

| 2 utterances, 3 000 steps, lr 1e-3 | flow loss | sampled-latent ρ |
|---|---|---|
| plain text conditioning | 0.064 | **0.2495** |
| **plan conditioning** | 0.064 | **0.2752** |

A **+10 % relative** improvement in ρ (0.2495 → 0.2752). Real, in the direction the plan's own loss
predicted, and *small*: it does not close a gap of 0.75. The honest verdict is that the coarse signal helps
a little and the binding constraint is still the training scale — the plan is in the long run because it is
measurably better, not because it is the papers' shape, and it is not the breakthrough.

One caveat on the comparison: both runs used the same data and step budget, but model initialisation is not
seeded by the harness, so part of the +0.026 could be run-to-run variation. It is recorded as a modest
effect, not a decisive one.

## 55. Wall time is not compute time on this machine (round 48)

A round's worth of wall clock passed with only ~50 minutes of CPU accumulated: **the machine sleeps**, so
elapsed time is a misleading unit here. Every estimate in this project should be read in **CPU-minutes**,
and the long run's budget is ~50 *CPU*-hours — which may be several days of wall clock if the machine is
idle part of the time. The logs carry the step counter, so progress never depends on the clock:

```powershell
Get-Content runs\flow_long_plan\train.log -Tail 3
```

## 56. The flow has learned *what latents look like*, not *which one to make* (round 49)

The plan checkpoint's synthesised audio has a median F0 of **93.7486 Hz** — exactly the 93.75 Hz latent
frame rate. The natural reading is "the samples are temporally white, so the decoder emits a pulse train at
the frame rate". That reading is **wrong**, and measuring it took one script:

| autocorrelation lag | teacher target | flow sample |
|---|---|---|
| 1 | 0.9011 | **0.8853** |
| 2 | 0.7375 | 0.7951 |
| 4 | 0.4851 | 0.6165 |
| 8 | 0.2048 | 0.3993 |
| 16 | 0.1224 | 0.2297 |

std: target 1.048, sample 0.809.

The samples are **smooth, and slightly smoother than the teacher** — the field has learned the marginal
*temporal process* of real latents quite well. What it has not learned is *which* latent belongs to *which*
text (ρ ≈ 0). So the failure has a precise description:

> **p(latent) is learned; p(latent | text) is not.**

That is consistent with everything else measured — the loss falling as the marginal is fitted (round 40),
the untrained field being indifferent to its text (round 39), the plan adding only +10 % (round 48) — and it
is the reason the audio is acoustically plausible and linguistically empty.

Two consequences worth keeping:

* the ρ measurement (value agreement per dimension) and this one (temporal structure) answer different
  questions, and both are needed: ρ ≈ 0 with teacher-like autocorrelation is a *different* failure from
  ρ ≈ 0 with white samples, and only the second would be a sampler or scale bug;
* `scripts/latent_autocorrelation.py` keeps the distinction checkable, and its docstring records the
  refuted hypothesis rather than quietly dropping it.

## 57. The residual form: principled, tested, and no better — which also questions the +10 % (round 50)

Round 49's diagnosis said p(latent | text) is the missing factor. The principled answer is to make the plan
*necessary*: instead of appending it as extra conditioning, have the field model the **residual**
`x1 − upsample(plan)`, so the target depends on the plan and the plan depends on the text — the objective
can no longer be satisfied by the marginal distribution alone. Implemented for training, one-shot synthesis
and blockwise streaming (where the plan is spread over the whole utterance once and sliced per block, so
streaming cannot silently disagree with offline).

Then measured, on the same task and budget as the previous two:

| 2 utterances, 3 000 steps, lr 1e-3 | flow loss | sampled-latent ρ |
|---|---|---|
| no plan | 0.064 | 0.2495 |
| advisory plan (memory token) | 0.064 | 0.2752 |
| **residual plan** | 0.082 | **0.2511** |

**All three are indistinguishable.** The residual form's loss is *lower* in early training purely because a
residual is a smaller target — not evidence of anything (the round 43 lesson, again).

The important consequence is what this does to the earlier result: the advisory plan's "+10 %" (0.2752 vs
0.2495) is **the same size as the spread between three variants that should differ**, on a task of two
utterances. The honest reading is that **the plan is not shown to help**, not that it helps by 10 %, and the
earlier claim is corrected here rather than left standing.

### A near-miss worth recording

Gating the residual behind `flow.plan_residual` was done in three places (loss, one-shot, streaming). A
failed edit left the two synthesis paths ungated, so in advisory mode the code would have *added a plan the
model never learned to subtract* — a training/inference mismatch that no metric would explain, and that the
test suite would have passed because the only synthesis test requested residual mode. It was caught by
reading the diff rather than by a red test; there is now a test that advisory mode does not add the plan
back, since that is exactly the kind of silent corruption that costs days.

### What was chosen and why

Both variants are exposed (`flow.plan_residual`, default **false**). With no measurable difference, the
default is the **advisory** form, whose worst case is "no better than plain" rather than "the target is
corrupted while the plan is still bad". The long run was restarted on it.

## 58. The text *is* in the objective — it contributes 14 % (round 51)

Rounds 39–49 inferred the conditioning's weakness from *samples*: an untrained field indifferent to its text,
ρ ≈ 0 with teacher-like temporal statistics. That leaves a question those measurements cannot answer — is
the text in the **objective** at all, or is it being ignored by construction?

`scripts/text_dependence.py` answers it directly with a shuffled-text control: pair each utterance's
acoustics with a **different utterance's text** and see whether the loss moves. On the 3 000-step plan
checkpoint:

```
flow loss with the RIGHT text     : 1.0642
flow loss with a SHUFFLED text    : 1.2146
difference                        : +0.1505 (+14.1% of the loss)
  -> the objective does depend on the text; the gap is optimisation or scale
```

So the conditioning path works and the text is genuinely part of the objective — the failure is not a
disconnected signal. It is a **magnitude** problem, and now it has a number: **the text accounts for ~14 %
of the loss**, while ρ against the right latent is ≈ 0. The model is extracting a little from the text and
the marginal distribution supplies the rest — which is a precise restatement of "p(latent) learned,
p(latent | text) not", with a quantity attached that can be tracked across checkpoints instead of inferred.

That also makes this a better *progress* indicator than the loss: a run that is learning the conditioning
should push the shuffled-text gap **up**, while the total loss can keep falling purely on the marginal.

One honest note on the gradient table this script prints: it shows **no gradient for the length
predictor**, which is an artefact of the probe calling `flow_loss` directly — the length term lives in
`stage_flow`, not in `flow_loss`. It is not a finding about the model, and the script says so where it
matters rather than leaving a misleading zero in a table.

## 59. The token bottleneck is nearly LOSSLESS — and a published claim is retracted (round 52)

**This section replaces a wrong claim made earlier in the same round.** The first version of this
measurement reported that the token bottleneck retains only ρ 0.118 of the frame latent, and concluded that
the Tiny route was capped at ~0.12 "in principle, no matter how well it is trained". Both were artefacts of
the probe, and the corrected measurement says the opposite.

What the probe does: feed the pipeline the **true** token latents, expand them back to frame rate with the
model's own geometry, and correlate against the true frame latents. That is the ceiling for any
`text → token → frame` route, and it needs no prediction.

| mean of 8 utterances | faithful geometry (correct) | crude inverse (wrong) |
|---|---|---|
| oracle ρ | **+0.9507** (range 0.933 … 0.976) | +0.1175 |
| different utterance's tokens | +0.0630 | +0.0134 |

**The token bottleneck preserves ρ = 0.95 of the frame latent.** It is a 2.1× compression (2250 → 1075
dims/s) and it is nearly lossless, which is exactly what a 2.1× compression of a smooth signal should be.

Two bugs produced the retracted number, and both are worth naming because they are the same class of mistake:

1. **Collapsing the sub-vectors.** `extract_signals` builds each token as `rate` **sub-span means** and
   inference expands each sub-vector back onto its own span. The first probe averaged the sub-vectors
   together and then stretched the result over the whole token, destroying the within-token detail the
   representation exists to keep.
2. **A transposed reshape.** Those sub-vectors are `torch.cat`ed, so the layout is `(rate, width)`; reshaping
   as `(width, rate)` silently transposes them. This alone took the number from 0.95 to 0.05.

### What this changes

* **The Tiny route is not structurally doomed.** Its measured 0.39 against token latents is a *learning*
  result, and with a near-lossless bottleneck a better text → token model would carry that improvement
  through to frame latents. The route is cheap (one forward pass, no sampling, a quarter of the size) and
  under the CPU-only constraint that makes it worth revisiting rather than discarding;
* **the data pipeline is consistent**, which the crude probe had also called into question;
* the retraction is the point: a measured claim went into a commit and the docs, and a second look at the
  probe — not at the model — showed it was wrong. The lesson is now attached to the tool itself, including
  a comment on the layout, so the next person measuring this cannot repeat it silently.

## 60. Reallocating the CPU: the cheap route is the better bet (round 53)

Round 52's correction — the token bottleneck is nearly lossless (ρ 0.95) — unblocked a comparison the
project could not make before, and it does not favour the flow:

| route | measured ρ | cost |
|---|---|---|
| flow (frame latents from text) | **≈ 0.00 – 0.06** (6 750 steps at 200 utterances; ≈ 0 at 1 175) | ~2 s/step → 200 000 steps ≈ **50 CPU-hours** |
| token route (token latents from text) | **0.190 validation / 0.321 training** per-dim correlation, and the bottleneck carries it at ρ 0.95 | ~0.8 s/step → 50 000 steps ≈ **11 CPU-hours** |

The token route is **20× closer to the 0.75 target and 4.5× cheaper**, and its generalisation gap is a
capacity story that already improved once (val 0.118 → 0.190 when the text side grew from 4.6M to 22.6M,
round 38). So the CPU was moved to it:

```powershell
.\.venv\Scripts\python.exe -u scripts\train.py --config configs\parakeet_tiny.yaml --stage distill-text `
  --cache runs\mixed_v2\latent_cache --warm-start runs\text_big\distill-text_step3000.pt `
  --steps 50000 --batch-size 16 --device cpu `
  --set text.dim=512 --set text.n_layers=6 --set flow.text_dim=512 --set flow.cond_dim=512 `
  --set train.save_every=5000 --out runs\text_long > runs\text_long.log 2>&1
```

* **22.64M trainable**, warm-started from the step-3 000 model with **0 missing / 0 unexpected keys** and a
  fresh optimiser and schedule (verified in the log, not assumed);
* checkpoint every 5 000 steps, so `fit_diagnosis.py` can be run at each and compared with the recorded
  baseline: **train 0.3212 / validation 0.1896** (`docs/evidence/fit_diagnosis_text_big_3000.json`);
* the flow's run is paused with its log and checkpoints kept as a reference point
  (`docs/evidence/flow_long_plan_advisory_baseline.log`), not deleted — if the token route stalls at ~0.2
  while the flow eventually moves, that is worth knowing.

What would make this the right call rather than a hopeful one: **validation ρ climbing past 0.190** at the
5 000/10 000/… checkpoints. If it does not move in 15 000 steps of a 22.6M model, then the text→acoustics
mapping itself is the wall — not the architecture, the bottleneck, the plan or the schedule, all of which
have now been measured and excluded one by one.

## 61. The speech-likeness gate has a false negative on *perfect* audio (round 54)

Measuring the token expansion's quality required decoding a real latent, and the result exposed the metric
rather than the model. Same utterance, three decodes:

| decode | intelligibility | voiced | median F0 | flatness | gate |
|---|---|---|---|---|---|
| direct AE round trip of a real latent | **WER 0.000** | 0.994 | 93.8 Hz | 0.591 | **NO** |
| through the token expansion | — | 1.000 | 93.8 Hz | 0.631 | NO |
| the Tiny model's own output | WER 1.000 | 1.000 | 93.7 Hz | 0.603 | NO |

**The gate rejects audio that transcribes perfectly.** The `voiced ≤ 0.9` bound fails on the autoencoder's
own reconstruction (0.99), and the components overlap almost completely between the intelligible round trip
and the unintelligible model output — so *no threshold on these three numbers separates them*. The gate is a
sanity check on voicing and spectral shape, not a quality verdict, and it now says so in
`speechlikeness`'s own comment (with these numbers), in `real_eval.py`'s output, and here.

Two consequences, one of which is a correction to how this project has been reporting:

* several earlier rounds used "speech_like: NO" as a headline for the Tiny and flow outputs. For the flow's
  *noise* that was informative (voiced 0.07), but for the Tiny path's buzz it was **not evidence of
  anything**, because the round trip gets the same verdict. **WER with its teacher control** was always the
  sound metric, and it remains the one to lead with;
* the token expansion is **not** a quality bottleneck: it matches the direct round trip on every component
  here, consistent with the piecewise-constant design being the deliberate fix for the round-22 seam that
  the code's own docstring describes.

Two refuted hypotheses from this round are recorded rather than dropped: the expansion was suspected of
causing the 93.7 Hz periodicity (it is in both the direct round trip *and* both model routes, so it is the
decoder's response to latents that are smooth but wrong), and a first version of the probe decoded a
**normalised** cache latent without denormalising — the convention that has its own test file, and the third
probe-side error in two rounds. The lesson each time is the same: when a measurement says something
dramatic, check the measurement first.

## 62. The checkpoint-loading class is closed (round 54)

`real_eval.py` built the model from the config and applied only *two* of the four geometry values, so the
22.6M text side (dim 512) failed to load with `size mismatch for latent_head.2.weight` — the **seventh**
occurrence of this class (`n_voices`, `latent_rate`, `text.dim`, `text.n_layers`). Six scripts still had
partial or missing inference.

Rather than patch the seventh one-off, `train.common.load_checkpoint_into(cfg, path, model=None)` now does
the whole load in one place: read the payload, apply the checkpoint's own geometry, build if needed, and
load only shape-compatible tensors. `real_eval.py` and `fit_diagnosis.py` use it; the remaining scripts
that build from a config are listed in the round's notes so they can be migrated the same way.

## 63. The 23 % duration shortfall is not the blocker (round 55)

The Tiny route under-predicts utterance length systematically — `fit_diagnosis` reports a duration ratio of
0.764 and `real_eval` sees generated audio at 0.77 of the reference. A systematic bias is separable from
variance, so it is worth one measurement: `real_eval.py --duration-scale` now passes a multiplier through to
the synthesizer (the parameter already existed there).

| duration scale | length ratio | WER | DNSMOS | voiced |
|---|---|---|---|---|
| 1.00 | 0.810 | **1.000** | 1.262 | 1.000 |
| 1.29 (1 / 0.775) | **1.062** | **1.000** | 1.195 | 1.000 |

The calibration does exactly what it should to the timing — 0.81 → 1.06, i.e. the shortfall was bias, not
variance — and it changes **nothing** about intelligibility, while DNSMOS drifts slightly down. So the
timing error is excluded as a cause: the content is wrong, which is the ρ ≈ 0.19 the token route measures.

The flag is kept: it is the right thing for a *report* to be able to separate timing from content, and a
future checkpoint whose latents are right will still want its durations calibrated.

## 64. Corpus expansion: the curation gate would have eaten the whole allowance (round 55)

The operator approved spending ~100k Speechify characters (of ~397k remaining) on more teacher audio, so the
pipeline was exercised end to end **before** spending anything:

* **New text, no leakage.** The local Gutenberg records are exhausted — with the existing 600 prompts and the
  validation set excluded, only ~155 new characters remain, and barely 5k across every local corpus.
  `scripts/expansion_prompts.py` therefore draws from freshly downloaded public-domain books, with a
  per-book quota so the budget spreads across sources rather than exhausting the first alphabetically, and
  skips any sentence whose text appears in validation. Result: **771 prompts, 100 025 characters**, balanced
  ~25k per book, filtered to real prose (no front matter, headings, markup or all-caps).
* **The curation gate rejects the teacher it is meant to protect.** A 3-prompt smoke test spent ~200
  characters and every record was rejected — `narrowband(4292Hz)` — against
  `min_bandwidth_hz: float = 5000.0  # [proposed]`. The **already-kept** Speechify records in the corpus have
  `bandwidth99` **median 4 799 Hz** (min 4 000, max 7 289): the proposed threshold is stricter than the
  corpus it guards. The existing 903 records never faced it because `curate_manifest()` was dead code when
  they were built (the docstring says so); wiring it in later made the pipeline reject its own training data.
  So the expansion runs with `--no-curate`, consistent with how the corpus was actually built — and the
  default is left alone, because changing a quality gate is a decision for the operator, not a side effect
  of a data run.

The full generation is running (771 prompts) concurrently with training, since API calls are I/O-bound.

Two lessons worth keeping, both cheap: a smoke test that costs 200 characters can save 100 000, and a gate
marked `[proposed]` in the source should be checked against the data before it is trusted to accept or
reject it.

## 65. The decision point answered YES: validation ρ is climbing (round 56)

Round 53 moved the CPU to the cheap token route on the strength of a comparison, and named the measurement
that would make the call right or wrong: **does validation ρ pass the 0.1896 baseline?** At the 5 000-step
checkpoint, measured with the same tool and the same cache as the baseline:

| | step 3 000 (warm start) | **step 5 000** |
|---|---|---|
| train dim correlation | 0.3212 | **0.4640** |
| **validation dim correlation** | 0.1896 | **0.3043** |
| duration ratio | 0.764 | **0.822** |
| latent cosine | 0.847 | 0.844 |

**+60 % on validation and +44 % on training in 2 000 steps**, with the duration ratio improving too. This is
the first route in this project whose *held-out* number moves at all, and it moved the moment it was given
the CPU. For a coarse extrapolation: another ~16 000 steps at that rate would reach 0.75 — a few CPU-hours,
against the flow's 50+ with no signal. The curve will saturate before that, and the next checkpoints will
show where; the point here is that the cheap route is no longer a hypothesis.

## 66. The corpus expansion is in the cache, merged and consistent (round 56)

The 100k-character allowance produced **770 utterances / 1.54 h** of new teacher audio, and getting it into
training exposed two more instances of known traps:

* **the eighth checkpoint-geometry mismatch**: `make_teacher_corpus.py` built from the config (`n_voices: 1`)
  and loaded a checkpoint with four voices. Fixed by routing it through `load_checkpoint_into`, the helper
  added for exactly this in round 54 — which is the argument for having centralised it;
* **a token-width trap that would have corrupted the merge silently**: the expansion's cache was first built
  at `latent_rate 1` (24-wide token targets) against the base cache's 3 (72-wide). The merge would have
  produced 770 items the model cannot read among 1 207 it can. `make_teacher_corpus.py` gained
  `--latent-rate` (applied *after* the geometry load, because the autoencoder checkpoint predates the rate
  change and still carries 1), and `merge_caches.py` now **refuses** a width mismatch instead of merging it,
  with a test that pins the refusal.

The merged cache: **1 977 items** (1 207 + 770), **13 voices** (the expansion's single unlabelled API voice
appended), token widths `{72}` throughout, and the base cache's latent normaliser preserved — verified by
loading it, not by reading the merge's own output.

## 67. The expansion is at a different scale, and the run continues on it anyway (round 57)

Training resumed on the merged cache, warm-started from the 0.3043 checkpoint. Two things the log and a
per-voice breakdown reveal, both worth recording before a curve is read:

* **the latent-head calibration moved**: predicted std 0.9533 → target **0.6717** on the merged cache against
  **0.9113** on the base cache, i.e. the merged target is 26 % narrower;
* **per voice**, the reason is clear: the ten base Speechify voices and the Kokoro voices all sit at token
  std 1.07–1.29, while the expansion's single unlabelled API voice sits at **0.643** (frame std 0.686 against
  1.077), with **1.5× longer utterances** (746 frames against ~500).

| half | items | token std | frame std |
|---|---|---|---|
| base | 1 207 | 1.096 | 1.077 |
| expansion | 770 | **0.643** | **0.686** |

So the expansion is not simply "more of the same": it is one new voice, quieter or less dynamic, speaking
longer sentences. That is a legitimate thing for a multi-voice corpus to contain — and it is a **confound
for the training curve**, because a single shared output scale must serve two halves 41 % apart.

Pre-emptive "fixes" (per-voice normalisation, regenerating at another level) are not obviously right, so the
disciplined move is the one already scheduled: read validation ρ at the 5 000-step checkpoint and compare it
with **0.3043**, measured at the same step count on the base cache alone. If the expansion helps, the scale
difference is tolerable; if it hurts, the fix is a per-source level match and this paragraph is the
justification for it.

The warm start also reported exactly what it should: `1 missing key` — `voice_embed.weight (12, 512) -> (13,
512)` — the new voice has no pretrained embedding and gets a fresh one, which is the correct behaviour and
visible rather than silent.

## 68. The validation split was measuring a *source*, not generalisation (round 57)

Comparing the expanded run against the base one produced a result that looked decisive — train cosine 0.873,
**validation cosine −0.366** — and the obvious reading was "the expansion hurts". It was wrong, and the
reason is in `fit_diagnosis`:

```python
validation_indices = list(range(max(0, len(dataset) - n), len(dataset)))   # the cache's TAIL
```

The caches are written in manifest order, so the tail is **whatever source was concatenated last**. For the
expanded cache that is the entire 770-item expansion: one low-scale voice (token std 0.643 against the base
half's 1.096). So "validation" was a source comparison, and a negative cosine was the expected outcome of
asking a base-scale model about a different-scale voice — not evidence about generalisation at all.

The split is now **shuffled** (`--split-seed`, default 0), the report carries the voice composition of both
sides so a source-shaped split stays visible, and both checkpoints were re-measured through it:

| step 5 000 | train ρ | **validation ρ** | validation cosine |
|---|---|---|---|
| base cache (1 207 items) | 0.375 | **0.388** | 0.868 |
| expanded cache (1 977 items) | 0.352 | **0.291** | 0.392 |

Three things follow, and only one of them is about the expansion:

* the **decision point still stands, and is better than reported**: 0.388 against the 0.1896 baseline. The
  baseline was measured with the same biased tail split, so that comparison was apples-to-apples; the
  absolute numbers were not;
* the **expansion genuinely hurts** once the split is fair (0.291 against 0.388, cosine 0.392 against 0.868).
  The pre-registered plan from round 57 applies — a per-source level match, with that paragraph as the
  justification for the work;
* the CPU went back to the base corpus, continuing from the step-5 000 checkpoint.

The general lesson is the one this project keeps paying for: a metric whose *split* is not examined can look
like a result about the model. This is the third instrumentation correction in five rounds, and the cheapest
kind of bug to fix — provided the split is looked at before the conclusion is written.

## 69. Why the expansion is at a different scale: a partial answer, and a refuted fix (round 58)

Round 57 pre-registered "a per-source level match" as the fix if the expansion hurt. Measuring it refutes
that fix and finds a partial cause — both worth having in writing before anyone spends the rest of the
allowance.

**It is not a level difference.** Reading the audio directly (`scripts/expansion_levels.py`, which resolves
the corpus's wavs through their manifest because they are not under the manifest itself):

| | rms | peak | crest |
|---|---|---|---|
| base corpus (Speechify) | −18.29 dB | −1.28 dB | 16.89 dB |
| expansion | −19.97 dB | −3.02 dB | 17.00 dB |

A **1.68 dB** level difference with a **0.11 dB** crest difference. The latent-scale gap is 0.643 against
1.096, i.e. **4.6 dB worth** — so gain cannot account for it, and the "level match" fix would not have
worked.

**The voice was part of it, not all of it.** `DEFAULT_VOICES` had no `"speechify"` entry, so the teacher's
`synthesize(text, voice=None)` fell back to the API's default speaker, and the manifest recorded an empty
voice. That is now wired to the corpus's own ten voices and a generated `alec` utterance records
`voice: 'alec'` — verified with a single prompt (~130 characters) rather than by re-running 771. But the
same utterance still encodes at **0.798** against the corpus's `alec` items at **1.041**, i.e. naming the
voice closes only about a quarter of the gap.

So the remaining cause is something else — the expansion's utterances are 1.5× longer (746 frames against
~500), which lowers within-utterance variance by construction, and the API's own output may differ from
whatever configuration produced the corpus. **Not diagnosed**, and stated as such. The practical
consequences:

* **do not spend more of the allowance on this shape of expansion**: with the cause only partly identified, a
  second 100k characters is a gamble, not a fix;
* the 770 utterances are a *distinct voice* with a smaller latent scale. The honest options are per-voice
  latent normalisation (a design change, and arguably right for a multi-voice corpus: it removes a
  per-voice scale the text cannot predict) or leaving them out. Both are decisions, neither is a bug fix;
* training continues on the base corpus, which measures better.

## 70. The free teacher does what the paid one could not (round 59)

The first expansion cost 100k Speechify characters and produced audio at a scale the model could not absorb.
The corpus has a **second** teacher, and it is local, Apache-2.0, and already represented in the corpus:
Kokoro (`data/teachers/kokoro-en-v0_19`), whose items measure token std **1.07–1.12** against the Speechify
voices' 1.07–1.29.

So the same 771 new prompts — the ones already built from four public-domain books, with validation leakage
excluded — are being rendered through Kokoro instead:

```powershell
.\.venv\Scripts\python.exe scripts\make_teacher_corpus.py --texts data\prompts\prompts_v2.txt `
    --teachers kokoro --out data\kokoro_v2 --no-curate
```

No API call, no allowance, no permission question, and — on the evidence available — **a scale that matches
the corpus**, which is the one property the paid expansion lacked. It is slower per utterance on CPU, which
costs nothing but wall clock.

The validation plan is already-tested tooling, in order:

1. `--cache-only --latent-rate 3` on the new corpus, so the token width matches (the merge now **refuses** a
   mismatch, so this cannot silently go wrong the way it nearly did in round 56);
2. `merge_caches.py` into `runs/expanded_v2`, which remaps voices and keeps the base normaliser;
3. **measure the two halves' token std before training anything** — if they match, the expansion is usable on
   the same basis as the paid one was rejected; if not, it is the same trap and gets the same treatment;
4. retrain warm-started from the best checkpoint and compare validation ρ through the **shuffled** split
   (0.388 baseline), not the tail split that misled round 57.

## 71. The token route is plateauing, and the expansion's cause is still unidentified (round 59)

**The climb is slowing.** With the split flaw fixed, validation ρ on the base corpus:

| total steps | validation ρ |
|---|---|
| 3 000 (warm start) | ~0.19 |
| 5 000 | 0.388 |
| 10 000 | **0.402** |

+0.014 over 5 000 steps, with training ρ moving 0.375 → 0.379. That is a plateau around **0.40**, not a
route to 0.75 — and 0.40 is below the 0.60 at which the recogniser already fails. More steps on this target
is no longer a plan.

**The expansion's cause is not found, and everything tried is ruled out.** The free Kokoro expansion (771
utterances, 1.38 h, Apache-2.0, local) reproduces the *paid* expansion's gap **exactly** — 0.697 against the
corpus's 1.151, a gap of 0.453 in both cases. That was a useful clue, and I followed it to the end of what
the budget allowed:

| hypothesis | test | result |
|---|---|---|
| the paid teacher / API | same prompts through local Kokoro | **same gap** — not the teacher |
| the voice | `DEFAULT_VOICES["speechify"]` wired to the corpus's voices | closes ~¼, not it |
| prompt length | 117 prompts at 30–90 chars (corpus median is 71) | **0.698 against 1.151** — not length |
| audio level | rms/peak/crest on both | 1.68 dB level, 0.11 dB crest — not level |
| the autoencoder checkpoint | re-encode cached waveforms with four different AEs | **identical** ratios, so not the checkpoint |
| the encode front-end | same builder path used for both | ruled out (all AEs agree with each other) |

So the corpus's Kokoro items and my Kokoro items — same teacher, same voices, same builder, same code path —
encode at 1.15 and 0.70. **Unresolved**, and recorded as unresolved rather than papered over. What that costs
the objective: the 770 paid utterances and the 771 free ones are both unusable as-is, and the remaining
~297k characters of allowance should not be spent reproducing the same shape.

`scripts/cache_encode_consistency.py` (kept from the paid-expansion probe) and
`scripts/expansion_levels.py` are the two tools that did this work, so the next person starts where this
stopped instead of at the beginning.

## 72. The plateau is the model, not the target — and the split was leaking text (round 60)

The token route stalled at ~0.40. The question worth money is whether that is the *target's* ceiling or the
*model's*. The corpus contains the experiment needed to ask: the operator's paragraph, and 331 other prompts,
are rendered in several voices, so **same text in different voices** can be correlated directly
(`scripts/target_ceiling.py`):

| comparison | token ρ | frame ρ |
|---|---|---|
| same text, different voices (n=108) | +0.248 | +0.171 |
| different text (control) | +0.031 | +0.069 |
| **the model at 10 000 steps** | **+0.402** | — |

**The model beats the cross-voice agreement.** That refutes the hypothesis the measurement was built to test:
cross-voice agreement is not an upper bound, because the model is *given the voice* while those pairs differ
in voice by construction. So a text+voice → latent model is not capped at 0.25, the target is **not** the
wall, and the honest ceiling for this target is still unmeasured. The script's own conclusion was rewritten
to say so rather than left printing the refuted reading.

It also exposed a **fourth instrumentation flaw**: 332 texts have multiple renditions, so the item-level
validation split was putting the same sentence on both sides. Splitting by **text group** — no prompt appears
in both halves — gives the honest numbers:

| total steps | train ρ | validation ρ |
|---|---|---|
| 5 000 | 0.369 | **0.372** |
| 10 000 | 0.358 | **0.371** |

Two things follow, and they matter more than the small correction:

* **the plateau is confirmed**: +0.371 → 0.371 across 5 000 steps;
* **the model is underfitting, not overfitting**: validation ρ ≈ training ρ (0.371 against 0.358–0.369). There
  is no generalisation gap to close, which — with the target ruled out above — points at capacity and
  architecture rather than data or steps. That is consistent with the only capacity experiment this project
  has run: 4.6M → 22.6M moved validation 0.118 → 0.190.

`fit_diagnosis` now splits by text group by default and reports the group count, so a leak of this kind
cannot quietly return.

## 73. The model *can* fit — so the plateau is the mapping, not the model (round 61)

Round 60 left a three-way ambiguity: the text side plateaus at ~0.37 with training ρ no higher, which could be
insufficient capacity, an optimisation problem, or a mapping that text alone does not determine. They call for
different spending, and memorisation separates them for a few minutes of CPU (`scripts/overfit_text.py`, the
token-route analogue of `overfit_flow.py`):

| 8 utterances, warm-started, lr 1e-3 | token ρ |
|---|---|
| before | +0.04 (a model that has never seen these items) |
| **after 300 steps** | **+0.9977** |
| after 3 000 steps, 64 utterances | **+0.9944** (from +0.8705: those items were already in its training corpus) |

**300 steps to ρ 0.998 on eight utterances.** Capacity and optimisation are therefore *adequate for fitting*:
the 22.6M text side is not too small and the schedule is not stuck. So the plateau at ~0.37 on the full
corpus is not a capacity problem — with round 60's result that the model already beats the same-text
cross-voice agreement, it is the **mapping**: one text admits many acoustics, and a deterministic regressor
must average them.

That closes a circle this project has walked several times. The papers do not predict acoustics from text
with a regressor; they *sample* them, precisely because the mapping is one-to-many. The deterministic route
was worth testing — it is a quarter of the size, one forward pass, no sampler — and it has now been measured
to its ceiling (~0.37–0.40), with the cause identified rather than assumed. The sampling route is the
structurally correct tool for this target; its own failure was diagnosed separately as the marginal being
learned first with weak conditioning (rounds 39–51), not as this.

## 74. Per-voice normalisation — the expansion gap, solved (round 62)

Rounds 58–59 ruled out the teacher, the voice *identity*, prompt length, audio level, the checkpoint and the
encode front-end — and left the 41 % scale gap unexplained. Fitting per-voice statistics
(`scripts/fit_voice_norm.py`) finally showed what it was, and it is boringly simple:

| voice | raw pooled std | after per-voice normalisation |
|---|---|---|
| `af_alloy`, `af_aoede` (the corpus's Kokoro voices) | 0.96, 0.85 | **1.000** |
| `alicia`, `alec`, `alfonso` (Speechify) | 1.11, 1.04, 1.07 | **1.000** |
| `af_heart` … `af_kore` (the expansion's Kokoro voices) | 0.66–0.89 | **1.000** |

**The scale is a property of the voice.** The corpus's Kokoro half used `af_alloy`/`af_aoede` (pooled std
0.83–0.85) while the expansion used eleven *other* Kokoro voices (0.44–0.52 pooled) — same teacher, same
builder, 1.7× different scale. Nothing about the data was wrong; a shared output head simply cannot serve two
scales at once, which is why mixing them measured worse (round 57: 0.291 against 0.388).

So the fix is an affine map per voice, in both directions:

* `scripts/fit_voice_norm.py` writes `voice_norm` into `cache_meta.json` — nothing is rewritten, so it is
  cheap and reversible, and **presence is what enables it**, leaving older caches unaffected;
* `LatentShardDataset` applies it to both targets (frames and tokens — the same 24-dimensional signal);
* `denormalize_voice` inverts it in **all three** synthesis paths: one-shot Tiny, one-shot flow, and
  blockwise streaming, always *before* the cache-level `latent_norm.denormalize`.

Two failures the tests caught before they shipped, both the silent kind:

* **the feature axis is not the same for both targets** — frames arrive as `(B, C, T)` and tokens as
  `(B, T, C = latent_dim·rate)`. Picking wrong raises a shape error at best, and scales the wrong axis at
  worst;
* **a wider voice table must not discard what was learned.** A shape-filtered load *skips*
  `voice_embed.weight` when the table grows, so a 12-voice checkpoint resumed on the 21-voice cache would
  rebuild **every** voice embedding from scratch. `load_checkpoint_into` now copies the rows that correspond
  and leaves the new voices at their initialisation.

* **	rain.py had its own warm-start path** and did not use the shared helper, so the very first\n  expanded retrain logged `skipped 1 key: voice_embed.weight (12,512)->(21,512)` and started with **every**\n  voice embedding random -- visible immediately as a loss of 1.45 at step 150 against ~0.9 for the same\n  trajectory before. With the growth logic applied there too: `voice table grew: copied 12 learned rows, 9\n  new voice(s) initialised fresh`, 0 missing keys, loss 0.94 at step 150. The lesson is the one from round 54\n  repeating: two code paths that do the same job will not stay the same, so the helper has to be the only one.\n\nNet effect: **1 541 utterances** (770 paid + 771 free) that measured as unusable are now on the same scale as
the corpus, verified by synthesising through base voices (0, 3) and expansion voices (12, 20) with the
21-row table. The corpus can now grow without a per-voice confound, which is the thing that has blocked every
expansion attempt since round 55.

## 75. The velocity field ignores the text, measured properly — and a mid-round self-correction (round 64)

The flow's conditioning was measured once (round 51, "the text carries 14 %") by shuffling the text and
watching the loss. That number was misleading, because the loss it measured **includes the plan term**, and a
plan head is a *supervised* text → token regression: it is text-dependent by construction. This round
decomposed it (`scripts/text_dependence.py` now reports total, flow-only and plan separately), and the
picture changes completely:

| 200 items, 160-frame crops | total | **flow term only** | plan term |
|---|---|---|---|
| advisory, step 1 000 | +19.3 % | **+3.4 %** | +67.0 % |
| advisory, step 2 000 | +37.4 % | **+2.0 %** | +111.1 % |
| residual, step 1 000 | +18.5 % | **+1.8 %** | +64.5 % |
| pre-fix architecture, step 1 000 | +4.4 % | — | — |

Read in order, this is the round's story:

* **I nearly reported a breakthrough.** The total rose 4.4 % → 19.3 % → 37.4 % across checkpoints and the
  obvious reading was "the conditioning is growing 4.4×". The decomposition, run because the number looked
  too good, shows the rise is **entirely the plan**, and the velocity field's own dependence is **2–3 %** —
  *lower* than the baseline's total, and flat or falling as the plan improves. The claim is retracted before
  it left the round;
* **the residual form does not fix it either**: making the target `x1 − upsample(plan)` was supposed to force
  the field to consume the plan, and it measures **+1.8 %** — no better than advisory. Two architectures,
  same answer, measured with an instrument that can now tell them apart;
* **so the flow's blocker is precisely stated**: the field's objective is ~2 % text-dependent, which is
  exactly consistent with ρ ≈ 0 on its samples. It is not that conditioning *cannot* be learned — the 2-item
  regime reached ρ 0.25, where the marginal has little to offer — it is that at this scale the marginal
  solution is cheap and the text's 2 % is not worth learning.

The tool is the durable part: any future conditioning change can be judged in one command by the **flow
term's** share, rather than by a total that a supervised auxiliary can inflate.

## 76. The long flow run's first gate reading (round 67)

The indicated spend is running: the flow on the **expanded, per-voice-normalised** corpus (1 978 items, 21
voices), 40 000 steps at ~0.74 s/step — about 8 CPU-hours — with a checkpoint every 2 000 so the gate can be
read as it goes. At step 2 000:

| | right text | shuffled text | share |
|---|---|---|---|
| total (flow + plan) | 1.2703 | 1.3957 | +9.9 % |
| **flow term only** | 0.5839 | 0.6072 | **+4.0 %** |
| plan term (supervised) | 0.6863 | 0.7885 | +14.9 % |

4.0 % sits inside the 1.9–7.4 % band that `conditioning_vs_scale.py` measured, so nothing has changed yet and
the *trajectory* is the signal to watch over the next checkpoints. Two things are worth having on record
before reading that trajectory:

* the reference points are noisy — the same 200-utterance size measured 2–3 % (batch 16) and 7.4 % (batch 8) —
  so a single reading means little and a monotone trend across several checkpoints means something;
* the run is also the first with **per-voice-normalised targets** and 1 541 recovered utterances, so if the
  trajectory differs from round 65's curve, those are the two things that changed.

Also verified this round, because it is the one place a missing per-voice inversion would not crash: the
**flow's** synthesis path inverts it correctly end to end. `load_latent_norm_from_cache` installs 19 voice
tables, and one-shot synthesis succeeds for base voices (0, 3) *and* expansion voices (12, 20), as does
blockwise streaming — wrong-scale latents with no error would have been the alternative.

## 77. The design matches the paper; the data does not (round 68)

Reading SupertonicTTS's own numbers rather than working from memory corrected a claim this project had been
carrying. The paper: *"With `Kc = 6` and `C = 24`, the model processes 144-dimensional vectors at a reduced
frame rate of approximately 14 Hz."* This project: the autoencoder emits 24 dims at 93.75 Hz and the flow
folds them by `compress: 6`, so the velocity field models **144-dimensional vectors at 15.6 Hz** — verified
from the config, not assumed.

| | latent the flow models | dims/s |
|---|---|---|
| SupertonicTTS | 144 dims @ ~14 Hz | ~2 000 |
| this project | 144 dims @ 15.6 Hz | 2 250 |

**They match**, and `docs/PLAN.md` had recommended a lower-rate learned bottleneck as the fix for a target it
called "unusually large". That recommendation is **retracted**: the `compress: 6` fold is exactly the
mechanism that closes the gap, and it has been in the config all along. The full comparison is in
[docs/PAPER_COMPARISON.md](../PAPER_COMPARISON.md), including the axes that *do* line up (latent size, folded
rate, ConvNeXt, character-level text, implicit alignment, `Ke = 4`, NFE 32 — and this model is **smaller**,
21.3M against 44M) and the one that does not:

**945 hours** of generative training data in the paper against **~1.9 h** in the corpus before this month's
generation (which added ~6.1 h; see §86 for the corrected arithmetic, and note that the corpus already contains
the earlier Speechify and Kokoro expansions). That reframes the remaining work from "find the missing
architectural trick" to "more teacher audio, or more compute, with the same architecture".

### The long run's second gate reading

| step | flow-term text dependence |
|---|---|
| 2 000 | +4.0 % |
| 4 000 | **+3.8 %** |

Flat. Two readings at the same value are not yet a trajectory, and the token route's own history says the
mapping can appear only after several thousand steps — so the run continues and the reading at 6 000, 8 000,
… is what decides. Recorded now, before the later points are known, because "flat so far" is the honest
description.

## 78. Acting on the paper comparison: the CPU goes to data (round 69)

Round 68 established where the gap is — 945 hours of generative training data against ~1.9 h here — so the
CPU moved from training to *data*, and the flow run was paused rather than left to spend 7 more CPU-hours
with a flat gate (its readings at 2 000 and 4 000 steps are on record in §77).

The free teacher makes this the one lever that costs nothing but wall clock. Kokoro is local, Apache-2.0, and
was already used for 304 corpus utterances; eight public-domain books (~9.3 MB) provide the text, and the
prompt builder excludes every sentence already used (1 407 prompts) so the new material is genuinely new:

| | |
|---|---|
| new prompts | **3 326**, 400 031 characters, ~50k from each of 8 books |
| measured rate | 261 utterances in ~10 min; 0.36 h of audio per 200 utterances |
| projected | **~4.6 hours of teacher audio** in ~2.1 h of CPU |
| corpus after | **~9.7 h** (1.9 h original + 3.2 h expansions + 4.6 h) |

Still two orders of magnitude short of the paper, and stated that way. But it is a 5× increase for zero
dollars, it uses the one teacher with no permission question, and round 65 measured that the flow's
conditioning **grows with corpus size** (1.9 % → 7.4 % over a 50× range) — so this is the lever the evidence
points at, not a hopeful one.

The pipeline for it is already tested end to end: `--no-curate` (the gate's proposed 5 kHz bandwidth floor
rejects this teacher's own audio, §64), `--latent-rate 3` (the merge now *refuses* a width mismatch), per-voice
normalisation (the scale trap of §69, fixed in §74), and a merge that remaps voices and keeps the base
normaliser.

## 79. The expansion pipeline, integration-tested before the big run (round 71)

`scripts/expand_corpus.py` was run for real — not just unit-tested — on the 117-utterance Kokoro set already on
disk, end to end:

```
latent rate overridden to 3 (token width 72)
n_voices 4 -> 11 for the cache write
merged: {"base": 1207, "extra": 117, "total": 1324, "voices": 21, "shards": 3}
fitted per-voice statistics for 19 voices
{
  "items": 1324, "new_items": 117, "voices": 21,
  "voices_with_normalisation": 19,
  "token_widths": [72],
  "worst_pooled_scale_error": 0.0688
}
```

Every guard that matters passed on live data: the cache was built at the right token width, the merge remapped
9 new voices onto the existing table, the normaliser was fitted, and the worst fitted voice pools within
**0.069** of unit scale against a 0.15 limit.

That is the whole chain that took four rounds to get right (56, 58–62) now being one command, and it means the
3 326-utterance run does not have to discover any of it again. Expected cost for the real corpus: the cache
build is ~5–8 min per 771 utterances, so roughly **20–30 minutes** for 3 326, then the merge and fit are
seconds.

## 80. Deliberate engineering checks worth calling out

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

## 82. Integrity pass, and two self-inflicted errors (rounds 73–75)

### The docs audit: a shell replacement that matches nothing is a no-op that looks like success

Round 73 found that **`docs/ROADMAP.md` had never been updated** by rounds of anchor-based shell replacements.
The replacement matched nothing, `Set-Content` wrote identical content, and the commit still succeeded because
*other* files had changed — so nothing ever failed loudly. Auditing what the commits actually touched:

| file | state |
|---|---|
| `docs/05-VERIFICATION.md` | **every section present (73–81), 2 560 lines** |
| `docs/PLAN.md` | updated |
| `docs/PAPER_COMPARISON.md` | new |
| `docs/ROADMAP.md` | **never changed** |

The authoritative record was never at risk — the measurements all live in this file. What was wrong is the
forward-looking checklist and the reports claiming it was current. Both are fixed, and the roadmap's section is
labelled as a reconstruction.

Round 74 then audited *this* file the same way and found the identical bug inside it: a duplicate `## 53.`
and `## 62. Environment notes` sitting at the end, after §80 — because several rounds reported "renumbering
Environment notes to 81" while the replacement targeted `## 80.` and the heading said `62`. Content intact;
labels fixed (53.1, 81) and re-audited clean. Numbers 35 and 38 have never existed and stay gaps: renumbering
twenty-five sections would break the cross-references that cite them.

**Either failure alone looks like nothing.** Both were caught only by asking what the commits had actually
touched, which is now the rule: any doc change gets read back.

### Two errors of my own in round 75

**I killed the corpus generation.** To stop a void experimental arm I ran "the highest-CPU python process",
which was the generator — 40 minutes older than the arm. The correct tool was `job_kill` with a job id. 725
utterances were lost; the resume generates only the remainder (2 601 prompts into `data/kokoro_v3b`) and both
halves merge, so the work is not discarded. The split was verified rather than assumed: 725 + 2 601 = 3 326
prompts, **zero overlap, zero missing** against the original file, so the recovery is exact.

**The A/B arm was void and only a printed count said so.** Warming the flow's text encoder from the token
checkpoint loaded **0 of 85 text tensors**: the autoencoder checkpoint's geometry is 256×4, the token route's
text encoder is 512×6, and the loader filters by exact shape equality. Round 73's probe *had* verified 85-of-85
compatibility — under a configuration the script does not use, because the probe set `text.dim` and
`n_layers` itself. **Verified-then-trusted.**

Both are fixed with the failure as the reason:

* `overfit_flow.py --set section.field=value`, applied **after** the checkpoint geometry and **before**
  `build_model`. The first attempt applied it after the warm-start load; the new guard fired again and caught
  the ordering bug — the override belongs where the model is *built*, not where weights are *loaded into*;
* a warm start that matches **no** text tensors when the checkpoint has them is now **fatal**, not a printed
  statistic. With the geometry matched: `loaded 190 tensors, 85 of them from the text encoder`.

The shared shape of both errors: a check that ran under conditions the real code does not use, and a kill that
targeted a proxy instead of an identity.

### The A/B was paused for the corpus, on measured grounds (round 77)

Arm A was consuming **22.5 CPU-seconds per step** under contention against 6 uncontended, so the two arms
would have taken about six hours and would have finished *after* the corpus that they were meant to inform.
The paper comparison (§77) puts the gap at data, not at conditioning levers, so the A/B was stopped by job id
-- the correct tool, after round 75 used a CPU proxy and killed the generator instead -- and the whole CPU went
to generation. Measured effect, ten minutes either side: **~10 wavs/min shared, 23 wavs/min alone**, cutting
the remaining generation from ~3 h to ~1.3 h.

The A/B reruns at full speed once the corpus is in, and the decision rule below still stands. pausing a
one-off experiment so the identified bottleneck finishes sooner is the trade this project's evidence supports.

### Pre-registered decision rule for the text-warm-start A/B

Both arms run on identical data (1 978 items, `runs/expanded_v2`) and identical geometry (512×6): arm A fresh,
arm B warm-started from the token route. The readout is the **flow term's own** text dependence at steps
2 000 and 4 000, against arm A's earlier readings of **+4.0 %** and **+3.8 %**.

| arm B at step 4 000 | reading | what it means |
|---|---|---|
| **≥ 8 %** | the warm start roughly doubles the field's use of the text | keep it, and retrain the corpus run from it |
| **5–8 %** | a real but modest gain | keep it only if it also holds on the new corpus |
| **≤ 4.5 %** | indistinguishable from arm A | drop it; round 43's null result survives the scale test |

Written now, before either arm has finished, because a rule invented afterwards is not a rule.

## 84. Auditing the no-op-normalisation class, and a hypothesis it produced (round 79)

Round 78's bug was "a reader that fits through a dataset which already applies the table". Every other caller
of `LatentShardDataset` was audited for the same class:

| caller | applies the table? | correct? |
|---|---|---|
| `merge_caches` | **never constructs the dataset** — it reads shards with `torch.load` and copies `payload["items"]` | **yes**, so merged caches store *raw* latents and nothing is normalised twice |
| `fit_voice_norm` | now reads raw (`apply_voice_norm=False`) | fixed |
| `expand_corpus.verify` | applies it, then checks the pooled scale | yes — this is what caught the bug |
| training and evaluation scripts (`overfit_flow`, `overfit_text`, `rho_curve`, `fit_diagnosis`, `real_eval`, …) | apply it | yes: they must train and measure in the space the model learns in |
| `latent_autocorrelation`, `bottleneck_ceiling` | apply it | harmless: an affine per-dimension map leaves per-dimension autocorrelation unchanged |

So the bug was confined to the one place that **refits** — which is why it survived from round 62 until a refit
actually happened, and why rehearsing the chained merge is what found it. The important row is `merge_caches`:
had the merge read through the dataset, all 3 326 new utterances would carry a double normalisation and every
guard would still pass, because the guards check scale, not provenance.

### A hypothesis the audit produced — tested, and refuted (round 80)

`scripts/target_ceiling.py` measures same-text **cross-voice** agreement — the 0.248 (token) / 0.171 (frame)
figures from round 61, read as "the target is inconsistent, so a deterministic model cannot do better". That
was measured on an **unfitted** cache. Per-voice normalisation applies an affine map per voice, so the
prediction was that the ceiling on a fitted cache would be **higher**.

`target_ceiling.py` gained `--cache` and `--no-voice-norm`, and the two arms ran on **identical items** (the
1 207 base utterances inside `runs/expanded_v2`):

| arm | same text, different voices | control (different text) |
|---|---|---|
| raw (normalisation off) | token **+0.248**, frame **+0.171** | +0.031 / +0.069 |
| normalised (round-62 table active) | token **+0.248**, frame **+0.171** | +0.031 / +0.069 |

**Identical to three decimals.** The raw arm also reproduces round 61's published figures exactly, which is a
useful confirmation that the measurement itself is stable across thirty rounds.

Why it could not have moved, in two parts:

* the eight takes' voices already share a scale — raw stds span 1.043–1.283, a **1.23×** spread, against the
  *expansion's* 0.44–0.52 — so the per-voice corrections are near-identical and a cross-voice correlation
  barely moves. Per-voice normalisation exists to make voices at **different** scales mixable, not to align
  voices that are already alike;
* more fundamentally, what the ceiling measures is the part of the acoustics the text does not determine. An
  affine map rescales; it cannot make two voices' *content* agree.

**Consequence: the cheap "retrain the token route on the normalised corpus to beat ρ 0.37" test is dropped.**
Its motivation was this hypothesis, the hypothesis is refuted, and the ceiling is unchanged — so the expected
gain is zero and the CPU belongs to the corpus. The flat-µ plateau keeps its round-61 interpretation: the limit
is the one-to-many mapping, and sampling is the structurally correct response to it.

## 85. Pre-flight for the bigger corpus (round 81)

LatentShardDataset loads every shard into memory at once -- merge_caches says so in its own docstring, which
is why the merge streams. The corpus is about to grow from 1 978 items to roughly 5 300, so the memory ceiling
was measured before the run rather than discovered during it:

| | |
|---|---|
| machine RAM | 31.2 GB total |
| uns/expanded_v2 on disk | 1 637 MB for 1 978 items |
| projected for ~5 300 items | **4.4 GB on disk**, ~5-6 GB in memory |
| plus model (21M fp32) and activations | well under 1 GB |

Comfortable, so no sharding work is needed and the flow can train on the whole expanded corpus. The check costs
a minute and rules out the failure mode that would otherwise appear as an unexplained allocation error halfway
into the run that this entire spend exists to enable.

## 86. The real merge, on the recovered half of the corpus (round 83)

The rehearsal used 117 utterances. This one used the 725 the generator actually produced before round 75's kill,
merged onto the base corpus with the fixed fitter:

`
merged: {"base": 1207, "extra": 725, "voices_added": 9, "total": 1932, "voices": 21, "shards": 4}
fitted per-voice statistics for 19 voices
token_widths: [72]
worst_pooled_scale_error: 0.0419
`

**0.0419** against the rehearsal's 0.0688 and the 0.15 limit -- lower on more data, which is what a per-voice
statistic estimated from 725 items rather than 117 should do. The guards that matter passed on live data: the
correct token width, nine new voices remapped onto the existing table, and every fitted voice within 4 % of unit
scale.

So the corpus stands at **1 932 items (~3.2 h of audio)** right now, and the second half (2 601 utterances,
~4.8 h) merges onto this result when the generator finishes.

> **Corrected in round 89.** This paragraph first said "~4.5 h" here and "~11 h" for the final corpus. Both were
> a double count: `runs/mixed_v2/latent_cache` is 1 207 items / **1.88 h** and *already contains* the earlier
> Speechify and Kokoro work, which I was adding a second time as "~3.2 h of expansions". Measured hours give
> **1.88 + 1.32 + ~4.81 = ~8.0 h** for ~4 533 items — a **4.3×** increase, not 6×. The same wrong figure reached
> `README.md`, `docs/PLAN.md` and `docs/PAPER_COMPARISON.md`, and all four are corrected. The lesson is the one
> from round 73: a number computed in prose and never checked against the cache itself.

The chain is the same one rehearsed in round 78 -- which is why this was one command and no surprises.

## 87. Test count corrected, and the count is now checked rather than estimated (round 85)

The commit that added the conditioning-gate tests said "Suite 337 -> 341". The real number was **336 -> 340**.
The earlier figure came from summing pytest's --collect-only per-file counts, which is not the same as the
number pytest reports as passed, and I quoted it without checking the summary line.

No document had the wrong number -- nothing in the repo states a test count -- so the correction is to the
commit message and to the method: pytest -p no:warnings and read the N passed line, which is what every
count in this log from now on comes from. It is a small thing, but "I reported a number I had not read" is the
same failure as the roadmap rounds, in miniature.

## 88. The second half's manifest will reconstruct exactly, and the finish is one command (round 87)

data/kokoro_v3b was generated before the round-82 durability fix, so it is in the same state the first half
was: wavs present, no manifest, and therefore unusable to the cache build. The same reconstruction applies, and
its alignment was **checked rather than assumed** before committing to it:

| check | result |
|---|---|
| wav indices | 0 .. 1596 contiguous, 1597 files |
| prompts_v3_rest.txt length | 2 601 (matches the 2 601 prompts it was cut for) |
| prompts_v3.txt[725] == prompts_v3_rest.txt[0] | **true** -- so index i maps to the right prompt |

So ebuild_manifest.py --texts data/prompts/prompts_v3_rest.txt produces correct records for that directory,
and the recovery that saved the first half works for the second.

scripts/build_expanded_v3.ps1 puts the whole finish in one command, in an order where every step has failed at
least once here: rebuild the manifest, cache with --latent-rate 3 (or the merge refuses), merge onto the
**existing** uns/expanded_v3a (whose voices are already remapped and whose normaliser covers the base
corpus), then train the flow with the gate on. Each step is skipped if its output already exists, so an
interruption costs only the step it was in. Verified: it parses cleanly and its guards report the correct state
of the filesystem right now.

## 90. Pre-registered decision rule for the expanded-corpus flow run (round 90)

The corpus is about to reach ~4 533 items / ~8.0 h and the flow will train on it for 12 000 steps with the gate
printed every 2 000. Written before any reading exists, because the same rule invented afterwards is not a rule.

What is known, and therefore what would count as news:

* the corpus-size effect was measured on a small regime — flow-term dependence **1.9 % (4 utterances) → 2.3 %
  (16) → 4.1 % (64) → 7.4 % (200)** at fixed steps (§65);
* at ~2 000 utterances it read **+4.0 % (step 2 000) / +3.8 % (step 4 000)** and flat (§76) — which is *lower*
  than the 200-utterance point, so either the 7.4 % was noise (its own run-to-run spread is ±3 points) or the
  curve had already turned over;
* so this run tests whether **2.3× more items than that flat run** moves the field, not whether a 50× range
  does.

| gate at step 12 000 | reading | decision |
|---|---|---|
| **≥ 8 %** | the corpus lever is real and still rising at this scale | keep buying data; the gap identified in §77 is the right one to attack |
| **5–8 %** | a modest but genuine gain | continue with data, and prioritise the text-warm-start A/B (§82) as the cheaper next lever |
| **≤ 4.5 %** | no movement from 2.3× the items | the corpus-size effect saturates below this regime: stop buying data, and either attack the marginal's cheapness or accept and document the ceiling |

Secondary readout, reported alongside: **validation ρ** on the 32 held-out utterances. Any value meaningfully
above 0 would be the first time this flow's samples have tracked the right latent rather than only the
marginal (§56), so it is worth recording whatever it is — including if it stays at ~0, which is what the 2 %
field dependence predicts.

## 91. The corpus is built and the flow run has started (round 91)

Generation finished cleanly at **2 601 / 2 601 utterances and 4.75 h**, and its manifest *was* written at the end
-- so the round-82 rebuild was not needed, and uild_expanded_v3.ps1's guard skipped it. That is the guard
working: it is there to handle the interrupted case, not to insist on running.

The chain then executed in one command:

`
corpus: 4533 items | training on 4501, holding out 32
override text.dim = 512 / text.n_layers = 6 / flow.text_dim = 512 / flow.cond_dim = 512
before training: sampled latent rho on 24 of the 4501 training utterances = -0.0004
holdout: 32 utterances never trained on (rho before = +0.0059)
schedule: 12000 steps with 600 warmup
`

Measured corpus accounting, which confirms the round-89 correction:

| | items | audio |
|---|---|---|
| mixed_v2 | 1 207 | 1.88 h |
| kokoro_v3 | 725 | 1.32 h |
| kokoro_v3b | 2 601 | 4.75 h |
| **total** | **4 533** | **7.95 h** — a **4.2×** increase |

**One thing to state plainly:** the run is **39.2M parameters (34.2M trainable)**, not the 21.3M in
docs/PAPER_COMPARISON.md. The --set text.dim=512 / n_layers=6 overrides were carried over from the
text-warm-start A/B, where matching the token route's geometry is what makes a warm start load at all. A corpus
run does not need them, and they roughly double the text encoder. So the documented architecture is the 256×4
build at 21.3M; this is a larger research variant of it, and the comparison table's parameter row should say
which build it describes.

The gate readings are due every 2 000 steps against the rule pre-registered in §90, which was written before
any of them existed.

## 92. A second optimistic rate estimate, from measuring the wrong corpus (round 92)

Round 88 measured the flow at 1.76 s/step contended and inferred ~0.9 alone, then right-sized this run to
12 000 steps on that basis. The run actually achieves about **2.7 s/step** — 600 steps in 27 minutes — so
12 000 steps is roughly **9 CPU-hours**, not the 2.5–5 projected.

The measurement was not wrong; it was taken on the **wrong corpus**. The 1 932-item cache's utterances are
shorter, and with `self_reference=True` every batch is padded to its longest member, so longer utterances cost
more per step — 6.6 s each here against 4.5 s there. This is round 88's lesson repeating one round later:
**measure the thing you are actually going to run, not the nearest thing available.**

The run continues rather than restarting — 900 steps and a falling loss (3.76 → 1.39) are already spent, and the
gate trend is what answers the question, not the exact step count. The readings at 2 000/4 000/6 000 still
satisfy §90's rule; the wall clock is now stated as ~9 CPU-hours rather than leaving the earlier figure standing.

## 94. The first gate reading on the expanded corpus: no gain, and the widest total/flow gap yet (round 93)

```
[gate] step 2000: FLOW-TERM text dependence +2.4% (total +23.1%, plan +39.9%)
```

Read against the two references this project already has, at the same step:

| run | items | audio | flow-term dependence at step 2 000 |
|---|---|---|---|
| `flow_long_crop` | 1 978 | ~1.9 h | **+4.0 %** |
| **`flow_v3`** | **4 533** | **7.95 h** | **+2.4 %** |

**2.3× the corpus, 4.2× the audio, and the velocity field's own text dependence went *down*.** 2.4 % against
4.0 % is outside the ±0.6 run-to-run spread this metric shows, so it is a decrease rather than noise. Applied to
the pre-registered rule in §90 — whose thresholds are stated for step 12 000, making this an intermediate
reading — the value sits in the **≤ 4.5 % band**: no movement from a larger corpus, which is the branch that
says stop buying data.

**The total would have said the opposite.** +23.1 % is more than double the smaller run's +9.9 %, and a reader
looking at the total alone would report that the conditioning had more than doubled. The velocity field's share
is 2.4 % of it — the plan head accounts for essentially all the rest. This is round 64's trap at its widest
divergence yet (23.1 against 2.4, or 10:1), and it is the clearest argument for the gate being automated and
decomposed rather than measured by hand when convenient.

**What it means, provisionally.** The corpus-size effect measured in §65 (1.9 % → 7.4 % across 4 to 200
utterances) does not extend to this regime: from ~2 000 to ~4 500 utterances the dependence, if anything, falls.
Five more readings are due (4 000 … 12 000); if the trajectory stays in this band, the pre-registered decision is
to stop spending on data and attack the marginal's cheapness instead — that is, make the unconditional solution
worse so the text's few percent becomes worth learning.

## 95. Both readouts agree at step 2 000, so the run was stopped and the A/B took its place (round 94)

The secondary pre-registered readout, on the same checkpoint:

```
step  2000:  rho trained -0.0007   rho held-out +0.0049
```

**~0**, which is exactly what a velocity field with 2.4 % text dependence should produce. So both readouts
agree, and they agree on a negative:

| readout | value | band / expectation |
|---|---|---|
| flow-term text dependence | **+2.4 %** | the §90 **≤ 4.5 %** branch: no movement from a larger corpus |
| sampled-latent ρ | **−0.0007 / +0.0049** | ~0, as predicted — the samples carry the marginal, not the target |

**The remaining 10 000 steps were cancelled, deliberately.** The earlier run on 2.3× less data was flat from
step 2 000 to 4 000, both readouts are already decisive at 2 000, and the run had ~7 CPU-hours left. Finishing it
would have bought a trajectory shape, not a decision. That is the pre-registration doing work in the *other*
direction: it let a decisive negative be recognised at step 2 000 rather than after another day of compute.

**What replaced it: the text-warm-start A/B**, relaunched on this corpus rather than the small one (§82
pre-registered it; round 75 voided its first attempt on a geometry mismatch, and round 84 built the guard that
catches that). It is the one mechanism here that has demonstrably learned the mapping — the token route's text
encoder regresses text → acoustics at ρ 0.37 — so warming the flow from it tests whether the field can be given
a head start instead of being asked to discover the mapping from noise. Both arms, same corpus, same 512×6
geometry, 2 000 steps, gate every 500 for four points each, plus validation ρ.

One correction to §94's wording while it is fresh: "attack the marginal's cheapness" was my phrase, and it is
probably the wrong frame. A flow-matching objective is *supposed* to fit the marginal — that is what the
unconditional term is — so the field's low text share is not a defect to be starved out. What the measurement
shows is narrower and more honest: **more data at this scale does not strengthen the conditioning**, and the
mechanism that has worked is the trained text encoder, which is what the A/B now tests.

## 96. The gate's noise floor, measured by the A/B's own control arm — and a claim withdrawn (round 95)

Arm A (fresh, 4 501 training utterances, 512×6) sampled the gate every 500 steps, which is the resolution no
previous run had:

| step | flow-term text dependence | total | plan |
|---|---|---|---|
| 500 | −2.4 % | −4.9 % | −7.2 % |
| 1 000 | **+6.6 %** | +20.8 % | +33.0 % |
| 1 500 | **+4.1 %** | +12.8 % | +19.5 % |
| 2 000 | **−1.4 %** | +1.3 % | +3.3 % |

**A ~9-point swing within a single run.** Every previous use of this gate sampled it once per run, and the
"±0.6" figure I have been quoting came from repeat measurements *of one checkpoint* — a different quantity
entirely. The per-checkpoint noise is ±3–4 points, and the run-to-run spread is at least that.

**Claim withdrawn.** §94 read round 93's step-2 000 values as "the velocity field's dependence went *down*, 2.4 %
against 4.0 % on 2.3× less data, outside the ±0.6 spread so a decrease rather than noise". With the measured
noise floor, 2.4 against 4.0 is **inside** it. What survives is the weaker, honest statement: **on this corpus
the flow term sits in the same band as before, so the 4.2× corpus expansion produced no measurable change
in the conditioning** — which is still §90's ≤ 4.5 %-band conclusion (no improvement, stop buying data), but it
is "no measurable change", not "a decrease".

> **Numbers in this section are superseded.** The band was quoted as "2–7 %" here and in the sections that follow
> (§§96–106). Round 114 found the gate was being measured in train mode, where 10 % conditioning dropout inflates
> it; measured properly the figure is **~1–2 %**, and at every individual timestep **0.3–1.3 %** (§107). Later
> sections of this log therefore quote the corrected number, and the corrections are kept in place rather than
> rewritten here because the sequence of what was believed and why is the record.

Two consequences, both applied from here:

* the gate is only interpretable from **several points in one run**, not from a single checkpoint — which the
  in-loop printing was built for and which this round finally used;
* a difference between runs has to exceed ~5 points before it means anything, so round 65's monotone 1.9 → 7.4 %
  across corpus sizes remains suggestive (it is a trend across four sizes) but is weaker evidence than it read.

One genuine positive, recorded because it is the secondary readout and it is not zero: **ρ after 2 000 steps on
this corpus is +0.0882**, against ~0 for the equivalent run on the smaller corpus. Small, single-run, and not yet
reproduced on arm B — but it is the first non-zero flow ρ at 2 000 steps this project has measured.

## 97. The A/B: warming the flow's text encoder from the token route changes nothing (round 96)

Both arms, same corpus (4 501 training utterances), same 512×6 geometry, same schedule; the only difference is
that arm B's text encoder starts from the token route's, which regresses text → acoustics at ρ 0.37 and is the
one mechanism in this project that has demonstrably learned the mapping.

| step | arm A (fresh) | arm B (warm text encoder) | difference |
|---|---|---|---|
| 500 | −2.4 % | −1.7 % | +0.7 |
| 1 000 | **+6.6 %** | **+6.8 %** | +0.2 |
| 1 500 | **+4.1 %** | **+3.9 %** | −0.2 |
| 2 000 | −1.4 % | *pending* | — |

**The arms track each other to within 0.2 points at every shared checkpoint.** Against a noise floor measured the
same day at ±3–4 points, that is indistinguishable — the pre-registered **≤ 4.5 %-ish branch applies: drop it**,
and round 43's null result survives the scale test that round 43 could not perform (it ran on 2 items, which
rounds 64–65 showed cannot discriminate anything).

**Why this negative is worth more than the others.** Every previous conditioning lever — cross-attention gain,
the pooled conditioning token, the plan, the residual plan, logit-normal `t`, more data — was ambiguous in one
way or another: measured on too few items, or confounded with architecture, or inside a noise floor nobody had
quantified. This one is clean: a *pretrained* text encoder carrying a working text → acoustic mapping, dropped
into the flow, produces no measurable change in how much the velocity field uses the text. The field is not
short of a text representation. It is not using one.

**Together with §96 and §94, the conditioning question now has a comprehensive answer at this scale.** Six
interventions tested (gain, pooled token, plan, residual plan, `t`-sampling, pretrained text encoder) plus a
4.2× corpus expansion, and the flow term stays in the 2–7 % band with sampled ρ ≈ 0.09. The measured difference
from SupertonicTTS is not conditioning *mechanism* — those now match the paper one by one (§77) — it is the
**945 h against 8 h** and the compute to use it.

**How complete the A/B actually is**, stated because it is not a clean finish: arm A ran all four checkpoints and
reported ρ = +0.0882. Arm B produced readings at 500/1 000/1 500 — the three that carry the comparison — and then
stopped at ~step 1 900, before its first checkpoint (`--save-every 2000`) and before the post-training ρ, so arm
B's ρ is missing rather than measured. No traceback was written and the wrapper never printed its completion line;
the machine appears to have ended the process. The primary readout is unaffected — that is the gate at the shared
checkpoints, where the arms differ by at most 0.2 points — but a paired ρ comparison is not available and is not
claimed.

## 98. Classifier-free guidance does not help either — and this is the cleanest negative in the project (round 99)

CFG is the standard remedy for weak conditioning, and it was already implemented (config default 1.5) but had
never been *swept* against the calibrating metric. It costs nothing but inference, so the noisy-training excuse
does not apply:

| cfg | frame ρ | output peak |
|---|---|---|
| 1.00 | **+0.0113** | 3.13 |
| 1.50 | +0.0109 | 3.58 |
| 2.00 | +0.0106 | 4.00 |
| 4.00 | +0.0094 | 5.41 |
| 8.00 | +0.0069 | 7.91 |

**Monotone decrease in ρ while the output magnitude grows 2.5×.** Guidance amplifies the text direction the field
learned — and that direction is *not* the direction that identifies the utterance. Extrapolating along it makes
the samples louder and no closer to the target.

**Why this one is worth more than the rest.** Every previous conditioning result involved training, and training
carries the ±3–4 point noise floor measured in §96, which left each of them arguable. This one is inference-only,
five weights, one checkpoint, same probe items, same seed: nothing but the guidance weight changes, and the effect
is monotone over an 8× range. It closes the last inference-side explanation for the flow's ρ ≈ 0.01 — the problem
is not that the field's text influence is under-weighted at sampling time; it is that no text-conditional
direction was learned during training.

**The conditioning question is now closed at this scale**, with eight interventions: cross-attention gain, the
pooled conditioning token, coarse-to-fine plan conditioning, the residual plan, logit-normal `t` sampling, a
4.2× larger corpus, a pretrained text encoder, and classifier-free guidance from 1.0 to 8.0. None moved the flow
term out of the 2–7 % band; none moved sampled ρ above 0.09. What the measurements jointly say is that this
velocity field, at this size and step budget, fits the marginal distribution and does not learn the text →
acoustic mapping — and §77's comparison is the explanation: matching architecture, **945 h against 8 h**.

## 99. The last untested dimension is training time — and ρ is noisy too (round 100)

Eight conditioning interventions are closed and none moved anything, so the remaining untested variable is the
simplest one: **more steps**. Every flow experiment in this project has run 2 000–4 000 steps, while
SupertonicTTS trains its 44M module against 945 h with a schedule to match, and the token route's own history
showed the mapping appearing only after several thousand steps (§48).

So arm A's step-2 000 checkpoint was continued for 8 000 more on the full corpus, gate every 1 000 for eight
points, with the readout pre-registered before the run:

| ρ at step 8 000 | reading |
|---|---|
| **> 0.20** | training time is the lever; the route is plausible with more compute |
| **0.05–0.20** | trending but far from the 0.75 bar; state the slope and extrapolate |
| **≤ 0.05, flat** | the route is closed at this model size, whatever the step budget |

**A calibration fell out of the warm start immediately.** Loading arm A's checkpoint, the continuation measures
**ρ = −0.0277** on its 24 probe items — while arm A itself reported **+0.0882** at the same step. Same model, same
cache, different probe draw: so ρ at this magnitude is *also* noisy, and §96's "first non-zero flow ρ" should be
treated as **not established** rather than as a small positive. The gate already had a measured ±3–4 point floor;
the sampled-ρ readout needs the same scepticism at values near zero, which is where all of them are.

That does not change any conclusion — every one of them was drawn from values inside the noise — but it does mean
the only clean signals in this project are the *decomposed* ones (the gate's flow term against its shuffled
control) and the *monotone* ones (the CFG sweep), not single sampled-ρ readings.

## 100. The readout now carries a spread, and the flow's samples are indistinguishable from zero (round 101)

The noise problem round 100 exposed has a fix, so it was applied rather than documented.
`parakeet/eval/gates.py` gains `sampled_rho_with_error()`: per-dimension correlation between sampled and target
latents over several probe seeds, returning mean, standard deviation, range and a `straddles_zero` flag.
`scripts/cfg_rho_sweep.py` uses it, so the sweep reports an interval instead of a point.

On arm A's step-2000 checkpoint, 5 seeds x 16 items:

  mean +0.0003 | std 0.0129 | range [-0.0178, +0.0147] | straddles zero: True
  per-seed: [-0.0178, -0.0031, -0.0031, 0.0107, 0.0147]

That explains the two contradictory readings as draws from a distribution centred on zero -- arm A's +0.0882 and
the continuation's -0.0277 -- and it means **round 95's "first non-zero flow rho at 2000 steps" is withdrawn**.
With its spread measured, the flow's sampled latents are indistinguishable from zero.

The CFG sweep re-run with error bars says the same at every weight:

| cfg | mean ρ | std | range | verdict |
|---|---|---|---|---|
| 1.00 | −0.0046 | 0.0175 | [−0.0197, +0.0146] | indistinguishable from zero |
| 4.00 | −0.0025 | 0.0083 | [−0.0088, +0.0069] | indistinguishable from zero |

**What this changes is what counts as evidence here**, not any conclusion: every flow result was already drawn
from values inside the noise. The clean signals are the decomposed ones (the gate's flow term against its shuffled
control) and the monotone ones (CFG's decline across an 8x range, inference-only and seed-controlled). A single
sampled-ρ reading is not evidence, and this project has been quoting them.

Recorded honestly: `sampled_rho_with_error` has no unit test. A stub model for it is feasible but fiddly (it
calls `conditions`, `text`, `predict_latent_frames` and `vf`), so it is verified by the measurement above
rather than by a test -- weaker than the rest of this suite and said out loud.

## 101. Training time: the interim readout, and why the run continues rather than being cancelled (round 103)

The 8 000-step continuation of arm A (effective 2 000 + run steps) has produced its first two data points:

| effective steps | flow-term gate | sampled ρ (mean ± std, 3 seeds) |
|---|---|---|
| 2 000 (arm A) | +6.6 % at its own step 1 000 | +0.0003 ± 0.0129 (5 seeds, §100) |
| 3 000 | **+1.7 %** | — |
| 4 000 | — | **+0.0045 ± 0.0395**, range [−0.0294, +0.0479] |

Three more gate readings have landed since, each 1 000 run-steps apart (round 105):

| run step | flow-term gate | total | plan |
|---|---|---|---|
| 1 000 | +1.7 % | +14.4 % | +23.6 % |
| 2 000 | −2.7 % | +2.5 % | +6.4 % |
| 3 000 | **−0.0 %** | +5.3 % | +8.4 % |

Mean +0.3 % against a measured ±3–4 point per-checkpoint floor: no trend, and nothing beyond what arm A showed
at 500-step resolution before this run began. The gate has now been sampled eight times across two runs on this
corpus, always inside the same band.

**The rate was mis-estimated a third time**, which is worth recording as a pattern rather than an anecdote: I
projected the 4 000-step checkpoint for ~17:30 and the run reached step 2 900 at 17:32, i.e. ~2.8 s/step against
the 1.8 measured on arm A with identical settings. The difference is conditions — arm A's timing was taken on a
quiet machine, this run has shared the CPU with dev probes. All three optimistic estimates (rounds 88, 92, 105)
came from measuring under conditions other than the run's, and "seconds per step" is evidently not a property of
the configuration.

Both readouts are in the same place they have been all along: the gate inside the 2–7 % band, and ρ
**indistinguishable from zero** with its spread measured.

**Why not cancel it here.** I have cancelled runs on decisive evidence twice (§94), and the pre-registered rule
for this one keys its three branches to **step 8 000**. At 4 000 steps the point estimate is +0.0045 — safely in
the ≤ 0.05 branch — but the *upper* end of its own interval is +0.048, sitting on the branch boundary, and the
movement from 2 000 to 4 000 (+0.0003 → +0.0045) is a small positive slope. A 4 000-to-8 000 continuation could
therefore plausibly land in the 0.05–0.20 branch, and cancelling now would be reading a trend as an endpoint —
the mirror image of the mistake the pre-registration exists to prevent.

So it runs on. The cost is ~2.5 CPU-hours; the alternative is a conclusion drawn from an extrapolation of two
points whose intervals overlap the boundary.

## 102. Training time: tested, and the last untested dimension is closed (round 108)

The 8 000-step continuation was stopped at run-step 4 000 (effective 6 000 from arm A) because the pre-registered
ρ branch was determined. Both readouts, complete:

| effective steps | flow-term gate | sampled ρ (mean ± std) |
|---|---|---|
| 2 000 | +6.6 % (arm A's step 1 000) | +0.0003 ± 0.0129, 5 seeds |
| 3 000 | +1.7 % | — |
| 4 000 | −2.7 % | +0.0045 ± 0.0395, range [−0.029, +0.048] |
| 5 000 | −0.0 % | — |
| 6 000 | **+7.9 %** | **+0.0045 ± 0.0081**, range [−0.004, +0.013] |

**The verdict is the rule's third branch: ρ ≤ 0.05, flat — the route is closed at this model size, whatever the
step budget.** The reason the endpoint was not needed: ρ moved **0.0000** between effective 4 000 and 6 000, and
its interval's upper bound at 6 000 (**+0.013**) sits **4× below** the branch's 0.05 threshold, so 4 000 more
steps cannot change the branch. The 8 000-step endpoint was pre-registered and would have been run if the
interval overlapped the boundary — that is what happened at 4 000 (+0.048), which is precisely why the run was
continued then and stopped now.

**Two things worth recording about the gate's behaviour.** It swung across the whole band — +1.7, −2.7, −0.0,
+7.9 — including the highest reading this project has ever measured (+7.9 %), *with no corresponding change in
ρ*. That combination is not a contradiction: loss and ρ have been measured as decoupled throughout (§45), and the
gate has a ±3–4 point per-checkpoint floor (§96). It is the cleanest illustration yet of why the ρ readout, with
its interval, is the one that decides, and the gate is the cheap early indicator that tells you when to bother
measuring.

**So the complete map of what has been tested and closed on the conditioning question, at this scale:**

| lever | outcome |
|---|---|
| cross-attention gain, pooled conditioning token | no effect |
| coarse-to-fine plan, residual plan | no effect |
| logit-normal `t` sampling | no effect |
| corpus ×4.2 (narration-style) | no measurable change |
| pretrained text encoder (ρ 0.37 source) | arms track within 0.2 points |
| classifier-free guidance 1.0 → 8.0 | ρ falls monotonically while output grows 2.5× |
| **training time (this round)** | **ρ flat at ~0.004 with interval 4× below threshold** |

Seven classes of intervention, none of which moves the velocity field's use of the text. What the architecture
comparison leaves is the one axis that differs from the paper: **945 h against 8 h**, and the compute to use it.

## 103. The flow's own ceiling, and it reaches about 1 % of it (round 109)

`scripts/target_ceiling.py` already measured SAME TEXT / DIFFERENT VOICES agreement — 0.248 token, 0.171 frame —
which has been read since round 61 as "the target's text-determined share". But **the flow is given the voice as
well as the text**, so the ceiling that actually applies to it is the agreement of two renditions sharing *both*.
One added arm (four lines) measures it:

| pair type | token ρ | frame ρ |
|---|---|---|
| SAME TEXT, **different** voices | +0.248 (n=108) | +0.171 |
| SAME TEXT, **SAME** voice | **+0.723** (n=4) | **+0.445** |
| DIFFERENT text (control) | +0.035 (n=120) | +0.045 |

**The flow's applicable ceiling is 0.445, and it achieves 0.0045 — about 1 % of it, on the very items its ρ is
measured on** (the probe uses training items). This is a different and much sharper statement than "the target is
one-to-many, so a deterministic model cannot do better", which is how the plateau has been explained:

* it does **not** say the architecture is wrong. The field fits its objective (flow loss 2.8 → 1.2) while its
  samples carry no target identity — the marginal-first behaviour measured since §56;
* it **does** rule out "there is nothing here to learn": there is an order of magnitude of headroom, on data the
  model has seen, from information it is given;
* it sharpens what remains: 4× the steps did not move ρ (§102), 4.2× the data did not (§94), and neither did any
  of eight conditioning interventions. **Optimisation is now the only open question.**

Caveats, since the sample is small and the metric is a sampler: n=4 same-voice pairs, and ρ comes from 4-NFE
samples, so sampling noise degrades it independently of how well the field is fitted. Both push toward the ceiling
being *under*-estimated, which makes the 1 % gap conservative.

One number deliberately not revised: the token route's 0.723 ceiling is for **seen** texts while its 0.37
validation figure is on held-out texts, so they are not comparable and the plateau's explanation stands.

## 104. The sampler is not the problem — and ρ against one recording is the wrong metric for a sampler (round 110)

**NFE does nothing.** Swept the sampling steps on the most-trained flow checkpoint (3 seeds × 6 items):

| NFE | mean ρ | range | verdict |
|---|---|---|---|
| 4 | +0.0257 | [+0.0036, +0.0463] | clear of zero (small sample) |
| 8 | +0.0012 | [−0.0105, +0.0126] | indistinguishable from zero |
| 16 | +0.0040 | [−0.0296, +0.0277] | indistinguishable from zero |
| **32** | **+0.0031** | [−0.0123, +0.0112] | indistinguishable from zero |

So the velocity field's **exact ODE solution** — NFE 32, full integration, the setting the paper prefers — is as
uncorrelated with the target as the 4-step one. The NFE-4 measurements were not the artifact I went looking for.

**And the metric is wrong for a sampler — which corrects a lot of this project's wording.** ρ is measured between
*one* sampled latent and *one* target utterance. A flow conditioned on (text, voice) is supposed to draw **one of
many** valid utterances for that pair, so a well-trained sampler will legitimately produce a different valid
rendition and score ~0 against that particular recording. **ρ-against-the-target is a metric for a deterministic
regressor; applied to a sampler it conflates "a different valid draw" with "wrong".**

That means every "the flow's ρ is ~0" statement in this log and the README is *not* evidence that the flow draws
invalid audio. It is equally consistent with drawing valid audio that happens not to be that recording.

**What does establish that the samples are wrong is the measurement independent of which valid draw you get: WER.**
The flow's audio transcribes at **1.000** against the teacher's **0.000** — unambiguous, because unintelligible is
unintelligible regardless of which utterance was intended.

**Consequent correction to §103.** That section compared the flow's ρ (0.0045) with a ceiling of 0.445 and concluded
it reaches "about 1 % of its ceiling". By the argument above the comparison is invalid for the same reason: 0.445
measures how strongly two renditions of one (text, voice) agree, and a sampler is entitled to produce a third
rendition that disagrees with both while remaining valid. Corrected statement: **the flow reaches ~1 % agreement
with the specific recordings it was trained on, which for a sampler is not yet evidence of failure.** The evidence
of failure is WER 1.000. Round 109's open question survives only in the weaker form: the field's conditional signal
is small, *and* the sampler's output is unintelligible by the metric that can tell.

The bottom line is unchanged — size and speed solved and measured, the speech unintelligible by WER — but the
correct next measurement for any flow attempt is **WER on its sampled audio**, not ρ against a recording.

## 105. The right metric, applied to the most-trained flow: WER 1.000 (round 111)

Round 110 established that ρ-against-a-recording cannot decide anything about a sampler and that WER can. This
applies it to the best flow checkpoint this project has — effective 6 000 steps of training on the ~8 h corpus,
sampled at **NFE 32** (full integration, the paper's preferred setting) — with the teacher control:

```
student WER 1.000   (faster-whisper base.en @16k)
teacher WER 0.000   <- control: the recogniser works on this text
```

All five checks pass: synthesis produces audio, DNSMOS and WER are available, the naturalness metric separates
teacher from student, and the recogniser transcribes the teacher. The instrument works and the answer is
unambiguous.

**Why this closes the loop.** Round 110's correction left one legitimate doubt: if the flow draws a different valid
rendition each time, a low ρ proves nothing — perhaps the audio is fine and merely not that recording. WER is
draw-independent, so the flow's audio being unintelligible is a real failure and not a metric artifact.

**The complete, corrected picture at this scale:**

| | |
|---|---|
| **what works** | autoencoder round trip WER 0.000; 9.63M parameters; 85× real time in int8 at 20.3 MB |
| **what fails** | text→acoustic *sampling*: WER 1.000 at NFE 32, after 6 000 effective steps on 8 h of audio |
| **why (measured)** | the velocity field's own objective is 2–7 % text-dependent; 4.2× data, 4× steps, 4× sampling steps and eight conditioning interventions all leave it there |
| **why (compared)** | the architecture matches SupertonicTTS axis by axis (§77); the one axis that differs is **945 h against 8 h** |

What to measure next, stated so it is not re-litigated: **WER on sampled audio with the teacher control**, on any
future flow attempt — not ρ against a recording.

## 106. The text's share of the target is 43 % on the flow's own representation — and my first reading was wrong (round 113)

A variance decomposition the project had never computed: add text-blind predictors in the order a marginal-first
model learns them, and see what is left for the text (`scripts/latent_variance_budget.py`).

**On the unfolded latent** (24 dims at 93.75 Hz — what the cache stores):

| predictor | R² | added |
|---|---|---|
| global per-dim mean | 0.201 | +0.201 |
| + per-voice mean | 0.402 | +0.202 |
| + per-utterance DC | 0.411 | +0.009 |
| + AR(1) smoothness | **0.914** | +0.502 |
| **residual** | | **0.086** |

Read alone that says "the target is 91 % predictable without text, so a low text dependence is expected". I wrote
that conclusion, then measured the same thing on the right representation.

**On the folded target the flow actually models** (144 dims at 15.6 Hz, `compress=6`):

| predictor | R² |
|---|---|
| per-voice mean | 0.404 |
| + AR(1) smoothness | 0.566 |
| **residual for the text** | **0.434** |

**Adjacent frames at 93.75 Hz are nearly identical**, so a one-step AR predictor captures 50 points of variance —
but that predictability is an artifact of the frame rate, and the flow never sees it: `fold_time` redistributes
those six near-identical frames across 144 dimensions, where the same predictor gets 17 points. The honest figure
is **43.4 % of the target's variance left for the text**, not 8.6 %. The measurement was right both times; the
first answered a question the flow does not face.

**What this settles.** The field uses 2–7 % of its objective on the text while **43 %** of the target's variance is
available beyond voice and smoothness — so "the low text dependence is simply the target's nature" is **refuted by
measurement**, with roughly **6–17× more conditional signal than the field uses**. Combined with WER:

* the conditional signal exists and is large (43 % of variance);
* the field uses a small part of it (2–7 %);
* the sampler's output is unintelligible anyway (**WER 1.000 at NFE 32**, §105);
* and data ×4.2, steps ×4, NFE ×8 and eight conditioning interventions move none of it.

So the lever is **not** the target representation — which is where the first reading would have pointed, and where
§68's retraction already pointed. It is whatever it takes to fit a large conditional signal in a 21–39M flow on
this corpus: the paper's regime of 945 h and the compute to use it.

## 107. The text contributes ~nothing at *every* timestep — and the gate was inflated by dropout (round 114)

**Per-timestep.** Replicated `flow_loss` exactly (`fold_time`, `make_xt`, `v_target = x1c - x0`) but with `t`
supplied explicitly, model in eval mode, the same `x0` for the right and shuffled passes:

| t | loss right | loss shuffled | difference | share |
|---|---|---|---|---|
| 0.05 | 0.8580 | 0.8611 | +0.0031 | **+0.4 %** |
| 0.20 | 0.5504 | 0.5519 | +0.0016 | +0.3 % |
| 0.40 | 0.4390 | 0.4404 | +0.0014 | +0.3 % |
| 0.60 | 0.4818 | 0.4879 | +0.0061 | +1.3 % |
| 0.80 | 0.6461 | 0.6523 | +0.0061 | +1.0 % |
| 0.95 | 1.0707 | 1.0819 | +0.0112 | +1.0 % |

The text contributes **0.3–1.3 % at every timestep**, so **no loss reweighting or curriculum can create it** — a
t-weighted objective is ruled out *before* spending a GPU hour on it, which is the whole point of measuring rather
than trying. The loss shape is visible too: 0.858 at t=0.05, minimum **0.439 at t=0.40**, 1.071 at t=0.95 — the
expected U.

**And the correction.** `text_dependence()` ran the model in **train** mode, and `flow_loss` applies **10 %
conditioning dropout** when training — so its "right text" pass was polluted by dropped conditioning on one batch
in ten, inflating the right-vs-shuffled gap. That is where the "2–7 %" band quoted in this log, the README,
`PLAN.md` and every summary came from. Fixed: the gate now measures in eval mode, restores the previous mode, and
reports `measured_in_eval_mode`. Re-measured on the same checkpoint: **+1.9 %**, consistent with the per-`t`
figures, against the +2.4 to +7.9 % the train-mode version reported.

No conclusion changes — both numbers say "small" — but every "2–7 %" in this project's docs was the inflated one,
and the honest figure is **~1–2 %**.

**What it adds up to:** the field's objective is ~1–2 % text-dependent *uniformly across the trajectory*, on a
target whose text-determined share is **43 %** of the variance (§106). The gap is larger than the inflated number
suggested, the ruled-out interventions are ruled out at every timestep rather than on average, and what must change
is **what the field is asked to do** — not how hard it is pushed, nor where in `t` it is pushed.

## 108. The plan route is dead — and that points at the one structural change with a measured floor (round 116)

Before spending 3–9 CPU-hours training the residual/plan form, its ceiling is free to measure: unfold the plan to
frames and correlate it with the target (`scripts/plan_ceiling.py`). I wrote in the script's own docstring that the
plan carries "the token route's ρ 0.37"; the measurement says otherwise:

```
plan vs target tokens : rho -0.0137
plan vs target frames : rho -0.0111   (median -0.0099, max +0.0160)
```

**The plan head has learned nothing.** The 0.37 belongs to the **token route** — a separately trained model (the
`distill-text` stage). The flow's own `plan_head` is a small auxiliary head supervised by `plan_weight` against the
*same* cached token latents, and it reaches ρ ≈ 0. Its loss falls to ~0.7 while its predictions stay uncorrelated
with the target: it minimises MSE by predicting the mean — the identical marginal-first failure measured everywhere
else in this project, now reproduced in a head with a fixed, supervised, well-posed target.

**Consequences:**

1. **The residual/plan route is dead** — and cheaply. `flow.plan_residual` asks the field to model `x1 − plan`; if the
   plan is uncorrelated with the target, that is just `x1` with extra noise. This closes the one structural idea the
   project had implemented but never judged by a sampler's metric, for two minutes instead of a training run.
2. **It was not an under-measurement.** I half-suspected the residual form was mis-read (round 64 judged it by ρ on 2
   items). It was not — the plan it depends on is empty.
3. **The comparison is the interesting part.** The *same* task — predict cached token latents from text — reaches
   **ρ 0.37** in the token route and **ρ ≈ 0** in this head. Same corpus, same targets, comparable capacity. The
   difference is **where the encoder is trained**: the token route's text encoder is trained against that regression
   and nothing else, while the flow's is trained only through the flow objective — which the marginal satisfies. *A
   text encoder trained through a marginal-first objective learns nothing a head can use.*
4. **So the structural change with a mechanism and a measured floor** is to take the plan from a model that *has*
   learned the mapping rather than from the flow's dead head:

   ```
   x1_residual = x1 - unfold(token_route(ids))     # rho 0.37, measured, checkpoint on disk
   flow models(x1_residual)                        # the field refines a 0.37-conditioned prior
   ```

   The floor is **0.37, measured not assumed**, and the field can only add to it. It is the first structural proposal
   here whose floor is above zero — and it explains why warm-starting the flow's encoder from the token route did
   nothing (§96): the encoder was never the bottleneck *at sampling time*; nothing forces its output into the
   sample. A residual built off that encoder does.

**Not claimed:** that 0.37 is enough. It is below the 0.75 intelligibility bar, so this route needs the field to add
substantially — which is exactly what the field has so far failed to do for anything. But it is the only route left
with a measured, nonzero floor, and establishing that floor took two minutes.

## 109. The residual route's floor is 0.406, measured — and the token route is at 91 % of its ceiling (round 117)

> **The 91 % figure in this section is superseded — see §119.** 0.445 is the *reliability* (how much two renditions
> of the same text and voice agree), not the ceiling that applies to a predictor of the shared signal: that is its
> square root, **0.667**, and the recogniser's fail point (0.60) lies **below** it. The token route is at **61 %** of
> the applicable ceiling, with **+0.261** of headroom, and **is not excluded** — the opposite of the pessimistic
> reading this heading invites. Nothing else in the section changes.

Round 116 proposed a residual against the token route's output and quoted a floor of 0.37 — which was the token
route's **token** ρ. A residual route inherits its **frame** ρ, never measured until now. Two minutes:

```
token route vs target FRAMES : rho +0.4064   (median +0.3565, max +0.8081)
token route vs target TOKENS : rho +0.4155
```

Measured **with the true durations supplied**, so this asks whether the predicted latents track the target frames
rather than whether the duration head is accurate — and a residual route would use predicted durations, which can
only lower the floor, never raise it.

**The comparison this makes possible**, all on the same corpus and items:

| quantity | value | share of the ceiling |
|---|---|---|
| ceiling for a model with text+voice (two renditions agreeing, §103) | 0.445 | — |
| the **token route's** frames | **0.406** | **91 %** |
| the **flow's** samples | 0.0045 | 1 % |

So the deterministic route — described in this project since round 61 as "at its ceiling, explained by the
one-to-many mapping" — is at **91 %** of the ceiling that actually applies to it. That description was written
against a 0.402 figure and a *different* ceiling (the cross-voice 0.171): 0.402 was validation ρ on held-out texts,
0.171 was the text-only share. Against the text+voice ceiling — the information set *both* models have — the token
route is nearly saturated and the flow is nowhere near it.

**What this does and does not change:**

* it does **not** make the token route intelligible: 0.406 is far below the 0.60 where the recogniser fails
  outright and the 0.75 where it reads a latent perfectly, and its own WER is 1.000. The plateau reading stands as
  a statement about quality;
* it **does** fix the residual route's floor at **0.406, measured** rather than 0.37 assumed, and makes the route's
  shape concrete:

  ```
  plan        = token_route(ids)                    # rho 0.406 frames, checkpoint on disk, no training needed
  x1_residual = x1 - plan                           # what the flow models instead of x1
  sample      = plan + flow_field(x0 -> residual)   # the output starts at 0.406 and the field can only add
  ```

  Even if the field adds nothing — which is what fields have done in this project — the samples would score **0.406
  instead of 0.0045**, a **90×** move on the metric that tracks the target, and the field's job becomes modelling a
  residual whose text-conditional part is carried by something that demonstrably learns.

For the honest record: this is the first *measured* route in this project with a nonzero floor, and the two
measurements that established it (the plan head is empty, the token route's frames are not) cost under five minutes
between them — both taken **before** building anything.

## 110. The plan removes 2 % of the energy but tracks at 0.406 — which reconciles the gate with ρ (round 118)

Building the residual route's cache (`scripts/make_residual_cache.py`) surfaced a design error and then a
conceptual one.

**The design error.** Subtracting the plan *unscaled* made things worse: the residual carried **2.16× the target's
energy**, because `energy(x − p) = energy(x) − 2cov + energy(p)`, and a predictor with ρ 0.406 at comparable
variance *adds* variance. Fixed by fitting the per-dimension regression coefficient `α = cov(x,p)/var(p)` before
writing anything — which minimises the residual energy and leaves the floor's ρ unchanged, because correlation is
scale-invariant. The sample still starts at 0.406, from a well-conditioned residual.

**The conceptual one, which is more valuable.** With `α` fitted, the plan removes only **2.1 % of the target's
*energy*** — while its per-dimension *correlation* is **0.406**. Those look contradictory and are not:

* ρ is scale-free: the plan tracks the target's **shape** at 0.406;
* the energy is dominated by a few high-variance dimensions the text does not predict, so the variance-weighted
  R² is ~0.02 even though the unweighted mean correlation is 0.41.

**Why this matters, and what it reconciles.** The gate is a **loss**-based measure — MSE, hence energy-weighted —
and it has said 1–2 % throughout. The best text predictor available explains **2.1 % of that same loss**. So:

* the field is **not** "ignoring" the text in any useful sense: there is only ~2 % of the loss to capture and it
  captures 1–2 % of it; measured against the loss, it is already near optimal;
* and that is exactly why the gate could never show the improvement that matters — the loss is energy-weighted,
  while intelligibility follows the **structure**, which is what ρ measures, and the project's own calibration says
  so (ρ ≥ 0.75 reads perfectly, 0.60 fails outright).

The two metrics were never in conflict; they measure different things, and the one that predicts audibility is ρ.
My own summary language has been sloppy about this for many rounds — *"the field barely uses the text"* is a
statement about the **loss**, and the loss was never where the text's contribution lives.

**What it implies for the residual route:** it stays promising, for the reason §109's floor suggested — the route's
benefit shows up in **ρ** (0.406 floor against the flow's 0.0045), not in the loss, so a gate reading would have said
"no change" even if it worked. Its evaluation must be **ρ and WER on `plan + field`**, never the gate.

## 111. The residual route is closed by arithmetic, before spending the CPU (round 119)

Every number needed to bound it was already measured:

| quantity | value | source |
|---|---|---|
| the token route's frames — the route's **floor** | ρ **0.406** | §109 |
| the text+voice reproducibility ceiling | ρ **0.445** | §103 |
| the deterministic route's fail threshold | ρ **0.60** | round 40 |
| what any field trained here has ever added | ~**nothing** | everywhere |

**The ceiling is below the fail threshold.** The plan already uses text *and* voice — that is its whole input — and
the field's inputs are the same two things. So the field cannot add information the plan does not have, and the
route's ceiling is the reproducibility of the target from (text, voice): **0.445**. A deterministic route reads a
latent perfectly at 0.75 and fails outright at 0.60, so 0.445 is not "a bit short of intelligible" — it is *below
the level at which the recogniser already fails*.

The floor is 0.406 and the ceiling 0.445: a **0.039 window**, against a field whose measured contribution to
everything else here has been indistinguishable from zero. Training it for 3–9 CPU-hours to sample somewhere inside
that window would move WER from 1.000 to 1.000.

**What the route would have bought, stated fairly:** ρ 0.0045 → ~0.4, a **90×** move on the metric this project has
used to track target agreement — a real improvement in a metric, and (per §110) the wrong metric: intelligibility
follows *structure*, and a sampler draws a different valid rendition rather than reproducing the recording (§104).
So the 90× would have moved a number that does not predict audibility, at the cost of a training run.

**A category error now named**, because it has sat in this project's framing for dozens of rounds. The
"ρ ≥ 0.75 to be intelligible" bar was calibrated on a **deterministic regressor judged against the recording**
(round 40). For a **sampler** it is not a meaningful target: a sampler that produces a perfectly valid, perfectly
intelligible utterance still scores ~0 against a *particular* recording of the same sentence. That is why the flow
has read "0.0045 against a 0.75 bar" for eighty rounds without its audio being any worse than the token route's
0.406 — the bar does not apply to it. The metric that applies to a sampler is **WER**, and both routes sit at
1.000.

**So the remaining path is only scale**, and this closes the last cheap structural idea rather than deferring it:
the architecture matches SupertonicTTS axis by axis (§77); the conditional signal exists (43 % of variance, §106)
and the deterministic route already extracts most of it (61 % of the *applicable* ceiling — 91 % of the
reliability, which is not the same thing, §119); the flow's field
fits the marginal and no intervention moves it; and both routes produce unintelligible audio at WER 1.000. The
difference from the paper is **945 h against 8**, and the compute to use it — which `docs/PLAN.md` now costs, with
three ordered runs and their criteria.

## 112. The dialogue domain leaves 1.75× more for the text — and the merged cache has a voice-mapping bug (round 121)

**1. The premise holds.** Both caches measured **raw**, same statistic, folded target (144 dims, what the flow
models):

| corpus | per-voice R² | + AR(1) | **residual for the text** |
|---|---|---|---|
| **dialogue** (1 461 turns, 1.8 h) | 0.484 | 0.626 | **0.374** |
| **narration** (`expanded_v3`) | 0.713 | 0.787 | **0.213** |

The dialogue domain's marginal — who is speaking, plus temporal smoothness — explains **62.6 %** of the target
against narration's **78.7 %**, so the share left for the text is **37.4 % against 21.3 %**. The hypothesis is that
a *harder* marginal makes the text worth learning; this measures its premise directly, and it is **confirmed**: the
dialogue set leaves **~1.75× more conditional signal**. The prediction for a training run is therefore concrete —
the flow term should rise on dialogue data — and it did not on narration.

Also a correction: §106 reported the narration residual as **0.434**, measured with per-voice normalisation applied
to a cache whose table turned out to be partly ineffective (below). Raw, the honest figure is **0.213**.

**2. The merged cache's voice mapping is wrong for some voices.** Probing `expanded_v3` item by item:

| voice | pooled std | \|per-dim mean\| | |
|---|---|---|---|
| 0 | 1.5943 | 1.1205 | **not normalised** |
| 1 | 1.0017 | 0.0187 | normalised |
| 2 | 1.4459 | 1.0050 | **not normalised** |
| 3 | 0.9973 | 0.0085 | normalised |

An **alternating** pattern, which *looked* like a permutation mismatch — and round 121 reported it as one. **That
claim is withdrawn** (§114): the direct check refutes it, and the real cause is different and more interesting.

**Round 122's check.** The base cache's `voice_names` matches the manifest's order of first appearance exactly, and
comparing **every** cache item's `voice` index against the manifest index of the corresponding record:

```
1207 items checked, 0 mismatches
```

**What the pattern actually is.** The alternation is not parity — it is **source**. `expanded_v3` was built as
`(mixed_v2 + kokoro_v3) + kokoro_v3b`, and:

* voices appearing in **both** the base corpus and the Kokoro expansion (`af_alloy`, `af_aoede`, `af_heart`, …) have
  their items at **two different scales** — the round-58 phenomenon — so a single pooled per-voice statistic is a
  *mixture*, normalising by it leaves each half off scale, and pooling the halves gives a std *larger* than either
  (the 1.59);
* voices appearing in **one** source only (`alicia`, `alec`, `alfonso` — all Speechify) have one scale, normalise
  cleanly, and sit at 1.00.

So the defect is real but it is **not** a mapping bug: **a per-voice statistic cannot normalise a voice whose items
come from two sources at different scales.** Per-voice normalisation was the right fix for the round-58 gap *between
corpora*; it is insufficient for a gap *between sources within one voice*, which is what a merge creates.

**Corrected blast radius:** the dialogue cache is single-source, so all eleven of its voices sit at std ~1.00 with
|mean| ~0.01 — the §112 premise result (37.4 % vs 21.3 %, measured **raw** on both sides) stands unqualified. The
narration numbers measured *with* normalisation applied are the affected ones: §106's 0.434 was one and is
superseded by the raw 0.213. Runs trained on `expanded_v3` (rounds 91–118) carried a scale/DC nuisance for the
voices that span sources — not corrupted targets, and within-run comparisons are unaffected.

## 114. The mapping-bug claim, withdrawn (round 122)

Round 121 reported a permutation mismatch in the merged cache's voice mapping and said every run since round 91 was
affected. **Withdrawn**: the direct check takes one minute and refutes it (1 207 items, 0 mismatches, above).

The real cause is the source-mixture effect described in §112, which *looks* like a permutation from a statistical
fingerprint: voices spanning two sources sit at a mixture scale, voices in one source do not, and the two alternate
in the merged table. Reporting the bug and its blast radius from the fingerprint — instead of from the direct check
against the manifest the cache was built from — is the third time this session the same mistake has appeared, and
the fix is the same each time: **compare the artefact against its own source before describing what is wrong with
it.**

## 115. The source-aware normalisation fix (round 123)

§112/§114 diagnosed a real defect and left it in place: a per-voice statistic cannot normalise a voice whose items
come from two source caches at different scales, which is what a merge creates. Fixed, in four places:

* merge_caches writes `item["source"]` (0 = base, 1 = extra, ...) — one int per item, and the thing that makes
  the correct statistic expressible at all;
* it_voice_norm returns a **second** table, oice_source_norm, keyed `"<voice>|<source>"`, produced **only**
  for voices that actually span more than one source — so single-source voices and every existing cache are
  unaffected;
* LatentShardDataset reads it and prefers it per item, falling back to the per-voice entry when there is no
  source tag or no matching pair;
* xpand_corpus writes both tables into the merged cache's meta.

Plus hardening: a malformed oice_norm entry (not a dict) is skipped rather than crashing the dataset with an
AttributeError — which is exactly what the signature change did to an existing test before it was updated.

**The test** pins both halves with a synthetic cache holding **one voice at two scales** (0.5 and 1.5 — the real
situation): the fit must produce per-source statistics with the right per-source stds *and*, once written, each
source must come back at pooled std ~1.0. Without the fix the two halves cannot both be normalised, which is the
defect stated as an assertion. Suite 344 → **345**.

**Still to do, so it is not mistaken for done:** uns/expanded_v3 has no source tags — it was merged before
this change — so the fix only takes effect for caches built by the updated chain, and that cache needs rebuilding
(~15–20 minutes) before any measurement that depends on its normalisation is trustworthy. The dialogue cache is
single-source and unaffected either way.

## 116. Per-source normalisation: applied, and one level deeper than the first attempt (round 124)

The §115 fix was in code but not in any cache, so this round rebuilt the chain. The rebuild immediately exposed a
**second copy of the same defect**, one level down: a single-int source tag is overwritten by the next merge, so
(base + kokoro_v3) + kokoro_v3b collapsed base and kokoro_v3 back into one source — step 1 produced 4
(voice, source) pairs and step 2 would have produced 4 again instead of 24. The tag is now a **compound path**
("0", then "0>0" / "0>1"), and each item's tag describes its own provenance rather than its position in the
last merge.

**Evidence from the rebuilt chain:**

| step | result |
|---|---|
| 1: mixed_v2 + kokoro_v3 | 1 932 items, **4** (voice, source) pairs |
| 2: + kokoro_v3b | 4 533 items, **24** (voice, source) pairs |

Both verify at 0.0419, and in the finished cache the 32 groups with n ≥ 4 all pool at std 1.000. The regression
test pins it on a synthetic cache holding **one voice at two scales** (0.5 and 1.5): the per-source statistics must
carry the right per-source stds *and* each source must come back at ~1.0 — impossible when the per-voice mixture is
used.

Also fixed en route: the dataset's table parser coerced the source to int while the per-item lookup built a
str, so **every** per-source entry was silently missed. The regression test caught it immediately.

**What this does not show, stated rather than implied:** that the original 1.59-on-real-data symptom is gone
*because of* this change. uns/expanded_v3 was refitted earlier in the session with the round-78 fix and the
pooled-std probe reads 1.000 on it too — but that statistic is partly **tautological**, since measuring the pooled
std of the very group the fit pooled returns 1.0 by construction. The clean evidence is the synthetic test (which
discriminates) and the 24 per-source entries that did not exist before. The original 1.59 came from a cache state
that no longer exists and cannot be re-measured.

**Going forward:** uns/v3_fixed — 4 533 items, same size, per-source normalisation — replaces xpanded_v3 as
the merged corpus.

## 117. The residual route's closure cited a threshold I had just declared inapplicable (round 129)

Round 119 closed the residual route by arithmetic — floor ρ 0.406, ceiling ρ 0.445, deterministic fail threshold
ρ 0.60, so its whole 0.039 window sits below the level where the recogniser already fails. **The same round also
established that the ρ bar was calibrated on a deterministic regressor judged against a particular recording and
does not apply to a sampler** — which the residual route is. So the closure cited a threshold that, by its own
round's finding, does not govern the thing being closed. **The conclusion survives; the reasoning did not.**

**The route is closed by the metric that applies.** Its floor is the token route's own audio — if the field adds
nothing, the sample *is* the plan — so the floor's WER decides. Measured fresh in 34 seconds (6 utterances,
faster-whisper base.en):

`
token route (distill-text step 5000)
  student WER 1.000
  teacher WER 0.000   <- control: the recogniser works on this text
`

Even the route's **floor** is unintelligible by the metric that governs a sampler, and the field would have to add
the entire difference — when its measured contribution to everything here has been indistinguishable from zero
across eight conditioning interventions and two architectures.

**Supported independently, and already:** §110's energy measurement — the plan removes only **2.1 %** of the
target's energy, because the energy is dominated by high-variance dimensions the text does not predict. A residual
still carrying ~98 % of the energy is not a materially easier task for a field that has never learned the easier
one.

Same conclusion as §111, now resting on WER rather than on a threshold that does not apply. Pinned as a model-card
claim (**41 verified, 0 failing**) with its report committed as evidence, so the number the closure depends on is
reproducible rather than quoted.

## 118. The domain A/B's instrument spread, measured before the result (round 130)

Built scripts/analyse_domain_ab.py to compare the two arms against the prediction registered before the runs, and
to report each arm's **own** spread — because that spread *is* the instrument's noise.

**First two points, arm N (narration):** +3.2 % at step 500, +0.5 % at step 1000 ⇒ mean +1.85, **spread 2.7
points**, sd 1.91.

Within **one** arm that has not changed what it is training on, the gate moves **2.7 points** between checkpoints —
against a predicted domain effect of ~1.5–2 points. This is the concern written into the pre-registration *before*
any result (§ "Domain A/B" in docs/ROADMAP.md): the in-loop gate ran epeats=2 against a ±3–4-point floor. It
is now **measured** rather than suspected.

**What that does and does not mean:**

* it does **not** invalidate the premise — that was measured directly and independently of any training (37.4 %
  against 21.3 %, §112) and is in the model card;
* it **does** mean the A/B's mean comparison carries an error of roughly sd/√4 ≈ 1 point per arm, so a ~1.5–2-point
  separation is **marginal rather than decisive**, and whichever way it lands the verdict must say so. The tooling
  enforces that: it declares UNRESOLVED whenever the difference is smaller than half an arm's own spread, rather
  than reporting a sign as a finding;
* and it is why the in-loop gate was raised to epeats=5 for future runs — these arms started before that change.

## 119. The ceiling was the reliability, not the true-score bound — the token route has more headroom than §109 said (round 132)

While re-reading §109 to check a *different* argument, the ceiling arithmetic turned out to be wrong — and wrong in
the pessimistic direction.

**What §109 did.** It measured the agreement between two renditions of the same text and same voice — frame ρ
**0.445** — called that "the ceiling that applies", and concluded that the token route at **0.406** sits at **91 %**
of it, i.e. nearly saturated.

**Why that is not the applicable ceiling.** 0.445 is a **reliability**: the correlation between two measurements of
the same underlying signal, each equal to that signal plus independent error. Classical test theory gives the
correlation a predictor of the *shared* signal can reach with a **single** such measurement as the square root of
the reliability:

```
true-score ceiling = sqrt(0.445) = 0.667
```

A model conditioned on (text, voice) has access to exactly the shared signal and nothing else, so **0.667**, not
0.445, bounds it. The recogniser's calibration (§37) puts its fail point at **0.60** and perfect reading at 0.75, so
the bar lies **inside** the range the ceiling permits:

| quantity | ρ |
|---|---|
| token route, measured | 0.406 |
| recogniser fails | 0.60 |
| **true-score ceiling (√0.445)** | **0.667** |
| two renditions agree (the reliability §109 quoted) | 0.445 |

So the token route is at **61 %** of the ceiling that applies, with **+0.261** of headroom — not 91 % and saturated.
**The deterministic route is not excluded by the ceiling**, the opposite of what §109's framing encourages.

**What this does not change.** The flow's samples are at 0.0045 against the same 0.667 — **0.7 %**, *worse* than the
1 % §109 reported, because the reference got larger. And the route-level verdicts rest on measurements that are
untouched: the token route's own audio is **WER 1.000** against the teacher's 0.000 (§117), and the field has not
moved under eight conditioning interventions. What changes is the *reason* the deterministic route might be stuck:
not "there is almost nothing left to learn", but "there are 0.26 points of it, and no headroom argument against
trying".

Corrected in the four other places the claim was published: `docs/ROADMAP.md`, the generated `docs/MODEL_CARD.md`
(via `scripts/model_card.py`), §111's parenthetical, and this section's forward pointer from §109. The README was
checked and does not carry the claim.

**Method note.** This is the seventh self-correction in the span and the second in three rounds. The pattern is
specific: a *derived* number inherited a name ("ceiling") from the measurement that produced it, and the name
outlived the definition. The measurements were right; the label did work it had not earned.

## 120. The ceiling has no measured value — it rests on 4 pairs (round 133)

Last round corrected §109's ceiling arithmetic (0.445 is a *reliability*, so the applicable bound is √0.445 =
0.667, putting the token route at 61 % rather than 91 %). Before acting on the new number I checked **its
precision**, and that makes both figures meaningless:

`
reliability r = 0.445 measured on n = 4 pairs
Fisher-z 95% CI for r:  [-0.902, +0.985]
therefore sqrt(r):      [0.000, 0.992]
`

**Four pairs do not determine a correlation.** The "text+voice ceiling" could be 0 — in which case nothing is
learnable from text and voice — or 0.99, in which case the token route's 0.354 is well short and the flow's 0.0045
is nowhere. Every ceiling figure this project has published, including the one published two rounds ago, is a point
estimate of a quantity that is **not measured**.

**What is actually established, separated from what is not:**

| quantity | value | status |
|---|---|---|
| token route, frames, **40 items** | ρ **0.354** | measured (a 12-item run gave 0.406 — the thin sample was optimistic) |
| token route audio, 6 utterances | WER **1.000** | measured (teacher control 0.000) |
| SAME TEXT across **different** voices, n = 108 | ρ 0.171 | well estimated |
| SAME TEXT, SAME voice, **n = 4** | ρ 0.445 | **not estimated** |
| the ceiling that follows from it | — | **unknown** |

The one same-text number with a real sample size is the **cross-voice** one (0.171, n=108), and it answers a
different question: different voices cannot share a rendition, so it bounds *cloning*, not text conditioning.

**Consequences, which cut toward doing more rather than less:**

* the "the deterministic route is at its ceiling" story — at 91 %, or at 61 % — has **no numerical basis**. Neither
  did "the target is one-to-many, so a deterministic model cannot do better";
* the honest statement is: a text-conditioned route measures ρ 0.354 and transcribes at WER 1.000, and **how much
  more is achievable is unknown**;
* so the cheap experiment is the right one, and it is cheap because the token route trains at ~0.8 s/step against
  the flow's 2.4: **a 5 000-step continuation is in flight** (uns/text_headroom, warm-started from step 5000,
  22.6 M trainable). **Pre-registered prediction: if validation ρ rises above 0.354 the headroom is real and the
  corpus/compute axis is confirmed; if it is flat, the mapping from text to this representation is the wall
  regardless of capacity** — which is what "one-to-many" was asserting without evidence.

**Repeat renditions cannot be manufactured here** to fix the measurement: Kokoro is deterministic and the corpus
has one take per (text, voice). A genuine reliability estimate needs repeated stochastic generations of the same
text and voice — a GPU-and-licence job, not a CPU one.

## 121. The ceiling is 0.845, not 0.445 — the reliability measured on 110 pairs instead of 4 (round 134)

The ceiling that was used to close the deterministic route — §109 quoted 0.445, §119 corrected it to √0.445 = 0.667,
§120 showed it rested on FOUR pairs with a CI of [−0.90, +0.99] — now has a real measurement.

**Method.** Repeat generations do not exist here (Kokoro is deterministic; one take per (text, voice)), so a
**conservative** stand-in was manufactured: 110 sentences with only their **punctuation** changed — identical words,
therefore identical phonetic content, different rendering. Punctuation alters prosody *systematically* rather than
independently, so the rendition difference is at least as large as between independent generations, which pushes
agreement **down**. A lower bound is what a headroom argument wants. The builder also lays the pairs out so the
second block starts at index 110 and **110 % 11 == 0**, making the generator's round-robin put both members of every
pair on the **same voice**, without touching the generator. 220 utterances, one free teacher, latents from the
trained autoencoder (§121's trap avoided).

**Result, 110 pairs, bootstrap CIs:**

| alignment | ρ | 95 % CI | implied ceiling √ρ |
|---|---|---|---|
| zero shift (lower bound) | **+0.7138** | [+0.6714, +0.7511] | **0.845** |
| best shift (upper bound) | **+0.7719** | [+0.7381, +0.8004] | **0.879** |

Median shift **0 frames**; length mismatch mean 0.055 s — so the zero-shift figure is not an alignment artefact.

**This reverses the reading.** The recogniser fails at ρ 0.60 and reads perfectly at 0.75 (§37). The *conservative*
reliability is 0.714 ⇒ ceiling **0.845**, i.e. **above both**. The deterministic text→latent route is **not excluded
by the data at all** — in principle a model that captured the shared signal perfectly would transcribe its own
corpus. Every "at its ceiling" reading — 91 %, 61 %, and the one-to-many story behind them — was an artefact of a
4-pair estimate that happened to land low.

**Where the token route actually is:** ρ **0.354** at n=40, which is **42 %** of this ceiling (not 61 %, not 91 %),
and the samples' 0.0045 is **0.5 %**. The honest description of the project's state changes: the wall is not "there
is almost nothing to learn from this text" but "**0.35 of an available 0.85 has been learned, and the reason for the
gap is not established**".

**Caveats, stated rather than buried:** this is a lower bound by construction (the true repeat-generation
reliability is *at least* 0.714, so the conclusion is robust in the direction that matters); punctuation variants
are not independent renders, which is exactly why the estimate is conservative; and it measures **this**
representation (the trained autoencoder's 24-dim latent at 93.75 Hz) on **this** voice set — so a different
autoencoder would have a different ceiling, which is itself an argument that the *representation* deserves
attention. The 0.845 ceiling says the task is possible; it does not say this model or this budget reaches it.

## 122. Environment notes

* CPU torch was installed from the PyTorch CPU index (no CUDA on this machine), in a dedicated
  Python 3.13 venv; the system Python 3.14 also has torch wheels available (2.14.1).
* `soundfile` is required for corpus I/O; `transformers`/`snac`/`kokoro` (teachers),
  `faster-whisper`, `funasr`, `onnxruntime` are optional and every code path that needs them
  reports unavailability instead of faking a metric.
* The repo's `runs/`, `data/`, `*.pt`, `*.wav` outputs are git-ignored; nothing large is committed.
