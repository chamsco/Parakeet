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
python -m pytest -q                                         # 140 tests
python scripts/bench_rtf.py --config configs/parakeet_tiny.yaml
```

## 1. Test suite

```
140 passed
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

## 9. Smoke test output (measured)

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

## 10. Deliberate engineering checks worth calling out

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

## 11. Environment notes

* CPU torch was installed from the PyTorch CPU index (no CUDA on this machine), in a dedicated
  Python 3.13 venv; the system Python 3.14 also has torch wheels available (2.14.1).
* `soundfile` is required for corpus I/O; `transformers`/`snac`/`kokoro` (teachers),
  `faster-whisper`, `funasr`, `onnxruntime` are optional and every code path that needs them
  reports unavailability instead of faking a metric.
* The repo's `runs/`, `data/`, `*.pt`, `*.wav` outputs are git-ignored; nothing large is committed.
