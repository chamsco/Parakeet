# Verification: what is actually proven by this repository

Everything below was produced by running the code in this repo on a Windows machine with an
AMD Ryzen 7 7800X3D (8 cores), 31 GB RAM, **no GPU**, Python 3.13.14, torch 2.9.1+cpu.

Reproduce with:

```bash
python scripts/smoke_test.py --steps 2 --out runs/smoke     # trains every stage, synthesises
python scripts/learn_demo.py                                # proves the stages learn (~10 min CPU)
python scripts/reflow_demo.py                               # validates NFE-2 sampling (~5 min CPU)
python scripts/export_onnx.py                               # int8 ONNX + runtime benchmark
python scripts/profile_pipeline.py                          # where does the time actually go
python -m pytest -q                                         # 109 tests
python scripts/bench_rtf.py --config configs/parakeet_tiny.yaml
```

## 1. Test suite

```
109 passed
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
| `test_onnx.py` | ONNX decoder matches PyTorch to **<1e-4**; dynamic time axis across 7/23/41 frames; int8 file is smaller and runs on CPU; the fp32↔int8 comparison reports size reduction, both latencies and the output deviation (skips if `onnx`/`onnxruntime` absent) |

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

`scripts/export_onnx.py` exports the **decoder compute** (causal ConvNeXt blocks + head) with
dynamic batch *and time* axes, quantises it to int8 with ONNX Runtime's static QDQ path
(per-channel, int8 weights / uint8 activations, calibrated on real latents), and benchmarks all
three runtimes on one CPU thread. The iSTFT stays in torch, so streaming is not lost.

```
parakeet-tiny: exporting decoder compute (latent_dim=24)
decoder compute benchmark (1 thread, 81 latent frames = 1.013s audio)
  PyTorch fp32          10.72 ms     94.52x real time
  ONNX fp32              8.72 ms     17.97 MB   1.23x vs PyTorch
  ONNX int8 (QDQ)        4.24 ms      4.68 MB   2.53x vs PyTorch
  int8 size reduction 3.84x | int8 speedup vs ONNX fp32 2.06x
  int8 deviation: |dlog_mag|max 0.0154, |dphase|max 0.0149 rad
```

* ONNX int8 is **2.53× faster than PyTorch** and **3.84× smaller** (18.0 → 4.7 MB) for the decoder
  compute, with a max log-magnitude deviation of 0.015 and max phase deviation of 0.015 rad — i.e.
  int8 changes the decoder's output by ~1.5 % in log-magnitude. (Paradee's equivalent claim is
  "int8 costs ~0 UTMOS"; ours is a measured output deviation, which is weaker evidence but honest —
  UTMOS needs a trained model.)
* This CPU is a Zen 4 (AVX-512 VNNI), which is why int8 *Conv* beats fp32 here. The benchmark
  measures rather than assumes precisely because that is hardware-dependent.
* Numerical parity of the fp32 graph with PyTorch is asserted in `tests/test_onnx.py` (< 1e-4 on
  the spectrogram), along with dynamic-time-axis behaviour and int8 size/run checks. Those tests
  skip cleanly when `onnx`/`onnxruntime` are absent.

## 4. Pipeline profile (measured)

The ONNX result above is only as useful as knowing where the time goes, so
`scripts/profile_pipeline.py` profiles a full Tiny synthesis, one CPU thread, warm caches, mean of
three sentences (0.565 s of audio, 4 NFE, 9.62 M params):

| component | ms | % of full | standalone × real time |
|---|---|---|---|
| text side (encoder + duration/F0/energy/latent heads) | 6.95 | **34.4 %** | 81× |
| latent construction (align + prosody projection + de-normalise) | 0.31 | 1.5 % | 1810× |
| decoder + iSTFT | 7.99 | **39.6 %** | 71× |
| phase-lock filter | 2.27 | 11.2 % | 249× |
| python/dispatch overhead | 2.66 | 13.2 % | — |
| **full synthesize** | **20.18** | 100 % | **28.0×** |

**Finding: the vocoder is not the bottleneck.** It is ~40 % of the budget, with the text side taking
an almost equal share, so int8-ONNX-ing the decoder (2.06× faster) buys roughly 20 % of the total
path, not a step change. The next real speed win is exporting the *text* side too (its attention is
MatMul-shaped, which int8 handles well) and trimming the 13 % Python/dispatch overhead — not more
vocoder work. That is now the top item in the roadmap's P4.

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

## 6. Smoke test output (measured)

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

## 7. Deliberate engineering checks worth calling out

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

## 8. Environment notes

* CPU torch was installed from the PyTorch CPU index (no CUDA on this machine), in a dedicated
  Python 3.13 venv; the system Python 3.14 also has torch wheels available (2.14.1).
* `soundfile` is required for corpus I/O; `transformers`/`snac`/`kokoro` (teachers),
  `faster-whisper`, `funasr`, `onnxruntime` are optional and every code path that needs them
  reports unavailability instead of faking a metric.
* The repo's `runs/`, `data/`, `*.pt`, `*.wav` outputs are git-ignored; nothing large is committed.
