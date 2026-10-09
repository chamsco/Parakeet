# Architecture specification

All numbers below are **measured** from the shipped configs (`python scripts/smoke_test.py` prints
the same table), not estimated from a paper.

## 1. Signal chain

| Stage | Shape | Rate | Notes |
|---|---|---|---|
| waveform | `(B, N)` | 24 000 Hz | 24 kHz matches both principal teachers (Orpheus/SNAC and Kokoro) — upsampling to 44.1 kHz would add no information |
| log-mel | `(B, 80, T)` | 93.75 Hz | `n_fft=1024, hop=256, win=1024`, Slaney-normalised triangular filterbank, `log(clamp(·, 1e-5))` |
| latent | `(B, 24, T)` | 93.75 Hz | continuous, produced by the autoencoder encoder |
| folded latent | `(B, 144, T/6)` | 15.625 Hz | `Kc = 6`; the flow sampler operates here |
| waveform out | `(B, (T−2)·hop)` | 24 000 Hz | decoder → iSTFT head (`center=True` semantics) |

The latent is ~2 250 floats/s (~8.4 kB/s fp32, ~2.1 kB/s int8).

## 2. Speech autoencoder (`SpeechAutoencoder`)

**Encoder** (non-causal — it sees the whole utterance, which is free because it runs once):

```
stem: Conv1d(80 → d0, k=7)
stage i: [Conv1d(k=3) transition] + n_i × ConvNeXtBlock(d_i, k=7)
to_latent: Conv1d(d_last → 24, k=1)
```

**Decoder** (causal, streaming-capable). Every depthwise convolution is a `CausalConv1d`
(left zero-padding only), which is what makes chunked decoding exactly equal to offline decoding:

```
from_latent: Conv1d(24 → d_dec, k=1)
blocks: n × ConvNeXtBlock(d_dec, k=7, expansion=4, dilation from a cyclic list, causal)
head: CausalConv1d(d_dec → 2·(n_fft/2+1), k=7)
       → log-magnitude (exp, clamped at 8) and phase → torch.polar → OLA iSTFT
```

`latent_receptive_field` reports the latent-frame history the decoder needs (48 frames for the
Tiny config); the streaming vocoder uses per-block history caches rather than a re-run window.

Measured sizes:

| Config | encoder dims / blocks | decoder | params |
|---|---|---|---|
| Tiny | `[64, 96, 128]` / `[1, 2, 2]` | 256 × 5, dilations `[1,2,4,8,16]` | 5.03 M |
| Small | `[128, 192, 256]` / `[2, 3, 3]` | 384 × 7, dilations `[1,2,4,8,16]` | 14.11 M |
| Small-44k | `[128, 192, 256]` / `[2, 3, 3]` | 384 × 8 | 13–14 M (228 mel @44.1 kHz) |

`LatentNormalizer` keeps a running mean/variance of the latent so the flow target is
unit-scale; the flow works in normalised latent space and the decoder de-normalises.

## 3. Text encoder (`TextEncoder`)

Character-level by default. Embedding → learned positional → `n_layers` × (multi-head
self-attention + FFN) → one local ConvNeXt block (kernel 5, catches grapheme clusters) →
LayerNorm → zeroed at padding.

* vocab 78 symbols: 4 specials (`<pad> <unk> <s> </s>`), 24 shared tags, 50 base characters.
  Phoneme mode adds 23 IPA symbols (101 total). Both fit the default `vocab_size=128`.
* Tiny: dim 256 × 4 layers. Small: dim 256 × 6 layers.

There is deliberately **no G2P module and no external aligner**; alignment is implicit through
cross-attention into the text memory, as in SupertonicTTS.

## 4. Duration (`DurationPredictor`, `UtteranceLengthPredictor`)

* `DurationPredictor`: 2 × (Conv1d → LayerNorm → GELU) → per-token log-duration. Used by Tiny,
  where the teacher's own durations are the regression targets (Paradee's trick: no alignment
  learning at all).
* `UtteranceLengthPredictor`: pooled text ⊕ speaker embedding → one scalar log-length, as in
  SupertonicTTS, keeping length decoupled from content generation for Small.

Tiny 0.66 M, Small 0.20 M (Small's is tiny because it is a single scalar head).

**Voice conditioning (Tiny).** A voice is *not* just timbre: it changes pitch range, energy,
timing and spectrum. `ParakeetTiny.voice_embed` therefore adds a per-voice vector to the **text
memory before every head**, so duration, F0, energy and the latent feature are all voice-conditioned.
The first implementation added it only to the latent feature, which left the prosody heads
voice-blind and made multi-voice training unable to separate voices at all. A single voice is the
learned constant `voice_embed.weight[0]` (Paradee replaces the style input with a learned constant).
The per-sample voice index is stored in the latent cache next to the teacher weight.

**Pitch targets.** `extract_signals` uses **YIN** (`estimate_f0_yin`: difference function, cumulative
mean normalisation, absolute threshold on the first *local* minimum, parabolic refinement on the
difference function) rather than autocorrelation, which is formant-biased — on a fixture utterance it
reported 168 Hz for an 81 Hz voice and marked only 52 % of frames voiced (YIN: 74 Hz, 100 %). Per-token
F0 targets average over **voiced frames only**, so a half-voiced token keeps its true pitch.

## 5. Speaker conditioning (`SpeakerConditioner`, PilotTTS §3.2)

```
reference mel ─► EcapaTdnnLite / CAM++ ─► 192-dim identity ─► id token
reference mel ─► MelMemoryEncoder ─► Q-Former (8 learnable queries × 2 layers) ─► 8 style tokens
conditioning memory = [id token ; style tokens]  (9 tokens, 256-dim)
```

* Identity and style are separate pathways; **cross-sample paired training** (style from a
  *different* utterance of the same speaker) plus `cosine_style_loss` is what decouples them.
* `voice_mode="constant"` replaces the style input with a learned constant (Tiny / single voice),
  exactly as Paradee does for a single-voice student.
* The shipped `EcapaTdnnLite` is a randomly-initialised stand-in for tests; production must load
  frozen CAM++ (`SpeakerConfig.checkpoint`, `freeze=True`).
* Size: 6.08 M (both variants).

## 6. Flow-matching estimator (`ConvNeXtVFEstimator`)

```
x_t (144, T/6) ─► Conv1d(144 → dim, k=1) ─► + TimeEmbedding(t)
                ─► depth × FlowBlock
FlowBlock: causal-free ConvNeXtBlock(dim, k=7) ─► cross-attention into
           [text memory ‖ speaker/style tokens] (projected to dim) ─► FFN
                ─► LayerNorm ─► Conv1d(dim → 144, k=1)  ⇒ velocity
```

* Rectified-flow convention: `x_t = (1−t)·x₀ + t·x₁`, target `v = x₁ − x₀`, `t=0` noise, `t=1` data.
* `null_memory` is a learned unconditional token; conditioning is dropped for 10 % of training
  items so classifier-free guidance (scale 3, as SupertonicTTS) is available at inference.
* **Streaming sampling.** Because the temporal mixing is *finite-support* convolutions rather than
  self-attention, the ODE can be integrated block by block: a block needs only
  `context = 3 × depth` frames each side (18 frames ≈ 1.15 s of audio for the Small config) plus a
  lookahead band that co-evolves and is discarded when the next block regenerates it. Conditions are
  global and computed once, so they impose no temporal dependency. `iter_blockwise_sample` /
  `blockwise_sample` implement this; `ParakeetFlow.synthesize_stream` decodes each block as it
  arrives. Measured: TTFA 7.81× better at 21.9 s of audio, at 1.07–2.81× total compute depending on
  block size (see [05-VERIFICATION.md](05-VERIFICATION.md) §6).
* `Ke = 4` **context-sharing batch expansion**: conditioning tensors are repeated `Ke` times so
  the estimator sees 4 independent noise/time draws per unique text+speaker pair
  (`expand_for_context_sharing`, verified by a test that asserts an effective batch of 3× for Ke=3).
  Cost: no extra text/speaker encoding; only the cheap latent batch grows.
* Small: dim 384, depth 6, 6 heads → 19.15 M (SupertonicTTS's text-to-latent module is 18.5 M —
  same order by construction).

## 7. Assembly and variants

| | `ParakeetTiny` | `ParakeetFlow` (Small) |
|---|---|---|
| text side | `text_side()` → log-duration, per-token latent, F0, energy | `conditions()` → memory |
| latent production | repeat per-token latent over durations + prosody projection, then de-normalise | few-step flow sample, unfold `Kc`, de-normalise |
| decode | shared causal decoder | shared causal decoder |
| post | phase-lock → int8 | phase-lock → int8 |
| params | 9.616 M | 45.038 M |

## 8. Deployment path

1. `quantize_weights_(bits=8, per_channel=True)` — weight-only, per-output-channel, fp16 scales.
   Refuses sub-8-bit without explicit per-channel acknowledgement (Paradee: 4-bit → UTMOS 3.98).
2. `save_int8_state_dict` — int8 weights + fp16 scales + fp16 norms/embeddings in one file.
3. **ONNX export + int8 QDQ** (`parakeet/inference/onnx_export.py`, `scripts/export_onnx.py`):
   * two exported graphs, because two halves each cost ~a third of the budget: the **text side**
     (`ids → log_duration, latent_token, f0, energy`) and the **decoder compute**
     (`from_latent → causal ConvNeXt blocks → head → (log_mag, phase)`), both with dynamic batch
     and time/token axes;
   * the text side uses the `torch.export`-based exporter (`dynamo=True`, needs `onnxscript`):
     the legacy TorchScript exporter bakes the dummy token count into `nn.MultiheadAttention`'s
     reshapes and yields a graph that *only* accepts that exact length;
   * the iSTFT overlap-add and the phase-lock filter stay in torch: one FFT per frame each, complex
     ops export badly, and together they are ~11 % of the budget;
   * int8 via `quantize_static` (QDQ, per-channel, int8 weights / uint8 activations) calibrated on
     real latents/token sequences, and the **measured** output deviation and PyTorch equivalence are
     reported alongside the size/speed change — ONNX Runtime's int8 *Conv* kernels need VNNI-era CPU
     support to actually beat fp32, so the speed-up must be measured rather than assumed;
   * `OnnxTinyPipeline` composes them; measured on one CPU thread it is **3.9× faster than PyTorch**
     (106× vs 27× real time) at **9.9 MB** total int8, with waveform cosine ≥ 0.992 vs the PyTorch
     pipeline on the same weights.
4. `Synthesizer.synthesize_stream` / `StreamingVocoder` — chunked, memory-bounded, numerically
   equal to offline decoding.
5. `phase_lock(wav, method="ramp" | "smooth")` — zero parameters, magnitude untouched, band 2–8 kHz.

Known gaps (tracked in [ROADMAP.md](ROADMAP.md)): a streaming *sampler* for Small (chunked decoding
exists; the flow still runs in one pass), and the ~13 % python/dispatch overhead measured by
`scripts/profile_pipeline.py`.
