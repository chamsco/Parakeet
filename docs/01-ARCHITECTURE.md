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
   * the exported graph is the **decoder compute** — `from_latent` → causal ConvNeXt blocks → head →
     `(log_mag, phase)` — with dynamic batch **and time** axes;
   * the iSTFT overlap-add stays outside the graph: it is one FFT per frame (cheap) and
     complex/`torch.stft`/`torch.istft` export badly, whereas the convolutions are the actual cost;
   * int8 via `quantize_static` (QDQ, per-channel, int8 weights / uint8 activations) calibrated on
     real latents, and the **measured** output deviation is reported alongside the size/speed change
     — because ONNX Runtime's int8 *Conv* kernels need VNNI-era CPU support to actually beat fp32,
     so the speed-up must be measured rather than assumed;
   * `OnnxVocoder` runs the session and keeps the streaming overlap-add in torch, so int8 deployment
     does not lose the chunked/streaming property.
4. `Synthesizer.synthesize_stream` / `StreamingVocoder` — chunked, memory-bounded, numerically
   equal to offline decoding.
5. `phase_lock(wav, method="ramp" | "smooth")` — zero parameters, magnitude untouched, band 2–8 kHz.

ONNX export is checked for numerical parity with PyTorch (`< 1e-4` on the spectrogram) in
`tests/test_onnx.py`, which skips cleanly when `onnx`/`onnxruntime` are absent.

Known gaps (tracked in [ROADMAP.md](ROADMAP.md)): wiring the ONNX path into a release script, and a
streaming *sampler* for Small (chunked decoding exists; the flow still runs in one pass).
