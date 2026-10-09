# Parakeet design: how to mix two very different teachers into one tiny model

This is the working design document. It explains *what* we are building, *why* each choice was
made, and *which* parts are proven versus estimated. Source-grounded research notes for every
external claim live in [`docs/research/`](research/).

---

## 1. The two teachers are not the same kind of thing

| | **Orpheus** (Canopy Labs) | **MiniMax speech-2.8-turbo** |
|---|---|---|
| Access | local open weights (`canopylabs/orpheus-3b-0.1-ft`), gated | HTTPS/WebSocket API only |
| Backbone | Llama-3.2-3B-Instruct geometry (3072 hidden, 28 layers, vocab 156 939) | undisclosed; the closest official description is `speech-02` (BPE text → 25 Hz VQ-on-mel tokenizer → AR transformer → latent flow matching + Flow-VAE) |
| Acoustic representation | SNAC 24 kHz, 7 codebooks, 1 super-frame @12 Hz (2048 samples, 83.3 ms), ~84 tokens/s | none exposed — **audio out only** |
| What we can distil from | audio **and** codes/logits; expressive inline tags `<laugh> <sigh> …` | audio only |
| Sample rate | 24 kHz | configurable (up to 44.1 kHz) |
| Expressivity | 8 English voices, 8 paralinguistic tags | 40 languages, 19 interjection tags (2.8-only), emotion control |
| Licence | Apache-2.0 code; weights gated Apache-2.0 tag over a Llama-3.2 base | proprietary; **terms bar using MiniMax Voice to develop foundation models** |

Two consequences follow immediately, and they shape the whole design:

1. **There is no shared token space.** Orpheus emits hierarchical RVQ codes at 12/23/47 Hz;
   MiniMax emits nothing but a waveform. Any design that depends on teacher tokens can only use
   one of the two teachers.
2. **There is no legal symmetry.** Orpheus is permissive (with a Llama-3.2 chain), MiniMax is
   restricted. A design that *requires* MiniMax audio imports that risk into every artefact.

## 2. The resolution: mix at the audio level, not the architecture level

Parakeet does **not** try to merge the teachers. It introduces one shared target representation —
a 24-dimensional continuous latent at 93.75 Hz produced by a Parakeet autoencoder — and
re-encodes every teacher's waveform into it:

```
Orpheus audio ─┐
MiniMax  audio ─┼─► frozen Parakeet autoencoder ─► latent (24 × 93.75 Hz) ─► student targets
Kokoro   audio ─┘                                       │
                                                        └─ fold Kc=6 ─► 144-dim @15.6 Hz for flow matching
```

This is the single most important design decision, and it buys four things at once:

* **Teacher-agnostic supervision.** The student never sees a teacher-specific representation, so
  teachers can be swapped, mixed, or weighted per sample without touching the model.
* **Frame-rate mismatch disappears.** SNAC's 12 Hz super-frames (7 codebooks), Kokoro's 40 Hz
  frames (F0 at 80 Hz) and MiniMax's arbitrary rate are all resampled by the encoder. (Kokoro's
  own 40 Hz layout is what Paradee used; we do not inherit it.)
* **One sampler, one decoder, one quantiser.** Nothing downstream needs to know which teacher
  produced a sample, which keeps the tiny deployment story simple.
* **Graceful degradation under legal pressure.** Removing MiniMax from the mixture changes the
  data, not the code. The default mixture is Orpheus 60 / Kokoro 40 precisely for that reason.

### How the two teachers contribute *different* things

Averaging two teachers would give a blurry average. Instead the mixture is *capability-split*,
following PilotTTS's use of teacher-generated parallel data for scarce capabilities:

| Contribution | Source | Mechanism |
|---|---|---|
| Naturalness / rendering quality | the highest-quality permitted teacher per utterance | acoustic loss + adversarial critic |
| Fidelity of prosody, stress, phrasing | whichever teacher generated that utterance | flow-matching target in latent space |
| Expressive control (`<laugh>`, `<sigh>`, …) | **Orpheus tags**, plus MiniMax's 19 interjection tags as a **taxonomy** for our own labelling | shared tag vocabulary in the tokeniser; tag-conditioned samples upweighted |
| Pronunciation robustness | all teachers | Whisper-WER agreement filter; disagreement is logged as difficulty, not trained through |
| Identity | frozen CAM++ speaker encoder (PilotTTS), **not** the teachers | cross-sample paired training |

Teacher mixing is therefore a **data mixture with per-sample weighting**, not an architectural
fusion:

* weights are capped so no teacher exceeds ~60 % of a shard (shard-local balance, implemented by
  interleaving with a golden-ratio sequence in `synthesize_corpus`);
* each sample is additionally weighted by its quality score (DNSMOS/UTMOS/WER from the pipeline);
* teachers that disagree (large WER spread on the same text) are *flagged*; the pipeline keeps
  both variants tagged, and the training weight falls.

**Where the weight actually goes** (and how it is verified): `build_latent_cache(teacher_weights=…)`
turns each sample's teacher into a raw weight (`share × quality`, floored at 0.05), stores it with
the sample, `collate` carries it into the batch, and both `TextSideDistillLoss` and
`ParakeetFlow.flow_loss` reduce **per sample** and rescale weights to mean 1 — so changing the
mixture re-weights the gradient without changing the effective learning rate.
`scripts/mixture_demo.py` demonstrates the effect: across a mixture sweep from 100 % low-pitched to
100 % high-pitched teacher, the student's predicted F0 moves monotonically over a 112.6 Hz span
(87 → 200 Hz) while the fit to the low-pitched teacher degrades in the same order.
*This plumbing was absent until round 6 — the mixer existed and was tested but nothing read it, so
the mixture was decoration. `tests/test_mixture.py` now pins every link.*

## 3. Why this architecture (and not something else)

| Decision | Reason |
|---|---|
| **Continuous 24-dim latent, not a codec** | A codec would re-import a token space, force code-prediction losses, and lock us to one teacher. SupertonicTTS shows a 24-dim latent is enough for 44M-parameter quality. |
| **ConvNeXt blocks, not a transformer, for the small model** | SupertonicTTS reaches competitive quality with ConvNeXt at 44M; transformers at this size are compute-hungry and awkward to stream. |
| **Causal dilated decoder + iSTFT head** | Streaming, and it removes the vocoder entirely (Vocos-style, as in SupertonicTTS). Our implementation is chunk-exact vs offline (see verification). |
| **Character-level text, no G2P** | Removes a morpheme-heavy dependency and a class of language-specific bugs; alignment is learned by cross-attention. (Phoneme mode remains available for single-voice Tiny.) |
| **Temporal compression `Kc=6`** | 6× fewer tokens for the same information at 15.6 Hz; the compressed sequence is what lets a small ConvNeXt compete with a transformer. |
| **Two halves trained separately (Tiny)** | Paradee: no alignment learning, no joint training, teacher can be discarded at training time — the cheapest possible route to a good 8–12M voice. Direct feature supervision beats end-to-end distillation (4.39 vs 3.78 UTMOS). |
| **Flow matching + Reflow for Small** | SupertonicTTS needs NFE 32; naive step cuts collapse (NFE 4 → WER 11.43 vs 2.64). Straightening the flow (2-rectified flow / consistency) is what makes NFE 2–4 usable. |
| **Q-Former + CAM++ factorisation** | This is how one student can be *both* expressive (Orpheus-like) and cloneable (MiniMax/PilotTTS-like) without style bleeding into identity. |
| **Weight-only int8 + phase-lock filter** | Paradee: int8 costs ~0 UTMOS (4.41 → 4.41) and lands 8.45 MB; 4-bit drops to 3.98. The filter is parameter-free and fixes the residual buzz. |

## 4. The recipe: five stages, two halves, one connection

```
Stage 0  DATA        teacher corpus (audio) ─► latent cache (frozen AE) ─► shards of cached signals
Stage 1  autoencoder  mel ─► latent ─► waveform        (mel + multi-res STFT + MPD/MSD, λ_spec = 3)
Stage 2  distill-text Tiny text side ─► durations, F0, energy, per-token latent feature  (L1/MSE)
Stage 3  distill-decoder Tiny decoder from *frozen teacher latents*; λ_spec annealed 45 → 10 → 3
Stage 4  flow        Small text+speaker ─► velocity field (CFM, Ke=4, CFG dropout 10 %)
Stage 5  reflow      straighten the flow with an EMA teacher's high-NFE endpoint ─► NFE 2–4
                      then: join halves (no joint training) ─► int8 ─► phase-lock ─► ship
```

Critical property: stages 2 and 3 consume **cached tensors**, so the teacher is not needed during
training and the two halves never have to be co-trained. Stage 1 is trained on *real* speech
(it is the representation everyone shares), which also blunts the "trained on synthetic audio"
failure mode Canopy warns about.

## 5. Budgets

| | Tiny | Small | SupertonicTTS (reference) | Paradee (reference) |
|---|---|---|---|---|
| parameters | **9.62 M** | **45.04 M** | 44 M | 8.07 M |
| composition | AE 5.03 / text 3.85 / duration 0.66 | AE 14.11 / text 5.43 / VF 19.15 / speaker 6.08 / length 0.20 | 0.5 DP + 18.5 T2F + 25 F2S | 4.23 text + 3.85 decoder |
| latent | 24-dim @93.75 Hz | same, `Kc=6` → 144 @15.6 Hz | 24-dim, Kc=6 → 144 @14 Hz | 40 Hz frames (Kokoro grid) |
| inference NFE | none | 2 (distilled) | 32 | none |
| int8 size | 12.0 MB | 56.3 MB | — | 8.45 MB |
| measured RTF, 1 CPU thread | **0.044 (22.6×)** | not yet meaningful | 0.313 @5 steps (their 99M shipping model) | 25.0× |

Where "light" actually comes from, in order of impact: (1) no vocoder at all — the decoder *is*
the vocoder; (2) `Kc=6` temporal compression on the generative half; (3) NFE 2 instead of 32;
(4) weight-only int8; (5) the Tiny variant removing the sampler entirely.

## 6. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| MiniMax terms bar training foundation models on its Voice output | **blocking for that teacher** | refused by default (`TeacherLicenseError`); tag taxonomy used instead; written permission required to enable — see [LEGAL.md](LEGAL.md) |
| Synthetic-only data → poor codec/allocation behaviour (Canopy's own warning) | high | AE + speaker encoder trained on *real* audio; teacher audio reserved for capability/render distillation; consider soft targets |
| Residual "buzz" on voiced speech 2–8 kHz | medium | phase-lock filter (zero parameters, two interchangeable lock references) + phase-linearity training loss |
| Teacher misalignment poisons duration targets | medium | durations come from the teacher (Kokoro) or a forced aligner; the energy-envelope fallback is explicitly marked as a fallback in `extract_signals` |
| NFE collapse when cutting steps | high | Reflow/consistency distillation stage, not naive step reduction |
| Q-Former entropy collapse (style tokens ignoring the reference) | medium | cross-sample paired training (different utterance, same speaker) + cosine consistency loss |
| Distillation licence ambiguity for Orpheus (Llama-3.2 chain) | medium | legal review before release; attribution + acceptable-use obligations travel with the weights |
| Overfitting a 12 k-sentence corpus (Paradee's budget) | medium | our prompts are our own; target ≥50 h for Small and ≥20 h for Tiny before judging quality |

## 7. What to do first (ordered)

1. **Freeze the representation.** Train the autoencoder on real speech until reconstructed mel
   is stable and the latent is well-conditioned (`LatentNormalizer` statistics sane). Everything
   downstream is capped by this.
2. **Build a small, honest teacher corpus.** 2–5 k prompts, Orpheus + Kokoro, with real-audio
   validation. Do not start with 200 k hours; SupertonicTTS proved 945 h is enough.
3. **Tiny first.** It is the cheapest path to a real, measurable model (Paradee needed one
   trained text side and one decoder), and it validates the whole cache/quantise/phase-lock path.
4. **Then Small + Reflow**, and only then chase zero-shot cloning quality.
5. Legal review in parallel with step 1 if MiniMax is to be used at all.
