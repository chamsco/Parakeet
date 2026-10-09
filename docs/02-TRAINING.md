# Training recipe

Reference recipe = **Paradee** (arXiv 2610.06817) for the two-half distillation discipline,
**SupertonicTTS** (arXiv 2503.23108) for the flow-matching training scheme, **PilotTTS**
(arXiv 2605.27258) for the conditioning losses. Numbers labelled *[paper]* are theirs;
numbers labelled *[ours]* are the shipped configs.

## Stage table

| # | Stage | Trains | Frozen | Data it reads |
|---|---|---|---|---|
| 1 | `autoencoder` | encoder + decoder | — | **real** speech mels |
| 2 | `distill-text` | Tiny text side (text encoder, duration, latent/F0/energy heads) | autoencoder | cached shards |
| 3 | `distill-decoder` | decoder only | encoder | cached shards (frozen teacher latents) |
| 4 | `flow` | text encoder, speaker conditioner, VF estimator, length predictor | autoencoder | cached shards |
| 5 | `reflow` | VF estimator | everything else | on-the-fly teacher endpoints |

Stages 2–3 consume fixed tensors, so the teacher is offline and the halves are never jointly
trained. That is the entire cost argument for the Tiny variant.

## Losses

```
Stage 1/3 (generator): L = λ_mel·L_mel + λ_spec·L_MRSTFT + λ_adv·L_adv + λ_fm·L_featmatch + λ_pl·L_phase
Stage 1/3 (discriminator): hinge loss, MPD(2,3,5,7,11) + MSD(3 scales)
Stage 2:  L = λ_dur·|log d̂ − log d| + λ_f0·|F̂0 − F0| + λ_en·|Ê − E| + λ_lat·MSE(latent_token)
Stage 4:  L = MSE(v̂, v) + λ_len·MSE(log len) + λ_style·(1 − cos(style_a, style_b))
Stage 5:  L = MSE(v̂_student, x₁_teacher − x₀)   with x₁_teacher from the EMA model at high NFE
```

### Target scaling is a correctness issue, not a detail

The stage-2 targets are all O(1) by construction: durations in log space, **F0 normalised log-Hz in
`[0, 1]`**, **energy normalised dBFS in `[0, 1]`**, per-token latent in normalised latent space.
This is not cosmetic.  With raw F0 bin indices (0–256) and raw dBFS the "distillation" loss was
**141**, of which the pitch term alone was ~110 and energy ~30 — a single-term loss wearing a
four-term objective as a disguise, with the duration and latent heads receiving almost no gradient.
`tests/test_learning.py` caught it; after normalising, the initial loss is ~3 and it falls by more
than half in 40 CPU steps.  (Paradee regresses `F0/100` for the same reason.)

### Stage freezing is stage-local

`run_stage` **resets** trainability at the start of every stage before applying that stage's freeze
pattern.  Previously a `distill-text` run (which freezes the whole autoencoder) left the decoder
frozen for a subsequent `distill-decoder` run in the same process, so the decoder stage silently
trained nothing while still reporting a finite loss.  This is now regression-tested.  Per stage:
`distill-text` freezes the autoencoder; `distill-decoder` freezes only the encoder (`stem`,
`encoder`, `to_latent`) and trains the decoder plus the prosody projection; `flow` and `reflow`
freeze the autoencoder, and `reflow` freezes everything except the vector field.

### The spectral-weight schedule is the single most important hyperparameter [paper]

| spectral weight | decoder UTMOS | full-student UTMOS |
|---|---|---|
| spectral only | 2.98 | 3.00 |
| 45 | 3.02 | — |
| 10 | 4.29 | 4.32 |
| 3 | 4.37 | 4.39 |
| 3 + phase-lock filter | — | **4.41** |

So `SpectralAnnealer` reproduces `45 @0 → 10 @3 000 → 3 @6 000` for the `distill-decoder` stage
and the shipped default is `spectral: 3.0` with `adversarial: 1.0`, `feature_match: 2.0`.
Wider/larger decoders did **not** fix the residual buzz; the loss balance did.

### Phase-linearity loss [ours]

`phase_linearity_loss` penalises `1 − cos(ψ[k+1] − 2ψ[k] + ψ[k−1])` in 2–8 kHz — i.e. it asks the
phase to be *coherent* (linear in frequency) without constraining its absolute value or slope.
Weight 0.05. It is a trainable complement to the inference-time phase-lock filter, not a
replacement for it.

## Hyperparameters

| | Stage 1 AE | Stage 2 text side | Stage 3 decoder | Stage 4 flow | Stage 5 reflow |
|---|---|---|---|---|---|
| Parakeet default | bs 16, lr 2e-4, AdamW β (0.8, 0.99), clip 1.0 | bs 16, 50 k steps | annealed, 25 k steps | bs 32, Ke 4, 200 k steps | bs 32, 20 k steps |
| Paradee [paper] | — | lr 5e-4, bs 32, **8 000 steps**, 500-step warmup + cosine, clip 1.0 | bs 16, lr 2e-4, clip 5, 1.6 s / 64-frame segments, **50 000 steps**, then +5 000 @mel-weight 10, +5 000 @3 | — | — |
| SupertonicTTS [paper] | 11 167 h corpus | — | — | 700 k iters, bs 64, Ke 4, 4× RTX 4090; CFG 3; NFE 32 | — |
| PilotTTS [paper] | — | — | — | 200 k steps on 60 k h subset for the ablation | — |

Everything uses AdamW, cosine decay with warmup (`cosine_warmup_scheduler`), grad clipping, and an
EMA of the student (`EMA_Model`, decay 0.999) which doubles as the Reflow teacher.

**Why Ke = 4:** SupertonicTTS shows higher expansion factors converge faster and deeper than
equivalent plain batch growth, and specifically reduce word-skip/repetition errors, at the cost of
only the cheap latent batch. Our test suite asserts the effective batch really grows by `Ke`.

**Why Reflow instead of fewer Euler steps:** at NFE 4 with a plain flow model, SupertonicTTS's WER
degrades to 11.43 (vs 2.64 at NFE 32) — the ODE trajectories are curved, so step count cannot be
cut for free. Reflow/2-rectified-flow training straightens them; consistency distillation is the
alternative if Reflow alone is not enough.

**Classifier-free guidance:** 10 % conditioning dropout during stages 4–5, guidance scale 3.0 at
inference. The unconditional branch uses a learned `null_memory` token, so no second forward pass
over the text encoder is needed.

## Data volumes

| Purpose | Volume | Source |
|---|---|---|
| AE + speaker encoder | 100–1 000 h real speech | Emilia / LibriLight / MLS (check each licence) |
| Tiny text side + decoder | **20–25 h** (Paradee used 12 000 sentences → 23.9 h) | teacher corpus |
| Small flow | 50–200 h teacher corpus; SupertonicTTS proved 945 h suffices | teacher corpus + real audio |
| Capability SFT (tags/emotion) | 1–10 h parallel, tag-annotated | teacher-generated, PilotTTS-style 50/50 mixed-prompt SFT |

## Compute budget (estimates, not measurements)

Assumptions: 24 kHz, 64-frame (≈0.68 s) segments, the parameter counts in
[01-ARCHITECTURE.md](01-ARCHITECTURE.md), a single 24 GB GPU, and roughly the per-step costs we
measured on CPU scaled by a typical 100–300× GPU speed-up.

| Work item | Estimate | Notes |
|---|---|---|
| AE training (Tiny, 300 h real audio, 100 k steps bs 16) | **2–4 GPU-days** | the most expensive single item; everything downstream depends on it |
| Tiny text side (8 k steps, Paradee's exact schedule) | **< 2 GPU-hours** | pure regression on cached tensors |
| Tiny decoder (60 k steps bs 16) | **≈ 1 GPU-day** | adversarial training is the cost driver |
| Small flow (200 k steps bs 32, Ke 4) | **4–8 GPU-days** | Ke 4 quadruples the estimator batch but not the encoder work |
| Reflow (20–50 k steps) | **≈ 1 GPU-day** | teacher endpoints cost `nfe` forward passes |
| End-to-end Tiny reproduction | **≈ 2 GPU-days** | matches Paradee's "fits one laptop / ≤2 concurrent trainers on 24 GB" claim |
| End-to-end Small reproduction | **≈ 1–2 GPU-weeks** | SupertonicTTS used 4× 4090 for 700 k iterations |

These are first-pass planning numbers. The first thing to do once data exists is measure
steps/second on the target GPU and replace this table (see [ROADMAP.md](ROADMAP.md) P2).

## Failure modes and diagnostics

| Symptom | Likely cause | Check |
|---|---|---|
| Speech sounds like a "robotic second voice" | spectral weight too high | run the anneal; log `spec_weight`; compare with `λ_spec = 3` |
| Residual buzz on voiced sounds | high-band phase | `phase_coherence` (2–8 kHz) before/after the filter; `phase_lock` loss curve |
| Word skipping / repetition | alignment instability | `Ke` expansion, CFG dropout, lower lr; inspect length-predictor error |
| Muffled output | decoder under-trained vs critic | feature-matching weight, discriminator lr, segment length |
| Nothing converges past a point | bad latent scale | `LatentNormalizer` statistics; re-fit on the AE's own outputs |
| One loss term dwarfs the others | unnormalised targets (raw dB, F0 in Hz, bin indices) | before/after probe as in `tests/test_learning.py`; normalise to O(1) |
| A stage reports a loss but changes nothing | freezing leaked in from the previous stage | `run_stage` resets trainability per stage — regression-tested |
| Student ignores the speaker | Q-Former collapse | cross-sample paired batches, style cosine loss, freeze CAM++ |
| Loss fine, audio noise | teacher signal cache mismatch (F0/duration off-by-one) | compare `log_mel` reconstructed from the cache against the source wav |
