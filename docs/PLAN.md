# What a viable run costs, and what to watch

Written after 65 rounds of measurement on this machine. Every number here is measured, not estimated, and
each one is traceable to `docs/05-VERIFICATION.md`.

## The constraint

**This machine trains on CPU.** Both GPU routes were investigated and closed: WSL2+ROCm is outside AMD's
support matrix for gfx1030 (Radeon RX 6950 XT, RDNA2), and DirectML cannot run the audio step at all. The
operator's decision was to keep training on CPU.

**Wall time is not compute time.** A round of wall clock has been measured passing with only ~50 minutes of
CPU accumulated, because the machine sleeps. Plan in **CPU-minutes**, expect wall time to be 2–4× the CPU
time, and trust the step counter in a log rather than a clock.

## Measured step costs

| stage | setting | cost | 50 000 steps |
|---|---|---|---|
| flow (marginal route) | batch 8, whole utterances | ~2.0 s/step | 28 CPU-hours |
| flow | batch 16, 160-frame crops, **measured** | **1.76 s/step contended, ~0.9 alone** | **12–25 CPU-hours** |
| token route (`distill-text`) | batch 16, 512-dim / 6-layer text side | ~0.8–1.2 s/step | 11–17 CPU-hours |
| audio autoencoder | batch 8 | ~0.5 s/step | 7 CPU-hours |

**Corrected in round 88.** This table claimed ~0.5 s/step for the cropped flow, which was an estimate from an
earlier measurement. Measured directly on the 1 932-item corpus by timing 100 and 300 steps and subtracting, so
the fixed setup cost cancels: **1.76 s/step while the corpus generator was running**, and about **0.9 s/step**
on a quiet machine — so 50 000 steps is 12–25 CPU-hours, not 7. The earlier figure was optimistic by roughly
two, and anything budgeted from it was too.

The practical consequence: a run long enough to be informative is **thousands of steps, not tens of thousands**.
The conditioning gate's trend is visible at 1 000-step resolution (round 65's curve used 1 000 per point), so a
12 000-step run with a reading every 2 000 gives six points for 2.5–5 CPU-hours — which is the right size for a
question that a wrong answer costs days to discover.

## What to watch, in order

1. **the flow term's text dependence** (`scripts/text_dependence.py`) — the share of the *velocity field's*
   objective that comes from the text. Round 64 measured it at **2–3 %** and showed that a total which looks
   like 37 % can be almost entirely a supervised auxiliary. This is the cheap early indicator: one forward
   pass per checkpoint, no sampling.
2. **sampled-latent ρ** (`scripts/rho_curve.py`) — the calibrating measurement: the recogniser reads a decoded
   latent perfectly at **ρ ≥ 0.75** and fails at 0.60. Nothing else in this project predicts audible quality
   as well, and unlike the mel proxy it cannot be satisfied by noise (ρ ≈ 0 with a log-cosine of 0.90 is on
   record).
3. **speech-likeness** — a sanity check only. It was measured to reject *perfectly intelligible* audio (the
   autoencoder's own round trip), so a `NO` there is not evidence of anything on its own.
4. **WER with its teacher control** — always reported together; the control reads 0.000, the current models
   1.000.

## The two candidate runs

**Token route** (deterministic, quarter the size, one forward pass, no sampler). Measured ceiling: it
memorises 8 utterances to ρ 0.998 in 300 steps, so capacity is not the limit; on the full corpus it plateaus
at **ρ ≈ 0.37** with validation ≈ training, i.e. it fits as well as it can and the remaining error is the
*one-to-many* mapping. More steps will not move it. **Not worth more compute.**

**Flow route** (sampling, the structurally correct tool for a one-to-many target). Its blocker is measured:
the velocity field's own objective is only a few percent text-dependent, while the 2-utterance regime reached
ρ 0.25 — so the field *can* learn conditioning when the marginal distribution has little to offer it.

**And the dependence grows with scale** (`scripts/conditioning_vs_scale.py`, same architecture, same 1 000
steps, only the corpus size changing):

| training utterances | flow-term text dependence |
|---|---|
| 4 | 1.9 % |
| 16 | 2.3 % |
| 64 | 4.1 % |
| 200 | **7.4 %** |

Monotone over a 50× range, and the **opposite** of the prediction this measurement was built to test (that a
bigger corpus would make the marginal cheaper and starve the conditioning further). So more data does feed the
conditioning — slowly. Treat the levels as noisy (the same 200-utterance size measured 2–3 % at batch 16
against 7.4 % at batch 8, so ±3 points), and the *trend* over the full range as the reliable part.

**This is where compute could pay**, and the number to move is the flow term's text dependence. The honest
expectation: it moves slowly, so budget a long run rather than a short one, and read the flow term every
1 000 steps before spending more.

## What would change the picture

* **more teacher audio.** This is the measured gap, not a hunch: SupertonicTTS trained its generative module
  on **945 hours** and this project has ~1.9 h (plus ~3.2 h of expansions, usable now that the per-voice scale
  is normalised). See [PAPER_COMPARISON.md](PAPER_COMPARISON.md) — the architectural axes line up one by one
  (144-dim folded latents at ~15 Hz, ConvNeXt, character-level text, implicit alignment, `Ke = 4`, NFE 32), so
  the difference is data and compute rather than a missing trick;
* **a GPU** (any CUDA card, even rented for a few hours): the code is device-agnostic, plain fp32, with no
  CUDA-specific assumptions and a verified `--resume`, so it would move as-is;
* ~~a lower-rate learned acoustic bottleneck~~ — **retracted.** The velocity field already models 144-dim
  vectors at 15.6 Hz, which is what SupertonicTTS does. The autoencoder's 93.75 Hz frame rate looked like the
  problem, but the `compress: 6` fold is exactly the mechanism that closes it.
