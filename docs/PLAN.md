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
| flow | batch 16, 160-frame aligned crops | ~0.5 s/step | **7 CPU-hours** |
| token route (`distill-text`) | batch 16, 512-dim / 6-layer text side | ~0.8–1.2 s/step | 11–17 CPU-hours |
| audio autoencoder | batch 8 | ~0.5 s/step | 7 CPU-hours |

So a serious run on this machine is **single-digit to low-double-digit CPU-hours** per 50 000 steps — days of
wall clock on a sleeping desktop, not weeks. Two artefacts already exist to keep that honest:
`scripts/rho_curve.py` reads ρ from checkpoints, and `scripts/text_dependence.py` reads the conditioning.

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
the velocity field's own objective is only ~2–3 % text-dependent at 200 utterances, while the 2-utterance
regime reached ρ 0.25 — so the field *can* learn conditioning when the marginal distribution has little to
offer it. **This is where compute could pay**, and the number to move is the flow term's text dependence.

## What would change the picture

* a **lower-rate learned acoustic bottleneck** (the papers' actual design: ~10–50 dims at 12.5–25 Hz). This
  project's frame latent is 24 dims at 93.75 Hz — 2 250 dims/s — so the flow's target is unusually large and
  its marginal unusually cheap to fit. The token latents are *not* a substitute: they measure ρ 0.95 against
  the frame latents, i.e. nearly lossless, so they inherit the same problem;
* **a GPU** (any CUDA card, even rented for a few hours): the code is device-agnostic, plain fp32, with no
  CUDA-specific assumptions and a verified `--resume`, so it would move as-is;
* a **larger, more diverse corpus** — with the caveat that `docs/05-VERIFICATION.md` §69 records why the two
  expansions built so far were not usable, and that per-voice normalisation (§74) fixed the scale half of it.
