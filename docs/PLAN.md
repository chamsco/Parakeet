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

> **Corrected after that measurement was tested — read the next two paragraphs before acting on this table.**
> The curve was taken at one checkpoint per size, and the per-checkpoint noise floor was later measured at
> **±3–4 points** (`§96`), so its 1.9 → 7.4 spread is within a single reading's error for the *individual* points;
> only the monotone ordering across four sizes was ever suggestive.
>
> **It does not extend to this project's scale.** On 2.3× more utterances (4 533 against 1 978) the flow term
> measured **2.4 %**, not more, and eight conditioning interventions — cross-attention gain, the pooled token,
> plan and residual-plan conditioning, logit-normal `t`, the 4.2× corpus, a pretrained text encoder, and
> classifier-free guidance from 1.0 to 8.0 — produced **no measurable change** (`§§94–101`). Sampled ρ stayed
> indistinguishable from zero with its spread measured (`§100`).
>
> So the sentence that used to follow this table — "this is where compute could pay" — is **withdrawn**. What the
> evidence supports now is narrower: at this model size and the step budgets tested (2 000–4 000 steps), nothing
> available here moves the flow's conditioning, and the remaining candidate is the paper's own regime — 945 h of
> audio and the compute to use it — not another few CPU-hours on 8 h. The one cheap test still open is simply
> *more steps*, which is what `runs/flow_v3_long` is running with a pre-registered endpoint (§101).

## What would change the picture

* **more teacher audio.** This is the measured gap, not a hunch: SupertonicTTS trained its generative module
  on **945 hours** and this project has **~8 h** (1.9 h corpus plus 6.1 h generated since, per-voice normalised
  so it is usable). See [PAPER_COMPARISON.md](PAPER_COMPARISON.md) — the architectural axes line up one by one
  (144-dim folded latents at ~15 Hz, ConvNeXt, character-level text, implicit alignment, `Ke = 4`, NFE 32), so
  the difference is data and compute rather than a missing trick;
* **a GPU** (any CUDA card, even rented for a few hours): the code is device-agnostic, plain fp32, with no
  CUDA-specific assumptions and a verified `--resume`, so it would move as-is;
* ~~a lower-rate learned acoustic bottleneck~~ — **retracted.** The velocity field already models 144-dim
  vectors at 15.6 Hz, which is what SupertonicTTS does. The autoencoder's 93.75 Hz frame rate looked like the
  problem, but the `compress: 6` fold is exactly the mechanism that closes it.

## If you have a GPU: the first three runs, in order

Everything below is a measurement this project already made, so each run has a *number to beat* rather than a hope.
Run them in this order and stop as soon as one fails its criterion — the point is not to be thorough, it is to
find out fast whether the paper's regime is the missing ingredient.

**Run 1 — the cheap falsification. (≈2 GPU-hours.)** Train the flow on the existing ~8 h corpus for 50 000 steps
at batch 16 with 160-frame crops, gate every 1 000, ρ every 5 000. On this machine the same setup ran at ~1–3
s/step; a rented card should make 50 000 steps trivial.

* **Watch:** the flow term's text dependence (it prints itself as `[gate] step N: …`; measure it in **eval** mode —
  the in-loop printout is correct, but an earlier version of `text_dependence.py` ran the model in train mode where
  10 % conditioning dropout inflates the number).
* **Beat:** the **~1–2 %** band it has been stuck in — uniformly across the trajectory, 0.3–1.3 % at every timestep
  from t=0.05 to t=0.95, so a reweighted or curriculum loss is not the answer. On CPU, 12 000 steps across two runs,
  eight conditioning interventions and 8× the sampling steps never left it.
* **What makes this the right test:** 43 % of the target's variance is text-determined on the representation the flow
  models (144 dims at 15.6 Hz, after voice and temporal smoothness are accounted for), so the signal is there and the
  field uses a small part of it. If 50 000 GPU steps do not move the flow term toward that 43 %, training volume on
  this architecture is not the missing ingredient and Run 3 is where the answer is. If it climbs, keep going.
* **Judge it by WER on the samples, not by ρ** — the sampler legitimately draws a different valid rendition each
  time, so ρ against a recording conflates "different valid draw" with "wrong" (see "How to judge" below).

**Run 2 — the corpus, only if Run 1 climbs. (≈2–5 GPU-days + generation time.)** Grow to at least 100 h using the
free local teacher: `scripts/expansion_prompts.py` → `make_teacher_corpus.py` → `scripts/expand_corpus.py`, which
caches, merges and per-voice-normalises in one command and fails loudly on each of the ways an expansion went
wrong here. Scale the teacher mixture beyond Kokoro if you also want the realism, and check licences first
(`docs/LEGAL.md` — Speechify's permission is scoped, and PersonaPlex's derivative clause is unread).

* **Criterion:** the paper's own ratio is ~100×. Anything less is a different experiment.

**Run 3 — the domain, if Run 1 fails. (≈1 GPU-day.)** This is the one untested hypothesis left, and it is cheap
because it changes the *data's character*, not the architecture: every corpus element here is **narration** — one
speaker, one register, smooth prosody, no turn-taking — which is an unusually *easy* target to model
unconditionally, and a hard target to model unconditionally is what makes the text worth using. Train on
conversational audio (Moshi's open weights, or any small conversational set; the corpus pipeline is agnostic) and
compare the flow term against Run 1's band.

**How to judge any of these.** By **WER on sampled audio with the teacher control** — `scripts/real_eval.py` —
never by ρ against a recording. A sampler legitimately draws a different valid rendition each time, so ρ-against-
the-target is a metric for a deterministic regressor and it conflates "different valid draw" with "wrong". The
current model's honest score is **WER 1.000 against the teacher's 0.000** at NFE 32. The intelligibility bar, when
you have samples worth measuring, is ρ ≥ 0.75 *for a deterministic route*; for the sampler, WER is the test that
cannot be gamed.

**What not to spend time on.** Cross-attention gain, the pooled conditioning token, plan or residual-plan
conditioning, logit-normal `t`, more sampling steps (NFE 4 → 32 changed nothing), and a pretrained text encoder:
all measured here, all null (`docs/05-VERIFICATION.md` §§94–104). More corpus on the *narration* domain is also
null (§94). Those are the cheap ideas, and they have been spent.
