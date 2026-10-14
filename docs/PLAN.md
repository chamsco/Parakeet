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

**Token route** (deterministic, quarter the size, one forward pass, no sampler). This section used to say it
"plateaus at ρ ≈ 0.37 with validation ≈ training" and that "more steps will not move it. **Not worth more
compute.**" That is **withdrawn**, and the measurements that withdraw it are in §134 of the verification log.

* **It had not plateaued.** Held-out frame ρ on 400 prompts from four unseen books, read every 1 000 steps, climbs
  **0.0932 → 0.1759 → 0.2102** on 1 207 items with a 9.62M text side, and **0.0908 → 0.1741 → 0.2113 → 0.2142 →
  0.2331** on 4 533 items with a ~30M one. Two configurations, ninefold different in text count and threefold in
  parameters, agreeing to within 0.005 the whole way.
* **Steps are the only thing that has moved it.** Across those two runs, neither text count nor capacity moved
  held-out ρ at matched steps. The one variable with an effect is the step counter.
* **The 0.37 figure was an in-sample number.** In-sample ρ at step 3 000 is **0.4157**, which is where that
  figure came from.
* **"Validation ≈ training" does not reproduce.** Held-out is **0.2127**, half of in-sample, a **2×** gap. At
  in-sample 0.42 the model cannot yet reproduce its own training data, so this configuration is under-fit
  rather than a converged fit against a one-to-many ceiling.
* **What is still open is the asymptote, and the increments are noisy.** The deltas run +0.0833, +0.0372, +0.0029,
  +0.0189. The +0.0029 was briefly read as a plateau and the reading was wrong: the next point came in at 0.2331,
  above everything before it. Against a measurement sd near 0.005, two adjacent deltas cannot tell "flat" from
  "slow", so the stop rule needs several consecutive points. `runs/text_long384` runs to 20 000 steps to find the
  asymptote. Taking the mean of the last three increments, +0.020 per 1 000 steps, the 0.60 the recogniser needs is
  about 18 000 steps away **if that rate held**, which a decelerating curve will not do. That is a bound, not a
  forecast.

So the correct statement is narrower than "not worth more compute" and narrower than "compute will fix it":
**more steps is the only untested lever left on CPU, its effect is real and its size is not yet measured, and the
decision to spend beyond it belongs to the asymptote measurement.**


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

* **The premise is already measured, so this run is not a fishing trip.** `scripts/latent_variance_budget.py` on the
  folded 144-dim target the flow models: narration's marginal (per-voice mean + AR(1) smoothness) explains **78.7 %**
  of the variance, leaving **21.3 %** for the text; a dialogue set built here from public-domain plays explains
  **62.6 %**, leaving **37.4 %** — **~1.75× more conditional signal** (§112, and in the model card). So the
  prediction is specific: the flow term should rise. It did not move on narration across 4.2× data and 4× steps.
* **The instrument is the limiting factor, and that is measured too.** The in-loop gate's own spread *within* one
  unchanged arm is **2.7 points** over two checkpoints against a predicted effect of ~1.5–2, so a two-arm mean
  comparison resolves this only marginally. If you run it, raise the gate's repeats (now 5, was 2) and read **more
  checkpoints per arm**, or accept an `UNRESOLVED` verdict — `scripts/analyse_domain_ab.py` reports one rather than
  a sign.
* **The corpus here may be too weak a version of the hypothesis.** `scripts/expansion_dialogue.py` builds it from
  Gutenberg plays, and **Kokoro reads the turns**: dialogue-shaped text in narration delivery — shorter utterances,
  more register variation, turn boundaries — not spontaneous prosody, backchannels or overlap. Moshi/PersonaPlex
  output or a licensed conversational set is the real test, and it needs a GPU.
* **Criterion:** the flow term climbs toward the 37.4 % that is available, and WER on the samples falls below 1.000
  with the teacher control intact. If it climbs in the gate but WER does not move, the gate is measuring the loss
  again — and §110 already showed the loss is not where the text's contribution lives.

**How to judge any of these.** By **WER on sampled audio with the teacher control** — `scripts/real_eval.py` —
never by ρ against a recording. A sampler legitimately draws a different valid rendition each time, so ρ-against-
the-target is a metric for a deterministic regressor and it conflates "different valid draw" with "wrong". The
current model's honest score is **WER 1.000 against the teacher's 0.000** at NFE 32. The intelligibility bar, when
you have samples worth measuring, is ρ ≥ 0.75 *for a deterministic route*; for the sampler, WER is the test that
cannot be gamed.

## Scaling to 7B, and adding a conversational (duplex) mode

Two separate problems, with very different costs. The short version: **7B is a GPU question and a fine-tuning
question, not a from-scratch question — and the conversational part does not need 7B at all.**

### The arithmetic, from measured rates

The 39.2M flow runs at ~**2.7 s/step** on this CPU (batch 16, 160-frame crops, measured). Parameters scale that
linearly at best, and attention scales worse with sequence length:

| model | parameters | CPU s/step (projected) | 10 000 steps |
|---|---|---|---|
| the 39.2M research flow (measured) | 39.2M | 2.7 | 8 hours |
| **7B** | 7000M (**179×**) | ~480 (8 min) | **~56 days** |

So on this machine 7B is not slow, it is **off the table** — a single step takes eight minutes, and an epoch is not
a thing that happens. The production candidate (9.63M) is 727× smaller than 7B again; "scale to 7B" is not a
continuation of this project's model line, it is a different model.

### What 7B would actually be

Not this architecture inflated. The model worth reaching is **Moshi-shaped**, because that is what PersonaPlex
fine-tuned and what a duplex conversational model has to be:

| piece | this project today | a duplex 7B |
|---|---|---|
| I/O | text → continuous latent → audio (TTS) | **two parallel streams** (user + assistant), full duplex |
| acoustics | 24 dims @ 93.75 Hz, folded to 144 @ 15.6 Hz | **Mimi**: 12.5 Hz, 8 codebooks (~100 tokens/s) |
| backbone | ConvNeXt text encoder + flow | Helium-class LLM, 7B, with an **inner/depth transformer** for the codebook axis |
| conditioning | voice embedding, optional plan | **text role prompt** + **voice embedding**, as PersonaPlex does |
| training | offline cached teacher corpus, single stream | streaming, interleaved, with turn-taking supervision |

What carries over, and is not nothing: the autoencoder work (a Mimi-like codec is the same problem this project
already solved to WER 0.000 round trip), the blockwise **streaming** inference and its measured cost, the whole
evaluation discipline, and `expand_corpus.py`'s pipeline for building a corpus without hitting the traps.

### The realistic path, in cost order

1. **Do not train 7B from scratch.** PersonaPlex did not either — it fine-tuned **Moshi**, whose weights are open.
   Start there. Fine-tuning a 7B is a **one-to-eight-GPU** job: full fine-tune with Adam wants ~80–120 GB of
   optimizer+gradient+activation state (≈8× A100/H100); **QLoRA on the frozen base fits a single 24–48 GB card**,
   which is a rented afternoon rather than a cluster.
2. **Add the two conditioning channels PersonaPlex has and this project does not**: a **text role prompt** and a
   **voice embedding**. Both are small, additive changes to a Moshi-shaped model, and both are things this project
   has already built separately (role-prompt text handling; per-voice conditioning with a measured effect).
3. **Then distil *down***, which is the actual objective. The point of a 7B duplex teacher is that its licence is
   permissive (NVIDIA Open Model Licence — whose derivative clause has **not** been read; `docs/LEGAL.md` records
   it as unverified) and that it *converses*, which no teacher here does. A distilled 7B→small student with duplex
   capability is a coherent target; a from-scratch 7B is not this project's goal.

### The conversational part, without 7B

This is the half that can be tested **here and now**, and its hypothesis is the one untested item in `PLAN.md`
above: every corpus element in this project is **narration** — one speaker, one register, smooth prosody, no
turn-taking — an unusually *easy* distribution to model unconditionally, which is exactly what makes the text look
worthless to a flow that only has to fit the marginal.

Started: `scripts/expansion_dialogue.py` extracts spoken turns from public-domain plays (Gutenberg: Hamlet,
Macbeth, The Importance of Being Earnest, A Doll's House) into `data/prompts/dialogue.txt` — **1 461 turns,
119 729 characters**, emitted **in play order** so that the generator's round-robin voice assignment makes
consecutive prompts **alternate between speakers**. Generated with the *same* 11-voice round-robin as the narration
corpus, so the only variable that changes is utterance style.

**Honest limitation:** Kokoro *reads* those turns. This is dialogue-shaped text in narration delivery — shorter
utterances, more register variation, turn boundaries — not spontaneous prosody, backchannels or overlap. It is a
**weaker** version of the hypothesis than a real duplex corpus, and it is the version this hardware can produce.
A true test needs either Moshi/PersonaPlex output (GPU) or a licensed conversational set.

**What not to spend time on.** Cross-attention gain, the pooled conditioning token, plan or residual-plan
conditioning, logit-normal `t`, more sampling steps (NFE 4 → 32 changed nothing), and a pretrained text encoder:
all measured here, all null (`docs/05-VERIFICATION.md` §§94–104). More corpus on the *narration* domain is also
null (§94). Those are the cheap ideas, and they have been spent.
