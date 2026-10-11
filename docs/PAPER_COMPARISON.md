# How this implementation compares with the papers

Written after 68 rounds, because the objective asks for a distillation of these specific papers and the
honest answer is "the architecture matches; the scale does not". Numbers are from the papers where cited and
measured here where they describe this code.

## The latent the flow actually sees

SupertonicTTS, in its own words: *"With `Kc = 6` and `C = 24`, the model processes 144-dimensional vectors at
a reduced frame rate of approximately 14 Hz."*

This project, from its config and cache: the autoencoder emits **24 dims at 93.75 Hz** (hop 256 at 24 kHz),
and the flow folds those by `compress: 6` before modelling them — so the velocity field is asked for
**144-dimensional vectors at 15.6 Hz**.

| | latent the flow models | dims/s |
|---|---|---|
| SupertonicTTS | 144 dims @ ~14 Hz | ~2 000 |
| **this project** | 144 dims @ 15.6 Hz | **2 250** |

**They match.** An earlier version of `docs/PLAN.md` claimed this project's target was "unusually large" and
recommended a lower-rate bottleneck as the fix; that was wrong, and it is corrected here rather than left
standing. The frame rate of the *autoencoder* is higher (93.75 Hz against their ~14 Hz), but the temporal
fold is exactly the mechanism that closes the gap, and it is already implemented.

## The rest of the design comparison

| axis | SupertonicTTS (paper) | this project | verdict |
|---|---|---|---|
| autoencoder latent | 24 dims, mel input, Vocos-style, causal streaming decoder | 24 dims, mel input, causal decoder | matches |
| flow input | 144 dims @ ~14 Hz | 144 dims @ 15.6 Hz | **matches** |
| blocks | ConvNeXt | ConvNeXt | matches |
| text | raw characters, no G2P, implicit alignment by cross-attention | raw characters, no G2P, implicit alignment | matches |
| alignment acceleration | context-sharing batch expansion `Ke = 4` | `context_expansion: 4`, applied in `flow_loss` | matches |
| NFE at inference | 32 preferred; 4 needs distillation | `nfe: 32`, `distilled_nfe: 4`, `reflow` stage present (not yet run) | matches, unexercised |
| parameters | 44M (0.5M duration + 18.5M text→latent + 25M latent→speech) | **21.2M** total / 18.8M trainable at the documented 256×4 text side (measured); **39.2M** / 34.2M at 512×6 | **smaller**, at the documented geometry |
| **training data (generative module)** | **945 hours** | **~8 h** (1.9 h corpus + 6.1 h generated this month) | **~120× short** |
| hardware | RTX 4090 | CPU only (both GPU routes closed with evidence) | — |

## What the comparison actually says

The design axes line up one by one — latent size, frame rate after folding, block type, character-level text,
implicit alignment, `Ke = 4`, NFE 32. The one axis that does **not** line up is **data**: 945 hours against
roughly two, and the training that produced the paper's model is correspondingly far out of reach on this
machine (its 44M model trained on 945 h, this one at 21M on ~2 h, on CPU).

That reframes the remaining work. It is not "find the missing architectural trick" — the architectural
questions that can be answered cheaply here have been answered, and the answers match. It is "get more
teacher audio, or more compute, and let the same architecture run longer". The per-voice normalisation of
round 62 is what makes the first of those actually usable; `docs/PLAN.md` costs the second in CPU-hours.

Reporting on this project's own terms, the one thing it has that the paper does not is the **measurement
discipline**: ρ as a calibrated target (≥ 0.75 for intelligibility, failing at 0.60), the flow term's text
dependence as a conditioning gate, and four instrumentation flaws found and fixed rather than inherited. That
does not make a model, but it does mean the next 945 hours would be spent on the right thing.

## What was adopted from each paper, and where it lives

The brief was to read these three and find the best way to mix them. This is that answer, item by item, with the
code that carries it and what measuring it established — including the cases where the paper's idea was adopted
and did **not** work here, which is the more useful half.

**[Paradee](https://www.alphaxiv.org/abs/2610.06817)** — Kokoro → 8M, two halves trained separately against a
cached teacher corpus.

| adopted | where | measured outcome |
|---|---|---|
| two halves trained **separately** against a frozen cached teacher | `make_teacher_corpus.py` + `runs/*/latent_cache`; stages `distill-text` and `flow` | it is what made CPU training tractable at all: the teacher is paid for once, then every experiment reads a cache |
| **low spectral weight** and adversarial decoding | `parakeet/models/` + the AE stage's adversarial term | the autoencoder's round trip is *perfect* (WER 0.000), so this half is solved |
| **weight-only int8** | `export_onnx.py --pipeline` | 9.63M → **20.3 MB** at **85× real time**, waveform cosine 0.9952 |
| direct feature supervision of the text half | the token route (`distill-text`) | ρ 0.37 on held-out prompts, with the ceiling explained as the one-to-many mapping rather than capacity |

**[SupertonicTTS](https://www.alphaxiv.org/abs/2503.23108)** — 44M, 945 h, the closest architectural match.

| adopted | where | measured outcome |
|---|---|---|
| 24-dim continuous latent + temporal compression `Kc = 6` | `autoencoder.latent_dim`, `flow.compress` | the velocity field models **144 dims at 15.6 Hz** against the paper's ~14 Hz — the axes match |
| ConvNeXt blocks, character-level text, implicit alignment | `parakeet/models/flow.py`, `text.mode` | implemented, no G2P, no aligner |
| **context-sharing batch expansion `Ke = 4`** | `flow.context_expansion`, applied in `flow_loss` | implemented as designed |
| the papers' reason for **sampling** rather than regressing | the flow stage | it is why the deterministic route's plateau (ρ 0.37) is a dead end rather than a tuning problem |
| — | — | **what did not transfer:** its conditioning, at this scale. Eight interventions (§§94–101) left the flow term at **~1–2 %** (2–7 % before round 114 found the gate was inflated by train-mode dropout) and ρ indistinguishable from zero. The difference is 945 h against 8 h |

**[PilotTTS](https://www.alphaxiv.org/abs/2605.27258)** — factorised conditioning, cross-sample paired training.

| adopted | where | measured outcome |
|---|---|---|
| **factorised conditioning**: identity separated from style | `speaker.py` + `voice_embed` + style tokens | identity/style conditioning is live (a change of reference moves the conditioning by 0.2254, §12) |
| **cross-sample paired training** — a *different* utterance of the same voice as the reference, plus a different-voice negative | `LatentShardBatchSource(pair_references=True)` | implemented; it also fixed a real bug (the reference was being dropped at collation, leaving zero gradient) |
| style as a **separation** loss rather than same-speaker consistency | the style term | consistency leaked identity into the style channel; separation is what the paper's factorisation implies |
| teacher-generated data for scarce capabilities | the Orpheus tag vocabulary, and this month's Kokoro expansion | the tag taxonomy is in the manifest; the expansion added ~6 h of free audio and is what the corpus work rests on |

**The mixture itself, measured.** The one experiment that tests "mixing teachers" as a mechanism rather than a
config value: a sweep between two synthetic teachers at 92 Hz and 215 Hz moves the student's predicted F0
monotonically **87 → 200 Hz**, with the fit degrading to the low teacher in the same order (§7). So the mixture
is real and controllable, not a label — which is what makes the Kokoro + Speechify corpus a *distillation* of two
teachers rather than one teacher plus filler.

## A fourth model, considered: NVIDIA PersonaPlex

Suggested as something that "might help a lot". Assessed against what this project has *measured*, rather than
against how impressive it is — it is impressive, and on four of the five axes below that does not matter.

What it is ([repo](https://github.com/NVIDIA/personaplex), [weights](https://huggingface.co/nvidia/personaplex-7b-v1),
[paper](https://arxiv.org/abs/2602.06053)): a **7B full-duplex speech-to-speech** conversational model, finetuned
from **Moshi** on top of the **Helium** LLM, with role prompts for persona and fixed voice embeddings for timbre.
Task tag on the model card: `audio-to-audio`. Licence: MIT code, `nvidia-open-model-license` weights, gated.

| question | answer | consequence |
|---|---|---|
| is it a TTS model? | **no** — it consumes speech and *responds*, so it cannot read a given sentence | its audio cannot be paired with the Gutenberg prompts the way Kokoro's is |
| size | **7B** | versus this project's 9.63M production candidate and 21–39M flow: a teacher, not a student |
| does it run here? | **no GPU on this machine** (both routes closed with evidence) | 7B duplex inference on CPU is not a practical corpus generator |
| licence | open weights under a named licence whose derivative clause has **not been read** | recorded in `docs/LEGAL.md` as *not established* rather than assumed |
| what would it actually give us? | **conversational audio** — spontaneous prosody, turn-taking, backchannels — which the current 8 h corpus has none of | the one hypothesis worth testing, below |

**The hypothesis it enables, which is the only reason to care.** This project's blocker is measured precisely: the
velocity field fits the **marginal** and ignores the text (`§§94–101`), and adding 4.2× more *narration-style*
audio did not change that. All of that audio was read-aloud prose from Kokoro and Speechify, which is unusually
*easy* to model unconditionally — one speaker, one register, smooth prosody, little turn-taking. PersonaPlex's
domain is the opposite: spontaneous conversation is the case where the acoustic outcome depends most strongly on
*what* is being said and *when*. If the marginal's cheapness is why the text term is not worth learning, then
**conversational targets should raise the flow term's share** — and that is a measurement this project already has
the instrument for (`scripts/text_dependence.py`, printed in-loop, with its noise floor characterised).

**The cheap version of that test does not need a 7B model.** Any conversational corpus does it, and Moshi's own
open weights or a small conversational dataset would answer the question far sooner than standing up PersonaPlex.
PersonaPlex is the *best* source of that domain, not the first one to try.

**What I did with it now:** recorded it as a teacher *candidate* with its licence state, and changed nothing about
the run in flight. A resource that would need a GPU, a licence reading and a new corpus domain does not get to
interrupt a measurement, and the honest summary is that it helps the *next* session — a GPU one — more than this
one.
