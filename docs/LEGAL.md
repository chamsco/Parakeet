# Legal and terms-of-service constraints

**Read this before you point the corpus builder at any teacher.** The code enforces the parts
that can be enforced (`parakeet/data/teacher.py`), but the rest is a judgement call that needs a
lawyer, not an engineer.

## 1. Teacher licence summary

| Teacher | Code licence | Weight licence | Train a student on its audio? | Enforced in code |
|---|---|---|---|---|
| **Orpheus** (Canopy Labs) | Apache-2.0 | `license: apache-2.0` but **gated** download, and the model card's base model is `meta-llama/Llama-3.2-3B-Instruct` | *Probably*, but the Llama-3.2 Community License conditions travel with derived weights | allowed (with attribution obligations) |
| **Kokoro-82M** | Apache-2.0 | Apache-2.0 | Yes | allowed |
| **MiniMax speech-2.8-turbo** | proprietary, API only | proprietary; you own the output but grant MiniMax a free worldwide licence to use it to improve the service | **No** — terms bar "developing foundation models using MiniMax Voice" without prior written permission; a separate platform clause bans "create derivative works of … our services" and "attempting to discover or decode … the algorithms" | **refused by default** (`TeacherLicenseError`) |

## 2. MiniMax: why it is refused by default

* The MiniMax **Voice** agreement (updated 2025-06-21) §三 explicitly prohibits, without prior
  written permission, *"使用MiniMax语音开发基础模型"* — using MiniMax Voice to develop foundation
  models — together with competing behaviour and scraping data through the APP/API.
* The MiniMax **platform** terms (effective 2026-03-30, Nanonoble Pte Ltd, Singapore law / SIAC)
  ban "create derivative works of … our services" and "attempting to discover or decode the source
  code, algorithms, or object code". *UNVERIFIED whether the platform/API terms carry the same
  foundation-model clause as the Voice agreement: the live page is a JS app whose text could not
  be extracted, so this needs a human browser check.*
* Output IP **does** vest in the user, but with a free worldwide grant-back to MiniMax for service
  improvement, and a deep-synthesis marking duty attaches to generated content in some
  jurisdictions.

**Consequences for Parakeet.**

* `check_teacher("minimax")` raises `TeacherLicenseError` unless you pass
  `acknowledge_restricted=True` **and** set `PARAKEET_ACCEPT_TEACHER_TOS=1` (or pass
  `--i-have-written-permission` to the corpus builder). Both flags exist so that the choice is
  deliberate and auditable, not accidental.
* The default mixture is **Orpheus 60 / Kokoro 40** — fully permissive — and every corpus record
  stores a `license` field so provenance survives into the manifest.
* MiniMax's genuinely useful contributions that carry **no** sampling risk are adopted anyway:
  its 19 interjection tags as a *labelling taxonomy*, and its word-level timestamps as an
  alignment-reference format. Neither requires training on its audio.
* If you have written permission, the code path already works
  (`MiniMaxBackend`), and the acoustic-only nature of its output is handled by design: the loss
  is applied in latent space, which needs no codes or logits.

## 3. Orpheus: the Llama-3.2 chain

Orpheus code is Apache-2.0 and SNAC is MIT, but the released weights are gated and built on
`meta-llama/Llama-3.2-3B-Instruct`, so the **Meta Llama 3.2 Community License** applies to the
weights and, by extension, plausibly to a distilled student. Practical obligations to plan for:

* acceptable-use policy compliance and a "Built with Llama" style attribution in your model card;
* redistribution of derivative weights is permitted but must carry the licence;
* the licence is not an Apache-2.0 blanket: do not describe a Parakeet checkpoint trained on
  Orpheus output as "Apache-2.0 without conditions" without review;
* **distillation is not addressed by either text.** Neither licence explicitly permits or forbids
  training a student on teacher outputs, so treat it as an open question requiring review before
  any public release of Orpheus-derived weights.

Our mitigation in the meantime: the Tiny and Small recipes are built so that the *safe* mixture
(Orpheus + Kokoro) is the default, and swapping to Kokoro-only is a one-line change
(`--teachers kokoro=1.0`), which yields a fully permissive lineage.

## 4. Text corpora used for the prompts

* Paradee's own repo is Apache-2.0 **except** `training/data/wikitext_sents.json`, which is
  CC BY-SA 3.0 — and WikiText-103 is derived from Wikipedia (also share-alike). Do not copy that
  file into a Parakeet corpus. Supply your own prompt file (`scripts/make_teacher_corpus.py
  --texts ...`), e.g. public-domain prose or text you own.
* If you later train on Emilia, LibriLight, MLS, etc., check each dataset's own terms; several
  are CC-BY-NC or research-only, which would make commercial release impossible regardless of the
  teacher question.

## 5. Output-side duties

Synthetic speech is regulated in some jurisdictions (EU AI Act transparency, and MiniMax's own
terms impose a marking duty). Plan for: an audible or watermark-style provenance signal, a model
card stating which teachers the checkpoint was distilled from, and a "no voice cloning without
consent" policy — the Small model is a zero-shot cloner by design, so misuse is a first-class
risk, not a footnote.

## 6. Policy as implemented

```python
from parakeet.data.teacher import check_teacher, TeacherLicenseError

check_teacher("kokoro")                        # ok
check_teacher("orpheus")                       # ok (Llama-3.2 obligations travel with weights)
check_teacher("minimax")                       # raises TeacherLicenseError
check_teacher("minimax", acknowledge_restricted=True)   # ok, only with written permission
```

* Every corpus record carries `teacher`, `license`, `voice`, and a content hash.
* `docs/LEGAL.md` (this file) is the single source of truth referenced from the code's error text.
* Nothing in this repository ships teacher weights or teacher audio.
