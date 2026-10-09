# Data pipeline

Two things happen in the data layer, and they are separable:

1. **Real-audio curation** for the autoencoder and the speaker encoder. This is the PilotTTS
   blueprint. PilotTTS's paper claims a released pipeline, but the repository is *inference-only*
   (no training or pipeline code, and issue #2 asking for it has no maintainer reply), so the
   pipeline below is our implementation of the paper's description plus the concrete thresholds
   from CosyVoice 3 where PilotTTS leaves them unspecified. Items marked **[published]** are from
   the papers; **[proposed]** are ours.
2. **Teacher corpus construction + latent caching** for distillation. This is Parakeet-specific
   and is implemented in `parakeet/data/`.

## 1. Real-audio curation (PilotTTS-style)

**Implemented and tested** in [`parakeet/data/curate.py`](../parakeet/data/curate.py) — run it over a
manifest with `curate_manifest(records, load_wav, out_dir, asr_fn=..., mos_fn=...)`.  Every
threshold below is tagged in `CurateConfig` as `[published]` or `[proposed]`, and two honesty rules
are enforced in code:

* **A MOS is never faked.**  DNSMOS needs a model; if you do not inject one the report records
  `mos_source="unavailable"` and MOS is simply not gated on.
* **A rejected item is never deleted.**  Rejects go to `rejected.jsonl` with all their reasons
  (all reasons, not just the first), so filters can be re-tuned and audited without re-decoding.

| # | Stage | Tool | Threshold / rule | Source |
|---|---|---|---|---|
| 1 | Decode + loudness normalise | ffmpeg/soundfile | single channel, 24 kHz target; peak `raw/max(raw)*0.6` | `[published]` CosyVoice 3 |
| 2 | Voice activity + speaker change | pyannote | drop segments without speech | `[published]` |
| 3 | Quality (MOS/SNR) | DNSMOS + SenseVoiceSmall **injected** | deficient if **MOS ≤ 3.5**; SNR estimate ≥ **15 dB** (proposed) | `[published]` MOS, `[proposed]` SNR |
| 4 | Enhancement for low-quality | resemble-enhance | only for items failing 3 but not 2 | `[published]` |
| 5 | Duration filter | built in | 3–30 s | `[published]` CosyVoice 3 |
| 6 | Clipping / silence / bandwidth | built in | clipping ≤ 1 %, silence ≤ 50 %, 99 % bandwidth ≥ 5 kHz | `[proposed]` |
| 7 | ASR cross-validation | Paraformer + FireRedASR + Whisper | keep if **average pairwise WER < 15 %** | `[published]` CosyVoice 3 |
| 8 | Punctuation via forced alignment | MFA / Qwen3-Force-Alignment | add comma if gap **≥ 300 ms**, remove if **≤ 50 ms** | `[published]` CosyVoice 3 |
| 9 | Text/speech length-ratio tails | built in | drop smallest **1 %** and largest **5 %** | `[published]` CosyVoice 3 |
| 10 | Overlap/truncation/synthetic-speech detectors | pyannote OSD + classifiers | drop on positive | `[published]` PilotTTS |
| 11 | Spectral rolloff check | built in (`spectral_stats`) | drop content-less bands | `[published]` PilotTTS |
| 12 | Speaker tagging + dedup | 3D-Speaker, MinHash text dedup | cluster identity; dedup near-identical text | `[published]` PilotTTS |
| 13 | **Keep rejects with tags** | built in | never delete | `[published]` PilotTTS |

**A caveat we discovered while implementing it:** the percentile SNR estimate is *not measurable*
on continuous speech with no pauses — the 10th and 90th percentiles of frame energy coincide, so
the ratio is meaningless.  `estimate_snr_db` therefore returns `None` and the gate is skipped
(`notes: snr_unevaluable(no_noise_floor_observed)`) rather than gating on a bogus 0 dB and
rejecting perfectly good continuous speech.  If you need a real SNR gate, replace it with a proper
VAD + noise-tracking estimator.

The AND of all gates over PilotTTS's ~200 k-hour pool retained ~200 k h in their setup. Aim far
lower for Parakeet: 100–1 000 h is plenty for the autoencoder, and SupertonicTTS showed 945 h is
enough for the generative half.

## 2. Teacher corpus

```
prompts.txt ─► mixture scheduler ─► {Orpheus, Kokoro[, MiniMax]} ─► wav/ + manifest.jsonl
                                                                          │
                                    Parakeet autoencoder (frozen) ────────┤
                                                                          ▼
                            latent shards: ids, durations, F0, energy, latent_token, latent, log_mel
```

* **Mixture scheduler** (`synthesize_corpus`): teachers are interleaved with a golden-ratio
  sequence so every shard contains the whole mixture. Shard-local balance matters — a shard that
  is 100 % one teacher produces a stretch of single-teacher gradient.
* **Voice assignment**: round-robin over each teacher's voice list (Orpheus's 8 English voices;
  Kokoro's voice set), so the student sees voice diversity even in the single-voice Tiny setup
  (useful for the speaker encoder and for the Small model).
* **Provenance**: every record carries `teacher`, `voice`, `license`, `duration_s`, `tags`, and a
  content hash. Licence enforcement happens here (see [LEGAL.md](LEGAL.md)).
* **Alignment source**: durations come from the teacher where it exposes them (Kokoro predicts
  them) or from a forced aligner (Qwen3-Force-Alignment, as PilotTTS) for Orpheus/MiniMax.
  `extract_signals` has an energy-envelope fallback which is **explicitly a fallback** — training
  on it long-term will produce wrong rhythm and it is marked as such in the code.
* **Cached signals** (`LatentShardWriter`, shards of 500 like Paradee): per-utterance
  `ids, n_frames, latent, log_mel, durations, f0, energy, latent_token`.  **Target conventions**
  (enforced by `extract_signals` / `token_targets_from_corpus`): durations in log space, F0 as
  normalised log-Hz in `[0, 1]`, energy as normalised dBFS in `[0, 1]`, and the per-token
  `latent_token` = mean normalised frame latent over each token's span (Paradee's "phoneme
  feature", and the target that beat end-to-end distillation 4.39 vs 3.78 UTMOS).
  The prosody targets are deliberately O(1): `tests/test_learning.py` caught that regressing raw
  dB or quantised F0 bin indices made a single term dominate the objective by ~100×, which starves
  the other heads of gradient.

Why two-stage (synthesise, then cache)? Re-rendering is the expensive part; caching makes stage 2
and 3 training pure tensor regression and lets you iterate on the student without ever re-running
the teacher.

## 3. Prompts

Paradee used 12 000 WikiText-103 sentences. We cannot: that file is CC BY-SA 3.0 and WikiText is
Wikipedia-derived. Supply your own prompt list (`--texts prompts.txt`). For tag supervision,
interleave tag-bearing prompts (`<laugh>`, `<sigh>`, …) so the model sees each tag with enough
context; PilotTTS's recipe is 50/50 mixed-prompt fine-tuning on a small capability set rather
than letting tags dominate the bulk corpus.

## 4. Commands

```bash
# 1. render the teacher corpus (permissive mixture by default)
python scripts/make_teacher_corpus.py \
    --texts data/prompts.txt --out data/teacher_corpus \
    --teachers orpheus=0.6,kokoro=0.4 --limit 20000

# 2. cache latents + teacher signals (needs a trained autoencoder)
python scripts/make_teacher_corpus.py --cache-only \
    --corpus data/teacher_corpus --config configs/parakeet_tiny.yaml \
    --ae-checkpoint runs/parakeet-tiny/autoencoder_last.pt \
    --cache-out data/latent_cache

# 3. train against the cache
python scripts/train.py --config configs/parakeet_tiny.yaml \
    --stage distill-text --cache data/latent_cache --steps 8000
```

`--dry-run` on `train.py` (or omitting `--cache`) uses shape-faithful random batches, which is how
the whole curriculum is validated on CPU without any data.

## 5. Known caveats

* **Synthetic-only training is a documented trap.** Canopy warns that training on synthetic audio
  maps everything onto the same codec tokens and destroys codebook utilisation. Parakeet's
  answer: the shared representation (AE) and the identity encoder are trained on **real** audio;
  teacher audio supervises rendering/expressivity. If you only ever train on teacher output, expect
  a "good at saying these sentences" model rather than a general one.
* MiniMax's word-level timestamps are genuinely useful for alignment supervision and carry no
  sampling risk — use them as a *format*, not as training audio.
* Frame rates differ across teachers: Kokoro 40 Hz frames with F0 at 80 Hz, Orpheus/SNAC 12 Hz
  super-frames with 7 codebooks, MiniMax arbitrary. All of this is resolved by re-encoding audio,
  not by resampling feature streams — do not try to align teacher *features*.
