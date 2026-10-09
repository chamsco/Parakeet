# PilotTTS and its Data Pipeline — Research Note for Parakeet

**Provenance.** All facts retrieved 2026-11-13 (UTC) from: [paper arXiv:2605.27258v2](https://arxiv.org/html/2605.27258v2)
(also read as a [PDF text render](https://r.jina.ai/https://arxiv.org/pdf/2605.27258v2) — identical numbers),
[released code](https://github.com/AMAPVOICE/PilotTTS), [released weights](https://huggingface.co/AmapVoice/PilotTTS),
[CosyVoice 3 arXiv:2505.17589v1](https://arxiv.org/html/2505.17589v1), [maintainer issue threads](https://github.com/AMAPVOICE/PilotTTS/issues),
[Orpheus TTS](https://raw.githubusercontent.com/canopyai/Orpheus-TTS/main/README.md), [MiniMax speech-2.8-turbo](https://replicate.com/minimax/speech-2.8-turbo).
Fetched page text was treated strictly as data. **Legend:** `[CODE]` = read in released source; `[PAPER]` = stated in
the paper; **UNVERIFIED** = absent or inconsistent across sources; *PROPOSED* = my own recommendation, not from a source.

---

## 1. PilotTTS implementation detail

### 1.1 Repo and released artifacts

- Repo is **inference-only**: `configs/`, `demo.py`, `inference.py`, `webui.py`, `pilot_voice/`, `third_party/{cosyvoice,Matcha-TTS}`, `tokenizer/`, `requirements.txt`, `assert/` — **no training code, no data-pipeline code** ([contents API](https://api.github.com/repos/AMAPVOICE/PilotTTS/contents/)). Apache-2.0, `master`, 219 stars / 25 forks / 11 open issues, created 2026-05-26, last push 2026-06-02 ([repo metadata](https://api.github.com/users/AMAPVOICE/repos?per_page=100)). No sibling pipeline repo exists under the org.
- Model card defects: placeholder clone URL (`github.com/xxx/pilot-tts.git`) and `snapshot_download('xxx/Pilot-TTS')`; English README says release "2025.05", Chinese says "2026.05" ([README](https://huggingface.co/AmapVoice/PilotTTS/raw/main/README.md), [README_zh](https://huggingface.co/AmapVoice/PilotTTS/raw/main/README_zh.md)).
- Released files ([HF tree API](https://huggingface.co/api/models/AmapVoice/PilotTTS/tree/main); repo total 5,611,393,795 B): `pilot_tts.pt` = 1,599,924,318 B; `pilot_tts_instruct.pt` = 1,599,822,744 B; `wav2vec2bert_stats.pt` = 9,343 B; `Fun-CosyVoice3-0.5B/{campplus.onnx, cosyvoice3.yaml, flow.pt, hift.pt, speech_tokenizer_v3.onnx}`; `Qwen3-0.6B/config.json` (**no LLM weights**); `assert/prompt.wav`.
- Consequences: (a) at inference the Qwen3 weights are unneeded — `init_from_pretrained=False` builds the LM from config and the PilotTTS checkpoint overwrites all LM params ([`model.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/model.py), [`engine.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/engine.py)); (b) the decoder path is `vocoder_only=True`, i.e. `flow.pt` + `hift.pt` only; (c) `facebook/w2v-bert-2.0` must be fetched separately from Meta.

### 1.2 Components and exact sizes

| Component | Spec | Source |
|---|---|---|
| AR backbone | Qwen3-0.6B | [PAPER](https://arxiv.org/html/2605.27258v2), [`model.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/model.py) |
| Audio-token embedding | `nn.Embedding(6563, embed_dim)` xavier, **concatenated** to Qwen `embed_tokens`; `lm_head` re-init xavier | [CODE] [`model.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/model.py) |
| Conditioner encoder | `ConformerEncoder(input_size=1024, output_size=512, linear_units=2048, heads=8, num_blocks=6, input_layer="linear")` | [CODE] same |
| Q-Former adapter | `PerceiverResampler(dim=LLM embed, dim_context=512, num_latents=32, depth=2, dim_head=64, heads=8, ff_mult=2)` → **32 condition tokens** | [CODE] [`model.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/model.py), [`perceiver.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/modules/perceiver.py) |
| Style features | frozen `facebook/w2v-bert-2.0`, `hidden_states[17]`, MaskGCT mean/var normalization | [CODE] [`model.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/model.py), [`utils.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/utils.py) |
| Speaker encoder | frozen CAMPPlus `campplus.onnx`, onnxruntime **CPU**; 1 embedding zero-padded to LLM dim and prepended | [CODE] [`engine.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/engine.py) |
| CFM decoder | Conditional Flow Matching + DiT, paper says ≈**300M** params, **10-step** denoising, cond. `[M_ref, s, e_Audio^tgt]` | [PAPER](https://arxiv.org/html/2605.27258v2) |
| Vocoder | HiFi-GAN (`hift.pt`), output **24 kHz** | [PAPER](https://arxiv.org/html/2605.27258v2), [config](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/configs/infer_pilot_tts_instruct.yaml) |

Whether the *released* `flow.pt` is really a 300M DiT is **UNVERIFIED** (no parameter count published).

### 1.3 Q-Former conditioning and cross-sample paired training

- Dual pathway: (1) Q-Former compressing reference w2v-BERT features into **32 fixed condition tokens**, plus (2) a frozen **CAMPPlus static speaker embedding** ([PAPER §3.1, §3.3.1](https://arxiv.org/html/2605.27258v2)).
- Stated rationale: token continuation clones well but degrades on noisy/short prompts and costs more inference; a pure speaker vector is robust but drops timbre detail and dynamic style; giving the speaker vector separately lets the Q-Former "focus on extracting dynamic speaking style" ([PAPER §3.3.1](https://arxiv.org/html/2605.27258v2)).
- **Cross-sample paired training** = for each sample, the reference used to compute `s` and `c` is a **different utterance of the same speaker**, forcing content-independent speaker attributes and enabling downstream emotion/dialect control ([PAPER §3.3.2](https://arxiv.org/html/2605.27258v2)).
- Paper Eq. 4 sequence: `[s, c(32), e_BT, |lang|, |emo|, e_Text, e_ET, e_BA, e_Audio, e_EA]`. In code, text tokens are offset by `vocab_size+2` (=6563), one prompt-start token = 6561 is appended, and the 32 condition tokens + speaker embedding are prepended **as embeddings, not vocabulary tokens** ([`engine.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/engine.py)).
- Text format: `<|0.00|><|{language}|>{text}<|0.02|>` ([`tools/text.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/tools/text.py)); the meaning of `<|0.00|>`/`<|0.02|>` is **UNVERIFIED** (most likely text-region boundaries `e_BT`/`e_ET`).
- Sampling: RAS sampling from VALL-E 2 ([`sampling.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/sampling.py), [VALL-E 2](https://arxiv.org/abs/2406.05370)); `generate()` defaults `top_k=30, top_p=0.8, temperature=1.0`, `total_len = min(2048, max_gen_len + prompt_len)`, EOS = 6562, index 6561 masked to `-inf`; shipped config uses `top_p=0.95, top_k=20, temperature=1.0, vocab_size=6561` ([`model.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/model.py), [config](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/configs/infer_pilot_tts_instruct.yaml)).
- Text tokenizer = Qwen3 BPE + **pinyin tokens**: initials `<PY1>ZH<PY1/>`, finals `<PY2>IANG<PY2/>`, tones `<PY3>1..5` ([`added_tokens.json`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/tokenizer/added_tokens.json)); deps include `WeTextProcessing`, `pypinyin`, `wetext`, `tiktoken` ([requirements.txt](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/requirements.txt)).
- Prompt handling: raw prompt truncated to **15 s** (`15 * 16000`) for w2v-BERT; CAMPPlus consumes a loudness-normalized copy ([`engine.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/engine.py)).

### 1.4 Frozen CAMPPlus speaker encoder

- [Cam++ (Interspeech 2023)](https://arxiv.org/abs/2303.00332), frozen, `CPUExecutionProvider` ([`tools/audio.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/tools/audio.py)).
- Front end: 80-dim Kaldi fbank, `dither=0`, `sample_frequency=16000`, per-utterance mean subtraction; input first loudness-normalized `gain_db = -24 - 10*log10(mean(wav^2))`, then clipped ±1 (same file). Output embedding dimension is **UNVERIFIED** (never documented/printed).
- Same `campplus.onnx` ships inside the CosyVoice 3 model dir; the SIM evaluation metric uses a *different* embedder (see 1.9).

### 1.5 Speech tokenizer: CosyVoice 3 single-codebook FSQ, adopted unchanged

- FSQ, single codebook, **25 Hz** (one token per **40 ms**), codebook **(2K+1)^D = 6,561** → since 6561 = 3^8, **D = 8, K = 1** (derived from the paper's own equation; only the product is printed) ([PAPER §3.2](https://arxiv.org/html/2605.27258v2)).
- Semantic content comes from CosyVoice 3's **5-task supervised training**: ASR, LID, SER, AED, SA ([PAPER §3.2](https://arxiv.org/html/2605.27258v2)).
- PilotTTS vocabulary extension: **6,563** audio slots — 0…6560 FSQ codes, 6561 reserved/masked, 6562 EOS ([`model.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/model.py), [config](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/configs/infer_pilot_tts_instruct.yaml)).
- CosyVoice 3's own construction (different system): FSQ inserted into the **voice encoder of MinMo** (12 Transformer blocks + RoPE) after CosyVoice 2 used SenseVoice-Large; multi-task set ≈**530,000 h** = ASR 365K, LID 85K, SER 48K, AED 21K, SA 11K ([CosyVoice 3 §2.1, Table 3](https://arxiv.org/html/2505.17589v1)).

### 1.6 Decoder stage

- CFM + DiT, 10 denoising steps, then HiFi-GAN ([PAPER §3.4](https://arxiv.org/html/2605.27258v2)); code calls CosyVoice 3's `frontend_zero_shot` → `token2wav(token=codes, prompt_token=flow_prompt_speech_token, prompt_feat=prompt_speech_feat, embedding=flow_embedding)`, i.e. the zero-shot flow decoder is reused unchanged ([`engine.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/engine.py)).
- Admitted limitations: no explicit style module (implicit style only), single-codebook capacity ceiling (hurts singing/BGM), lossy mel + separate vocoder ([PAPER §5](https://arxiv.org/html/2605.27258v2)).

### 1.7 Training stages and schedules

| Stage | Data | Notes |
|---|---|---|
| Pre-training | ~**200,000 h** zh+en from public sources | base zero-shot model ([PAPER §4.1](https://arxiv.org/html/2605.27258v2)) |
| Ablation runs | cleaned **60K-h** subset, **200K steps** each, 3 conditioning variants | fixed budget ([PAPER §4.6](https://arxiv.org/html/2605.27258v2)) |
| Emotion post-training | ~**2,200 h** = **1,000 h** high-quality + **1,200 h** augmented | open-source sets + internal annotations + model-augmented ([PAPER §4.1](https://arxiv.org/html/2605.27258v2)) |
| Paralinguistic SFT | ~**200 h** | pipeline outputs + internal collections ([PAPER §3.3.4](https://arxiv.org/html/2605.27258v2)) |
| Dialect fine-tuning | **16,000 h**, 14 dialects, from dialectal ASR corpora | 50/50 mixed-prompt sampling ([PAPER §3.3.5](https://arxiv.org/html/2605.27258v2)) |

**UNVERIFIED:** optimizer, LR, schedule, batch size, sequence length, epochs, hardware, GPU-hours, and freeze order
(the code exposes `freeze_encoder()` / `freeze_llm()` but no training script shows their use).

### 1.8 Evaluation numbers (Seed-TTS Eval) and control evals

Zero-shot, Table 1 ([PAPER](https://arxiv.org/html/2605.27258v2)); CER via Paraformer-zh, WER via Whisper, SIM = cosine of speaker embeddings.
`test-zh CER / SIM`: **PilotTTS 0.87 / 0.862**; Seed-TTS 1.12 / 0.796; F5-TTS 1.56 / 0.741; FireRedTTS-2 1.14 / 0.736;
CosyVoice-3-0.5B 1.16 / 0.780; VoxCPM-0.5B 0.93 / 0.772; Qwen3-TTS-25Hz-0.6B 1.18 / –; MiniMax-Speech 0.83 / –;
VibeVoice-1.5B 1.16 / 0.744.
`test-en WER / SIM`: **1.50 / 0.815**; 2.25 / 0.762; 1.83 / 0.647; 1.95 / 0.655; 2.02 / 0.718; 1.85 / 0.729; 1.64 / –; 1.65 / –; 3.04 / 0.689.

Ablation, Table 6 (same 200K steps, 60K-h subset) Full / w/o spk / w/o both: `test-zh` CER 1.130 / 1.022 / 1.412;
`test-en` WER 1.940 / 1.860 / 2.710; `test-hc` CER 7.830 / 8.866 / 10.623; SIM (zh, en, hc) Full 0.8626 / 0.8157 / 0.8470,
w/o both 0.8617 / 0.8027 / 0.8435. Reading: condition tokens carry content/prosody (~35% relative CER penalty on hard
cases when removed); the speaker embedding mainly buys SIM, largest gain on hard cases 0.8355 → 0.8470 ([PAPER §4.6](https://arxiv.org/html/2605.27258v2)).

Emotion control (human eval; 51 prompt speakers = 15 expressive characters + 36 ordinary, 18F/18M; success requires
*simultaneous* timbre preservation and recognizable emotion), Table 2: **PilotTTS avg primary 88.1%**, avg all-eleven 85.7%;
CosyVoice 3 83.8% / 82.5%; Fish-Speech S2 60.0% / 60.2%; VoxCPM 26.0% / 31.9%; IndexTTS covers only 7 of 11.
Per-category: Serious 93.2, Surprise 93.2, Happy 86.4, Disgust 65.5 (weakest), Psychology 98.2. Speaker SIM under
emotion control, Table 3: **0.8101 → 0.7329** (highest in both conditions, smallest drop); CosyVoice 3 0.7963 → 0.6940 ([PAPER §4.3](https://arxiv.org/html/2605.27258v2)).

Paralinguistic (21 prompts per category, human success %), Table 4: **PilotTTS LAUGH 97.6, COUGH 64.3, BREATH 81.0,
overall 85.1**, plus LAUGH_SPAN **94.6** and CRY **61.9** (unique to PilotTTS); CosyVoice 3 83.3 / 59.5 / 95.2 / 80.4;
Fish-Speech S2 54.8 / 64.3 / 83.3 / 64.3 ([PAPER §4.4](https://arxiv.org/html/2605.27258v2)).

Dialect, Table 5: Same-Dialect **91.80%**, Mandarin-to-Dialect **86.46%**, Cross-Dialect **85.38%**; a sample fails if
>10% of pronunciations are non-target-dialect ([PAPER §4.5](https://arxiv.org/html/2605.27258v2)).

### 1.9 Reproduction caveats

1. **SIM protocol differs from common practice** — a project collaborator states SIM is the cosine similarity of
   **FunASR pretrained speaker-recognition** embeddings, in reply to a user whose fine-tuned WavLM-large gave very
   different numbers and who never received the promised README update ([issue #5](https://github.com/AMAPVOICE/PilotTTS/issues/5)). Always pin the embedder.
2. **The Q-Former does not transfer prompt prosody** — user report "音色 clone 的比较像，但无法学到 prompt audio 的韵律",
   closed without maintainer explanation ([issue #9](https://github.com/AMAPVOICE/PilotTTS/issues/9)); consistent with PAPER limitation #1.
3. **Dialect looks disabled in the released instruct checkpoint** — the dialect demo is commented out with "马上开放方言模型"
   ([`demo.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/demo.py)); README/paper say 14 dialects, the
   [project page](https://amapvoice.github.io/PilotTTS/) says 13.
4. **Speed complaints exist** (4070 Ti SUPER user: "Toooo....slow") ([issue #8](https://github.com/AMAPVOICE/PilotTTS/issues/8)) — the 10-step CFM + HiFi-GAN dominates latency, not the 0.6B AR model.
5. `requirements.txt` pins `transformers<=4.52.4`, `torch<=2.5.1` — older than current stacks ([requirements.txt](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/requirements.txt)).

---

## 2. The data pipeline recipe

### 2.1 What the paper specifies, stage by stage ([PAPER §2](https://arxiv.org/html/2605.27258v2))

**Stage 1 — quality assessment & enhancement:** (1) standardize format/sampling rate; (2) segment with **SAD + speaker
change detection** (pyannote power-set diarization line of work); (3) estimate three attributes in parallel — **DNSMOS**
perceptual MOS, a **speech/non-speech classifier** (`iic/SenseVoiceSmall`), and **SNR**; (4) assign quality status —
"acoustically deficient" if **MOS ≤ 3.5**, or non-speech, or "insufficient SNR"; (5) run deficient segments through
**denoising/enhancement** (`resemble-ai/resemble-enhance`) instead of dropping them.

**Stage 2 — label annotation:** (1) **multi-ASR transcription** with cross-system consistency checking — Paraformer,
FireRedASR, Whisper, plus internal ASRs; (2) **overlapping-speech detection** with `pyannote/segmentation-3.0` and a
**forced aligner** for audio↔text timing; (3) **prosody annotation** built on Qwen3-Force-Alignment, hierarchical
prosodic structure; (4) **speaker tagging** with 3D-Speaker-Toolkit for speaker-consistent samples; (5) **spectral
rolloff analysis** to catch low-bandwidth (insufficient high-frequency) recordings.

**Stage 3 — quality filtering:** (1) **truncation detector** for incomplete starts/ends; (2) **synthetic-speech detector**
for TTS audio mixed into crawled data; (3) final module ANDing eight criteria — acoustic quality, speech validity,
transcription reliability, overlap, speaker consistency, truncation risk, synthesis likelihood, spectral quality;
(4) **rejected samples are retained** with tags/metadata so other quality tiers can be rebuilt; filter output = **~200,000 h** zh+en.

### 2.2 Tool inventory (every operator is a named public artifact)

pyannote (SAD/SCD, diarization; [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)) · DNSMOS ·
[SenseVoiceSmall](https://www.modelscope.cn/models/iic/SenseVoiceSmall) · [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) ·
[Paraformer](https://arxiv.org/html/2605.27258v2) · [FireRedASR](https://arxiv.org/abs/2501.14350) · [Whisper](https://arxiv.org/abs/2212.04356) ·
Qwen3-Force-Alignment / [Qwen3-ASR](https://arxiv.org/abs/2601.21337) · [3D-Speaker-Toolkit](https://github.com/modelscope/3D-Speaker) ·
[CosyVoice FSQ tokenizer](https://github.com/FunAudioLLM/CosyVoice) · [MaskGCT wav2vec2bert stats](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct).

### 2.3 Published vs missing thresholds

Published: **MOS ≤ 3.5** (DNSMOS); **~200K h** retained; **15 s** inference-prompt cap; **25 Hz / 6,561** tokens;
post-training budgets **2,200 / 200 / 16,000 h**.
**UNVERIFIED (must be re-invented):** SNR cutoff, min/max utterance duration, loudness target and filter, cross-ASR
agreement threshold, spectral-rolloff cutoff, speaker-consistency criterion, DNSMOS variant, identities of the
truncation and synthetic-speech detectors, per-source hours, and zh:en mixing ratio. Section 2 is a design narrative,
not a threshold table.

### 2.4 The pipeline is described, not released

The abstract and README claim the complete pipeline recipe and code are released, but the repo has **no pipeline or
training code** and there is no companion repo ([contents](https://api.github.com/repos/AMAPVOICE/PilotTTS/contents/), [org repos](https://api.github.com/users/AMAPVOICE/repos?per_page=100)).
[Issue #2](https://github.com/AMAPVOICE/PilotTTS/issues/2) asks exactly this; nine "+1"-style comments, a community
assessment "应该不会开放，这个才是重中之重" ("probably won't be open-sourced, this is the crown jewel"), and a
follow-up citing the paper's promise — **no maintainer reply** through 2026-07-11. Plan accordingly: Section 2 is a
blueprint we must fill in ourselves.

### 2.5 Supplemental: CosyVoice 3's fully specified pipeline (different system — attribute carefully)

CosyVoice 3 publishes real numbers, so borrow these directly ([CosyVoice 3 §3](https://arxiv.org/html/2505.17589v1)):
(1) diarization → VAD → audio event detection, producing **speaker-level segments < 30 s** (in-house modules, replaceable
by open source); (2) **MossFormer2** noise reduction, drop utterances starting/ending with incomplete words, trim
leading/trailing silence; (3) language ID with **Faster-Whisper Large-V3**, then transcribe with **Faster-Whisper
Large-V3 + NVIDIA NeMo Canary-1B + seamlessM4T-V2-large**, keeping only transcriptions whose **average pairwise WER < 15%**;
(4) forced alignment with **Montreal Forced Aligner**, **add comma when the gap ≥ 300 ms**, **remove pause punctuation
when the gap ≤ 50 ms**; (5) volume normalization `raw / max(raw) * 0.6`; (6) discard the **smallest 1%** and **largest 5%**
of speech-token/text-token length ratios; (7) tokenizer data **530K h** (breakdown in §1.5), instruction-following data
grown **1,500 h → 5,000 h** with **100+ style types** ([§2.5](https://arxiv.org/html/2505.17589v1)).

---

## 3. Emotion, paralinguistic and dialect labeling

### 3.1 Emotion (11 categories)

- Taxonomy: **7 primary** — happy, sad, angry, fear, contempt, serious, surprise — plus **4 extended** — concern,
  blue (melancholy), disgust, psychology (inner monologue) ([PAPER §3.3.3](https://arxiv.org/html/2605.27258v2)).
- Released tags add **neutral** and **unknown** (13 usable) ([README](https://huggingface.co/AmapVoice/PilotTTS/raw/main/README.md));
  the tokenizer carries open/close pairs `<|happy|>…<|/happy|>` for angry, blue, concern, disdain, disgust, fear, happy,
  neutral, psychology, sad, serious, surprise, unknown ([added_tokens.json](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/tokenizer/added_tokens.json)) — note README "disdain" = paper "contempt".
- Data: **~2,200 h** = **1,000 h high-quality + 1,200 h augmented** (open-source datasets, internal annotations,
  model-augmented) ([PAPER §4.1](https://arxiv.org/html/2605.27258v2)).
- Mechanism: training-free tag wrapping at inference, `text = f"<|{emotion}|>{text}<|/{emotion}|>"` ([`demo.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/demo.py)).
- **UNVERIFIED:** who/what produced the labels (human vs SER model), guidelines, per-category hours, inter-annotator
  agreement, how "psychology" is operationalized, and how the augmented 1,200 h were generated.

### 3.2 Paralinguistic categories

- Four targets: **LAUGH, BREATH, CRY, COUGH**, plus wrapped **LAUGH_SPAN** for laughter temporally coupled with speech
  ([PAPER §3.3.4](https://arxiv.org/html/2605.27258v2)). Two modes: **implicit** (infer from text; varied laughter types
  such as restrained chuckle / soft giggle / hearty laughter) and **explicit** (onomatopoeia tags).
- Data: **~200 h** from pipeline outputs plus internal collections, realized via SFT ([PAPER §3.3.4](https://arxiv.org/html/2605.27258v2)).
- Documented tags: `<|LAUGH|>`, `<|BREATH|>`, `<|COUGH|>`, `<|CRY|>`, `<|LAUGH_SPAN|>…<|/LAUGH_SPAN|>` ([README](https://huggingface.co/AmapVoice/PilotTTS/raw/main/README.md)).
  The tokenizer additionally defines `<|CRY_SPAN|>`, `<|SIGH|>`, `<|ELONG|>`, `<|EMPH|>`, `<|RPT|>` and vocal bursts
  `<|Uhm|> <|ah|> <|oh|> <|ei|> <|ay|> <|yi|> <|yo|> <|wa|> <|heh|> <|huh|> <|hng|> <|uh|>` ([added_tokens.json](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/tokenizer/added_tokens.json)) — which of these the released checkpoint actually emits is **UNVERIFIED**.
- COUGH is hardest for every system (~60%) due to acoustic variability and low prevalence ([PAPER §4.4](https://arxiv.org/html/2605.27258v2)).

### 3.3 Dialect

- Paper/README: **14 dialects**; project page: **13** — inconsistent (**UNVERIFIED**) ([PAPER §3.3.5](https://arxiv.org/html/2605.27258v2), [project page](https://amapvoice.github.io/PilotTTS/)).
- README tags (14): `zh-dongbei, zh-shandong, zh-henan, zh-shan1xi, zh-minnan, zh-gansu, zh-ningxia, zh-shanghai,
  zh-chongqing, zh-hubei, zh-hunan, zh-jiangxi, zh-guizhou, zh-yunnan`. The **released tokenizer has 21** — the 14 plus
  `zh-beijing, zh-cantonese, zh-cantonese-hk, zh-shan2xi, zh-sichuan, zh-suhang, zh-tianjin` ([added_tokens.json](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/tokenizer/added_tokens.json)); [issue #6](https://github.com/AMAPVOICE/PilotTTS/issues/6) asks about Cantonese/Sichuanese.
- Data: **16,000 h**, all from **dialectal ASR corpora** ([PAPER §4.1](https://arxiv.org/html/2605.27258v2)).
- Most copyable idea in the paper: because dialect audio is scarce, **synthesize three Mandarin utterances per dialect
  speaker** with the pre-trained model to build "dialect–Mandarin" parallel data, then fine-tune with **mixed-prompt
  sampling** — target always dialect, prompt drawn **50/50** from a Mandarin or dialect utterance of the same speaker —
  so the model extracts speaker identity from stylistically diverse prompts instead of copying prompt style ([PAPER §3.3.5](https://arxiv.org/html/2605.27258v2)).

### 3.4 Emoji labeling

**Nothing published.** No emoji vocabulary, emoji-labeling scheme, or emoji-annotated categories appear in the paper,
repo, tokenizer, or model card; control is entirely angle-bracket tags (`<|happy|>`, `<|LAUGH|>`, `<|zh-henan|>`).
Treat emoji control as **absent** from this recipe (UNVERIFIED at best); Parakeet would have to map emoji → this tag inventory itself.

### 3.5 Labeling gaps

No published information on annotation tooling, labeler count/qualification, inter-annotator agreement,
mixed-emotion or overlapping-event handling, span-boundary conventions for `LAUGH_SPAN`/`CRY_SPAN`, or the prosody
label schema from the Qwen3-Force-Alignment step; utterance-level vs span-level labeling is implied but **UNVERIFIED**.

---

## 4. Supplemental: CosyVoice 3 tokenizer and Q-Former usage in speech

### 4.1 CosyVoice 3 tokenizer (origin of PilotTTS's tokens)

- Discrete tokens via **Finite Scalar Quantization** inserted into the voice encoder of **MinMo** (large speech-understanding
  model), replacing CosyVoice 2's SenseVoice-Large insertion point; `H → Proj_down → ROUND[−K,K] → Proj_up` with
  straight-through gradients; index = base-(2K+1) positional sum; **25 Hz** ([CosyVoice 3 §2.1, Eq. 1–2](https://arxiv.org/html/2505.17589v1)).
- The 5-task supervision (ASR, LID, SER, AED, SA) on ~530K h is exactly what lets a *single* codebook carry paralinguistic
  information (emotion, pronunciation style) — i.e. the reason single-codebook is viable for expressive control at all
  ([CosyVoice 3 §2.1, Table 3](https://arxiv.org/html/2505.17589v1)).
- CosyVoice 3's own post-training (DiffRO differentiable reward over token logits with ASR/SER/MOS/AED multi-task rewards)
  is orthogonal; PilotTTS does not use RL ([CosyVoice 3 §2.2](https://arxiv.org/html/2505.17589v1)).
- CosyVoice 3's control vocabulary is richer than PilotTTS's: `[laughter]` / `[breath]` vocal bursts, `<strong>XXX</strong>`
  emphasis, natural-language instructions terminated by `<|endofprompt|>`, 100+ style types ([CosyVoice 3 §2.5, Table 1](https://arxiv.org/html/2505.17589v1)).

### 4.2 Q-Former / Perceiver lineage

- Q-Former originates in [BLIP-2](https://arxiv.org/abs/2301.12597) as learnable queries compressing a frozen encoder into a
  fixed token set; PilotTTS cites [Kim et al., EMNLP 2024 Findings](https://arxiv.org/html/2605.27258v2) for efficient Q-Former use.
- Implementation is a **Perceiver Resampler** whose header credits `lucidrains/naturalspeech2-pytorch` ([`perceiver.py`](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/pilot_voice/modules/perceiver.py)); the Perceiver Resampler pattern comes from [Flamingo](https://arxiv.org/abs/2204.14198). So "Q-Former" here = latent-query cross-attention bottleneck (depth 2, 32 latents, RMSNorm), *not* BLIP-2's full dual-encoder Q-Former.
- Practical consequence: a 32-token, 2-layer cross-attention adapter over frozen w2v-BERT-2.0 features is a few million
  params and, per the ablation, carries most of the content/prosody conditioning signal ([PAPER §4.6](https://arxiv.org/html/2605.27258v2)).

---

## Actionable takeaways for Parakeet

### Top 5 things to copy

1. **Dual-pathway conditioning + cross-sample paired training.** 32 latent condition tokens (frozen w2v-BERT-2.0 layer-17
   features → MaskGCT-normalized → 6-block conformer 1024→512 → 2-layer Perceiver Resampler) **plus** a separate frozen
   CAMPPlus speaker embedding; the reference for `s`/`c` is always a **different utterance of the same speaker**. This is
   the mechanism behind the headline SIM 0.862 / 0.815 at only 200K h ([PAPER §3.3.2, §4.6](https://arxiv.org/html/2605.27258v2)).
2. **Single-codebook 25 Hz FSQ semantic tokenizer** (6,561 codes = D=8, K=1) instead of RVQ: no hierarchical prediction
   head, no tokenizer training if CosyVoice 3's is reusable — accepting the admitted capacity ceiling for singing/BGM ([PAPER §3.2, §5](https://arxiv.org/html/2605.27258v2)).
3. **Synthetic-parallel-data + mixed-prompt sampling for scarce capabilities.** Generate **3** teacher utterances per
   speaker per target style/dialect, then SFT with a **50/50** mix of easy/target-style prompts from the same speaker.
   This is precisely how Parakeet should use API-only teachers (MiniMax) and English-centric ones (Orpheus) as *data
   generators* rather than as architectures ([PAPER §3.3.5](https://arxiv.org/html/2605.27258v2)).
4. **Small, targeted capability SFT sets with a tag-based control surface** (~2.2K h emotion, ~200 h paralinguistic, large
   dialect set) plus a tokenizer extended with emotion, dialect, paralanguage, and **pinyin initial/final/tone** tokens —
   pronunciation and style control without a phonemizer ([PAPER §4.1](https://arxiv.org/html/2605.27258v2), [added_tokens.json](https://raw.githubusercontent.com/AMAPVOICE/PilotTTS/master/tokenizer/added_tokens.json)).
5. **Metadata-preserving filtering built only from public operators** — never delete rejected audio; keep quality tags and
   annotations so tiers are a query, not a rerun ([PAPER §2.3](https://arxiv.org/html/2605.27258v2)).

### Ordered data-pipeline spec to implement

`[published]` = number comes from PilotTTS §2 or CosyVoice 3 §3; *[PROPOSED]* = my default, must be calibrated.

**A. Ingest and standardization** — (1) decode to mono; keep a 16 kHz rendition for all analysis/models and a target-rate
(24 kHz) rendition for training targets *[PROPOSED]*. (2) EBU R128/BS.1770 to −23 LUFS for training audio, and a separate
CAMPPlus branch normalized exactly as PilotTTS does: `gain_db = -24 - 10*log10(mean(wav^2))`, clip ±1 `[published, PilotTTS code]`.
(3) diarize → VAD → AED; cut **speaker-level segments < 30 s** `[published, CV3]`. (4) drop segments truncated mid-word at
either edge, trim silence `[published, CV3]`; *[PROPOSED: keep training clips 2–20 s, prompt pool 3–10 s]*.

**B. Quality assessment (parallel per segment)** — (5) DNSMOS; **MOS ≤ 3.5 → deficient/enhance** `[published, PilotTTS]`.
(6) speech/non-speech via SenseVoiceSmall; reject music/noise/events `[published, PilotTTS]`. (7) SNR — *[PROPOSED: reject
< 10 dB, enhance if 10–15 dB]* (paper cutoff unpublished). (8) spectral rolloff — *[PROPOSED: reject if 95% rolloff < 8 kHz]*
(cutoff unpublished). (9) persist a status tag `ok | enhanced | deficient` per segment.

**C. Enhancement (deficient only)** — (10) run resemble-enhance (or MossFormer2 `[published, CV3]`); re-tag `enhanced` if
fixed, else `rejected` but retained `[published, PilotTTS]`.

**D. Annotation** — (11) transcribe with **3 independent ASRs** (Paraformer-zh, FireRedASR, Whisper) `[published, PilotTTS]`.
(12) keep transcriptions with **average pairwise WER < 15%** `[published threshold, CV3]`; *[PROPOSED: majority text +
`low_conf` tag when no consensus]*. (13) overlap detection with `pyannote/segmentation-3.0`; tag overlaps and exclude them
from single-speaker TTS training `[published, PilotTTS]`. (14) forced alignment (Qwen3-Force-Alignment or MFA) → word/phone
timings; punctuation repair **add comma if gap ≥ 300 ms, remove pause punctuation if gap ≤ 50 ms** `[published, CV3]`.
(15) prosody annotation from the alignment output, stored as side-car labels `[published, PilotTTS]`. (16) speaker tagging with
3D-Speaker-Toolkit; *[PROPOSED: require within-cluster cosine ≥ 0.7 and ≥ 2 utterances/speaker for paired training]*.
(17) compute FSQ speech tokens at 25 Hz and text token counts `[published]`.

**E. Filtering (logical AND gates)** — (18) truncation detector `[published stage, unpublished tool]`. (19) synthetic-speech
detector `[published stage, unpublished tool]` — critical for us, since our own teachers generate audio. (20) length-ratio
gate: drop **smallest 1%** and **largest 5%** of speech-token/text-token ratios `[published, CV3]`. (21) final AND over the
eight criteria listed in §2.1 `[published, PilotTTS]`. (22) write everything (including rejects) to a queryable manifest
(parquet) with all tags `[published design, PilotTTS]`.

**F. Capability labels via teachers** — (23) use Orpheus emote tags `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan>
<yawn> <gasp>` and its `{voice}:` prompt format to *generate* labeled emotion/paralanguage data, and MiniMax
speech-2.8-turbo (voice cloning, emotion control, 40+ languages) for synthetic pairs ([Orpheus](https://raw.githubusercontent.com/canopyai/Orpheus-TTS/main/README.md), [MiniMax](https://replicate.com/minimax/speech-2.8-turbo)); map them onto our own 7+4 emotion and 4 paralinguistic classes.
(24) dialect/style pairs: **3** teacher utterances per speaker per target style, then 50/50 mixed-prompt SFT `[published, PilotTTS]`.
(25) **keep synthetic audio out of the bulk pre-training mix**: Orpheus's pretraining notes warn synthetic data "lacks
diversity and map[s] to the same set of tokens when tokenised (i.e. lead to poor codebook utilisation)" ([Orpheus](https://raw.githubusercontent.com/canopyai/Orpheus-TTS/main/README.md)) —
mirror PilotTTS's own split (1,000 h real + 1,200 h augmented emotion, 200 h paralinguistic, 16K h dialect) instead of
flooding pre-training.

### What not to copy

- The 10-step CFM + separate HiFi-GAN decode path is the latency hotspot (users complain) and the paper's limitation #3
  admits lossy mel reconstruction ([PAPER §5](https://arxiv.org/html/2605.27258v2), [issue #8](https://github.com/AMAPVOICE/PilotTTS/issues/8)); Parakeet should cut steps and/or decode latents/waveform directly.
- Implicit-only style capture: the paper's limitation #1 plus the community finding that the Q-Former misses prompt prosody
  means 32 latents are **not** a style encoder ([issue #9](https://github.com/AMAPVOICE/PilotTTS/issues/9)).
- Never benchmark SIM against 0.862 without matching the embedder — they used FunASR speaker embeddings ([issue #5](https://github.com/AMAPVOICE/PilotTTS/issues/5)).

### Open questions (all currently UNVERIFIED)

1. License compatibility of reusing CosyVoice 3's FSQ tokenizer / SenseVoice / CAMPPlus artifacts in a commercial distillation.
2. Real `flow.pt` parameter count and step count versus "300M DiT, 10 steps".
3. The unpublished filter thresholds (SNR, duration, loudness, rolloff, ASR agreement) — we must define and document ours, ideally A/B'd against a Seed-TTS-Eval-style probe set.
4. Whether span tags (`LAUGH_SPAN`, `CRY_SPAN`) need span-level supervision or emerge from utterance-level SFT.
5. Whether our tokenizer choice still allows consistent tokenization of both teachers' audio (teacher audio must be re-tokenized under our tokenizer, or the teachers must feed a shared semantic space).
