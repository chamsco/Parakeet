# Teacher research: Orpheus TTS, MiniMax speech-2.8-turbo, and the Paradee distillation recipe

Scope: facts needed to distill a "Parakeet" student from a **mix** of (a) Canopy Labs' Orpheus TTS and
(b) MiniMax `speech-2.8-turbo`, reusing the methodology of *Paradee* ([arXiv:2610.06817](https://arxiv.org/abs/2610.06817)).
All page content was read as data. Unconfirmed items are marked **UNVERIFIED**.

---

## Task 1 — Orpheus / Canopy Labs: what was actually released

| Artifact | Repo ID | Notes |
|---|---|---|
| Finetuned prod (English) | `canopylabs/orpheus-tts-0.1-finetune-prod` → [`canopylabs/orpheus-3b-0.1-ft`](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) | everyday TTS, 8 English voices |
| Pretrained (English) | [`canopylabs/orpheus-3b-0.1-pretrained`](https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained) | "trained on 100k+ hours of English speech data" |
| Multilingual family | [collection](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release) | 14 models = 7 pretrain/finetune pairs; verified `3b-zh-{pretrain,ft}`, `3b-hi-{pretrain,ft}`, `3b-ko-ft`, `3b-es_it-ft` |

The repo [README](https://github.com/canopyai/Orpheus-TTS) (raw: [jsDelivr mirror](https://cdn.jsdelivr.net/gh/canopyai/orpheus-tts@main/README.md)) releases only two English
models plus the multilingual preview, and its checklist explicitly leaves **`1b, 400m, 150m` unchecked** — **no sub-3B
Orpheus exists publicly.** HF labels two multilingual cards `4B`, but that is the file-size-derived number, not a parameter count.

**Backbone.** The [HF API](https://huggingface.co/api/models/canopylabs/orpheus-3b-0.1-ft) reports
`architectures: ["LlamaForCausalLM"]`, `base_model: ["meta-llama/Llama-3.2-3B-Instruct", "canopylabs/orpheus-3b-0.1-pretrained"]`,
`license: apache-2.0`, `gated: "auto"`, 30,996 downloads, `safetensors.parameters.F32 = 3,782,986,752`,
`usedStorage = 56.7 GB`. The ungated mirror [`unsloth/orpheus-3b-0.1-pretrained/config.json`](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained/raw/main/config.json)
confirms Llama-3.2-3B geometry: `hidden_size 3072`, `28` layers, `24` heads, `8` KV heads,
`intermediate_size 8192`, `head_dim 128`, `rope_theta 500000`, `rope_scaling {factor 32.0, original_max_position_embeddings 8192}`,
`max_position_embeddings 131072`, `torch_dtype bfloat16`, **`vocab_size 156939`**.
So Orpheus = **Llama-3.2-3B-Instruct geometry, autoregressive next-token prediction over SNAC audio tokens.**
The 3.78B tensor count is explained by the enlarged vocabulary plus an untied head (156,939 × 3,072 ≈ 482M
per matrix vs Llama's 394M): ≈ 3.21B − 394M + 964M ≈ 3.78B — **arithmetic inference; effective size ≈ 3.2–3.3B (UNVERIFIED)**.

### SNAC codec and the exact token stream

Decoding uses [SNAC `hubertsiuzdak/snac_24khz`](https://huggingface.co/hubertsiuzdak/snac_24khz):
24 kHz mono, **19.8 M params, 0.98 kbps, 3 RVQ levels at 12 / 23 / 47 Hz**, code lengths `[12, 24, 48]` per second.
From the standalone C++ port [CrispASR `orpheus_snac.h`](https://github.com/CrispStrobe/CrispASR/blob/main/src/orpheus_snac.h):
`n_codebooks = 3`, `hop_length = 512`, VQ strides `[4, 2, 1]`, shape relation `n0 = T_super, n1 = 2·T_super, n2 = 4·T_super`,
output samples `= n2 · 512 = 2048 · T_super`. **Rate math:** one super-frame = 1/12 s = 83.33 ms = 2048 samples,
and **every 7 emitted LM tokens form one super-frame** → **≈ 84 audio tokens per second of speech**;
a 2,048-token context caps out near 2,048/7/12 ≈ **24 s** of audio before text tokens.

De-interleaving is explicit in the official decoder ([`orpheus_tts_pypi/orpheus_tts/decoder.py`](https://raw.githubusercontent.com/canopyai/Orpheus-TTS/main/orpheus_tts_pypi/orpheus_tts/decoder.py)):
for group `i = 7j`, `codes_0 += frame[i]`, `codes_1 += frame[i+1], frame[i+4]`, `codes_2 += frame[i+2], frame[i+3], frame[i+5], frame[i+6]`;
all codes must lie in `[0, 4096]`. Tokenizer layout, from the [Sunbird Orpheus-3B card](https://huggingface.co/Sunbird/orpheus-3b-tts-multilingual): `end_of_text` 128009,
`START_OF_SPEECH` 128257, `END_OF_SPEECH` (eos) 128258, `START_OF_HUMAN`/`END_OF_HUMAN` 128259/128260, `START_OF_AI`/`END_OF_AI` 128261/128262,
`PAD_TOKEN` 128263, and the audio codebooks at `128266 + N·4096`, `N ∈ {0..6}` — so the last audio id is `128266 + 7·4096 − 1 = 156938`, exactly one below `vocab_size 156939`.
**Streaming granularity:** once `count > 27` and `count % 7 == 0`, the decoder takes `buffer[-28:]`, decodes whole 7-token super-frames, and slices
`audio_hat[:, :, 2048:4096]` — **2,048 samples (85.3 ms) emitted per step after a 28-token warm-up**.

### Voices, tags, prompting, sampling

- English voices, "in order of conversational realism": `tara, leah, jess, leo, dan, mia, zac, zoe`; prompt format `{name}: {text}` ([README](https://cdn.jsdelivr.net/gh/canopyai/orpheus-tts@main/README.md)).
  Per-language voice lists were linked to `canopylabs.ai/releases/orpheus_can_speak_any_language#info`, which now serves marketing copy only — **no longer published (UNVERIFIED)**.
- Emotion tags: `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>`. `repetition_penalty >= 1.1` is stated as **required** for stable generation; raising `repetition_penalty`/`temperature` makes it speak faster.
- Context: the streaming example passes `max_model_len=2048`, pretraining used **8192**, and the config advertises `max_position_embeddings 131072` with RoPE factor 32 — **treat 2048 as the practical deployed context (UNVERIFIED for multilingual models).**
- Latency: "**~200 ms streaming latency** … reducible to **~100 ms** with input streaming"; serving is vLLM-based (`orpheus-speech`, `vllm==0.7.3` pin workaround), with Baseten the preferred partner at fp8/fp16 ([README](https://cdn.jsdelivr.net/gh/canopyai/orpheus-tts@main/README.md)).
  Community figures ([orpheus-streaming](https://github.com/taresh18/orpheus-streaming) TTFB ≈ 160 ms; [issue #221](https://github.com/canopyai/Orpheus-TTS/issues/221) "sub 100ms TTFT") are **not vendor-verified**. **SNAC decode on CPU costs ~50–150 ms per utterance**; bf16 serving needs ≥ 14 GB VRAM (24 GB for vLLM at `max_model_len=4096`) ([Sunbird card](https://huggingface.co/Sunbird/orpheus-3b-tts-multilingual)).

### Published training material — and the synthetic-data warning

Official finetune path ([README](https://cdn.jsdelivr.net/gh/canopyai/orpheus-tts@main/README.md), [config.yaml](https://raw.githubusercontent.com/canopyai/Orpheus-TTS/main/finetune/config.yaml)): HF dataset
in `canopylabs/zac-sample-dataset` format → preprocessing notebook → `finetune/train.py`, with `epochs: 1`, `batch_size: 1`, `num_processes: 1`, `learning_rate: 5.0e-5`,
`save_steps: 5000`, `pad_token: 128263`, base `canopylabs/orpheus-tts-0.1-pretrained`. "You should start to see high quality results after ~50 examples but for best results,
aim for 300 examples/speaker." Pretraining: 100k+ hours, sequence length 8192, `input_ids` chained, with a text dataset mixed in (format in [issue #37](https://github.com/canopyai/Orpheus-TTS/issues/37)).

> **Load-bearing for Parakeet** — the Orpheus authors advise against training on synthetic data: "I recommend not
> using synthetic data for training as it produces worse results when you try to finetune specific voices, probably
> because synthetic voices lack diversity and map to the same set of tokens when tokenised (i.e. lead to poor
> codebook utilisation)." — [README, Pretrain Model](https://cdn.jsdelivr.net/gh/canopyai/orpheus-tts@main/README.md)

A concrete de-facto community recipe ([Sunbird card](https://huggingface.co/Sunbird/orpheus-3b-tts-multilingual)): LoRA `r=64, α=64, dropout=0` on `q,k,v,o,gate,up,down_proj`;
`adamw_8bit`, wd `0.001`, linear schedule, lr `2e-4`, warmup `5`; per-device batch 1 × grad-accum 4; 3 epochs; `max_seq_length 4096`; bf16; one RTX 4090; merge to 16-bit.
Data prep encodes each clip with `snac_24khz` into 7 codes/frame with per-position offsets, drops consecutive duplicate frames, and builds
`[SOH] + text + [EOT] + [EOH] + [SOA] + [SOS] + codes + [EOS] + [EOA]`.

---

## Task 1b — MiniMax `speech-2.8-turbo` (teacher #2)

- **API**: `POST https://api.minimax.io/v1/t2a_v2`; model enum includes `speech-2.8-hd`, **`speech-2.8-turbo`**, `speech-2.6-*`, `speech-02-*`, `speech-01-*` ([API reference](https://platform.minimax.io/docs/api-reference/speech-t2a-http)).
- **Knobs**: `text` < 10,000 chars (streaming recommended > 3,000); `stream`; `voice_setting{voice_id, speed 0.5–2, vol 0–10, pitch −12..12}`;
  `audio_setting{sample_rate, bitrate, format, channel}`; `language_boost` (40 values incl. `auto`); `pronunciation_dict.tone`
  (Pinyin with tone number, IPA, or Jyutping); `voice_modify{pitch,intensity,timbre,sound_effects}`;
  `subtitle_enable` + `subtitle_type ∈ {sentence, word, word_streaming}`.
- **Audio**: sample rates `8000, 16000, 22050, 24000, 32000, 44100`; formats `mp3, wav, flac` (streaming: **mp3 only**); `output_format ∈ {url, hex}` (default `hex`, URL valid 24 h); doc example returns `audio_sample_rate: 32000`, `bitrate: 128000`, `channel: 1` ([input schema](https://developers.cloudflare.com/ai/models/minimax/speech-2.8-turbo/schema-input.json)).
- **Interjection tags (2.8 only)**: `(laughs) (chuckle) (coughs) (clear-throat) (groans) (breath) (pant) (inhale) (exhale) (gasps) (sniffs) (sighs) (snorts) (burps) (lip-smacking) (humming) (hissing) (emm) (sneezes)`; pause markers `<#x#>` with `x ∈ [0.01, 99.99]` s.
- **Emotion**: `happy, sad, angry, fearful, disgusted, surprised, calm, fluent`; voice cloning from a 10-second sample ([launch post](https://www.minimax.io/news/minimax-speech-28)).
- **Pricing**: **$60 / M characters** (turbo), $100 / M (hd), same for async T2A; voice cloning $1.50/voice, voice design $3/voice ([pricing](https://platform.minimax.io/docs/pricing/overview)). Cloudflare Workers AI lists the same model at **$0.00006/character** with **zero data retention** ([model page](https://developers.cloudflare.com/ai/models/minimax/speech-2.8-turbo/)).
- **Architecture, parameter count, tokenizer, logits, RTF, MOS/UTMOS/WER: not published anywhere I could find → UNVERIFIED.** No weights, no codec, no token stream. **The only distillation signal from MiniMax is audio** (plus word/sentence timestamps) — versus Orpheus, where exact discrete codes and per-frame intermediates are available.

---

## Task 2 — Paradee: the recipe in implementation detail

Sources: [abstract](https://arxiv.org/abs/2610.06817), [full HTML](https://arxiv.org/html/2610.06817v1), [alphaXiv](https://www.alphaxiv.org/abs/2610.06817),
[repo README](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/README.md),
[training/README](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/training/README.md), plus the scripts themselves.

**Repo layout** ([trees API](https://api.github.com/repos/sahilmahendrakar/paradee/git/trees/main?recursive=1)): `paper/Paradee-Mahendrakar-2026.pdf`;
`paradee/{__init__,__main__,tts}.py`; `web/misaki.js`; `LICENSE` (Apache-2.0);
`training/data/wikitext_sents.json` (1.31 MB); `training/scripts/{common,gen_teacher,regen_audio,student,student_decoder,discriminators,train_text,train_decoder,phase_lock,quantize_student,export_paradee,eval_full,rtf,utmos,wer}.py`.
There are **no config files — every hyperparameter is a CLI default**; runs resume from `checkpoints/<run>/last.pt`.

### Stage 0 — teacher corpus ("save every intermediate")

[`gen_teacher.py`](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/training/scripts/gen_teacher.py):
`KPipeline(lang_code="a", device="mps", repo_id="hexgrad/Kokoro-82M")`, voice `af_heart`, over **12,000 WikiText-103
sentences → 23.9 h of audio**, saved as **shards of 500** (6.9 GB total); rows > 400 phonemes are dropped; the style
vector is indexed by phoneme count (`ref[T-3]`), `r[:, :128]` → decoder, `r[:, 128:]` → prosody predictor.
Saved per row: `ids` (int64 `[T]`, phoneme ids with a `0` pad at each end), `dur` (float `[T]`, duration **before**
rounding) and `pred_dur` (int64 `[T]`), `d` (fp16 prosody-predictor hidden), `t_en` (**512-ch phoneme features**,
fp16 `[512, T]`), `F0` and `N` (pitch and energy at **2× frame rate**, `[2F]`), and `audio` (int16, 24 kHz).

[`regen_audio.py`](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/training/README.md) then re-renders every waveform **on CPU
with a fixed seed per sentence**, because "Kokoro's output differs slightly between CPU and GPU, and the decoder is
trained to match the CPU version"; `train_decoder.py` re-derives the teacher's harmonic excitation with the same seed.
**Frame grid:** Kokoro's decoder frame is **600 samples @ 24 kHz = 40 Hz**; F0/energy are at **2× = 80 Hz**; the phase
filter uses STFT `n_fft 1024, hop 256`.

### Stage 1 — text side (4.23 M params)

Architecture ([paper §4.3](https://arxiv.org/html/2610.06817v1)): ALBERT with **256 hidden, 6 layers, 4 heads**, plus a prosody
predictor and text encoder at **192 channels, 3 layers**, and a **2-layer MLP 192 → 512** to reach the decoder's width.
"Since the voice is fixed, we remove the style input from both halves of the student and replace it with a **learned
constant** vector" — so the student must also learn Kokoro's utterance-length dependence of the style vector. Variants:
linear projection (4.0 M) and wider (7.1 M).

Training ([`train_text.py`](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/training/scripts/train_text.py)): AdamW, `lr 5e-4` (paper value; script default `5e-4`),
batch 32, **8,000 steps** (README command `train_text.py s+mlp --steps 8000`; the script's own default is 20,000),
schedule `min(1, s/500) · ½(1 + cos(π·min(s,steps)/steps))` (500-step warmup then cosine), grad-clip 1.0. Losses are
**masked L1 against the teacher's saved values with teacher-forced durations**:

| loss | definition |
|---|---|
| `l_dur` | `L1(log(pred_dur), log(teacher_dur))` over text positions |
| `l_f0` | `L1(pred_F0, teacher_F0) / 100` over frame positions |
| `l_n` | `L1(pred_N, teacher_N)` over frame positions |
| `l_asr` | `L1(pred_features, teacher_t_en) / 512 × asr_weight (1.0)` |

Optional `--through-decoder` adds a log-mel L1 through the **frozen teacher decoder** (48-frame segments, `td_bs 8`,
`td_weight 1.0`). That variant is the ablation loser: it matched the teacher with the teacher decoder but "fell to
**3.78** with the student decoder, against **4.39** for the directly trained one" ([paper §1](https://arxiv.org/html/2610.06817v1)).

### Stage 2 — decoder (3.85 M params)

Kokoro's decoder layout at **256 AdaIN channels (teacher 1,024)** and **generator `upsample_initial_channel` 128
(teacher 512)**, keeping Kokoro's upsampling factors, resblock kernels `[3, 7, 11]`, excitation module and iSTFT head;
`style_dim 16`, learned constant style. **3.36 GFLOP per second of audio (15× less than the teacher's decoder)**;
generator alone 1.19 M params ([`student_decoder.py`](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/training/scripts/student_decoder.py)).
The code also defines presets `B–F` (thinner), `W/W2`, and `L=(384,192,[3,7,11])` — "7.86M: capacity probe, 1.5× wider
everywhere than A".

Shared trainer settings ([`train_decoder.py`](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/training/scripts/train_decoder.py)): AdamW `betas=(0.8, 0.99)`,
`weight_decay=0.01`, grad-clip 5.0, 64-frame (~1.6 s) random segments, cosine schedule with 1,000-step warmup.

**Stage 2a — spectral only**: 50,000 steps, batch 16, lr 2e-4.
```
loss = logmel_L1 + MRSTFT
logmel : torchaudio MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80), L1 on log(mel)
MRSTFT : for n_fft in (512, 1024, 2048), hop = n_fft/4, Hann window:
         SC = ||T| − |Y|| / ||T||  +  L1(log|Y|, log|T|);  averaged over the 3 resolutions
```
**Stage 2b — adversarial**: +5,000 steps at `--mel-weight 10` (batch 8), then +5,000 at `--mel-weight 3` (chained via
`--init-from`). Code defaults: `--gan-weight 1.0`, `--fm-weight 2.0`, `--mel-weight 45.0` (the losing default),
`--d-mult 1`, `--d-lr-mult 1.0`.
```
total = mel_weight · (logmel_L1 + MRSTFT) + gan_weight · adv + fm_weight · fm
```
Discriminators ([`discriminators.py`](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/training/scripts/discriminators.py)): **both** a HiFi-GAN-style
**multi-period discriminator** over `periods=(2, 3, 5, 7, 11)` and a **multi-resolution spectrogram discriminator**
(`ResD`) over `n_ffts=(512, 1024, 2048)` — 3.2 M params together. `PeriodD` channels `(32,64,128,256)`, kernels
`(5,1)`/`(3,1)`, reflection padding; `ResD` channels `(32,64,128,128)`, `(3,9)` convs stride `(1,2)`, spectrogram
computed internally at `n_fft/hop = n/4`. **Least-squares GAN**: `d_loss = Σ[mean((r−1)²) + mean(f²)]`;
`g_loss adv = Σ mean((f−1)²)`; `fm = Σ mean_over_layers L1(feat_fake, feat_real.detach())`.

Documented-but-off-by-default student options: `--wave-weight`, `--cstft-weight`, `--head-weight` (distil the teacher
iSTFT head's log-magnitude branch `[:, :11]`), `--phase-weight` (L1 on `sin(x)` of the phase branch `[:, 11:]`),
`--hf-weight` (band-limited 1.5–8 kHz STFT, window 1024 / hop 128), `--src-bands` (per-harmonic (cos, sin) phase
channels, zero-init), `--full-phase` (unbounded zero-init additive phase). All `--head-*`/`--phase-*` need
`--matched-source` (they compare against the teacher's own excitation); the generator runs `m_source` under `no_grad`,
so the harmonic mixing layer never trains (extra harmonics fixed at −0.2 = the teacher's 9-column mean).

### Assembly, phase filter, quantization, export

**Assembly** ([paper §4.4](https://arxiv.org/html/2610.06817v1)): connect student text side → student decoder, **no joint training**; the text side predicts
durations, `alignment()` stretches phoneme features to one per audio frame, then it predicts frame-level F0/energy.

**Phase-locking filter** ([`phase_lock.py`](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/training/scripts/phase_lock.py); [paper §4.5, §7](https://arxiv.org/html/2610.06817v1)) — **parameter-free**, applied
*after* synthesis, band **2–8 kHz**, motivated by "the phase of voiced speech between 2 and 8 kHz, just above where
Kokoro's harmonic excitation stops":
```python
n, hop, SR = 1024, 256, 24000;  W = hann_window(n);  FR = arange(n//2+1) * SR / n
BAND = (FR >= 2000) & (FR < 8000)
S = stft(student);  E = stft(harmonic_excitation)          # E from a generator.m_source forward hook
v  = F0[frame] > 60                                        # voiced decision (F0 at 300-sample hops)
rel = S * E.conj() / E.abs().clamp_min(1e-6)               # student phase relative to its own source
ph  = where(BAND & v, angle(E) + angle(smooth(rel, L)), angle(S))
out = istft(polar(S.abs(), ph), n, hop, window=W, length=k)
```
`smooth` is a Hann-weighted moving average over `L` frames (`conv1d`, `padding=L//2`) or `"seg"` (one mean offset per
contiguous voiced run); the source note records "9 frames slightly better; 17 frames as good as teacher phase".
Cost: **~5 % of run time**. Effect: full-student UTMOS **4.39 → 4.41** ([paper §6.1](https://arxiv.org/html/2610.06817v1)).

**int8 quantization** ([`quantize_student.py`](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/training/scripts/quantize_student.py); [paper §4.6](https://arxiv.org/html/2610.06817v1)) — **weight-only, per output channel, fp16 scale**:
```python
if p.dim() >= 2 and name.split(".")[-1].startswith("weight"):
    p.data = fq(p.data, bits)                       # per-output-channel int8
    nbytes += p.numel()*bits//8 + p.shape[0]*2      # + one fp16 scale per output channel
else:
    nbytes += p.numel()*2                           # everything else stays fp16
```
(`remove_weight_norm` and `remove_parametrizations(..., leave_parametrized=True)` are applied first.) Measured
**8.45 MB int8 vs 32.5 MB fp32; UTMOS 4.41 at both precisions**. Released ONNX **9.0 MB**; teacher 325 MB fp32 ONNX /
82 MB int8. **4-bit** dropped the decoder to **UTMOS 3.98** in a 3-sentence test — "A small model has no redundant
channels left to absorb the rounding"; and "The common int8 download of Kokoro sounds worse mainly because it also
quantizes activations" ([paper §6.1, App. B](https://arxiv.org/html/2610.06817v1)).

**Export / serving surface** ([training/README](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/training/README.md), [README](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/README.md)):
one ONNX graph, inputs `input_ids` int64 `[1, T]` (Kokoro vocabulary, `0` pad at each end, **max 512 total**) and
`speed` float32 `[1]`, output `waveform` float32 `[1, samples]` at 24 kHz, phase filter baked in; **17.8× real time**
in onnxruntime on one thread. Gotcha worth copying: Kokoro's `kokoro.custom_stft.CustomSTFT` does **not** match
`torch.stft` and "costs about 0.5 UTMOS"; the export uses its own exact `ConvSTFT`.

### Results and ablations (200 held-out sentences, Apple M4 Pro, 1 CPU thread)

| Model | Params | Size | Speed | UTMOS | WER |
|---|---:|---:|---:|---:|---:|
| Kokoro-82M teacher (`af_heart`) | 81.8 M | 325 MB | 7.6× | 4.52 ± 0.003 | 5.7% |
| **Paradee** (`af_heart`) | **8.07 M** | **8.45 MB** | **25.0×** | **4.41 ± 0.01** | **5.7%** |
| Paradee, released ONNX | 8.07 M | 9.0 MB | 17.8× | 4.41 | 6.0% |
| Kokoro-7M-Distill (`af_msa`) | 7.48 M | 30.1 MB | 35.5× | 4.18 | 7.4% |
| Piper en_US-lessac-medium | 15.7 M | 63.2 MB | 15.4× | 4.36 | 8.8% |
| KittenTTS nano 0.8 (Bella) | 14.0 M | 56.8 MB | 10.7× | 4.01 | 5.5% |

Also: teacher decoder alone **20.5×** real time on onnxruntime; **speaker similarity to teacher 0.958** (a different
Kokoro voice, `af_bella`, scores 0.842); compute 55 → **3.4 GFLOP/s** (15×); "3.3× faster than the teacher in this test, not 25×" ([alphaXiv](https://www.alphaxiv.org/abs/2610.06817)).

**Loss-balance ablation (the headline finding).** Spectral losses only → **UTMOS 2.98** (full student 3.00), with an
audible "robotic second voice"; adversarial training at the default spectral weight **45** → **3.02**; weight
**10 → 4.29 (dec) / 4.32 (full)**; weight **3 → 4.37 (dec) / 4.39 (full)**; + phase filter → **4.41**. "Larger or
wider decoders did not solve the residual buzz… the balance of losses matters more here than decoder size" ([paper §7](https://arxiv.org/html/2610.06817v1)).

**Half-wise ablation.** A **4.2 M text side scored 4.36 through the teacher's decoder** and a **7.1 M one did no
better**; the **3.9 M decoder scored 4.37 from the teacher's text side**; joined, the two halves scored **4.39** —
"their errors do not add up" ([paper §1](https://arxiv.org/html/2610.06817v1)).

**Compute, data, metrics.** All training on one MacBook Pro (M4 Pro, 24 GB), **MPS** backend; "run at most two trainers
at once on a 24 GB machine" ([training/README](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/training/README.md)).
**Per-stage wall-clock is not reported anywhere → UNVERIFIED** (the scripts print `s/step`, so it is re-measurable).
Evaluation ([paper §5](https://arxiv.org/html/2610.06817v1)): **UTMOS `utmos22_strong`** as the primary metric; WER from **Whisper (base)**; **DTW log-mel L1**
for teacher-forced comparisons (saturates for free-running students — "a 3% speed change alone scores 0.36"; teacher
seed-to-seed floor 0.09); speaker similarity from **Resemblyzer**. Held-out = first 200 rows of shard 0, "fixed forever".
Licensing: WikiText-103 sentences **CC BY-SA 3.0**; code and model **Apache-2.0** ("the same as Kokoro")
([README](https://raw.githubusercontent.com/sahilmahendrakar/paradee/main/README.md)). **No multi-voice or multi-teacher
extension is attempted or discussed** — the stated boundary is "one English voice only".

---

## Task 3 — Licenses, gating, and distillation/ToS constraints

### Canopy Labs / Orpheus

| Component | License | Evidence |
|---|---|---|
| `canopyai/Orpheus-TTS` code | **Apache-2.0** | full text at [repo LICENSE](https://raw.githubusercontent.com/canopyai/Orpheus-TTS/main/LICENSE) |
| `canopylabs/orpheus-3b-0.1-ft` weights | tag **`license: apache-2.0`**, **gated** (`gated: "auto"`) | [HF API](https://huggingface.co/api/models/canopylabs/orpheus-3b-0.1-ft); `raw/main/README.md` → **HTTP 401 "Access to model … is restricted"** |
| Upstream base | `meta-llama/Llama-3.2-3B-Instruct` | `base_model` in the [same API record](https://huggingface.co/api/models/canopylabs/orpheus-3b-0.1-ft) |
| SNAC code + weights | **MIT** (© 2023 Descript, © 2024 Hubert Siuzdak) | [snac LICENSE](https://raw.githubusercontent.com/hubertsiuzdak/snac/main/LICENSE); `license: mit` in the [model card](https://huggingface.co/hubertsiuzdak/snac_24khz) |

Gating means accepting terms and authenticating with an HF token even for research, and the repo costs 56.7 GB of storage. Apache-2.0 on the card does **not** erase
the Llama-3.2 Community License flowing from the base weights — a downstream card logs the chain explicitly: "It transitively inherits obligations from … the
Llama-3 base architecture and weights — Meta Llama 3 Community License" ([Sunbird card](https://huggingface.co/Sunbird/orpheus-3b-tts-multilingual)). **Neither the Apache-2.0 text, the
README, nor the (gated) model card says anything about training on model outputs or about distillation — no permission and no prohibition found → UNVERIFIED, needs
counsel.** The practical signal is neutral-to-negative on quality grounds, not legal: the Orpheus authors warn that synthetic-audio training degrades their model.

### MiniMax

Governing document: [MiniMax Terms of Service](https://platform.minimax.io/protocol/terms-of-service), **Effective 2026-03-30**,
provider **Nanonoble Pte Ltd** (Singapore), **Singapore law**, **SIAC arbitration**. Verbatim clauses bearing on distillation:

- **User Rights and Obligations §7(a)**: "Copy, modify, **create derivative works of**, reverse engineer, decompile,
  translate, disassemble, or otherwise attempt to extract any or all source code from our services (unless expressly
  permitted by applicable law, in which case you must contact us to obtain the necessary information before attempting
  reverse engineering)."
- **Intellectual Property (Non-Transfer) and Indemnity Obligations §3**: "Except with express permission from us, you
  may not copy, imitate, modify, translate, adapt, lease, sell, sublicense, distribute over the internet, publish, or
  transfer any part of our products and services … You are also prohibited from reverse engineering, reverse assembling,
  decompiling, or **attempting to discover or decode the source code, algorithms, or object code** of the products and services."
- **Intellectual Property §3**: "As between you and us, and to the extent permitted by applicable laws, **you retain your
  ownership rights in Client input and generated content.** We may use the input and generated content to provide,
  maintain, develop, and improve our Services…"
- **AI Output §4**: "You are solely responsible for all use of any AI Output by you and your end users…"
- **User Rights and Obligations §1** (deep-synthesis duties): add "non-intrusive identifiers", keep logs, and "place a
  prominent mark … within generated or edited content to inform the public of the use of deep synthesis technology."

I grepped the full fetched ToS for `train`, `compete`, `distill`, `benchmark`: **no clause names model training, distillation, benchmarking, or competing-model
construction.** The exposure is the general "create derivative works of our services" / "decode the algorithms" language, not an explicit anti-distillation ban.
**UNVERIFIED / unresolved:** whether a distilled student is a "derivative work of the services" — the ToS does not answer it. Two separate documents exist
([platform](https://platform.minimax.io/protocol/terms-of-service) vs the [consumer app](https://agent.minimaxi.com/doc/en/terms-of-service.html)), the Cloudflare card links
`https://www.minimaxi.com/terms` which redirects cross-origin to `www.minimax.cn`, and the serving entity differs by surface — **pin the applicable terms per
integration.** One tension to flag: Cloudflare states **"Zero data retention: Yes"** while MiniMax's platform ToS reserves the right to use "input and generated
content" to improve services; these are different parties' statements, but the combination is **UNVERIFIED**.

---

## Actionable takeaways for Parakeet

1. **Split the two teachers into two distillation problems.** Orpheus yields *exact discrete codes* (SNAC, 7 tokens per
   83.33 ms super-frame, ids `128266 + N·4096`, de-interleave 1/2/4 per level, ~84 tok/s) plus all intermediate tensors
   if the 3B teacher stays resident. MiniMax yields **audio only** (8–44.1 kHz, mp3/wav/flac, plus `sentence`/`word`/
   `word_streaming` timestamps) — no logits, codes, weights, or published size. Run a token/feature-level path for
   Orpheus and an audio-level path (mel + MRSTFT + adversarial, optionally UTMOS-ranked selection) for MiniMax.
2. **The transferable result is the loss balance, not the architecture.** Spectral-only 2.98 → weight 45 gives 3.02 →
   weight 10 gives 4.32 → weight 3 gives 4.39 → 4.41 with the phase filter; wider/larger decoders did not help. Adopt
   `w_mel·(logmel_L1 + MRSTFT@[512,1024,2048]) + 1.0·adv + 2.0·fm` with `w_mel ∈ [3,10]`, AdamW `betas=(0.8,0.99)`, `wd=0.01`, grad-clip 5, 64-frame (~1.6 s) segments, MPD `(2,3,5,7,11)` **and** MRD `(512,1024,2048)`, LS-GAN + feature matching.
3. **Copy the phase-locking filter verbatim — free quality.** Parameter-free, post-synthesis, 2–8 kHz, voiced = `F0 > 60 Hz`,
   `rel = S·conj(E)/max(|E|,1e-6)`, smooth `rel`'s phase over `L ≈ 9–17` frames (Hann conv1d), STFT `n=1024, hop=256`, keep
   `|S|`; ~5 % of runtime. It needs the vocoder's **harmonic excitation**, so Parakeet's decoder must expose an equivalent source module.
4. **Quantize weight-only, per output channel, int8, fp16 scales, everything else fp16** (cost ~0 UTMOS: 4.41 → 4.41).
   Do **not** go to 4 bits (3.98) and **do not quantize activations**. Size accounting: `numel·bits/8 + out_channels·2`
   for quantized matrices, `numel·2` elsewhere.
5. **Supervise the text/prosody half on teacher features directly, never end-to-end through a decoder** (3.78 vs 4.39).
   MiniMax exposes no intermediates, so its durations/F0/energy must be *derived* (forced aligner, F0 tracker, RMS) and
   labelled as pseudo-targets, not distilled ones.
6. **Do not train an Orpheus-family code predictor on teacher audio alone.** Synthetic audio "maps to the same set of
   tokens" and causes "poor codebook utilisation". Either license a real speech corpus and use teacher outputs as
   auxiliary/soft targets, or distil the *distribution* (temperature + sequence-level KD over the 7-token super-frame).
7. **The corpus budget is small and reproducible:** 12,000 WikiText-103 sentences → 23.9 h → 6.9 GB of intermediates,
   shards of 500, 200 fixed held-out — enough to reach UTMOS 4.41 from a 4.52 teacher on one laptop-class machine
   (M4 Pro 24 GB, MPS) with ≤ 2 concurrent trainers.
8. **Align frame grids before mixing teachers.** Kokoro/Paradee: 600 samples @ 24 kHz = 40 Hz frames, F0/energy at 80 Hz.
   Orpheus/SNAC: 2048 samples = 12 Hz super-frames carrying 7 codebooks. MiniMax: arbitrary sample rate with word
   timestamps. Pick one student frame rate (40 Hz is proven) and resample every teacher stream onto it, including the
   excitation the phase filter consumes.
9. **Legal work is on the critical path, in risk order:** (i) **MiniMax** audio-derived distillation — "create derivative
   works of our services" + "decode the algorithms", no explicit training permission, plus a deep-synthesis marking duty;
   (ii) **Orpheus weights** — gated repo, Apache-2.0 tag, but a Llama-3.2-3B-Instruct base drags in Meta's Community
   License, and distillation is not addressed at all; (iii) **SNAC** — clean MIT; (iv) **Paradee** — Apache-2.0 code but
   CC BY-SA 3.0 on `training/data/wikitext_sents.json` (share-alike obligation on that data).
10. **Cheap export wins:** one ONNX graph `input_ids [1,T] → waveform`, phase filter baked in, `T ≤ 512` with pad-0 ends,
    `speed` as a scalar input, and avoid any STFT op not numerically identical to the one used in training (`CustomSTFT` cost ~0.5 UTMOS).
11. **Scope honesty:** Paradee demonstrates this pipeline for **one English voice** with an automatic predictor (UTMOS)
    plus informal listening — no formal listening study, no multi-voice, no multi-teacher, speed measured on one machine.

**Open items / explicitly UNVERIFIED:** MiniMax `speech-2.8-turbo` parameter count, architecture, tokenizer, latency/RTF
and MOS/UTMOS/WER (nothing published); the Orpheus multilingual voice list and per-language tags (the linked page no
longer publishes them); Orpheus effective parameter count (≈3.2–3.3B vs the 3.78B tensor count) and whether `lm_head` is
tied; whether MiniMax permits distillation in writing, and whether a distilled student is a "derivative work of the
services"; Paradee per-stage wall-clock and total compute hours; whether the Orpheus weights' `gated: "auto"` acceptance
conditions modify the license grant (the raw card required auth).
