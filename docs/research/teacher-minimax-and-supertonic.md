# Teacher research: MiniMax speech-2.8-turbo, Orpheus, and SupertonicTTS

Research note for **Parakeet** — a tiny, fast, high-realism TTS distilled from a *mix* of two teachers (Canopy Labs Orpheus + MiniMax `speech-2.8-turbo`), reusing architecture ideas from *SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System* ([arXiv 2503.23108](https://arxiv.org/abs/2503.23108), [alphaXiv](https://www.alphaxiv.org/abs/2503.23108)).

**Provenance legend:** **[OFFICIAL]** vendor/paper/repo primary source · **[3P]** third-party measurement, harness named · **[MARKETING]** promotional claim, not independently measured · **[UNVERIFIED]** not confirmable from a primary source in this pass. All numbers are quoted from the linked source; nothing is interpolated, and single-account benchmarks are flagged as such.

## 1. MiniMax speech-2.8-turbo (and the speech-01 / 02 / 2.5 / 2.6 lineage)

### 1.1 API surface and hard limits [OFFICIAL]

- Endpoint `POST https://api.minimax.io/v1/t2a_v2`; US-West variant `https://api-uw.minimax.io/v1/t2a_v2` ([T2A HTTP reference](https://platform.minimax.io/docs/api-reference/speech-t2a-http), [WebSocket guide](https://platform.minimax.io/docs/guides/speech-t2a-websocket)).
- Streaming: `wss://api.minimax.io/ws/v1/t2a_v2` (text sent sentence by sentence) and bidirectional `wss://api.minimax.io/ws/v1/t2a_v2_bidi` (text streamed at any granularity, even character-by-character; the server buffers into sentences) ([docs index](https://platform.minimax.io/docs/llms.txt)).
- Input limits: sync text **must be under 10,000 characters** (streaming recommended above 3,000) ([T2A HTTP reference](https://platform.minimax.io/docs/api-reference/speech-t2a-http)); async long-form allows **up to 1,000,000 characters per request** ([Async TTS guide](https://platform.minimax.io/docs/guides/speech-t2a-async)).
- Model enum on `t2a_v2`: `speech-2.8-hd`, `speech-2.8-turbo`, `speech-2.6-hd`, `speech-2.6-turbo`, `speech-02-hd`, `speech-02-turbo`, `speech-01-hd`, `speech-01-turbo` ([same reference](https://platform.minimax.io/docs/api-reference/speech-t2a-http)). **No `speech-2.5-*` id is in the current enum** — the 2.5 preview generation appears superseded.
- Output: non-streaming `mp3` / `wav` / `flac`, streaming `mp3` only; `output_format` is `hex` (default) or `url` (valid 24 h); the documented example payload uses `sample_rate: 32000`, `bitrate: 128000`, `channel: 1` ([T2A HTTP reference](https://platform.minimax.io/docs/api-reference/speech-t2a-http)). The full supported sample-rate range (commonly cited as 8–44.1 kHz) is **[UNVERIFIED]**.
- Voice cloning: `POST https://api.minimax.io/v1/files/upload` with `purpose="voice_clone"` (mp3/m4a/wav, **10 s – 5 min**, ≤20 MB), optional `prompt_audio` (<8 s), then `POST https://api.minimax.io/v1/voice_clone` with a custom `voice_id` ([Voice Clone guide](https://platform.minimax.io/docs/guides/speech-voice-clone)).
- Language: the `language_boost` enum lists 40 options including `auto`, `Chinese,Yue`, `English`, `Japanese`, `Korean`, `Thai`, `Persian`, `Filipino`, `Tamil`, `Afrikaans` ([T2A HTTP reference](https://platform.minimax.io/docs/api-reference/speech-t2a-http)); the guide claims **40 languages** and warns that `speech-01`/`speech-02` do **not** currently support Persian, Filipino or Tamil ([WebSocket guide](https://platform.minimax.io/docs/guides/speech-t2a-websocket)).
- Paralinguistics: **19 inline interjection tags**, only on `speech-2.8-hd` / `speech-2.8-turbo` — `(laughs) (chuckle) (coughs) (clear-throat) (groans) (breath) (pant) (inhale) (exhale) (gasps) (sniffs) (sighs) (snorts) (burps) (lip-smacking) (humming) (hissing) (emm) (sneezes)` ([T2A HTTP reference](https://platform.minimax.io/docs/api-reference/speech-t2a-http)). This tag set is the single most relevant 2.8-vs-2.6 capability delta for a teacher-distillation target.
- Other controls: `voice_modify` (`pitch`, `intensity`, `timbre`, `sound_effects` e.g. `spacious_echo`), `pronunciation_dict.tone`, `voice_setting` (`speed`/`vol`/`pitch`), plus `subtitle_enable` / `subtitle_type` ∈ {`sentence`, `word`, `word_streaming`} emitting **word-level timestamps** ([T2A HTTP reference](https://platform.minimax.io/docs/api-reference/speech-t2a-http)) — directly usable as teacher alignment supervision.
- Official SDK package names for speech: **[UNVERIFIED]** — the docs index lists raw cURL/Python/WebSocket examples; SDK pages found cover text (OpenAI/Anthropic/AI SDK), not TTS ([docs index](https://platform.minimax.io/docs/llms.txt)).

### 1.2 Architecture: what is actually disclosed

- **The only official architecture disclosure is for the `speech-02` generation**: *MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder* ([arXiv 2505.07916](https://arxiv.org/abs/2505.07916), May 2025). Reported components: a **BPE text tokenizer**; an **audio tokenizer with Encoder–VQ–Decoder architecture applying VQ on mel-spectrograms at 25 tokens/second with CTC supervision**; an **autoregressive Transformer** emitting discrete audio tokens; and a **latent flow-matching model** comprising a transformer-based flow-matching module plus a novel **Flow-VAE** whose decoder is a neural vocoder, with KL-divergence regularization ([HTML §3](https://arxiv.org/html/2505.07916v1)).
- A **learnable speaker encoder is trained jointly with the AR Transformer** and emits a fixed-size conditioning vector, enabling zero-shot cloning from untranscribed reference audio — the paper's headline contribution, explicitly contrasted against using a frozen pre-trained speaker-verification encoder ([§3](https://arxiv.org/html/2505.07916v1)).
- Emotion control is reported as **LoRA** on the AR Transformer; text-to-voice (T2V) compresses timbre features to **128 dimensions via PCA** and maps natural-language descriptions onto that space; "professional voice cloning" freezes the AR Transformer and fine-tunes only the speaker conditional embedding ([§5](https://arxiv.org/html/2505.07916v1)). The paper reports **32 languages** and a top position on the public TTS Arena leaderboard ([abstract](https://arxiv.org/abs/2505.07916)).
- **Whether `speech-2.8-turbo` preserves this architecture is [UNVERIFIED].** No technical report, model card, parameter count, codec spec or token rate for 2.6/2.8 was found; the 2.8 docs describe only capability deltas. Parameter count of any MiniMax speech model: **[UNVERIFIED]**.

### 1.3 Latency — separate the claim from the measurement

| Source | Kind | Claim |
|---|---|---|
| [MiniMax WebSocket guide](https://platform.minimax.io/docs/guides/speech-t2a-websocket) | **[OFFICIAL]**, qualitative | Recommends `api-uw.minimax.io` for **"reduced Time to First Audio (TTFA)"**; publishes no ms figure |
| [WaveSpeedAI launch post](https://wavespeed.ai/blog/posts/introducing-minimax-speech-2-8-turbo-on-wavespeedai/) | **[MARKETING]** (reseller) | "processing latency **under 250 milliseconds** and no cold starts **on WaveSpeedAI**" |
| [minimax-ai.chat 48-call benchmark](http://minimax-ai.chat/guide/speech-2-8-multilingual-test/) | **[3P]**: 8 languages × 2 models × 3 reps, one account, sequential | Turbo **p50 3,566 ms** vs HD 3,694 ms full-response; **p95 3,961.4 ms** vs 4,329.2 ms; RTF < 1 |

Critical caveat: that harness measured **time to last byte of a non-streaming HTTP response, not streaming TTFA** — stated explicitly by its authors ([methodology](http://minimax-ai.chat/guide/speech-2-8-multilingual-test/)). No independent streaming time-to-first-audio number for 2.8 was found, so **real TTFA is [UNVERIFIED]**. Effectively ~3.2–3.8 s p50 per 8–10 s utterance, i.e. RTF ≈ 0.4 over a single sequential connection.

### 1.4 Pricing [OFFICIAL]

- `speech-2.8-turbo` **$60 per million characters**; `speech-2.8-hd` **$100 / M characters**, for both sync and async T2A. Legacy in the same table: `speech-2.6-turbo` / `speech-02-turbo` $60/M, `speech-2.6-hd` / `speech-02-hd` $100/M ([Pay-as-you-go pricing](https://platform.minimax.io/docs/guides/pricing-paygo)).
- Voice Design **$3 per voice** (charged on first synthesis use); Rapid/LLM-powered Voice Cloning **$1.50 per voice** ([same](https://platform.minimax.io/docs/guides/pricing-paygo)).
- Audio Subscription $5 / $30 / $99 / $249 / $999 per month including 100k / 300k / 1.1M / 3.3M / 20M audio points, with 10 / 100 / 250 / 500 / 800 voice slots and 10 / 50 / 200 / 500 / 800 RPM ([Audio Subscription](https://platform.minimax.io/docs/guides/pricing-speech)).
- Independent cost check at list price: 4,752 API-reported characters ⇒ **$0.28512 Turbo vs $0.47520 HD**, a 40% reduction ([3P](http://minimax-ai.chat/guide/speech-2-8-multilingual-test/)).

### 1.5 Developer terms relevant to distillation — the load-bearing finding

**We found an explicit prohibition on using MiniMax audio to develop foundation models.** In the *MiniMax Voice (MiniMax语音) user agreement*, updated **2025-06-21**, §三 (service use rules), verbatim:

> 未经我们书面许可，任何人均不得自行或授权、允许、协助他人对MiniMax语音产品和服务的信息内容（包括但不限于图片、文字、音视频、代码、电子信息等）进行任何形式的改变、复制、传播、收集、编辑、开发、垂直搜索、镜像、反向工程（对系统和算法源代码的反编译等）、**使用MiniMax语音开发基础模型**或用于其他与我们竞争的行为、通过我们的APP、API等任何渠道抓取我们的数据…

Translation (ours): *"Without our written permission, no one may — on their own or by authorizing, permitting or assisting others — perform any form of alteration, copying, distribution, collection, editing, development, vertical search, mirroring, or reverse engineering (decompilation of system and algorithm source code, etc.) on the information content of the MiniMax Voice products and services (including but not limited to images, text, audio/video, code, electronic information), **use MiniMax Voice to develop foundation models**, or engage in other behaviour competing with us, scrape our data through our APP, API or any other channel, …"* — source: [MiniMax语音用户协议 §三](https://www.minimax.cn/audio/doc/terms-of-service.html).

The same section separately bars reverse engineering of the model/algorithm: *"对本产品和服务进行反向工程、反向汇编、反向编译、翻译或者以其他方式尝试发现本软件的源代码、模型、算法和系统的源代码或底层组件"* ([same §三](https://www.minimax.cn/audio/doc/terms-of-service.html)).

Countervailing clause — **output ownership does vest in the user**, but with a broad grant back: *"在您与我们之间，在适用法律允许的范围内，**输出内容的知识产权及相关权益归属于您**。… 对于输入内容和您享有知识产权等权益的输出内容（如有），您同意授予我们和/或关联方一项**免费的、无需标注您个人身份的、全球范围内的使用权** … 例如用于产品和服务的提升和优化、品牌推广和宣传。"* ([§四 知识产权](https://www.minimax.cn/audio/doc/terms-of-service.html)). So you own the output IP, but may not use the service to *develop foundation models*, and you grant MiniMax a free worldwide licence to your inputs and outputs for service improvement.

**[UNVERIFIED]:** whether the *platform/API* terms (rather than the consumer audio-product agreement) carry equivalent language. `https://platform.minimax.io/protocol/terms-of-service` and `/protocol/privacy-policy` both return HTTP 200 but render as a JavaScript application with **no extractable text body** (raw HTML 115,611 bytes → 20 characters of text); `https://www.minimax.io/terms` and `/terms-of-service` both 301-redirect to the marketing homepage. A human must open the platform ToS in a browser and confirm before any distillation from MiniMax output.

**Practical read for Parakeet:** even if the platform ToS is silent, the vendor's own audio-product agreement asserts a "no foundation models from our output" restriction. Treat MiniMax-generated audio as **NOT SAFE for distillation-training data** absent written permission.

## 2. SupertonicTTS in implementation detail ([arXiv 2503.23108](https://arxiv.org/abs/2503.23108) v3, 22 pp)

Three components trained in three phases: speech autoencoder → text-to-latent flow-matching module → utterance-level duration predictor ([§3/§4](https://arxiv.org/html/2503.23108v3)).

### 2.1 Speech autoencoder (Vocos-based) [OFFICIAL]

Built on **Vocos** ([arXiv 2306.00814](https://arxiv.org/abs/2306.00814)) with the Fourier head removed.

- **Latent encoder**: first conv + BatchNorm maps a **228-dim mel spectrogram** → **512-dim** hidden at unchanged sequence length; **10 ConvNeXt blocks**, intermediate dim **2048**; final linear + LayerNorm projects 512 → **24-dim latent**; all convs kernel size **7**. *"compresses a 228-dimensional mel spectrogram into 24-dimensional latents while maintaining the original sequence length"* ([Appendix A.1.1](https://arxiv.org/html/2503.23108v3)).
- **Latent decoder**: 24-dim latent → 512; **10 dilated ConvNeXt blocks**, intermediate dim **2048**, depthwise dilation rates **[1, 2, 4, 1, 2, 4, 1, 1, 1, 1]**; kernel-3 conv to **2048**; final linear to **512 frame-level channels**, reshaped to a single-channel waveform. **All decoder conv layers are causal**, which is what enables streaming ([Appendix A.1.2](https://arxiv.org/html/2503.23108v3)).
- Mel config: FFT **2048** (46.43 ms), Hann window 2048, hop **512** (11.61 ms), **228 mel bands** — at 44.1 kHz that is **~86 Hz** ([§4.2](https://arxiv.org/html/2503.23108v3), [§3.2.1](https://arxiv.org/html/2503.23108v3)).
- Adversarial training cropped real/generated speech to **0.19 s**; loss weights `λ_adv = 1`, `λ_fm = 0.1`; trained **1.5M iterations**, AdamW ([§4.2](https://arxiv.org/html/2503.23108v3)). Discriminators: multi-period (periods 2, 3, 5, 7, 11; six conv layers with hidden sizes 16/64/256/512/512/1) and multi-resolution with FFT sizes 512/1024/2048 and hop = FFT/4 ([Appendix](https://arxiv.org/html/2503.23108v3)).

### 2.2 Temporal compression factor `Kc` [OFFICIAL]

Given factor `Kc`, a latent `(C, T)` becomes `(Kc·C, T/Kc)` — channel-expanding, frame-reducing, and exactly invertible ("all temporal information is preserved … allowing for perfect inversion") ([§3.2.1](https://arxiv.org/html/2503.23108v3)).

- **`Kc = 6`** with **`C = 24`** ⇒ text-to-latent module input dim **`Kc·C = 144`**.
- Rationale: aligns the autoencoder to **~86 Hz** (typical vocoder rate) and the text-to-latent module to **~14 Hz** (typical semantic-token rate). Claimed benefit: fewer speech frames ⇒ cheaper attention and easier text-speech alignment ([§3.2.1](https://arxiv.org/html/2503.23108v3)).
- `Kc` values other than 6 were searched for but no sweep table exists in the text ⇒ **[UNVERIFIED]**.

### 2.3 Text encoder: character-level, no G2P [OFFICIAL]

*"A key design choice is the exclusion of G2P and other pretrained modules, ensuring the model learns everything directly from the character input."* ([§3.2.3](https://arxiv.org/html/2503.23108v3))

- Structure: **embedder → 6 ConvNeXt blocks → 4 self-attention blocks → 2 cross-attention layers**.
- Embedder: character → **128-dim** lookup table. ConvNeXt kernel size **5**, intermediate dim **512**.
- Self-attention: transformer-encoder style, **512 filter channels, 4 heads, rotary position embedding**.
- Cross-attention: three linear layers with equal in/out dims; the **first cross-attention layer uses 50 learnable vectors (dim 128)** as reference keys — the same 50 vectors are reused as keys in the VF estimator ([Appendix A.2](https://arxiv.org/html/2503.23108v3)).

### 2.4 Reference / speaker conditioning [OFFICIAL]

- Reference encoder: linear **144 → 128**; **6 ConvNeXt blocks** (kernel 5, intermediate 512); **2 cross-attention layers**; **50 learnable vectors of dim 128** in the first attention block produce a fixed number of reference value vectors. Design follows the timbre-token block of **NANSY++** ([arXiv 2211.09407](http://arxiv.org/pdf/2211.09407v1)) ([§3.2.3](https://arxiv.org/html/2503.23108v3)).
- Training-time reference audio = random crop of the input with duration **0.2 s – 9 s**, capped at half the original utterance duration ([§4.2](https://arxiv.org/html/2503.23108v3)).

### 2.5 Duration predictor (DP) [OFFICIAL]

- Total budget **~0.5 M parameters** — deliberately lightweight, because predicting *total* duration is simpler than phoneme-level duration ([Appendix A.3](https://arxiv.org/html/2503.23108v3)).
- DP reference encoder: linear **144 → 64**; **4 ConvNeXt blocks** (kernel 5, intermediate 256); **2 cross-attention layers** projecting to **16-dim** vectors; **8 learnable query vectors**; outputs stacked to a **64-dim** vector.
- DP text encoder: embedder → **64-dim**; **6 ConvNeXt blocks** (kernel 5, intermediate 256); **2 self-attention blocks** (256 channels, 2 heads, RoPE); a learnable **64-dim utterance token** is prepended; the first output vector passes a linear layer to give an utterance-level text embedding.
- Duration estimator: two linear layers with PReLU; the first keeps **164** dims in/out (the exact 164 composition is **[UNVERIFIED]**), the second maps to **one scalar**.
- Trained only **3,000 iterations**, AdamW, lr **5e-4**, batch **128**, single RTX 4090 ([§4.2](https://arxiv.org/html/2503.23108v3)).

### 2.6 Text-to-latent flow-matching module (`#T2F`) [OFFICIAL]

- Vector field (VF) estimator: first linear maps **144-dim noisy latents → 256**; main block = **4 dilated ConvNeXt blocks (dilations 1, 2, 4, 8) + 2 standard ConvNeXt blocks + TimeCondBlock + TextCondBlock + RefCondBlock**, every ConvNeXt with kernel **5** and intermediate dim **1024**; then **4 more ConvNeXt blocks** and a final linear 256 → **144** ([Appendix A.2](https://arxiv.org/html/2503.23108v3)).
- Time conditioning: a **64-dim** time embedding (Grad-TTS-style) projected by a single linear layer and added **globally** to the input sequence. Text and reference conditioning both use cross-attention with the conditional variables as keys/values ([§3.2.3](https://arxiv.org/html/2503.23108v3)).
- Inference: **Euler method**, **NFE = 32**, **CFG coefficient = 3**, `σ_min = 1e-8`; training used `p_uncond = 0.05` ([§4.2](https://arxiv.org/html/2503.23108v3), [Appendix D.1](https://arxiv.org/html/2503.23108v3)).
- NFE ablation (LS-clean, RTX 4090): NFE 4 → RTF 0.006 / WER 11.43 / SIM 0.335; 8 → 0.011 / 2.818 / 0.472; 16 → 0.019 / 2.679 / **0.476**; **32 → 0.037 / 2.639 / 0.472**; 64 → 0.071 / 2.705 / 0.470; 128 → 0.140 / 2.693 / 0.468 (NISQA rises monotonically to 4.070 at NFE 128). **NFE 4 is catastrophic (WER 11.43)** — for Parakeet, sub-8-NFE operation is the real engineering risk ([Table 8](https://arxiv.org/html/2503.23108v3)).
- Follow-up work introduces LARoPE (length-aware RoPE) for this cross-attention alignment ([arXiv 2509.11084](https://arxiv.org/abs/2509.11084)) and self-purifying flow matching ([arXiv 2509.19091](https://arxiv.org/abs/2509.19091)); both are listed as core technologies in the official release notes ([Supertonic README](https://raw.githubusercontent.com/supertone-inc/supertonic/main/README.md)).

### 2.7 Context-sharing batch expansion (`Ke`) [OFFICIAL]

Definition: for expansion factor `Ke`, draw **`Ke` independent noise–timestep pairs** from the same input but **reuse the same conditioning variables across all `Ke` samples**. Because conditions are pre-encoded, this mimics a larger batch at lower compute and I/O cost ([§3.2.2](https://arxiv.org/html/2503.23108v3)).

- Training used **`Ke = 4`** ([§4.2](https://arxiv.org/html/2503.23108v3)).
- Claim, backed by Fig. 6: increasing `Ke` accelerates convergence of *both* validation loss and WER, whereas increasing plain batch size `B` reduces validation loss but **slows WER convergence**; the method also reduces word skipping, repetition and mispronunciation ([Appendix D.2](https://arxiv.org/html/2503.23108v3)). A pseudo-algorithm is in Appendix C ([Algorithm 1](https://arxiv.org/html/2503.23108v3)).

### 2.8 Training data and optimization [OFFICIAL]

| Stage | Data | Optimization |
|---|---|---|
| Speech autoencoder | **11,167 hours**, ~**14,000 speakers**, public + internal ([§4.1](https://arxiv.org/html/2503.23108v3)) | 1.5M iters, AdamW ([§4.2](https://arxiv.org/html/2503.23108v3)) |
| Text-to-latent + DP | **945 hours**, **2,576 English speakers** from public sets (VCTK, Hi-Fi TTS, LibriTTS, …), resampled to **44.1 kHz** ([§4.1](https://arxiv.org/html/2503.23108v3)) | 700k iters, batch 64, `Ke=4`, lr 5e-4 halved every 300k, **4× RTX 4090** ([§4.2](https://arxiv.org/html/2503.23108v3)) |
| Duration predictor | same 945 h | 3,000 iters, batch 128, lr 5e-4, 1× RTX 4090 ([§4.2](https://arxiv.org/html/2503.23108v3)) |

- Latents are normalized with precomputed channel-wise mean/variance before entering the text-to-latent module ([§4.2](https://arxiv.org/html/2503.23108v3)).
- Evaluation sets: LT-clean, LT-other (LibriTTS test), LS-clean (LibriSpeech test-clean), LS-PC-clean (Chen et al.); samples 4–10 s ([§5](https://arxiv.org/html/2503.23108v3)).
- **Data-efficiency headline**: 945 h vs 55k–100k h for baselines ([Table 5](https://arxiv.org/html/2503.23108v3)).

### 2.9 Parameter counts and speed [OFFICIAL]

From Tables 5/6 (`#DP` duration predictor, `#T2F` text-to-feature, `#F2S` feature-to-speech, `#All` whole system) ([Table 5](https://arxiv.org/html/2503.23108v3)):

| System | Data (h) | #DP | #T2F | #F2S | #All | RTF |
|---|---|---|---|---|---|---|
| SupertonicTTS (**Ours**) | **945** | **0.5 M** | **18.5 M** | **25 M** | **44 M** | **0.02** (RTX 4090) / **0.05** (RTX 3090) |
| VALL-E | 60k | — | 403 M† | 7 M | 410 M† | ~0.64 |
| VoiceBox | 60k | 28 M | 330 M | 13 M | 371 M | ~0.62 |
| CLaM-TTS | 55k | — | >1.23 B† | 112 M | >1.3 B† | 0.42 (A100) |
| DiTTo-TTS | 55k | 33 M | 825 M | 112 M | 940 M | 0.16 (A100) |
| FireRedTTS | 248k* | — | 538 M | 235 M | 773 M | 0.84 (RTX 3090) |
| F5-TTS | 100k* | — | 335.8 M | 13.5 M | 349 M | 0.31 (RTX 3090) |

`†` estimated from the baseline's architecture description. The text-to-latent module at **18.5 M** is **~18× smaller** than the next smallest baseline's T2F (VoiceBox, 330 M). WER: 2.64 on LS-clean, 2.41 on LS-PC-clean (vs GT 2.18 / 1.86) ([§6](https://arxiv.org/html/2503.23108v3)). The speech autoencoder is reported **>20× faster than BigVGAN** at comparable NISQA/UTMOSv2 ([§5.1](https://arxiv.org/html/2503.23108v3)).

### 2.10 Release status — important gap [OFFICIAL]

- The paper's **44 M** configuration is **not** the released artifact. The Supertone lineage is **Supertonic 1 ≈ 66 M** (English only), **Supertonic 2 ≈ 66 M** (5 languages), **Supertonic 3 ≈ 99 M** (**31 languages**, 10 inline expression tags e.g. `<laugh>`, `<breath>`, `<sigh>`; 44.1 kHz output) ([Supertonic README, Models & Versions](https://raw.githubusercontent.com/supertone-inc/supertonic/main/README.md)).
- **Code and weights were released, then archived**: source at [supertone-oss-archive/supertonic](https://github.com/supertone-oss-archive/supertonic); weights at [HF supertonic-3](https://huggingface.co/supertone-oss-archive/supertonic-3), [supertonic-2](https://huggingface.co/supertone-oss-archive/supertonic-2), [supertonic](https://huggingface.co/supertone-oss-archive/supertonic). The README now states: *"This repository is archived. Development and support have ended."*
- **Licensing is split**: *"This project's sample code is released under the MIT License… The accompanying model is released under the OpenRAIL-M License"* ([README §License](https://raw.githubusercontent.com/supertone-inc/supertonic/main/README.md)) → **code MIT, weights OpenRAIL-M**.
- Deployment target is ONNX Runtime (CPU, WebGPU, Raspberry Pi, e-reader); the README cites an **average RTF of 0.3×** on an Onyx Boox Go 6 e-reader ([README](https://raw.githubusercontent.com/supertone-inc/supertonic/main/README.md)). Voice cloning is **not** in the open-weight repo; a hosted "Voice Builder" produced per-voice JSON and was scheduled to go inaccessible after **2026-08-31** ([README service notice](https://raw.githubusercontent.com/supertone-inc/supertonic/main/README.md)).

### 2.11 Reference architectures it builds on

| Paper | ID | Relevance |
|---|---|---|
| Vocos | [2306.00814](https://arxiv.org/abs/2306.00814) | Backbone of both latent encoder and decoder (ConvNeXt blocks; Fourier head removed) |
| ConvNeXt | [2201.03545](https://arxiv.org/abs/2201.03545) | The block primitive used everywhere (text enc., reference enc., VF estimator, AE) |
| Flow Matching for Generative Modeling | [2210.02747](https://arxiv.org/abs/2210.02747) | Training objective for the text-to-latent module |
| NANSY++ | [2211.09407](http://arxiv.org/pdf/2211.09407v1) | Timbre-token block design behind the reference encoder |
| Grad-TTS | (cited for time embedding) | Source of the 64-dim time-embedding scheme |
| LARoPE | [2509.11084](https://arxiv.org/abs/2509.11084) | Length-aware RoPE for text-speech cross-attention alignment |
| Self-Purifying Flow Matching | [2509.19091](https://arxiv.org/abs/2509.19091) | Training flow matching with noisy/unreliable labels |
| RobustSpeechFlow | [2605.22083](https://arxiv.org/abs/2605.22083) | Augmentation-based contrastive flow matching (2026 follow-up) |

## 3. Practical comparators for a tiny TTS

### 3.1 Master comparison table

| Model | Params | Architecture family | Audio / latent rate | Latency / RTF | Code+weights public | License | Note |
|---|---|---|---|---|---|---|---|
| **SupertonicTTS (paper)** | **44 M** (0.5 + 18.5 + 25) | Vocos AE + latent flow matching + utterance DP | 44.1 kHz; latent 24-d @ ~86 Hz; T2L @ ~14 Hz (`Kc=6`) | **RTF 0.02** RTX 4090 / **0.05** RTX 3090 at NFE 32 | **No** — paper only; the 44 M config was never released | paper CC BY-NC-SA 4.0 | [Table 5](https://arxiv.org/html/2503.23108v3) |
| **Supertonic 3 (released)** | ~**99 M** | same lineage, ONNX Runtime | 44.1 kHz | **[3P]** mean RTF **0.313** @5-step, **0.165** @2-step (4-core EPYC 7763 CPU, no GPU) | **Yes**, archived | code **MIT** / weights **OpenRAIL-M** | 2-step output "robotic, unclear" per [3P CPU benchmark](https://heyneo.com/blog/kokoro-tts-vs-supertonic-3-tts) |
| **Kokoro-82M** | **82 M** | StyleTTS2 + ISTFTNet; espeak-ng/misaki G2P | 24 kHz | **[3P]** RTF **0.469** (PyTorch CPU) / **0.509** (ONNX CPU) | **Yes** | **Apache-2.0** (code + weights) | 8 langs / 54 voices; 1000 A100-h, ~$1000 ([card](https://huggingface.co/hexgrad/Kokoro-82M)) |
| **ZipVoice** | **123 M** | Zipformer flow-matching VF estimator + text encoder + flow distillation | **[UNVERIFIED]** | **4–8 NFE** (distilled); paper claims 3× smaller and up to **30× faster** than a DiT flow-matching baseline | **Yes** | **Apache-2.0** | 100k h Emilia; [README: "only 123M parameters"](https://github.com/k2-fsa/ZipVoice) |
| **Kitten TTS Nano 0.8** | **15 M** (~50 MB) | StyleTTS2 architecture | 24 kHz | **[UNVERIFIED]** | **Yes** | **Apache-2.0** | 8 voices, English; [model card](https://huggingface.co/KittenML/kitten-tts-nano-0.8) |
| **Piper / VITS-class** | **[UNVERIFIED]** | VITS + **espeak-ng** phonemization | 16 kHz / 22.05 kHz voice tiers | **[UNVERIFIED]** | **Yes** | **GPL-3.0** (`piper1-gpl`) | [repo](https://github.com/OHF-Voice/piper1-gpl); GPL is a real constraint for a permissive Parakeet |
| **F5-TTS** | **349 M** (335.8 T2F + 13.5 F2S) | DiT-style flow matching, no duration predictor | **[UNVERIFIED]** | **RTF 0.31** RTX 3090 | **Yes** | **CC-BY-NC-4.0** (weights) | 100k h Emilia; **non-commercial** ([card](https://huggingface.co/SWivid/F5-TTS)) |
| **dots.tts-mf** | **2 B** | Qwen2.5-1.5B LLM + AR flow-matching head over **48 kHz AudioVAE**; MeanFlow distillation | 48 kHz | **NFE 4** (range 2–4), single model eval/step, CFG fused | **Yes** | **Apache-2.0** | no discrete codec tokens; [card](https://huggingface.co/dots-studio/dots.tts-mf) |
| **Orpheus (teacher 1)** | **3 B** | Llama-family autoregressive TTS | **[UNVERIFIED]** | **[UNVERIFIED]** | **Yes** | **Apache-2.0** | English, `llama` arch tag ([card](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)) |
| **MiniMax speech-2.8-turbo (teacher 2)** | **[UNVERIFIED]** | (02 generation) BPE text tok. + VQ audio tok. @25 tok/s + AR Transformer + Flow-VAE flow matching | 32 kHz (documented example payload) | **[3P]** non-streaming p50 3,566 ms, RTF <1; **[MARKETING]** "<250 ms" only on WaveSpeedAI | **No** (proprietary) | proprietary ToS; **foundation-model use restricted** (§1.5) | [tech report](https://arxiv.org/abs/2505.07916) covers `speech-02`, not 2.8 |

### 3.2 What the comparator table actually says

- **Sub-20M is real but not yet good**: Kitten TTS Nano at 15 M is the only public checkpoint in Parakeet's target size class, and it is English-only with **no published RTF or quality benchmark** ([card](https://huggingface.co/KittenML/kitten-tts-nano-0.8)) — quality is **[UNVERIFIED]**.
- **The 44–123 M band is where the evidence is**: SupertonicTTS at 44 M and ZipVoice at 123 M both approach 350 M–1 B baselines on WER/SIM while being far faster. This is the strongest support for Parakeet's size target.
- **CPU RTF ≈ 0.3 at 5 NFE is the realistic deployment bar** for a sub-100 M model in 2026; GPU RTF 0.02–0.05 is achievable but does not describe the edge target ([3P benchmark](https://heyneo.com/blog/kokoro-tts-vs-supertonic-3-tts), [Table 5](https://arxiv.org/html/2503.23108v3)).
- **Licence hygiene is a live risk**: two of the strongest quality references are unusable as-is for a permissive product — F5-TTS weights are **CC-BY-NC-4.0**, Supertonic weights are **OpenRAIL-M**, Piper is **GPL-3.0**. The Apache-2.0 options are Kokoro, ZipVoice, Kitten TTS and dots.tts.
- **G2P-free is a differentiator**: SupertonicTTS, ZipVoice, dots.tts and Orpheus consume text directly (characters or BPE), whereas Kokoro (misaki/espeak-ng) and Piper (espeak-ng) need an external G2P — a portability *and* licensing consideration, since espeak-ng is GPL-3.0.

## Actionable takeaways for Parakeet

1. **Copy SupertonicTTS's geometry almost verbatim; it is fully specified.** 24-dim latent at ~86 Hz (44.1 kHz, hop 512, 228 mel bands), `Kc = 6` → 144-dim / ~14 Hz for the flow module; Vocos-style 10-block ConvNeXt encoder/decoder with hidden 512 and intermediate 2048; causal dilated decoder for streaming; character-level text encoder (char→128, 6 ConvNeXt + 4 self-attn @512/4 heads/RoPE + 2 cross-attn with 50 learnable 128-d vectors); VF estimator 144→256 with 4 dilated (1,2,4,8) + 2 plain ConvNeXt (kernel 5, intermediate 1024) plus time/text/ref conditioning blocks. All per [Appendix A](https://arxiv.org/html/2503.23108v3).
2. **Budget ~44 M total: 0.5 M duration predictor + 18.5 M text-to-latent + 25 M decoder/vocoder.** The DP is nearly free — do not over-invest there ([Table 5](https://arxiv.org/html/2503.23108v3)).
3. **Adopt `Ke`-style context-sharing batch expansion at `Ke = 4`.** It demonstrably beats plain batch-size scaling on alignment/WER convergence and reduces skipping and repetition, at low memory/IO cost ([Appendix D.2](https://arxiv.org/html/2503.23108v3)).
4. **Do not chase NFE below 8.** SupertonicTTS at NFE 4 collapses to **WER 11.43** vs 2.64 at NFE 32; the knee is 8–16. If low NFE is required, use a *flow-distillation* objective (ZipVoice's approach, down to 4 steps with a distilled student) rather than naive step reduction ([Table 8](https://arxiv.org/html/2503.23108v3), [ZipVoice](https://arxiv.org/abs/2506.13053)).
5. **The 945-hour data budget is a feature, not a limitation** — 945 h beat baselines trained on 55k–100k h on WER (2.64 LS-clean, 2.41 LS-PC-clean) at 44 M params ([Table 5](https://arxiv.org/html/2503.23108v3)). A teacher-distillation corpus should target a comparable order of magnitude, not millions of hours.
6. **Do not train on MiniMax output without written permission.** The MiniMax Voice user agreement explicitly bars *"使用MiniMax语音开发基础模型"* ("using MiniMax Voice to develop foundation models") without prior written consent ([§三](https://www.minimax.cn/audio/doc/terms-of-service.html)). The English platform ToS could not be machine-read and needs human review. Orpheus weights, by contrast, are **Apache-2.0** ([card](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)) — a Parakeet distilled primarily from Orpheus or from Apache-2.0 corpora is the low-risk path.
7. **Borrow MiniMax 2.8's *targets*, not its weights.** The 19 interjection tags and emitted word-level timestamps are a ready-made supervision taxonomy: generate a tag-annotated, timestamped corpus and train Parakeet to reproduce those paralinguistic behaviours, treating the tags as extra character vocabulary handled by SupertonicTTS-style cross-attention alignment ([T2A reference](https://platform.minimax.io/docs/api-reference/speech-t2a-http)).
8. **Match the licence posture deliberately.** Ship Apache-2.0 code + weights (like Kokoro/ZipVoice/Kitten) if Parakeet is to be broadly reusable; avoid inheriting OpenRAIL-M (Supertonic weights), CC-BY-NC-4.0 (F5-TTS) or GPL-3.0 (Piper, espeak-ng).
9. **Ship ONNX and a streaming causal decoder.** The "fast + tiny" story in this space is told on **CPU** (RTF ~0.3 at 5 NFE, 0.165 at 2 NFE on 4 EPYC cores; Supertonic README cites RTF 0.3× on an e-reader) — a GPU-only RTF 0.02 will not win the comparison ([3P benchmark](https://heyneo.com/blog/kokoro-tts-vs-supertonic-3-tts), [README](https://raw.githubusercontent.com/supertone-inc/supertonic/main/README.md)).
10. **Open questions to close before committing:** (a) MiniMax platform-level ToS text — manual browser review required; (b) whether `speech-2.8` retains the `speech-02` Flow-VAE architecture; (c) Kitten TTS Nano's actual quality/RTF at 15 M; (d) ZipVoice's audio sample rate and measured CPU RTF; (e) the exact composition of the duration predictor's 164-dim input.
