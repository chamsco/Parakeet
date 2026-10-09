# Evaluation

Four numbers decide whether Parakeet is a success. They are listed in priority order, because a
TTS model that is not fast is not the product we are building.

| # | Metric | Why | Tool | Implemented |
|---|---|---|---|---|
| 1 | **RTF on one CPU thread** | the entire premise is "lightning fast, on a laptop" | `measure_rtf` (dependency-free, pins `torch.set_num_threads(1)`) | ✅ |
| 1b | **TTFA** (time to first audio) | for interactive use the first chunk matters more than the total | `scripts/streaming_demo.py`, `Synthesizer.synthesize_stream` | ✅ |
| 2 | **UTMOS** (1–5 neural naturalness) | the metric Paradee's ablations are stated in | optional `utmos`/`torch.hub` | optional (declared unavailable if missing, never faked) |
| 3 | **WER** (Whisper large-v3) | intelligibility; catches word skips/repeats | optional `faster-whisper` | optional |
| 4 | **SECS** speaker similarity | required for the zero-shot Small model | frozen CAM++ via `funasr` | optional |
| — | MCD (dB), log-mel L1, spectral convergence, segmental SNR, **phase coherence 2–8 kHz** | always-available diagnostics; CI uses these | `parakeet.eval` | ✅ |

`list_metric_availability()` reports which optional metrics are live in the current environment, so
a report can never silently present an untrained-encoder number as SECS.

## Targets to beat

| Model | Params | RTF (CPU) | UTMOS | WER | Notes |
|---|---|---|---|---|---|
| **Paradee** (Paradee's own result) | 8.07 M | 25.0× real time, 1 thread (17.8× ONNX); 3.4 GFLOP/s | 4.41 ± 0.01 (teacher Kokoro 4.52) | 5.7 % (teacher 5.7 %) | single English voice, 23.9 h corpus, int8 8.45 MB |
| **Kokoro-82M** | 82 M | 0.469 (4-core EPYC) | — | — | Apache-2.0, needs espeak-ng/misaki G2P |
| **Supertonic-3** | ~99 M | 0.313 @5 steps (0.165 @2 steps, robotic) | — | — | the shipping model, not the paper's 44 M |
| **SupertonicTTS (paper)** | 44 M | 0.02 RTF on a 4090 | — | 2.64 % @NFE 32; 11.43 % @NFE 4 | 945 h training data |
| **ZipVoice** | 123 M | — | — | — | flow distillation, 4–8 NFE, Apache-2.0 |
| **Kitten TTS Nano 0.8** | 15 M | UNVERIFIED | UNVERIFIED | UNVERIFIED | closest public model to our Tiny size class |
| **Parakeet-Tiny (target)** | **9.6 M** | **≥ 25× real time, 1 thread** | **≥ 4.35** | ≤ 6 % | must beat Paradee on size, match it on speed/quality |
| **Parakeet-Small (target)** | **45 M** | ≤ 0.3 CPU / RTF ≤ 0.02 GPU | ≥ 4.40 | ≤ 3 % | must be competitive with SupertonicTTS at 44 M |

Comparison protocol for fairness: 200 held-out sentences, 95 % CIs, the same machine for every
model, one CPU thread for speed, and `phase_lock` **off** when comparing raw decoder quality (on
when reporting the shipped configuration), because the filter is part of our system but not of
theirs.

## Diagnostic interpretation

* **`phase_coherence_2_8k`** — higher is better (less buzz). Measured on a synthetic
  randomised-phase voiced signal, the filter raises it substantially; on an untrained model's
  output the change is small, which is expected since the base output is noise-like. Use it
  *before/after* the filter on a **trained** model, and track it as a release gate.
* **MCD** — mostly a speaker/timbre-consistency check. A large MCD with a good UTMOS usually means
  the identity conditioning is weak.
* **Segmental SNR** — sensitive to phase; it will move when you enable the phase lock, which is
  the point.
* **`stream_vs_offline_max_diff`** — not a quality metric but a correctness gate; it must stay at
  float precision (measured 5.6e-09). A regression here means the streaming decoder broke.

## Harness

`scripts/smoke_test.py` already prints the correctness and speed halves of this table on every run:
per-stage losses, parameter counts, int8 size, phase coherence before/after, streaming vs offline
difference, and RTF at one thread. Additional CPU-verifiable harnesses:

| Script | What it establishes |
|---|---|
| `scripts/learn_demo.py` | the stages actually learn (before/after metrics with pass/fail) |
| `scripts/reflow_demo.py` | NFE-2 sampling is viable after Reflow, and reports what the experiment *cannot* conclude |
| `scripts/streaming_demo.py` | TTFA speed-up vs total-compute cost, blockwise agreement with the one-shot sampler, and a positive control proving the metric detects error |
| `scripts/export_onnx.py` | PyTorch vs ONNX-fp32 vs ONNX-int8 latency, size, and output deviation |
| `scripts/profile_pipeline.py` | component-wise share of a full synthesis, so optimisation targets the right thing |

Use `profile_pipeline.py` as a release gate too: if the "bottleneck" line moves to the text side or
to python/dispatch overhead, the ONNX work is done and the next lever is elsewhere.
`parakeet/eval/metrics.py` provides `evaluate_pair()` (student vs teacher) and `EvalReport.to_json()`
for run-over-run comparison; wire it into a release script once a checkpoint exists
(`docs/ROADMAP.md` P4).
