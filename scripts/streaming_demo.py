"""Streaming (blockwise) sampling validation for Parakeet-Small.

    python scripts/streaming_demo.py --quick      # ~1 minute
    python scripts/streaming_demo.py              # ~8 minutes on 8 CPU cores

The one-shot sampler must integrate the whole latent before any audio exists, so time-to-first-audio
grows with the utterance.  Parakeet's vector field mixes time with *finite-support* depthwise
convolutions (not a transformer), so the ODE can be integrated block by block with a bounded window.

Two things have to be true, and both are measured here rather than asserted:

1. **Agreement.**  Blockwise sampling is an approximation of the full-sequence ODE.  The yardstick is
   not "zero error" -- it is *whether the approximation stays within the sampling variability of the
   model itself*, i.e. closer to the one-shot result than an independent draw from the same
   conditioning.  Measured in latent space and in audio space.
2. **Speed.**  Time-to-first-audio (TTFA) for streaming vs one-shot, as a function of utterance
   length.  The streaming window is `context + block + lookahead` compressed frames regardless of
   length, so TTFA should stay roughly flat while the one-shot sampler grows linearly.

The autoencoder and flow model here are trained briefly on synthetic audio purely as a fixture: the
question is whether the *mechanism* works, not whether the model is good.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.audio.mel import MelSpectrogram  # noqa: E402
from parakeet.config import load_config  # noqa: E402
from parakeet.data.features import fit_latent_normalizer  # noqa: E402
from parakeet.data.synthetic import SyntheticSpeechBatchSource, make_corpus  # noqa: E402
from parakeet.data.text import TextTokenizer  # noqa: E402
from parakeet.inference import StreamingPhaseLock, phase_coherence, phase_lock  # noqa: E402
from parakeet.models import build_model  # noqa: E402
from parakeet.models.flow import (  # noqa: E402
    blockwise_sample,
    consistency_sample,
    unfold_time,
    vf_context_frames,
)
from parakeet.train.common import build_optimizer, seed_everything  # noqa: E402
from parakeet.train.stages import flow_step, run_stage  # noqa: E402
from parakeet.inference.synthesize import Synthesizer  # noqa: E402


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(a.reshape(-1)[None], b.reshape(-1)[None]).item()
    )


@torch.no_grad()
def temporal_coupling(
    vf, memory: torch.Tensor, memory_mask: torch.Tensor, shape, t: float = 0.5
) -> dict:
    """How much does the vector field actually mix across time?

    Perturb a single frame and measure the response at nearby and distant frames.  This matters
    because ``layer_scale_init=1e-6`` starts every ConvNeXt branch essentially disabled, so a
    *briefly trained* model can have negligible temporal coupling -- in which case blockwise
    sampling is exact for a trivial reason and proves nothing by itself.
    """
    x = torch.randn(shape)
    tt = torch.full((shape[0],), t)
    base = vf(x, tt, memory, memory_mask)
    perturbed = x.clone()
    perturbed[:, :, 0] += 1.0
    delta = (vf(perturbed, tt, memory, memory_mask) - base).abs().amax(dim=1)
    near = float(delta[:, :3].max().item())
    far = float(delta[:, 4:].max().item()) if delta.shape[-1] > 4 else 0.0
    return {
        "response_at_perturbed_frame": near,
        "response_beyond_4_frames": far,
        "far_over_near": far / max(near, 1e-12),
    }


def _coupled_copy(vf, gamma: float = 0.5):
    """A copy of the vector field with layer scales forced open, for sensitivity testing.

    The scale lives on the inner ConvNeXt block (``FlowBlock.conv.gamma``); with the initial
    ``layer_scale_init=1e-6`` the whole temporal branch is essentially disabled, so a model that
    has barely trained behaves as a *per-frame* function and blockwise sampling is exact for a
    trivial reason.  Opening the scales restores temporal mixing and makes the control meaningful.
    """
    import copy as _copy

    clone = _copy.deepcopy(vf)
    touched = 0
    for blk in clone.blocks:
        inner = getattr(blk, "conv", None)
        gamma_param = getattr(inner, "gamma", None)
        if gamma_param is not None:
            with torch.no_grad():
                gamma_param.fill_(gamma)
            touched += 1
    if touched == 0:
        raise RuntimeError("found no layer-scale parameters to open; the control would be a no-op")
    return clone.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate streaming blockwise sampling")
    ap.add_argument("--config", default="configs/parakeet_small.yaml")
    ap.add_argument("--utterances", type=int, default=16)
    ap.add_argument("--steps-ae", type=int, default=60)
    ap.add_argument("--steps-flow", type=int, default=250)
    ap.add_argument("--steps-sample", type=int, default=2, help="NFE")
    ap.add_argument("--block-frames", type=int, default=16)
    ap.add_argument("--agreement-frames", type=int, default=200,
                    help="compressed frames used for the agreement test (must span many blocks)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048],
                    help="latent frame counts to measure TTFA at")
    ap.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32, 64, 128],
                    help="streaming block sizes for the TTFA/total-compute trade-off sweep")
    ap.add_argument("--out", default="runs/streaming_demo")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps_ae, args.steps_flow = 6, 24
        args.lengths = [64, 128]
        args.agreement_frames = 80

    cfg = load_config(args.config)
    cfg.train.save_every = 0
    cfg.train.log_every = 10**9
    seed_everything(cfg.train.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    corpus = make_corpus(args.utterances, cfg.audio, seed=cfg.train.seed)
    _banner(f"{cfg.name} | {len(corpus)} synthetic utterances | block {args.block_frames} frames")
    t0 = time.perf_counter()

    model = build_model(cfg)
    rf = vf_context_frames(model.vf)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
          f"vector-field temporal context: {rf} frames each side (~"
          f"{rf / (cfg.audio.sample_rate / cfg.audio.hop_length / cfg.flow.compress):.2f}s of audio)")

    # ---------------- fixture ----------------
    wave_source = SyntheticSpeechBatchSource(corpus, batch_size=2, seed=cfg.train.seed)
    run_stage("autoencoder", cfg, model=model, batches=wave_source, max_steps=args.steps_ae, out_dir=str(out))
    fit_latent_normalizer(model.latent_norm, model.autoencoder, wave_source, cfg, max_batches=4)
    mel = MelSpectrogram(cfg.audio)
    tokenizer = TextTokenizer(mode=cfg.text.mode)

    # two utterances of the *same* duration layout, so they can be stacked into one batch
    pair = [u for u in corpus if u.layout == corpus[0].layout][:2]
    wavs = torch.stack([u.wav for u in pair], dim=0)
    log_mel = mel.log_mel(wavs)
    with torch.no_grad():
        latent_target = model.latent_norm.normalize(model.autoencoder.encode(log_mel))
    ids = torch.stack([tokenizer.encode(u.text, add_special=False) for u in pair], dim=0)
    text_mask = torch.ones_like(ids, dtype=torch.bool)

    opt = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    for _ in range(args.steps_flow):
        batch = {
            "ids": ids,
            "text_mask": text_mask,
            "latent": latent_target,
            "ref_mel": log_mel,
            "ref_mask": torch.ones(log_mel.shape[0], log_mel.shape[-1], dtype=torch.bool),
        }
        loss, _ = flow_step(cfg, model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
    model.eval()
    print(f"[fixture] AE {args.steps_ae} steps + flow {args.steps_flow} steps | "
          f"final flow loss {float(loss.detach()):.4f} | {time.perf_counter() - t0:.0f}s elapsed")

    # ---------------- 1. agreement ----------------
    _banner("agreement: blockwise vs full-sequence (identical noise and conditioning)")
    with torch.no_grad():
        memory, memory_mask, _ = model.conditions(ids, text_mask, log_mel)
        t_latent = latent_target.shape[-1]
        # The agreement test must span *many* blocks, otherwise a single block makes blockwise
        # sampling trivially identical to the full run and the test proves nothing.  The vector
        # field's temporal length is independent of the conditioning, so the sequence is forced.
        tc = args.agreement_frames
        shape = (ids.shape[0], cfg.flow.latent_dim * cfg.flow.compress, tc)
        n_blocks = (tc + args.block_frames - 1) // args.block_frames
        print(f"  forcing {tc} compressed frames (~{tc / (cfg.audio.sample_rate / cfg.audio.hop_length / cfg.flow.compress):.1f}s audio) "
              f"= {n_blocks} blocks of {args.block_frames}")
        x0 = torch.randn(shape)
        x0_other = torch.randn(shape)
        full = consistency_sample(model.vf, memory, memory_mask, shape, steps=args.steps_sample,
                                  device=x0.device, cfg_scale=1.0, x0=x0)
        independent = consistency_sample(model.vf, memory, memory_mask, shape, steps=args.steps_sample,
                                         device=x0.device, cfg_scale=1.0, x0=x0_other)

        rows = []
        variants = {
            # production candidates
            "interpolated": dict(context_mode="interpolated"),
            "final": dict(context_mode="final"),
            "no_lookahead": dict(context_mode="interpolated", lookahead=0),
            # positive control: starving the window must *break* agreement on a model that does mix
            # time, otherwise the metric cannot detect boundary error at all
            "starved_context": dict(context_mode="interpolated", context=2, lookahead=2),
        }
        coupling = temporal_coupling(model.vf, memory, memory_mask, shape)
        print(f"  temporal coupling of the trained fixture: response at the perturbed frame "
              f"{coupling['response_at_perturbed_frame']:.2e}, beyond 4 frames "
              f"{coupling['response_beyond_4_frames']:.2e}")
        for name, kw in variants.items():
            t_start = time.perf_counter()
            streamed = blockwise_sample(
                model.vf, memory, memory_mask, shape, steps=args.steps_sample,
                block_frames=args.block_frames, x0=x0, **kw,
            )
            elapsed = time.perf_counter() - t_start
            row = {
                "variant": name,
                "mse_vs_full": float(F.mse_loss(streamed, full).item()),
                "cosine_vs_full": _cos(streamed, full),
                "cosine_independent_draw": _cos(independent, full),
                "seconds": elapsed,
            }
            rows.append(row)
            print(f"  {name:14s} latent MSE {row['mse_vs_full']:.4f} | cosine vs full "
                  f"{row['cosine_vs_full']:+.4f} | independent-draw cosine "
                  f"{row['cosine_independent_draw']:+.4f} | {elapsed:.2f}s")

        # audio-space agreement for the default variant
        def decode(x1c: torch.Tensor) -> torch.Tensor:
            latent = unfold_time(x1c, cfg.flow.compress, t_out=tc * cfg.flow.compress)
            return model.autoencoder.decode(model.latent_norm.denormalize(latent))

        default = blockwise_sample(
            model.vf, memory, memory_mask, shape, steps=args.steps_sample,
            block_frames=args.block_frames, x0=x0, context_mode="interpolated",
        )
        wav_full, wav_block, wav_indep = decode(full), decode(default), decode(independent)
        n = min(wav_full.shape[-1], wav_block.shape[-1], wav_indep.shape[-1])

        def mel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
            return float(F.l1_loss(mel.log_mel(a[..., :n]), mel.log_mel(b[..., :n])).item())

        audio = {
            "mel_l1_block_vs_full": mel_l1(wav_block, wav_full),
            "mel_l1_independent_vs_full": mel_l1(wav_indep, wav_full),
            "waveform_cosine_block_vs_full": _cos(wav_block[..., :n], wav_full[..., :n]),
            "waveform_cosine_independent_vs_full": _cos(wav_indep[..., :n], wav_full[..., :n]),
        }
        for k, v in audio.items():
            print(f"  {k}: {v:+.4f}")

        # ---- sensitivity: does the metric detect boundary error on a model that *does* mix time?
        coupled = _coupled_copy(model.vf, gamma=0.5)
        coupling_coupled = temporal_coupling(coupled, memory, memory_mask, shape)
        full_coupled = consistency_sample(
            coupled, memory, memory_mask, shape, steps=args.steps_sample, cfg_scale=1.0, x0=x0
        )
        sensitivity = []
        print(f"  positive control on a layer-scale-opened copy: response at the perturbed frame "
              f"{coupling_coupled['response_at_perturbed_frame']:.3f}, beyond 4 frames "
              f"{coupling_coupled['response_beyond_4_frames']:.3f}")
        for name, kw in variants.items():
            streamed_c = blockwise_sample(
                coupled, memory, memory_mask, shape, steps=args.steps_sample,
                block_frames=args.block_frames, x0=x0, **kw,
            )
            entry = {
                "variant": name,
                "mse_vs_full": float(F.mse_loss(streamed_c, full_coupled).item()),
                "cosine_vs_full": _cos(streamed_c, full_coupled),
            }
            sensitivity.append(entry)
            print(f"    {name:16s} latent MSE {entry['mse_vs_full']:.4f} | cosine "
                  f"{entry['cosine_vs_full']:+.4f}")

    # ---------------- 2. TTFA scaling ----------------
    _banner("time to first audio: blockwise streaming vs one-shot")
    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=False)
    timings = []
    for n_latent in args.lengths:
        text = "the quick brown fox jumps over the lazy dog"
        t0 = time.perf_counter()
        with torch.no_grad():
            one_shot = synth.synthesize(text, steps=args.steps_sample, n_latent_frames=n_latent)
        ttfa_one_shot = time.perf_counter() - t0

        t0 = time.perf_counter()
        chunks = []
        for chunk in synth.synthesize_stream(text, chunk_frames=args.block_frames,
                                             steps=args.steps_sample, n_latent_frames=n_latent):
            chunks.append(chunk)
            break  # first chunk only -> that is the time-to-first-audio
        ttfa_stream = time.perf_counter() - t0
        first_chunk_samples = chunks[0].shape[0] if chunks else 0

        t0 = time.perf_counter()
        all_chunks = list(
            synth.synthesize_stream(text, chunk_frames=args.block_frames,
                                    steps=args.steps_sample, n_latent_frames=n_latent)
        )
        total_stream = time.perf_counter() - t0
        total_samples = sum(c.shape[0] for c in all_chunks)

        row = {
            "n_latent_frames": n_latent,
            "audio_seconds": n_latent / (cfg.audio.sample_rate / cfg.audio.hop_length),
            "ttfa_one_shot_s": ttfa_one_shot,
            "ttfa_stream_s": ttfa_stream,
            "ttfa_speedup": ttfa_one_shot / max(ttfa_stream, 1e-9),
            "first_chunk_samples": first_chunk_samples,
            "n_chunks": len(all_chunks),
            "total_stream_s": total_stream,
            "total_samples": total_samples,
            "one_shot_samples": one_shot.shape[-1],
        }
        timings.append(row)
        print(f"  {n_latent:5d} latent frames ({row['audio_seconds']:.2f}s audio): "
              f"TTFA one-shot {ttfa_one_shot*1000:7.1f} ms | streaming {ttfa_stream*1000:6.1f} ms | "
              f"{row['ttfa_speedup']:5.2f}x | {row['n_chunks']:3d} chunks | "
              f"total {total_stream:.3f}s vs {ttfa_one_shot:.3f}s")

    # ---------------- 3. block-size trade-off ----------------
    _banner("streaming trade-off: block size vs TTFA vs total compute (longest utterance)")
    longest = args.lengths[-1]
    text = "the quick brown fox jumps over the lazy dog"
    t0 = time.perf_counter()
    with torch.no_grad():
        one_shot_long = synth.synthesize(text, steps=args.steps_sample, n_latent_frames=longest)
    one_shot_total = time.perf_counter() - t0
    sweep = []
    for block in args.block_sizes:
        t0 = time.perf_counter()
        first = next(iter(synth.synthesize_stream(text, chunk_frames=block, steps=args.steps_sample,
                                                  n_latent_frames=longest)), None)
        ttfa = time.perf_counter() - t0
        t0 = time.perf_counter()
        chunks = list(synth.synthesize_stream(text, chunk_frames=block, steps=args.steps_sample,
                                              n_latent_frames=longest))
        total = time.perf_counter() - t0
        sweep.append({
            "block_frames": block,
            "ttfa_s": ttfa,
            "ttfa_vs_one_shot": one_shot_total / max(ttfa, 1e-9),
            "total_s": total,
            "total_compute_multiplier": total / max(one_shot_total, 1e-9),
            "n_chunks": len(chunks),
            "first_chunk_samples": int(first.shape[0]) if first is not None else 0,
            "total_samples": sum(int(c.shape[0]) for c in chunks),
        })
        print(f"  block {block:4d}: TTFA {ttfa*1000:7.1f} ms ({one_shot_total/max(ttfa,1e-9):5.2f}x vs "
              f"one-shot {one_shot_total*1000:.1f} ms) | total {total:5.3f}s "
              f"({total/max(one_shot_total,1e-9):4.2f}x one-shot compute) | {len(chunks)} chunks")
    print("  (streaming re-samples the context/lookahead band for every block, so it trades total "
          "compute for latency; larger blocks reduce that overhead)")

    # ---------------- 4. chunked phase lock ----------------
    _banner("chunked phase-lock filter vs offline filter")
    with torch.no_grad():
        wav = wav_full
        offline = phase_lock(
            wav, sample_rate=cfg.audio.sample_rate, n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
        )
        streamer = StreamingPhaseLock(
            sample_rate=cfg.audio.sample_rate, n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
        )
        pieces = []
        step = 2048
        for i in range(0, wav.shape[-1], step):
            filtered = streamer.push(wav[..., i : i + step])
            if filtered.shape[-1]:
                pieces.append(filtered)
        pieces.append(streamer.flush())
        chunked = torch.cat(pieces, dim=-1)
    m = min(offline.shape[-1], chunked.shape[-1])
    phase = {
        "max_abs_diff": float((offline[..., :m] - chunked[..., :m]).abs().max().item()),
        "waveform_cosine": _cos(offline[..., :m], chunked[..., :m]),
        "coherence_offline": float(phase_coherence(offline).item()),
        "coherence_chunked": float(phase_coherence(chunked).item()),
    }
    print(f"  chunked vs offline: max|diff| {phase['max_abs_diff']:.2e} | cosine "
          f"{phase['waveform_cosine']:+.4f} | coherence {phase['coherence_offline']:.4f} vs "
          f"{phase['coherence_chunked']:.4f}")

    # ---------------- report ----------------
    def row_for(name: str) -> dict:
        return next(r for r in rows if r["variant"] == name)

    def sens_for(name: str) -> dict:
        return next(r for r in sensitivity if r["variant"] == name)

    production = [row_for(n) for n in ("interpolated", "final", "no_lookahead")]
    best_ttfa = max(sweep, key=lambda r: r["ttfa_vs_one_shot"])
    checks = {
        # production variants must not be measurably worse than the one-shot sampler, judged
        # against the model's own sampling variability (an independent draw)
        "blockwise_within_sampling_variability": all(
            r["cosine_vs_full"] > r["cosine_independent_draw"] for r in production
        ),
        # and the measurement must be able to detect boundary error at all: on a model whose layer
        # scales are opened (so the temporal branch is active), removing the right-hand lookahead
        # must visibly degrade agreement with the full-sequence sampler
        "positive_control_detects_error": sens_for("no_lookahead")["mse_vs_full"]
        > 3.0 * max(sens_for("interpolated")["mse_vs_full"], 1e-12),
        "ttfa_improves_with_length": best_ttfa["ttfa_vs_one_shot"] > 1.5,
        "streaming_length_matches_one_shot": abs(timings[-1]["total_samples"] - timings[-1]["one_shot_samples"])
        <= cfg.audio.hop_length * cfg.flow.compress,
    }
    # `--quick` runs toy-length utterances whose TTFA is dominated by fixed per-call overhead, so
    # that criterion is reported but not enforced there (quick mode validates the harness only).
    enforced = {
        k: v for k, v in checks.items() if not (args.quick and k == "ttfa_improves_with_length")
    }
    report = {
        "config": args.config,
        "vf_context_frames": rf,
        "block_frames": args.block_frames,
        "agreement_frames": tc,
        "nfe": args.steps_sample,
        "flow_loss_final": float(loss.detach()),
        "temporal_coupling": coupling,
        "temporal_coupling_opened_layer_scales": coupling_coupled,
        "agreement": rows,
        "audio_agreement": audio,
        "sensitivity_opened_layer_scales": sensitivity,
        "ttfa": timings,
        "block_size_sweep": sweep,
        "phase_lock": phase,
        "checks": checks,
        "seconds_total": time.perf_counter() - t0,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _banner("RESULT")
    for name, ok in checks.items():
        note = " (informational in --quick)" if name not in enforced else ""
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{note}")
    print(f"\nreport -> {out/'report.json'}")
    print("STREAMING DEMO " + ("PASSED" if all(enforced.values()) else "FAILED"))
    return 0 if all(enforced.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
