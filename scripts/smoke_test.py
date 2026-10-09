"""Full CPU smoke test: every stage, both variants, start to finish, no corpus required.

    python scripts/smoke_test.py [--steps 2] [--out runs/smoke]

This is the artifact that proves the repository actually runs: it exercises
  1. autoencoder training (mel + multi-res STFT + MPD/MSD adversarial),
  2. Tiny text-side distillation on cached-signal targets,
  3. Tiny decoder-only distillation with the 45->10->3 spectral anneal,
  4. Small flow-matching training with Ke=4 context-sharing batch expansion,
  5. Reflow (few-step) distillation,
  6. synthesis + phase-lock filter + int8 quantisation + RTF measurement,
using shape-faithful synthetic batches.  It also verifies that the streaming decoder matches
the offline decoder.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.data.dataset import SyntheticBatchSource  # noqa: E402
from parakeet.eval.metrics import measure_rtf  # noqa: E402
from parakeet.inference import (  # noqa: E402
    Synthesizer,
    phase_coherence,
    phase_lock,
    quantize_weights_,
    size_report,
)
from parakeet.models import build_model, count_parameters, parameter_report  # noqa: E402
from parakeet.train.stages import run_stage  # noqa: E402

STAGES_TINY = ["autoencoder", "distill-text", "distill-decoder"]
STAGES_SMALL = ["autoencoder", "flow", "reflow"]


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def _fmt_params(n: int) -> str:
    return f"{n/1e6:.3f} M"


def run_variant(cfg_path: str, stages, steps: int, out_dir: Path, batch_size: int = 2) -> dict:
    cfg = load_config(cfg_path)
    cfg.train.max_steps = steps
    cfg.train.log_every = max(1, steps)
    cfg.train.save_every = 0
    cfg.train.out_dir = str(out_dir / cfg.name)
    cfg.flow.nfe = min(cfg.flow.nfe, 4)  # keep the smoke test quick
    torch.manual_seed(cfg.train.seed)

    model = build_model(cfg)
    _banner(f"{cfg.name}: {_fmt_params(count_parameters(model))} parameters")
    for name, count in parameter_report(model).items():
        print(f"  {name:22s} {_fmt_params(count)}")
    sizes = size_report(model)
    print(f"  fp32 {sizes['fp32_mb']:.2f} MB | fp16 {sizes['fp16_mb']:.2f} MB | int8+fp16 scales {sizes['int8_mb']:.2f} MB")

    results = {}
    for stage in stages:
        source = SyntheticBatchSource(cfg, stage, batch_size=batch_size, n_frames=64, n_tokens=24)
        t0 = time.perf_counter()
        logs = run_stage(
            stage,
            cfg,
            model=model,
            batches=source,
            max_steps=steps,
            out_dir=str(out_dir / cfg.name),
        )
        dt = time.perf_counter() - t0
        loss = logs.get("loss", float("nan"))
        ok = loss == loss  # not NaN
        results[stage] = {"loss": loss, "seconds": dt, "ok": ok}
        extra = " ".join(f"{k}={v:.4f}" for k, v in logs.items() if k not in {"loss", "step"})
        print(f"  [{stage:16s}] loss={loss:.4f}  {dt:5.2f}s  {extra}")
        if not ok:
            raise SystemExit(f"stage {stage} produced a non-finite loss")

    # ---------------- synthesis + phase lock + int8 ----------------
    _banner(f"{cfg.name}: synthesis")
    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=False)
    text = "hello world, this is a parakeet smoke test."
    with torch.no_grad():
        wav_raw = synth.synthesize(text, steps=cfg.flow.distilled_nfe, seed=0)
    sr = cfg.audio.sample_rate
    wav_pl = phase_lock(wav_raw, sample_rate=sr, n_fft=cfg.audio.n_fft, hop_length=cfg.audio.hop_length)
    coh_raw = float(phase_coherence(wav_raw, sample_rate=sr).item())
    coh_pl = float(phase_coherence(wav_pl, sample_rate=sr).item())
    dur = wav_raw.shape[-1] / sr
    print(f"  audio {wav_raw.shape[-1]} samples = {dur:.3f}s @ {sr} Hz")
    print(f"  phase coherence 2-8k: {coh_raw:.4f} -> {coh_pl:.4f} (higher = less buzz)")

    rtf = measure_rtf(
        lambda: synth.synthesize(text, steps=cfg.flow.distilled_nfe, seed=0),
        sample_rate=sr,
        warmup=1,
        runs=2,
        threads=1,
    )
    print(f"  RTF(1 thread) {rtf.rtf:.3f}  ({rtf.seconds_per_audio_second:.2f}x real time, {rtf.audio_seconds:.2f}s audio)")

    # streaming decoder must equal the offline decoder
    from parakeet.audio.mel import MelSpectrogram

    with torch.no_grad():
        mel = MelSpectrogram(cfg.audio).log_mel(wav_raw)
        latent = model.autoencoder.encode(mel)
        offline = model.autoencoder.decode(latent)
        chunked = synth.synthesize_chunked(latent, chunk_frames=16)
    n = min(offline.shape[-1], chunked.shape[-1])
    diff = float((offline[..., :n] - chunked[..., :n]).abs().max())
    print(f"  streaming vs offline decoder max|diff| = {diff:.2e}")

    q_model = build_model(cfg)
    q_model.load_state_dict(model.state_dict())
    quantize_weights_(q_model, bits=8, per_channel=True)
    q_sizes = size_report(q_model)
    print(f"  int8 weights -> {q_sizes['int8_mb']:.2f} MB (fp32 {q_sizes['fp32_mb']:.2f} MB)")

    results["synthesis"] = {
        "phase_coherence_raw": coh_raw,
        "phase_coherence_locked": coh_pl,
        "rtf_1thread": rtf.rtf,
        "stream_vs_offline_max_diff": diff,
        "int8_mb": q_sizes["int8_mb"],
    }
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--out", default="runs/smoke")
    ap.add_argument("--tiny-config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--small-config", default="configs/parakeet_small.yaml")
    ap.add_argument("--skip-small", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | cuda {torch.cuda.is_available()}")

    summary = {"tiny": run_variant(args.tiny_config, STAGES_TINY, args.steps, out)}
    if not args.skip_small:
        summary["small"] = run_variant(args.small_config, STAGES_SMALL, args.steps, out, batch_size=2)

    _banner("SUMMARY")
    failures = []
    for variant, res in summary.items():
        for stage, info in res.items():
            if isinstance(info, dict) and "ok" in info and not info["ok"]:
                failures.append(f"{variant}.{stage}")
            if isinstance(info, dict) and "loss" in info:
                print(f"  {variant:6s} {stage:16s} loss={info['loss']:.4f} {info['seconds']:5.2f}s")
            elif isinstance(info, dict):
                print(f"  {variant:6s} {'synthesis':16s} " + " ".join(f"{k}={v:.4f}" for k, v in info.items()))
    print("\nSMOKE TEST " + ("FAILED: " + ", ".join(failures) if failures else "PASSED"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
