"""Export the Parakeet vocoder to int8 ONNX and benchmark it against PyTorch.

    python scripts/export_onnx.py --config configs/parakeet_tiny.yaml
    python scripts/export_onnx.py --config configs/parakeet_tiny.yaml --checkpoint runs/.../autoencoder_last.pt

Reports, for the decoder compute:
  * fp32 ONNX latency vs int8 ONNX latency (single thread, CPU)
  * model size before/after int8
  * the actual output deviation introduced by int8 (so size/speed is judged against fidelity)
  * PyTorch decode latency on the same latents, for the "PyTorch vs ONNX" comparison Paradee
    reports (25.0x vs 17.8x real time)

The latents used are synthetic unless a checkpoint is supplied; with a randomly-initialised model
the *timings* are architecture-representative (convolutions dominate) but the deviation numbers are
not meaningful as a quality statement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.audio.mel import MelSpectrogram  # noqa: E402
from parakeet.config import load_config  # noqa: E402
from parakeet.data.synthetic import make_corpus  # noqa: E402
from parakeet.inference import write_wav  # noqa: E402
from parakeet.inference.onnx_export import OnnxVocoder, compare_fp32_int8  # noqa: E402
from parakeet.models import build_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Export + benchmark int8 ONNX vocoder")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="runs/onnx")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--pipeline", action="store_true",
                    help="export text side + vocoder and benchmark the full ONNX pipeline")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg).eval()
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("ema", {}).get("shadow", payload["model"])
        model.load_state_dict(state, strict=False)
        print(f"loaded checkpoint {args.checkpoint}")
    else:
        print("WARNING: random weights -- timings are representative, quality deviation is not")

    torch.set_num_threads(args.threads)
    out = Path(args.out)

    if args.pipeline:
        from parakeet.inference.onnx_export import compare_pipelines

        texts = [
            "the quick brown fox jumps over the lazy dog",
            "parakeet is a small and fast text to speech model",
            "real time factor is the number that matters on a laptop",
        ]
        print(f"{cfg.name}: full pipeline export (text side + vocoder), fp32 and int8")
        t0 = time.perf_counter()
        report = compare_pipelines(
            model, cfg, out, texts, runs=args.runs, threads=args.threads, opset=args.opset
        )
        report["config"] = args.config
        report["checkpoint"] = args.checkpoint
        report["export_seconds"] = time.perf_counter() - t0
        (out / "pipeline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"\n{'=' * 78}\nfull Tiny pipeline ({args.threads} thread(s), "
              f"mean {report['mean_audio_seconds']:.3f}s audio)\n{'=' * 78}")
        print(f"  PyTorch fp32    {report['mean_ms']['torch']:8.2f} ms  "
              f"{report['torch_x_realtime']:7.2f}x real time")
        print(f"  ONNX fp32       {report['mean_ms']['onnx_fp32']:8.2f} ms  "
              f"{report['mean_audio_seconds']/(report['mean_ms']['onnx_fp32']/1000):7.2f}x real time")
        print(f"  ONNX int8       {report['mean_ms']['onnx_int8']:8.2f} ms  "
              f"{report['int8_x_realtime']:7.2f}x real time   "
              f"({report['int8_vs_torch_speedup']:.2f}x vs PyTorch)")
        print(f"  ONNX int8 + phase lock (shipped) {report['mean_ms']['onnx_int8_shipped']:6.2f} ms  "
              f"{report['int8_shipped_x_realtime']:7.2f}x real time   "
              f"({report['int8_shipped_vs_torch_speedup']:.2f}x vs PyTorch)")
        print(f"  total size: {report['total_fp32_mb']:.2f} -> {report['total_int8_mb']:.2f} MB "
              f"(text side {report['text_side_mb']['fp32']:.2f} -> {report['text_side_mb']['int8']:.2f} MB"
              f" | vocoder {report['vocoder_mb']['fp32']:.2f} -> {report['vocoder_mb']['int8']:.2f} MB)")
        print(f"  int8 vs PyTorch equivalence: mel L1 {report['int8_vs_torch_mel_l1']:.4f}, "
              f"waveform cosine {report['int8_vs_torch_waveform_cosine']:.4f}")
        print(f"\nreport -> {out/'pipeline_report.json'}")
        return 0

    corpus = make_corpus(max(4, args.runs // 2), cfg.audio, seed=0)
    mel = MelSpectrogram(cfg.audio)
    calibration = [mel.log_mel(u.wav[None]) for u in corpus]

    print(f"{cfg.name}: exporting decoder compute (latent_dim={cfg.autoencoder.latent_dim})")
    t0 = time.perf_counter()
    report = compare_fp32_int8(model, cfg.audio, out, calibration, runs=args.runs, opset=args.opset)
    export_seconds = time.perf_counter() - t0

    # PyTorch reference latency on the same latents
    with torch.no_grad():
        cal_latents = [model.autoencoder.encode(m) for m in calibration]
        model.autoencoder.decode(cal_latents[0])  # warm-up
        t0 = time.perf_counter()
        for _ in range(args.runs):
            for lat in cal_latents:
                wav_t = model.autoencoder.decode(lat)
        torch_ms = (time.perf_counter() - t0) / (args.runs * len(cal_latents)) * 1000.0
    audio_seconds = wav_t.shape[-1] / cfg.audio.sample_rate
    torch_rtf = torch_ms / 1000.0 / max(audio_seconds, 1e-9)

    voc = OnnxVocoder(out / "vocoder_int8.onnx", cfg.audio, intra_op_threads=args.threads)
    wav = voc.decode(cal_latents[0].numpy())
    write_wav(out / "onnx_int8_decode.wav", wav, cfg.audio.sample_rate)

    report.update(
        {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "threads": args.threads,
            "params": sum(p.numel() for p in model.parameters()),
            "pytorch": {
                "mean_ms": torch_ms,
                "rtf": torch_rtf,
                "x_realtime": 1.0 / max(torch_rtf, 1e-9),
                "audio_seconds": audio_seconds,
            },
            "export_seconds": export_seconds,
            "onnx_vs_pytorch_speedup": torch_ms / max(report["fp32"]["mean_ms"], 1e-9),
            "onnx_int8_vs_pytorch_speedup": torch_ms / max(report["int8"]["mean_ms"], 1e-9),
        }
    )
    (out / "onnx_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'=' * 78}\ndecoder compute benchmark ({args.threads} thread(s), "
          f"{report['fp32']['latent_frames']:.0f} latent frames = "
          f"{audio_seconds:.3f}s audio)\n{'=' * 78}")
    print(f"  PyTorch fp32       {torch_ms:8.2f} ms   {1/max(torch_rtf,1e-9):7.2f}x real time")
    print(f"  ONNX fp32          {report['fp32']['mean_ms']:8.2f} ms   "
          f"{report['fp32']['model_mb']:7.2f} MB   {report['onnx_vs_pytorch_speedup']:.2f}x vs PyTorch")
    print(f"  ONNX int8 (QDQ)    {report['int8']['mean_ms']:8.2f} ms   "
          f"{report['int8']['model_mb']:7.2f} MB   {report['onnx_int8_vs_pytorch_speedup']:.2f}x vs PyTorch")
    print(f"  int8 size reduction {report['size_reduction_x']:.2f}x | "
          f"int8 speedup vs ONNX fp32 {report['speedup_x']:.2f}x")
    print(f"  int8 deviation: |dlog_mag|max {report['int8_max_log_mag_deviation']:.4f}, "
          f"|dphase|max {report['int8_max_phase_deviation_rad']:.4f} rad")
    print(f"\nreport -> {out/'onnx_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
