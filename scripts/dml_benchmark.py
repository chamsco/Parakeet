"""Benchmark a real training step on CPU versus the DirectML GPU.

    .venv-dml\\Scripts\\python.exe scripts/dml_benchmark.py --cache runs/mixed_train/latent_cache

The project has trained on CPU since round 1 -- 1600 steps took about an hour, which is what made every
experiment expensive (round 28 even found the model was undertrained at 5.5 passes, because that *was*
the affordable budget).  There is a Radeon RX 6950 XT in this machine.  On Windows its route into
PyTorch is DirectML, and DirectML does not implement every op a TorchScript-free audio model uses, so
this measures the step that actually matters (`distill-audio`, the mel/spectral/aux objective) rather
than assuming: it times real forward+backward passes and reports which op fails, if any.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.data.dataset import LatentShardDataset, collate  # noqa: E402
from parakeet.models import build_model  # noqa: E402
from parakeet.audio.mel import MelSpectrogram  # noqa: E402
from parakeet.train.losses import (  # noqa: E402
    DistillSignalWeights,
    LogMelLoss,
    MultiResolutionSTFTLoss,
    TextSideDistillLoss,
)


def resolve(device_name: str) -> torch.device:
    if device_name == "dml":
        import torch_directml  # type: ignore

        return torch_directml.device()
    return torch.device(device_name)


def make_batch(cache: Path, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    dataset = LatentShardDataset(cache)
    items = [dataset[i] for i in range(batch_size)]
    batch = collate(items)
    return {key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()}


def run_steps(args) -> Dict:
    cfg = load_config(args.config)
    cfg.autoencoder.latent_rate = args.latent_rate
    # the cache knows how many voices it holds; the config alone would index an embedding out of range
    from parakeet.train.common import derive_n_voices_from_cache

    cfg.n_voices = derive_n_voices_from_cache(args.cache)
    device = resolve(args.device)
    torch.manual_seed(0)
    model = build_model(cfg).to(device)
    for parameter in model.autoencoder.parameters():
        parameter.requires_grad_(False)
    model.train()

    criterion = TextSideDistillLoss(DistillSignalWeights(latent_contrast=args.contrast))
    mel = MelSpectrogram(cfg.audio).to(device)
    # DirectML has no complex dtype (`Invalid or unsupported data type ComplexFloat`) and both the mel
    # and the multi-resolution STFT losses are built on torch.stft, so they run on the CPU while the
    # model runs on the GPU.  The tensors crossing that boundary are the rendered and target audio.
    loss_device = device  # the real-valued STFT path keeps every graph on one device
    spectral = MultiResolutionSTFTLoss().to(loss_device)
    criterion = criterion.to(loss_device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    batch = make_batch(Path(args.cache), args.batch_size, device)

    timings: List[float] = []
    losses: List[float] = []
    for step in range(args.steps):
        start = time.perf_counter()
        try:
            from parakeet.train.stages import text_audio_step

            total, logs, _recon = text_audio_step(
                cfg, model, batch, criterion=criterion,
                losses={"mel": LogMelLoss(mel), "spectral": spectral},
                loss_device=loss_device,
            )
        except Exception as exc:  # noqa: BLE001 - the point is to report what fails
            return {"device": args.device, "error": f"{type(exc).__name__}: {exc}"}
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        if device.type != "cpu" and hasattr(device, "type"):
            pass
        if args.device == "dml":
            import torch_directml  # type: ignore

            torch_directml.synchronize(device)
        timings.append(time.perf_counter() - start)
        losses.append(float(total))
    warm = timings[1:] if len(timings) > 1 else timings
    return {
        "device": args.device,
        "steps": args.steps,
        "seconds_per_step": sum(warm) / len(warm),
        "audio_seconds_per_step": float(batch["wav"].shape[-1] / cfg.audio.sample_rate),
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU vs DirectML training-step benchmark")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--cache", default="runs/mixed_train/latent_cache")
    ap.add_argument("--latent-rate", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--devices", default="cpu,dml")
    ap.add_argument("--losses-on-cpu", action="store_true", default=False,
                    help="run the STFT-based losses on the CPU: DirectML has no complex dtype")
    ap.add_argument("--out", default="runs/dml_benchmark.json")
    args = ap.parse_args()

    results = []
    for device_name in args.devices.split(","):
        device_name = device_name.strip()
        if not device_name:
            continue
        try:
            results.append(run_steps(argparse.Namespace(**{**vars(args), "device": device_name})))
            print(f"  {device_name}: {results[-1]}")
        except Exception as exc:  # noqa: BLE001
            results.append({"device": device_name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {device_name}: FAILED - {type(exc).__name__}: {exc}")

    cpu = next((r for r in results if r["device"] == "cpu" and "error" not in r), None)
    gpu = next((r for r in results if r["device"] != "cpu" and "error" not in r), None)
    report = {
        "results": results,
        "speedup": (cpu["seconds_per_step"] / gpu["seconds_per_step"]) if cpu and gpu else None,
        "note": (
            "measured on the real distill-audio step, so an op DirectML cannot run shows up as an "
            "error rather than as optimistic latency"
        ),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["speedup"]:
        print(f"  speedup: {report['speedup']:.2f}x  ({cpu['seconds_per_step']:.3f}s -> "
              f"{gpu['seconds_per_step']:.3f}s per step)")
    print(f"report -> {args.out}")
    return 0 if cpu and gpu else 1


if __name__ == "__main__":
    raise SystemExit(main())
