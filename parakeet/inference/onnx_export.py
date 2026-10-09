"""ONNX export and int8 quantisation for the Parakeet vocoder.

Why this exists: the "lightning fast on a laptop" claim is a *deployment* claim, and PyTorch is not
the deployment target.  Paradee reports 25.0x real time in PyTorch on one CPU thread but 17.8x via
ONNX — the numbers move with the runtime, so both have to be measured.

Scope: the exported graph is the **decoder compute** (`from_latent` -> causal ConvNeXt blocks ->
head -> log-magnitude/phase).  Real-valued outputs only, with a dynamic time axis.  The iSTFT
overlap-add stays outside the graph: it is cheap (an FFT per frame) and exporting
complex/`torch.stft`/`torch.istft` through ONNX is fragile, whereas the convolutions are the cost.

Int8: ``quantize_static`` (QDQ format, per-channel, int8 weights / uint8 activations) with a
calibration set drawn from real latents.  Note that ONNX Runtime's int8 *Conv* kernels need VNNI-era
CPU support to actually beat fp32 — so :func:`benchmark` measures rather than assumes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio.istft import OLAISTFT
from ..config import AudioConfig


class _DecoderHead(nn.Module):
    """ONNX-friendly view of the autoencoder decoder: real tensors in, real tensors out."""

    def __init__(self, autoencoder: nn.Module) -> None:
        super().__init__()
        self.ae = autoencoder

    def forward(self, latent: torch.Tensor):
        h = self.ae.from_latent(latent)
        h = self.ae.decoder(h)
        out = self.ae.head(h)
        log_mag, phase = out.chunk(2, dim=1)
        return log_mag, phase


def export_onnx(
    model: nn.Module,
    audio: AudioConfig,
    path: str | Path,
    opset: int = 17,
    latent_frames: int = 24,
    dynamo: bool = False,
) -> Path:
    """Export the decoder to ONNX with dynamic batch and time axes.

    ``dynamo=False`` selects the legacy TorchScript exporter, which needs only ``onnx`` and emits
    ops our ConvNeXt/LayerNorm stack maps onto cleanly.  PyTorch 2.9 also offers the new
    ``torch.export``-based exporter (``dynamo=True``), which additionally requires ``onnxscript``;
    switch if you need it, but verify the graph numerics either way (``tests/test_onnx.py``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _DecoderHead(model.autoencoder).eval()
    dummy = torch.randn(1, model.autoencoder.latent_dim, latent_frames)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(path),
        input_names=["latent"],
        output_names=["log_mag", "phase"],
        dynamic_axes={
            "latent": {0: "batch", 2: "frames"},
            "log_mag": {0: "batch", 2: "frames"},
            "phase": {0: "batch", 2: "frames"},
        },
        opset_version=opset,
        dynamo=dynamo,
    )
    return path


def quantize_int8(
    src: str | Path,
    dst: str | Path,
    calibration_latents: Sequence[np.ndarray],
    per_channel: bool = True,
) -> Path:
    """Static QDQ int8 quantisation with a calibration set of real latents."""
    from onnxruntime.quantization import (  # type: ignore
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class _Reader(CalibrationDataReader):
        def __init__(self, latents: Sequence[np.ndarray]) -> None:
            self._items = [{"latent": np.asarray(x, dtype=np.float32)} for x in latents]
            self._i = 0

        def get_next(self):  # noqa: D102
            if self._i >= len(self._items):
                return None
            item = self._items[self._i]
            self._i += 1
            return item

        def rewind(self) -> None:  # noqa: D102
            self._i = 0

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(src),
        str(dst),
        _Reader(calibration_latents),
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    return dst


class OnnxVocoder:
    """Runtime wrapper: ONNX decoder compute + torch overlap-add iSTFT (streaming-capable)."""

    def __init__(
        self,
        onnx_path: str | Path,
        audio: AudioConfig,
        providers: Optional[Sequence[str]] = None,
        intra_op_threads: int = 1,
    ) -> None:
        import onnxruntime as ort  # type: ignore

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=list(providers or ["CPUExecutionProvider"])
        )
        self.audio = audio
        self.istft = OLAISTFT(audio.n_fft, audio.hop_length, audio.win_length, center=True)
        self.input_name = self.session.get_inputs()[0].name

    @property
    def providers(self) -> List[str]:
        return self.session.get_providers()

    def spectrogram(self, latent: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = latent if isinstance(latent, np.ndarray) else latent.detach().cpu().numpy()
        log_mag, phase = self.session.run(None, {self.input_name: np.asarray(x, dtype=np.float32)})
        mag = torch.exp(torch.from_numpy(log_mag).clamp(max=8.0))
        return torch.polar(mag, torch.from_numpy(phase))

    def decode(self, latent: np.ndarray | torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        return self.istft(self.spectrogram(latent), length=length)


def benchmark(
    onnx_path: str | Path,
    audio: AudioConfig,
    latents: Sequence[np.ndarray],
    runs: int = 10,
    threads: int = 1,
) -> Dict[str, float]:
    """Mean latency of one decode pass, plus model size.  Measures; does not assume."""
    voc = OnnxVocoder(onnx_path, audio, intra_op_threads=threads)
    frames = int(np.asarray(latents[0]).shape[-1])
    for latent in latents[:1]:  # warm-up
        voc.decode(latent)
    times: List[float] = []
    for _ in range(runs):
        for latent in latents:
            t0 = time.perf_counter()
            wav = voc.decode(latent)
            times.append((time.perf_counter() - t0) * 1000.0)
    audio_seconds = wav.shape[-1] / audio.sample_rate
    mean_ms = float(np.mean(times))
    return {
        "mean_ms": mean_ms,
        "audio_seconds": audio_seconds,
        "rtf_fixed_frames": mean_ms / 1000.0 / max(audio_seconds, 1e-9),
        "latent_frames": float(frames),
        "model_bytes": float(Path(onnx_path).stat().st_size),
        "model_mb": Path(onnx_path).stat().st_size / 1e6,
        "providers": ",".join(voc.providers),
    }


def compare_fp32_int8(
    model: nn.Module,
    audio: AudioConfig,
    out_dir: str | Path,
    calibration_mels: Iterable[torch.Tensor],
    runs: int = 10,
    opset: int = 17,
) -> Dict[str, object]:
    """End-to-end: export, quantise, benchmark both, and report the quality drift.

    Returns a dict suitable for JSON dumping, including the max absolute spectrogram deviation
    introduced by int8 so the size/speed change can be judged against a real fidelity number.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = export_onnx(model, audio, out_dir / "vocoder_fp32.onnx", opset=opset)

    with torch.no_grad():
        cal_latents = [model.autoencoder.encode(m).numpy().astype(np.float32) for m in calibration_mels]
    int8_path = quantize_int8(fp32_path, out_dir / "vocoder_int8.onnx", cal_latents)

    fp32 = benchmark(fp32_path, audio, cal_latents, runs=runs)
    int8 = benchmark(int8_path, audio, cal_latents, runs=runs)

    with torch.no_grad():
        ref = _DecoderHead(model.autoencoder).eval()(torch.from_numpy(cal_latents[0]))
        voc = OnnxVocoder(int8_path, audio)
        log_mag, phase = voc.session.run(None, {voc.input_name: cal_latents[0]})
        dev_mag = float(np.abs(log_mag - ref[0].numpy()).max())
        dev_phase = float(np.abs(phase - ref[1].numpy()).max())

    return {
        "fp32": fp32,
        "int8": int8,
        "size_reduction_x": fp32["model_mb"] / max(int8["model_mb"], 1e-9),
        "speedup_x": fp32["mean_ms"] / max(int8["mean_ms"], 1e-9),
        "int8_max_log_mag_deviation": dev_mag,
        "int8_max_phase_deviation_rad": dev_phase,
    }
