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
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio.istft import OLAISTFT
from ..config import AudioConfig
from ..models.duration import normalized_to_durations


def onnx_artifact_bytes(path: str | Path) -> int:
    """Total bytes of an ONNX artifact, including an external-data sidecar if one exists.

    The ``torch.export``-based exporter writes large weight sets to ``<name>.onnx.data`` by
    default, so reporting ``path.stat().st_size`` alone understates the model by ~40x.  Every size
    this module reports goes through here.
    """
    path = Path(path)
    total = path.stat().st_size
    for sidecar in path.parent.glob(path.name + ".data"):
        total += sidecar.stat().st_size
    return total


def _quiet_export_logs() -> None:
    """The dynamo exporter emits very verbose ``torch.__trace`` DEBUG lines."""
    import logging

    for name in ("torch.__trace", "torch.onnx", "onnxscript"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _ensure_utf8_stdout() -> None:
    """The dynamo exporter prints emoji, which crashes on a cp1252 Windows console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - non-reconfigurable streams
            pass


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


def onnx_input_spec(path: str | Path) -> tuple[str, np.dtype]:
    """Read the first graph input's name and numpy dtype.

    Quantisation calibration feeds the graph directly, so the input key and dtype have to match
    the model (token ids are int64, latents are float32).  Reading them from the graph avoids a
    class of silent mistakes.
    """
    import onnx  # type: ignore

    model = onnx.load(str(path))
    inp = model.graph.input[0]
    elem = inp.type.tensor_type.elem_type
    mapping = {
        1: np.dtype(np.float32),
        2: np.dtype(np.uint8),
        3: np.dtype(np.int8),
        6: np.dtype(np.int32),
        7: np.dtype(np.int64),
        10: np.dtype(np.float16),
        11: np.dtype(np.float64),
    }
    return inp.name, mapping.get(elem, np.dtype(np.float32))


def quantize_int8(
    src: str | Path,
    dst: str | Path,
    calibration_inputs: Sequence[np.ndarray],
    per_channel: bool = True,
    input_name: Optional[str] = None,
    dtype: Optional[np.dtype] = None,
) -> Path:
    """Static QDQ int8 quantisation with a calibration set.

    The input name and dtype default to whatever the graph declares, so this works for both the
    text side (``ids``, int64) and the vocoder (``latent``, float32).
    """
    from onnxruntime.quantization import (  # type: ignore
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    if input_name is None or dtype is None:
        detected_name, detected_dtype = onnx_input_spec(src)
        input_name = input_name or detected_name
        dtype = dtype or detected_dtype

    class _Reader(CalibrationDataReader):
        def __init__(self, items: Sequence[np.ndarray]) -> None:
            self._items = [{input_name: np.asarray(x, dtype=dtype)} for x in items]
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
        _Reader(calibration_inputs),
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
        x = np.asarray(x)
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"latent must be floating point, got {x.dtype}")
        log_mag, phase = self.session.run(None, {self.input_name: x.astype(np.float32, copy=False)})
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
    total_bytes = onnx_artifact_bytes(onnx_path)
    return {
        "mean_ms": mean_ms,
        "audio_seconds": audio_seconds,
        "rtf_fixed_frames": mean_ms / 1000.0 / max(audio_seconds, 1e-9),
        "latent_frames": float(frames),
        "model_bytes": float(total_bytes),
        "model_mb": total_bytes / 1e6,
        "providers": ",".join(voc.providers),
    }


class _TextSide(nn.Module):
    """ONNX-friendly view of the Tiny text side.

    ``mask=None`` means "every token is valid", which is exactly true at inference where we
    synthesise one unpadded sequence at a time -- and it keeps the exported graph free of
    mask-shape logic.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, ids: torch.Tensor):
        side = self.model.text_side(ids, None)
        return side["log_duration"], side["latent_token"], side["f0"], side["energy"]


def export_text_side_onnx(
    model: nn.Module,
    path: str | Path,
    opset: int = 18,
    tokens: int = 16,
    dynamo: bool = True,
) -> Path:
    """Export the Tiny text side: ``ids (B, T) -> log_duration, latent_token, f0, energy``.

    This matters because profiling showed the text side is ~34 % of a full synthesis, on par with
    the vocoder -- so exporting only the decoder cannot deliver the int8 win.

    ``dynamo=True`` (the ``torch.export``-based exporter, needs ``onnxscript``) is the *default
    here on purpose*: the legacy TorchScript exporter bakes the dummy sequence length into
    `nn.MultiheadAttention`'s internal reshapes, so the resulting graph only accepts the exact
    token count it was traced with -- a landmine that only shows up at inference time with a
    different sentence length.  The new exporter propagates the dynamic axis correctly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _TextSide(model).eval()
    vocab = getattr(getattr(model, "cfg", None), "text", None)
    vocab_size = getattr(vocab, "vocab_size", 128)
    dummy = torch.randint(0, vocab_size, (1, tokens))
    names = ["ids"]
    outputs = ["log_duration", "latent_token", "f0", "energy"]
    dynamic = {0: "batch", 1: "tokens"}
    if dynamo:
        _ensure_utf8_stdout()
        _quiet_export_logs()
        try:
            torch.onnx.export(
                wrapper,
                (dummy,),
                str(path),
                input_names=names,
                output_names=outputs,
                opset_version=opset,
                dynamo=True,
                dynamic_shapes={"ids": dict(dynamic)},
                external_data=False,
            )
        except TypeError:  # pragma: no cover - older exporter without the argument
            torch.onnx.export(
                wrapper,
                (dummy,),
                str(path),
                input_names=names,
                output_names=outputs,
                opset_version=opset,
                dynamo=True,
                dynamic_shapes={"ids": dict(dynamic)},
            )
    else:
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(path),
            input_names=names,
            output_names=outputs,
            dynamic_axes={"ids": dict(dynamic), **{o: dict(dynamic) for o in outputs}},
            opset_version=opset,
            dynamo=False,
        )
    return path


class OnnxTextSide:
    """Runtime wrapper for the exported Tiny text side."""

    def __init__(self, onnx_path: str | Path, intra_op_threads: int = 1) -> None:
        import onnxruntime as ort  # type: ignore

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, ids: np.ndarray | torch.Tensor) -> Dict[str, np.ndarray]:
        x = ids if isinstance(ids, np.ndarray) else ids.detach().cpu().numpy()
        x = np.asarray(x)
        # ONNX Runtime happily *casts* a float input to the int64 embedding indices instead of
        # erroring, which silently produces nonsense; fail loudly at the boundary.
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError(f"text-side ids must be integers, got {x.dtype}")
        log_duration, latent_token, f0, energy = self.session.run(
            None, {self.input_name: x.astype(np.int64, copy=False)}
        )
        return {"log_duration": log_duration, "latent_token": latent_token, "f0": f0, "energy": energy}


class OnnxTinyPipeline:
    """Full Tiny synthesis with the heavy modules in ONNX and the cheap ones in torch.

    Division of labour, chosen by measurement (see ``scripts/profile_pipeline.py``):

    ===========================  ======  ==========================================
    stage                        share   where it runs
    ===========================  ======  ==========================================
    text side                    34 %    **ONNX** (fp32 or int8)
    latent construction           1.5 %   torch -- dynamic ``repeat_interleave`` is not
                                          ONNX-friendly, and it costs almost nothing
    decoder + head               40 %    **ONNX** (fp32 or int8)
    iSTFT overlap-add            ~cheap  torch (complex ops export badly)
    phase-lock filter            11 %    torch (FFT-based)
    ===========================  ======  ==========================================
    """

    def __init__(
        self,
        text_side_path: str | Path,
        decoder_path: str | Path,
        model: nn.Module,
        cfg,
        intra_op_threads: int = 1,
        apply_phase_lock: bool = True,
        phase_lock_strength: float = 0.7,
        phase_lock_method: str = "ramp",
        tokenizer=None,
    ) -> None:
        from ..data.text import TextTokenizer

        self.cfg = cfg
        self.model = model.eval()
        self.text = OnnxTextSide(text_side_path, intra_op_threads=intra_op_threads)
        self.vocoder = OnnxVocoder(decoder_path, cfg.audio, intra_op_threads=intra_op_threads)
        self.tokenizer = tokenizer or TextTokenizer(mode=cfg.text.mode)
        self.apply_phase_lock = apply_phase_lock
        self.phase_lock_strength = phase_lock_strength
        self.phase_lock_method = phase_lock_method

    @torch.no_grad()
    def synthesize_ids(self, ids: np.ndarray | torch.Tensor) -> torch.Tensor:
        """``(B, T)`` token ids -> waveform ``(B, N)``."""
        side = self.text(ids)
        durations = normalized_to_durations(torch.from_numpy(side["log_duration"]))
        token_latent = torch.from_numpy(side["latent_token"])
        f0 = torch.from_numpy(side["f0"])
        energy = torch.from_numpy(side["energy"])
        latent, _ = self.model.decoder_latent_from_tokens(token_latent, durations, f0, energy)
        wav = self.vocoder.decode(latent.numpy())
        if self.apply_phase_lock:
            from .phase_lock import phase_lock

            wav = phase_lock(
                wav,
                sample_rate=self.cfg.audio.sample_rate,
                n_fft=self.cfg.audio.n_fft,
                hop_length=self.cfg.audio.hop_length,
                strength=self.phase_lock_strength,
                method=self.phase_lock_method,
            )
        return wav

    def synthesize(self, text: str, **_: object) -> torch.Tensor:
        ids, _mask = self.tokenizer.batch([text], max_len=self.cfg.text.max_len, add_special=False)
        return self.synthesize_ids(ids.numpy())


def compare_pipelines(
    model: nn.Module,
    cfg,
    out_dir: str | Path,
    texts: Sequence[str],
    runs: int = 5,
    threads: int = 1,
    opset: int = 17,
) -> Dict[str, object]:
    """Export both halves, build the ONNX pipelines, and benchmark + verify against PyTorch.

    Equivalence is measured rather than assumed: the ONNX pipelines must reproduce the PyTorch
    waveform (mel L1 and correlation), otherwise a speed-up is meaningless.
    """
    from ..audio.mel import MelSpectrogram
    from ..data.text import TextTokenizer
    from ..inference.synthesize import Synthesizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mel = MelSpectrogram(cfg.audio)
    torch.set_num_threads(threads)
    tokenizer = TextTokenizer(mode=cfg.text.mode)

    text_fp32 = export_text_side_onnx(
        model, out_dir / "text_side_fp32.onnx", opset=max(opset, 18), dynamo=True
    )
    dec_fp32 = export_onnx(model, cfg.audio, out_dir / "vocoder_fp32.onnx", opset=opset)

    # calibration inputs: several token sequences (no specials -- the same convention used for
    # training the text side, see TextTokenizer.batch(add_special=False))
    cal_texts = list(texts) + [
        "the quick brown fox jumps over the lazy dog",
        "hello there",
        "parakeet is a small and fast text to speech model",
    ]
    cal_sequences = [
        tokenizer.encode(t, add_special=False).numpy().astype(np.int64)[None] for t in cal_texts
    ]
    with torch.no_grad():
        cal_ids = np.concatenate(cal_sequences, axis=1)
        side = OnnxTextSide(text_fp32)(cal_ids)
        durations = normalized_to_durations(torch.from_numpy(side["log_duration"]))
        cal_latents, _ = model.decoder_latent_from_tokens(
            torch.from_numpy(side["latent_token"]),
            durations,
            torch.from_numpy(side["f0"]),
            torch.from_numpy(side["energy"]),
        )
    text_int8 = quantize_int8(text_fp32, out_dir / "text_side_int8.onnx", cal_sequences)
    dec_int8 = quantize_int8(
        dec_fp32, out_dir / "vocoder_int8.onnx", [cal_latents.numpy().astype(np.float32)]
    )

    torch_pipe = Synthesizer(model, cfg, device="cpu", apply_phase_lock=False)
    onnx_fp32 = OnnxTinyPipeline(text_fp32, dec_fp32, model, cfg, threads, apply_phase_lock=False)
    onnx_int8 = OnnxTinyPipeline(text_int8, dec_int8, model, cfg, threads, apply_phase_lock=False)
    # the *shipped* configuration includes the parameter-free phase-lock filter
    onnx_int8_shipped = OnnxTinyPipeline(text_int8, dec_int8, model, cfg, threads, apply_phase_lock=True)

    def bench(fn, runs: int = runs) -> float:
        fn()
        t0 = time.perf_counter()
        for _ in range(runs):
            out = fn()
        return (time.perf_counter() - t0) / runs * 1000.0, out

    rows = []
    for text in texts:
        t_torch, wav_torch = bench(lambda t=text: torch_pipe.synthesize(t, seed=0))
        t_fp32, wav_fp32 = bench(lambda t=text: onnx_fp32.synthesize(t))
        t_int8, wav_int8 = bench(lambda t=text: onnx_int8.synthesize(t))
        t_ship, wav_ship = bench(lambda t=text: onnx_int8_shipped.synthesize(t))
        n = min(wav_torch.shape[-1], wav_int8.shape[-1])
        a, b = mel.log_mel(wav_torch[..., :n]), mel.log_mel(wav_int8[..., :n])
        corr = float(
            torch.nn.functional.cosine_similarity(
                wav_torch[..., :n].reshape(-1)[None], wav_int8[..., :n].reshape(-1)[None]
            ).item()
        )
        rows.append(
            {
                "text": text,
                "torch_ms": t_torch,
                "onnx_fp32_ms": t_fp32,
                "onnx_int8_ms": t_int8,
                "onnx_int8_shipped_ms": t_ship,
                "audio_seconds": n / cfg.audio.sample_rate,
                "int8_vs_torch_speedup": t_torch / max(t_int8, 1e-9),
                "int8_shipped_vs_torch_speedup": t_torch / max(t_ship, 1e-9),
                "int8_vs_torch_mel_l1": float(F.l1_loss(a, b).item()),
                "int8_vs_torch_waveform_cosine": corr,
            }
        )

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    torch_ms, int8_ms = mean("torch_ms"), mean("onnx_int8_ms")
    audio_s = mean("audio_seconds")
    return {
        "rows": rows,
        "text_side_mb": {
            "fp32": onnx_artifact_bytes(text_fp32) / 1e6,
            "int8": onnx_artifact_bytes(text_int8) / 1e6,
        },
        "vocoder_mb": {
            "fp32": onnx_artifact_bytes(dec_fp32) / 1e6,
            "int8": onnx_artifact_bytes(dec_int8) / 1e6,
        },
        "mean_ms": {
            "torch": torch_ms,
            "onnx_fp32": mean("onnx_fp32_ms"),
            "onnx_int8": int8_ms,
            "onnx_int8_shipped": mean("onnx_int8_shipped_ms"),
        },
        "mean_audio_seconds": audio_s,
        "torch_rtf": (torch_ms / 1000.0) / max(audio_s, 1e-9),
        "int8_rtf": (int8_ms / 1000.0) / max(audio_s, 1e-9),
        "int8_shipped_rtf": (mean("onnx_int8_shipped_ms") / 1000.0) / max(audio_s, 1e-9),
        "torch_x_realtime": audio_s / max(torch_ms / 1000.0, 1e-9),
        "int8_x_realtime": audio_s / max(int8_ms / 1000.0, 1e-9),
        "int8_shipped_x_realtime": audio_s / max(mean("onnx_int8_shipped_ms") / 1000.0, 1e-9),
        "int8_vs_torch_speedup": torch_ms / max(int8_ms, 1e-9),
        "int8_shipped_vs_torch_speedup": torch_ms / max(mean("onnx_int8_shipped_ms"), 1e-9),
        "int8_vs_torch_mel_l1": mean("int8_vs_torch_mel_l1"),
        "int8_vs_torch_waveform_cosine": mean("int8_vs_torch_waveform_cosine"),
        "total_int8_mb": (
            onnx_artifact_bytes(text_int8) + onnx_artifact_bytes(dec_int8)
        )
        / 1e6,
        "total_fp32_mb": (
            onnx_artifact_bytes(text_fp32) + onnx_artifact_bytes(dec_fp32)
        )
        / 1e6,
        "threads": threads,
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
