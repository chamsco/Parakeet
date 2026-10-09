"""Evaluation metrics.

The four numbers that decide whether Parakeet is a success, in the order they matter:

1. **RTF on one CPU thread** -- the entire premise is "lightning fast, on a laptop".  Paradee
   reports 25.0x real time (17.8x via ONNX) on one CPU thread; a public CPU benchmark puts
   Supertonic-3 at RTF 0.313 (5-step) and Kokoro at 0.469, so CPU is the bar to beat.
2. **UTMOS** -- neural naturalness predictor, 1-5.  Paradee: teacher 4.52, student 4.41.
3. **WER** (Whisper) -- intelligibility.  Paradee: 5.7% for both teacher and student.
4. **Speaker similarity** (SECS) -- for the zero-shot Small model.

Metrics that need optional heavy dependencies are declared ``available=False`` with a reason
rather than being silently faked.  The dependency-free diagnostics (MCD, log-mel distance,
phase coherence, RTF) always work and are what CI runs.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..audio.mel import MelSpectrogram
from ..config import ParakeetConfig
from ..inference.phase_lock import phase_coherence


# --------------------------------------------------------------------------------------
# dependency-free diagnostics
# --------------------------------------------------------------------------------------
def _align(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[-1], b.shape[-1])
    return a.reshape(-1)[:n], b.reshape(-1)[:n]


def spectral_convergence(a: torch.Tensor, b: torch.Tensor, n_fft: int = 1024, hop: int = 256) -> float:
    a, b = _align(a, b)
    win = torch.hann_window(n_fft, device=a.device)
    sa = torch.stft(a, n_fft, hop, n_fft, win, return_complex=True).abs()
    sb = torch.stft(b, n_fft, hop, n_fft, win, return_complex=True).abs()
    num = torch.linalg.norm(sb - sa)
    den = torch.linalg.norm(sb).clamp_min(1e-8)
    return float((num / den).item())


def log_mel_l1(a: torch.Tensor, b: torch.Tensor, cfg: ParakeetConfig) -> float:
    mel = MelSpectrogram(cfg.audio)
    n = min(a.shape[-1], b.shape[-1])
    ma = mel.log_mel(a.reshape(1, -1)[..., :n])
    mb = mel.log_mel(b.reshape(1, -1)[..., :n])
    return float(F.l1_loss(ma, mb).item())


def mel_cepstral_distortion(
    a: torch.Tensor, b: torch.Tensor, cfg: ParakeetConfig, n_ceps: int = 13
) -> float:
    """MCD in dB via DCT of the log-mel spectra (dependency-free approximation)."""
    mel = MelSpectrogram(cfg.audio)
    n = min(a.shape[-1], b.shape[-1])
    ma = mel.log_mel(a.reshape(1, -1)[..., :n])
    mb = mel.log_mel(b.reshape(1, -1)[..., :n])
    k = torch.arange(n_ceps, device=ma.device)[:, None]
    m = torch.arange(ma.shape[-2], device=ma.device)[None, :].float()
    basis = torch.cos(math.pi / ma.shape[-2] * (m + 0.5) * k)
    ca = torch.matmul(basis, ma)[0]
    cb = torch.matmul(basis, mb)[0]
    diff = (ca - cb)[1:]  # skip c0 (energy)
    return float((10.0 / math.log(10.0)) * torch.sqrt(2.0 * diff.pow(2).sum(dim=0)).mean().item())


def segmental_snr(reference: torch.Tensor, estimate: torch.Tensor, frame: int = 2048) -> float:
    r, e = _align(reference, estimate)
    n = (r.numel() // frame) * frame
    if n == 0:
        return float("nan")
    r = r[:n].reshape(-1, frame)
    e = e[:n].reshape(-1, frame)
    noise = (r - e).pow(2).mean(dim=-1)
    signal = r.pow(2).mean(dim=-1)
    snr = 10 * torch.log10((signal + 1e-8) / (noise + 1e-8))
    return float(snr.mean().item())


def buzz(wav: torch.Tensor, cfg: ParakeetConfig) -> float:
    """Phase coherence in 2-8 kHz: higher is *less* buzz (see inference.phase_lock)."""
    return float(
        phase_coherence(
            wav,
            sample_rate=cfg.audio.sample_rate,
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
        ).item()
    )


# --------------------------------------------------------------------------------------
# optional heavy metrics
# --------------------------------------------------------------------------------------
@dataclass
class OptionalMetric:
    value: Optional[float]
    available: bool
    reason: str = ""


def utmos(wavs: Sequence[torch.Tensor], sample_rate: int = 24000) -> OptionalMetric:
    """UTMOS22 strong predictor (pip install utmos / torch.hub)."""
    try:
        import utmos  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return OptionalMetric(None, False, f"`utmos` not installed ({exc.__class__.__name__})")
    try:  # pragma: no cover - requires the optional dependency
        predictor = utmos.Score(sample_rate=sample_rate)
        scores = [predictor.score(w.reshape(-1).numpy()) for w in wavs]
        return OptionalMetric(float(np.mean(scores)), True)
    except Exception as exc:  # pragma: no cover
        return OptionalMetric(None, False, str(exc))


def whisper_wer(audio: Sequence[torch.Tensor], texts: Sequence[str], sample_rate: int = 24000) -> OptionalMetric:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # pragma: no cover
        return OptionalMetric(None, False, f"`faster-whisper` not installed ({exc.__class__.__name__})")
    try:  # pragma: no cover
        import re

        model = WhisperModel("large-v3", device="auto", compute_type="int8")
        total_err = total_ref = 0
        for w, ref in zip(audio, texts):
            segments, _ = model.transcribe(w.reshape(-1).numpy(), language="en")
            hyp = " ".join(s.text for s in segments)
            norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).split()
            r, h = norm(ref), norm(hyp)
            err = sum(1 for a, b in zip(r, h) if a != b) + abs(len(r) - len(h))
            total_err += err
            total_ref += max(1, len(r))
        return OptionalMetric(total_err / max(1, total_ref), True)
    except Exception as exc:  # pragma: no cover
        return OptionalMetric(None, False, str(exc))


def speaker_similarity(
    a: torch.Tensor, b: torch.Tensor, cfg: ParakeetConfig
) -> OptionalMetric:
    """SECS.  Uses the frozen CAM++ encoder when available, else reports unavailable.

    We deliberately do **not** fall back to our randomly-initialised ECAPA stand-in: a
    similarity number from an untrained encoder is worse than no number.
    """
    try:  # pragma: no cover - optional dependency
        from funasr import AutoModel  # type: ignore
    except Exception as exc:
        return OptionalMetric(None, False, f"CAM++/funasr not installed ({exc.__class__.__name__})")
    return OptionalMetric(None, False, "funasr present but CAM++ checkpoint not configured")


# --------------------------------------------------------------------------------------
# real-time factor
# --------------------------------------------------------------------------------------
@dataclass
class RTFResult:
    rtf: float
    samples_per_second: float
    seconds_per_audio_second: float
    threads: int
    n_runs: int
    audio_seconds: float


def measure_rtf(
    synthesize_fn: Callable[[], torch.Tensor],
    sample_rate: int = 24000,
    warmup: int = 1,
    runs: int = 3,
    threads: int = 1,
) -> RTFResult:
    """Time ``synthesize_fn`` and compare against the audio it produced."""
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        audio = None
        for _ in range(warmup):
            audio = synthesize_fn()
        t0 = time.perf_counter()
        for _ in range(runs):
            audio = synthesize_fn()
        elapsed = (time.perf_counter() - t0) / max(1, runs)
    finally:
        torch.set_num_threads(prev)
    n = int(audio.reshape(-1).shape[0]) if audio is not None else 0
    audio_seconds = n / sample_rate
    rtf = elapsed / max(audio_seconds, 1e-9)
    return RTFResult(
        rtf=rtf,
        samples_per_second=n / max(elapsed, 1e-9),
        seconds_per_audio_second=1.0 / max(rtf, 1e-9),
        threads=threads,
        n_runs=runs,
        audio_seconds=audio_seconds,
    )


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
@dataclass
class EvalReport:
    n_utterances: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    unavailable: Dict[str, str] = field(default_factory=dict)
    rtf: Optional[Dict[str, float]] = None
    extras: Dict[str, object] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def evaluate_pair(
    generated: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
    texts: Sequence[str],
    cfg: ParakeetConfig,
    include_optional: bool = True,
) -> EvalReport:
    """Compare a student against a teacher (or against ground truth)."""
    report = EvalReport(n_utterances=len(generated))
    if not generated:
        return report
    sc = [spectral_convergence(g, r) for g, r in zip(generated, reference)]
    lm = [log_mel_l1(g, r, cfg) for g, r in zip(generated, reference)]
    mcd = [mel_cepstral_distortion(g, r, cfg) for g, r in zip(generated, reference)]
    snr = [segmental_snr(r, g) for g, r in zip(generated, reference)]
    bz = [buzz(g, cfg) for g in generated]
    report.metrics.update(
        {
            "spectral_convergence": float(np.mean(sc)),
            "log_mel_l1": float(np.mean(lm)),
            "mcd_db": float(np.mean(mcd)),
            "segmental_snr_db": float(np.nanmean(snr)),
            "phase_coherence_2_8k": float(np.mean(bz)),
        }
    )
    if include_optional:
        for name, res in (("utmos", utmos(generated, cfg.audio.sample_rate)),):
            if res.available and res.value is not None:
                report.metrics[name] = res.value
            else:
                report.unavailable[name] = res.reason
        wer = whisper_wer(generated, texts, cfg.audio.sample_rate)
        if wer.available and wer.value is not None:
            report.metrics["wer"] = wer.value
        else:
            report.unavailable["wer"] = wer.reason
    return report


def list_metric_availability() -> Dict[str, bool]:
    """Report which optional metrics are live in this environment (used by CI and docs)."""
    wav = torch.randn(1, 24000)
    cfg = ParakeetConfig()
    out = {
        "utmos": utmos([wav]).available,
        "whisper_wer": whisper_wer([wav], ["test"]).available,
        "speaker_similarity": speaker_similarity(wav, wav, cfg).available,
    }
    return out
