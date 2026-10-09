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
    #: which configuration produced the value (e.g. the Whisper size).  A WER from `base.en` is not
    #: comparable to one from `large-v3`, so the number must travel with its own provenance.
    detail: str = ""


def utmos(wavs: Sequence[torch.Tensor], sample_rate: int = 24000) -> OptionalMetric:
    """UTMOS22 strong predictor (pip install utmos / torch.hub)."""
    try:
        import utmos  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return OptionalMetric(None, False, f"`utmos` not installed ({exc.__class__.__name__})")
    try:  # pragma: no cover - requires the optional dependency
        predictor = utmos.Score(sample_rate=sample_rate)
        scores = [predictor.score(w.reshape(-1).numpy()) for w in wavs]
        return OptionalMetric(float(np.mean(scores)), True, detail="utmos22 strong")
    except Exception as exc:  # pragma: no cover
        return OptionalMetric(None, False, str(exc))


def dnsmos_score(audio: Sequence[torch.Tensor], sample_rate: int = 24000) -> OptionalMetric:
    """DNSMOS P.835 (pip install speechmos) -- reference-free naturalness.

    This is the metric the curation gate already references (``CurateConfig.min_dnsmos`` = 3.5, from
    PilotTTS), and unlike UTMOS it installs here: ``utmos`` needs ``fairseq``, whose sdist cannot
    build on this Python.  DNSMOS wants 16 kHz input, and it resamples with a polyphase filter
    (scipy is present with librosa), falling back to linear interpolation.

    Returns the **overall** MOS (``ovrl_mos``); the signal/background sub-scores and the P.808 score
    travel in ``detail``, because they are what diagnose *why* a sample scores low.
    """
    try:
        from speechmos import dnsmos  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        return OptionalMetric(None, False, f"`speechmos` not installed ({exc.__class__.__name__})")
    try:  # pragma: no cover - requires the optional dependency
        results = []
        clipped = []
        for wav in audio:
            mono = wav.reshape(-1)
            # the decoder is not guaranteed to stay inside [-1, 1] (measured: a token-rate sweep hit
            # "np.ndarray values must be between -1 and 1" from DNSMOS on one utterance), and a metric
            # should measure the audio rather than refuse it -- but the clipping is reported, because
            # a hot output is itself a finding
            over = float((mono.abs() > 1.0).float().mean()) if mono.numel() else 0.0
            clipped.append(over)
            mono = mono.clamp(-1.0, 1.0)
            if sample_rate != 16000:
                try:
                    from scipy.signal import resample_poly

                    mono = torch.from_numpy(
                        resample_poly(mono.numpy(), 16000, sample_rate).astype("float32")
                    )
                except Exception:  # noqa: BLE001 - scipy is optional; linear is a fallback
                    target = int(mono.numel() * 16000 / sample_rate)
                    mono = torch.nn.functional.interpolate(
                        mono.reshape(1, 1, -1), size=target, mode="linear", align_corners=False
                    ).reshape(-1)
                # polyphase interpolation can *overshoot* the input range, so the clamp has to happen
                # again after resampling (measured: it did, and DNSMOS refused the array)
                mono = mono.clamp(-1.0, 1.0)
            results.append(dnsmos.run(mono.numpy(), 16000, return_df=False))
        # keys are `ovrl_mos`, `sig_mos`, `bak_mos`, `p808_mos` -- NOT `mos_ovrl`, which is what the
        # first version of this wrapper looked for, silently returning 0.0 for real speech
        overall = [
            float(r.get("ovrl_mos", r.get("p808_mos", float("nan")))) if isinstance(r, dict)
            else float(np.asarray(r).reshape(-1)[-1])
            for r in results
        ]
        if results and isinstance(results[0], dict):
            detail = (
                f"dnsmos p835 ovrl | sig {np.mean([r['sig_mos'] for r in results]):.2f} "
                f"bak {np.mean([r['bak_mos'] for r in results]):.2f} "
                f"p808 {np.mean([r['p808_mos'] for r in results]):.2f}"
            )
        else:
            detail = "dnsmos p835 ovrl"
        clipped_frac = float(np.mean(clipped)) if clipped else 0.0
        if clipped_frac > 0:
            detail += f" | {100 * clipped_frac:.2f}% of samples clipped to [-1, 1]"
        return OptionalMetric(float(np.mean(overall)), True, detail=detail)
    except Exception as exc:  # pragma: no cover
        return OptionalMetric(None, False, str(exc))


def whisper_wer(
    audio: Sequence[torch.Tensor],
    texts: Sequence[str],
    sample_rate: int = 24000,
    model_size: str = "large-v3",
) -> OptionalMetric:
    """WER of synthesized audio against the intended text, via faster-whisper.

    ``model_size`` matters and is recorded in ``detail``: the papers report WER with a large ASR
    model, and a smaller one (practical on CPU) yields a different, usually higher, number.  A WER
    without its recogniser attached is not a result.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # pragma: no cover
        return OptionalMetric(None, False, f"`faster-whisper` not installed ({exc.__class__.__name__})")
    try:  # pragma: no cover
        import re

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        total_err = total_ref = 0
        for w, ref in zip(audio, texts):
            segments, _ = model.transcribe(w.reshape(-1).numpy(), language="en")
            hyp = " ".join(s.text for s in segments)
            norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).split()
            r, h = norm(ref), norm(hyp)
            err = sum(1 for a, b in zip(r, h) if a != b) + abs(len(r) - len(h))
            total_err += err
            total_ref += max(1, len(r))
        return OptionalMetric(total_err / max(1, total_ref), True, detail=f"faster-whisper {model_size}")
    except Exception as exc:  # pragma: no cover
        return OptionalMetric(None, False, str(exc))


def whisper_wer_with_control(
    audio: Sequence[torch.Tensor],
    texts: Sequence[str],
    control_audio: Optional[Sequence[torch.Tensor]] = None,
    sample_rate: int = 24000,
    model_size: str = "base.en",
    attempts: int = 2,
) -> Dict[str, object]:
    """WER measured against the recogniser's **own control**, with a retry.

    The control is the teacher's audio for the same text: if the recogniser cannot transcribe *that*,
    the WER it produces for the student is not a measurement of the student.

    This is not hypothetical.  A run of ``scripts/ae_train.py`` reported teacher WER **0.926** where
    the same three clips, same model, same code path scored **0.000**; re-running reproduced 0.000,
    and the audio tensors were verified byte-identical (no in-place mutation anywhere in the mel /
    encode / decode / phase-coherence / DNSMOS path).  A transient int8 CTranslate2 failure produced a
    meaningless number, and the only reason it was caught is that the control was reported next to it.

    So: run the control **first**, retry the whole measurement once if it fails, and report the WER as
    *unavailable* -- with the failure as the reason -- if the control still fails.  Returns
    ``{"wer", "control", "attempts", "note"}``.
    """
    last_control = OptionalMetric(None, False, "not run")
    last_wer = OptionalMetric(None, False, "not run")
    for attempt in range(1, max(1, attempts) + 1):
        if control_audio is not None:
            last_control = whisper_wer(
                control_audio, texts, sample_rate=sample_rate, model_size=model_size
            )
            control_ok = bool(last_control.available and last_control.value < 0.5)
        else:
            control_ok = True
        last_wer = whisper_wer(audio, texts, sample_rate=sample_rate, model_size=model_size)
        if control_ok and last_wer.available:
            return {
                "wer": last_wer,
                "control": last_control,
                "attempts": attempt,
                "note": "the recogniser transcribed the teacher's audio correctly",
            }
    reason = (
        f"the recogniser failed its own control (teacher WER "
        f"{last_control.value if last_control.value is not None else 'unavailable'}), so the "
        f"student WER from the same call is not reported"
    )
    return {
        "wer": OptionalMetric(None, False, reason, detail=f"faster-whisper {model_size}"),
        "control": last_control,
        "attempts": attempts,
        "note": reason,
    }


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
