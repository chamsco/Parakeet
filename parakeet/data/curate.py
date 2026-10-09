"""Executable audio-curation pipeline, following the published parts of the PilotTTS recipe.

PilotTTS's paper describes a data pipeline and claims to release it, but the repository is
**inference-only** -- no pipeline or training code, and the issue asking for it has no maintainer
reply.  So the stages below are our implementation of the paper's description, with the concrete
numeric thresholds taken from CosyVoice 3 where PilotTTS leaves them unspecified.  Every threshold
is tagged in :class:`CurateConfig`:

* ``[published]`` -- stated in PilotTTS or CosyVoice 3;
* ``[proposed]``  -- ours, because the paper does not publish a number.

Two honesty rules are enforced here, because a curation pipeline is very easy to fool yourself
with:

1. **Never fake a MOS.**  Neural MOS (DNSMOS) needs a model; if the caller does not inject one we
   record ``mos_source="unavailable"`` and simply do not gate on MOS.  Objective measurements
   (clipping, SNR estimate, bandwidth, silence) are always recorded.
2. **Never delete a rejected item.**  Rejects are written to ``rejected.jsonl`` with the reasons,
   so filters can be re-tuned and audited without re-decoding the corpus.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass
class CurateConfig:
    min_duration_s: float = 3.0          # [published] CosyVoice 3: segments < 30 s
    max_duration_s: float = 30.0         # [published]
    min_snr_db: float = 15.0             # [proposed]
    min_rms_dbfs: float = -45.0          # [proposed] absolute level floor (silence detector)
    min_dnsmos: float = 3.5              # [published] PilotTTS: deficient if MOS <= 3.5
    max_clipping_ratio: float = 0.01     # [proposed]
    max_silence_ratio: float = 0.50      # [proposed]
    min_bandwidth_hz: float = 5000.0     # [proposed]
    asr_max_avg_wer: float = 0.15        # [published] CosyVoice 3: keep if pairwise WER < 15 %
    peak_factor: float = 0.6             # [published] CosyVoice 3: raw/max(raw)*0.6
    add_comma_gap_ms: float = 300.0      # [published] CosyVoice 3
    remove_comma_gap_ms: float = 50.0    # [published] CosyVoice 3
    length_ratio_low_pct: float = 1.0    # [published] CosyVoice 3: drop smallest 1 %
    length_ratio_high_pct: float = 5.0   # [published] CosyVoice 3: drop largest 5 %
    silence_floor_db: float = 40.0       # [proposed] frame energy below peak-40 dB counts as silence


# --------------------------------------------------------------------------------------
# objective measurements
# --------------------------------------------------------------------------------------
@dataclass
class AudioQuality:
    duration_s: float
    peak_db: float
    rms_db: float
    clipping_ratio: float
    silence_ratio: float
    snr_db: Optional[float]
    rolloff95_hz: float
    bandwidth99_hz: float
    mos: Optional[float] = None
    mos_source: str = "unavailable"
    keep: bool = True
    reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _dbfs(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-8))


def estimate_snr_db(
    wav: torch.Tensor,
    sr: int,
    frame_ms: float = 20.0,
    min_dynamic_range_db: float = 6.0,
) -> Optional[float]:
    """Percentile SNR estimate, or ``None`` when it is not measurable.

    Noise is the 10th percentile of frame energy and speech the 90th.  That only means anything
    when the recording *has* a noise floor to observe: on a signal with no pauses (continuous
    speech, or a stationary tone) the two percentiles coincide and the ratio carries no
    information.  Returning ``None`` and skipping the gate is the honest behaviour -- gating on a
    meaningless 0 dB would reject perfectly good continuous speech.
    """
    frame = max(64, int(sr * frame_ms / 1000.0))
    x = wav.reshape(-1)
    n = (x.numel() // frame) * frame
    if n < frame * 4:
        return None
    frames = x[:n].reshape(-1, frame)
    energy = frames.pow(2).mean(dim=-1).clamp_min(1e-12)
    speech = torch.quantile(energy, 0.90)
    noise = torch.quantile(energy, 0.10)
    dynamic = float(10.0 * torch.log10(speech / noise).item())
    if dynamic < min_dynamic_range_db:
        return None
    return dynamic


def spectral_stats(wav: torch.Tensor, sr: int, n_fft: int = 1024) -> Tuple[float, float]:
    """Return ``(95 % rolloff Hz, 99 % bandwidth Hz)`` averaged over frames."""
    x = wav.reshape(-1)
    if x.numel() < n_fft:
        x = F.pad(x, (0, n_fft - x.numel()))
    window = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(x, n_fft, n_fft // 4, n_fft, window, return_complex=True).abs().pow(2)
    if spec.numel() == 0:
        return 0.0, 0.0
    freqs = torch.linspace(0.0, sr / 2.0, spec.shape[0], device=x.device)
    total = spec.sum(dim=0).clamp_min(1e-12)
    cumulative = spec.cumsum(dim=0) / total[None]
    rolloff = freqs[(cumulative >= 0.95).float().argmax(dim=0)].mean()
    bandwidth = freqs[(cumulative >= 0.99).float().argmax(dim=0)].mean()
    return float(rolloff.item()), float(bandwidth.item())


def analyze_audio(
    wav: torch.Tensor,
    sample_rate: int,
    cfg: Optional[CurateConfig] = None,
    mos_fn: Optional[Callable[[torch.Tensor, int], float]] = None,
) -> AudioQuality:
    """Measure one waveform and decide whether it survives the gates."""
    cfg = cfg or CurateConfig()
    x = wav.reshape(-1).float()
    peak = float(x.abs().max().item()) if x.numel() else 0.0
    rms = float(x.pow(2).mean().sqrt().item()) if x.numel() else 0.0
    clipping = float((x.abs() >= 0.99).float().mean().item()) if x.numel() else 1.0

    frame = 1024
    n = (x.numel() // frame) * frame
    if n and peak > 1e-6:
        energies = x[:n].reshape(-1, frame).pow(2).mean(dim=-1).clamp_min(1e-12)
        peak_energy = energies.max()
        silence = float(
            (energies < peak_energy * 10 ** (-cfg.silence_floor_db / 10)).float().mean().item()
        )
    else:
        # a signal with no discernible peak is entirely silence.  Without this branch digital
        # silence scored silence_ratio 0.0 -- the "below peak - 40 dB" rule is degenerate when the
        # peak is zero -- and then passed every gate except duration/bandwidth.
        silence = 1.0

    rolloff, bandwidth = spectral_stats(x, sample_rate)
    quality = AudioQuality(
        duration_s=x.numel() / sample_rate,
        peak_db=_dbfs(peak),
        rms_db=_dbfs(rms),
        clipping_ratio=clipping,
        silence_ratio=silence,
        snr_db=estimate_snr_db(x, sample_rate),
        rolloff95_hz=rolloff,
        bandwidth99_hz=bandwidth,
    )
    if quality.snr_db is None:
        quality.notes.append("snr_unevaluable(no_noise_floor_observed)")
    if mos_fn is not None:
        quality.mos = float(mos_fn(x, sample_rate))
        quality.mos_source = "injected"
    return apply_gates(quality, cfg)


def apply_gates(q: AudioQuality, cfg: CurateConfig) -> AudioQuality:
    """Apply every threshold; collect ALL reasons rather than short-circuiting at the first."""
    reasons: List[str] = []
    if q.duration_s < cfg.min_duration_s:
        reasons.append(f"too_short({q.duration_s:.2f}s<{cfg.min_duration_s})")
    if q.duration_s > cfg.max_duration_s:
        reasons.append(f"too_long({q.duration_s:.2f}s>{cfg.max_duration_s})")
    # absolute level, not just a ratio against the file's own peak: a file that failed to render
    # (silence, or a near-silent teacher response) must be rejected even when every relative and
    # duration gate is relaxed.  Without this, digital silence was *kept* under relaxed gates.
    if q.rms_db < cfg.min_rms_dbfs:
        reasons.append(f"low_level({q.rms_db:.1f}dBFS<{cfg.min_rms_dbfs})")
    if q.clipping_ratio > cfg.max_clipping_ratio:
        reasons.append(f"clipping({q.clipping_ratio:.3f})")
    if q.silence_ratio > cfg.max_silence_ratio:
        reasons.append(f"mostly_silent({q.silence_ratio:.2f})")
    if q.snr_db is not None and q.snr_db < cfg.min_snr_db:
        reasons.append(f"low_snr({q.snr_db:.1f}dB)")
    if q.bandwidth99_hz < cfg.min_bandwidth_hz:
        reasons.append(f"narrowband({q.bandwidth99_hz:.0f}Hz)")
    if q.mos is not None and q.mos < cfg.min_dnsmos:
        reasons.append(f"low_mos({q.mos:.2f})")
    q.reasons = reasons
    q.keep = not reasons
    return q


# --------------------------------------------------------------------------------------
# text / alignment stages
# --------------------------------------------------------------------------------------
def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein WER on whitespace tokens (no external dependency)."""
    ref = _tokens(reference)
    hyp = _tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def _tokens(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def average_pairwise_wer(transcripts: Sequence[str]) -> float:
    """Mean WER over all unordered pairs of ASR hypotheses for the same utterance.

    CosyVoice 3 keeps a segment when the average pairwise WER across several ASR systems is below
    15 %.  Agreement between independent systems is a cheap proxy for transcription reliability.
    """
    texts = [t for t in transcripts if t is not None]
    if len(texts) < 2:
        return 0.0
    values: List[float] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            values.append(max(word_error_rate(a, b), word_error_rate(b, a)))
    return sum(values) / len(values)


def apply_punctuation_gaps(
    words: Sequence[str],
    boundaries: Sequence[Tuple[float, float]],
    cfg: Optional[CurateConfig] = None,
) -> str:
    """Re-punctuate from forced-alignment gaps.

    ``boundaries[i] = (start_s, end_s)`` for ``words[i]``.  A gap of at least
    ``add_comma_gap_ms`` gets a comma; an existing comma is dropped when its surrounding gap is at
    most ``remove_comma_gap_ms`` (CosyVoice 3's rule, applied here to a plain word list).
    """
    cfg = cfg or CurateConfig()
    if len(words) != len(boundaries):
        raise ValueError("words and boundaries must have the same length")
    out: List[str] = []
    cleaned = [w.strip().rstrip(",") for w in words]
    for i, word in enumerate(cleaned):
        out.append(word)
        if i + 1 < len(cleaned):
            gap_ms = (boundaries[i + 1][0] - boundaries[i][1]) * 1000.0
            if gap_ms >= cfg.add_comma_gap_ms:
                out[-1] = out[-1] + ","
            elif gap_ms <= cfg.remove_comma_gap_ms:
                out[-1] = out[-1].rstrip(",")
    text = " ".join(out)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def length_ratio_mask(
    ratios: Sequence[float], low_pct: float = 1.0, high_pct: float = 5.0
) -> List[bool]:
    """Drop the extreme text-length/speech-length ratio tails (CosyVoice 3: 1 % low, 5 % high)."""
    if not ratios:
        return []
    values = torch.tensor(list(ratios), dtype=torch.float32)
    lo = float(torch.quantile(values, low_pct / 100.0).item())
    hi = float(torch.quantile(values, 1.0 - high_pct / 100.0).item())
    return [bool(lo <= r <= hi) for r in ratios]


def normalize_loudness(wav: torch.Tensor, peak_factor: float = 0.6) -> torch.Tensor:
    """Peak normalisation ``raw / max(|raw|) * peak_factor`` (CosyVoice 3's rule)."""
    x = wav.reshape(-1).float()
    peak = x.abs().max()
    if float(peak) < 1e-8:
        return x
    return x / peak * peak_factor


# --------------------------------------------------------------------------------------
# manifest driver
# --------------------------------------------------------------------------------------
@dataclass
class CurationReport:
    n_total: int = 0
    n_kept: int = 0
    n_rejected: int = 0
    reason_counts: Dict[str, int] = field(default_factory=dict)
    hours_kept: float = 0.0


def curate_manifest(
    records: Iterable[Dict],
    load_wav: Callable[[str], Tuple[torch.Tensor, int]],
    out_dir: str | Path,
    cfg: Optional[CurateConfig] = None,
    asr_fn: Optional[Callable[[torch.Tensor, int], Sequence[str]]] = None,
    mos_fn: Optional[Callable[[torch.Tensor, int], float]] = None,
    normalize: bool = True,
) -> CurationReport:
    """Run the gates over a manifest and write ``kept.jsonl`` + ``rejected.jsonl``.

    ``asr_fn`` returns several transcripts (one per ASR system) and is checked against
    :attr:`CurateConfig.asr_max_avg_wer`; without it the ASR stage is skipped and recorded as such.
    Rejected items are always written out with their reasons.
    """
    cfg = cfg or CurateConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: List[Dict] = []
    rejected: List[Dict] = []
    report = CurationReport()

    for rec in records:
        report.n_total += 1
        wav, sr = load_wav(rec["wav_path"])
        quality = analyze_audio(wav, sr, cfg, mos_fn=mos_fn)
        entry = {**rec, "quality": asdict(quality)}

        if quality.keep and asr_fn is not None:
            transcripts = list(asr_fn(wav, sr))
            avg_wer = average_pairwise_wer(transcripts)
            entry["quality"]["asr_pairwise_wer"] = avg_wer
            entry["quality"]["asr_transcripts"] = transcripts
            if avg_wer > cfg.asr_max_avg_wer:
                quality.keep = False
                quality.reasons.append(f"asr_disagreement({avg_wer:.3f})")
        elif asr_fn is None:
            entry["quality"]["asr_pairwise_wer"] = None
            entry["quality"]["notes"].append("asr_stage_skipped(no asr_fn injected)")

        entry["quality"]["keep"] = quality.keep
        entry["quality"]["reasons"] = quality.reasons
        if quality.keep:
            report.n_kept += 1
            report.hours_kept += quality.duration_s / 3600.0
            if normalize:
                entry["normalized_peak_factor"] = cfg.peak_factor
            kept.append(entry)
        else:
            report.n_rejected += 1
            for reason in quality.reasons:
                key = reason.split("(")[0]
                report.reason_counts[key] = report.reason_counts.get(key, 0) + 1
            rejected.append(entry)

    (out_dir / "kept.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + ("\n" if kept else ""),
        encoding="utf-8",
    )
    (out_dir / "rejected.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rejected) + ("\n" if rejected else ""),
        encoding="utf-8",
    )
    (out_dir / "curation_report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    return report
