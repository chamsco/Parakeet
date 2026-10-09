"""F0 / energy extraction, and the F0 quantisation used by the Tiny distillation path.

Paradee caches *durations, pitch, energy and phoneme features* from the teacher and trains a
small text side to regress them.  This module provides the pitch/energy half of that cache
without a heavy dependency.  For research-grade F0 we recommend ``praat-parselmouth`` or
``pyin``; :func:`estimate_f0` is a normalized-autocorrelation estimator that is good enough
for regression targets and for the phase-locking filter.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def frame_energy_db(
    wav: torch.Tensor,
    frame_length: int = 1024,
    hop_length: int = 256,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Frame RMS in dB.  ``(B, N) -> (B, T)``."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    pad = frame_length // 2
    x = F.pad(wav, (pad, pad))
    frames = x.unfold(-1, frame_length, hop_length)
    rms = frames.pow(2).mean(dim=-1).clamp_min(eps).sqrt()
    return 20.0 * torch.log10(rms)


def _frames(wav: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    pad = frame_length // 2
    x = F.pad(wav.unsqueeze(-2), (pad, pad)) if False else F.pad(wav, (pad, pad))
    return x.unfold(-1, frame_length, hop_length)


@torch.no_grad()
def estimate_f0_yin(
    wav: torch.Tensor,
    sample_rate: int,
    hop_length: int = 256,
    frame_length: int = 1024,
    fmin: float = 60.0,
    fmax: float = 500.0,
    threshold: float = 0.50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """YIN pitch tracker (de Cheveigné & Kawahara, 2002).

    YIN is used instead of autocorrelation because autocorrelation has a **formant bias**: for a
    strongly formant-shaped harmonic stack the tallest normalized peak can land on a subharmonic or
    a formant period, which is not a corner case here -- it is the whole signal model.  Measured on
    the fixture voices, the autocorrelation estimator reported 134.5 Hz for an 81 Hz voice, and
    those wrong values become the F0 *targets* the student is trained to imitate.

    The default CMND threshold is 0.50 rather than the paper's canonical 0.10-0.15 because this
    implementation additionally requires a **local minimum** below the threshold, a stricter
    criterion than the paper's.  Measured on real Kokoro speech (round 17): raising it from 0.15 to
    0.55 lifts the voiced fraction from 0.39 to 0.81 (af_heart), 0.08 to 0.73 (af_sky) and 0.23 to
    0.50 (af_bella) **while the median F0 stays flat** (204->199, 100->105, 158->159 Hz) -- the
    extra frames are genuinely voiced, not noise.  At the previous 0.25 most real speech was
    discarded, so per-token F0 targets were averaged over a small minority of frames.  Above ~0.70
    the median starts moving (142 Hz for af_sky), so the useful range is 0.45-0.55.

    Returns ``(f0_hz, voiced, confidence)`` of shape ``(B, T)``; unvoiced frames are 0.
    """
    frames = _frames(wav, frame_length, hop_length)
    if frames.numel() == 0:
        empty = torch.zeros(wav.shape[0], 0)
        return empty, empty.bool(), empty
    frames = frames - frames.mean(dim=-1, keepdim=True)
    b, t, w = frames.shape

    # difference function d(tau) = sum_j (x[j] - x[j+tau])^2, computed from the autocorrelation
    nfft = 1 << max(1, (2 * w - 1).bit_length())
    spec = torch.fft.rfft(frames, n=nfft, dim=-1)
    acf = torch.fft.irfft(spec * spec.conj(), n=nfft, dim=-1)[..., :w]
    squares = frames.pow(2)
    prefix = F.pad(torch.cumsum(squares, dim=-1), (1, 0))  # prefix[k] = sum_{j<k} x[j]^2
    tau = torch.arange(w, device=frames.device)
    term1 = prefix[..., (w - tau).clamp(min=0)]  # sum_{j < w - tau}
    term2 = prefix[..., w : w + 1] - prefix[..., tau]  # sum_{j >= tau}
    diff = (term1 + term2 - 2 * acf).clamp_min(0.0)
    diff[..., 0] = 0.0

    # cumulative mean normalised difference
    cumsum = torch.cumsum(diff, dim=-1)
    denom = cumsum / tau.clamp_min(1).to(diff.dtype)
    cmnd = diff / denom.clamp_min(1e-9)
    cmnd[..., 0] = 1.0

    lo = max(2, int(sample_rate / fmax))
    hi = min(w - 2, int(sample_rate / fmin))
    if hi <= lo:
        empty = torch.zeros(b, t, device=wav.device)
        return empty, empty.bool(), empty
    window = cmnd[..., lo : hi + 1]

    # YIN step 3-4: the *first local minimum* of the CMND below the absolute threshold (falling
    # back to the global minimum).  Taking the first threshold *crossing* instead lands short of the
    # true period -- on a pure 200 Hz tone it reported 226 Hz.
    prev = torch.cat([window[..., :1], window[..., :-1]], dim=-1)
    nxt = torch.cat([window[..., 1:], window[..., -1:]], dim=-1)
    is_min = (window <= prev) & (window <= nxt)
    below = is_min & (window < threshold)
    has_below = below.any(dim=-1)
    idx = torch.where(has_below, below.float().argmax(dim=-1), window.argmin(dim=-1)) + lo
    idx = idx.clamp(min=lo + 1, max=hi - 1)

    # parabolic refinement on the *difference* function (smooth near the minimum), not the CMND
    left = torch.gather(diff, -1, (idx - 1).unsqueeze(-1)).squeeze(-1)
    centre = torch.gather(diff, -1, idx.unsqueeze(-1)).squeeze(-1)
    right = torch.gather(diff, -1, (idx + 1).unsqueeze(-1)).squeeze(-1)
    denom_p = (left - 2 * centre + right).abs().clamp_min(1e-12)
    delta = (0.5 * (left - right) / denom_p).clamp(-1.0, 1.0)
    tau_best = idx.to(wav.dtype) + delta

    f0 = sample_rate / tau_best.clamp_min(1.0)
    confidence = (1.0 - torch.gather(cmnd, -1, idx.unsqueeze(-1)).squeeze(-1)).clamp(0.0, 1.0)
    voiced = has_below & (f0 >= fmin) & (f0 <= fmax)
    f0 = torch.where(voiced, f0, torch.zeros_like(f0))
    return f0, voiced, confidence


@torch.no_grad()
def estimate_f0(
    wav: torch.Tensor,
    sample_rate: int,
    hop_length: int = 256,
    frame_length: int = 1024,
    fmin: float = 60.0,
    fmax: float = 500.0,
    threshold: Optional[float] = None,
    method: str = "yin",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pitch tracking.  ``method="yin"`` (default) or ``"autocorr"`` (legacy, formant-biased).

    ``threshold`` defaults are **per method** and deliberately different: 0.50 for the CMND of YIN
    (its local-minimum requirement makes it stricter than the paper's threshold; see
    :func:`estimate_f0_yin` for the real-speech measurement behind that number) and 0.30 for the
    normalised autocorrelation peak, its historical value.  A single shared default would silently
    make one of the two paths wrong.

    Returns ``(f0_hz, voiced, confidence)`` each of shape ``(B, T)``; unvoiced frames have
    ``f0_hz == 0``.
    """
    if method == "yin":
        return estimate_f0_yin(
            wav, sample_rate, hop_length, frame_length, fmin, fmax,
            threshold=0.50 if threshold is None else threshold,
        )
    if method != "autocorr":
        raise ValueError(f"unknown pitch method {method!r}")
    threshold = 0.30 if threshold is None else threshold

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    n = wav.shape[-1]
    pad = frame_length // 2
    x = F.pad(wav, (pad, pad))
    frames = x.unfold(-1, frame_length, hop_length)  # (B, T, L)
    frames = frames - frames.mean(dim=-1, keepdim=True)
    win = torch.hann_window(frame_length, device=wav.device, dtype=wav.dtype)
    frames = frames * win
    nfft = 1 << max(1, (2 * frame_length - 1).bit_length())
    spec = torch.fft.rfft(frames, n=nfft, dim=-1)
    acf = torch.fft.irfft(spec * spec.conj(), n=nfft, dim=-1)[..., :frame_length]
    zero = acf[..., :1].clamp_min(1e-8)
    acf = acf / zero

    min_lag = max(2, int(sample_rate / fmax))
    max_lag = min(frame_length - 2, int(sample_rate / fmin))
    segment = acf[..., min_lag:max_lag]

    best_val, best_idx = segment.max(dim=-1)
    lag = (best_idx + min_lag).to(wav.dtype)

    left = torch.gather(acf, -1, (best_idx + min_lag - 1).clamp(min=0).long().unsqueeze(-1)).squeeze(-1)
    right = torch.gather(
        acf, -1, (best_idx + min_lag + 1).clamp(max=frame_length - 1).long().unsqueeze(-1)
    ).squeeze(-1)
    denom = (left - 2 * best_val + right).abs().clamp_min(1e-8)
    delta = 0.5 * (left - right) / denom
    delta = delta.clamp(-0.5, 0.5)
    lag = (lag + delta).clamp(min=1.0)

    f0 = sample_rate / lag
    voiced = (best_val > threshold) & (f0 >= fmin) & (f0 <= fmax)
    f0 = torch.where(voiced, f0, torch.zeros_like(f0))
    return f0, voiced, best_val.clamp(0.0, 1.0)


def f0_to_normalized(
    f0_hz: torch.Tensor,
    voiced: torch.Tensor | None = None,
    fmin: float = 60.0,
    fmax: float = 500.0,
) -> torch.Tensor:
    """Map F0 to a continuous, *O(1)* regression target in ``[0, 1]`` (0 == unvoiced).

    This is the training target for the Tiny text side.  Regressing raw Hz (or worse, quantised
    bin *indices* that run to 256) makes one loss term dominate every other by two orders of
    magnitude -- Paradee regresses ``F0/100`` for the same reason.  Log-spacing matches pitch
    perception, so an L1 error means roughly the same thing across the range.
    """
    f0 = f0_hz.to(torch.float32)
    if voiced is not None:
        f0 = torch.where(voiced.to(torch.bool), f0, torch.zeros_like(f0))
    lo = math.log2(max(fmin, 1e-6))
    hi = math.log2(max(fmax, fmin * 1.001))
    scaled = (torch.log2(f0.clamp_min(1e-6)) - lo) / max(hi - lo, 1e-6)
    return torch.where(f0 > 0, scaled.clamp(0.0, 1.0), torch.zeros_like(scaled))


def normalized_to_f0(
    value: torch.Tensor,
    fmin: float = 60.0,
    fmax: float = 500.0,
) -> torch.Tensor:
    """Inverse of :func:`f0_to_normalized` (0 -> 0 Hz, i.e. unvoiced)."""
    lo = math.log2(max(fmin, 1e-6))
    hi = math.log2(max(fmax, fmin * 1.001))
    v = value.to(torch.float32).clamp(0.0, 1.0)
    f0 = torch.pow(2.0, lo + v * (hi - lo))
    return torch.where(value > 0, f0, torch.zeros_like(f0))


def energy_to_normalized(
    energy_db: torch.Tensor, floor_db: float = -60.0, ceil_db: float = 0.0
) -> torch.Tensor:
    """Map frame/token energy in dBFS to an *O(1)* target in ``[0, 1]``.

    Same reasoning as :func:`f0_to_normalized`: regressing raw dB puts a term of magnitude ~20-40
    into an objective whose other terms are ~1, so the loss becomes a single-term loss in disguise
    and the other heads stop receiving useful gradient.  (This is the failure the learning tests
    caught: with raw targets the "distillation" loss was 141, of which pitch alone was ~110.)
    """
    e = energy_db.to(torch.float32)
    return ((e - floor_db) / max(ceil_db - floor_db, 1e-6)).clamp(0.0, 1.0)


def normalized_to_energy(
    value: torch.Tensor, floor_db: float = -60.0, ceil_db: float = 0.0
) -> torch.Tensor:
    """Inverse of :func:`energy_to_normalized`, in dBFS."""
    v = value.to(torch.float32).clamp(0.0, 1.0)
    return floor_db + v * (ceil_db - floor_db)


def f0_to_bins(
    f0_hz: torch.Tensor,
    voiced: torch.Tensor | None = None,
    fmin: float = 60.0,
    fmax: float = 500.0,
    n_bins: int = 256,
) -> torch.Tensor:
    """Quantise log-F0 into ``n_bins`` bins (0 == unvoiced), as an integer code.

    Quantised pitch makes the Tiny text-side regression target robust to mis-voicing in the
    teacher cache (it behaves like a coarse pitch contour rather than a fragile float).
    """
    up = max(2 * n_bins, 65536)
    f0 = f0_hz.to(torch.float32)
    if voiced is not None:
        f0 = torch.where(voiced, f0, torch.zeros_like(f0))
    log_f0 = torch.log2(f0.clamp_min(fmin))
    lo, hi = float(torch.log2(torch.tensor(fmin + 1e-6))), float(torch.log2(torch.tensor(fmax)))
    scaled = (log_f0 - lo) / max(hi - lo, 1e-6)
    bins = torch.round(scaled * (n_bins - 1)).clamp(0, n_bins - 1)
    bins = torch.where(f0 > 0, bins + 1, torch.zeros_like(bins))
    del up
    return bins.to(torch.long)


def bins_to_f0(
    bins: torch.Tensor,
    fmin: float = 60.0,
    fmax: float = 500.0,
    n_bins: int = 256,
) -> torch.Tensor:
    """Inverse of :func:`f0_to_bins` (bin 0 -> 0 Hz, i.e. unvoiced)."""
    lo, hi = float(torch.log2(torch.tensor(fmin + 1e-6))), float(torch.log2(torch.tensor(fmax)))
    b = bins.to(torch.float32) - 1.0
    scaled = b / max(n_bins - 1, 1)
    f0 = torch.pow(2.0, lo + scaled * (hi - lo))
    return torch.where(bins > 0, f0, torch.zeros_like(f0))


