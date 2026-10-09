"""Phase-locking post-filter.

Paradee reports a residual "buzz" on voiced sounds that survives adversarial training and gets
*worse* with bigger decoders, and traces it to the phase of voiced speech between 2 and 8 kHz.
Their fix is a phase-locking filter applied after synthesis: no training, no extra parameters,
UTMOS 4.39 -> 4.41.

Our implementation of that idea:

* Work on the STFT.  Magnitudes are **never** touched (so loudness/timbre are unchanged).
* A glottal pulse train delayed by ``tau`` has the phase spectrum ``psi(f) ~= -2 pi f tau + c``
  i.e. phase is *linear in frequency*.  We search a small grid of ``tau`` per frame and keep
  the one that maximises the phase concentration
  ``R(tau) = | mean_f exp(i(psi(f) + 2 pi f tau)) |``.
* Replace the phase in the band by the best linear model, blended with weight
  ``strength * voicing`` so unvoiced fricatives keep their natural aperiodic phase.

:func:`phase_coherence` returns the same concentration statistic and is used by the tests and
the evaluation harness to quantify the buzz before/after.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..audio.f0 import estimate_f0


def _band_slice(sample_rate: int, n_fft: int, band: Tuple[float, float]) -> Tuple[int, int]:
    lo = int(round(band[0] / (sample_rate / n_fft)))
    hi = int(round(band[1] / (sample_rate / n_fft))) + 1
    return max(1, lo), min(n_fft // 2 + 1, hi)


@torch.no_grad()
def phase_coherence(
    wav: torch.Tensor,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    band: Tuple[float, float] = (2000.0, 8000.0),
    n_tau: int = 64,
) -> torch.Tensor:
    """Mean phase concentration in ``band`` (1.0 == perfectly locked, ~0 == random phase)."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft, hop_length, n_fft, window, return_complex=True)
    lo, hi = _band_slice(sample_rate, n_fft, band)
    sub = spec[:, lo:hi, :]
    freqs = torch.arange(lo, hi, device=wav.device, dtype=torch.float32) * (sample_rate / n_fft)
    taus = torch.linspace(0.0, 1.0 / 60.0, n_tau, device=wav.device)
    phase = torch.angle(sub)  # (B, Fb, T)
    e = torch.exp(1j * phase)
    # (B*T, Fb) x (Fb, G) -> (B*T, G)
    b, fb, t = e.shape
    d = torch.exp(1j * 2 * torch.pi * freqs[None, :] * taus[:, None])  # (G, Fb)
    r = (e.permute(0, 2, 1).reshape(b * t, fb) @ d.to(e.dtype).T).abs() / fb
    return r.max(dim=-1).values.mean()


def _hann_smooth(x: torch.Tensor, length: int) -> torch.Tensor:
    """Causal-ish moving average with a Hann kernel along the last axis."""
    if length <= 1:
        return x
    k = torch.hann_window(length, periodic=False, device=x.device, dtype=torch.float32)
    k = k / k.sum()
    real = torch.nn.functional.conv1d(
        x.real.reshape(-1, 1, x.shape[-1]), k.view(1, 1, -1), padding=length // 2
    ).reshape(x.shape[:-1] + (x.shape[-1],))
    imag = torch.nn.functional.conv1d(
        x.imag.reshape(-1, 1, x.shape[-1]), k.view(1, 1, -1), padding=length // 2
    ).reshape(x.shape[:-1] + (x.shape[-1],))
    return torch.complex(real, imag)


@torch.no_grad()
def phase_lock(
    wav: torch.Tensor,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    band: Tuple[float, float] = (2000.0, 8000.0),
    strength: float = 0.7,
    n_tau: int = 64,
    smooth_frames: int = 13,
    method: str = "ramp",
    f0: Optional[torch.Tensor] = None,
    voiced: Optional[torch.Tensor] = None,
    length: Optional[int] = None,
) -> torch.Tensor:
    """Apply the zero-parameter phase-locking filter to a waveform.

    Two interchangeable lock references (both leave the magnitude spectrum untouched):

    ``method="ramp"``
        Physically motivated: a glottal pulse train delayed by ``tau`` has phase
        ``psi(f) = -2 pi f tau + c``, i.e. *linear in frequency*.  We pick the ``tau`` that
        maximises ``|mean_f exp(i(psi + 2 pi f tau))|`` and use the resulting linear phase as
        the reference.
    ``method="smooth"``
        Paradee's variant: a locally smoothed complex reference ``E`` (Hann window of
        ``smooth_frames`` frames along time, L~9-17 in the paper, within the 2-8 kHz band),
        taking ``rel = S * conj(E) / |E|`` so that the lock reference is ``angle(E)``.

    ``strength`` blends the reference with the original phase; ``voiced`` (when supplied)
    reduces the correction on unvoiced frames so fricatives keep their aperiodic phase.
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    win_length = win_length or n_fft
    window = torch.hann_window(win_length, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(
        wav, n_fft, hop_length, win_length, window, center=True, return_complex=True
    )
    mag = spec.abs()
    phase = torch.angle(spec)

    lo, hi = _band_slice(sample_rate, n_fft, band)
    freqs = torch.arange(lo, hi, device=wav.device, dtype=torch.float32) * (sample_rate / n_fft)
    b, full_f, t = phase.shape

    if method == "smooth":
        e = _hann_smooth(spec[:, lo:hi, :], smooth_frames)
        lock_ref = torch.angle(e)
        # concentration of the phase around the smoothed reference drives the blend weight
        rel = spec[:, lo:hi, :] * torch.conj(e) / e.abs().clamp_min(1e-8)
        w_band = rel.real.clamp(-1.0, 1.0)
    elif method == "ramp":
        taus = torch.linspace(0.0, 1.0 / 60.0, n_tau, device=wav.device)
        e = torch.exp(1j * phase[:, lo:hi, :])  # (B, Fb, T)
        fb = e.shape[1]
        d = torch.exp(1j * 2 * torch.pi * freqs[None, :] * taus[:, None]).to(e.dtype)  # (G, Fb)
        proj = e.permute(0, 2, 1).reshape(b * t, fb) @ d.T  # (B*T, G)
        r = proj.abs() / fb
        best = r.argmax(dim=-1)
        tau = taus[best].view(b, t)
        c = proj.gather(1, best[:, None]).view(b, t) / fb
        lock_ref = torch.angle(c)[:, None, :] - 2 * torch.pi * freqs[None, :, None] * tau[:, None, :]
        w_band = r.max(dim=-1).values.view(b, t)
    else:
        raise ValueError(f"unknown phase-lock method {method!r}")

    w = (w_band * strength).clamp(0.0, 1.0)
    if voiced is not None:
        v = voiced.to(w.dtype)
        if v.shape[-1] != w.shape[-1]:
            v = torch.nn.functional.interpolate(
                v[:, None, :], size=w.shape[-1], mode="nearest"
            )[:, 0, :]
        w = w * (0.25 + 0.75 * v)  # keep partial locking on unvoiced/uncertain frames

    delta = torch.atan2(
        torch.sin(lock_ref - phase[:, lo:hi, :]), torch.cos(lock_ref - phase[:, lo:hi, :])
    )
    out_phase = phase.clone()
    out_phase[:, lo:hi, :] = phase[:, lo:hi, :] + w[:, None, :] * delta

    new_spec = torch.polar(mag, out_phase)
    out = torch.istft(
        new_spec, n_fft, hop_length, win_length, window, center=True, length=length or wav.shape[-1]
    )
    return out


def phase_lock_with_f0(
    wav: torch.Tensor,
    sample_rate: int = 24000,
    **kwargs,
) -> torch.Tensor:
    """Convenience wrapper that estimates F0/voicing first (when the caller has none)."""
    f0, voiced, _ = estimate_f0(
        wav, sample_rate, hop_length=kwargs.get("hop_length", 256), frame_length=kwargs.get("n_fft", 1024)
    )
    return phase_lock(wav, sample_rate=sample_rate, f0=f0, voiced=voiced, **kwargs)
