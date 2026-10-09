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

from typing import Dict, Optional, Tuple

import torch

from ..audio.f0 import estimate_f0


def _band_slice(sample_rate: int, n_fft: int, band: Tuple[float, float]) -> Tuple[int, int]:
    lo = int(round(band[0] / (sample_rate / n_fft)))
    hi = int(round(band[1] / (sample_rate / n_fft))) + 1
    return max(1, lo), min(n_fft // 2 + 1, hi)


#: Per-call constants (window, band grid, delay grid and the complex search matrix) do not depend on
#: the audio, yet they were rebuilt on every call.  Profiling the shipped int8 pipeline showed
#: ``torch.exp``/``torch.polar`` on those grids costing ~0.8 ms of a 6.9 ms call -- 12 % of the
#: latency for arithmetic that never changes.  Cached per (device, dtype, geometry, method), bounded
#: so a caller sweeping band/method combinations cannot grow it without limit.
_CONSTANT_CACHE: Dict[tuple, Dict[str, torch.Tensor]] = {}
_CONSTANT_CACHE_LIMIT = 8


def _lock_constants(
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    band: Tuple[float, float],
    n_tau: int,
    device: torch.device,
    dtype: torch.dtype,
    method: str,
    kind: str = "filter",
) -> Dict[str, torch.Tensor]:
    # device and dtype are hashable as-is: building their str() per call was measurable overhead in
    # the very cache meant to remove overhead
    key = (
        kind, device, dtype, sample_rate, n_fft, hop_length, win_length,
        tuple(band), n_tau, method,
    )
    entry = _CONSTANT_CACHE.get(key)
    if entry is None:
        lo, hi = _band_slice(sample_rate, n_fft, band)
        complex_dtype = torch.complex64 if dtype is torch.float32 else torch.complex128
        freqs = torch.arange(lo, hi, device=device, dtype=torch.float32) * (sample_rate / n_fft)
        taus = torch.linspace(0.0, 1.0 / 60.0, n_tau, device=device)
        entry = {
            "lo": lo,
            "hi": hi,
            "freqs": freqs,
            "taus": taus,
            # the delay search matrix: (n_tau, band bins), the expensive part of the filter's setup
            "d": torch.exp(1j * 2 * torch.pi * freqs[None, :] * taus[:, None]).to(complex_dtype),
        }
        if len(_CONSTANT_CACHE) >= _CONSTANT_CACHE_LIMIT:
            _CONSTANT_CACHE.pop(next(iter(_CONSTANT_CACHE)))
        _CONSTANT_CACHE[key] = entry
    return entry


def clear_constant_cache() -> None:
    """Drop cached grids (for tests that assert on construction, or to free memory)."""
    _CONSTANT_CACHE.clear()


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
    const = _lock_constants(
        sample_rate, n_fft, hop_length, n_fft, band, n_tau, wav.device, wav.dtype, "ramp",
        kind="coherence",
    )
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft, hop_length, n_fft, window, return_complex=True)
    lo, hi = const["lo"], const["hi"]
    sub = spec[:, lo:hi, :]
    phase = torch.angle(sub)  # (B, Fb, T)
    e = torch.exp(1j * phase)
    # (B*T, Fb) x (Fb, G) -> (B*T, G)
    b, fb, t = e.shape
    d = const["d"].to(e.dtype)  # (G, Fb), cached
    r = (e.permute(0, 2, 1).reshape(b * t, fb) @ d.T).abs() / fb
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
    #: Delay-grid resolution for the linear-phase reference.  This is not an arbitrary knob: the
    #: achievable lock across a 2-8 kHz band is bounded by it.  With 64 points over [0, 1/60 s] the
    #: spacing is 260 us, i.e. more than two periods of phase error at 8 kHz.  Measured (round 12,
    #: scripts/phase_lock_ab.py): going 64 -> 256 at matched strength triples the coherence the
    #: filter adds to glottal-locked speech (+0.0058 -> +0.0120) while *reducing* what it adds to
    #: white noise (+0.0353 -> +0.0208) -- less of the effect is its own arithmetic.  Cost is
    #: ~1.5 -> 1.6 ms per audio second.
    n_tau: int = 256,
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
    original_length = wav.shape[-1]
    if original_length < n_fft:
        # ``torch.stft(center=True)`` pads n_fft//2 on each side, which is invalid for signals
        # shorter than n_fft -- and that is exactly what the final chunk of a streamed utterance
        # looks like.  Pad to one frame, filter, then trim.
        wav = torch.nn.functional.pad(wav, (0, n_fft - original_length))
    win_length = win_length or n_fft
    const = _lock_constants(
        sample_rate, n_fft, hop_length, win_length, band, n_tau, wav.device, wav.dtype, method
    )
    window = torch.hann_window(win_length, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(
        wav, n_fft, hop_length, win_length, window, center=True, return_complex=True
    )
    mag = spec.abs()
    phase = torch.angle(spec)

    lo, hi = const["lo"], const["hi"]
    freqs = const["freqs"]
    b, full_f, t = phase.shape

    if method == "smooth":
        e = _hann_smooth(spec[:, lo:hi, :], smooth_frames)
        lock_ref = torch.angle(e)
        # concentration of the phase around the smoothed reference drives the blend weight
        rel = spec[:, lo:hi, :] * torch.conj(e) / e.abs().clamp_min(1e-8)
        w_band = rel.real.clamp(-1.0, 1.0)
    elif method == "ramp":
        taus = const["taus"]
        e = torch.exp(1j * phase[:, lo:hi, :])  # (B, Fb, T)
        fb = e.shape[1]
        d = const["d"].to(e.dtype)  # (G, Fb), cached
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
    return out[..., :original_length]


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


class StreamingPhaseLock:
    """Chunked phase-lock filter with an ``n_fft`` look-ahead, for streaming synthesis.

    Offline synthesis filters the whole utterance; a streaming pipeline cannot wait for that.  This
    keeps one ``n_fft`` of unemitted samples as overlap, filters the buffer, and emits everything
    except the overlap -- so every emitted sample has a full right-hand context and the interior of
    the stream is identical to what the offline filter would produce.  Only the very first frames of
    the stream (where the offline filter also sees centre padding) and the buffer boundaries are
    approximate.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        band: Tuple[float, float] = (2000.0, 8000.0),
        strength: float = 0.7,
        method: str = "ramp",
        #: must match :func:`phase_lock`'s default: the delay-grid resolution bounds the achievable
        #: lock, and a coarse grid mostly adds phase structure the signal never had (see the A/B in
        #: scripts/phase_lock_ab.py)
        n_tau: int = 256,
        smooth_frames: int = 13,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.band = band
        self.strength = strength
        self.method = method
        self.n_tau = n_tau
        self.smooth_frames = smooth_frames
        self._pending: Optional[torch.Tensor] = None
        self.n_emitted = 0

    def _filter(self, wav: torch.Tensor) -> torch.Tensor:
        return phase_lock(
            wav,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            band=self.band,
            strength=self.strength,
            method=self.method,
            n_tau=self.n_tau,
            smooth_frames=self.smooth_frames,
        )

    @torch.no_grad()
    def push(self, chunk: torch.Tensor) -> torch.Tensor:
        """Filter ``chunk`` and return the newly finalised samples (possibly empty)."""
        x = chunk.reshape(1, -1) if chunk.dim() == 1 else chunk
        buf = x if self._pending is None else torch.cat([self._pending, x], dim=-1)
        if buf.shape[-1] <= self.n_fft:
            self._pending = buf
            return buf.new_zeros(buf.shape[0], 0)
        filtered = self._filter(buf)
        keep = buf.shape[-1] - self.n_fft
        self._pending = buf[..., keep:]
        out = filtered[..., :keep]
        self.n_emitted += out.shape[-1]
        return out

    @torch.no_grad()
    def flush(self) -> torch.Tensor:
        if self._pending is None or self._pending.shape[-1] == 0:
            return torch.zeros(1, 0)
        out = self._filter(self._pending)
        self._pending = None
        return out
