"""Overlap-add iSTFT that works both offline and *streaming*.

The SupertonicTTS decoder is causal so that audio can be emitted before the full latent is
available.  To actually stream we also need an iSTFT that can be pushed frame-by-frame; the
implementations below share one code path so that streaming output is numerically identical
to the offline output (up to float32 accumulation order).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_window(n_fft: int, win_length: Optional[int], device, dtype) -> torch.Tensor:
    win_length = n_fft if win_length is None else win_length
    window = torch.hann_window(win_length, periodic=True, device=device, dtype=dtype)
    if win_length < n_fft:
        pad = (n_fft - win_length) // 2
        window = F.pad(window, (pad, n_fft - win_length - pad))
    return window


def device_supports_complex(device) -> bool:
    """Can this device hold a complex tensor?

    DirectML cannot ("Invalid or unsupported data type ComplexFloat"), and it is the only route into
    this machine's Radeon from Windows PyTorch.  The autoencoder's head is `torch.polar` + iSTFT and
    the mel/spectral losses are `torch.stft`, so without this check the GPU could not train anything
    that touches the decoder -- even though the decoder's conv stack is **6.1x faster** there than on
    the CPU (measured: 67 ms -> 11 ms for 4x900 frames).
    """
    return torch.device(device).type in {"cpu", "cuda", "mps", "xla"}


@lru_cache(maxsize=16)
def _inverse_dft_basis(n_fft: int, dtype_name: str) -> tuple:
    """Cosine and sine bases for a **real-valued** inverse DFT of a real signal's half spectrum.

    ``irfft`` normalisation is folded in: the real and imaginary parts of the half spectrum multiply
    ``cos(2*pi*k*n/N)`` and ``-sin(2*pi*k*n/N)``, with unity weight at DC and Nyquist and 2x for the
    interior bins, all scaled by 1/N.
    """
    dtype = getattr(torch, dtype_name)
    n = torch.arange(n_fft, dtype=torch.float64)
    k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
    angle = 2.0 * math.pi * k[:, None] * n[None, :] / n_fft
    weight = torch.full((n_fft // 2 + 1,), 2.0, dtype=torch.float64)
    weight[0] = 1.0
    if n_fft % 2 == 0:
        weight[-1] = 1.0
    cos_basis = (weight[:, None] * torch.cos(angle)) / n_fft
    sin_basis = (-weight[:, None] * torch.sin(angle)) / n_fft
    return cos_basis.to(dtype), sin_basis.to(dtype)


def _forward_dft_basis(n_fft: int, dtype_name: str) -> tuple:
    """Cosine and sine bases for a **real-valued** forward DFT of a real frame."""
    dtype = getattr(torch, dtype_name)
    n = torch.arange(n_fft, dtype=torch.float64)
    k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
    angle = 2.0 * math.pi * k[None, :] * n[:, None] / n_fft
    return torch.cos(angle).to(dtype), (-torch.sin(angle)).to(dtype)


def real_magnitude_spectrogram(
    wav: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int],
    window: torch.Tensor,
    center: bool = True,
) -> torch.Tensor:
    """``(B, N)`` waveform -> ``(B, F, T)`` magnitude, matching ``torch.stft(...).abs()``.

    Framing and padding mirror `torch.stft`: reflection padding of ``n_fft // 2`` on both sides when
    ``center``, then `unfold` with the hop, then the window, then the DFT as a matrix multiply.  The
    normalisation is the plain (unnormalised) DFT, so magnitudes are directly comparable.
    """
    if wav.dim() == 3 and wav.shape[1] == 1:
        wav = wav[:, 0, :]
    if center:
        pad = n_fft // 2
        if wav.shape[-1] <= pad:
            wav = F.pad(wav, (pad, pad), mode="replicate")
        else:
            wav = F.pad(wav, (pad, pad), mode="reflect")
    frames = wav.unfold(-1, n_fft, hop_length)  # (B, T, n_fft)
    if win_length is not None and win_length != n_fft:
        pad = (n_fft - win_length) // 2
        window = F.pad(window, (pad, n_fft - win_length - pad))
    frames = frames * window
    cos_basis, sin_basis = _forward_dft_basis(n_fft, str(wav.dtype).replace("torch.", ""))
    cos_basis = cos_basis.to(wav.device)
    sin_basis = sin_basis.to(wav.device)
    real = frames @ cos_basis  # (B, T, F)
    imag = frames @ sin_basis
    return torch.sqrt(real.pow(2) + imag.pow(2) + 1e-12).transpose(1, 2)  # (B, F, T)


def irfft_frames(real: torch.Tensor, imag: torch.Tensor, n_fft: int) -> torch.Tensor:
    """``(B, F, T)`` real/imag -> ``(B, T, n_fft)`` frames, without a complex dtype.

    Uses ``torch.fft.irfft`` where the device supports it (exact, and fast) and a real matrix
    multiply where it does not; `tests/test_real_istft.py` asserts the two agree.
    """
    if device_supports_complex(real.device):
        spec = torch.complex(real, imag)
        return torch.fft.irfft(spec.transpose(1, 2), n=n_fft)
    cos_basis, sin_basis = _inverse_dft_basis(n_fft, str(real.dtype).replace("torch.", ""))
    cos_basis = cos_basis.to(real.device)
    sin_basis = sin_basis.to(real.device)
    r = real.transpose(1, 2)
    i = imag.transpose(1, 2)
    return r @ cos_basis + i @ sin_basis


_IDENTITY_CACHE: dict = {}


def _identity_kernel(n_fft: int, dtype, device) -> torch.Tensor:
    """``(n_fft, 1, n_fft)`` identity kernel for the transposed-convolution overlap-add.

    Built on the CPU and cached: `torch.eye` is **broken on DirectML** -- it falls back to the CPU and
    returns an *empty* tensor ``(0,)``, so the kernel has to be made where the op works.
    """
    key = (n_fft, str(dtype))
    cached = _IDENTITY_CACHE.get(key)
    if cached is None:
        cached = torch.eye(n_fft, dtype=torch.float32).unsqueeze(1)
        _IDENTITY_CACHE[key] = cached
    return cached.to(dtype=dtype, device=device)


def _overlap_add(frames: torch.Tensor, hop: int, total: int) -> torch.Tensor:
    """``(B, Tc, n_fft)`` frames -> ``(B, total)`` overlap-added signal (no windowing).

    Each frame is placed at ``t*hop`` and added in, which is a transposed convolution with an identity
    kernel.  It is written this way rather than with `F.fold` because fold's backward is
    ``aten::col2im``, which DirectML does not implement -- and the plugin's CPU fallback leaves the
    autograd graph crossing devices, after which the backward pass dies inside the plugin's own error
    handler (a `UnicodeDecodeError` decoding a Windows error string, which is how this was found).
    `conv_transpose1d` has the same forward and a backward the device supports.
    """
    b, tc, n_fft = frames.shape
    weight = _identity_kernel(n_fft, frames.dtype, frames.device)
    out = F.conv_transpose1d(frames.transpose(1, 2), weight, stride=hop)
    return out[:, 0, :total]


def _envelope(tc: int, window: torch.Tensor, hop: int, total: int, device) -> torch.Tensor:
    """``(1, 1, total)`` sum of squared windows covering each sample.

    Needs no gradient (it depends only on the window, the hop and the frame count), so it is built on
    the CPU and cached rather than recomputed on the device every step.
    """
    key = (tc, hop, total)
    cached = _ENVELOPE_CACHE.get(key)
    if cached is None:
        profile = window.pow(2).to(torch.float64).expand(1, tc, window.numel()).contiguous()
        cached = _overlap_add(profile, hop, total).view(1, 1, total).to(window.dtype)
        _ENVELOPE_CACHE[key] = cached
    return cached.to(device)


_ENVELOPE_CACHE: dict = {}


class StreamingOLA:
    """Incremental overlap-add with a growing (but trimmed) buffer.

    Feed complex spectrogram frames with :meth:`push`; it returns the waveform samples that
    became fully determined by the frames pushed so far.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int] = None,
        center: bool = True,
        device=None,
        dtype=torch.float32,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.window = _make_window(n_fft, win_length, device, dtype)
        self.reset()

    def reset(self) -> None:
        self._y: Optional[torch.Tensor] = None  # (B, 1, L) contiguous overlap-added signal
        self._e: Optional[torch.Tensor] = None  # (B, 1, L) window energy envelope
        self._n_frames = 0
        self._n_emitted = 0
        self._padded = self.n_fft // 2 if self.center else 0

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def push(self, spec: torch.Tensor, imag: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``(B, F, Tc)`` frames -> new samples ``(B, n_new)``.

        Either a complex spectrum, or its real part with the imaginary part passed separately, for
        devices with no complex dtype (see :func:`device_supports_complex`).
        """
        if imag is not None:
            real = spec
        else:
            real, imag = spec.real, spec.imag
        if real.dim() == 2:
            real = real.unsqueeze(0)
            imag = imag.unsqueeze(0)
        tc = real.shape[-1]
        if tc == 0:
            b = real.shape[0]
            return torch.zeros(b, 0, dtype=self.window.dtype, device=self.window.device)

        frames = irfft_frames(real, imag, self.n_fft)  # (B, Tc, n_fft)
        b = frames.shape[0]
        frames = frames * self.window

        total = (tc - 1) * self.hop_length + self.n_fft
        if self._y is None:
            self._y = torch.zeros(b, 1, total, dtype=frames.dtype, device=frames.device)
            self._e = torch.zeros_like(self._y)
            self._base = 0
        else:
            needed = (self._n_frames + tc - 1) * self.hop_length + self.n_fft
            cur = self._y.shape[-1]
            if needed > cur:
                self._y = F.pad(self._y, (0, needed - cur))
                self._e = F.pad(self._e, (0, needed - cur))

        start = self._n_frames * self.hop_length
        self._y[..., start : start + total] += _overlap_add(
            frames, self.hop_length, total
        ).view(b, 1, total)
        self._e[..., start : start + total] += _envelope(
            tc, self.window, self.hop_length, total, frames.device
        )
        self._n_frames += tc

        # samples fully covered by every frame that can touch them: s < n_frames * hop
        determined = self._n_frames * self.hop_length
        out = self._y[..., :determined] / self._e[..., :determined].clamp_min(1e-8)
        new = out[..., self._n_emitted :]
        self._n_emitted = determined
        return new[:, 0, :]

    def finalize(self) -> torch.Tensor:
        """Return the remaining (tail) samples and stop."""
        if self._y is None:
            return torch.zeros(1, 0)
        total = self._n_frames * self.hop_length
        out = (self._y[..., :total] / self._e[..., :total].clamp_min(1e-8))[..., self._n_emitted :]
        self._n_emitted = total
        return out[:, 0, :]


class OLAISTFT(nn.Module):
    """Stateless iSTFT matching ``torch.istft(..., center=center)`` semantics."""

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        center: bool = True,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.center = center
        self.register_buffer(
            "window", _make_window(n_fft, win_length, None, torch.float32), persistent=False
        )

    def forward(
        self, spec: torch.Tensor, length: Optional[int] = None, imag: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if imag is None and spec.dim() == 2:
            spec = spec.unsqueeze(0)
        tc = spec.shape[-1]
        st = StreamingOLA(
            self.n_fft,
            self.hop_length,
            self.win_length,
            center=False,
            device=spec.device,
            dtype=self.window.dtype,
        )
        y = st.push(spec, imag=imag)
        y = y[..., : tc * self.hop_length]
        if self.center:
            trim = self.n_fft // 2
            y = y[..., trim:]
        if length is not None:
            if y.shape[-1] < length:
                y = F.pad(y, (0, length - y.shape[-1]))
            else:
                y = y[..., :length]
        return y

    def output_length(self, n_frames: int) -> int:
        if self.center:
            return n_frames * self.hop_length
        return (n_frames - 1) * self.hop_length + self.n_fft


