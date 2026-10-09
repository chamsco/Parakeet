"""Overlap-add iSTFT that works both offline and *streaming*.

The SupertonicTTS decoder is causal so that audio can be emitted before the full latent is
available.  To actually stream we also need an iSTFT that can be pushed frame-by-frame; the
implementations below share one code path so that streaming output is numerically identical
to the offline output (up to float32 accumulation order).
"""

from __future__ import annotations

import math
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

    def push(self, spec: torch.Tensor) -> torch.Tensor:
        """``(B, F, Tc)`` complex frames -> new samples ``(B, n_new)``."""
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        tc = spec.shape[-1]
        if tc == 0:
            b = spec.shape[0]
            return torch.zeros(b, 0, dtype=self.window.dtype, device=self.window.device)

        frames = torch.fft.irfft(spec.transpose(1, 2), n=self.n_fft)  # (B, Tc, n_fft)
        b = frames.shape[0]
        frames = frames * self.window
        env = self.window.pow(2).expand(tc, self.n_fft)

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
        sig_fold = F.fold(
            frames.transpose(1, 2),  # (B, n_fft, Tc)
            output_size=(1, total),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        )  # (B, 1, 1, total)
        env_fold = F.fold(
            env.transpose(0, 1).unsqueeze(0).expand(1, self.n_fft, tc),  # (1, n_fft, Tc)
            output_size=(1, total),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        )
        self._y[..., start : start + total] += sig_fold.view(b, 1, total)
        self._e[..., start : start + total] += env_fold.view(1, 1, total)
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

    def forward(self, spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        if spec.dim() == 2:
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
        y = st.push(spec)
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


