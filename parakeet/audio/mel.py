"""Log-mel front-end (pure torch, no torchaudio/librosa needed)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AudioConfig


def hz_to_mel(freq: torch.Tensor | float) -> torch.Tensor:
    """HTK mel scale."""
    freq_t = torch.as_tensor(freq, dtype=torch.float32)
    return 2595.0 * torch.log10(1.0 + freq_t / 700.0)


def mel_to_hz(mel: torch.Tensor | float) -> torch.Tensor:
    mel_t = torch.as_tensor(mel, dtype=torch.float32)
    return 700.0 * (torch.pow(10.0, mel_t / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    norm: str = "slaney",
) -> torch.Tensor:
    """Triangular mel filterbank, shape ``(n_mels, n_fft // 2 + 1)``."""
    fmax = float(sample_rate / 2) if fmax is None else float(fmax)
    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs)
    mel_pts = torch.linspace(float(hz_to_mel(fmin)), float(hz_to_mel(fmax)), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(n_mels):
        left, center, right = float(hz_pts[m]), float(hz_pts[m + 1]), float(hz_pts[m + 2])
        if right <= left:
            continue
        up = (fft_freqs - left).clamp(min=0.0) / max(center - left, 1e-8)
        down = (right - fft_freqs).clamp(min=0.0) / max(right - center, 1e-8)
        fb[m] = torch.minimum(up, down)
        if norm == "slaney":
            fb[m] *= 2.0 / max(right - left, 1e-8)
    return fb


class MelSpectrogram(nn.Module):
    """STFT -> mel.  ``center=True`` mirrors librosa/torchaudio defaults."""

    def __init__(self, cfg: AudioConfig, power: float = 1.0, center: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.power = power
        self.center = center
        window = torch.hann_window(cfg.win_length, periodic=True)
        self.register_buffer("window", window, persistent=False)
        self.register_buffer(
            "fb",
            mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax),
            persistent=False,
        )

    @property
    def n_mels(self) -> int:
        return self.cfg.n_mels

    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        """``(B, N) -> (B, F, T)`` complex spectrogram.  A ``(B, 1, N)`` input is accepted too."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 3 and wav.shape[1] == 1:
            wav = wav[:, 0, :]
        if wav.dim() != 2:
            raise ValueError(f"expected waveform of shape (B, N), got {tuple(wav.shape)}")
        return torch.stft(
            wav,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """``(B, N) -> (B, n_mels, T)`` linear-mel."""
        spec = self.stft(wav)
        mag = spec.abs()
        if self.power != 1.0:
            mag = mag.pow(self.power)
        return torch.matmul(self.fb, mag)

    def log_mel(self, wav: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
        eps = self.cfg.log_eps if eps is None else eps
        return torch.log(self.forward(wav).clamp_min(eps))

    def t_frames(self, n_samples: int) -> int:
        """Number of frames produced for ``n_samples`` input samples."""
        if self.center:
            return n_samples // self.cfg.hop_length + 1
        return max(0, (n_samples - self.cfg.n_fft) // self.cfg.hop_length + 1)


