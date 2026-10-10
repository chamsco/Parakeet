"""Speech autoencoder: mel -> continuous latent -> waveform.

Architecture follows SupertonicTTS (arXiv 2503.23108) §3.1, which itself builds on Vocos:
a ConvNeXt encoder producing a *low-dimensional* latent at the mel frame rate, and a
**causal dilated** ConvNeXt decoder with an iSTFT head, so the decoder can stream.

Why a continuous latent instead of discrete codec tokens?
  * It is teacher-agnostic: any teacher's audio can be re-encoded, which is what lets us
    distill a *mixture* of teachers (Orpheus / MiniMax / Kokoro) into one student.
  * No codec-LM vocabulary mismatch, no delayed/interleaved codebook schedules to reproduce.
  * At 24 dims and a 256-hop at 24 kHz the latent is ~2.2k floats/s (~8.4 kB/s fp32),
    cheap enough to predict with flow matching on a CPU-class budget.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio.istft import OLAISTFT, device_supports_complex
from ..config import AudioConfig, AutoencoderConfig
from .blocks import CausalConv1d, ConvNeXtBlock


class SpeechAutoencoder(nn.Module):
    def __init__(self, audio: AudioConfig, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.audio = audio
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        n_freq = audio.n_fft // 2 + 1

        # ---- encoder (non-causal: it sees the whole utterance) ----
        dims = list(cfg.encoder_dims)
        self.stem = nn.Conv1d(
            audio.n_mels, dims[0], cfg.kernel_size, padding=cfg.kernel_size // 2
        )
        enc: list[nn.Module] = []
        for i, (dim, n_blocks) in enumerate(zip(dims, cfg.encoder_blocks)):
            if i > 0:
                enc.append(nn.Conv1d(dims[i - 1], dim, 3, padding=1))
            enc.extend(
                ConvNeXtBlock(dim, cfg.kernel_size, layer_scale_init=cfg.layer_scale_init)
                for _ in range(n_blocks)
            )
        self.encoder = nn.Sequential(*enc)
        self.to_latent = nn.Conv1d(dims[-1], cfg.latent_dim, 1)

        # ---- decoder (causal: streaming capable) ----
        conv_cls = CausalConv1d if cfg.causal_decoder else nn.Conv1d
        self.from_latent = nn.Conv1d(cfg.latent_dim, cfg.decoder_dim, 1)
        dilations = list(cfg.decoder_dilations)
        self.decoder = nn.Sequential(
            *[
                ConvNeXtBlock(
                    cfg.decoder_dim,
                    cfg.kernel_size,
                    expansion=cfg.decoder_expansion,
                    dilation=dilations[i % len(dilations)],
                    causal=cfg.causal_decoder,
                    layer_scale_init=cfg.layer_scale_init,
                )
                for i in range(cfg.decoder_blocks)
            ]
        )
        self.head = conv_cls(cfg.decoder_dim, 2 * n_freq, cfg.kernel_size)
        self.istft = OLAISTFT(audio.n_fft, audio.hop_length, audio.win_length, center=True)
        self._receptive_field = self._compute_receptive_field()

    # ------------------------------------------------------------------ helpers
    def _compute_receptive_field(self) -> int:
        rf = 0
        for m in self.decoder:
            if isinstance(m, ConvNeXtBlock):
                rf += m.dwconv.receptive_field
        rf += getattr(self.head, "receptive_field", 0)
        return rf

    @property
    def latent_receptive_field(self) -> int:
        """Latent frames of history the decoder needs to emit one new frame."""
        return self._receptive_field

    def latent_length(self, n_samples: int) -> int:
        """Latent frames produced for ``n_samples`` of audio (center-padded STFT)."""
        return n_samples // self.audio.hop_length + 1

    def waveform_length(self, n_latent_frames: int) -> int:
        return self.istft.output_length(n_latent_frames)

    # ------------------------------------------------------------------ forward
    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """``(B, n_mels, T) -> (B, latent_dim, T)``."""
        h = self.stem(mel)
        h = self.encoder(h)
        return self.to_latent(h)

    def spectrogram(
        self, latent: torch.Tensor, length: Optional[int] = None
    ) -> torch.Tensor:
        """``(B, latent_dim, T) -> complex spec (B, F, T)`` (log-magnitude + phase head).

        On a device with no complex dtype (DirectML) this returns the **real** part and the imaginary
        part is obtained from :meth:`spectrogram_parts`; callers should use that instead when the
        device cannot hold complex tensors.
        """
        real, imag = self.spectrogram_parts(latent)
        if not device_supports_complex(real.device):
            raise RuntimeError(
                "this device has no complex dtype; use spectrogram_parts()/decode() instead of "
                "spectrogram()"
            )
        return torch.complex(real, imag)

    def spectrogram_parts(self, latent: torch.Tensor):
        """``(B, latent_dim, T) -> (real, imag)`` of the (B, F, T) spectrogram.

        Split out of :meth:`spectrogram` because `torch.polar` needs a complex dtype, which DirectML
        does not have -- and the conv stack feeding it is 6.1x faster on this machine's GPU than on
        the CPU.  The magnitude/phase head is unchanged, so an autoencoder trained with the complex
        path keeps working: the two are the same computation.
        """
        h = self.from_latent(latent)
        h = self.decoder(h)
        out = self.head(h)
        log_mag, phase = out.chunk(2, dim=1)
        mag = torch.exp(log_mag.clamp(max=8.0))
        return mag * torch.cos(phase), mag * torch.sin(phase)

    def decode(self, latent: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """``(B, latent_dim, T) -> waveform (B, N)``."""
        real, imag = self.spectrogram_parts(latent)
        if device_supports_complex(real.device):
            return self.istft(torch.complex(real, imag), length=length)
        return self.istft(real, length=length, imag=imag)

    def forward(self, mel: torch.Tensor, length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(mel)
        return self.decode(latent, length=length), latent


class LatentNormalizer(nn.Module):
    """Running normalisation of the latent space (stabilises flow-matching training)."""

    def __init__(self, latent_dim: int, momentum: float = 0.99, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("var", torch.ones(latent_dim))
        self.register_buffer("n", torch.zeros(1))
        self.momentum = momentum
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        # x: (B, C, T)
        flat = x.transpose(1, 2).reshape(-1, x.shape[1])
        m = flat.mean(0)
        v = flat.var(0, unbiased=False)
        if float(self.n) < 1.0:
            self.mean.copy_(m)
            self.var.copy_(v)
        else:
            a = self.momentum
            self.mean.mul_(a).add_(m, alpha=1 - a)
            self.var.mul_(a).add_(v, alpha=1 - a)
        self.n += 1

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        shape = (1, -1, 1)
        return (x - self.mean.view(shape)) / (self.var.view(shape) + self.eps).sqrt()

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        shape = (1, -1, 1)
        return x * (self.var.view(shape) + self.eps).sqrt() + self.mean.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)
