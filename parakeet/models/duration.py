"""Duration modelling.

Two heads, both used by Parakeet:

* :class:`DurationPredictor` -- per-token durations.  The Tiny model needs these to place
  phonemes on the time axis (Paradee regresses the teacher's own durations, so no alignment
  learning and no external aligner are required).
* :class:`UtteranceLengthPredictor` -- a single scalar total length, as in SupertonicTTS,
  which keeps duration prediction decoupled from content generation for the flow model.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DurationConfig
from .blocks import ChannelLayerNorm, CausalConv1d, masked_mean


class DurationPredictor(nn.Module):
    """Predicts log-duration per text token."""

    def __init__(self, cfg: DurationConfig, input_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(max(1, cfg.n_layers)):
            layers.append(nn.Conv1d(d, cfg.hidden, cfg.kernel_size, padding=cfg.kernel_size // 2))
            layers.append(ChannelLayerNorm(cfg.hidden))
            layers.append(nn.GELU())
            d = cfg.hidden
        self.net = nn.Sequential(*layers)
        self.proj = nn.Conv1d(cfg.hidden, 1, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x``: ``(B, T, C)`` -> log durations ``(B, T)``."""
        h = self.net(x.transpose(1, 2))
        out = self.proj(h).squeeze(1)
        if mask is not None:
            out = out.masked_fill(~mask.to(torch.bool), 0.0)
        return out

    def durations(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, min_frames: int = 1) -> torch.Tensor:
        """Positive integer frame counts (rounded, clamped)."""
        return self.forward(x, mask).exp().round().clamp_min(min_frames).long()


class UtteranceLengthPredictor(nn.Module):
    """Predicts total latent length (in compressed frames) from pooled text+speaker."""

    def __init__(self, cfg: DurationConfig, text_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        d = text_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(d, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(
        self,
        text_mem: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = masked_mean(text_mem, text_mask, dim=1)
        if cond is not None:
            pooled = torch.cat([pooled, cond.mean(dim=1) if cond.dim() == 3 else cond], dim=-1)
        return self.net(pooled).squeeze(-1)  # log-length


def align_tokens_to_frames(
    token_features: torch.Tensor,
    durations: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repeat token features across frames; returns ``(frames, frame_mask)``."""
    b, t, c = token_features.shape
    repeats = durations.clamp_min(0).long()
    lengths = repeats.sum(dim=-1)
    t_max = int(lengths.max().item()) if lengths.numel() else 0
    out = torch.zeros(b, t_max, c, dtype=token_features.dtype, device=token_features.device)
    mask = torch.zeros(b, t_max, dtype=torch.bool, device=token_features.device)
    for i in range(b):
        idx = torch.repeat_interleave(torch.arange(t, device=token_features.device), repeats[i])
        n = min(idx.numel(), t_max)
        out[i, :n] = token_features[i, idx[:n]]
        mask[i, :n] = True
    return out, mask


class CausalDurationUpsampler(nn.Module):
    """Streaming-safe upsampling from token rate to frame rate."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.smooth = CausalConv1d(dim, dim, 5)

    def forward(self, tokens: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        x, _ = align_tokens_to_frames(tokens, durations)
        return self.smooth(x.transpose(1, 2)).transpose(1, 2)
