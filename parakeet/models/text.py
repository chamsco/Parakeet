"""Text encoders.

Parakeet uses **character-level** text by default, following SupertonicTTS: no G2P module and
no external aligner, with alignment learned implicitly through cross-attention.  A phoneme
mode is also supported because the Tiny model follows Paradee/Kokoro and can afford a
deterministic G2P (espeak-ng) for a single voice.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import TextConfig
from .blocks import ConvNeXtBlock, SelfAttentionBlock


class TextEncoder(nn.Module):
    def __init__(self, cfg: TextConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=0)
        self.pos = (
            nn.Embedding(cfg.max_len, cfg.dim) if cfg.use_pos_emb else None
        )
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(cfg.dim, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )
        # A little local convolution helps character-level models capture grapheme clusters.
        self.conv = ConvNeXtBlock(cfg.dim, kernel_size=5, layer_scale_init=1e-6)
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``ids``: ``(B, T)`` long -> memory ``(B, T, dim)``."""
        x = self.embed(ids)
        if self.pos is not None:
            pos = torch.arange(ids.shape[1], device=ids.device)
            x = x + self.pos(pos)[None]
        pad_mask = None if mask is None else ~mask.to(torch.bool)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x

    def forward_with_lengths(self, ids: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        from .blocks import sequence_mask

        mask = sequence_mask(lengths, ids.shape[1])
        return self.forward(ids, mask), mask
