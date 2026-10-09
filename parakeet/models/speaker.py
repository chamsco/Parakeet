"""Speaker conditioning.

Parakeet follows PilotTTS (arXiv 2605.27258) in *factorising* voice conditioning into two
pathways:

* a **global identity embedding** from a frozen speaker-verification network (CAM++ /
  CAMPPlus).  Identity is a slowly-varying, clip-level attribute.
* a set of **Q-Former style tokens** attending into the reference mel, which carry the
  dynamic, utterance-level attributes (prosody, emotion, speaking rate, recording colour).

Keeping the two pathways separate is what makes cross-sample paired training possible: you
can swap style tokens between two clips of the *same* speaker (or keep identity and change
style) and add a consistency loss, which is how PilotTTS decouples identity from style.

The real pipeline loads pretrained CAM++ (modelscope: ``iic/speech_campplus_sv_zh-cn_16k-common``).
:class:`EcapaTdnnLite` is a small randomly-initialised stand-in with the same interface so
that the package is runnable and unit-testable without network access.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import SpeakerConfig
from .blocks import ChannelLayerNorm, CrossAttention, masked_mean


class EcapaTdnnLite(nn.Module):
    """Compact ECAPA-TDNN-style speaker encoder.  ``(B, n_mels, T) -> (B, emb_dim)``."""

    def __init__(self, n_mels: int = 80, channels: tuple[int, ...] = (256, 384, 384, 384), emb_dim: int = 192) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.stem = nn.Conv1d(n_mels, channels[0], 5, padding=2)
        blocks: list[nn.Module] = []
        for i, ch in enumerate(channels):
            in_ch = channels[i - 1] if i else channels[0]
            dilation = 2 if i % 2 else 1
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, ch, 3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.merge = nn.Conv1d(sum(channels), 512, 1)
        self.pool_attn = nn.Sequential(nn.Conv1d(512, 128, 1), nn.Tanh(), nn.Conv1d(128, 1, 1))
        self.fc = nn.Linear(1024, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)

    def forward(self, mel: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.stem(mel)
        feats = []
        for blk in self.blocks:
            h = blk(h)
            feats.append(h)
        h = self.merge(torch.cat(feats, dim=1))
        attn = self.pool_attn(h)
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, :].to(torch.bool), float("-inf"))
        w = torch.softmax(attn, dim=-1)
        mean = (h * w).sum(dim=-1)
        std = (h.pow(2) * w).sum(dim=-1).clamp_min(1e-8).sqrt()
        emb = self.fc(torch.cat([mean, std], dim=-1))
        if emb.shape[0] > 1:
            emb = self.bn(emb)
        return F.normalize(emb, dim=-1)


class MelMemoryEncoder(nn.Module):
    """Reference-mel -> Q-Former memory ``(B, T, mem_dim)``.

    In production this is a frozen speech-representation model (PilotTTS uses features from a
    speech encoder); the conv stack here is the dependency-free equivalent and is trained
    jointly with the student.
    """

    def __init__(self, n_mels: int = 80, mem_dim: int = 512, channels: tuple[int, ...] = (256, 384, 512)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_mels
        for ch in channels:
            layers += [
                nn.Conv1d(in_ch, ch, 5, padding=2),
                ChannelLayerNorm(ch),
                nn.GELU(),
            ]
            in_ch = ch
        layers.append(nn.Conv1d(in_ch, mem_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.net(mel).transpose(1, 2)


class QFormerStyleEncoder(nn.Module):
    """Learnable query tokens that attend into a frozen reference-mel memory (BLIP-2 Q-Former)."""

    def __init__(self, cfg: SpeakerConfig, mem_dim: int = 512) -> None:
        super().__init__()
        self.cfg = cfg
        self.query = nn.Parameter(torch.randn(1, cfg.n_query, cfg.style_dim) * 0.02)
        self.mem_proj = nn.Linear(mem_dim, cfg.style_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.LayerNorm(cfg.style_dim),
                        CrossAttention(cfg.style_dim, cfg.style_dim, cfg.qformer_heads),
                        nn.LayerNorm(cfg.style_dim),
                        nn.Sequential(
                            nn.Linear(cfg.style_dim, 4 * cfg.style_dim),
                            nn.GELU(),
                            nn.Linear(4 * cfg.style_dim, cfg.style_dim),
                        ),
                    ]
                )
                for _ in range(cfg.qformer_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.style_dim)

    def forward(self, memory: torch.Tensor, memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``memory``: ``(B, S, mem_dim)`` -> style tokens ``(B, n_query, style_dim)``."""
        h = self.mem_proj(memory)
        q = self.query.expand(h.shape[0], -1, -1)
        for ln1, attn, ln2, ffn in self.layers:
            q = q + attn(ln1(q), h, memory_mask)
            q = q + ffn(ln2(q))
        return self.norm(q)


class SpeakerConditioner(nn.Module):
    """Combines the global identity embedding and the Q-Former style tokens into memory tokens."""

    def __init__(self, cfg: SpeakerConfig, mem_dim: int = 512) -> None:
        super().__init__()
        self.cfg = cfg
        self.speaker = EcapaTdnnLite(cfg.n_mels, tuple(cfg.channels), cfg.emb_dim)
        self.mem_encoder = MelMemoryEncoder(cfg.n_mels, mem_dim)
        self.qformer = QFormerStyleEncoder(cfg, mem_dim)
        self.id_proj = nn.Linear(cfg.emb_dim, cfg.style_dim)
        self.type_embed = nn.Parameter(torch.randn(2, cfg.style_dim) * 0.02)
        #: learned constant voice (Tiny / single-voice variant, "replaces the style input
        #: with a learned constant" -- Paradee §2)
        self.constant_style = nn.Parameter(torch.randn(1, cfg.n_query, cfg.style_dim) * 0.02)

    def forward(
        self,
        mel: Optional[torch.Tensor] = None,
        mel_mask: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        style: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns conditioning tokens ``(B, 1 + n_query, style_dim)``.

        ``mel`` is the reference log-mel in ``(B, n_mels, T)`` layout; ``mel_mask`` is
        ``(B, T)`` with True == valid frame.
        """
        if mel is None and speaker_emb is None and style is None:
            raise ValueError(
                "SpeakerConditioner needs at least one of mel / speaker_emb / style; "
                "callers that want an unconditional pass should pass an explicit zero "
                "speaker_emb (see ParakeetFlow.conditions)"
            )
        if mel is not None:
            b = mel.shape[0]
        elif speaker_emb is not None:
            b = speaker_emb.shape[0]
        else:
            b = style.shape[0]
        if speaker_emb is None:
            if mel is None:
                speaker_emb = torch.zeros(b, self.cfg.emb_dim, device=self.id_proj.weight.device)
            else:
                speaker_emb = self.speaker(mel, mel_mask)
        id_token = self.id_proj(speaker_emb)[:, None, :] + self.type_embed[0][None, None, :]
        if style is None:
            if mel is None:
                style = self.constant_style.expand(b, -1, -1)
            else:
                style = self.qformer(self.mem_encoder(mel), mel_mask)
        style = style + self.type_embed[1][None, None, :]
        return torch.cat([id_token, style], dim=1)

    def encode_style(
        self, mel: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Style tokens for a reference mel (Q-Former over the mel memory encoder)."""
        return self.qformer(self.mem_encoder(mel), mask)

    @staticmethod
    def cosine_style_loss(style_a: torch.Tensor, style_b: torch.Tensor) -> torch.Tensor:
        """Same-speaker consistency between two style-token sets (PilotTTS §3.2 pairing).

        Off by default in the recipe: pushing two *same-speaker* style sets together encourages the
        style tokens to encode speaker identity, which is the opposite of the identity/style
        decoupling cross-sample pairing is for.  Kept for callers who want to stabilise style
        tokens early in training; the default term is :meth:`style_separation_loss`.
        """
        a = F.normalize(style_a.reshape(style_a.shape[0], -1), dim=-1)
        b = F.normalize(style_b.reshape(style_b.shape[0], -1), dim=-1)
        return (1.0 - (a * b).sum(dim=-1)).mean()

    @staticmethod
    def style_separation_loss(
        style_same: torch.Tensor, style_other: torch.Tensor, margin: float = 0.0
    ) -> torch.Tensor:
        """Push a speaker's style tokens away from *another speaker's* (identity debiasing).

        With cross-sample pairing the positive reference is a different utterance of the same
        speaker, so any style similarity across *different* speakers is identity leaking into the
        style channel.  A hinge on the cosine similarity removes it without constraining the style
        tokens' own structure.
        """
        a = F.normalize(style_same.reshape(style_same.shape[0], -1), dim=-1)
        b = F.normalize(style_other.reshape(style_other.shape[0], -1), dim=-1)
        return F.relu((a * b).sum(dim=-1) - margin).mean()


def speaker_embedding_from_audio(
    model: SpeakerConditioner, mel: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return model.speaker(mel, mask)
