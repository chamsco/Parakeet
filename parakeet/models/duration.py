"""Duration modelling.

Two heads, both used by Parakeet:

* :class:`DurationPredictor` -- per-token durations.  The Tiny model needs these to place
  phonemes on the time axis (Paradee regresses the teacher's own durations, so no alignment
  learning and no external aligner are required).
* :class:`UtteranceLengthPredictor` -- a single scalar total length, as in SupertonicTTS,
  which keeps duration prediction decoupled from content generation for the flow model.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

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


def subtoken_spans(
    durations: torch.Tensor, rate: int, n_frames: Optional[int] = None
) -> List[List[Tuple[int, int]]]:
    """Split each token's frame span into ``rate`` sub-spans, evenly.

    This exists so the **cache builder and inference cannot disagree** about the geometry.  Round 22
    measured that a token path carrying one latent per token (24 numbers per ~6 frames, 3.9
    dimensions per frame) accounts for most of the token->frame seam, and that carrying 2-3 per token
    recovers it: WER 0.722 at rate 1, 0.204 at rate 2, 0.093 at rate 3, against 0.167 for the frame
    latent itself.  Splitting the span in two places -- once when building targets, once when
    expanding predictions -- is exactly how such a change silently misaligns, so both callers use
    this function.

    A token shorter than ``rate`` frames yields spans clamped to the token; callers must fill any
    missing sub-vector with the token's own mean so the width stays ``rate * dim``.
    """
    spans: List[List[Tuple[int, int]]] = []
    start = 0
    for length in durations.tolist():
        end = start + int(length)
        if n_frames is not None:
            end = min(int(n_frames), end)
        token_spans: List[Tuple[int, int]] = []
        if end > start:
            edges = torch.linspace(start, end, rate + 1).round().long().tolist()
            for k in range(rate):
                a, b = int(edges[k]), int(edges[k + 1])
                b = min(max(b, a + 1), end)
                token_spans.append((a, b) if b > a else (start, end))
        spans.append(token_spans)
        start = end
    return spans


def align_subtokens_to_frames(
    subtokens: torch.Tensor, durations: torch.Tensor, n_frames: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(B, T, rate, C)`` per-token sub-latents -> ``(B, frames, C)`` using the shared geometry.

    The inverse of how :func:`subtoken_spans` defines the spans, so a latent built here matches one
    whose sub-vectors were averaged over those same spans.
    """
    b, t, rate, dim = subtokens.shape
    if durations.dim() == 1:  # a single item
        durations = durations.unsqueeze(0)
    if n_frames is not None:
        total = int(n_frames)
    else:
        total = int(durations.sum(dim=-1).max().item())
    out = torch.zeros(b, total, dim, dtype=subtokens.dtype, device=subtokens.device)
    mask = torch.zeros(b, total, dtype=torch.bool, device=subtokens.device)
    for i in range(b):
        item_total = int(durations[i].sum().item()) if n_frames is None else total
        item_total = min(item_total, total)
        geometry = subtoken_spans(durations[i], rate, item_total)
        for token_index, spans in enumerate(geometry):
            if token_index >= t:
                break
            for k, (a, bb) in enumerate(spans):
                if a >= total:
                    continue
                bb = min(bb, total)
                if bb <= a:
                    continue
                out[i, a:bb] = subtokens[i, token_index, k if k < rate else rate - 1]
                mask[i, a:bb] = True
    return out, mask


class CausalDurationUpsampler(nn.Module):
    """Streaming-safe upsampling from token rate to frame rate."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.smooth = CausalConv1d(dim, dim, 5)

    def forward(self, tokens: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        x, _ = align_tokens_to_frames(tokens, durations)
        return self.smooth(x.transpose(1, 2)).transpose(1, 2)

# --------------------------------------------------------------------------------------
# duration-target normalisation
# --------------------------------------------------------------------------------------
#: log-duration statistics, **measured on the real Kokoro corpus** (round 19):
#: 1141 tokens, mean 1.728, std 0.390 (median 1.609).  Durations were the one prosody target still
#: regressed in raw log space while F0 and energy had been normalised to O(1) since round 2 -- the
#: same fix simply had not been applied to the third signal.
#:
#: The consequence was measurable and severe: with targets around log(6) = 1.79 and a head whose
#: random initialisation outputs near zero, the mean-absolute-error gradient on the head's *weights*
#: is divided by the token count, so after 400 steps the head had learned only a constant ~1.75
#: frames per token -- a 0.29x duration collapse on real speech, and 1.2 of the 2.6 total loss was
#: this single term.  Normalising makes the target zero-mean and unit-variance, so the head starts
#: near the answer and only has to learn the deviation.
LOG_DURATION_MEAN = 1.728
LOG_DURATION_STD = 0.390


def durations_to_normalized(durations: torch.Tensor) -> torch.Tensor:
    """Frame counts -> the normalised target the text side regresses."""
    log_duration = torch.log(durations.clamp_min(1).float())
    return (log_duration - LOG_DURATION_MEAN) / LOG_DURATION_STD


def normalized_to_log_duration(pred: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`durations_to_normalized` in log space."""
    return pred * LOG_DURATION_STD + LOG_DURATION_MEAN


def normalized_to_durations(
    pred: torch.Tensor, duration_scale: float = 1.0, min_frames: int = 1
) -> torch.Tensor:
    """Normalised head output -> integer frame counts.  **Every** inference path must come here.

    The `distill-text` stage regresses normalised log-durations, so any consumer that calls
    ``log_duration.exp()`` directly is off by a factor of ``exp(LOG_DURATION_MEAN) ~ 5.6``.
    """
    frames = torch.exp(normalized_to_log_duration(pred)) * duration_scale
    return frames.round().clamp_min(min_frames).long()
