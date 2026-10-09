"""Neural building blocks shared by all Parakeet modules."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Left-padded 1D convolution (no future leakage), used by the streaming decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, groups=groups, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_pad:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)

    def forward_no_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve without implicit left padding (used by the layer-cached streaming decoder,
        where the caller supplies exact history, including literal zeros at the utterance start)."""
        return self.conv(x)

    @property
    def receptive_field(self) -> int:
        return (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of a ``(B, C, T)`` tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (Liu et al., 2022) in 1D, optionally causal.

    depthwise conv -> channel LayerNorm -> pointwise expand -> GELU -> pointwise project
    -> layer-scale -> residual.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion: int = 4,
        dilation: int = 1,
        causal: bool = False,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        cls = CausalConv1d if causal else nn.Conv1d
        kwargs = {"dilation": dilation} if causal else {"dilation": dilation, "padding": dilation * (kernel_size - 1) // 2}
        self.dwconv = cls(dim, dim, kernel_size, groups=dim, **kwargs)
        self.norm = ChannelLayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.pwconv1(h.transpose(1, 2))
        h = self.act(h)
        h = self.pwconv2(h)
        if self.gamma is not None:
            h = self.gamma * h
        return x + h.transpose(1, 2)

    def forward_with_history(self, x_with_history: torch.Tensor, n_new: int) -> torch.Tensor:
        """Streaming variant: ``x_with_history`` already contains this block's exact input
        history (``left_pad`` frames) followed by ``n_new`` new frames; return the ``n_new``
        new outputs.

        This exists because prefilling a *deeper* layer's input with zeros is not the same as
        the zero padding that layer sees offline: the responses of earlier layers to a zero
        input are themselves non-zero.  Keeping each block's real input history (literal zeros
        only at the true utterance start) makes chunked decoding bit-comparable with offline.
        """
        h = self.dwconv.forward_no_pad(x_with_history)[:, :, -n_new:]
        h = self.norm(h)
        h = self.pwconv1(h.transpose(1, 2))
        h = self.act(h)
        h = self.pwconv2(h)
        if self.gamma is not None:
            h = self.gamma * h
        return x_with_history[:, :, -n_new:] + h.transpose(1, 2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """``t``: ``(B,)`` in ``[0, 1]`` -> ``(B, dim)``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None] * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = hidden or dim * 4
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.mlp[0].in_features))


class CrossAttention(nn.Module):
    """Multi-head cross attention over a memory sequence ``(B, S, C)``."""

    def __init__(self, dim: int, ctx_dim: int, n_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError("dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(ctx_dim, dim)
        self.v = nn.Linear(ctx_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        s = memory.shape[1]
        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(memory).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(memory).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if memory_mask is not None:
            attn_mask = memory_mask[:, None, None, :].to(torch.bool)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o(out)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim), nn.GELU(), nn.Linear(ffn_mult * dim, dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h
        return x + self.ffn(self.norm2(x))


def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """``(B,)`` lengths -> ``(B, T)`` bool mask (True == valid)."""
    max_len = int(max_len or lengths.max().item())
    idx = torch.arange(max_len, device=lengths.device)[None, :]
    return idx < lengths[:, None]


def lengths_to_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.to(torch.bool)


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int = 1) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    m = mask.to(x.dtype)
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    return (x * m).sum(dim=dim) / m.sum(dim=dim).clamp_min(1e-6)


class DurationUpsampler(nn.Module):
    """Repeat each token's feature vector for ``duration`` frames (Tiny path)."""

    @staticmethod
    def forward(x: torch.Tensor, durations: torch.Tensor, max_frames: Optional[int] = None) -> torch.Tensor:
        """``x``: ``(B, T, C)``, ``durations``: ``(B, T)`` ints -> ``(B, T', C)``."""
        outputs = []
        for i in range(x.shape[0]):
            d = durations[i].clamp_min(0).long()
            idx = torch.repeat_interleave(torch.arange(d.numel(), device=x.device), d)
            if max_frames is not None:
                idx = idx[:max_frames]
            outputs.append(x[i, idx] if idx.numel() else x[i, :0])
        t_max = max(o.shape[0] for o in outputs)
        padded = [F.pad(o, (0, 0, 0, t_max - o.shape[0])) for o in outputs]
        return torch.stack(padded, dim=0)
