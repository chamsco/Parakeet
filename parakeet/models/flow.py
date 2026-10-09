"""Text-to-latent generation with rectified flow (conditional flow matching).

Follows SupertonicTTS §3.2 with three deliberate additions:

1. **Temporal compression** (``Kc``).  The latent ``(C, T)`` is folded to ``(C*Kc, T/Kc)`` so
   the vector-field estimator runs at ~1/Kc the frame rate: 6x fewer tokens for the same
   information, and the shortened sequence is what keeps a small ConvNeXt competitive with a
   transformer.
2. **Context-sharing batch expansion** (``Ke``): each text/reference pair is expanded into
   ``Ke`` independent noise/timestep draws, so the alignment learning signal per unit of
   memory/IO grows by ``Ke`` (SupertonicTTS reports faster convergence and fewer
   word-skip/repeat errors; we use ``Ke=4``).
3. **Few-step distillation.**  A 32-step flow ODE is unusable for "lightning fast" CPU
   synthesis, so we distill the sampler itself: 2-rectified-flow / Reflow training
   (:func:`reflow_pair`) plus optional consistency distillation, targeting NFE 1-4.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import FlowConfig
from .blocks import ChannelLayerNorm, ConvNeXtBlock, CrossAttention, TimeEmbedding, sequence_mask


# --------------------------------------------------------------------------------------
# temporal (de)compression
# --------------------------------------------------------------------------------------
def fold_time(x: torch.Tensor, k: int) -> torch.Tensor:
    """``(B, C, T) -> (B, C*k, T//k)`` (drop the remainder)."""
    if k == 1:
        return x
    b, c, t = x.shape
    t_keep = (t // k) * k
    x = x[..., :t_keep]
    return x.reshape(b, c, t_keep // k, k).permute(0, 1, 3, 2).reshape(b, c * k, t_keep // k)


def unfold_time(x: torch.Tensor, k: int, t_out: Optional[int] = None) -> torch.Tensor:
    """Inverse of :func:`fold_time`; ``t_out`` right-pads to the original frame count."""
    if k == 1:
        return x
    b, ck, tk = x.shape
    c = ck // k
    x = x.reshape(b, c, k, tk).permute(0, 1, 3, 2).reshape(b, c, tk * k)
    if t_out is not None and t_out > x.shape[-1]:
        x = F.pad(x, (0, t_out - x.shape[-1]))
    return x


# --------------------------------------------------------------------------------------
# vector field estimator
# --------------------------------------------------------------------------------------
class FlowBlock(nn.Module):
    """ConvNeXt mixing + cross-attention into (text || speaker/style) memory."""

    def __init__(self, cfg: FlowConfig) -> None:
        super().__init__()
        self.conv = ConvNeXtBlock(cfg.dim, cfg.kernel_size, cfg.ffn_mult, layer_scale_init=1e-6)
        self.norm = nn.LayerNorm(cfg.dim)
        self.cross = CrossAttention(cfg.dim, cfg.dim, cfg.n_heads, cfg.dropout)
        self.ffn_norm = nn.LayerNorm(cfg.dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.dim, cfg.ffn_mult * cfg.dim), nn.GELU(), nn.Linear(cfg.ffn_mult * cfg.dim, cfg.dim)
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor, memory_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """``x``: channels-first ``(B, dim, Tc)``; ``memory``: ``(B, S, dim)``."""
        x = self.conv(x)
        x = x + self.cross(self.norm(x.transpose(1, 2)), memory, memory_mask).transpose(1, 2)
        return x + self.ffn(self.ffn_norm(x.transpose(1, 2))).transpose(1, 2)


class ConvNeXtVFEstimator(nn.Module):
    """Predicts the flow velocity ``v = x1 - x0`` at (compressed) latent time."""

    def __init__(self, cfg: FlowConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.latent_dim * cfg.compress
        self.in_proj = nn.Conv1d(in_dim, cfg.dim, 1)
        self.time_mlp = TimeEmbedding(cfg.dim)
        self.mem_proj = nn.Linear(cfg.cond_dim, cfg.dim)
        self.blocks = nn.ModuleList([FlowBlock(cfg) for _ in range(cfg.depth)])
        self.out_norm = ChannelLayerNorm(cfg.dim)
        self.out_proj = nn.Conv1d(cfg.dim, in_dim, 1)
        #: learned unconditional (null) conditioning for classifier-free guidance
        self.null_memory = nn.Parameter(torch.randn(1, 1, cfg.cond_dim) * 0.02)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        memory: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor] = None,
        drop_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``x_t``: ``(B, C*Kc, Tc)``; ``t``: ``(B,)`` in ``[0,1]``; ``memory``: ``(B, S, cond_dim)``.

        ``memory`` is projected to the model dimension once and then used as the cross-attention
        context by every block.  ``drop_cond`` implements classifier-free guidance by replacing
        selected items' memory with a learned null token (also used for the unconditional pass).
        """
        if memory is None:
            b = x_t.shape[0]
            memory = self.mem_proj(self.null_memory.expand(b, -1, -1))
            memory_mask = None
        else:
            if drop_cond is not None and bool(drop_cond.any()):
                null = self.null_memory.expand(memory.shape[0], memory.shape[1], -1)
                sel = drop_cond.view(-1, 1, 1).to(memory.dtype)
                memory = memory * (1 - sel) + null * sel
            memory = self.mem_proj(memory)

        h = self.in_proj(x_t)
        h = h + self.time_mlp(t)[:, :, None]
        for blk in self.blocks:
            h = blk(h, memory, memory_mask)
        return self.out_proj(self.out_norm(h))


# --------------------------------------------------------------------------------------
# flow matching utilities
# --------------------------------------------------------------------------------------
def make_xt(x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Rectified-flow interpolation: ``x_t = (1-t) x0 + t x1`` (t=0 noise, t=1 data)."""
    while t.dim() < x1.dim():
        t = t.unsqueeze(-1)
    return (1.0 - t) * x0 + t * x1


def flow_velocity(x1: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    return x1 - x0


def sample_timesteps(
    batch: int, device, sigma_min: float = 1e-4, mode: str = "uniform"
) -> torch.Tensor:
    if mode == "logit_normal":
        u = torch.randn(batch, device=device)
        t = torch.sigmoid(u)  # concentrated around 0.5
    else:
        t = torch.rand(batch, device=device)
    return t.clamp(sigma_min, 1.0 - sigma_min)


def reflow_pair(x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a straight-line (2-rectified flow) training pair from a teacher ODE endpoint.

    Given the noise ``x0`` and the *teacher sampler's* output ``x1`` for that noise, the
    student is trained on the chord ``(1-t) x0 + t x1`` with target ``x1 - x0``.  Iterating
    this "reflow" step straightens the trajectories so that 1-4 Euler steps suffice.
    """
    b = x0.shape[0]
    t = sample_timesteps(b, x0.device)
    x_t = make_xt(x1, x0, t)
    v = flow_velocity(x1, x0)
    return x_t, t, v


@torch.no_grad()
def euler_sample(
    model: ConvNeXtVFEstimator,
    memory: Optional[torch.Tensor],
    memory_mask: Optional[torch.Tensor],
    shape: Tuple[int, int, int],
    steps: int,
    device,
    cfg_scale: float = 1.0,
    t_start: float = 0.0,
    x0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Integrate the flow ODE from ``t_start`` to 1 with ``steps`` Euler updates."""
    was_training = model.training
    model.eval()
    try:
        b, c, tc = shape
        if x0 is None:
            x0 = torch.randn(shape, device=device)
        x = x0
        ts = torch.linspace(t_start, 1.0, steps + 1, device=device)
        for i in range(steps):
            t = ts[i].expand(b)
            v = model(x, t, memory, memory_mask)
            if cfg_scale != 1.0:
                v_uncond = model(x, t, None, None)
                v = v_uncond + cfg_scale * (v - v_uncond)
            dt = ts[i + 1] - ts[i]
            x = x + dt * v
        return x
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def consistency_sample(
    model: ConvNeXtVFEstimator,
    memory: Optional[torch.Tensor],
    memory_mask: Optional[torch.Tensor],
    shape: Tuple[int, int, int],
    steps: int = 2,
    device=None,
    cfg_scale: float = 1.0,
    x0: Optional[torch.Tensor] = None,
    t_start: float = 0.0,
) -> torch.Tensor:
    """Few-step sampler used after Reflow/consistency distillation (NFE == ``steps``).

    ``x0`` fixes the initial noise, which matters for A/B comparisons (blockwise vs one-shot must
    start from the same draw).
    """
    return euler_sample(
        model,
        memory,
        memory_mask,
        shape,
        steps=steps,
        device=device,
        cfg_scale=cfg_scale,
        x0=x0,
        t_start=t_start,
    )


# --------------------------------------------------------------------------------------
# streaming sampling
# --------------------------------------------------------------------------------------
def vf_context_frames(model: ConvNeXtVFEstimator) -> int:
    """Frames of temporal context the vector field reads on each side.

    The estimator mixes time with *symmetric* (non-causal) depthwise convolutions, so every block
    adds ``(k-1)//2`` frames on each side.  Because the mixing is convolutional rather than a full
    self-attention, the needed context is finite -- which is what makes blockwise streaming
    sampling possible at all: a transformer-based vector field would require the whole sequence.
    """
    total = 0
    for blk in model.blocks:
        conv = getattr(blk, "conv", None)
        dw = getattr(conv, "dwconv", None)
        inner = getattr(dw, "conv", dw)
        if inner is None:
            continue
        kernel = int(inner.kernel_size[0])
        dilation = int(inner.dilation[0])
        total += (kernel - 1) * dilation // 2
    return total


@torch.no_grad()
def iter_blockwise_sample(
    model: ConvNeXtVFEstimator,
    memory: Optional[torch.Tensor],
    memory_mask: Optional[torch.Tensor],
    shape: Tuple[int, int, int],
    steps: int = 2,
    block_frames: int = 16,
    context: Optional[int] = None,
    lookahead: Optional[int] = None,
    context_mode: str = "interpolated",
    cfg_scale: float = 1.0,
    x0: Optional[torch.Tensor] = None,
    t_start: float = 0.0,
):
    """Yield ``(start, end, x1_block)`` for each compressed-frame block, in order.

    Streaming consumers (see :meth:`ParakeetFlow.synthesize_stream`) decode each block as it
    arrives instead of waiting for the whole latent, which is what turns a one-shot sampler into a
    low-time-to-first-audio one.  :func:`blockwise_sample` is a thin consumer of this generator, so
    the measured endpoint error applies to the streaming path exactly.
    """
    if context_mode not in {"interpolated", "final"}:
        raise ValueError(f"unknown context_mode {context_mode!r}")
    b, c, tc = shape
    if x0 is None:
        x0 = torch.randn(shape, device=memory.device if memory is not None else None)
    rf = vf_context_frames(model) if context is None else int(context)
    ahead = rf if lookahead is None else int(lookahead)
    x1 = torch.zeros_like(x0)
    ts = torch.linspace(t_start, 1.0, steps + 1, device=x0.device)
    was_training = model.training
    model.eval()
    try:
        for start in range(0, tc, block_frames):
            end = min(tc, start + block_frames)
            lo = max(0, start - rf)
            hi = min(tc, end + ahead)
            new = end - start
            x_ctx = x1[:, :, lo:start]
            x_ctx0 = x0[:, :, lo:start]
            x_new = x0[:, :, start:hi].clone()
            for i in range(steps):
                t = ts[i].expand(b)
                x_in = torch.cat([x_ctx, x_new], dim=-1) if x_ctx.shape[-1] else x_new
                v = model(x_in, t, memory, memory_mask)
                if cfg_scale != 1.0:
                    v_uncond = model(x_in, t, None, None)
                    v = v_uncond + cfg_scale * (v - v_uncond)
                v_new = v[:, :, -x_new.shape[-1] :]
                x_new = x_new + (ts[i + 1] - ts[i]) * v_new
                if context_mode == "interpolated" and x_ctx.shape[-1]:
                    x_ctx = (1.0 - ts[i + 1]) * x_ctx0 + ts[i + 1] * x1[:, :, lo:start]
            x1[:, :, start:end] = x_new[:, :, :new]
            yield start, end, x1[:, :, start:end]
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def blockwise_sample(
    model: ConvNeXtVFEstimator,
    memory: Optional[torch.Tensor],
    memory_mask: Optional[torch.Tensor],
    shape: Tuple[int, int, int],
    steps: int = 2,
    block_frames: int = 16,
    context: Optional[int] = None,
    lookahead: Optional[int] = None,
    context_mode: str = "interpolated",
    cfg_scale: float = 1.0,
    x0: Optional[torch.Tensor] = None,
    t_start: float = 0.0,
) -> torch.Tensor:
    """Sample the latent ODE in blocks so audio can start before the whole utterance exists.

    This is an *approximation* of :func:`euler_sample`, and the approximation is worth stating
    precisely, because it is the whole reason it is cheap:

    * the vector field is convolutional with finite context ``rf``, so a block's outputs depend
      only on ``[block_start - rf, block_end + rf]``.  Text and speaker conditioning are global and
      computed once, so they impose no temporal dependency;
    * frames to the **left** are already finalised, so they are held at their generated values --
      optionally *interpolated back* along the same rectified-flow path they travelled
      (``context_mode="interpolated"``), which is what the full-sequence integration would have
      shown at that timestep, given their own noise draw;
    * frames to the **right** within ``lookahead`` are not finalised, so they are integrated
      alongside the block (co-evolving, exactly as they do in the full run) and then discarded when
      the next block regenerates them with proper left context.

    ``context_mode="final"`` is cheaper (no interpolation bookkeeping) but presents the model with
    out-of-distribution finalised neighbours at every timestep.  ``scripts/streaming_demo.py``
    measures the resulting endpoint error against the full-sequence sampler.
    """
    b, c, tc = shape
    out = torch.empty(
        shape, device=memory.device if memory is not None else (x0.device if x0 is not None else None)
    )
    for start, end, block in iter_blockwise_sample(
        model,
        memory,
        memory_mask,
        shape,
        steps=steps,
        block_frames=block_frames,
        context=context,
        lookahead=lookahead,
        context_mode=context_mode,
        cfg_scale=cfg_scale,
        x0=x0,
        t_start=t_start,
    ):
        out[:, :, start:end] = block
    return out


def build_memory(
    text_mem: torch.Tensor,
    text_mask: Optional[torch.Tensor],
    cond_tokens: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Concatenate text memory and speaker/style tokens into a single attention memory."""
    if cond_tokens is None:
        return text_mem, text_mask
    memory = torch.cat([text_mem, cond_tokens], dim=1)
    if text_mask is None:
        return memory, None
    cond_mask = torch.ones(
        cond_tokens.shape[0], cond_tokens.shape[1], dtype=text_mask.dtype, device=text_mask.device
    )
    return memory, torch.cat([text_mask, cond_mask], dim=1)


