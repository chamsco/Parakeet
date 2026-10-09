"""Top-level Parakeet models.

Two variants share every building block, differing only in *how the acoustic latent is
produced* -- which is exactly the split Paradee exploits (train the two halves separately,
then connect them):

``ParakeetTiny``  (~8-12M)  text -> (durations, F0, energy, per-token latent feature)
                           -> frame-rate latent -> causal decoder -> waveform.
                           Single/few voices, no flow sampler, CPU realtime.  Trained by
                           regression onto *cached teacher signals*: no alignment learning,
                           no joint training required.

``ParakeetFlow``  (~44-60M) text + speaker/style -> flow-matching vector field
                           -> compressed latent -> causal decoder -> waveform.
                           Zero-shot cloning, GPU/CPU realtime with 1-4 NFE after
                           sampler distillation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ParakeetConfig
from .autoencoder import LatentNormalizer, SpeechAutoencoder
from .blocks import sequence_mask
from .duration import (
    DurationPredictor,
    UtteranceLengthPredictor,
    align_tokens_to_frames,
    normalized_to_durations,
)
from .flow import (
    ConvNeXtVFEstimator,
    build_memory,
    consistency_sample,
    fold_time,
    iter_blockwise_sample,
    make_xt,
    sample_timesteps,
    unfold_time,
)
from .speaker import SpeakerConditioner
from .text import TextEncoder


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)
    )


def parameter_report(module: nn.Module, prefix: str = "") -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, child in module.named_children():
        n = count_parameters(child)
        if n:
            out[f"{prefix}{name}"] = n
    out[f"{prefix}TOTAL"] = count_parameters(module)
    return out


# --------------------------------------------------------------------------------------
# Tiny: Paradee-style two-half distillation
# --------------------------------------------------------------------------------------
class ParakeetTiny(nn.Module):
    variant = "tiny"

    def __init__(self, cfg: ParakeetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.autoencoder = SpeechAutoencoder(cfg.audio, cfg.autoencoder)
        self.latent_norm = LatentNormalizer(cfg.autoencoder.latent_dim)
        self.text = TextEncoder(cfg.text)
        self.duration = DurationPredictor(cfg.duration, cfg.text.dim)
        #: per-token acoustic state, predicted by the small text side (Paradee's "phoneme
        #: features"): a vector in the autoencoder's latent space.
        self.latent_head = nn.Sequential(
            nn.Linear(cfg.text.dim, cfg.text.dim), nn.GELU(), nn.Linear(cfg.text.dim, cfg.autoencoder.latent_dim)
        )
        self.f0_head = nn.Linear(cfg.text.dim, 1)
        self.energy_head = nn.Linear(cfg.text.dim, 1)
        #: frame-level refinement: F0/energy contour modulates the repeated token latents
        self.prosody_proj = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(), nn.Linear(64, cfg.autoencoder.latent_dim)
        )
        #: voice: conditions the *whole* text side, not just the latent feature.  A voice differs in
        #: pitch range, energy, timing and timbre, so adding it only to the latent (as the first
        #: version did) left the F0/energy/duration heads voice-blind -- which made multi-voice
        #: training unable to separate voices at all.  "Replaces the style input with a learned
        #: constant" for a single voice (Paradee).
        self.voice_embed = nn.Embedding(max(1, cfg.n_voices), cfg.text.dim)
        self.register_buffer("f0_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("f0_std", torch.tensor(1.0), persistent=False)

    # ------------------------------------------------------------------ text side
    def text_side(
        self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None, voice: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        h = self.text(ids, mask)
        if voice is not None:
            h = h + self.voice_embed(voice)[:, None, :]
        else:
            h = h + self.voice_embed.weight[0][None, None, :]
        log_dur = self.duration(h, mask)
        latent_tok = self.latent_head(h)
        f0 = self.f0_head(h).squeeze(-1)
        energy = self.energy_head(h).squeeze(-1)
        return {
            "text_memory": h,
            "log_duration": log_dur,
            "latent_token": latent_tok,
            "f0": f0,
            "energy": energy,
        }

    def latent_from_tokens(
        self,
        latent_tok: torch.Tensor,
        durations: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        max_frames: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the frame-rate latent the decoder consumes."""
        frames, frame_mask = align_tokens_to_frames(latent_tok, durations)
        if max_frames is not None:
            frames = frames[:, :max_frames]
            frame_mask = frame_mask[:, :max_frames]
        if f0 is not None and energy is not None:
            f0f, _ = align_tokens_to_frames(f0.unsqueeze(-1), durations)
            ef, _ = align_tokens_to_frames(energy.unsqueeze(-1), durations)
            f0f = f0f[:, : frames.shape[1]]
            ef = ef[:, : frames.shape[1]]
            frames = frames + self.prosody_proj(torch.cat([f0f, ef], dim=-1))
        latent = frames.transpose(1, 2)
        return latent, frame_mask

    def decoder_latent_from_tokens(
        self,
        latent_tok: torch.Tensor,
        durations: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        max_frames: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the **decoder-input** latent from (predicted or teacher) token signals.

        The text side predicts *normalised* per-token latents (that is what ``latent_norm`` is
        for), while the autoencoder decoder was trained on raw encoder output -- so this is the
        one place where the de-normalisation happens.  Both inference and the
        ``distill-decoder`` stage go through here, which guarantees the decoder is trained on
        exactly the distribution it will see at synthesis time.
        """
        latent, frame_mask = self.latent_from_tokens(latent_tok, durations, f0, energy, max_frames)
        return self.latent_norm.denormalize(latent), frame_mask

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def synthesize(
        self,
        ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        voice: Optional[torch.Tensor] = None,
        duration_scale: float = 1.0,
        max_frames: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        side = self.text_side(ids, mask, voice)
        durations = normalized_to_durations(side["log_duration"], duration_scale)
        latent, _ = self.decoder_latent_from_tokens(
            side["latent_token"], durations, side["f0"], side["energy"], max_frames
        )
        return self.autoencoder.decode(latent)

    def forward(self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None, **kw) -> Dict[str, torch.Tensor]:
        return self.text_side(ids, mask, kw.get("voice"))


# --------------------------------------------------------------------------------------
# Small: SupertonicTTS-style flow matching with PilotTTS conditioning
# --------------------------------------------------------------------------------------
class ParakeetFlow(nn.Module):
    variant = "small"

    def __init__(self, cfg: ParakeetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.autoencoder = SpeechAutoencoder(cfg.audio, cfg.autoencoder)
        self.latent_norm = LatentNormalizer(cfg.autoencoder.latent_dim)
        self.text = TextEncoder(cfg.text)
        self.speaker = SpeakerConditioner(cfg.speaker)
        self.cond_proj = nn.Linear(cfg.speaker.style_dim, cfg.flow.cond_dim)
        self.vf = ConvNeXtVFEstimator(cfg.flow)
        self.length_predictor = UtteranceLengthPredictor(cfg.duration, cfg.text.dim, cfg.flow.cond_dim)
        if cfg.voice_mode == "constant":
            self.voice_embed = nn.Embedding(max(1, cfg.n_voices), cfg.speaker.style_dim)
        else:
            self.voice_embed = None

    # ------------------------------------------------------------------ conditioning
    def conditions(
        self,
        ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ref_mel: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        voice: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Returns ``(memory, memory_mask, cond_tokens)``."""
        text_mem = self.text(ids, mask)
        b = ids.shape[0]
        style = None
        if self.voice_embed is not None:
            v = voice if voice is not None else torch.zeros(b, dtype=torch.long, device=ids.device)
            style = self.voice_embed(v)[:, None, :].expand(-1, self.cfg.speaker.n_query, -1)
        if ref_mel is None and speaker_emb is None and style is None:
            # no reference at all: fall back to a zero identity embedding so that pre-training on
            # a fixed voice (or an unconditional pass) still works and stays differentiable.
            speaker_emb = torch.zeros(b, self.cfg.speaker.emb_dim, device=ids.device)
        cond = self.speaker(
            mel=ref_mel if ref_mel is not None else None,
            mel_mask=ref_mask,
            speaker_emb=speaker_emb,
            style=style,
        )
        cond = self.cond_proj(cond)
        memory, memory_mask = build_memory(text_mem, mask, cond)
        return memory, memory_mask, cond

    # ------------------------------------------------------------------ shapes
    def latent_frames(self, n_samples: int) -> int:
        return n_samples // self.cfg.audio.hop_length + 1

    def compressed_frames(self, n_latent_frames: int) -> int:
        return max(1, n_latent_frames // self.cfg.flow.compress)

    def predict_latent_frames(
        self,
        text_mem: torch.Tensor,
        mask: Optional[torch.Tensor],
        cond: torch.Tensor,
        duration_scale: float = 1.0,
    ) -> torch.Tensor:
        """Predicted total latent frame count per item (SupertonicTTS utterance duration)."""
        log_len = self.length_predictor(text_mem, cond, mask)
        return (log_len.exp() * duration_scale).round().clamp_min(1).long()

    # ------------------------------------------------------------------ losses
    def flow_loss(
        self,
        ids: torch.Tensor,
        mask: Optional[torch.Tensor],
        x1: torch.Tensor,
        ref_mel: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        voice: Optional[torch.Tensor] = None,
        context_expansion: Optional[int] = None,
        reflow: bool = False,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Conditional flow-matching MSE.  ``x1`` is the (normalised) latent ``(B, C, T)``.

        ``sample_weight`` carries the per-sample teacher-mixture weight (see
        :class:`~parakeet.train.losses.MultiTeacherMixer`); it is repeated along with the batch when
        context-sharing expansion is active, so the mixture re-weights the gradient exactly.
        """
        ke = context_expansion or self.cfg.flow.context_expansion
        t_latent = x1.shape[-1]
        x1c = fold_time(x1, self.cfg.flow.compress)
        b, c, tc = x1c.shape
        if ke > 1:
            ids = ids.repeat_interleave(ke, dim=0)
            mask = None if mask is None else mask.repeat_interleave(ke, dim=0)
            ref_mel = None if ref_mel is None else ref_mel.repeat_interleave(ke, dim=0)
            ref_mask = None if ref_mask is None else ref_mask.repeat_interleave(ke, dim=0)
            speaker_emb = None if speaker_emb is None else speaker_emb.repeat_interleave(ke, dim=0)
            voice = None if voice is None else voice.repeat_interleave(ke, dim=0)
            x1c = x1c.repeat_interleave(ke, dim=0)
            sample_weight = (
                None if sample_weight is None else sample_weight.repeat_interleave(ke, dim=0)
            )
        memory, memory_mask, _ = self.conditions(
            ids, mask, ref_mel, ref_mask, speaker_emb, voice
        )
        x0 = torch.randn_like(x1c)
        t = sample_timesteps(x1c.shape[0], x1c.device, self.cfg.flow.sigma_min)
        x_t = make_xt(x1c, x0, t)
        v_target = x1c - x0
        drop = (torch.rand(x1c.shape[0], device=x1c.device) < 0.1) if self.training else None
        v_pred = self.vf(x_t, t, memory, memory_mask, drop_cond=drop)
        from ..train.losses import weighted_mean

        per_sample = (v_pred - v_target).pow(2).mean(dim=tuple(range(1, v_pred.dim())))
        loss = weighted_mean(per_sample, sample_weight)
        aux = {"t_latent": torch.tensor(float(t_latent)), "tc": torch.tensor(float(tc))}
        if reflow:
            loss = loss * 1.0  # reflow pairs are supplied by the caller as (x0, x1)
        return loss, aux

    @torch.no_grad()
    def teacher_endpoint(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        shape: Tuple[int, int, int],
        steps: Optional[int] = None,
        x0: Optional[torch.Tensor] = None,
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Run the *high-NFE* sampler to produce targets for Reflow / consistency distillation."""
        steps = steps or self.cfg.flow.nfe
        cfg_scale = 1.0 if cfg_scale is None else cfg_scale
        return consistency_sample(
            self.vf, memory, memory_mask, shape, steps=steps, device=memory.device, cfg_scale=cfg_scale
        )

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def synthesize(
        self,
        ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ref_mel: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        voice: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        duration_scale: float = 1.0,
        n_latent_frames: Optional[int] = None,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        memory, memory_mask, cond = self.conditions(
            ids, mask, ref_mel, ref_mask, speaker_emb, voice
        )
        text_mem = self.text(ids, mask)
        if n_latent_frames is None:
            n_latent_frames = int(
                self.predict_latent_frames(text_mem, mask, cond, duration_scale).max().item()
            )
        tc = self.compressed_frames(n_latent_frames)
        b = ids.shape[0]
        shape = (b, self.cfg.flow.latent_dim * self.cfg.flow.compress, tc)
        steps = steps or self.cfg.flow.distilled_nfe
        cfg_scale = self.cfg.flow.cfg_scale if cfg_scale is None else cfg_scale
        x1c = consistency_sample(
            self.vf, memory, memory_mask, shape, steps=steps, device=ids.device, cfg_scale=cfg_scale
        )
        latent = unfold_time(x1c, self.cfg.flow.compress, t_out=n_latent_frames)
        latent = self.latent_norm.denormalize(latent)
        return self.autoencoder.decode(latent)

    @torch.no_grad()
    def synthesize_stream(
        self,
        ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ref_mel: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        voice: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        duration_scale: float = 1.0,
        n_latent_frames: Optional[int] = None,
        block_frames: int = 16,
        context: Optional[int] = None,
        lookahead: Optional[int] = None,
        context_mode: str = "interpolated",
        voice_stream_chunk: Optional[int] = None,
    ):
        """Yield waveform chunks as the latent is being sampled.

        Time-to-first-audio becomes "one block of sampling + one block of decoding" instead of
        "the whole latent", which matters most for long utterances: the sampling window is
        ``context + block + lookahead`` frames regardless of utterance length, so TTFA stays
        roughly constant while total time grows with the text.

        The blockwise sampler is an approximation of the full-sequence ODE (see
        :func:`parakeet.models.flow.blockwise_sample`); ``scripts/streaming_demo.py`` measures the
        endpoint error against the one-shot sampler.
        """
        from ..inference.synthesize import StreamingVocoder

        self.eval()
        memory, memory_mask, cond = self.conditions(
            ids, mask, ref_mel, ref_mask, speaker_emb, voice
        )
        text_mem = self.text(ids, mask)
        if n_latent_frames is None:
            n_latent_frames = int(
                self.predict_latent_frames(text_mem, mask, cond, duration_scale).max().item()
            )
        tc = self.compressed_frames(n_latent_frames)
        b = ids.shape[0]
        shape = (b, self.cfg.flow.latent_dim * self.cfg.flow.compress, tc)
        steps = steps or self.cfg.flow.distilled_nfe
        cfg_scale = self.cfg.flow.cfg_scale if cfg_scale is None else cfg_scale

        vocoder = StreamingVocoder(
            self.autoencoder, chunk_frames=voice_stream_chunk or block_frames
        )
        remaining = int(n_latent_frames)
        for _start, _end, block in iter_blockwise_sample(
            self.vf,
            memory,
            memory_mask,
            shape,
            steps=steps,
            block_frames=block_frames,
            context=context,
            lookahead=lookahead,
            context_mode=context_mode,
            cfg_scale=cfg_scale,
        ):
            latent = unfold_time(block, self.cfg.flow.compress)
            if latent.shape[-1] > remaining:
                latent = latent[..., :remaining]
            remaining -= latent.shape[-1]
            latent = self.latent_norm.denormalize(latent)
            wav = vocoder.push(latent)
            if wav.shape[-1]:
                yield wav
            if remaining <= 0:
                break
        tail = vocoder.flush()
        if tail.shape[-1]:
            yield tail

    def forward(self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None, **kw):
        return self.conditions(ids, mask, **kw)


# --------------------------------------------------------------------------------------
def build_model(cfg: ParakeetConfig) -> nn.Module:
    cfg.validate()
    if cfg.variant == "tiny":
        return ParakeetTiny(cfg)
    if cfg.variant == "small":
        return ParakeetFlow(cfg)
    raise ValueError(f"unknown variant {cfg.variant!r}")


