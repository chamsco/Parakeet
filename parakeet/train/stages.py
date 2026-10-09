"""Training stages.

Parakeet is trained in the order Paradee prescribes -- **separately, then connected**:

======  ==========================  =====================================================
stage   trains                      objective
======  ==========================  =====================================================
``autoencoder``      AE enc+dec    mel + multi-resolution STFT + MPD/MSD adversarial
``distill-decoder``  AE decoder    same, but from *teacher* latents; encoder frozen;
                                   spectral weight annealed 45 -> 10 -> 3 (Paradee §4)
``distill-text``     Tiny text side L1/CE regression on cached teacher signals
                                   (durations, F0, energy, phoneme/latent features)
``flow``             flow VF + len  conditional flow matching, Ke=4 context sharing
``reflow``           flow VF       2-rectified-flow / consistency distillation to NFE 1-4
======  ==========================  =====================================================

Because ``distill-text`` and ``distill-decoder`` consume a *frozen cached* teacher corpus,
no teacher is needed at training time and the two halves never have to be jointly trained --
which is what keeps the Tiny recipe cheap enough to run on rented GPU hours.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio.mel import MelSpectrogram
from ..config import ParakeetConfig
from ..models import build_model
from ..models.flow import consistency_sample, fold_time, make_xt, reflow_pair, sample_timesteps
from .common import (
    EMAModel,
    Meter,
    build_optimizer,
    cosine_warmup_scheduler,
    freeze_,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from .losses import (
    AdversarialVocoderLoss,
    LogMelLoss,
    MultiResolutionSTFTLoss,
    SpectralAnnealer,
    TextSideDistillLoss,
)


# --------------------------------------------------------------------------------------
# individual steps (pure functions so they can be unit-tested without a data pipeline)
# --------------------------------------------------------------------------------------
def autoencoder_step(
    cfg: ParakeetConfig,
    model: nn.Module,
    wav: torch.Tensor,
    losses: Dict[str, nn.Module],
    spectral_weight: float = 3.0,
    adversarially: bool = True,
    decoder_only: bool = False,
    latent: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """One generator step of the autoencoder / decoder-only distillation stage.

    ``latent`` lets the caller supply a **pre-computed** latent -- from the shard cache, or built
    from cached teacher signals (durations/F0/energy/latent features).  That is what the real
    ``distill-decoder`` stage does: the teacher signals are frozen tensors, so there is no reason
    to re-encode audio every step, and the decoder then trains on exactly the latent
    distribution the text side will produce at synthesis time.
    """
    ae = model.autoencoder
    if latent is None:
        mel = losses["mel_module"].log_mel(wav)
        latent = ae.encode(mel)
        if decoder_only:
            latent = latent.detach()
    recon = ae.decode(latent, length=wav.shape[-1])

    l_mel = losses["mel"](recon, wav)
    l_spec, spec_parts = losses["spectral"](recon, wav)
    total = cfg.train.loss.mel * l_mel + spectral_weight * l_spec
    logs = {"mel": l_mel.detach(), "spectral": l_spec.detach(), "spec_weight": torch.tensor(spectral_weight)}
    logs.update({f"spec_{k}": v.detach() for k, v in spec_parts.items()})

    if adversarially:
        adv, fm = losses["adversarial"](wav, recon, mode="generator")
        total = total + cfg.train.loss.adversarial * adv + cfg.train.loss.feature_match * fm
        logs["adv"] = adv.detach()
        logs["feat_match"] = fm.detach()

    from .losses import phase_lock_loss

    pl = phase_lock_loss(recon, cfg.audio.sample_rate, cfg.audio.n_fft, cfg.audio.hop_length)
    total = total + cfg.train.loss.phase_lock * pl
    logs["phase_lock"] = pl.detach()
    return total, logs, recon


def discriminator_step(
    losses: Dict[str, nn.Module], real: torch.Tensor, fake: torch.Tensor
) -> torch.Tensor:
    loss, _ = losses["adversarial"](real, fake.detach(), mode="discriminator")
    return loss


def flow_step(
    cfg: ParakeetConfig,
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Conditional flow-matching + utterance-length regression."""
    loss, aux = model.flow_loss(
        batch["ids"],
        batch.get("text_mask"),
        batch["latent"],
        ref_mel=batch.get("ref_mel"),
        ref_mask=batch.get("ref_mask"),
        speaker_emb=batch.get("speaker_emb"),
        voice=batch.get("voice"),
        sample_weight=batch.get("teacher_weight"),
    )
    text_mem = model.text(batch["ids"], batch.get("text_mask"))
    _, _, cond = model.conditions(
        batch["ids"],
        batch.get("text_mask"),
        batch.get("ref_mel"),
        batch.get("ref_mask"),
        batch.get("speaker_emb"),
        batch.get("voice"),
    )
    pred_frames = model.length_predictor(text_mem, cond, batch.get("text_mask"))
    target_frames = torch.log(batch["latent"].shape[-1] * torch.ones_like(pred_frames))
    l_len = F.mse_loss(pred_frames, target_frames)
    total = loss + cfg.train.loss.duration * l_len
    logs = {"flow": loss.detach(), "length": l_len.detach()}

    # Style-token objectives.  The tokens are derived here rather than by the loader because they
    # are a *model* product (Q-Former over the mel memory encoder), and because collation must stay
    # tensor-only.  Two mutually exclusive terms, both driven by the paired references:
    #   * a different-speaker reference -> push style away from speaker identity (default), and
    #   * a same-speaker reference pair  -> the optional consistency regulariser.
    style_same = batch.get("style_tokens")
    style_alt = batch.get("style_tokens_alt")
    if style_same is None and batch.get("ref_mel") is not None:
        style_same = model.speaker.encode_style(batch["ref_mel"], batch.get("ref_mask"))
    if style_alt is None and batch.get("ref_mel_neg") is not None:
        style_alt = model.speaker.encode_style(batch["ref_mel_neg"], batch.get("ref_mask_neg"))

    from ..models.speaker import SpeakerConditioner

    if style_same is not None and style_alt is not None and batch.get("ref_mel_neg") is not None:
        l_style = SpeakerConditioner.style_separation_loss(style_same, style_alt)
        total = total + cfg.train.loss.style_separation * l_style
        logs["style_separation"] = l_style.detach()
    elif style_same is not None and style_alt is not None:
        l_style = SpeakerConditioner.cosine_style_loss(style_same, style_alt)
        total = total + cfg.train.loss.style_consistency_pair * l_style
        logs["style_consistency"] = l_style.detach()
    return total, logs


@torch.no_grad()
def reflow_targets(
    cfg: ParakeetConfig,
    teacher_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    teacher_steps: Optional[int] = None,
    cfg_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the high-NFE teacher from fresh noise; return ``(memory, mask, x0, teacher_x1)``."""
    memory, memory_mask, _ = teacher_model.conditions(
        batch["ids"],
        batch.get("text_mask"),
        batch.get("ref_mel"),
        batch.get("ref_mask"),
        batch.get("speaker_emb"),
        batch.get("voice"),
    )
    b, _, t_latent = batch["latent"].shape
    tc = teacher_model.compressed_frames(t_latent)
    shape = (b, cfg.flow.latent_dim * cfg.flow.compress, tc)
    x0 = torch.randn(shape, device=batch["latent"].device)
    x1 = consistency_sample(
        teacher_model.vf,
        memory,
        memory_mask,
        shape,
        steps=teacher_steps or cfg.flow.nfe,
        device=x0.device,
        cfg_scale=cfg_scale,
    )
    return memory, memory_mask, x0, x1


def reflow_step(
    cfg: ParakeetConfig,
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    teacher_model: Optional[nn.Module] = None,
    teacher_steps: Optional[int] = None,
    cfg_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Straighten the flow with a 2-rectified-flow (Reflow) pair, enabling NFE 1-4."""
    teacher = teacher_model or model
    memory, memory_mask, x0, x1 = reflow_targets(cfg, teacher, batch, teacher_steps, cfg_scale)
    x_t, t, v_target = reflow_pair(x0, x1)
    v_pred = model.vf(x_t, t, memory, memory_mask)
    loss = F.mse_loss(v_pred, v_target)
    return loss, {"reflow": loss.detach(), "x1_std": x1.std().detach()}


def tiny_text_step(
    cfg: ParakeetConfig,
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: Optional[TextSideDistillLoss] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Regress cached teacher signals with the small text side (Paradee stage 2)."""
    criterion = criterion or TextSideDistillLoss()
    pred = model.text_side(batch["ids"], batch.get("text_mask"), batch.get("voice"))
    target = {
        "durations": batch["durations"],
        "f0": batch["f0"],
        "energy": batch["energy"],
        "latent_token": batch["latent_token"],
    }
    total, logs = criterion(
        pred, target, batch.get("text_mask"), sample_weight=batch.get("teacher_weight")
    )
    return total, logs


# --------------------------------------------------------------------------------------
# generic loop
# --------------------------------------------------------------------------------------
STAGE_STEPS: Dict[str, str] = {
    "autoencoder": "autoencoder",
    "distill-decoder": "decoder",
    "distill-text": "text",
    "flow": "flow",
    "reflow": "reflow",
}


def run_stage(
    stage: str,
    cfg: ParakeetConfig,
    model: Optional[nn.Module] = None,
    batches: Optional[Callable[[], Dict[str, torch.Tensor]]] = None,
    max_steps: Optional[int] = None,
    out_dir: Optional[str] = None,
    device: Optional[str] = None,
    log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ema_model: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Minimal, dependency-free training loop.

    ``batches`` is a zero-argument callable returning a batch dict (a DataLoader iterator in
    practice, a synthetic generator in the smoke test).  Keeping it a callable is what lets
    the CPU smoke test exercise every stage without any real corpus.
    """
    if stage not in STAGE_STEPS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(STAGE_STEPS)}")
    if batches is None:
        raise ValueError("`batches` callable is required")
    seed_everything(cfg.train.seed)
    dev = resolve_device(device or cfg.train.device)
    model = model or build_model(cfg)
    model.to(dev).train()
    out_dir = Path(out_dir or cfg.train.out_dir)
    steps = max_steps or cfg.train.max_steps

    # Reset trainability for THIS stage.  A previous stage in the same process may have frozen
    # modules (``distill-text`` freezes the whole autoencoder, ``flow`` freezes the encoders), and
    # inheriting that silently means a stage does not train what it claims to train -- e.g.
    # ``distill-decoder`` would leave the decoder frozen after a ``distill-text`` run.
    for p in model.parameters():
        p.requires_grad = True
    if stage == "distill-decoder":
        for module in (model.autoencoder.stem, model.autoencoder.encoder, model.autoencoder.to_latent):
            for p in module.parameters():
                p.requires_grad = False
    elif stage == "distill-text":
        for p in model.autoencoder.parameters():
            p.requires_grad = False
    elif stage in {"flow", "reflow"}:
        # the autoencoder is a frozen representation for the generative half
        for p in model.autoencoder.parameters():
            p.requires_grad = False
        if stage == "reflow":
            # sampler distillation only adapts the vector field; everything else must hold still
            for name, p in model.named_parameters():
                if not name.startswith("vf."):
                    p.requires_grad = False

    opt = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    sched = cosine_warmup_scheduler(opt, cfg.train.warmup_steps, steps)
    ema = EMAModel(model, cfg.train.ema_decay)
    meter = Meter()
    logs: Dict[str, float] = {}

    mel_module = MelSpectrogram(cfg.audio)
    losses: Dict[str, nn.Module] = {
        "mel_module": mel_module,
        "mel": LogMelLoss(mel_module),
        "spectral": MultiResolutionSTFTLoss(),
        "adversarial": AdversarialVocoderLoss().to(dev),
    }
    disc_opt = (
        build_optimizer(losses["adversarial"], cfg.train.lr, cfg.train.weight_decay)
        if stage in {"autoencoder", "distill-decoder"}
        else None
    )
    anneal = SpectralAnnealer()
    text_criterion = TextSideDistillLoss()
    teacher = ema_model

    for step in range(steps):
        batch = batches()
        batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        extra_logs: Dict[str, float] = {}

        if stage in {"autoencoder", "distill-decoder"}:
            spectral_weight = anneal(step) if stage == "distill-decoder" else cfg.train.loss.spectral
            loss, step_logs, fake = autoencoder_step(
                cfg,
                model,
                batch["wav"],
                losses,
                spectral_weight=spectral_weight,
                adversarially=True,
                decoder_only=(stage == "distill-decoder"),
                latent=batch.get("latent"),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            if disc_opt is not None:
                disc_opt.zero_grad(set_to_none=True)
                d_loss = discriminator_step(losses, batch["wav"], fake)
                d_loss.backward()
                disc_opt.step()
                extra_logs["disc"] = float(d_loss.detach())
        elif stage == "flow":
            loss, step_logs = flow_step(cfg, model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
        elif stage == "reflow":
            loss, step_logs = reflow_step(cfg, model, batch, teacher_model=teacher)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
        elif stage == "distill-text":
            loss, step_logs = tiny_text_step(cfg, model, batch, text_criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
        else:  # pragma: no cover - guarded above
            raise AssertionError(stage)

        sched.step()
        ema.update(model)
        step_logs = {k: float(v) for k, v in step_logs.items()}
        step_logs["loss"] = float(loss.detach())
        step_logs["lr"] = float(sched.get_last_lr()[0])
        step_logs.update(extra_logs)
        meter.update(step_logs)

        if (step + 1) % max(1, cfg.train.log_every) == 0:
            logs = meter.mean()
            logs["step"] = step + 1
            if log_fn:
                log_fn(logs)
            meter.reset()
        if cfg.train.save_every and (step + 1) % cfg.train.save_every == 0:
            save_checkpoint(
                out_dir / f"{stage}_step{step+1}.pt", model, opt, step + 1, cfg, ema=ema
            )

    final = meter.mean() or dict(logs)
    final["step"] = steps
    save_checkpoint(out_dir / f"{stage}_last.pt", model, opt, steps, cfg, ema=ema)
    return final


def train_all_stages(
    cfg: ParakeetConfig,
    batches_by_stage: Dict[str, Callable[[], Dict[str, torch.Tensor]]],
    steps_by_stage: Optional[Dict[str, int]] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Run the full curriculum on one model instance (small helper for scripts/tests)."""
    results: Dict[str, Dict[str, float]] = {}
    model = build_model(cfg)
    for stage in ["autoencoder", "distill-text", "distill-decoder", "flow", "reflow"]:
        if cfg.variant == "tiny" and stage in {"flow", "reflow"}:
            continue
        if cfg.variant == "small" and stage in {"distill-text", "distill-decoder"}:
            continue
        if stage not in batches_by_stage:
            continue
        results[stage] = run_stage(
            stage,
            cfg,
            model=model,
            batches=batches_by_stage[stage],
            max_steps=(steps_by_stage or {}).get(stage),
            out_dir=out_dir,
        )
    return results
