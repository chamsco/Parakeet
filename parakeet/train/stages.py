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

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio.mel import MelSpectrogram
from ..config import ParakeetConfig
from ..models import build_model
from ..models.duration import normalized_to_durations
from ..models.flow import consistency_sample, fold_time, make_xt, reflow_pair, sample_timesteps
from .common import (
    EMAModel,
    Meter,
    build_optimizer,
    cosine_warmup_scheduler,
    count_trainable,
    freeze_,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    seed_everything,
    write_run_metadata,
)
from .losses import (
    AdversarialVocoderLoss,
    DistillSignalWeights,
    LogMelLoss,
    MultiResolutionSTFTLoss,
    SpectralAnnealer,
    TextSideDistillLoss,
    consistency_distillation_loss,
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
        latent_token=batch.get("latent_token"),
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
    # surface the coarse-to-fine plan term: if the plan is being ignored, its loss says so, and a hidden
    # auxiliary objective is exactly how the conditioning problem in round 39 went unnoticed for so long
    for key, value in aux.items():
        if isinstance(value, torch.Tensor) and value.dim() == 0 and key not in logs:
            logs[key] = value.detach()

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
    # the named loss rather than a second inline MSE: they were duplicates, and the named one
    # already supports a frame mask for callers that need it
    loss = consistency_distillation_loss(v_pred, v_target)
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
def text_audio_step(
    cfg: ParakeetConfig,
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    losses: Dict[str, nn.Module],
    criterion: Optional[TextSideDistillLoss] = None,
    loss_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """Train the text side **through the decoder**, against the target audio (round 23).

    ``distill-text`` regresses cached teacher latents with an L1 in latent space, and round 22 measured
    what that costs: the text side fits its objective well (loss 0.655) while the audio it renders is
    unintelligible, because a small error in latent space is not a small error in the decoder's output.
    This step closes the loop instead: predict the token signals, build the decoder input with the same
    call synthesis makes, decode, and compare the **audio** to the teacher's.

    Everything except the autoencoder is trainable, so the gradients reach the text side, the duration
    head and -- for the first time in a path that is trained rather than merely executed --
    ``prosody_proj``.  The token-signal terms stay in as an auxiliary objective (``audio_aux`` weight):
    an audio-only loss leaves the *length* free, since rounding durations to frames is not
    differentiable, and the cached signals are the only thing that pins it.
    """
    # the weights come from the config so the mean-invariant latent term can be A/B'd with `--set`
    # instead of being a constant only a code change can reach
    if criterion is None:
        criterion = TextSideDistillLoss(
            DistillSignalWeights(
                latent_contrast=float(getattr(cfg.train.loss, "signal_latent_contrast", 0.0))
            )
        )
    pred = model.text_side(batch["ids"], batch.get("text_mask"), batch.get("voice"))
    durations = normalized_to_durations(pred["log_duration"])
    latent, frame_mask = model.decoder_latent_from_tokens(
        pred["latent_token"], durations, pred["f0"], pred["energy"]
    )
    target_audio = batch["wav"]
    recon = model.autoencoder.decode(latent, length=target_audio.shape[-1])

    # The mel and multi-resolution STFT losses are built on torch.stft, which needs a complex dtype.
    # DirectML does not have one ("Invalid or unsupported data type ComplexFloat"), so on a DirectML
    # device the *losses* run on the CPU while the model runs on the GPU: what crosses the boundary is
    # the rendered and target audio, a couple of megabytes a step, and autograd carries the gradient
    # back across it.  `loss_device` is None on CPU-only runs, which keeps the original path identical.
    if loss_device is not None and loss_device != recon.device:
        l_mel = losses["mel"](recon.to(loss_device), target_audio.to(loss_device))
        l_spec, spec_parts = losses["spectral"](
            recon.to(loss_device), target_audio.to(loss_device)
        )
    else:
        l_mel = losses["mel"](recon, target_audio)
        l_spec, spec_parts = losses["spectral"](recon, target_audio)
    total = cfg.train.loss.audio_mel * l_mel + cfg.train.loss.audio_spectral * l_spec
    logs = {
        "audio_mel": l_mel.detach(),
        "audio_spectral": l_spec.detach(),
        "rendered_frames": torch.tensor(float(latent.shape[-1])),
        "target_frames": torch.tensor(float(target_audio.shape[-1] / cfg.audio.hop_length)),
    }
    logs.update({f"audio_spec_{k}": v.detach() for k, v in spec_parts.items()})

    aux = float(cfg.train.loss.audio_aux)
    if aux > 0:
        target = {
            "durations": batch["durations"],
            "f0": batch["f0"],
            "energy": batch["energy"],
            "latent_token": batch["latent_token"],
        }
        aux_loss, aux_logs = criterion(
            pred, target, batch.get("text_mask"), sample_weight=batch.get("teacher_weight")
        )
        total = total + aux * aux_loss
        logs["aux"] = aux_loss.detach()
        logs.update({f"aux_{k}": v.detach() for k, v in aux_logs.items()})
    return total, logs, recon


STAGE_STEPS: Dict[str, str] = {
    "autoencoder": "autoencoder",
    "distill-decoder": "decoder",
    "distill-text": "text",
    "distill-audio": "text_audio",
    "flow": "flow",
    "reflow": "reflow",
}


def _gradients_are_finite(model: nn.Module) -> bool:
    """Every gradient that exists must be finite.  Clipping does not rescue a NaN gradient."""
    return all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
    )


def _record_divergence(
    count: int, first: Optional[int], step: int
) -> Tuple[int, Optional[int]]:
    return count + 1, (step + 1 if first is None else first)


def _optimise(step: int, model: nn.Module, opt: torch.optim.Optimizer, loss: torch.Tensor, grad_clip: float) -> bool:
    """Backward, guard, clip, step.  Returns False when the update was skipped as divergent.

    Skipping is the difference between "this run diverged at step N" and "the report is all NaN":
    round 24 measured 2245 of 4000 skipped steps in one run, all of which would otherwise have written
    NaN into the parameters.
    """
    if not bool(torch.isfinite(loss)):
        return False
    loss.backward()
    if not _gradients_are_finite(model):
        return False
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return True


def _named_top_level(model: nn.Module):
    """Direct child modules with names, for the frozen/trainable report."""
    return list(model.named_children())


def _module_trainable(module: nn.Module) -> bool:
    """True if the module has parameters and at least one is trainable.

    A module with *no* parameters (a normaliser holding only buffers, say) is neither trainable nor
    frozen -- reporting it as frozen would be a false alarm in the diagnostic that exists to catch
    stages that train nothing.
    """
    params = list(module.parameters())
    return any(p.requires_grad for p in params) if params else True


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
    run_metadata: Optional[Dict[str, Any]] = None,
    resume_from: Optional[str] = None,
) -> Dict[str, float]:
    """Minimal, dependency-free training loop.

    ``batches`` is a zero-argument callable returning a batch dict (a DataLoader iterator in
    practice, a synthetic generator in the smoke test).  Keeping it a callable is what lets
    the CPU smoke test exercise every stage without any real corpus.

    ``run_metadata`` is merged into the provenance written to ``<out_dir>/run.json`` (git revision,
    config hash, trainable parameter count, plus whatever the caller knows -- teacher mixture,
    corpus fingerprint).  The trainable count is also logged, because a stage that trains far fewer
    parameters than intended has silently done nothing: that exact bug (an inherited freeze list)
    made ``distill-decoder`` train nothing at all before it was caught.
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
    elif stage == "distill-audio":
        # the autoencoder is the fixed renderer: only the text side (and the prosody projection inside
        # the token->frame path) learns from the audio comparison
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

    total_params = sum(p.numel() for p in model.parameters())
    trainable = count_trainable(model)
    frozen = [
        name for name, module in _named_top_level(model) if not _module_trainable(module)
    ]
    logs["trainable_params"] = float(trainable)
    logs["frozen_modules"] = float(len(frozen))
    if log_fn is not None:
        # No "loss" key: this is a banner, not a measurement, and the placeholder used to be
        # `float("nan")` -- which reads as a diverging run and cost time to investigate twice.
        log_fn(
            {
                "step": 0,
                "trainable_params": float(trainable),
                "total_params": float(total_params),
                "frozen": ",".join(frozen) if frozen else "none",
            }
        )
    write_run_metadata(
        out_dir,
        cfg,
        stage=stage,
        extra={
            "max_steps": int(steps),
            "total_params": int(total_params),
            "trainable_params": int(trainable),
            "frozen_modules": frozen,
            **(run_metadata or {}),
        },
    )

    mel_module = MelSpectrogram(cfg.audio)
    losses: Dict[str, nn.Module] = {
        "mel_module": mel_module,
        "mel": LogMelLoss(mel_module),
        "spectral": MultiResolutionSTFTLoss(),
        "adversarial": AdversarialVocoderLoss().to(dev),
    }
    disc_opt = (
        build_optimizer(losses["adversarial"], cfg.train.lr, cfg.train.weight_decay)
        # a reconstruction-only phase is legitimate (and 24x cheaper per step): with the adversarial
        # weight at zero the discriminator would be trained against a generator that is not chasing it
        if stage in {"autoencoder", "distill-decoder"} and cfg.train.loss.adversarial > 0.0
        else None
    )
    anneal = SpectralAnnealer()
    text_criterion = TextSideDistillLoss()
    teacher = ema_model

    # ------------------------------------------------------------------ resume
    start_step = 0
    nonfinite_steps = 0
    first_nonfinite_step: Optional[int] = None
    if resume_from:
        payload = load_checkpoint(
            resume_from,
            model,
            optimizer=opt,
            ema=ema,
            discriminator=losses["adversarial"] if disc_opt is not None else None,
        )
        start_step = int(payload.get("step") or 0)
        # Put the LR schedule where it left off.  The loop calls sched.step() *after* each update,
        # so advancing once here makes iteration `start_step` use fn(start_step) -- exactly the LR an
        # uninterrupted run would have used.  Without this a resumed run silently restarts the
        # cosine schedule from the warmup peak.
        if start_step > 0:
            sched.last_epoch = start_step - 1
            # torch warns that the scheduler stepped before the optimizer stepped, which is exactly
            # what repositioning a schedule looks like; the alternative is a private-API poke at
            # _step_count.  The LR below is verified by test_resume_continues_the_lr_schedule.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sched.step()
        batch_state = (payload.get("extra") or {}).get("batch_state")
        if batch_state is not None and hasattr(batches, "load_state_dict"):
            batches.load_state_dict(batch_state)
        resumed_from_step = start_step
        print(f"[stage {stage}] resumed from {resume_from} at step {start_step} "
              f"(lr {sched.get_last_lr()[0]:.3e})")

    for step in range(start_step, steps):
        batch = batches()
        batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        extra_logs: Dict[str, float] = {}

        if stage in {"autoencoder", "distill-decoder"}:
            spectral_weight = anneal(step) if stage == "distill-decoder" else cfg.train.loss.spectral
            # `distill-decoder` exists to train the decoder on the distribution *inference* produces.
            # It used to consume the cached frame-level latent (`batch["latent"]`), which is clean --
            # and round 20 measured the consequence: the token-expanded path that synthesis actually
            # builds scored WER 0.870 against 0.167 for the frame latent.  Building the input through
            # `decoder_latent_from_tokens` also puts `prosody_proj` in the graph, so it finally
            # receives gradients instead of staying a randomly initialised module applied at
            # inference only.
            stage_latent = batch.get("latent")
            if stage == "distill-decoder" and batch.get("wav") is None:
                raise ValueError(
                    "the `distill-decoder` stage needs the target waveform in its batch, which the "
                    "latent cache only started storing in round 21; rebuild the cache with "
                    "`cache_teacher_corpus` (its shards now carry `wav`).  Before that this stage "
                    "could only run under --dry-run"
                )
            if (
                stage == "distill-decoder"
                and cfg.autoencoder.decoder_uses_token_latents
                and batch.get("latent_token") is not None
            ):
                stage_latent, _ = model.decoder_latent_from_tokens(
                    batch["latent_token"],
                    batch["durations"],
                    batch.get("f0"),
                    batch.get("energy"),
                )
                stage_latent = stage_latent.detach()
                extra_logs["decoder_input"] = 1.0  # 1 == token-expanded, 0 == cached frame latent
            loss, step_logs, fake = autoencoder_step(
                cfg,
                model,
                batch["wav"],
                losses,
                spectral_weight=spectral_weight,
                adversarially=bool(cfg.train.loss.adversarial > 0.0),
                decoder_only=(stage == "distill-decoder"),
                latent=stage_latent,
            )
            if not torch.isfinite(loss):
                # Divergence is a *finding*, not something to discover from NaNs in a final report.
                # Round 20: a reconstruction-only run on real speech went mel 2.06 -> 0.50 and then
                # to NaN somewhere between steps 1200 and 1800, and the only visible symptom was a
                # report of all-NaN metrics at the end.  Skip the update, keep the schedule moving so
                # the budget still maps to the LR curve, and record the step.
                nonfinite_steps += 1
                if first_nonfinite_step is None:
                    first_nonfinite_step = step + 1
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            loss.backward()
            # A finite loss does not imply finite gradients: round 24's scaled-corpus run went
            # non-finite at step 1756, and with only the loss checked the parameters were written with
            # NaN grads, so every later loss was NaN (2245 of 4000 steps "skipped" while the weights
            # stayed broken).  `_optimise` checks the gradients too and skips the update if any is
            # non-finite.
            if not _gradients_are_finite(model):
                nonfinite_steps, first_nonfinite_step = _record_divergence(
                    nonfinite_steps, first_nonfinite_step, step
                )
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            if disc_opt is not None:
                disc_opt.zero_grad(set_to_none=True)
                d_loss = discriminator_step(losses, batch["wav"], fake)
                if not torch.isfinite(d_loss):
                    nonfinite_steps += 1
                    if first_nonfinite_step is None:
                        first_nonfinite_step = step + 1
                    disc_opt.zero_grad(set_to_none=True)
                else:
                    d_loss.backward()
                    disc_opt.step()
                    extra_logs["disc"] = float(d_loss.detach())
        elif stage == "flow":
            loss, step_logs = flow_step(cfg, model, batch)
            if not _optimise(step, model, opt, loss, cfg.train.grad_clip):
                nonfinite_steps, first_nonfinite_step = _record_divergence(
                    nonfinite_steps, first_nonfinite_step, step
                )
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
        elif stage == "reflow":
            loss, step_logs = reflow_step(cfg, model, batch, teacher_model=teacher)
            if not _optimise(step, model, opt, loss, cfg.train.grad_clip):
                nonfinite_steps, first_nonfinite_step = _record_divergence(
                    nonfinite_steps, first_nonfinite_step, step
                )
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
        elif stage == "distill-text":
            loss, step_logs = tiny_text_step(cfg, model, batch, text_criterion)
            if not _optimise(step, model, opt, loss, cfg.train.grad_clip):
                nonfinite_steps, first_nonfinite_step = _record_divergence(
                    nonfinite_steps, first_nonfinite_step, step
                )
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
        elif stage == "distill-audio":
            loss, step_logs, _recon = text_audio_step(cfg, model, batch, losses, text_criterion)
            if not _optimise(step, model, opt, loss, cfg.train.grad_clip):
                nonfinite_steps, first_nonfinite_step = _record_divergence(
                    nonfinite_steps, first_nonfinite_step, step
                )
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
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
                out_dir / f"{stage}_step{step+1}.pt",
                model,
                opt,
                step + 1,
                cfg,
                ema=ema,
                discriminator=losses["adversarial"] if disc_opt is not None else None,
                extra={"stage": stage, "batch_state": _batch_state(batches)},
            )

    final = meter.mean() or dict(logs)
    final["step"] = steps
    if nonfinite_steps:
        # surface divergence in the returned logs *and* in the provenance, so it cannot be missed
        final["nonfinite_steps"] = float(nonfinite_steps)
        final["first_nonfinite_step"] = float(first_nonfinite_step or 0)
        print(
            f"[stage {stage}] WARNING: {nonfinite_steps} of {steps} steps produced a non-finite loss "
            f"(first at step {first_nonfinite_step}); those updates were skipped"
        )
    else:
        final["nonfinite_steps"] = 0.0
    if resume_from:
        # carried through the *final* dict: the per-interval logs are replaced by meter.mean(),
        # so writing it into `logs` earlier silently disappeared
        final["resumed_from"] = float(start_step)
    save_checkpoint(
        out_dir / f"{stage}_last.pt",
        model,
        opt,
        steps,
        cfg,
        ema=ema,
        discriminator=losses["adversarial"] if disc_opt is not None else None,
        extra={"stage": stage, "batch_state": _batch_state(batches)},
    )
    if nonfinite_steps:
        (out_dir / "divergence.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "steps": steps,
                    "nonfinite_steps": nonfinite_steps,
                    "first_nonfinite_step": first_nonfinite_step,
                    "skipped_updates": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return final


def _batch_state(batches) -> Optional[Dict[str, Any]]:
    """The batch source's RNG/order state, when it exposes one (needed for an exact resume)."""
    getter = getattr(batches, "state_dict", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:  # noqa: BLE001 - a source without serialisable state must not break saving
        return None


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
