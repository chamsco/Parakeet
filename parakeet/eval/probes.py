"""Learning probes: cheap, honest checks that a training pipeline actually fits its targets.

These are *not* quality metrics.  They answer a narrower and more useful question during
development: "did this stage move in the right direction, and do the two halves compose?"  They
are deliberately measured on fixed, comparable inputs (the same corpus before and after, the same
target set) rather than on a running training loss, because a training loss can fall while the
model gets worse.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from ..audio.mel import MelSpectrogram
from ..config import ParakeetConfig
from ..models.duration import normalized_to_durations


def ae_reconstruction_l1(
    model, wavs: Sequence[torch.Tensor], cfg: ParakeetConfig
) -> float:
    """Mean log-mel L1 between input audio and its autoencoder round-trip.

    Independent of the training objective (which includes adversarial and spectral terms), so it
    is a fair before/after measure of the *representation* rather than of the loss balance.
    """
    mel = MelSpectrogram(cfg.audio)
    values = []
    with torch.no_grad():
        for wav in wavs:
            w = wav.reshape(1, -1)
            target = mel.log_mel(w)
            latent = model.autoencoder.encode(target)
            recon = model.autoencoder.decode(latent, length=w.shape[-1])
            values.append(float(F.l1_loss(mel.log_mel(recon), target).item()))
    return sum(values) / max(1, len(values))


def teacher_signal_loss(model, targets: Sequence[Mapping], cfg: ParakeetConfig) -> float:
    """The Tiny distillation objective on each target individually (Tiny models only).

    The sample's own ``teacher_weight`` and ``voice`` are passed through when present: dropping
    either would evaluate the model under conditioning it was not given, which quietly makes a
    voice-conditioned model look worse than an unconditioned one.
    """
    from ..train.stages import tiny_text_step

    values = []
    with torch.no_grad():
        for t in targets:
            batch = {
                "ids": t["ids"][None],
                "text_mask": torch.ones(1, t["ids"].numel(), dtype=torch.bool),
                "durations": t["durations"][None],
                "f0": t["f0"][None],
                "energy": t["energy"][None],
                "latent_token": t["latent_token"][None],
            }
            for key in ("teacher_weight", "voice"):
                value = t.get(key) if isinstance(t, Mapping) else None
                if value is None:
                    continue
                tensor = torch.as_tensor(value).reshape(-1)
                batch[key] = tensor[:1] if tensor.numel() else None
            loss, _ = tiny_text_step(cfg, model, batch)
            values.append(float(loss.item()))
    return sum(values) / max(1, len(values))


def end_to_end_mel_l1(model, targets: Sequence[Mapping], cfg: ParakeetConfig) -> float:
    """Mean log-mel L1 between a text-only synthesis and the target audio.

    For Tiny this runs the whole distilled path: text -> durations/F0/energy/latent features ->
    frame latent -> decoder -> waveform.  It therefore measures composition, not just regression.
    """
    mel = MelSpectrogram(cfg.audio)
    values = []
    with torch.no_grad():
        for t in targets:
            ids = t["ids"][None]
            mask = torch.ones_like(ids, dtype=torch.bool)
            wav = model.synthesize(ids, mask)
            a = mel.log_mel(wav)
            b = mel.log_mel(t["wav"].reshape(1, -1))
            n = min(a.shape[-1], b.shape[-1])
            values.append(float(F.l1_loss(a[..., :n], b[..., :n]).item()))
    return sum(values) / max(1, len(values))


def duration_error_frames(model, targets: Sequence[Mapping]) -> float:
    """Mean absolute per-token duration error, in latent frames."""
    values = []
    with torch.no_grad():
        for t in targets:
            ids = t["ids"][None]
            side = model.text_side(ids)
            pred = normalized_to_durations(side["log_duration"])
            values.append(float((pred[0] - t["durations"].float()).abs().mean().item()))
    return sum(values) / max(1, len(values))


def latent_normalizer_summary(model) -> dict:
    """Latent statistics, for sanity-checking that the normaliser was fitted."""
    norm = getattr(model, "latent_norm", None)
    if norm is None:
        return {}
    return {
        "abs_mean": float(norm.mean.abs().mean().item()),
        "mean_sigma": float(norm.var.sqrt().mean().item()),
        "updates": int(norm.n.item()),
    }
