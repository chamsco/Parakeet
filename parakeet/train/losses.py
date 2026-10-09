"""Loss functions for Parakeet's staged distillation.

The single most important empirical result we took from Paradee (arXiv 2610.06817, §4) is
that **loss balance dominates decoder size** when distilling a vocoder:

    spectral-only            -> UTMOS 2.98   (robotic second voice)
    spectral weight 45       -> UTMOS 3.02
    spectral weight 10       -> UTMOS 4.29
    spectral weight  3       -> UTMOS 4.37   (teacher 4.52)

So :class:`LossConfig` defaults to ``spectral=3.0`` with ``adversarial=1.0``, and
:class:`SpectralAnnealer` reproduces the 45 -> 10 -> 3 schedule for reproducing that study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio.mel import MelSpectrogram
from ..models.duration import durations_to_normalized


# --------------------------------------------------------------------------------------
# Reconstruction losses
# --------------------------------------------------------------------------------------
class MultiResolutionSTFTLoss(nn.Module):
    """Spectral-convergence + log-magnitude L1 over several STFT resolutions."""

    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Sequence[int] = (128, 256, 512),
        win_sizes: Sequence[int] = (512, 1024, 2048),
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = list(hop_sizes)
        self.win_sizes = list(win_sizes) if win_sizes else list(fft_sizes)
        self.eps = eps
        for n, w in zip(self.fft_sizes, self.win_sizes):
            self.register_buffer(f"win_{n}", torch.hann_window(w), persistent=False)

    def _stft(self, x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        window = getattr(self, f"win_{n_fft}")
        x = x.reshape(-1, x.shape[-1])
        return torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=win, window=window,
            center=True, return_complex=True,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        n = min(x.shape[-1], y.shape[-1])
        x, y = x[..., :n], y[..., :n]
        sc_total = x.new_zeros(())
        mag_total = x.new_zeros(())
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            sx = self._stft(x, n_fft, hop, win)
            sy = self._stft(y, n_fft, hop, win)
            mx, my = sx.abs(), sy.abs()
            sc = torch.linalg.norm(my - mx, dim=(-2, -1)) / (torch.linalg.norm(my, dim=(-2, -1)) + self.eps)
            lm = F.l1_loss(torch.log(mx + self.eps), torch.log(my + self.eps))
            sc_total = sc_total + sc.mean()
            mag_total = mag_total + lm
        k = len(self.fft_sizes)
        total = (sc_total + mag_total) / k
        return total, {"sc": sc_total / k, "log_mag": mag_total / k}


class LogMelLoss(nn.Module):
    def __init__(self, mel: MelSpectrogram) -> None:
        super().__init__()
        self.mel = mel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = min(x.shape[-1], y.shape[-1])
        return F.l1_loss(self.mel.log_mel(x[..., :n]), self.mel.log_mel(y[..., :n]))


class SpectralAnnealer:
    """Reproduces Paradee's spectral-weight study: 45 -> 10 -> 3."""

    def __init__(self, schedule: Sequence[Tuple[int, float]] = ((0, 45.0), (3000, 10.0), (6000, 3.0))) -> None:
        self.schedule = list(schedule)

    def __call__(self, step: int) -> float:
        w = self.schedule[0][1]
        for at, value in self.schedule:
            if step >= at:
                w = value
        return w


# --------------------------------------------------------------------------------------
# Adversarial losses (HiFi-GAN style MPD/MSD)
# --------------------------------------------------------------------------------------
class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, channels: Sequence[int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        self.period = period
        blocks: List[nn.Module] = []
        in_ch = 1
        for ch in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, ch, (5, 1), (3, 1), padding=(2, 0)),
                    nn.LeakyReLU(0.1),
                )
            )
            in_ch = ch
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if x.dim() == 2:
            x = x[:, None, :]
        b, c, t = x.shape
        pad = (-t) % self.period
        if pad:
            x = F.pad(x, (0, pad), mode="reflect")
        x = x.reshape(b, c, (t + pad) // self.period, self.period)
        feats: List[torch.Tensor] = []
        h = x
        for blk in self.blocks:
            h = blk(h)
            feats.append(h)
        logits = self.head(h)
        return logits, feats


class ScaleDiscriminator(nn.Module):
    def __init__(self, channels: Sequence[int] = (32, 64, 128, 256), kernel: int = 15) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Conv1d(1, channels[0], kernel, padding=kernel // 2)]
        in_ch = channels[0]
        for ch in channels[1:]:
            layers += [
                nn.LeakyReLU(0.1),
                nn.Conv1d(in_ch, ch, kernel, stride=2, padding=kernel // 2),
            ]
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv1d(in_ch, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if x.dim() == 2:
            x = x[:, None, :]
        feats: List[torch.Tensor] = []
        h = x
        for m in self.net:
            h = m(h)
            if isinstance(m, nn.Conv1d):
                feats.append(h)
        return self.head(h), feats


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11)) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        return [d(x) for d in self.discriminators]


class MultiScaleDiscriminator(nn.Module):
    """3 scales with average pooling between (HiFi-GAN MSD, simplified with Conv1d)."""

    def __init__(self, n_scales: int = 3) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([ScaleDiscriminator() for _ in range(n_scales)])
        self.pool = nn.AvgPool1d(4, 2, padding=2, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        outs = []
        for i, d in enumerate(self.discriminators):
            outs.append(d(x))
            x = self.pool(x)
        return outs


def discriminator_loss(
    real_outs: List[Tuple[torch.Tensor, List[torch.Tensor]]],
    fake_outs: List[Tuple[torch.Tensor, List[torch.Tensor]]],
    hinge: bool = True,
) -> torch.Tensor:
    loss = 0.0
    for (real, _), (fake, _) in zip(real_outs, fake_outs):
        if hinge:
            loss = loss + F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
        else:
            loss = loss + F.mse_loss(real, torch.ones_like(real)) + F.mse_loss(fake, torch.zeros_like(fake))
    return loss / max(1, len(real_outs))


def generator_adversarial_loss(
    fake_outs: List[Tuple[torch.Tensor, List[torch.Tensor]]], hinge: bool = True
) -> torch.Tensor:
    loss = 0.0
    for fake, _ in fake_outs:
        loss = loss + (-fake.mean() if hinge else F.mse_loss(fake, torch.ones_like(fake)))
    return loss / max(1, len(fake_outs))


def feature_matching_loss(
    real_outs: List[Tuple[torch.Tensor, List[torch.Tensor]]],
    fake_outs: List[Tuple[torch.Tensor, List[torch.Tensor]]],
) -> torch.Tensor:
    loss = 0.0
    n = 0
    for (_, real_feats), (_, fake_feats) in zip(real_outs, fake_outs):
        for rf, ff in zip(real_feats, fake_feats):
            loss = loss + F.l1_loss(ff, rf.detach())
            n += 1
    return loss / max(1, n)


class AdversarialVocoderLoss(nn.Module):
    """Combined GAN objective used in the 'distill-decoder' and autoencoder stages."""

    def __init__(self, hinge: bool = True) -> None:
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.hinge = hinge

    def discriminate(self, x: torch.Tensor):
        return self.mpd(x) + self.msd(x)

    def forward(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        mode: str = "generator",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        real_outs = self.discriminate(real.detach() if mode == "generator" else real)
        fake_outs = self.discriminate(fake)
        if mode == "discriminator":
            return discriminator_loss(real_outs, fake_outs, self.hinge), None
        adv = generator_adversarial_loss(fake_outs, self.hinge)
        fm = feature_matching_loss(real_outs, fake_outs)
        return adv, fm


# --------------------------------------------------------------------------------------
# Phase losses (Paradee's "buzz" fix, made trainable)
# --------------------------------------------------------------------------------------
def _band_bins(freqs: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return (freqs >= lo) & (freqs <= hi)


def phase_linearity_loss(
    spec: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    band: Tuple[float, float] = (2000.0, 8000.0),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalise non-linear phase-vs-frequency in a high band.

    A glottal pulse train has a phase spectrum that is *linear in frequency* (slope set by the
    pulse position).  Random/unstructured phase in 2-8 kHz is exactly what Paradee identifies
    as the residual "buzz", so we penalise the second difference of the phase across frequency:

        L = mean(1 - cos(psi[k+1] - 2 psi[k] + psi[k-1]))

    which is invariant to the absolute phase and to a constant slope -- i.e. it only asks the
    phase to be *coherent*, not to be a particular value.  Differentiable, no extra parameters.
    """
    freqs = torch.linspace(0.0, sample_rate / 2, spec.shape[-2], device=spec.device)
    idx = torch.nonzero(_band_bins(freqs, *band), as_tuple=False).flatten()
    if idx.numel() < 3:
        return spec.new_zeros(())
    psi = torch.angle(spec[:, idx, :])
    second = psi[:, 2:, :] - 2.0 * psi[:, 1:-1, :] + psi[:, :-2, :]
    return (1.0 - torch.cos(second)).mean()


def phase_lock_loss(
    wav: torch.Tensor,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    band: Tuple[float, float] = (2000.0, 8000.0),
) -> torch.Tensor:
    return phase_linearity_loss(
        torch.stft(
            wav.reshape(-1, wav.shape[-1]),
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=wav.device),
            return_complex=True,
        ),
        sample_rate,
        n_fft,
        band=band,
    )


# --------------------------------------------------------------------------------------
# Distilled teacher-signal losses (Tiny text side)
# --------------------------------------------------------------------------------------
@dataclass
class DistillSignalWeights:
    duration: float = 1.0
    f0: float = 1.0
    energy: float = 1.0
    latent: float = 1.0


class TextSideDistillLoss(nn.Module):
    """Regress the *teacher's own cached signals*: durations, F0, energy, phoneme features.

    This is the Paradee trick that removes alignment learning entirely: because the targets
    are fixed tensors saved at corpus-build time, the small text side is a plain regression
    problem and needs no teacher at training time (the teacher is frozen and offline).

    Target conventions matter here and are enforced by the pipeline: durations in log space,
    **F0 as normalised log-Hz in [0, 1]**, **energy as normalised dBFS in [0, 1]**, and the
    per-token latent feature in the autoencoder's (normalised) latent space.  Regressing raw dB or
    quantised F0 bin *indices* makes one term dominate by two orders of magnitude -- caught by
    ``tests/test_learning.py``, which is why both prosody targets are O(1).
    """

    def __init__(self, weights: Optional[DistillSignalWeights] = None) -> None:
        super().__init__()
        self.w = weights or DistillSignalWeights()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """``sample_weight`` is the per-sample teacher-mixture weight (see MultiTeacherMixer).

        Every term is reduced **per sample** and then combined with the (batch-normalised) weight,
        so changing the teacher mixture re-weights the gradient instead of being silently ignored.
        """
        log_dur = durations_to_normalized(target["durations"].to(pred["log_duration"].dtype))
        l_dur = weighted_mean(per_sample_l1(pred["log_duration"], log_dur, mask), sample_weight)
        l_f0 = weighted_mean(per_sample_l1(pred["f0"], target["f0"], mask), sample_weight)
        l_en = weighted_mean(per_sample_l1(pred["energy"], target["energy"], mask), sample_weight)
        l_lat = weighted_mean(
            per_sample_mse(pred["latent_token"], target["latent_token"], mask), sample_weight
        )

        total = (
            self.w.duration * l_dur
            + self.w.f0 * l_f0
            + self.w.energy * l_en
            + self.w.latent * l_lat
        )
        return total, {
            "duration": l_dur.detach(),
            "f0": l_f0.detach(),
            "energy": l_en.detach(),
            "latent": l_lat.detach(),
        }


# --------------------------------------------------------------------------------------
# Multi-teacher mixing
# --------------------------------------------------------------------------------------
class MultiTeacherMixer:
    """Turns a teacher mixture into **per-sample weights that actually reach the loss**.

    Why a *mix* of teachers rather than one:

    * Orpheus -> expressive, tag-controllable, but 3B and codec-limited fidelity;
    * MiniMax -> high-fidelity prosody, but closed and (often) ToS-restricted;
    * Kokoro  -> permissive and fast, but flat affect.

    They are combined at the *audio/latent* level (teacher-agnostic), and each sample carries a
    weight from its teacher's mixture share multiplied by the data pipeline's quality score, so a
    bad synthesis cannot drag the student down.

    Two halves, deliberately separated:

    * :meth:`weights` builds the raw per-sample weight when the **cache is written**
      (:func:`parakeet.data.features.build_latent_cache`).  It is stored with the sample so the
      provenance survives into training.
    * :meth:`normalize` rescales a batch of weights to mean 1 **inside the loss**, so that changing
      the mixture re-weights the gradient *without* changing the overall learning rate.

    An important subtlety: normalising per item would make every weight 1.0, so normalisation must
    happen at batch level -- which is why the loss takes a tensor and not the teacher ids.
    """

    def __init__(
        self,
        teacher_weights: Optional[Dict[str, float]] = None,
        min_weight: float = 0.05,
    ) -> None:
        self.teacher_weights = teacher_weights or {}
        self.min_weight = min_weight

    def weights(
        self,
        teacher_ids: Sequence[str],
        quality: Optional[torch.Tensor] = None,
        device=None,
        dtype=torch.float32,
    ) -> torch.Tensor:
        """Raw per-sample weight = mixture share x quality (clamped, never dropped entirely)."""
        w = torch.tensor(
            [self.teacher_weights.get(t, 1.0) for t in teacher_ids], device=device, dtype=dtype
        )
        if quality is not None:
            w = w * quality.to(device=device, dtype=dtype)
        return w.clamp_min(self.min_weight)

    def normalize(self, weight: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Rescale a batch of weights to mean 1 (identity for ``None``)."""
        if weight is None:
            return None
        w = weight.detach().to(dtype=torch.float32).clamp_min(self.min_weight)
        return (w / w.mean().clamp_min(1e-8)).to(weight.dtype)


def weighted_mean(per_sample: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean of a per-sample tensor, optionally re-weighted (weights are normalised to mean 1)."""
    if weight is None:
        return per_sample.mean()
    w = weight.detach().to(dtype=per_sample.dtype).clamp_min(1e-8)
    w = w / w.mean().clamp_min(1e-8)
    return (per_sample * w).mean()


def _masked_count(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Number of valid elements per sample, broadcast to every non-batch dimension.

    The mask must be expanded to the full shape before counting: a ``(B, T)`` mask against a
    ``(B, T, C)`` tensor has to divide by ``T * C``, not ``T``.  Getting this wrong inflated the
    latent term of the distillation loss by a factor of ``C`` (24x) and would have silently made it
    dominate the prosody terms again.
    """
    m = mask.to(like.dtype)
    while m.dim() < like.dim():
        m = m.unsqueeze(-1)
    return m.expand_as(like)


def per_sample_l1(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Mean absolute error **per sample** (masked positions excluded, not merely zeroed).

    ``F.l1_loss`` averages over padded positions too, which silently dilutes the loss on short
    samples; per-sample normalisation by the valid count is both correct and what per-sample
    teacher weighting needs.
    """
    diff = (pred - target).abs()
    dims = tuple(range(1, diff.dim()))
    if mask is None:
        return diff.mean(dim=dims)
    m = _masked_count(mask, diff)
    return (diff * m).sum(dim=dims) / m.sum(dim=dims).clamp_min(1.0)


def per_sample_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Mean squared error **per sample** (see :func:`per_sample_l1`)."""
    diff = (pred - target).pow(2)
    dims = tuple(range(1, diff.dim()))
    if mask is None:
        return diff.mean(dim=dims)
    m = _masked_count(mask, diff)
    return (diff * m).sum(dim=dims) / m.sum(dim=dims).clamp_min(1.0)


def consistency_distillation_loss(
    student_pred: torch.Tensor,
    teacher_target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Latent-space consistency loss for few-step sampler distillation."""
    if mask is None:
        return F.mse_loss(student_pred, teacher_target)
    m = mask.unsqueeze(1).to(student_pred.dtype)
    return ((student_pred - teacher_target).pow(2) * m).sum() / m.sum().clamp_min(1.0)


def build_loss_bundle(cfg) -> Dict[str, nn.Module]:
    """Convenience factory for the standard Parakeet objective set."""
    mel = MelSpectrogram(cfg.audio)
    return {
        "mel": LogMelLoss(mel),
        "spectral": MultiResolutionSTFTLoss(),
        "adversarial": AdversarialVocoderLoss(),
    }
