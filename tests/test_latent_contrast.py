"""The mean-invariant latent term, and the measurement that motivated it.

Round 30's fit diagnosis: the text side matched the teacher's token latent at a flattened cosine of
0.805 while its **per-dimension correlation was 0.126** -- it had learned the average latent, which is
what MSE between raw values rewards.  The mel proxy looked healthy at WER 1.0, in the same run.

`per_sample_latent_contrast` centres both sides across tokens, per dimension, so it is blind to the mean
and can only be reduced by matching the variation.  A test of "it decreases during training" would be
slow and flaky; what matters is the three algebraic properties that make it the right term.
"""

from __future__ import annotations

import torch

from parakeet.train.losses import (
    DistillSignalWeights,
    TextSideDistillLoss,
    per_sample_latent_contrast,
)


def _targets(tokens: int = 6, dims: int = 8, batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch, tokens, dims, generator=generator)


def test_contrast_is_zero_for_a_perfect_prediction():
    target = _targets()
    assert float(per_sample_latent_contrast(target.clone(), target).abs().max()) < 1e-6


def test_contrast_ignores_a_per_dimension_shift():
    """The failure being fixed is prediction = the mean, which a raw MSE cannot distinguish."""
    target = _targets()
    wrong_mean = target + 5.0
    raw = float((wrong_mean - target).pow(2).mean())
    assert raw > 20.0
    assert float(per_sample_latent_contrast(wrong_mean, target).abs().max()) < 1e-6, (
        "a per-dimension offset is the mean, and this term must not see it"
    )


def test_a_constant_prediction_cannot_cheat():
    """Collapsing to the mean must cost the target's variance, not zero."""
    target = _targets()
    constant = torch.zeros_like(target)
    contrast = float(per_sample_latent_contrast(constant, target).mean())
    variance = float(target.var(dim=1, unbiased=False).mean())
    assert abs(contrast - variance) < 1e-4, (contrast, variance)


def test_the_loss_only_pays_for_the_contrast_when_asked():
    pred = {"log_duration": torch.zeros(1, 6), "f0": torch.zeros(1, 6),
            "energy": torch.zeros(1, 6), "latent_token": _targets(batch=1)}
    target = {"durations": torch.full((1, 6), 4.0), "f0": torch.zeros(1, 6),
              "energy": torch.zeros(1, 6), "latent_token": _targets(batch=1)}
    without = TextSideDistillLoss(DistillSignalWeights())
    _, logs_off = without(pred, target)
    assert "latent_contrast" not in logs_off, "the default must not change existing behaviour"

    with_term = TextSideDistillLoss(DistillSignalWeights(latent_contrast=1.0))
    total, logs_on = with_term(pred, target)
    assert "latent_contrast" in logs_on
    assert float(total) > float(logs_on["latent_contrast"]) - 1e-6
