"""Losses: reconstruction, adversarial, phase, distillation, multi-teacher mixing."""

import pytest
import torch
import torch.nn.functional as F

from parakeet.audio.mel import MelSpectrogram
from parakeet.config import ParakeetConfig
from parakeet.train.losses import (
    AdversarialVocoderLoss,
    LogMelLoss,
    MultiResolutionSTFTLoss,
    MultiTeacherMixer,
    SpectralAnnealer,
    TextSideDistillLoss,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    phase_linearity_loss,
    phase_lock_loss,
    weighted_mean,
)


def test_mrstft_zero_for_identical_signals(tiny_cfg):
    loss_fn = MultiResolutionSTFTLoss()
    wav = torch.randn(2, 4096)
    loss, parts = loss_fn(wav, wav.clone())
    assert loss.item() < 1e-4
    assert set(parts) == {"sc", "log_mag"}


def test_mrstft_positive_and_differentiable(tiny_cfg):
    loss_fn = MultiResolutionSTFTLoss()
    a = torch.randn(1, 4096)
    b = torch.randn(1, 4096)
    x = a.clone().requires_grad_(True)
    loss, _ = loss_fn(x, b)
    assert loss.item() > 0
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_log_mel_loss(tiny_cfg):
    mel = MelSpectrogram(tiny_cfg.audio)
    loss_fn = LogMelLoss(mel)
    wav = torch.randn(1, 8192)
    assert loss_fn(wav, wav.clone()).item() < 1e-6
    assert loss_fn(wav, torch.randn(1, 8192)).item() > 0


def test_adversarial_vocoder_generator_and_discriminator(tiny_cfg):
    adv = AdversarialVocoderLoss()
    real = 0.1 * torch.randn(2, 4096)
    fake = (0.1 * torch.randn(2, 4096)).requires_grad_(True)
    g_loss, fm = adv(real, fake, mode="generator")
    assert torch.isfinite(g_loss) and torch.isfinite(fm)
    (g_loss + fm).backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()

    d_loss, _ = adv(real, fake.detach(), mode="discriminator")
    assert torch.isfinite(d_loss)
    d_loss.backward()


def test_adversarial_helpers_are_normalised():
    adv = AdversarialVocoderLoss()
    real = 0.1 * torch.randn(1, 2048)
    fake = 0.1 * torch.randn(1, 2048)
    real_outs = adv.discriminate(real)
    fake_outs = adv.discriminate(fake)
    assert torch.isfinite(discriminator_loss(real_outs, fake_outs))
    assert torch.isfinite(generator_adversarial_loss(fake_outs))
    assert torch.isfinite(feature_matching_loss(real_outs, fake_outs))


def test_spectral_annealer_schedule():
    ann = SpectralAnnealer()
    assert ann(0) == 45.0
    assert ann(2999) == 45.0
    assert ann(3000) == 10.0
    assert ann(5000) == 10.0
    assert ann(6000) == 3.0
    assert ann(10**6) == 3.0


def _synthetic_spec(kind: str, n_fft: int = 1024, t: int = 16) -> torch.Tensor:
    freqs = torch.arange(n_fft // 2 + 1, dtype=torch.float32)
    taus = torch.full((t,), 3.0)
    if kind == "locked":
        phase = -2 * torch.pi * freqs[None, :] * taus[:, None] / n_fft
    else:
        g = torch.Generator().manual_seed(0)
        phase = torch.rand(n_fft // 2 + 1, t, generator=g).transpose(0, 1) * 2 * torch.pi
    mag = torch.ones(t, n_fft // 2 + 1) * (0.5 + freqs[None, :] / n_fft)
    return torch.polar(mag, phase).transpose(0, 1)[None]


def test_phase_linearity_prefers_locked_phase():
    locked = _synthetic_spec("locked")
    random_phase = _synthetic_spec("random")
    l_locked = phase_linearity_loss(locked, 24000, 1024)
    l_random = phase_linearity_loss(random_phase, 24000, 1024)
    assert l_locked.item() < l_random.item()
    assert l_locked.item() < 1e-4


def test_phase_lock_loss_on_waveform(tiny_cfg):
    wav = 0.2 * torch.randn(1, 8192)
    loss = phase_lock_loss(wav, tiny_cfg.audio.sample_rate, tiny_cfg.audio.n_fft, tiny_cfg.audio.hop_length)
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 2.0


def test_text_side_distill_loss(fast_cfg):
    from parakeet.models import build_model

    model = build_model(fast_cfg)
    ids = torch.randint(1, 50, (2, 12))
    pred = model.text_side(ids)
    target = {
        "durations": torch.randint(2, 9, (2, 12)),
        "f0": torch.randn(2, 12),
        "energy": torch.randn(2, 12),
        "latent_token": torch.randn(2, 12, fast_cfg.autoencoder.latent_dim),
    }
    criterion = TextSideDistillLoss()
    loss, parts = criterion(pred, target)
    assert torch.isfinite(loss)
    assert set(parts) == {"duration", "f0", "energy", "latent"}
    loss.backward()
    assert any(p.grad is not None for p in model.text.parameters())


def test_multi_teacher_mixer_normalises_weights():
    mixer = MultiTeacherMixer({"orpheus": 0.6, "kokoro": 0.4})
    per_sample = torch.ones(4)
    weights = mixer.weights(["orpheus", "orpheus", "kokoro", "kokoro"])
    # raw shares are preserved, and the loss-level mean is weight-invariant (mean-1 rescaling)
    assert torch.allclose(weights, torch.tensor([0.6, 0.6, 0.4, 0.4]))
    assert torch.allclose(weighted_mean(per_sample, weights), torch.ones(()))

    quality = torch.tensor([1.0, 0.0, 1.0, 1.0])
    out2 = weighted_mean(per_sample, mixer.weights(
        ["orpheus", "orpheus", "kokoro", "kokoro"], quality=quality
    ))
    assert torch.isfinite(out2)

    # a restricted/low-quality teacher is down-weighted, never dropped entirely
    mixer2 = MultiTeacherMixer({"orpheus": 0.1, "minimax": 10.0}, min_weight=0.05)
    w = mixer2.weights(["orpheus", "minimax"], quality=torch.tensor([1.0, 0.0001]))
    assert float(w[0]) == pytest.approx(0.1)
    assert float(w[1]) == pytest.approx(0.05), "the floor keeps it in the mixture"
