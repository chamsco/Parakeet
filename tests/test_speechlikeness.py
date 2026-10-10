"""Speech-likeness: the acceptance criterion the mel proxies could not supply.

Round 35 measured both paths at WER 1.000 while their envelope proxies looked healthy -- and then
measured why: the Tiny regression path emits a near-silent buzz (98 % "voiced" at 491 Hz, the tracker's
ceiling, at 0.012 amplitude against the teacher's 0.49) and the flow emits unvoiced noise (7 % voiced,
spectral flatness 0.60 against the teacher's 0.30).  Neither is speech, and the mel proxy rewarded both.

The metric must separate three cases a generator can produce: real voiced speech, white noise (flat
spectrum, no pitch), and a pure tone (a pitch, but pinned at one frequency with no formant structure).
"""

from __future__ import annotations

import torch

from parakeet.eval.metrics import speechlikeness


def _voiced_tone(seconds: float = 2.0, f0: float = 130.0, rate: int = 24000) -> torch.Tensor:
    """A harmonic-rich tone with a slowly varying pitch: about as speech-like as a synthetic signal
    gets without being speech."""
    t = torch.arange(int(rate * seconds)) / rate
    wobble = 1.0 + 0.05 * torch.sin(2 * torch.pi * 3.0 * t)
    wave = torch.zeros_like(t)
    for harmonic in range(1, 9):
        wave = wave + torch.sin(2 * torch.pi * f0 * harmonic * t * wobble) / harmonic
    return wave * 0.2


def test_the_gate_accepts_measured_good_audio_and_rejects_the_two_failures():
    """Thresholds calibrated on measurements, not intuition.

    The anchor that forced a correction: the autoencoder round trip of a *real* latent transcribes at
    WER 0.000 with flatness 0.51 -- above a first version's 0.5 cutoff, so the gate rejected good
    audio.  The failing cases are the opposite extremes: a buzz is ~96 % voiced, noise is <10 %.
    """
    rate = 24000
    t = torch.arange(rate * 2) / rate
    buzz = torch.sin(2 * torch.pi * 495 * t) * 0.05
    assert speechlikeness([buzz], rate)["speech_like"] is False

    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(rate * 2, generator=generator) * 0.2
    assert speechlikeness([noise], rate)["speech_like"] is False


def test_the_real_autoencoder_round_trip_passes_the_gate():
    """The end-to-end anchor: a real cached latent, denormalised with the cache's statistics and decoded
    through the trained autoencoder, is intelligible (WER 0.000, measured in round 36).  If this fails,
    either the gate is miscalibrated or the latent convention regressed -- both worth failing for."""
    from pathlib import Path

    import pytest

    from parakeet.config import load_config
    from parakeet.data.dataset import LatentShardDataset
    from parakeet.eval.metrics import speechlikeness as gate
    from parakeet.models import build_model
    from parakeet.train.common import derive_n_voices_from_cache, load_latent_norm_from_cache

    cache = Path("runs/mixed_v2/latent_cache")
    autoencoder = Path("runs/ae_scaled/adversarial/autoencoder_last.pt")
    if not cache.exists() or not autoencoder.exists():
        pytest.skip("the trained autoencoder / cache are not present in this workspace")

    cfg = load_config("configs/parakeet_tiny.yaml")
    cfg.n_voices = derive_n_voices_from_cache(cache)
    payload = torch.load(autoencoder, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    model = build_model(cfg)
    current = model.state_dict()
    model.load_state_dict(
        {k: v for k, v in state.items() if k in current and tuple(current[k].shape) == tuple(v.shape)},
        strict=False,
    )
    assert load_latent_norm_from_cache(model, cache) is True, "the cache must carry its statistics"
    model.eval()
    latent = model.latent_norm.denormalize(LatentShardDataset(cache)[0]["latent"][None])
    with torch.no_grad():
        audio = model.autoencoder.decode(latent, length=latent.shape[-1] * cfg.audio.hop_length)
    measured = gate([audio.reshape(-1)], cfg.audio.sample_rate)
    assert measured["speech_like"] is True, measured


def test_a_voiced_tone_is_speech_like_and_noise_is_not():
    generator = torch.Generator().manual_seed(0)
    tone = speechlikeness([_voiced_tone()], 24000)
    noise = speechlikeness([torch.randn(48000, generator=generator) * 0.2], 24000)
    assert tone["available"] and noise["available"]
    assert tone["voiced_fraction"] > 0.55, tone  # the 5 % pitch wobble costs a few frames
    assert tone["speech_like"] is True, tone
    assert noise["voiced_fraction"] < 0.3, noise
    assert noise["speech_like"] is False, noise
    assert noise["spectral_flatness"] > tone["spectral_flatness"], (noise, tone)


def test_a_pure_buzz_at_the_tracker_ceiling_is_rejected():
    """The Tiny path's failure mode: 98 % voiced at 491 Hz is not a human pitch, it is a tone."""
    rate = 24000
    t = torch.arange(rate * 2) / rate
    buzz = torch.sin(2 * torch.pi * 495 * t) * 0.05
    measured = speechlikeness([buzz], rate)
    assert measured["voiced_fraction"] > 0.9, measured
    assert measured["speech_like"] is False, (
        "a tone pinned at the tracker's ceiling must not count as speech"
    )


def test_silence_and_very_short_audio_are_reported_as_unavailable():
    assert speechlikeness([], 24000)["available"] is False
    assert speechlikeness([torch.zeros(512)], 24000)["available"] is False


def test_the_reference_is_measured_for_comparison():
    measured = speechlikeness([_voiced_tone()], 24000, reference=[_voiced_tone(f0=110.0)])
    assert "reference" in measured and measured["reference"]["available"] is True
