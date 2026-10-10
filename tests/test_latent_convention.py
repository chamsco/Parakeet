"""The latent-space convention that silently corrupted every run.

The cache stores **normalised** latents: `real_train_demo.py` fits a `LatentNormalizer` on the corpus and
the cache builder applies `normalize` before writing each item.  Every decode path is supposed to undo
that, and the code looked right -- `decoder_latent_from_tokens` calls `denormalize`, `ParakeetFlow.
synthesize` calls it too.  What was missing is the *statistics*: the fitted normaliser lived for one
process, so a model loading a checkpoint later got `mean 0 / var 1` (the identity), or stale values from
an older fit.

Measured on one utterance's cached latent (round 36):

    encoder(teacher audio) -> decode   WER 0.000
    cached latent          -> decode   WER 1.000
    cached * scale + shift -> decode   WER 0.000     (residual of the affine fit: 0.0000)

so the decoder was receiving a per-dimension shifted and rescaled latent in every run.  These tests pin
the invariant, the persistence, and the loader.
"""

from __future__ import annotations

import json

import torch

from parakeet.models.autoencoder import LatentNormalizer
from parakeet.train.common import load_latent_norm_from_cache


def test_denormalize_undoes_normalize_and_the_identity_does_not():
    """The invariant the whole bug violated: a *fitted* normaliser round-trips; a fresh one does not."""
    torch.manual_seed(0)
    latent = torch.randn(2, 24, 50) * 0.8

    fresh = LatentNormalizer(24)
    fitted = LatentNormalizer(24)
    for _ in range(4):
        fitted.update(latent)
    assert float(fitted.n) > 0

    round_trip = fitted.denormalize(fitted.normalize(latent))
    assert torch.allclose(round_trip, latent, atol=1e-4), "a fitted normaliser must round-trip"

    identity_round_trip = fresh.denormalize(fitted.normalize(latent))
    assert not torch.allclose(identity_round_trip, latent, atol=1e-3), (
        "an unfitted normaliser must NOT undo a fitted transform -- that is exactly the bug: the "
        "decoder received the normalised latent"
    )


def test_statistics_are_persisted_in_the_cache_and_loaded_back(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    stats = {"mean": [0.5] * 24, "var": [0.25] * 24, "samples": 4}
    (cache / "cache_meta.json").write_text(json.dumps({"latent_norm": stats}), encoding="utf-8")

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_norm = LatentNormalizer(24)

    holder = Holder()
    assert load_latent_norm_from_cache(holder, cache) is True
    assert torch.allclose(holder.latent_norm.mean, torch.full((24,), 0.5), atol=1e-6)
    assert torch.allclose(holder.latent_norm.var, torch.full((24,), 0.25), atol=1e-6)
    assert float(holder.latent_norm.n) == 4.0


def test_a_cache_without_statistics_reports_that_rather_than_pretending(tmp_path):
    cache = tmp_path / "old_cache"
    cache.mkdir()
    (cache / "cache_meta.json").write_text(json.dumps({"voice_names": ["a"]}), encoding="utf-8")

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_norm = LatentNormalizer(24)

    holder = Holder()
    assert load_latent_norm_from_cache(holder, cache) is False
    assert torch.allclose(holder.latent_norm.mean, torch.zeros(24)), "the identity must be left alone"
    assert load_latent_norm_from_cache(holder, None) is False
    assert load_latent_norm_from_cache(holder, tmp_path / "missing") is False


def test_a_dimension_mismatch_is_refused_rather_than_copied():
    """A cache from a different autoencoder must not silently overwrite the normaliser."""
    assert LatentNormalizer(24).mean.numel() == 24
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory)
        (cache / "cache_meta.json").write_text(
            json.dumps({"latent_norm": {"mean": [0.0] * 8, "var": [1.0] * 8, "samples": 1}}),
            encoding="utf-8",
        )

        class Holder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.latent_norm = LatentNormalizer(24)

        holder = Holder()
        assert load_latent_norm_from_cache(holder, cache) is False
