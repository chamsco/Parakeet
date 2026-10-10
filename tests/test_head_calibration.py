"""Two calibrations that decide whether a run can succeed at all, and one that must not regress.

Round 33 measured the same arithmetic bug twice: AdamW moves a parameter by roughly the learning rate
per step regardless of gradient size, so **the distance from the initialisation to the target sets a
floor on the steps needed**:

* the flow's utterance-length head started 0.35 against a 6.75 target in log frames (~32 000 steps to
  travel) — fixed by initialising its bias;
* the text side's latent head started with output std 0.25-0.31 against a target of ~1.03 (3.6x, about
  18 000 steps) — fixed by calibrating that head to the data before training.

The same run also hit a silent shape mismatch (a 24-wide head against 72-wide cached targets) that
surfaced deep inside the loss as "tensor a (24) must match tensor b (72)".
"""

from __future__ import annotations

import json

import pytest
import torch

from parakeet.config import load_config
from parakeet.models import build_model
from parakeet.train.common import calibrate_head_scale, derive_latent_rate_from_cache


def test_latent_rate_is_read_from_the_cache_data(tmp_path):
    """The cache decides the sub-latent width; the config must not be able to disagree silently."""
    cache = tmp_path / "cache"
    cache.mkdir()
    torch.save(
        [{"latent_token": torch.zeros(7, 72), "latent": torch.zeros(24, 20)}], cache / "shard_000.pt"
    )
    (cache / "cache_meta.json").write_text(json.dumps({"voice_names": ["a"]}), encoding="utf-8")
    assert derive_latent_rate_from_cache(cache, latent_dim=24) == 3

    torch.save([{"latent_token": torch.zeros(7, 24)}], cache / "shard_only.pt")
    (cache / "shard_000.pt").unlink()
    assert derive_latent_rate_from_cache(cache, latent_dim=24) == 1
    assert derive_latent_rate_from_cache(tmp_path / "missing", latent_dim=24) is None


def test_head_calibration_matches_the_target_scale():
    """A regression head that starts 3.6x too small cannot reach the target in a short run."""
    cfg = load_config("configs/parakeet_tiny.yaml")
    cfg.n_voices = 2
    torch.manual_seed(0)
    model = build_model(cfg)
    batch = {
        "ids": torch.randint(1, 40, (2, 12)),
        "text_mask": torch.ones(2, 12, dtype=torch.bool),
        "voice": torch.zeros(2, dtype=torch.long),
        "latent_token": torch.randn(2, 12, cfg.autoencoder.latent_dim * cfg.autoencoder.latent_rate)
        * 1.03,
    }
    with torch.no_grad():
        before = float(model.text_side(batch["ids"], batch["text_mask"], batch["voice"])["latent_token"].std())
    measured = calibrate_head_scale(model, batch)
    with torch.no_grad():
        after = float(model.text_side(batch["ids"], batch["text_mask"], batch["voice"])["latent_token"].std())
    assert measured["ratio"] > 1.0, "this model is expected to start too small"
    assert abs(after - float(batch["latent_token"].std())) < 0.2, (before, after)
    assert abs(measured["predicted_std_after"] - after) < 1e-4


def test_calibration_is_a_no_op_for_a_model_without_the_head():
    cfg = load_config("configs/parakeet_tiny.yaml")
    model = build_model(cfg)
    model.latent_head = None  # type: ignore[assignment]
    assert calibrate_head_scale(model, {"ids": torch.ones(1, 4, dtype=torch.long)}) == {}
