"""Config system: defaults, YAML round-trip, validation, rejection of typos."""

import pytest

from parakeet.config import (
    ParakeetConfig,
    dataclass_diff,
    describe,
    from_dict,
    load_config,
    save_config,
    to_dict,
)


def test_defaults_are_valid():
    cfg = ParakeetConfig().validate()
    assert cfg.variant == "tiny"
    assert cfg.audio.hop_length == 256
    assert cfg.flow.compress == 6
    assert abs(cfg.audio.frame_rate - 93.75) < 1e-6


def test_yaml_roundtrip(tmp_path):
    cfg = ParakeetConfig(variant="small", voice_mode="reference")
    cfg.flow.dim = 512
    cfg.name = "roundtrip"
    path = save_config(cfg, tmp_path / "c.yaml")
    again = load_config(path)
    assert dataclass_diff(cfg, again) == []
    assert to_dict(cfg)["flow"]["dim"] == 512


def test_shipped_configs_load(tiny_yaml, small_yaml):
    tiny = load_config(tiny_yaml)
    small = load_config(small_yaml)
    assert tiny.variant == "tiny"
    assert small.variant == "small"
    assert small.flow.context_expansion == 4
    assert small.train.loss.spectral == 3.0, "Paradee's low spectral weight must be the default"


def test_unknown_key_is_rejected():
    with pytest.raises(KeyError):
        from_dict({"variant": "tiny", "nonsense": 1})


def test_validation_catches_mismatch():
    cfg = ParakeetConfig()
    cfg.flow.latent_dim = 32
    with pytest.raises(ValueError, match="latent_dim"):
        cfg.validate()

    cfg = ParakeetConfig()
    cfg.flow.cond_dim = 128
    with pytest.raises(ValueError, match="cond_dim"):
        cfg.validate()

    with pytest.raises(ValueError):
        ParakeetConfig(variant="huge").validate()


def test_describe_is_informative():
    text = describe(ParakeetConfig())
    assert "93.8Hz" in text and "nfe=32->4" in text


def test_optional_fmax_none():
    cfg = from_dict({"audio": {"fmax": None}})
    assert cfg.audio.fmax is None
