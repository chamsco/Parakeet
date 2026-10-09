import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import ParakeetConfig, load_config  # noqa: E402

torch.manual_seed(0)


@pytest.fixture(scope="session")
def tiny_cfg() -> ParakeetConfig:
    cfg = ParakeetConfig(variant="tiny")
    return cfg.validate()


@pytest.fixture(scope="session")
def small_cfg() -> ParakeetConfig:
    cfg = ParakeetConfig(variant="small", voice_mode="reference")
    return cfg.validate()


@pytest.fixture(scope="session")
def fast_cfg() -> ParakeetConfig:
    """Small dimensions for fast, shape-correct tests."""
    cfg = ParakeetConfig(variant="tiny")
    cfg.autoencoder.encoder_dims = [32, 48, 64]
    cfg.autoencoder.decoder_dim = 64
    cfg.autoencoder.decoder_blocks = 3
    cfg.text.dim = 64
    cfg.text.n_layers = 2
    cfg.text.n_heads = 4
    cfg.flow.dim = 64
    cfg.flow.depth = 2
    cfg.flow.n_heads = 4
    cfg.flow.text_dim = 64
    cfg.flow.cond_dim = 64
    cfg.speaker.style_dim = 64
    cfg.speaker.emb_dim = 64
    cfg.speaker.channels = [32, 48]
    cfg.speaker.n_query = 4
    cfg.duration.hidden = 64
    cfg.train.log_every = 1
    cfg.train.save_every = 0
    return cfg.validate()


@pytest.fixture(scope="session")
def tiny_yaml() -> Path:
    return ROOT / "configs" / "parakeet_tiny.yaml"


@pytest.fixture(scope="session")
def small_yaml() -> Path:
    return ROOT / "configs" / "parakeet_small.yaml"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT
