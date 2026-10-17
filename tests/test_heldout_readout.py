"""Held-out readouts must load the trained voice table and map voices by name, not by cache-local index."""

import json

import pytest
import torch
import torch.nn as nn

from parakeet.train.common import cache_voice_names, load_state_checked, voice_index_remap


def test_remap_follows_names_not_indices():
    train = ["af_alloy", "alicia", "af_heart", "af_bella"]
    held_out = ["af_heart", "af_bella"]
    assert voice_index_remap(held_out, train) == {0: 2, 1: 3}


def test_remap_refuses_an_untrained_voice():
    with pytest.raises(ValueError, match="af_new"):
        voice_index_remap(["af_heart", "af_new"], ["af_heart"])


def test_shape_mismatch_raises_instead_of_leaving_random_weights():
    trained = nn.Embedding(21, 8)
    narrower = nn.Embedding(11, 8)
    with pytest.raises(ValueError, match="weight"):
        load_state_checked(narrower, trained.state_dict())


def test_matching_state_loads_and_reports_unknown_keys():
    src, dst = nn.Linear(4, 4), nn.Linear(4, 4)
    state = {**src.state_dict(), "extra.weight": torch.zeros(1)}
    assert load_state_checked(dst, state) == ["extra.weight"]
    assert torch.equal(dst.weight, src.weight)


def test_cache_voice_names(tmp_path):
    (tmp_path / "cache_meta.json").write_text(json.dumps({"voice_names": ["a", "b"]}), encoding="utf-8")
    assert cache_voice_names(tmp_path) == ["a", "b"]
    assert cache_voice_names(tmp_path / "missing") == []
