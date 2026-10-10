"""Aligned crops: text and acoustics must be cut together, or the conditioning lies.

Training a flow on whole utterances spends each step on one long sequence, while the thing that has to be
learned — which latent belongs to which text — needs *variety* per step. Cropping is the standard answer,
but it is only sound if the text is cropped with the audio: keeping the whole sentence while cutting the
frames teaches the model to predict an arbitrary part of the utterance from all of it.

The cache carries per-token frame counts, so a frame window maps to a token span. These tests pin the
invariant (the cropped tokens cover the cropped frames, within one token), the per-token slicing, and the
degenerate cases.
"""

from __future__ import annotations

import pytest
import torch

from parakeet.data.dataset import crop_item_to_tokens


def _item(tokens: int = 40, frames_per_token: int = 10, latent_dim: int = 24, rate: int = 3):
    return {
        "ids": torch.arange(tokens) % 40 + 1,
        "durations": torch.full((tokens,), frames_per_token, dtype=torch.long),
        "latent": torch.randn(latent_dim, tokens * frames_per_token),
        "latent_token": torch.randn(tokens, latent_dim * rate),
        "f0": torch.rand(tokens),
        "energy": torch.rand(tokens),
    }


def test_cropped_tokens_cover_the_cropped_frames():
    item = _item()
    piece = crop_item_to_tokens(item, start_frame=100, n_frames=150)
    assert piece is not None
    assert piece["latent"].shape[-1] == 150, "the frame crop must be exact"
    covered = int(piece["durations"].sum())
    assert abs(covered - 150) <= int(item["durations"].max()), (
        f"cropped tokens cover {covered} frames for a 150-frame window"
    )
    start, window, first, last = [int(v) for v in piece["crop"]]
    assert (start, window) == (100, 150)
    assert last > first


def test_per_token_tensors_are_sliced_along_the_token_axis():
    """`latent_token` is (T, features): slicing the last axis would cut the features instead."""
    item = _item(tokens=40, latent_dim=24, rate=3)
    piece = crop_item_to_tokens(item, start_frame=0, n_frames=100)
    tokens = piece["ids"].numel()
    assert piece["latent_token"].shape == (tokens, 72), piece["latent_token"].shape
    assert piece["f0"].numel() == tokens
    assert piece["durations"].numel() == tokens


def test_the_crop_is_inside_the_utterance():
    item = _item(tokens=10, frames_per_token=10)
    piece = crop_item_to_tokens(item, start_frame=95, n_frames=50)  # would run past the end
    assert piece is not None
    start, window, _first, _last = [int(v) for v in piece["crop"]]
    assert start + window <= 100, (start, window)
    assert piece["latent"].shape[-1] == window


@pytest.mark.parametrize(
    "item",
    [
        {},                                                          # nothing to crop
        {"ids": torch.ones(4, dtype=torch.long), "latent": torch.randn(24, 100)},  # no durations
        {"ids": torch.ones(1, dtype=torch.long), "durations": torch.tensor([10]),
         "latent": torch.randn(24, 10)},                             # one token
    ],
)
def test_uncroppable_items_are_refused_rather_than_producing_a_wrong_pair(item):
    assert crop_item_to_tokens(item, 0, 50) is None
