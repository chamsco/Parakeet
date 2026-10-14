"""A cache has to record which autoencoder encoded it.

The metadata carried the audio config, the latent width and the normalisation statistics, but not the autoencoder's
identity, so retraining one and reusing a cache -- or merging a new cache into an old one -- would put latents from two
different spaces into one training set with nothing in the metadata to show it. The expansion's merge check compares
pooled scales, which catches a gross mismatch and not a subtle one.

What matters is stability, so these pin the two properties that make the hash usable rather than a slow end-to-end
test: it must not change while a model is being trained, and it must change the moment a weight does.
"""

from __future__ import annotations

import torch

from parakeet.data.features import _autoencoder_fingerprint


def test_the_same_weights_hash_the_same_and_a_changed_weight_does_not():
    model = torch.nn.Linear(4, 3)
    first = _autoencoder_fingerprint(model)
    assert _autoencoder_fingerprint(model) == first, "same weights must hash the same"
    with torch.no_grad():
        model.weight.add_(1.0)
    assert _autoencoder_fingerprint(model) != first, "changed weights must hash differently"


def test_the_hash_ignores_gradient_state():
    """A fingerprint that moved during training would make every checkpoint look like a different autoencoder."""
    model = torch.nn.Linear(4, 3)
    before = _autoencoder_fingerprint(model)
    model(torch.zeros(1, 4)).sum().backward()
    assert model.weight.grad is not None
    assert _autoencoder_fingerprint(model) == before
