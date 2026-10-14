"""The sampled-ρ readout must report a spread, and must not lie when the output is degenerate.

`sampled_rho_with_error` is the metric the whole flow verdict rests on, and round 101 shipped it without a test.
Writing one immediately found a bug: the degenerate path (a model whose output is constant, so no dimension has
variance) returned a dict *without* `straddles_zero`, which every caller reads — so a zero-output model would have
raised KeyError instead of reporting "cannot tell".

These tests drive it with a stub model and a stubbed sampler, so the statistics are exercised directly.
"""

from __future__ import annotations
from pathlib import Path

import pytest
import torch


from parakeet.eval import gates
from parakeet.eval.gates import sampled_rho_with_error


class _FlowCfg:
    latent_dim = 24
    compress = 6
    cfg_scale = 1.0


class _Cfg:
    flow = _FlowCfg()


class _StubModel(torch.nn.Module):
    """The minimum surface the readout touches: conditions, text, predict_latent_frames, compressed_frames, vf.

    `vf` is a submodule that is passed to the sampler but never called: the sampler itself is stubbed, so the
    readout's arithmetic is what these tests exercise, not the network.
    """

    def __init__(self) -> None:
        super().__init__()
        self.vf = torch.nn.Identity()

    def conditions(self, ids, mask, voice=None):
        return torch.zeros(1, 1, 4), torch.ones(1, 1), None

    def text(self, ids, mask):
        return torch.zeros(1, 1, 4)

    def predict_latent_frames(self, text, mask, cond):
        return torch.tensor(6.0)

    def compressed_frames(self, n_latent_frames: int) -> int:
        return max(1, n_latent_frames // _FlowCfg.compress)


class _Dataset:
    """Two items with frame latents that have real variance, so ρ is computable."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self.items = [{
            "ids": torch.tensor([[1, 2, 3, 4]]),
            "latent": torch.randn(24, 12, generator=generator),
            "voice": torch.tensor([0]),
        } for _ in range(2)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def _stub_sampler(monkeypatch, produce, unfold=None):
    """Replace the sampler so the 'sampled latent' is whatever the test wants.

    `unfold` defaults to the real `unfold_time`; tests that want the sampled frame latent to be an exact tensor
    pass `unfold=lambda x, c, t_out=None: x` so nothing reshapes it afterwards.
    """
    def fake(vf, memory, memory_mask, shape, steps, device, cfg_scale, x0=None):
        return produce(shape)

    monkeypatch.setattr("parakeet.models.flow.consistency_sample", fake)
    if unfold is not None:
        monkeypatch.setattr("parakeet.models.parakeet.unfold_time", unfold)


def test_degenerate_output_reports_cannot_tell_rather_than_raising(monkeypatch):
    """A model emitting zeros has no variance to correlate: it must report it, not crash."""
    _stub_sampler(monkeypatch, lambda shape: torch.zeros(*shape))
    result = sampled_rho_with_error(_StubModel(), _Cfg(), _Dataset(), items=2, seeds=(0, 1))
    assert result["n_seeds"] == 0
    assert result["straddles_zero"] is True, "a degenerate reading is not evidence of a correlation"
    assert result["mean"] != result["mean"]  # nan


def test_a_seed_varying_sampler_produces_a_spread(monkeypatch):
    """The whole point of the function: a range and a std, not a point."""
    counter = {"n": 0}

    def produce(shape):
        counter["n"] += 1
        generator = torch.Generator().manual_seed(counter["n"])
        return torch.randn(*shape, generator=generator)

    _stub_sampler(monkeypatch, produce)
    result = sampled_rho_with_error(_StubModel(), _Cfg(), _Dataset(), items=2, seeds=(0, 1, 2, 3))
    assert result["n_seeds"] == 4
    assert len(result["per_seed"]) == 4
    assert result["std"] >= 0.0
    assert result["min"] <= result["mean"] <= result["max"]
    assert "straddles_zero" in result


def test_the_verdict_is_not_always_straddling(monkeypatch):
    """A sampler that returns the target itself must report a clear positive, or the metric is useless."""
    dataset = _Dataset()
    target = dataset[0]["latent"]  # (24, 12)

    def produce(shape):
        # the sampled frame latent IS the target, so every dimension correlates perfectly
        return target.unsqueeze(0)

    _stub_sampler(monkeypatch, produce, unfold=lambda x, c, t_out=None: x)
    # one probe item, because the sampler returns item 0's latent for whatever item it is asked about: with two
    # items the mean would be ~0.5 (perfect on one, uncorrelated on the other), which is the function behaving
    # correctly rather than a bug.
    result = sampled_rho_with_error(_StubModel(), _Cfg(), dataset, items=1, seeds=(0,))
    assert result["n_seeds"] == 1
    assert result["mean"] == result["mean"], "a perfect match must not be nan"
    assert result["mean"] > 0.9, f"a perfect match must read positive, got {result['mean']}"
    assert result["straddles_zero"] is False


def test_the_early_return_keys_match_the_normal_return(monkeypatch):
    """Both paths must expose the same keys -- the bug this test was written to catch."""
    _stub_sampler(monkeypatch, lambda shape: torch.zeros(*shape))
    degenerate = sampled_rho_with_error(_StubModel(), _Cfg(), _Dataset(), items=2, seeds=(0,))

    def produce(shape):
        generator = torch.Generator().manual_seed(7)
        return torch.randn(*shape, generator=generator)

    _stub_sampler(monkeypatch, produce)
    normal = sampled_rho_with_error(_StubModel(), _Cfg(), _Dataset(), items=2, seeds=(0, 1))
    assert set(degenerate) == set(normal), (sorted(degenerate), sorted(normal))
