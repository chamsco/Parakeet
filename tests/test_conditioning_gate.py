"""The conditioning gate must not credit the velocity field with the plan's work.

Round 64's mistake is the reason this module exists: the flow-matching loss *total* includes the supervised
plan head, which is a text -> token regression and therefore text-dependent by construction. The total rose
4.4 % -> 19.3 % -> 37.4 % across checkpoints and read as "the conditioning is growing", while the velocity
field's own share was 2 % and flat. These tests pin the decomposition with models whose behaviour is known by
construction rather than measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from parakeet.eval.gates import text_dependence  # noqa: E402


class _Batch:
    """The batch shape `flow_loss` receives: ids, an optional mask, targets and a voice."""

    def __init__(self, n: int = 4, tokens: int = 5, width: int = 24) -> None:
        generator = torch.Generator().manual_seed(0)
        self.data = {
            "ids": torch.randint(1, 20, (n, 6), generator=generator),
            "text_mask": torch.ones(n, 6),
            "latent": torch.randn(n, width, 16, generator=generator),
            "latent_token": torch.randn(n, tokens, width, generator=generator),
            "voice": torch.zeros(n, dtype=torch.long),
        }

    def __call__(self):
        return dict(self.data)


class _Model(torch.nn.Module):
    """Loss built from the batch, with the text's influence chosen by construction.

    `flow_weight` and `plan_weight` say exactly which part of the objective depends on the text, which is the
    distinction the gate has to report.  A weight of 0 multiplies the text signal by zero, so that term is
    provably text-independent rather than accidentally so.
    """

    def __init__(self, flow_weight: float, plan_weight: float) -> None:
        super().__init__()
        self.flow_weight = flow_weight
        self.plan_weight = plan_weight

    def flow_loss(self, ids, mask, latent, voice=None, latent_token=None):
        target = latent.mean(dim=(1, 2), keepdim=True)          # (n, 1, 1)
        text = ids.float().mean(dim=1).view(-1, 1, 1) / 20.0    # (n, 1, 1), from the text
        flow = torch.nn.functional.mse_loss(self.flow_weight * text, target)
        plan = self.plan_weight * torch.nn.functional.mse_loss(text, target)
        return flow + plan, {"plan": plan}


def test_a_text_blind_objective_reports_no_dependence():
    model = _Model(flow_weight=0.0, plan_weight=0.0)
    result = text_dependence(model, _Batch())
    assert abs(result["flow_share"]) < 1e-6, result
    assert abs(result["total_share"]) < 1e-6, result


def test_a_text_driven_field_is_reported():
    """A field whose loss genuinely depends on the text must show a positive share."""
    model = _Model(flow_weight=1.0, plan_weight=0.0)
    result = text_dependence(model, _Batch())
    assert result["flow_share"] > 0, result
    assert abs(result["plan_share"]) < 1e-6, "there is no plan term here"


def test_a_text_driven_plan_does_not_inflate_the_field(tmp_path=None):
    """The round-64 bug, exactly: total high, flow term zero, because the plan carries the signal."""
    model = _Model(flow_weight=0.0, plan_weight=1.0)
    result = text_dependence(model, _Batch())
    assert abs(result["flow_share"]) < 1e-6, (
        f"the flow term is text-blind but the gate reports {result['flow_share']:+.2f}%: it is crediting the "
        f"field with the plan's work"
    )
    assert result["difference_flow"] == 0.0


def test_the_plan_term_is_reported_separately():
    model = _Model(flow_weight=0.0, plan_weight=1.0)
    result = text_dependence(model, _Batch())
    assert "difference_plan" in result and "difference_flow" in result
    # the flow term is blind, so its difference is zero while the plan's is not necessarily
    assert result["difference_flow"] == 0.0
