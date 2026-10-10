"""Coarse-to-fine plan conditioning: the plan must exist, be supervised, and reach the memory.

Rounds 40-47 measured the flow fitting the marginal velocity field while its text conditioning stayed weak,
so the papers' split was implemented: predict a token-level acoustic plan from the text and put it in the
conditioning memory.  The failure mode to guard against is the one already measured once -- a conditioning
path that exists but is ignored -- which is why the plan is supervised directly against the cached token
latents rather than left for the flow loss to discover.
"""

from __future__ import annotations

import torch

from parakeet.config import load_config
from parakeet.models import build_model


def _model(use_plan: bool, n_voices: int = 3):
    cfg = load_config("configs/parakeet_flow.yaml")
    cfg.n_voices = n_voices
    cfg.flow.use_plan = use_plan
    torch.manual_seed(0)
    return cfg, build_model(cfg)


def _batch(cfg, batch: int = 2, tokens: int = 12, frames: int = 48):
    ids = torch.randint(1, 100, (batch, tokens))
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    width = cfg.autoencoder.latent_dim * max(1, int(cfg.autoencoder.latent_rate))
    return {
        "ids": ids,
        "text_mask": mask,
        "latent": torch.randn(batch, cfg.autoencoder.latent_dim, frames),
        "latent_token": torch.randn(batch, tokens, width),
        "voice": torch.zeros(batch, dtype=torch.long),
    }


def test_the_plan_is_off_by_default_so_existing_runs_are_unchanged():
    """The long full-corpus run is training on the default path; enabling the plan must be opt-in."""
    cfg, model = _model(use_plan=False)
    assert model.use_plan is False
    assert not any("plan" in name for name, _ in model.named_parameters())
    batch = _batch(cfg)
    memory, memory_mask, _cond = model.conditions(batch["ids"], batch["text_mask"], voice=batch["voice"])
    assert memory.shape[1] == memory_mask.shape[1]
    _, planned = _model(use_plan=True)
    planned_memory, _planned_mask, _ = planned.conditions(
        batch["ids"], batch["text_mask"], voice=batch["voice"]
    )
    assert memory.shape[1] < planned_memory.shape[1], "off-by-default must not add plan tokens"


def test_the_plan_widens_the_memory_by_one_token_per_text_token():
    cfg, with_plan = _model(use_plan=True)
    _, without = _model(use_plan=False)
    batch = _batch(cfg)
    plain, plain_mask, _ = without.conditions(batch["ids"], batch["text_mask"], voice=batch["voice"])
    planned, planned_mask, _ = with_plan.conditions(batch["ids"], batch["text_mask"], voice=batch["voice"])
    tokens = batch["ids"].shape[1]
    assert planned.shape[1] == plain.shape[1] + tokens
    assert planned_mask.shape[1] == planned.shape[1]
    assert bool(planned_mask.all()), "plan tokens must be attended to, not masked out"


def test_the_plan_is_supervised_against_the_cached_token_latents():
    """Without this the plan can be ignored, which is the failure measured in round 39."""
    cfg, model = _model(use_plan=True)
    batch = _batch(cfg)
    model.train()
    loss, aux = model.flow_loss(
        batch["ids"], batch["text_mask"], batch["latent"],
        voice=batch["voice"], latent_token=batch["latent_token"],
    )
    assert "plan" in aux, "the plan term must be reported so its progress is visible"
    assert torch.isfinite(loss) and torch.isfinite(aux["plan"])
    loss.backward()
    grads = [p.grad for p in model.plan_head.parameters() if p.grad is not None]
    assert grads, "no gradient reached the plan head"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_the_plan_improves_when_it_is_trained_on_a_fixed_batch():
    cfg, model = _model(use_plan=True)
    batch = _batch(cfg)
    model.train()
    optimiser = torch.optim.AdamW(model.plan_head.parameters(), lr=5e-3)
    first = None
    for _ in range(40):
        optimiser.zero_grad()
        loss, aux = model.flow_loss(
            batch["ids"], batch["text_mask"], batch["latent"],
            voice=batch["voice"], latent_token=batch["latent_token"],
        )
        (cfg.flow.plan_weight * (loss - loss.detach()) + loss).backward()
        optimiser.step()
        if first is None:
            first = float(aux["plan"])
    assert float(aux["plan"]) < first * 0.7, (first, float(aux["plan"]))
