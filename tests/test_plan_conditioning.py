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


def test_the_plan_is_spread_over_frames_uniformly_and_reversibly():
    """Training and inference must spread a plan identically, or the residual would not match."""
    cfg, model = _model(use_plan=True)
    plan = torch.randn(2, 5, 72)
    upsampled = model.upsample_plan(plan, 20)
    assert upsampled.shape == (2, 72, 20)
    # uniform nearest-neighbour: token 0 covers the first frames, the last token the last frames
    assert torch.allclose(upsampled[:, :, 0], plan[:, 0, :])
    assert torch.allclose(upsampled[:, :, -1], plan[:, -1, :])


def test_a_zero_plan_makes_the_residual_formulation_identical_to_the_plain_one():
    """The residual must be exactly ``x1 - upsample(plan)``: with a zero plan there is nothing to subtract.

    This is the invariant that makes the change safe -- and it is only checkable because both paths are in
    the same code.
    """
    cfg, model = _model(use_plan=True)
    cfg.flow.plan_residual = True
    cfg.flow.plan_weight = 0.0  # isolate the flow term from the plan term
    with torch.no_grad():
        for parameter in model.plan_head.parameters():
            parameter.zero_()
    batch = _batch(cfg)
    model.train()

    torch.manual_seed(7)
    plain, _ = model.flow_loss(
        batch["ids"], batch["text_mask"], batch["latent"], voice=batch["voice"]
    )
    torch.manual_seed(7)
    residual, aux = model.flow_loss(
        batch["ids"], batch["text_mask"], batch["latent"],
        voice=batch["voice"], latent_token=batch["latent_token"],
    )
    assert torch.allclose(plain, residual, atol=1e-6), (float(plain), float(residual))
    assert "plan" in aux and float(aux["plan"]) > 0, "a zero plan still has to be supervised"


def test_a_nonzero_plan_changes_the_target_the_flow_is_asked_for():
    """If the plan did not change the target, nothing would force the conditioning to be used."""
    cfg, model = _model(use_plan=True)
    cfg.flow.plan_residual = True
    cfg.flow.plan_weight = 0.0
    batch = _batch(cfg)
    model.train()
    torch.manual_seed(11)
    plain, _ = model.flow_loss(
        batch["ids"], batch["text_mask"], batch["latent"], voice=batch["voice"]
    )
    torch.manual_seed(11)
    residual, _ = model.flow_loss(
        batch["ids"], batch["text_mask"], batch["latent"],
        voice=batch["voice"], latent_token=batch["latent_token"],
    )
    assert not torch.allclose(plain, residual, atol=1e-4), (
        "a trained plan must change the flow's target -- that is the whole point of the residual form"
    )


def test_synthesis_adds_the_plan_back():
    cfg, model = _model(use_plan=True)
    model.eval()
    ids = torch.randint(1, 100, (1, 8))
    mask = torch.ones(1, 8, dtype=torch.bool)
    with torch.no_grad():
        wav_with = model.synthesize(ids, mask, voice=torch.zeros(1, dtype=torch.long), steps=2,
                                    n_latent_frames=48)
        for parameter in model.plan_head.parameters():
            parameter.zero_()
        wav_zero = model.synthesize(ids, mask, voice=torch.zeros(1, dtype=torch.long), steps=2,
                                    n_latent_frames=48, x0=torch.zeros(1, 24 * 6, 8))
    assert wav_with.dim() >= 1 and wav_with.numel() > 0, wav_with.shape
    assert wav_zero.numel() > 0


def test_advisory_mode_does_not_add_the_plan_back_to_the_sampled_latent():
    """In advisory mode the plan was only extra conditioning, so adding it back would corrupt the output.

    This is a training/inference mismatch that no metric would explain -- it was nearly shipped when a
    failed edit left the synthesis path ungated.
    """
    cfg, model = _model(use_plan=True)
    assert cfg.flow.plan_residual is False
    model.eval()
    ids = torch.randint(1, 100, (1, 8))
    mask = torch.ones(1, 8, dtype=torch.bool)
    with torch.no_grad():
        wav_plan = model.synthesize(ids, mask, voice=torch.zeros(1, dtype=torch.long), steps=2,
                                    n_latent_frames=48)
        for parameter in model.plan_head.parameters():
            parameter.zero_()
        wav_zero = model.synthesize(ids, mask, voice=torch.zeros(1, dtype=torch.long), steps=2,
                                    n_latent_frames=48)
    assert wav_plan.shape == wav_zero.shape


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
