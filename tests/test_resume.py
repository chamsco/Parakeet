"""Checkpoint and resume: does resuming continue a run, or restart it with warm weights?

Round 13 found that ``train.py --resume`` restored **only the model**.  The optimizer moments, the
EMA (the Reflow teacher and the better final weights), the discriminator, the LR schedule position,
the RNG and the batch order were all discarded -- so a resumed run silently restarted the cosine
schedule from its warmup peak and reshuffled its data.  On a 50k-step GPU run that is expensive to
discover and impossible to notice from the loss alone.

The central test here is the only one that really proves it: an uninterrupted run and an
interrupted-then-resumed run must produce **identical** parameters and losses.
"""

import copy
import json

import pytest
import torch

from parakeet.config import ParakeetConfig
from parakeet.data.dataset import SyntheticBatchSource
from parakeet.models import build_model
from parakeet.train.common import EMAModel, build_optimizer, load_checkpoint, save_checkpoint
from parakeet.train.stages import run_stage


def _cfg(tmp_path=None) -> ParakeetConfig:
    cfg = ParakeetConfig(variant="tiny")
    cfg.audio.n_mels = 40
    cfg.speaker.n_mels = 40
    cfg.text.dim = 48
    cfg.text.n_layers = 2
    cfg.text.n_heads = 4
    cfg.autoencoder.encoder_dims = [24, 32]
    cfg.autoencoder.encoder_blocks = [1, 1]
    cfg.autoencoder.decoder_dim = 32
    cfg.autoencoder.decoder_blocks = 2
    cfg.duration.hidden = 32
    cfg.flow.text_dim = cfg.text.dim
    cfg.flow.cond_dim = cfg.text.dim
    cfg.train.warmup_steps = 4
    cfg.train.save_every = 0
    cfg.train.log_every = 1000
    cfg.train.batch_size = 2
    return cfg.validate()


def _train(cfg, out_dir, steps, resume_from=None, save_every=0):
    cfg = copy.deepcopy(cfg)
    cfg.train.save_every = save_every
    source = SyntheticBatchSource(
        cfg, "distill-text", batch_size=cfg.train.batch_size, n_frames=24, n_tokens=8, seed=123
    )
    model = build_model(cfg)
    final = run_stage(
        "distill-text",
        cfg,
        model=model,
        batches=source,
        max_steps=steps,
        out_dir=str(out_dir),
        resume_from=resume_from,
    )
    return model, source, final


def _params(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ------------------------------------------------------------------ component round-trip
def test_checkpoint_round_trips_every_training_component(fast_cfg, tmp_path):
    cfg = _cfg()
    model = build_model(cfg)
    opt = build_optimizer(model, cfg.train.lr)
    ema = EMAModel(model, 0.9)
    # make the state non-trivial so a missing component is detectable
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    ema.update(model)

    path = save_checkpoint(tmp_path / "ckpt.pt", model, opt, 7, cfg, ema=ema)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["step"] == 7
    assert {"model", "optimizer", "ema", "config", "rng"} <= set(payload)

    fresh_model = build_model(cfg)
    fresh_opt = build_optimizer(fresh_model, cfg.train.lr)
    fresh_ema = EMAModel(fresh_model, 0.9)
    load_checkpoint(path, fresh_model, optimizer=fresh_opt, ema=fresh_ema)

    for key, value in model.state_dict().items():
        assert torch.allclose(fresh_model.state_dict()[key], value), key
    # Adam moments: a non-zero exp_avg proves the optimizer state survived
    opt_state = fresh_opt.state_dict()["state"]
    assert any(float(s["exp_avg"].abs().sum()) > 0 for s in opt_state.values()), (
        "optimizer moments must be restored, not reinitialised"
    )
    ema_state = fresh_ema.state_dict()
    original_state = ema.state_dict()
    assert all(
        torch.allclose(ema_state["shadow"][k], original_state["shadow"][k])
        for k in original_state["shadow"]
    )


def test_checkpoint_records_the_rng_state(fast_cfg, tmp_path):
    """A checkpoint carries the RNG stream, so a resumed run draws the same numbers.

    The model is built *before* seeding: parameter initialisation consumes the global RNG, so
    building it inside the measured window would compare two different streams (which is how the
    first version of this test managed to fail against a correct implementation).
    """
    cfg = _cfg()
    model = build_model(cfg)
    torch.manual_seed(7)
    path = save_checkpoint(tmp_path / "ckpt.pt", model, None, 1, cfg)

    torch.manual_seed(999)  # unrelated state
    load_checkpoint(path, model)  # restores the saved stream
    after_resume = torch.randn(3)

    torch.manual_seed(7)  # the stream the checkpoint was saved from
    expected = torch.randn(3)
    assert torch.allclose(after_resume, expected), "the saved RNG stream must be restored"


def test_load_can_skip_rng_restoration(fast_cfg, tmp_path):
    cfg = _cfg()
    model = build_model(cfg)
    torch.manual_seed(7)
    path = save_checkpoint(tmp_path / "ckpt.pt", model, None, 1, cfg)

    torch.manual_seed(0)
    first = torch.randn(3)  # the caller's stream
    torch.manual_seed(0)
    payload = load_checkpoint(path, model, restore_rng=False)
    assert payload["rng_restored"] is False
    assert torch.allclose(torch.randn(3), first), "an opt-out load must not disturb the RNG stream"


# ------------------------------------------------------------------ the real claim
def test_resumed_run_is_identical_to_an_uninterrupted_one(fast_cfg, tmp_path):
    """The only test that proves resume works: same trajectory, not merely a warm start."""
    torch.manual_seed(0)
    cfg = _cfg()
    uninterrupted_model, _, uninterrupted_final = _train(cfg, tmp_path / "full", steps=10)

    torch.manual_seed(0)
    torch.manual_seed(0)
    first_model, first_source, _ = _train(cfg, tmp_path / "part1", steps=5, save_every=5)
    assert (tmp_path / "part1" / "distill-text_step5.pt").exists()

    # a *fresh process equivalent*: new model, new optimizer, new source, new EMA
    resumed_model, _, resumed_final = _train(
        cfg, tmp_path / "part2", steps=10,
        resume_from=str(tmp_path / "part1" / "distill-text_step5.pt"),
    )

    a, b = _params(uninterrupted_model), _params(resumed_model)
    assert set(a) == set(b)
    worst = max(float((a[k] - b[k]).abs().max()) for k in a)
    assert worst == 0.0, (
        f"resumed run diverged from the uninterrupted one (max parameter difference {worst:.3e}); "
        "check optimizer/EMA/RNG/batch-order restoration and the LR schedule position"
    )
    assert uninterrupted_final["step"] == resumed_final["step"] == 10


def test_resume_continues_the_lr_schedule(fast_cfg, tmp_path):
    """A resumed run must not restart the cosine schedule at the warmup peak."""
    cfg = _cfg()
    cfg.train.warmup_steps = 2
    cfg.train.save_every = 0
    cfg.train.log_every = 1
    seen = {}

    def record(logs):
        if "step" in logs and "lr" in logs:
            seen[int(logs["step"])] = float(logs["lr"])

    source = SyntheticBatchSource(cfg, "distill-text", batch_size=2, n_frames=24, n_tokens=8, seed=1)
    model = build_model(cfg)
    run_stage("distill-text", cfg, model=model, batches=source, max_steps=4, out_dir=str(tmp_path / "a"),
              log_fn=record)
    uninterrupted = dict(seen)

    seen.clear()
    source = SyntheticBatchSource(cfg, "distill-text", batch_size=2, n_frames=24, n_tokens=8, seed=1)
    run_stage("distill-text", cfg, model=build_model(cfg), batches=source, max_steps=2,
              out_dir=str(tmp_path / "b"), log_fn=record)
    source = SyntheticBatchSource(cfg, "distill-text", batch_size=2, n_frames=24, n_tokens=8, seed=1)
    run_stage("distill-text", cfg, model=build_model(cfg), batches=source, max_steps=4,
              out_dir=str(tmp_path / "c"), log_fn=record,
              resume_from=str(tmp_path / "b" / "distill-text_last.pt"))

    # steps 1-2 of the resumed run must repeat the LR of the uninterrupted run at those steps
    for step in (3, 4):
        assert seen[step] == pytest.approx(uninterrupted[step], rel=1e-9), (
            f"step {step}: resumed lr {seen[step]:.4e} != uninterrupted {uninterrupted[step]:.4e}"
        )


def test_resume_reports_the_step_it_restarted_from(fast_cfg, tmp_path):
    cfg = _cfg()
    source = SyntheticBatchSource(cfg, "distill-text", batch_size=2, n_frames=24, n_tokens=8, seed=1)
    model = build_model(cfg)
    save_checkpoint(tmp_path / "start.pt", model, None, 3, cfg)
    logs = run_stage("distill-text", cfg, model=model, batches=source, max_steps=5,
                     out_dir=str(tmp_path / "out"), resume_from=str(tmp_path / "start.pt"))
    assert logs["resumed_from"] == 3.0
    assert logs["step"] == 5


def test_resume_of_a_missing_file_raises(fast_cfg, tmp_path):
    cfg = _cfg()
    source = SyntheticBatchSource(cfg, "distill-text", batch_size=2, n_frames=24, n_tokens=8, seed=1)
    with pytest.raises(FileNotFoundError):
        run_stage("distill-text", cfg, model=build_model(cfg), batches=source, max_steps=1,
                  out_dir=str(tmp_path), resume_from=str(tmp_path / "nope.pt"))


def test_saved_checkpoint_carries_batch_order_state(fast_cfg, tmp_path):
    """Without the batch-order state a resumed run reshuffles and diverges."""
    cfg = _cfg()
    _, source, _ = _train(cfg, tmp_path / "s", steps=3, save_every=3)
    payload = torch.load(
        tmp_path / "s" / "distill-text_step3.pt", map_location="cpu", weights_only=False
    )
    state = payload["extra"]["batch_state"]
    assert state and state["generator"] is not None
    assert payload["extra"]["stage"] == "distill-text"
    # and it must be loadable back into a fresh source of the same shape
    fresh = SyntheticBatchSource(cfg, "distill-text", batch_size=2, n_frames=24, n_tokens=8, seed=999)
    fresh.load_state_dict(state)
    assert torch.equal(fresh()["ids"], source()["ids"]), "the same batch must follow"
