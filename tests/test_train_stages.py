"""Staged training: every stage must run, produce finite losses and checkpoint."""

import copy

import pytest
import torch

from parakeet.config import ParakeetConfig, load_config
from parakeet.data.dataset import SyntheticBatchSource
from parakeet.models import build_model
from parakeet.train.stages import STAGE_STEPS, run_stage, train_all_stages


def _fast_small() -> ParakeetConfig:
    cfg = ParakeetConfig(variant="small", voice_mode="reference")
    cfg.autoencoder.encoder_dims = [32, 48, 64]
    cfg.autoencoder.encoder_blocks = [1, 1, 1]
    cfg.autoencoder.decoder_dim = 64
    cfg.autoencoder.decoder_blocks = 3
    cfg.autoencoder.decoder_dilations = [1, 2]
    cfg.text.dim = 64
    cfg.text.n_layers = 2
    cfg.text.n_heads = 4
    cfg.flow.dim = 64
    cfg.flow.depth = 2
    cfg.flow.n_heads = 4
    cfg.flow.text_dim = 64
    cfg.flow.cond_dim = 64
    cfg.flow.nfe = 2
    cfg.flow.distilled_nfe = 2
    cfg.speaker.style_dim = 64
    cfg.speaker.emb_dim = 64
    cfg.speaker.channels = [32, 48]
    cfg.speaker.n_query = 4
    cfg.duration.hidden = 64
    cfg.train.log_every = 1
    cfg.train.save_every = 0
    return cfg.validate()


TINY_STAGES = ["autoencoder", "distill-text", "distill-decoder"]
SMALL_STAGES = ["autoencoder", "flow", "reflow"]


@pytest.mark.parametrize("stage", TINY_STAGES)
def test_tiny_stage_runs(fast_cfg, stage, tmp_path):
    cfg = copy.deepcopy(fast_cfg)
    cfg.train.max_steps = 1
    model = build_model(cfg)
    source = SyntheticBatchSource(cfg, stage, batch_size=2, n_frames=32, n_tokens=12)
    logs = run_stage(stage, cfg, model=model, batches=source, max_steps=1, out_dir=str(tmp_path))
    assert logs["loss"] == logs["loss"], f"{stage} loss is NaN"
    assert (tmp_path / f"{stage}_last.pt").exists()


def test_decoder_stage_consumes_the_token_expanded_distribution(fast_cfg, tmp_path):
    """With the flag on, the decoder's input comes from `decoder_latent_from_tokens` -- the same call
    synthesis makes -- which is what puts `prosody_proj` in the graph and lets it learn.  The log key
    records which distribution was used, so a silent revert is visible."""
    cfg = copy.deepcopy(fast_cfg)
    cfg.train.max_steps = 1
    model = build_model(cfg)
    source = SyntheticBatchSource(cfg, "distill-decoder", batch_size=2, n_frames=16, n_tokens=6)

    cfg.autoencoder.decoder_uses_token_latents = True
    logs_on = run_stage("distill-decoder", cfg, model=model, batches=source, max_steps=1,
                        out_dir=str(tmp_path / "on"))
    assert logs_on.get("decoder_input") == 1.0, "the token-expanded path must be the one used"

    cfg.autoencoder.decoder_uses_token_latents = False
    logs_off = run_stage("distill-decoder", cfg, model=model, batches=source, max_steps=1,
                         out_dir=str(tmp_path / "off"))
    assert "decoder_input" not in logs_off, "with the flag off the cached frame latent is used"


def test_distill_audio_trains_the_text_side_through_the_decoder(fast_cfg, tmp_path):
    """The whole point of the stage: audio-space gradients must reach the text side.

    Round 22 measured that the latent-space L1 lets the text side fit its objective while rendering
    unintelligible audio, so this stage decodes the predicted tokens and compares waveforms.  The
    contract is precise: the text side and the prosody projection learn, and the autoencoder -- the
    frozen renderer -- does not.
    """
    cfg = copy.deepcopy(fast_cfg)
    cfg.train.max_steps = 3
    cfg.train.log_every = 1
    model = build_model(cfg)
    source = SyntheticBatchSource(cfg, "distill-audio", batch_size=2, n_frames=16, n_tokens=6)
    # `text_side` is a method, so the text side is probed through its heads
    probes = {
        "latent_head": [p.detach().clone() for p in model.latent_head.parameters()],
        "prosody_proj": [p.detach().clone() for p in model.prosody_proj.parameters()],
        "autoencoder": [p.detach().clone() for p in model.autoencoder.parameters()],
    }
    logs = run_stage("distill-audio", cfg, model=model, batches=source, max_steps=3,
                     out_dir=str(tmp_path))
    assert logs["loss"] == logs["loss"], "the loss must be finite"
    assert logs["audio_mel"] == logs["audio_mel"], "the audio objective must be in the logs"
    assert "aux" in logs, "the cached-signal auxiliary objective must be in the logs"

    def moved(module) -> bool:
        return any(
            not torch.equal(a, b) for a, b in zip(probes[module], getattr(model, module).parameters())
        )

    assert moved("latent_head"), "the text side must receive gradient through the decoder"
    assert moved("prosody_proj"), "prosody_proj runs inside the trained path and must learn"
    assert not moved("autoencoder"), "the autoencoder is the frozen renderer"


def test_reconstruction_only_phase_skips_the_discriminator(fast_cfg, tmp_path):
    """A reconstruction-only phase is 24x cheaper per step on this CPU (round 20), so it must be a
    first-class option: with the adversarial weight at zero the discriminator is never stepped and
    no `disc` loss appears in the logs."""
    cfg = copy.deepcopy(fast_cfg)
    cfg.train.loss.adversarial = 0.0
    cfg.train.loss.feature_match = 0.0
    cfg.train.max_steps = 1
    model = build_model(cfg)
    before = {k: v.detach().clone() for k, v in model.autoencoder.state_dict().items()}
    source = SyntheticBatchSource(cfg, "autoencoder", batch_size=2, n_frames=32, n_tokens=12)
    logs = run_stage("autoencoder", cfg, model=model, batches=source, max_steps=1, out_dir=str(tmp_path))
    assert logs["loss"] == logs["loss"]
    assert "disc" not in logs, "the discriminator must not run when its weight is zero"
    assert any(not torch.equal(before[k], v) for k, v in model.autoencoder.state_dict().items()), (
        "the generator must still train"
    )


def test_nonfinite_gradients_are_skipped_even_with_a_finite_loss(fast_cfg, tmp_path, monkeypatch):
    """Round 24: the scaled-corpus run went non-finite at step 1756 and, with only the *loss* checked,
    NaN gradients were written into the parameters so every later loss was NaN (2245 of 4000 steps
    "skipped" while the weights stayed broken).  Clipping a NaN gradient does not help."""
    from parakeet.train import stages as stages_module

    cfg = copy.deepcopy(fast_cfg)
    cfg.train.max_steps = 1
    model = build_model(cfg)
    source = SyntheticBatchSource(cfg, "distill-text", batch_size=2, n_frames=16, n_tokens=6)
def test_nonfinite_gradients_are_skipped_even_with_a_finite_loss(fast_cfg, tmp_path):
    """Round 24: the scaled-corpus run went non-finite at step 1756 and, with only the *loss* checked,
    NaN gradients were written into the parameters, so every later loss was NaN -- 2245 of 4000 steps
    "skipped" while the weights stayed broken.  Clipping a NaN gradient does not help."""
    from parakeet.train import stages as stages_module

    cfg = copy.deepcopy(fast_cfg)
    cfg.train.max_steps = 1
    model = build_model(cfg)
    source = SyntheticBatchSource(cfg, "distill-text", batch_size=2, n_frames=16, n_tokens=6)
    # a hook that replaces a real gradient with NaN: the loss stays finite, the gradient does not
    param = next(model.text.parameters())
    hook = param.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    try:
        logs = stages_module.run_stage(
            "distill-text", cfg, model=model, batches=source, max_steps=2, out_dir=str(tmp_path)
        )
    finally:
        hook.remove()
    assert logs["nonfinite_steps"] >= 1, "the poisoned gradient must be detected"
    assert logs["first_nonfinite_step"] == 1
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters()), (
        "no parameter may be left non-finite"
    )


def test_nonfinite_loss_is_skipped_and_recorded(fast_cfg, tmp_path, monkeypatch):
    """Divergence must be a *finding*, not an all-NaN report at the end.

    A reconstruction-only run on real speech went mel 2.06 -> 0.50 and then to NaN with no visible
    symptom until the final report.  The loop now skips non-finite updates, keeps the schedule moving,
    reports the first offending step, writes `divergence.json`, and leaves the parameters finite.
    """
    import json

    from parakeet.train import stages as stages_module

    cfg = copy.deepcopy(fast_cfg)
    cfg.train.max_steps = 1
    model = build_model(cfg)
    source = SyntheticBatchSource(cfg, "autoencoder", batch_size=2, n_frames=32, n_tokens=12)
    real_step = stages_module.autoencoder_step
    calls = {"n": 0}

    def exploding_step(*args, **kwargs):
        calls["n"] += 1
        total, logs, recon = real_step(*args, **kwargs)
        if calls["n"] == 1:
            return total * float("nan"), logs, recon
        return total, logs, recon

    monkeypatch.setattr(stages_module, "autoencoder_step", exploding_step)
    logs = stages_module.run_stage(
        "autoencoder", cfg, model=model, batches=source, max_steps=3, out_dir=str(tmp_path)
    )
    assert calls["n"] >= 3
    assert logs["nonfinite_steps"] >= 1
    assert logs["first_nonfinite_step"] == 1
    for name, value in model.autoencoder.state_dict().items():
        assert torch.isfinite(value).all(), f"{name} became non-finite despite the guard"
    payload = json.loads((tmp_path / "divergence.json").read_text(encoding="utf-8"))
    assert payload["first_nonfinite_step"] == 1
    assert payload["skipped_updates"] is True


@pytest.mark.parametrize("stage", SMALL_STAGES)
def test_small_stage_runs(stage, tmp_path):
    cfg = _fast_small()
    cfg.train.max_steps = 1
    model = build_model(cfg)
    source = SyntheticBatchSource(cfg, stage, batch_size=2, n_frames=32, n_tokens=12)
    logs = run_stage(stage, cfg, model=model, batches=source, max_steps=1, out_dir=str(tmp_path))
    assert logs["loss"] == logs["loss"], f"{stage} loss is NaN"
    assert (tmp_path / f"{stage}_last.pt").exists()


def test_stage_registry_covers_every_stage():
    assert set(STAGE_STEPS) == {
        "autoencoder",
        "distill-decoder",
        "distill-text",
        "distill-audio",
        "flow",
        "reflow",
    }


def test_unknown_stage_rejected(fast_cfg):
    with pytest.raises(ValueError):
        run_stage("nonsense", fast_cfg, batches=lambda: {})


def test_distill_decoder_freezes_encoder(fast_cfg, tmp_path):
    cfg = copy.deepcopy(fast_cfg)
    model = build_model(cfg)
    source = SyntheticBatchSource(cfg, "distill-decoder", batch_size=1, n_frames=32, n_tokens=12)
    run_stage("distill-decoder", cfg, model=model, batches=source, max_steps=1, out_dir=str(tmp_path))
    assert not any(p.requires_grad for p in model.autoencoder.encoder.parameters())
    assert any(p.requires_grad for p in model.autoencoder.decoder.parameters())


def test_distill_text_freezes_autoencoder(fast_cfg, tmp_path):
    cfg = copy.deepcopy(fast_cfg)
    model = build_model(cfg)
    source = SyntheticBatchSource(cfg, "distill-text", batch_size=1, n_frames=16, n_tokens=8)
    run_stage("distill-text", cfg, model=model, batches=source, max_steps=1, out_dir=str(tmp_path))
    assert not any(p.requires_grad for p in model.autoencoder.parameters())


def test_training_reduces_loss_on_a_tiny_problem(fast_cfg):
    """Sanity: the distilled text side actually fits its cached targets."""
    cfg = copy.deepcopy(fast_cfg)
    cfg.train.lr = 5e-3
    model = build_model(cfg)
    fixed_batch = SyntheticBatchSource(cfg, "distill-text", batch_size=2, n_frames=16, n_tokens=12, seed=7)
    batch = fixed_batch()
    from parakeet.train.stages import tiny_text_step

    with torch.no_grad():
        first, _ = tiny_text_step(cfg, model, batch)
    for _ in range(20):
        run_stage(
            "distill-text", cfg, model=model, batches=lambda: batch, max_steps=1, out_dir=str(cfg.train.out_dir)
        )
    with torch.no_grad():
        last, _ = tiny_text_step(cfg, model, batch)
    assert last.item() < first.item(), f"loss did not decrease: {first.item():.4f} -> {last.item():.4f}"


def test_train_all_stages_curriculum(tmp_path):
    cfg = _fast_small()
    cfg.train.out_dir = str(tmp_path)
    sources = {
        stage: SyntheticBatchSource(cfg, stage, batch_size=1, n_frames=24, n_tokens=8)
        for stage in SMALL_STAGES
    }
    results = train_all_stages(
        cfg, sources, steps_by_stage={s: 1 for s in SMALL_STAGES}, out_dir=str(tmp_path)
    )
    assert set(results) == set(SMALL_STAGES)
    for logs in results.values():
        assert logs["loss"] == logs["loss"]


def test_load_shipped_configs_build():
    for path in ("configs/parakeet_tiny.yaml", "configs/parakeet_small.yaml", "configs/parakeet_small_44k.yaml"):
        cfg = load_config(path)
        assert cfg.variant in {"tiny", "small"}
        assert cfg.flow.compress == 6
        assert cfg.audio.hop_length in {256, 512}
