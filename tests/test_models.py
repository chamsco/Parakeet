"""Model assembly: parameter budgets, shapes, flow-matching mechanics."""

import pytest
import torch

from parakeet.config import ParakeetConfig, load_config
from parakeet.models import (
    ParakeetFlow,
    ParakeetTiny,
    SpeechAutoencoder,
    build_model,
    count_parameters,
    parameter_report,
)
from parakeet.models.flow import (
    consistency_sample,
    euler_sample,
    fold_time,
    make_xt,
    unfold_time,
)


def test_fold_unfold_roundtrip():
    x = torch.randn(2, 24, 18)
    folded = fold_time(x, 6)
    assert folded.shape == (2, 144, 3)
    back = unfold_time(folded, 6, t_out=18)
    assert back.shape == (2, 24, 18)
    assert torch.allclose(back, x)


def test_fold_drops_remainder():
    x = torch.randn(1, 4, 7)
    assert fold_time(x, 3).shape == (1, 12, 2)


def test_rectified_flow_interpolation_endpoints():
    x1 = torch.ones(1, 3, 4)
    x0 = torch.zeros(1, 3, 4)
    assert torch.allclose(make_xt(x1, x0, torch.zeros(1)), x0)
    assert torch.allclose(make_xt(x1, x0, torch.ones(1)), x1)
    v = x1 - x0
    mid = make_xt(x1, x0, torch.full((1,), 0.5))
    assert torch.allclose(v * 0.5, mid)


def test_tiny_parameter_budget(tiny_cfg):
    model = build_model(tiny_cfg)
    n = count_parameters(model)
    assert 8e6 < n < 12e6, f"tiny must stay 8-12M, got {n/1e6:.2f}M"
    assert isinstance(model, ParakeetTiny)


def test_shipped_small_config_parameter_budget(small_yaml):
    cfg = load_config(small_yaml)
    model = build_model(cfg)
    n = count_parameters(model)
    assert 40e6 < n < 52e6, f"small must land near the 44M SupertonicTTS budget, got {n/1e6:.2f}M"
    assert isinstance(model, ParakeetFlow)


def test_tiny_text_side_shapes(tiny_cfg):
    model = build_model(tiny_cfg)
    ids = torch.randint(1, 100, (2, 20))
    out = model.text_side(ids)
    assert out["log_duration"].shape == (2, 20)
    assert out["latent_token"].shape == (2, 20, tiny_cfg.autoencoder.latent_dim)
    durations = torch.full((2, 20), 5, dtype=torch.long)
    latent, mask = model.latent_from_tokens(out["latent_token"], durations, out["f0"], out["energy"])
    assert latent.shape == (2, tiny_cfg.autoencoder.latent_dim, 100)
    assert mask.shape == (2, 100)
    wav = model.autoencoder.decode(latent)
    assert wav.shape[-1] > 0 and torch.isfinite(wav).all()


def test_autoencoder_length_bookkeeping(tiny_cfg):
    ae = build_model(tiny_cfg).autoencoder
    n_samples = 24000
    frames = ae.latent_length(n_samples)
    assert frames == n_samples // tiny_cfg.audio.hop_length + 1
    assert ae.istft.output_length(frames) == frames * tiny_cfg.audio.hop_length


def test_flow_loss_backward(small_cfg):
    cfg = ParakeetConfig(variant="small", voice_mode="reference")
    cfg.flow.depth = 2
    cfg.flow.dim = 64
    cfg.flow.n_heads = 4
    cfg.flow.text_dim = 64
    cfg.flow.cond_dim = 64
    cfg.text.dim = 64
    cfg.text.n_layers = 2
    cfg.autoencoder.encoder_dims = [32, 48, 64]
    cfg.autoencoder.decoder_dim = 64
    cfg.autoencoder.decoder_blocks = 3
    cfg.speaker.style_dim = 64
    cfg.speaker.emb_dim = 64
    cfg.speaker.channels = [32, 48]
    cfg.speaker.n_query = 4
    cfg.duration.hidden = 64
    cfg.validate()
    model = build_model(cfg)
    ids = torch.randint(1, 100, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.bool)
    latent = torch.randn(2, cfg.autoencoder.latent_dim, 36)
    ref_mel = torch.randn(2, cfg.audio.n_mels, 80)
    loss, aux = model.flow_loss(ids, mask, latent, ref_mel=ref_mel)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.vf.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert aux["tc"].item() == 6.0


def test_context_sharing_expands_batch(small_cfg):
    cfg = ParakeetConfig(variant="small", voice_mode="reference")
    cfg.flow.depth = 1
    cfg.flow.dim = 64
    cfg.flow.n_heads = 4
    cfg.flow.text_dim = 64
    cfg.flow.cond_dim = 64
    cfg.text.dim = 64
    cfg.text.n_layers = 1
    cfg.text.n_heads = 4
    cfg.autoencoder.encoder_dims = [16, 32]
    cfg.autoencoder.decoder_dim = 32
    cfg.autoencoder.decoder_blocks = 2
    cfg.speaker.style_dim = 64
    cfg.speaker.emb_dim = 32
    cfg.speaker.channels = [16]
    cfg.speaker.n_query = 2
    cfg.duration.hidden = 32
    cfg.flow.context_expansion = 3
    cfg.validate()
    model = build_model(cfg)
    seen = {}

    def hook(module, inputs, output):
        seen["batch"] = inputs[0].shape[0]

    handle = model.vf.in_proj.register_forward_hook(hook)
    ids = torch.randint(1, 50, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.bool)
    latent = torch.randn(2, cfg.autoencoder.latent_dim, 24)
    try:
        loss, _ = model.flow_loss(ids, mask, latent)
    finally:
        handle.remove()
    assert torch.isfinite(loss)
    assert seen["batch"] == 6, "Ke=3 context sharing must triple the effective batch"


def test_euler_and_consistency_samplers(small_cfg):
    cfg = ParakeetConfig(variant="small")
    cfg.flow.depth = 1
    cfg.flow.dim = 64
    cfg.flow.n_heads = 4
    cfg.flow.text_dim = 64
    cfg.flow.cond_dim = 64
    cfg.text.dim = 64
    cfg.text.n_layers = 1
    cfg.autoencoder.encoder_dims = [16, 32]
    cfg.autoencoder.decoder_dim = 32
    cfg.autoencoder.decoder_blocks = 2
    cfg.speaker.style_dim = 64
    cfg.speaker.emb_dim = 32
    cfg.speaker.channels = [16]
    cfg.speaker.n_query = 2
    cfg.validate()
    model = build_model(cfg)
    memory = torch.randn(1, 10, cfg.flow.cond_dim)
    shape = (1, cfg.flow.latent_dim * cfg.flow.compress, 4)
    x0 = torch.randn(shape)
    a = euler_sample(model.vf, memory, None, shape, steps=4, device="cpu", x0=x0)
    b = consistency_sample(model.vf, memory, None, shape, steps=2, device="cpu")
    assert a.shape == shape and b.shape == shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not torch.allclose(a, b)


def test_training_flag_preserved_after_sampling(small_cfg):
    model = build_model(small_cfg)
    model.train()
    shape = (1, small_cfg.flow.latent_dim * small_cfg.flow.compress, 2)
    euler_sample(model.vf, None, None, shape, steps=1, device="cpu")
    assert model.training, "euler_sample must restore train mode"


def test_parameter_report_sums(tiny_cfg):
    model = build_model(tiny_cfg)
    rep = parameter_report(model)
    assert rep["TOTAL"] == count_parameters(model)
    assert sum(v for k, v in rep.items() if k != "TOTAL") <= rep["TOTAL"]
