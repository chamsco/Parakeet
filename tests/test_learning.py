"""Learning tests: the stages must not only run, they must fit, and the freeze pattern must be
stage-local (a previous stage's frozen modules must not silently disable the next stage).
"""

import copy

import pytest
import torch

from parakeet.config import ParakeetConfig
from parakeet.data.features import (
    TokenTargetBatchSource,
    fit_latent_normalizer,
    token_targets_from_corpus,
)
from parakeet.data.synthetic import (
    LAYOUTS,
    SyntheticSpeechBatchSource,
    corpus_hours,
    make_corpus,
    make_utterance,
    render_token,
)
from parakeet.data.text import TextTokenizer
from parakeet.eval import (
    ae_reconstruction_l1,
    duration_error_frames,
    end_to_end_mel_l1,
    latent_normalizer_summary,
    teacher_signal_loss,
)
from parakeet.models import build_model
from parakeet.train.losses import build_loss_bundle
from parakeet.train.stages import autoencoder_step, run_stage


def _cfg(fast_cfg, **overrides) -> ParakeetConfig:
    cfg = copy.deepcopy(fast_cfg)
    cfg.train.save_every = 0
    cfg.train.log_every = 100
    for key, value in overrides.items():
        setattr(cfg.train, key, value)
    return cfg


def _corpus(cfg, n: int = 4, seed: int = 0):
    return make_corpus(n, cfg.audio, seed=seed)


# ------------------------------------------------------------------ synthetic fixture
def test_synthetic_utterance_is_structurally_exact():
    cfg = ParakeetConfig()
    utt = make_utterance(cfg.audio, layout="short", f0=150.0, seed=0)
    assert utt.n_tokens == len(LAYOUTS["short"])
    assert utt.token_frames == list(LAYOUTS["short"])
    assert utt.wav.numel() == utt.n_frames * cfg.audio.hop_length
    assert abs(float(utt.wav.abs().max()) - 0.3) < 1e-4
    assert utt.token_f0[0] > utt.token_f0[-1], "F0 should decline across the utterance"
    assert all(torch.isfinite(torch.tensor(utt.token_energy_db)))
    assert len(utt.text) == utt.n_tokens


def test_synthetic_corpus_covers_both_layouts():
    cfg = ParakeetConfig()
    corpus = make_corpus(8, cfg.audio, seed=3)
    assert {u.layout for u in corpus} == set(LAYOUTS)
    assert corpus_hours(corpus, cfg.audio.sample_rate) > 0
    # every utterance length matches its layout exactly (that is what makes batching exact)
    for utt in corpus:
        assert utt.wav.numel() == sum(LAYOUTS[utt.layout]) * cfg.audio.hop_length


def test_render_token_is_finite_and_bandlimited():
    tok = render_token(200.0, 700.0, 1200.0, 4096, 24000, generator=torch.Generator().manual_seed(0))
    assert tok.shape == (4096,)
    assert torch.isfinite(tok).all()
    assert tok.abs().max() > 0
    spec = torch.fft.rfft(tok * torch.hann_window(4096)).abs()
    freqs = torch.linspace(0, 12000, spec.numel())
    centroid = float((spec * freqs).sum() / spec.sum())
    assert centroid < 5000.0, f"harmonic stack should sit low, got centroid {centroid:.0f} Hz"


# ------------------------------------------------------------------ targets + normaliser
def test_latent_normalizer_fits_and_inverts(fast_cfg):
    cfg = _cfg(fast_cfg)
    model = build_model(cfg)
    corpus = _corpus(cfg, 4, seed=1)
    source = SyntheticSpeechBatchSource(corpus, batch_size=2, seed=0)
    fit_latent_normalizer(model.latent_norm, model.autoencoder, source, cfg, max_batches=3)
    summary = latent_normalizer_summary(model)
    assert summary["updates"] == 3
    assert summary["mean_sigma"] > 0

    x = torch.randn(1, cfg.autoencoder.latent_dim, 5) * 0.3 + 0.5
    assert torch.allclose(model.latent_norm.denormalize(model.latent_norm.normalize(x)), x, atol=1e-4)


def test_token_targets_are_exact_and_normalised(fast_cfg):
    cfg = _cfg(fast_cfg)
    model = build_model(cfg)
    corpus = _corpus(cfg, 4, seed=2)
    fit_latent_normalizer(
        model.latent_norm, model.autoencoder, SyntheticSpeechBatchSource(corpus, 2, seed=0), cfg, max_batches=2
    )
    targets = token_targets_from_corpus(model, corpus, cfg, TextTokenizer(mode=cfg.text.mode))
    assert len(targets) == len(corpus)
    for t in targets:
        n_tok = len(t["durations"])
        assert t["ids"].numel() == n_tok, "one text token per duration (character-level)"
        assert t["latent_token"].shape == (n_tok, cfg.autoencoder.latent_dim)
        assert t["f0"].numel() == n_tok and t["energy"].numel() == n_tok
        assert t["wav"].shape[-1] == int(t["durations"].sum()) * cfg.audio.hop_length
    # normalised targets should be roughly unit scale
    scale = float(targets[0]["latent_token"].std())
    assert 0.2 < scale < 5.0


def test_token_target_batch_source_aligns_decoder_latent(fast_cfg):
    cfg = _cfg(fast_cfg)
    model = build_model(cfg)
    corpus = _corpus(cfg, 4, seed=4)
    targets = token_targets_from_corpus(model, corpus, cfg, TextTokenizer(mode=cfg.text.mode))

    text_batch = TokenTargetBatchSource(targets, "distill-text", batch_size=2, seed=0)()
    assert {"ids", "text_mask", "durations", "f0", "energy", "latent_token"} <= set(text_batch)
    assert text_batch["ids"].shape[0] == 2
    assert text_batch["text_mask"].dtype == torch.bool and text_batch["text_mask"].all()

    dec_batch = TokenTargetBatchSource(targets, "distill-decoder", batch_size=2, seed=0, model=model)()
    needed = dec_batch["wav"].shape[-1] // cfg.audio.hop_length + 2
    assert dec_batch["latent"].shape == (2, cfg.autoencoder.latent_dim, needed)

    with pytest.raises(ValueError):
        TokenTargetBatchSource(targets, "distill-decoder", batch_size=1)()


def test_autoencoder_step_accepts_precomputed_latent(fast_cfg):
    cfg = _cfg(fast_cfg)
    model = build_model(cfg)
    wav = 0.1 * torch.randn(1, 4096)
    frames = 4096 // cfg.audio.hop_length + 2
    latent = torch.randn(1, cfg.autoencoder.latent_dim, frames)
    losses = build_loss_bundle(cfg)
    loss, logs, fake = autoencoder_step(cfg, model, wav, losses, latent=latent)
    assert torch.isfinite(loss)
    loss.backward()
    # nothing may have been encoded, so the encoder path must be gradient-free
    assert all(p.grad is None for p in model.autoencoder.stem.parameters())
    assert all(p.grad is None for p in model.autoencoder.encoder.parameters())
    assert fake.shape == wav.shape


# ------------------------------------------------------------------ freeze pattern / learning
def test_stage_freezing_does_not_leak_between_stages(fast_cfg, tmp_path):
    """Regression: ``distill-text`` freezes the autoencoder; ``distill-decoder`` must unfreeze the
    decoder again, otherwise the decoder stage silently trains nothing."""
    cfg = _cfg(fast_cfg)
    cfg.train.lr = 1e-3
    model = build_model(cfg)
    corpus = _corpus(cfg, 4, seed=5)
    targets = token_targets_from_corpus(model, corpus, cfg, TextTokenizer(mode=cfg.text.mode))

    run_stage(
        "distill-text", cfg, model=model,
        batches=TokenTargetBatchSource(targets, "distill-text", 2, seed=0),
        max_steps=1, out_dir=str(tmp_path),
    )
    assert not any(p.requires_grad for p in model.autoencoder.parameters())

    decoder_before = [p.detach().clone() for p in model.autoencoder.decoder.parameters()]
    run_stage(
        "distill-decoder", cfg, model=model,
        batches=TokenTargetBatchSource(targets, "distill-decoder", 2, seed=0, model=model),
        max_steps=1, out_dir=str(tmp_path),
    )
    assert any(p.requires_grad for p in model.autoencoder.decoder.parameters())
    assert not any(p.requires_grad for p in model.autoencoder.encoder.parameters())
    changed = [
        not torch.equal(a, b)
        for a, b in zip(decoder_before, model.autoencoder.decoder.parameters())
    ]
    assert any(changed), "the decoder stage must actually update decoder weights"


def test_pipeline_learns_on_structured_synthetic_speech(fast_cfg, tmp_path):
    """The headline learning assertion: the representation and the distilled text side both fit."""
    cfg = _cfg(fast_cfg)
    cfg.train.lr = 2e-3
    model = build_model(cfg)
    corpus = _corpus(cfg, 4, seed=6)
    wavs = [u.wav for u in corpus]

    ae_before = ae_reconstruction_l1(model, wavs, cfg)
    run_stage(
        "autoencoder", cfg, model=model,
        batches=SyntheticSpeechBatchSource(corpus, batch_size=2, seed=0),
        max_steps=40, out_dir=str(tmp_path),
    )
    ae_after = ae_reconstruction_l1(model, wavs, cfg)
    assert ae_after < ae_before * 0.95, f"autoencoder did not learn: {ae_before:.4f} -> {ae_after:.4f}"

    fit_latent_normalizer(
        model.latent_norm, model.autoencoder, SyntheticSpeechBatchSource(corpus, 2, seed=0), cfg, max_batches=2
    )
    targets = token_targets_from_corpus(model, corpus, cfg, TextTokenizer(mode=cfg.text.mode))
    text_before = teacher_signal_loss(model, targets, cfg)
    run_stage(
        "distill-text", cfg, model=model,
        batches=TokenTargetBatchSource(targets, "distill-text", batch_size=2, seed=0),
        max_steps=40, out_dir=str(tmp_path),
    )
    text_after = teacher_signal_loss(model, targets, cfg)
    assert text_after < text_before * 0.80, (
        f"text side did not fit the cached teacher signals: {text_before:.4f} -> {text_after:.4f}"
    )

    # probes must be callable end to end (values are not asserted: 40 steps is not a checkpoint)
    assert duration_error_frames(model, targets) > 0
    assert end_to_end_mel_l1(model, targets, cfg) > 0
