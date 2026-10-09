"""Streaming (blockwise) sampling, chunked phase lock, and streaming synthesis.

The interesting property to pin down is *why* blockwise sampling works at all: the vector field
mixes time with finite-support convolutions, so a block only needs a bounded window.  These tests
check the receptive-field helper empirically, that a single block is exactly the one-shot sampler,
that multi-block output stays within the model's own sampling variability, and that the streaming
audio path lines up with the offline one.
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from parakeet.config import ParakeetConfig
from parakeet.inference import StreamingPhaseLock, Synthesizer, phase_lock
from parakeet.inference.synthesize import StreamingVocoder
from parakeet.models import build_model
from parakeet.models.flow import (
    blockwise_sample,
    consistency_sample,
    iter_blockwise_sample,
    vf_context_frames,
)


def _fast_small() -> ParakeetConfig:
    cfg = ParakeetConfig(variant="small", voice_mode="reference")
    cfg.audio.n_fft = 512
    cfg.audio.hop_length = 128
    cfg.audio.win_length = 512
    cfg.autoencoder.encoder_dims = [16, 24, 32]
    cfg.autoencoder.encoder_blocks = [1, 1, 1]
    cfg.autoencoder.decoder_dim = 32
    cfg.autoencoder.decoder_blocks = 2
    cfg.autoencoder.decoder_dilations = [1, 2]
    cfg.text.dim = 32
    cfg.text.n_layers = 1
    cfg.text.n_heads = 4
    cfg.flow.dim = 32
    cfg.flow.depth = 3
    cfg.flow.n_heads = 4
    cfg.flow.text_dim = 32
    cfg.flow.cond_dim = 32
    cfg.flow.nfe = 2
    cfg.flow.distilled_nfe = 2
    cfg.speaker.style_dim = 32
    cfg.speaker.emb_dim = 32
    cfg.speaker.channels = [16]
    cfg.speaker.n_query = 2
    cfg.duration.hidden = 32
    return cfg.validate()


def _context(cfg: ParakeetConfig, model, batch: int = 1, tokens: int = 8, tc: int = 60):
    ids = torch.randint(1, cfg.text.vocab_size, (batch, tokens))
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    ref = torch.randn(batch, cfg.audio.n_mels, 40)
    with torch.no_grad():
        memory, memory_mask, _ = model.conditions(ids, mask, ref)
    shape = (batch, cfg.flow.latent_dim * cfg.flow.compress, tc)
    return memory, memory_mask, shape


# ------------------------------------------------------------------ receptive field
def test_vf_context_frames_matches_measured_response():
    """Empirically: perturb one frame and see how far the response reaches."""
    cfg = _fast_small()
    model = build_model(cfg).eval()
    # open the layer scales, otherwise the temporal branch is disabled at init (1e-6) and the
    # response is pointwise -- which is itself worth knowing, see the docstring
    for blk in model.vf.blocks:
        blk.conv.gamma.data.fill_(1.0)
    rf = vf_context_frames(model.vf)
    assert rf == 3 * cfg.flow.depth, "each kernel-7 block adds (7-1)//2 = 3 frames per side"

    memory, memory_mask, shape = _context(cfg, model, tc=40)
    x = torch.randn(shape)
    t = torch.full((shape[0],), 0.5)
    centre = 20
    with torch.no_grad():
        base = model.vf(x, t, memory, memory_mask)
        perturbed = x.clone()
        perturbed[:, :, centre] += 1.0
        delta = (model.vf(perturbed, t, memory, memory_mask) - base).abs().amax(dim=1)[0]
    inside = delta[centre - rf : centre + rf + 1].max()
    outside = torch.cat([delta[: centre - rf], delta[centre + rf + 1 :]]).max()
    assert float(inside) > 1e-3
    assert float(outside) < 1e-6, "response must not reach beyond the reported context"


def test_layer_scale_init_disables_temporal_mixing():
    """A freshly built vector field is effectively per-frame; that is why streaming is exact early."""
    cfg = _fast_small()
    model = build_model(cfg).eval()
    memory, memory_mask, shape = _context(cfg, model, tc=16)
    x = torch.randn(shape)
    t = torch.full((shape[0],), 0.5)
    with torch.no_grad():
        base = model.vf(x, t, memory, memory_mask)
        perturbed = x.clone()
        perturbed[:, :, 0] += 1.0
        delta = (model.vf(perturbed, t, memory, memory_mask) - base).abs().amax(dim=1)[0]
    assert float(delta[1:].max()) < 1e-4, "at init the branch is scaled by 1e-6"
    assert float(delta[0]) > 1e-3


# ------------------------------------------------------------------ blockwise sampler
def test_single_block_equals_one_shot():
    cfg = _fast_small()
    model = build_model(cfg).eval()
    memory, memory_mask, shape = _context(cfg, model, tc=12)
    x0 = torch.randn(shape)
    with torch.no_grad():
        one_shot = consistency_sample(model.vf, memory, memory_mask, shape, steps=2, x0=x0)
        blocked = blockwise_sample(
            model.vf, memory, memory_mask, shape, steps=2, block_frames=shape[-1], x0=x0
        )
    assert torch.allclose(one_shot, blocked, atol=1e-6)


def test_blockwise_within_sampling_variability():
    cfg = _fast_small()
    model = build_model(cfg).eval()
    memory, memory_mask, shape = _context(cfg, model, tc=64)
    x0 = torch.randn(shape)
    x0_other = torch.randn(shape)
    with torch.no_grad():
        full = consistency_sample(model.vf, memory, memory_mask, shape, steps=2, x0=x0)
        independent = consistency_sample(model.vf, memory, memory_mask, shape, steps=2, x0=x0_other)
        blocked = blockwise_sample(
            model.vf, memory, memory_mask, shape, steps=2, block_frames=16, x0=x0
        )
    cos = lambda a, b: float(F.cosine_similarity(a.reshape(-1)[None], b.reshape(-1)[None]).item())
    assert blocked.shape == full.shape
    assert torch.isfinite(blocked).all()
    assert cos(blocked, full) > cos(independent, full), (
        "blockwise must be closer to the one-shot result than an independent draw"
    )


def test_iter_blockwise_covers_all_frames_in_order():
    cfg = _fast_small()
    model = build_model(cfg).eval()
    memory, memory_mask, shape = _context(cfg, model, tc=50)
    x0 = torch.randn(shape)
    blocks, cursor = [], 0
    with torch.no_grad():
        for start, end, block in iter_blockwise_sample(
            model.vf, memory, memory_mask, shape, steps=2, block_frames=16, x0=x0
        ):
            assert start == cursor
            assert end > start
            assert block.shape[-1] == end - start
            blocks.append(block)
            cursor = end
        assembled = torch.cat(blocks, dim=-1)
        reference = blockwise_sample(
            model.vf, memory, memory_mask, shape, steps=2, block_frames=16, x0=x0
        )
    assert cursor == shape[-1]
    assert torch.allclose(assembled, reference, atol=1e-6)


def test_blockwise_rejects_bad_context_mode():
    cfg = _fast_small()
    model = build_model(cfg).eval()
    memory, memory_mask, shape = _context(cfg, model, tc=32)
    with pytest.raises(ValueError):
        next(
            iter_blockwise_sample(
                model.vf, memory, memory_mask, shape, steps=1, block_frames=16,
                context_mode="nonsense",
            )
        )


def test_blockwise_leaves_training_flag_alone():
    cfg = _fast_small()
    model = build_model(cfg)
    model.train()
    memory, memory_mask, shape = _context(cfg, model, tc=32)
    blockwise_sample(model.vf, memory, memory_mask, shape, steps=1, block_frames=16)
    assert model.training


# ------------------------------------------------------------------ streaming audio
def test_streaming_phase_lock_matches_offline_interior():
    cfg = _fast_small()
    g = torch.Generator().manual_seed(0)
    wav = 0.2 * torch.randn(1, 40000, generator=g)
    offline = phase_lock(
        wav, sample_rate=cfg.audio.sample_rate, n_fft=cfg.audio.n_fft,
        hop_length=cfg.audio.hop_length, win_length=cfg.audio.win_length,
    )
    streamer = StreamingPhaseLock(
        sample_rate=cfg.audio.sample_rate, n_fft=cfg.audio.n_fft,
        hop_length=cfg.audio.hop_length, win_length=cfg.audio.win_length,
    )
    pieces = []
    step = 4096
    for i in range(0, wav.shape[-1], step):
        out = streamer.push(wav[..., i : i + step])
        if out.shape[-1]:
            pieces.append(out)
    pieces.append(streamer.flush())
    chunked = torch.cat(pieces, dim=-1)
    n = min(offline.shape[-1], chunked.shape[-1])
    cos = float(
        F.cosine_similarity(offline[..., :n].reshape(-1)[None], chunked[..., :n].reshape(-1)[None]).item()
    )
    assert abs(chunked.shape[-1] - wav.shape[-1]) <= cfg.audio.n_fft
    assert cos > 0.99, f"chunked phase lock diverged from offline (cosine {cos:.4f})"


def test_streaming_phase_lock_holds_back_overlap():
    cfg = _fast_small()
    streamer = StreamingPhaseLock(
        sample_rate=cfg.audio.sample_rate, n_fft=cfg.audio.n_fft, hop_length=cfg.audio.hop_length
    )
    short = torch.randn(1, cfg.audio.n_fft // 2)
    assert streamer.push(short).shape[-1] == 0, "must buffer at least n_fft samples"
    tail = streamer.flush()
    assert tail.shape[-1] > 0


def test_synthesize_stream_yields_audio_chunks():
    cfg = _fast_small()
    model = build_model(cfg).eval()
    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=False)
    text = "hello streaming parakeet"
    # blocks are counted in *compressed* frames (T / Kc), so ask for several blocks' worth
    n_latent = 16 * cfg.flow.compress * 3
    with torch.no_grad():
        whole = synth.synthesize(text, steps=2, n_latent_frames=n_latent, seed=0)
    chunks = list(synth.synthesize_stream(text, chunk_frames=16, steps=2, n_latent_frames=n_latent))
    assert len(chunks) >= 3, "three blocks' worth of latent must produce several chunks"
    total = sum(c.shape[0] for c in chunks)
    assert abs(total - whole.shape[-1]) <= cfg.audio.hop_length * cfg.flow.compress
    assert all(torch.isfinite(torch.from_numpy(c)).all() for c in chunks)


def test_synthesize_stream_with_phase_lock_runs():
    cfg = _fast_small()
    model = build_model(cfg).eval()
    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=True)
    chunks = list(synth.synthesize_stream("locked streaming", chunk_frames=16, steps=2, n_latent_frames=48))
    assert chunks, "streaming with the chunked phase lock must still yield audio"
    joined = torch.cat([torch.from_numpy(c) for c in chunks])
    assert joined.numel() > 0
    assert torch.isfinite(joined).all()
