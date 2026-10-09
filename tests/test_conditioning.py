"""Speaker/style conditioning from a latent cache, with cross-sample pairing.

Round 9 found the third instance of the same class of bug, this time in the *flagship* path:
`collate` dropped `log_mel` entirely, so the Small/flow model trained from a cache received
`ref_mel=None` -- a zero speaker embedding and no style tokens -- and the cross-sample paired
training that PilotTTS's identity/style decoupling depends on was never implemented.  Every demo
that exercised conditioning built its own `ref_mel` by hand, so nothing noticed.

These tests pin the path (cache batch -> pairing -> style tokens -> loss -> gradients) and use a
control: with the pre-fix behaviour (no reference) the conditioning parameters get *no* gradient.
"""

import copy
import math

import pytest
import torch

from parakeet.data.dataset import LatentShardBatchSource, LatentShardDataset, collate
from parakeet.data.features import build_latent_cache
from parakeet.data.text import TextTokenizer
from parakeet.models import build_model
from parakeet.models.speaker import SpeakerConditioner
from parakeet.train.stages import flow_step

from test_voice import VOICES, _multi_voice_manifest


def _flow_cfg(fast_cfg):
    cfg = copy.deepcopy(fast_cfg)
    cfg.variant = "small"
    cfg.voice_mode = "reference"
    cfg.n_voices = 3
    cfg.speaker.channels = [16, 24]
    cfg.speaker.emb_dim = 32
    cfg.speaker.style_dim = 32
    cfg.speaker.n_query = 4
    cfg.flow.text_dim = cfg.text.dim
    cfg.flow.cond_dim = cfg.text.dim
    return cfg.validate()


def _cache(cfg, tmp_path, texts=("hello there", "another line", "a third utterance")):
    import soundfile as sf

    manifest = _multi_voice_manifest(cfg, tmp_path, voices=VOICES, texts=texts)
    model = build_model(cfg)
    cache = build_latent_cache(
        manifest, tmp_path / "cache", cfg, model.autoencoder,
        tokenizer=TextTokenizer(mode=cfg.text.mode),
    )
    return cache, build_model(cfg)


# ------------------------------------------------------------------ collation carries references
def test_cache_batch_carries_a_padded_reference(fast_cfg, tmp_path):
    cfg = _flow_cfg(fast_cfg)
    cache, model = _cache(cfg, tmp_path)
    dataset = LatentShardDataset(cache)

    batch = collate([dataset[0], dataset[1]])
    assert "ref_mel" in batch and "ref_mask" in batch
    assert batch["ref_mel"].shape[-1] == batch["ref_mask"].shape[-1]
    assert batch["ref_mask"].all(), "a self-reference from a cache item is fully valid"
    n_mels = cfg.audio.n_mels
    assert batch["ref_mel"].shape[1] == n_mels

    # ragged lengths must be padded and masked, not silently truncated to the first item
    short = dict(dataset[0])
    long = dict(dataset[1])
    short["ref_mel"] = short["log_mel"][:, :40]
    long["ref_mel"] = long["log_mel"]
    padded = collate([short, long])
    assert padded["ref_mel"].shape[-1] == int(long["log_mel"].shape[-1])
    assert int(padded["ref_mask"][0].sum()) == 40
    assert padded["ref_mask"][0, 40:].sum() == 0

    # and max_ref_frames truncates (PilotTTS caps the prompt at 15 s)
    capped = collate([long, long], max_ref_frames=50)
    assert capped["ref_mel"].shape[-1] == 50
    assert int(capped["ref_mask"].sum()) == 2 * 50


def test_cache_batch_without_references_has_no_ref_key(fast_cfg, tmp_path):
    """Legacy caches (no log_mel) must still collate rather than crash."""
    cfg = _flow_cfg(fast_cfg)
    cache, _ = _cache(cfg, tmp_path)
    dataset = LatentShardDataset(cache)
    stripped = [dict(dataset[0]), dict(dataset[1])]
    for item in stripped:
        item.pop("log_mel")
        item.pop("ref_mel", None)
    batch = collate(stripped)
    assert "ref_mel" not in batch
    assert "latent" in batch


# ------------------------------------------------------------------ cross-sample pairing
def _reference_source(dataset, ref: torch.Tensor, width: int) -> int:
    """Which cache item did this (padded) reference come from?  -1 if none matches."""
    for j in range(len(dataset)):
        candidate = dataset[j]["log_mel"]
        w = min(width, int(candidate.shape[-1]))
        if torch.allclose(ref[:, :w], candidate[:, :w], atol=1e-6):
            return j
    return -1


def test_pairing_picks_a_different_utterance_of_the_same_voice(fast_cfg, tmp_path):
    cfg = _flow_cfg(fast_cfg)
    cache, _ = _cache(cfg, tmp_path)
    dataset = LatentShardDataset(cache)
    voices = [int(dataset[i]["voice"]) for i in range(len(dataset))]
    # fixture layout: one voice per text, so voice 0 is items 0, 3, 6, ...
    assert voices[0] == 0 and voices[1] == 1

    source = LatentShardBatchSource(
        dataset, batch_size=2, shuffle=False, seed=0, pair_references=True
    )
    source.order, source.pos = [0, 1], 0
    batch = source()
    width = int(batch["ref_mask"][0].sum())
    ref0 = batch["ref_mel"][0]
    assert ref0[:, width:].abs().sum() == 0, "padding must be zero, not stale data"
    origin = _reference_source(dataset, ref0, width)
    assert origin != 0, "the reference must not be the target utterance"
    assert voices[origin] == voices[0], "the partner must be the same voice"
    assert origin == 3, "the only other 'low' utterance in the fixture"


def test_pairing_supplies_a_different_voice_negative(fast_cfg, tmp_path):
    cfg = _flow_cfg(fast_cfg)
    cache, _ = _cache(cfg, tmp_path)
    dataset = LatentShardDataset(cache)
    voices = [int(dataset[i]["voice"]) for i in range(len(dataset))]

    source = LatentShardBatchSource(
        dataset, batch_size=2, shuffle=False, seed=0, pair_references=True
    )
    source.order, source.pos = [0, 1], 0
    batch = source()
    assert "ref_mel_neg" in batch and "ref_mask_neg" in batch
    width = int(batch["ref_mask_neg"][0].sum())
    origin = _reference_source(dataset, batch["ref_mel_neg"][0], width)
    assert origin != -1, "the negative must come from the corpus"
    assert voices[origin] != voices[0], "the negative reference must be a different voice"


def test_self_reference_is_opt_out(fast_cfg, tmp_path):
    cfg = _flow_cfg(fast_cfg)
    cache, _ = _cache(cfg, tmp_path)
    dataset = LatentShardDataset(cache)
    source = LatentShardBatchSource(
        dataset, batch_size=2, shuffle=False, seed=0, pair_references=False
    )
    source.order, source.pos = [0, 1], 0
    batch = source()
    ref = batch["ref_mel"][0, :, : int(batch["ref_mask"][0].sum())]
    assert torch.allclose(ref, dataset[0]["log_mel"])


# ------------------------------------------------------------------ the loss is actually live
def test_flow_step_uses_references_and_the_control_sends_no_gradient(fast_cfg, tmp_path):
    """Control: without a reference the *reference-dependent* modules get no gradient at all --
    which is precisely what the cached training path did before this was wired.

    The frozen-fallback path still trains ``constant_style`` (the learned constant that replaces
    the style input for a single voice, Paradee) and ``id_proj.bias`` (a bias on a zero input learns
    a constant offset), so the control measures exactly the three modules that can only ever learn
    identity or style *from a reference*: the ECAPA speaker encoder, the mel memory encoder and the
    Q-Former.
    """
    cfg = _flow_cfg(fast_cfg)
    cache, _ = _cache(cfg, tmp_path)
    dataset = LatentShardDataset(cache)
    source = LatentShardBatchSource(
        dataset, batch_size=2, shuffle=False, seed=0, pair_references=True
    )
    source.order, source.pos = [0, 1], 0

    def reference_encoder_grad(model) -> float:
        modules = [model.speaker.mem_encoder, model.speaker.qformer, model.speaker.speaker]
        return sum(
            float(p.grad.abs().sum())
            for module in modules
            for p in module.parameters()
            if p.grad is not None
        )

    def run(with_refs: bool):
        model = build_model(cfg)
        batch = source()
        if not with_refs:
            for key in ("ref_mel", "ref_mask", "ref_mel_neg", "ref_mask_neg"):
                batch.pop(key, None)
        model.zero_grad(set_to_none=True)
        loss, logs = flow_step(cfg, model, batch)
        loss.backward()
        return float(loss.detach()), logs, reference_encoder_grad(model)

    loss_ref, logs_ref, grad_ref = run(True)
    loss_none, logs_none, grad_none = run(False)

    assert math.isfinite(loss_ref)
    assert grad_ref > 0, "the speaker/style conditioner must receive gradient with references"
    assert "style_separation" in logs_ref, "the separation term must be active"
    assert grad_none == 0.0, (
        "control failed: without a reference the speaker encoder / Q-Former / memory encoder are "
        "unreachable, so the pre-fix cached path could never learn identity or style"
    )
    assert "style_separation" not in logs_none
    assert loss_ref != loss_none


def test_style_separation_loss_pushes_different_speakers_apart():
    a = torch.randn(2, 4, 8)
    b = torch.randn(2, 4, 8)
    opposite = SpeakerConditioner.style_separation_loss(a, -a)
    identical = SpeakerConditioner.style_separation_loss(a, a)
    assert float(opposite) == pytest.approx(0.0, abs=1e-6)
    assert float(identical) > 0.5, "identical style sets across speakers must be penalised"

    # a same-speaker consistency term is still available, and is not the default
    assert float(SpeakerConditioner.cosine_style_loss(a, a)) == pytest.approx(0.0, abs=1e-6)
    assert float(SpeakerConditioner.cosine_style_loss(a, -a)) == pytest.approx(2.0, abs=1e-4)


def test_reflow_and_streaming_still_get_references(fast_cfg, tmp_path):
    """The pairing loader must not break the ordinary single-reference path."""
    cfg = _flow_cfg(fast_cfg)
    cache, _ = _cache(cfg, tmp_path)
    dataset = LatentShardDataset(cache)
    batch = LatentShardBatchSource(dataset, batch_size=2, shuffle=False, seed=0)()
    assert batch["ref_mel"].shape[0] == 2
    assert batch["ref_mel"].shape[1] == cfg.audio.n_mels
    assert batch["ref_mask"].any()
