"""Multi-teacher mixture: the weights must actually reach the gradient.

This file exists because they previously did not. ``MultiTeacherMixer`` was implemented, exported
and unit-tested, but nothing in the cache, the dataset collation or the training stages ever read
it -- so the "mix training of both teachers" claim was a documented config value rather than a
mechanism. These tests pin the whole path: raw weights when the cache is written, collation of the
per-sample metadata, per-sample reduction inside the losses, and finally that the mixture visibly
steers what the student learns.
"""

import copy
import json

import pytest
import torch

from parakeet.config import ParakeetConfig
from parakeet.data.dataset import LatentShardDataset, collate
from parakeet.data.features import build_latent_cache
from parakeet.data.synthetic import LAYOUTS, make_corpus
from parakeet.data.text import TextTokenizer
from parakeet.models import build_model
from parakeet.train.losses import (
    MultiTeacherMixer,
    TextSideDistillLoss,
    per_sample_l1,
    per_sample_mse,
    weighted_mean,
)
from parakeet.train.stages import flow_step, tiny_text_step
from parakeet.audio.f0 import normalized_to_f0


# ------------------------------------------------------------------ weight construction
def test_weights_are_raw_shares_times_quality():
    mixer = MultiTeacherMixer({"orpheus": 0.6, "kokoro": 0.4}, min_weight=0.05)
    w = mixer.weights(["orpheus", "kokoro"])
    assert torch.allclose(w, torch.tensor([0.6, 0.4])), "raw shares, not normalised per item"

    # an unknown teacher falls back to 1.0 rather than silently vanishing
    assert float(mixer.weights(["mystery"])[0]) == 1.0

    # quality multiplies, and the floor keeps a poor sample in the mixture
    wq = mixer.weights(["orpheus", "kokoro"], quality=torch.tensor([1.0, 0.0]))
    assert float(wq[0]) == pytest.approx(0.6)
    assert float(wq[1]) == pytest.approx(0.05)


def test_weighted_mean_renormalises_but_preserves_emphasis():
    per_sample = torch.tensor([0.0, 10.0])
    unweighted = weighted_mean(per_sample, None)
    assert float(unweighted) == pytest.approx(5.0)

    # weights are renormalised to mean 1, so the overall scale does not change...
    favouring_low = weighted_mean(per_sample, torch.tensor([0.9, 0.1]))
    favouring_high = weighted_mean(per_sample, torch.tensor([0.1, 0.9]))
    assert float(favouring_low) == pytest.approx(1.0)
    assert float(favouring_high) == pytest.approx(9.0)

    # ...and scaling every weight by a constant must not change the loss at all
    scaled = weighted_mean(per_sample, torch.tensor([9.0, 1.0]))
    assert float(scaled) == pytest.approx(float(favouring_low))


def test_per_sample_masking_divides_by_valid_count():
    """Masked positions must be excluded, not merely zeroed (F.l1_loss would dilute the mean)."""
    pred = torch.tensor([[1.0, 0.0, 0.0]])  # error only in the first (valid) position
    target = torch.zeros(1, 3)
    mask = torch.tensor([[True, False, False]])
    assert float(per_sample_l1(pred, target, mask)[0]) == pytest.approx(1.0)
    assert float(per_sample_l1(pred, target, None)[0]) == pytest.approx(1.0 / 3.0)


def test_per_sample_masking_counts_channels():
    """Regression: a ``(B, T)`` mask on a ``(B, T, C)`` tensor must divide by ``T * C``, not ``T``.

    The first version of the per-sample helpers forgot to expand the mask across the channel
    dimension, which inflated the latent term of the distillation loss by a factor of ``C`` (24x)
    -- reintroducing exactly the 'one loss term dominates' failure this project already fixed once.
    """
    b, t, c = 2, 3, 4
    pred = torch.zeros(b, t, c)
    target = torch.ones(b, t, c)  # every element has squared error 1
    mask = torch.ones(b, t, dtype=torch.bool)
    assert float(per_sample_mse(pred, target, mask)[0]) == pytest.approx(1.0)
    assert float(per_sample_l1(pred, target, mask)[0]) == pytest.approx(1.0)

    # with a partial mask, only the valid positions may count
    mask[1, 1:] = False
    assert float(per_sample_mse(pred, target, mask)[1]) == pytest.approx(1.0)

    # and the masked per-sample value must equal the unmasked one when the mask is all-True
    assert torch.allclose(per_sample_mse(pred, target, mask), per_sample_mse(pred, target), atol=1e-6)


# ------------------------------------------------------------------ loss-level weighting
def test_text_side_loss_responds_to_sample_weight():
    """A sample with a huge error must dominate when it carries the weight, and not when it doesn't."""
    criterion = TextSideDistillLoss()
    b, t, c = 2, 4, 6
    pred = {
        "log_duration": torch.zeros(b, t),
        "latent_token": torch.zeros(b, t, c),
        "f0": torch.zeros(b, t),
        "energy": torch.zeros(b, t),
    }
    target = {
        "durations": torch.ones(b, t, dtype=torch.long),
        "f0": torch.zeros(b, t),
        "energy": torch.zeros(b, t),
        "latent_token": torch.zeros(b, t, c),
    }
    target["latent_token"][1] = 100.0  # sample 1 is catastrophically wrong
    weight_ok = torch.tensor([0.95, 0.05])  # de-emphasise the bad sample
    weight_bad = torch.tensor([0.05, 0.95])  # emphasise it

    loss_ok, _ = criterion(pred, target, None, sample_weight=weight_ok)
    loss_bad, _ = criterion(pred, target, None, sample_weight=weight_bad)
    assert float(loss_bad) > float(loss_ok) * 5, "weighting must change the objective"

    # unit weights reproduce the unweighted loss exactly
    loss_none, _ = criterion(pred, target, None, sample_weight=None)
    loss_ones, _ = criterion(pred, target, None, sample_weight=torch.ones(b))
    assert float(loss_ones) == pytest.approx(float(loss_none), rel=1e-6)


def _flow_cfg(fast_cfg) -> ParakeetConfig:
    """The Tiny fixture has no flow module; the same dims with the flow variant do."""
    cfg = copy.deepcopy(fast_cfg)
    cfg.variant = "small"
    cfg.voice_mode = "constant"
    return cfg.validate()


def test_flow_loss_responds_to_sample_weight(fast_cfg):
    cfg = _flow_cfg(fast_cfg)
    model = build_model(cfg)
    ids = torch.randint(1, 50, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.bool)
    latent = torch.randn(2, cfg.autoencoder.latent_dim, 36)
    weights = torch.tensor([0.9, 0.1])

    torch.manual_seed(0)
    loss_unweighted, _ = model.flow_loss(ids, mask, latent, context_expansion=1)
    torch.manual_seed(0)
    loss_unit, _ = model.flow_loss(ids, mask, latent, context_expansion=1, sample_weight=torch.ones(2))

    # a lopsided weight must change the value (same seeds, so the only difference is the weight)
    produced = set()
    for w in (torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9])):
        torch.manual_seed(0)
        value, _ = model.flow_loss(ids, mask, latent, context_expansion=1, sample_weight=w)
        produced.add(round(float(value), 6))
    assert len(produced) == 2, "lopsided weights must change the flow loss"
    # unit weights are a no-op
    assert float(loss_unit) == pytest.approx(float(loss_unweighted), rel=1e-5)


def test_flow_loss_repeats_weights_with_context_expansion(fast_cfg):
    """With Ke>1 the estimator batch is repeated, so the weights must be repeated with it."""
    cfg = _flow_cfg(fast_cfg)
    cfg.flow.context_expansion = 3
    model = build_model(cfg)
    ids = torch.randint(1, 50, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.bool)
    latent = torch.randn(2, cfg.autoencoder.latent_dim, 24)
    seen = {}

    def hook(module, inputs, output):
        seen["batch"] = inputs[0].shape[0]

    handle = model.vf.in_proj.register_forward_hook(hook)
    try:
        torch.manual_seed(0)
        loss, _ = model.flow_loss(
            ids, mask, latent, sample_weight=torch.tensor([0.8, 0.2]), context_expansion=3
        )
    finally:
        handle.remove()
    assert seen["batch"] == 6
    assert torch.isfinite(loss)


# ------------------------------------------------------------------ cache + collation plumbing
def _corpus_with_two_teachers(cfg, tmp_path, per_teacher: int = 3):
    import soundfile as sf

    (tmp_path / "wav").mkdir(parents=True, exist_ok=True)
    lines = []
    for name, f0_range in (("orpheus", (90.0, 110.0)), ("kokoro", (190.0, 210.0))):
        for i, utt in enumerate(make_corpus(per_teacher, cfg.audio, seed=hash(name) % 100, f0_range=f0_range)):
            path = tmp_path / "wav" / f"{name}_{i}.wav"
            sf.write(str(path), utt.wav.numpy(), cfg.audio.sample_rate)
            lines.append(
                json.dumps(
                    {
                        "utt_id": f"{name}_{i}",
                        "text": utt.text,
                        "teacher": name,
                        "voice": "v0",
                        "wav_path": f"wav/{name}_{i}.wav",
                        "sample_rate": cfg.audio.sample_rate,
                        "duration_s": float(utt.wav.numel() / cfg.audio.sample_rate),
                    }
                )
            )
    (tmp_path / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path / "manifest.jsonl"


def test_cache_preserves_teacher_provenance_and_weights(fast_cfg, tmp_path):
    cfg = copy.deepcopy(fast_cfg)
    manifest = _corpus_with_two_teachers(cfg, tmp_path, per_teacher=3)
    model = build_model(cfg)
    cache = build_latent_cache(
        manifest,
        tmp_path / "cache",
        cfg,
        model.autoencoder,
        tokenizer=TextTokenizer(mode=cfg.text.mode),
        teacher_weights={"orpheus": 0.9, "kokoro": 0.1},
    )
    meta = json.loads((cache / "cache_meta.json").read_text(encoding="utf-8"))
    assert meta["teacher_names"] == ["orpheus", "kokoro"]
    assert meta["teacher_weights"] == {"orpheus": 0.9, "kokoro": 0.1}

    dataset = LatentShardDataset(cache)
    weights, indices = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        assert "teacher_weight" in item and "teacher_index" in item
        weights.append(float(item["teacher_weight"]))
        indices.append(int(item["teacher_index"]))
    assert set(indices) == {0, 1}
    assert max(weights) == pytest.approx(0.9) and min(weights) == pytest.approx(0.1)

    # and collation must carry them into the batch the trainer sees
    batch = collate([dataset[0], dataset[3]])
    assert batch["teacher_weight"].shape == (2,)
    assert batch["teacher_index"].shape == (2,)
    assert len(set(batch["teacher_weight"].tolist())) == 2


def test_training_step_reads_weights_from_the_batch(fast_cfg):
    """The stage functions must pick the weight up without any extra plumbing."""
    cfg = copy.deepcopy(fast_cfg)
    model = build_model(cfg)
    batch = {
        "ids": torch.randint(1, 50, (2, 6)),
        "text_mask": torch.ones(2, 6, dtype=torch.bool),
        "durations": torch.randint(2, 8, (2, 6)),
        "f0": torch.rand(2, 6),
        "energy": torch.rand(2, 6),
        "latent_token": torch.randn(2, 6, cfg.autoencoder.latent_dim),
    }
    lopsided = {**batch, "teacher_weight": torch.tensor([0.9, 0.1])}
    flipped = {**batch, "teacher_weight": torch.tensor([0.1, 0.9])}
    a, _ = tiny_text_step(cfg, model, lopsided)
    b, _ = tiny_text_step(cfg, model, flipped)
    assert float(a.detach()) != float(b.detach()), "tiny_text_step must consume teacher_weight"

    flow_cfg = _flow_cfg(fast_cfg)
    flow_model = build_model(flow_cfg)
    ids = torch.randint(1, 50, (2, 6))
    flow_batch = {
        "ids": ids,
        "text_mask": torch.ones(2, 6, dtype=torch.bool),
        "latent": torch.randn(2, flow_cfg.autoencoder.latent_dim, 24),
    }
    torch.manual_seed(1)
    a, _ = flow_step(flow_cfg, flow_model, {**flow_batch, "teacher_weight": torch.tensor([0.9, 0.1])})
    torch.manual_seed(1)
    b, _ = flow_step(flow_cfg, flow_model, {**flow_batch, "teacher_weight": torch.tensor([0.1, 0.9])})
    assert float(a.detach()) != float(b.detach()), "flow_step must consume teacher_weight"


# ------------------------------------------------------------------ does it steer the student?
def _targets_for(model, cfg, corpora, weights, tokenizer):
    """Token targets for each teacher's corpus, tagged with that teacher's mixture weight."""
    from parakeet.data.features import token_targets_from_corpus

    out = []
    for corpus, weight in zip(corpora, weights):
        for target in token_targets_from_corpus(model, corpus, cfg, tokenizer):
            target["teacher_weight"] = torch.tensor(float(weight))
            out.append(target)
    return out


def _batch_from(targets, indices):
    picked = [targets[i] for i in indices]
    ids = torch.stack([t["ids"] for t in picked])
    return {
        "ids": ids,
        "text_mask": torch.ones_like(ids, dtype=torch.bool),
        "durations": torch.stack([t["durations"] for t in picked]),
        "f0": torch.stack([t["f0"] for t in picked]),
        "energy": torch.stack([t["energy"] for t in picked]),
        "latent_token": torch.stack([t["latent_token"] for t in picked]),
        "teacher_weight": torch.tensor([float(t["teacher_weight"]) for t in picked]),
    }


@torch.no_grad()
def _mean_predicted_f0_hz(model, targets) -> float:
    """Average predicted F0 (Hz) over a fixed set of inputs, de-normalising the target space."""
    values = []
    for t in targets:
        ids = t["ids"][None]
        side = model.text_side(ids)
        values.append(normalized_to_f0(side["f0"][0]).mean())
    return float(torch.stack(values).mean().item())


def test_mixture_weights_steer_what_the_text_side_learns(fast_cfg):
    """Two 'teachers' with very different pitch; the mixture must tilt the student accordingly.

    Same initialisation, same data, same step count -- only the per-sample weights differ.  If the
    weights reach the gradient, favouring the low-pitch teacher must yield a lower predicted F0.
    """
    cfg = copy.deepcopy(fast_cfg)
    cfg.train.lr = 2e-3
    tokenizer = TextTokenizer(mode=cfg.text.mode)

    low = make_corpus(4, cfg.audio, seed=11, f0_range=(85.0, 105.0))
    high = make_corpus(4, cfg.audio, seed=12, f0_range=(200.0, 220.0))
    low_truth = sum(u.token_f0[0] for u in low) / len(low)
    high_truth = sum(u.token_f0[0] for u in high) / len(high)
    assert high_truth > low_truth * 1.8

    def train_with(low_weight: float, high_weight: float):
        torch.manual_seed(cfg.train.seed)
        model = build_model(cfg)
        targets = _targets_for(model, cfg, [low, high], [low_weight, high_weight], tokenizer)
        order = torch.Generator().manual_seed(0)
        for _ in range(60):
            idx = torch.randint(0, len(targets), (4,), generator=order).tolist()
            loss, _ = tiny_text_step(cfg, model, _batch_from(targets, idx))
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p -= cfg.train.lr * p.grad
                        p.grad = None
        return _mean_predicted_f0_hz(model, targets)

    favouring_low = train_with(0.95, 0.05)
    favouring_high = train_with(0.05, 0.95)

    assert favouring_high > favouring_low, (
        f"mixture did not steer the student: low-weighted {favouring_low:.1f} Hz vs "
        f"high-weighted {favouring_high:.1f} Hz (teachers at {low_truth:.0f}/{high_truth:.0f} Hz)"
    )
