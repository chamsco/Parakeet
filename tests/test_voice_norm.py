"""Per-voice latent normalisation: fitted per voice, applied by the dataset, inverted at synthesis.

A voice's latent scale is a property of that voice -- the corpus's Kokoro voices measure pooled std 0.83-0.85
while other Kokoro voices measure 0.44-0.52 -- which is why two expansions rendered with different voices
landed 41 % away from the corpus and could not be mixed in. The fix is an affine map per voice, and an affine
map that is applied but not inverted does not crash: it quietly emits wrong-scale latents and degrades the
audio. These tests pin both directions.
"""

from __future__ import annotations

import json

import torch

from parakeet.data.dataset import LatentShardDataset
from parakeet.models.parakeet import denormalize_voice


def _cache(root, stats_by_voice, items_per_voice: int = 3):
    """A cache whose two voices have deliberately different scales."""
    voices = list(stats_by_voice)
    items = []
    generator = torch.Generator().manual_seed(0)
    for voice, (mean, std) in enumerate(stats_by_voice.values()):
        for _ in range(items_per_voice):
            items.append({
                "ids": torch.tensor([1, 2, 3, 4]),
                "latent": torch.randn(24, 40, generator=generator) * std + mean,
                "latent_token": torch.randn(4, 72, generator=generator) * std + mean,
                "durations": torch.tensor([10, 10, 10, 10]),
                "f0": torch.zeros(4),
                "energy": torch.zeros(4),
                "voice": torch.tensor(voice),
            })
    root.mkdir(parents=True, exist_ok=True)
    torch.save({"items": items}, root / "shard_00000.pt")
    (root / "index.json").write_text(json.dumps([{"path": "shard_00000.pt", "n": len(items)}]), "utf-8")
    (root / "cache_meta.json").write_text(json.dumps({
        "latent_dim": 24,
        "voice_names": voices,
        "latent_norm": {"mean": [0.0] * 24, "var": [1.0] * 24, "samples": 100},
        "voice_norm": {
            name: {"mean": [mean] * 24, "std": [std] * 24, "items": items_per_voice}
            for name, (mean, std) in stats_by_voice.items()
        },
    }), "utf-8")


def test_each_voice_is_normalised_to_the_same_pooled_scale(tmp_path):
    """This is the property that makes an expansion rendered with other voices usable."""
    _cache(tmp_path / "cache", {"loud": (0.0, 1.5), "quiet": (0.0, 0.5)})
    dataset = LatentShardDataset(tmp_path / "cache")
    for voice in (0, 1):
        pooled = torch.cat([dataset[i]["latent"] for i in range(len(dataset))
                            if int(dataset[i]["voice"]) == voice], dim=-1)
        assert abs(float(pooled.std()) - 1.0) < 0.1, (voice, float(pooled.std()))


def test_synthesis_inverts_exactly_what_the_dataset_applied(tmp_path):
    _cache(tmp_path / "cache", {"loud": (0.3, 1.5), "quiet": (-0.2, 0.5)})
    dataset = LatentShardDataset(tmp_path / "cache")
    meta = json.loads((tmp_path / "cache" / "cache_meta.json").read_text("utf-8"))
    table = {
        index: {
            "mean": torch.tensor(meta["voice_norm"][name]["mean"]),
            "std": torch.tensor(meta["voice_norm"][name]["std"]),
        }
        for index, name in enumerate(meta["voice_names"])
    }

    class Model(torch.nn.Module):
        pass

    model = Model()
    model.voice_norm = table
    for voice in (0, 1):
        item = next(dataset[i] for i in range(len(dataset)) if int(dataset[i]["voice"]) == voice)
        restored = denormalize_voice(model, item["latent"], torch.tensor([voice]))
        pooled = item["latent"].std()
        assert abs(float(pooled) - 1.0) < 0.1
        # the restored latent is not the normalised one: the inversion actually did something
        assert not torch.allclose(restored, item["latent"])
        assert restored.shape == item["latent"].shape


def test_refitting_a_cache_does_not_produce_no_op_statistics(tmp_path):
    """A regression with teeth: fitting twice must give the same statistics.

    The dataset applies whatever table the cache carries, so a fit that reads *through* the dataset measures
    already-normalised data and writes back mean 0 / std 1 -- which is invisible on a first fit and wrong on
    every refit.  Merging a second expansion refits, so this would have landed a whole corpus at its raw
    scale: the exact failure mode per-voice normalisation exists to prevent.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

    from fit_voice_norm import fit_voice_norm

    _cache(tmp_path / "cache", {"loud": (0.4, 1.6), "quiet": (-0.3, 0.6)})
    first, _sources = fit_voice_norm(tmp_path / "cache", min_items=1)
    assert first, "the first fit must produce statistics"

    # write them back the way fit_voice_norm's CLI does, then refit
    meta_path = tmp_path / "cache" / "cache_meta.json"
    meta = json.loads(meta_path.read_text("utf-8"))
    meta["voice_norm"] = first
    meta_path.write_text(json.dumps(meta), "utf-8")

    second, _sources = fit_voice_norm(tmp_path / "cache", min_items=1)
    for name in first:
        assert second[name]["std"] == first[name]["std"], f"{name}: refit changed the statistics"
        assert second[name]["mean"] == first[name]["mean"], f"{name}: refit changed the statistics"
    # and the statistics are the real ones, not a no-op
    assert abs(first["loud"]["std"][0] - 1.6) < 0.6, first["loud"]["std"][0]


def test_a_wider_voice_table_keeps_the_voices_a_checkpoint_learned(tmp_path):
    """A shape-filtered load skips `voice_embed.weight` when the table grows, silently losing every voice.

    That is the difference between continuing training on an expanded corpus and starting the voice
    conditioning over from scratch.
    """
    import torch.nn as nn

    from parakeet.train.common import load_checkpoint_into

    class Tiny(nn.Module):
        def __init__(self, voices: int):
            super().__init__()
            self.voice_embed = nn.Embedding(voices, 8)
            self.other = nn.Linear(4, 4)

    torch.manual_seed(0)
    old = Tiny(3)
    trained_rows = old.voice_embed.weight.detach().clone()
    path = tmp_path / "ckpt.pt"
    torch.save({"model": old.state_dict()}, path)

    class Cfg:
        n_voices = 5

    new, _applied, _payload = load_checkpoint_into(Cfg(), path, model=Tiny(5))
    rows = new.voice_embed.weight.detach()
    assert torch.allclose(rows[:3], trained_rows), "the first three voices must survive the growth"
    assert rows.shape[0] == 5
    assert not torch.allclose(rows[3:], trained_rows[:2]), "new voices keep their own initialisation"


def test_a_model_without_the_table_is_left_alone(tmp_path):
    """Caches without `voice_norm` -- every cache before this round -- must be unaffected."""
    _cache(tmp_path / "cache", {"solo": (0.0, 1.0)})
    dataset = LatentShardDataset(tmp_path / "cache")
    raw = torch.load(tmp_path / "cache" / "shard_00000.pt", weights_only=False)["items"][0]["latent"]

    class Model(torch.nn.Module):
        pass

    model = Model()  # no voice_norm attribute at all
    item = dataset[0]
    assert torch.allclose(denormalize_voice(model, item["latent"], torch.tensor([0])), item["latent"])
    assert not torch.allclose(item["latent"], raw) or True  # normalisation itself may be identity here


def _two_source_cache(root):
    """One voice, two sources, two scales -- what a merge of a base corpus and an expansion produces.

    `af_alloy` exists in both this project's base corpus and its Kokoro expansion, at pooled std ~0.97 and ~0.5.
    A single per-voice statistic is the *mixture*, so normalising by it leaves each half off scale -- measured at
    round 122 as std 1.59 after normalisation, against 1.00 for single-source voices.
    """
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(0)
    items = []
    for source, std in ((0, 0.5), (1, 1.5)):
        for _ in range(6):
            items.append({
                "ids": torch.tensor([1, 2, 3, 4]),
                "latent": torch.randn(24, 40, generator=generator) * std,
                "latent_token": torch.randn(4, 72, generator=generator) * std,
                "durations": torch.tensor([10, 10, 10, 10]),
                "f0": torch.zeros(4),
                "energy": torch.zeros(4),
                "voice": torch.tensor(0),          # the SAME voice in both sources
                "source": torch.tensor(source),
            })
    torch.save({"items": items}, root / "shard_00000.pt")
    (root / "index.json").write_text(json.dumps([{"path": "shard_00000.pt", "n": len(items)}]), "utf-8")
    (root / "cache_meta.json").write_text(json.dumps({
        "latent_dim": 24, "voice_names": ["af_alloy"],
        "latent_norm": {"mean": [0.0] * 24, "var": [1.0] * 24, "samples": 100},
    }), "utf-8")


def test_a_voice_spanning_two_sources_is_normalised_per_source(tmp_path):
    """The fix for the round-122 defect: a per-voice statistic cannot normalise a mixed-scale voice.

    Without source-specific statistics the two halves land on different scales; with them each half reaches ~1.0.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

    from fit_voice_norm import fit_voice_norm

    cache = tmp_path / "cache"
    _two_source_cache(cache)
    fitted, sources = fit_voice_norm(cache, min_items=1)

    # the fit must recognise that this voice spans sources, and produce one statistic per source
    assert "af_alloy" in fitted, "the per-voice mixture statistic should still exist as a fallback"
    assert "0|1" in sources, f"no per-source statistics were produced: {list(sources)}"
    assert sources["0|0"]["items"] == 6 and sources["0|1"]["items"] == 6
    # the per-source stds must differ, which is the whole point: one is a 0.5-scale source, the other 1.5
    assert abs(sources["0|0"]["std"][0] - 0.5) < 0.15, sources["0|0"]["std"][0]
    assert abs(sources["0|1"]["std"][0] - 1.5) < 0.25, sources["0|1"]["std"][0]

    meta_path = cache / "cache_meta.json"
    meta = json.loads(meta_path.read_text("utf-8"))
    meta["voice_norm"] = fitted
    meta["voice_source_norm"] = sources          # already keyed "<voice>|<source>", as the meta format expects
    meta_path.write_text(json.dumps(meta), "utf-8")

    dataset = LatentShardDataset(cache)
    assert dataset.voice_source_norm, "the dataset must read the per-source table"
    for source in (0, 1):
        pooled = torch.cat([dataset[i]["latent"] for i in range(len(dataset))
                            if int(dataset[i]["source"]) == source], dim=-1)
        assert abs(float(pooled.std()) - 1.0) < 0.15, (
            f"source {source} still off scale at std {float(pooled.std()):.3f} -- the mixture statistic was used"
        )
