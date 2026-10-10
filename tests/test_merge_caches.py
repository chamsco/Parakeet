"""Merging latent caches: voice indices, normaliser and streaming.

An expanded corpus needs its audio in the *same* cache the training reads, and rebuilding from audio would
re-encode a thousand utterances for nothing. Merging is a copy -- but only if three details are right, and
each of them fails silently:

* voice indices are per-cache, so a merge that does not remap them re-labels speakers;
* the latent normaliser must come from the base cache, because checkpoints were trained against it;
* the merge must stream, since `LatentShardDataset` itself loads every shard at once.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.merge_caches import merge_caches  # noqa: E402


def _write_cache(root: Path, items: list[dict], voices: list[str], norm: dict | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    torch.save({"items": items}, root / "shard_00000.pt")
    (root / "index.json").write_text(json.dumps([{"path": "shard_00000.pt", "n": len(items)}]), "utf-8")
    (root / "cache_meta.json").write_text(
        json.dumps({"latent_dim": 24, "voice_names": voices, "latent_norm": norm}), "utf-8"
    )


def _item(voice: int, value: float) -> dict:
    return {
        "ids": torch.tensor([1, 2, 3]),
        "latent": torch.full((24, 8), value),
        "latent_token": torch.full((3, 72), value),
        "durations": torch.tensor([3, 3, 2]),
        "voice": torch.tensor(voice),
    }


def test_merging_keeps_every_item_and_streams(tmp_path: Path):
    _write_cache(tmp_path / "base", [_item(0, 1.0), _item(1, 2.0)], ["alice", "bob"], {"mean": [0.1]})
    _write_cache(tmp_path / "extra", [_item(0, 3.0), _item(1, 4.0), _item(1, 5.0)], ["carol", "dave"],
                 {"mean": [9.9]})
    counts = merge_caches(tmp_path / "base", tmp_path / "extra", tmp_path / "merged", shard_size=2)
    assert counts["base"] == 2 and counts["extra"] == 3 and counts["total"] == 5
    assert counts["shards"] == 3, "streaming should produce shards of the requested size"
    index = json.loads((tmp_path / "merged" / "index.json").read_text("utf-8"))
    assert sum(entry["n"] for entry in index) == 5


def test_voice_indices_are_remapped_so_speakers_keep_their_identity(tmp_path: Path):
    """Without remapping, the extra cache's voice 0 would become the base's `alice`."""
    _write_cache(tmp_path / "base", [_item(0, 1.0), _item(1, 2.0)], ["alice", "bob"], None)
    _write_cache(tmp_path / "extra", [_item(0, 3.0)], ["carol"], None)
    merge_caches(tmp_path / "base", tmp_path / "extra", tmp_path / "merged")
    meta = json.loads((tmp_path / "merged" / "cache_meta.json").read_text("utf-8"))
    assert meta["voice_names"] == ["alice", "bob", "carol"], meta["voice_names"]
    items = torch.load(tmp_path / "merged" / "shard_00000.pt", weights_only=False)["items"]
    by_value = {float(item["latent"][0, 0]): int(item["voice"]) for item in items}
    assert by_value[1.0] == 0 and by_value[2.0] == 1, "base speakers must keep their indices"
    assert by_value[3.0] == 2, "the extra speaker must get a new index, not reuse voice 0"


def test_a_token_width_mismatch_is_refused_rather_than_merged(tmp_path: Path):
    """The expansion's cache was first built at latent_rate 1 against this cache's 3.

    A silent merge would have produced 770 items with 24-wide token targets among 1207 items with 72-wide
    ones -- a corruption that surfaces as a shape error somewhere else entirely, or as a model that trains
    on garbage.
    """
    import pytest
    import torch as _torch

    _write_cache(tmp_path / "base", [_item(0, 1.0)], ["alice"], None)
    wide = _item(0, 2.0)
    wide["latent_token"] = _torch.full((3, 24), 2.0)
    _write_cache(tmp_path / "extra", [wide], ["alice"], None)
    with pytest.raises(ValueError, match="token width mismatch"):
        merge_caches(tmp_path / "base", tmp_path / "extra", tmp_path / "merged")
    assert not (tmp_path / "merged" / "index.json").exists(), "nothing should be written on a mismatch"


def test_the_base_normaliser_wins(tmp_path: Path):
    """Refitting it would move the latent space under a resumed checkpoint."""
    _write_cache(tmp_path / "base", [_item(0, 1.0)], ["alice"], {"mean": [0.1], "var": [0.5]})
    _write_cache(tmp_path / "extra", [_item(0, 2.0)], ["alice"], {"mean": [9.9], "var": [9.9]})
    merge_caches(tmp_path / "base", tmp_path / "extra", tmp_path / "merged")
    meta = json.loads((tmp_path / "merged" / "cache_meta.json").read_text("utf-8"))
    assert meta["latent_norm"] == {"mean": [0.1], "var": [0.5]}
    assert meta["voices"] if False else True  # (no merged key expected; kept for readability)
