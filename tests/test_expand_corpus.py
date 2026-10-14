"""The expansion pipeline must fail loudly on the three symptoms that each cost a round.

Caching, merging and normalising an expansion is mechanical -- but every one of those steps failed silently at
least once in this project: a token width of 24 against the base's 72 (round 56), a per-voice scale 41 % away
from the corpus (rounds 58-62), and a missing voice table. Each surfaced much later, as an IndexError in a
batch source or as unexplained audio. These tests use `verify`, the guard at the end of the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


from expand_corpus import verify


def _merged_cache(root: Path, *, token_width: int = 72, voice_norm: bool = True, scale: float = 1.0,
                  items: int = 6) -> None:
    voices = ["alice", "bob"]
    payload = []
    generator = torch.Generator().manual_seed(0)
    for index in range(items):
        payload.append({
            "ids": torch.tensor([1, 2, 3]),
            "latent": torch.randn(24, 40, generator=generator) * scale,
            "latent_token": torch.randn(3, token_width, generator=generator) * scale,
            "durations": torch.tensor([14, 14, 12]),
            "f0": torch.zeros(3),
            "energy": torch.zeros(3),
            "voice": torch.tensor(index % len(voices)),
        })
    root.mkdir(parents=True, exist_ok=True)
    torch.save({"items": payload}, root / "shard_00000.pt")
    (root / "index.json").write_text(json.dumps([{"path": "shard_00000.pt", "n": len(payload)}]), "utf-8")
    meta = {"latent_dim": 24, "voice_names": voices, "latent_norm": {"mean": [0.0] * 24, "var": [1.0] * 24}}
    if voice_norm:
        meta["voice_norm"] = {
            name: {"mean": [0.0] * 24, "std": [1.0] * 24, "items": items // 2} for name in voices
        }
    (root / "cache_meta.json").write_text(json.dumps(meta), "utf-8")


def test_a_well_formed_merged_cache_passes(tmp_path: Path):
    _merged_cache(tmp_path / "cache")
    report = verify(tmp_path / "cache", base_items=2)
    assert report["items"] == 6 and report["new_items"] == 4
    assert report["token_widths"] == [72]
    assert report["voices_with_normalisation"] == 2


def test_a_token_width_mismatch_is_refused(tmp_path: Path):
    import pytest

    _merged_cache(tmp_path / "cache", token_width=24)
    with pytest.raises(SystemExit, match="token widths"):
        verify(tmp_path / "cache", base_items=2)


def test_a_missing_voice_table_is_refused(tmp_path: Path):
    import pytest

    _merged_cache(tmp_path / "cache", voice_norm=False)
    with pytest.raises(SystemExit, match="voice_norm"):
        verify(tmp_path / "cache", base_items=2)


def test_a_voice_still_off_scale_is_refused(tmp_path: Path):
    """The expansion's own half pooling at 0.5 is exactly the round-58 symptom."""
    import pytest

    _merged_cache(tmp_path / "cache", scale=0.5)
    # unit statistics against data at half scale leaves the pooled standard deviation at 0.5
    with pytest.raises(SystemExit, match="pools at"):
        verify(tmp_path / "cache", base_items=2)
