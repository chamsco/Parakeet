"""Merge latent caches into one, so a corpus expansion can train with the original data.

`LatentShardDataset` reads exactly one directory, so an expanded corpus needs its own cache *and* a way to
train on both. Rebuilding the whole cache from audio would re-encode a thousand utterances for nothing;
merging the shards is a copy.

Three details decide whether the result is usable:

* **voice indices must be remapped.** Each cache has its own `voice_names` table, and a voice index is only
  meaningful inside its own table. Merging without remapping silently re-labels speakers -- the kind of bug
  that shows up as a model that cannot hold an identity.
* **the latent normaliser must come from the base cache.** The statistics are what every decode path applies,
  and checkpoints were trained against them; refitting on the merged corpus would silently move the space
  under a resumed run.
* **streaming.** `LatentShardDataset` loads every shard into memory at once; the merge must not, or doubling
  a corpus would need the whole thing resident twice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch


def _shard_entries(cache: Path) -> List[dict]:
    return json.loads((cache / "index.json").read_text(encoding="utf-8"))


def _meta(cache: Path) -> dict:
    return json.loads((cache / "cache_meta.json").read_text(encoding="utf-8"))


def merge_caches(base: Path, extra: Path, out: Path, shard_size: int = 500) -> Dict[str, object]:
    base_meta, extra_meta = _meta(base), _meta(extra)
    base_voices = list(base_meta.get("voice_names") or [])
    voice_map = {name: index for index, name in enumerate(base_voices)}
    out.mkdir(parents=True, exist_ok=True)

    written: List[dict] = []
    buffer: List[dict] = []
    shard_index = 0

    def flush() -> None:
        nonlocal buffer, shard_index
        if not buffer:
            return
        name = f"shard_{shard_index:05d}.pt"
        torch.save({"items": buffer}, out / name)
        written.append({"path": name, "n": len(buffer)})
        shard_index += 1
        buffer = []

    counts = {"base": 0, "extra": 0, "voices_added": 0}
    widths: Dict[str, int] = {}
    for cache, names, label in ((base, base_voices, "base"), (extra, None, "extra")):
        extra_names = list(extra_meta.get("voice_names") or []) if label == "extra" else names
        for entry in _shard_entries(cache):
            payload = torch.load(cache / entry["path"], map_location="cpu", weights_only=False)
            for item in payload["items"]:
                item = dict(item)
                # A width mismatch merges silently and produces items the model cannot read: the expansion's
                # cache was first built at latent_rate 1 (24-wide tokens) against this cache's 3 (72-wide).
                # Refusing loudly is the only safe behaviour.
                token = item.get("latent_token")
                if token is not None:
                    width = int(token.shape[-1])
                    if label not in widths:
                        widths[label] = width
                    elif widths[label] != width:
                        raise ValueError(
                            f"{label} cache mixes token widths {widths[label]} and {width}"
                        )
                    if "base" in widths and widths[label] != widths["base"]:
                        raise ValueError(
                            f"token width mismatch: base {widths['base']} vs {label} {width}.  The caches "
                            "were built at different latent rates and cannot be merged."
                        )
                voice = item.get("voice")
                if voice is not None and extra_names:
                    position = int(voice)
                    name = extra_names[position] if position < len(extra_names) else f"voice_{position}"
                    if name not in voice_map:
                        voice_map[name] = len(voice_map)
                        counts["voices_added"] += 1
                    item["voice"] = torch.tensor(voice_map[name])
                buffer.append(item)
                counts[label] += 1
                if len(buffer) >= shard_size:
                    flush()
    flush()

    (out / "index.json").write_text(json.dumps(written, indent=2), encoding="utf-8")
    merged_meta = dict(base_meta)
    merged_meta["voice_names"] = list(voice_map)
    # the base cache's normaliser wins: checkpoints were trained against it (see the module docstring)
    merged_meta["latent_norm"] = base_meta.get("latent_norm")
    (out / "cache_meta.json").write_text(json.dumps(merged_meta, indent=2), encoding="utf-8")
    counts["total"] = counts["base"] + counts["extra"]
    counts["voices"] = len(voice_map)
    counts["shards"] = len(written)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge latent caches (streaming, voices remapped)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-size", type=int, default=500)
    args = ap.parse_args()
    counts = merge_caches(Path(args.base), Path(args.extra), Path(args.out), args.shard_size)
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
