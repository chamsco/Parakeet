"""Per-voice latent normalisation: fit each voice's own mean/variance and record it in the cache.

Two measurements point at the same fix:

* the expansions (770 paid Speechify + 771 free Kokoro) encode at token std ~0.70 against the corpus's
  ~1.15 -- 41 % apart, per voice, with teacher, prompt length, level, checkpoint and front-end all ruled
  out. Per-voice normalisation removes exactly that difference, because it is a *per-voice* scale;
* the text side plateaus at rho ~0.37 because one text admits many acoustics (round 61), and a per-voice
  scale is one of the things the text cannot predict. Removing it leaves a target that is closer to what the
  text determines.

This script fits the statistics and writes them into `cache_meta.json` as `voice_norm`; nothing is rewritten,
so it is cheap and reversible. Applying them is a separate, flag-gated step in the dataset.

    python scripts/fit_voice_norm.py --cache runs/expanded_v2 --dry-run
    python scripts/fit_voice_norm.py --cache runs/expanded_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from parakeet.data.dataset import LatentShardDataset


def fit_voice_norm(cache: Path, min_items: int = 4) -> Dict[str, Dict[str, List[float]]]:
    """Per-voice mean/std of the cached latents, computed on frames (the flow's target).

    Frames rather than tokens: the flow generates frame latents, so that is the scale a synthesis path has to
    invert. Tokens follow the same statistics per voice, and the token route's own head is calibrated
    against whatever scale it is given.
    """
    dataset = LatentShardDataset(cache)
    meta = json.loads((cache / "cache_meta.json").read_text(encoding="utf-8"))
    names = list(meta.get("voice_names") or [])
    per_voice: Dict[int, List[torch.Tensor]] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        voice = item.get("voice")
        if voice is None:
            continue
        per_voice.setdefault(int(voice), []).append(item["latent"])

    out: Dict[str, Dict[str, List[float]]] = {}
    for voice, latents in sorted(per_voice.items()):
        name = names[voice] if voice < len(names) and names[voice] else f"voice_{voice}"
        if len(latents) < min_items:
            continue  # too few items to estimate a variance from
        stacked = torch.cat(latents, dim=-1)
        out[name] = {
            "mean": stacked.mean(dim=-1).tolist(),
            "std": stacked.std(dim=-1).clamp_min(1e-6).tolist(),
            "items": len(latents),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit per-voice latent statistics into cache_meta.json")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--min-items", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    fitted = fit_voice_norm(cache, args.min_items)
    print(f"{len(fitted)} voices fitted (min {args.min_items} items):")
    for name, stats in fitted.items():
        mean = sum(stats["mean"]) / len(stats["mean"])
        std = sum(stats["std"]) / len(stats["std"])
        print(f"  {name[:14]:>14s}  n={stats['items']:5d}  mean {mean:+.4f}  std {std:.4f}")

    if args.dry_run:
        print("\n(dry run -- cache_meta.json untouched)")
        return 0

    path = cache / "cache_meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["voice_norm"] = fitted
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote voice_norm for {len(fitted)} voices -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
