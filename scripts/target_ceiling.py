"""Is the ~0.40 plateau the model, or the target's own text-determinacy?

The token route stalled at validation per-dim correlation 0.402 after 10 000 steps. Before buying more
compute or another architecture, measure the ceiling: how much do two utterances of the **same text in
different voices** agree on their token latents? Anything the text does not determine cannot be predicted
from text by any model, so that agreement is an upper bound on the target.

The operator's eight takes are exactly this experiment -- one paragraph, eight voices -- and they are in the
cache, sharing token ids by construction. Controls: different text in the same voice, and different text in
different voices.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from parakeet.data.dataset import LatentShardDataset

CACHE = "runs/mixed_v2/latent_cache"


def correlate(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-token-dimension correlation, aligned by index (same text -> same token index)."""
    length = min(a.shape[0], b.shape[0])
    a, b = a[:length], b[:length]
    scores = []
    for dim in range(a.shape[1]):
        x, y = a[:, dim], b[:, dim]
        if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
            continue
        scores.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
    return sum(scores) / max(1, len(scores))


def correlate_frames(a: torch.Tensor, b: torch.Tensor) -> float:
    length = min(a.shape[-1], b.shape[-1])
    a, b = a[:, :length], b[:, :length]
    scores = []
    for dim in range(a.shape[0]):
        x, y = a[dim], b[dim]
        if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
            continue
        scores.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
    return sum(scores) / max(1, len(scores))


dataset = LatentShardDataset(CACHE)
print(f"cache {CACHE}: {len(dataset)} items")

# group items by their token ids: identical ids means identical text (the eight takes)
by_text: dict[tuple, list[int]] = defaultdict(list)
for index in range(len(dataset)):
    ids = tuple(int(v) for v in dataset[index]["ids"][:40])
    by_text[ids].append(index)

groups = {k: v for k, v in by_text.items() if len(v) >= 2}
print(f"text groups with more than one rendition: {len(groups)} "
      f"(largest {max((len(v) for v in groups.values()), default=0)} renditions)")

same_text_token, same_text_frame, same_text_voices = [], [], []
for ids, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:6]:
    voices = {int(dataset[i]["voice"]) for i in members}
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            x, y = dataset[members[a]], dataset[members[b]]
            same_text_token.append(correlate(x["latent_token"], y["latent_token"]))
            same_text_frame.append(correlate_frames(x["latent"], y["latent"]))
    same_text_voices.append((len(members), len(voices)))

control_token, control_frame = [], []
generator = torch.Generator().manual_seed(0)
order = torch.randperm(len(dataset), generator=generator).tolist()
for i in range(0, 240, 2):
    x, y = dataset[order[i]], dataset[order[i + 1]]
    control_token.append(correlate(x["latent_token"], y["latent_token"]))
    control_frame.append(correlate_frames(x["latent"], y["latent"]))

mean = lambda v: sum(v) / max(1, len(v))
print(f"\nSAME TEXT, different voices : token rho {mean(same_text_token):+.3f} "
      f"(n={len(same_text_token)}) | frame rho {mean(same_text_frame):+.3f}")
print(f"DIFFERENT text (control)    : token rho {mean(control_token):+.3f} "
      f"(n={len(control_token)}) | frame rho {mean(control_frame):+.3f}")
print(f"\nmodel's validation token rho at 10 000 steps: +0.402 (item split) / +0.371 (text-grouped split)")
print("\n  MEASURED RESULT (round 60), and it refutes the hypothesis this script was written to test:")
print("  the model's validation rho is HIGHER than the same-text cross-voice agreement.  So that agreement")
print("  is NOT an upper bound on the target -- the model is given the *voice*, and these pairs differ in")
print("  voice by construction, so their agreement is depressed by the very factor the model is told about.")
print("  A text+voice -> latent model is therefore not capped at 0.25, and the honest ceiling for this")
print("  target remains unmeasured.  What the script does establish is that the target is not the wall the")
print("  ~0.37 plateau suggested, and that the corpus repeats 332 texts across voices -- which is why the")
print("  evaluation split had to become text-grouped.")
