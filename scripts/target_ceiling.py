"""Is the ~0.40 plateau the model, or the target's own text-determinacy?

The token route stalled at validation per-dim correlation 0.402 after 10 000 steps. Before buying more
compute or another architecture, measure the ceiling: how much do two utterances of the **same text in
different voices** agree on their token latents? Anything the text does not determine cannot be predicted
from text by any model, so that agreement is an upper bound on the target.

The operator's eight takes are exactly this experiment -- one paragraph, eight voices -- and they are in the
cache, sharing token ids by construction. Controls: different text in the same voice, and different text in
different voices.

Round 79 added `--cache` and `--no-voice-norm` because those eight takes live in `runs/expanded_v2`, which now
carries per-voice statistics. That matters to the *measurement*: per-voice normalisation is an affine map per
voice, and it removes a per-voice scale difference the model is told about through the voice embedding -- so
same-text cross-voice agreement measured on a fitted cache should be **higher** than the 0.248/0.171 figures
this script first reported on an unfitted one. Toggling the flag on the same cache isolates exactly that.

    python scripts/target_ceiling.py --cache runs/expanded_v2 [--no-voice-norm]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from parakeet.data.dataset import LatentShardDataset

_ap = argparse.ArgumentParser(description="Same-text cross-voice agreement: the target's own ceiling")
_ap.add_argument("--cache", default="runs/mixed_v2/latent_cache")
_ap.add_argument("--items", type=int, default=0, help="0 = the whole cache")
_ap.add_argument("--no-voice-norm", action="store_true",
                 help="read raw latents, i.e. the measurement as it was taken before per-voice fitting")
_args = _ap.parse_args()

CACHE = _args.cache


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


_indices = list(range(_args.items)) if _args.items else None
dataset = LatentShardDataset(CACHE, indices=_indices, apply_voice_norm=not _args.no_voice_norm)
print(f"cache {CACHE}: {len(dataset)} items | voice_norm applied: {not _args.no_voice_norm} "
      f"| table present: {bool(dataset.voice_norm)}")

# group items by their token ids: identical ids means identical text (the eight takes)
by_text: dict[tuple, list[int]] = defaultdict(list)
for index in range(len(dataset)):
    ids = tuple(int(v) for v in dataset[index]["ids"][:40])
    by_text[ids].append(index)

groups = {k: v for k, v in by_text.items() if len(v) >= 2}
print(f"text groups with more than one rendition: {len(groups)} "
      f"(largest {max((len(v) for v in groups.values()), default=0)} renditions)")

same_text_token, same_text_frame, same_text_voices = [], [], []
same_voice_token, same_voice_frame = [], []      # text AND voice shared: the flow's own information set
for ids, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:6]:
    voices = {int(dataset[i]["voice"]) for i in members}
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            x, y = dataset[members[a]], dataset[members[b]]
            token_score = correlate(x["latent_token"], y["latent_token"])
            frame_score = correlate_frames(x["latent"], y["latent"])
            same_text_token.append(token_score)
            same_text_frame.append(frame_score)
            # The flow is given the voice as well as the text, so the ceiling that applies to it is the
            # agreement of two renditions sharing BOTH.  Comparing that with the cross-voice figure says how
            # much of the target the voice alone resolves -- and therefore how high a conditioned model could
            # reach, which is the number the flow's rho should be judged against.
            if int(x["voice"]) == int(y["voice"]):
                same_voice_token.append(token_score)
                same_voice_frame.append(frame_score)
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
if same_voice_token:
    print(f"SAME TEXT, SAME voice       : token rho {mean(same_voice_token):+.3f} "
          f"(n={len(same_voice_token)}) | frame rho {mean(same_voice_frame):+.3f}")
    print("  ^ the flow is given the voice as well as the text, so THIS is the ceiling that applies to it:")
    print("    whatever remains here is the part of the acoustics that neither the text nor the voice fixes.")
else:
    print("SAME TEXT, SAME voice       : no such pairs in this cache (each text was rendered once per voice)")
print(f"\nmodel's validation token rho at 10 000 steps: +0.402 (item split) / +0.371 (text-grouped split)")
print("\n  MEASURED RESULT (round 60), and it refutes the hypothesis this script was written to test:")
print("  the model's validation rho is HIGHER than the same-text cross-voice agreement.  So that agreement")
print("  is NOT an upper bound on the target -- the model is given the *voice*, and these pairs differ in")
print("  voice by construction, so their agreement is depressed by the very factor the model is told about.")
print("  A text+voice -> latent model is therefore not capped at 0.25, and the honest ceiling for this")
print("  target remains unmeasured.  What the script does establish is that the target is not the wall the")
print("  ~0.37 plateau suggested, and that the corpus repeats 332 texts across voices -- which is why the")
print("  evaluation split had to become text-grouped.")
