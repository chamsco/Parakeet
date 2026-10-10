"""The token bottleneck, measured with the *faithful* inverse.

The first version of this measurement collapsed each token's `rate` sub-vectors into one and then stretched
it over the token's frames. That is not the inverse the model uses: `extract_signals` builds each token as
`rate` **sub-span means** (via `subtoken_spans`), and `decoder_latent_from_tokens` expands them by putting
each sub-vector back on its own span. Collapsing first destroys exactly the within-token detail the
representation is designed to keep, so the earlier 0.118 was an artefact of the probe, not a property of the
bottleneck.

This repeats the oracle with the real geometry, so the number is a ceiling for the token route rather than
for my approximation of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from parakeet.data.dataset import LatentShardDataset
from parakeet.models.duration import subtoken_spans

CACHE = sys.argv[1] if len(sys.argv) > 1 else "runs/mixed_v2/latent_cache"
ITEMS = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def correlate(a: torch.Tensor, b: torch.Tensor) -> float:
    length = min(a.shape[-1], b.shape[-1])
    a, b = a[:, :length], b[:, :length]
    scores = []
    for dim in range(a.shape[0]):
        x, y = a[dim], b[dim]
        if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
            continue
        scores.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
    return sum(scores) / max(1, len(scores))


def faithful_expand(token: torch.Tensor, durations: torch.Tensor, frames: int, width: int) -> torch.Tensor:
    """`latent_token` -> frame latent using the same sub-span geometry the model uses."""
    rate = max(1, token.shape[-1] // width)
    geometry = subtoken_spans(durations, rate, frames)
    # `extract_signals` builds a token by `torch.cat`ing `rate` width-vectors, so the layout is
    # (rate, width) -- reshaping as (width, rate) silently transposes the sub-vectors, which is how an
    # earlier version of this probe produced a number that was pure artefact
    pieces = token.reshape(token.shape[0], rate, width)
    out = torch.zeros(width, frames)
    filled = torch.zeros(frames, dtype=torch.bool)
    for index, spans in enumerate(geometry):
        if index >= pieces.shape[0]:
            break
        for sub, (a, b) in enumerate(spans[:rate]):
            a = max(0, min(int(a), frames))
            b = max(a + 1, min(int(b), frames))
            out[:, a:b] = pieces[index, sub, :, None]
            filled[a:b] = True
    if not bool(filled.all()):  # unfilled frames get the nearest filled value, never zeros
        last = out[:, 0]
        for frame in range(frames):
            if filled[frame]:
                last = out[:, frame]
            else:
                out[:, frame] = last
    return out


dataset = LatentShardDataset(CACHE, indices=list(range(ITEMS)))
oracles, wrongs = [], []
print(f"{'item':>5s} {'faithful oracle rho':>20s} {'wrong utterance':>16s}")
for index in range(len(dataset)):
    item = dataset[index]
    latent = item["latent"]
    token = item["latent_token"]
    durations = item["durations"]
    frames = int(latent.shape[-1])
    width = int(latent.shape[0])
    oracle = faithful_expand(token, durations, frames, width)
    other = dataset[(index + 1) % len(dataset)]
    wrong = faithful_expand(other["latent_token"], other["durations"], frames, width)
    oracle_rho, wrong_rho = correlate(oracle, latent), correlate(wrong, latent)
    oracles.append(oracle_rho)
    wrongs.append(wrong_rho)
    print(f"{index:5d} {oracle_rho:20.4f} {wrong_rho:16.4f}")

mean = lambda v: sum(v) / max(1, len(v))
print(f"\nmean faithful oracle rho: {mean(oracles):+.4f}   (earlier crude inverse: +0.1175)")
print(f"mean with a different utterance's tokens: {mean(wrongs):+.4f}")
print("\n  This is the ceiling for any text -> token -> frame route, with the model's own geometry.")
