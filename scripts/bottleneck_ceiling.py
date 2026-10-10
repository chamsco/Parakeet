"""What is the ceiling?  How much of the frame latent survives the token-latent bottleneck?

The intelligibility threshold is rho >= 0.75 against the *frame* latent (round 37).  Every architecture
tried here goes through a text-conditioned bottleneck: text -> token latents -> frame latents.  If the
token latents do not carry enough of the frame latent to begin with, then no amount of training reaches
0.75 and the representation is the blocker, not the optimisation.

So measure the *oracle*: give the pipeline the true token latents (no prediction involved), spread them
over the utterance's frames, and correlate against the true frame latents.  That is an upper bound for any
text -> token -> frame route, and it costs seconds.

Controls, because an oracle number alone is not interpretable:
  * a *different* utterance's token latents (the floor: what you get with no information at all);
  * the same measurement on a randomly-shuffled token order (destroys ordering, keeps statistics).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from parakeet.data.dataset import LatentShardDataset

CACHE = sys.argv[1] if len(sys.argv) > 1 else "runs/mixed_v2/latent_cache"
ITEMS = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def correlate(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-dimension correlation between two (C, T) latents."""
    length = min(a.shape[-1], b.shape[-1])
    a, b = a[:, :length], b[:, :length]
    scores = []
    for dim in range(a.shape[0]):
        x, y = a[dim], b[dim]
        if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
            continue
        scores.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
    return sum(scores) / max(1, len(scores))


def spread_tokens(token: torch.Tensor, durations: torch.Tensor, frames: int, width: int) -> torch.Tensor:
    """`(S, S*width)` token latents -> `(width, frames)` by repeating each token over its frames."""
    unfolded = token.reshape(token.shape[0], width, -1).mean(dim=-1)      # (S, width) pooled per token
    pieces = []
    for index in range(unfolded.shape[0]):
        repeat = max(1, int(durations[index].item()))
        pieces.append(unfolded[index].unsqueeze(-1).repeat(1, repeat))
    stacked = torch.cat(pieces, dim=-1)
    if stacked.shape[-1] >= frames:
        return stacked[:, :frames]
    pad = frames - stacked.shape[-1]
    return torch.cat([stacked, stacked[:, -1:].repeat(1, pad)], dim=-1)


dataset = LatentShardDataset(CACHE, indices=list(range(ITEMS)))
rows = []
for index in range(len(dataset)):
    item = dataset[index]
    latent = item["latent"]
    token = item["latent_token"]
    durations = item["durations"]
    frames = int(latent.shape[-1])
    width = int(latent.shape[0])
    oracle = spread_tokens(token, durations, frames, width)
    rows.append((index, correlate(oracle, latent), token, durations, latent, frames, width))

print(f"{'item':>5s} {'oracle rho':>11s} {'wrong utterance':>16s} {'shuffled tokens':>16s}")
oracles, wrongs, shuffles = [], [], []
for index, oracle_rho, token, durations, latent, frames, width in rows:
    other = rows[(index + 1) % len(rows)]
    wrong = spread_tokens(other[2], other[3], frames, width)
    generator = torch.Generator().manual_seed(0)
    shuffled = spread_tokens(token[torch.randperm(token.shape[0], generator=generator)], durations,
                             frames, width)
    wrong_rho = correlate(wrong, latent)
    shuffled_rho = correlate(shuffled, latent)
    oracles.append(oracle_rho)
    wrongs.append(wrong_rho)
    shuffles.append(shuffled_rho)
    print(f"{index:5d} {oracle_rho:11.4f} {wrong_rho:16.4f} {shuffled_rho:16.4f}")

mean = lambda v: sum(v) / max(1, len(v))
print(f"\nmean oracle rho (true token latents -> frame latents): {mean(oracles):+.4f}")
print(f"mean with a DIFFERENT utterance's tokens            : {mean(wrongs):+.4f}")
print(f"mean with the token order SHUFFLED                  : {mean(shuffles):+.4f}")
print("\n  The oracle is the ceiling for any text -> token -> frame route.  If it is far below the 0.75")
print("  intelligibility threshold, the representation -- not the training -- is the blocker.")
