"""Measure the rendition reliability with a real sample size, from near-identical-text pairs.

Pairs are the SAME WORDS with different punctuation, rendered by the SAME voice (the prompt builder lays them out so
the generator's round-robin lands both members on one voice). So the words -- and therefore the phonetic content --
are identical, and the difference between the two latents is rendition.

That makes the measured agreement a LOWER bound on a true repeat-generation reliability: punctuation changes prosody
systematically rather than independently, so the rendition difference here is at least as large as between two
independent generations, which pushes agreement DOWN. A conservative estimate is what an argument about headroom
wants.

Two alignments are reported, because the members pace differently:

  * zero shift -- a lower bound: the same frame index in both, which penalises any pacing difference;
  * best shift within a small window -- an upper bound: what the agreement would be if pacing were irrelevant.

The recogniser's calibration (round 37) is the bar: decoded latents are read perfectly at rho >= 0.75 and fail at
0.60. So the question is whether the conservative reliability implies a ceiling at or above 0.60.

    python scripts/rendition_reliability.py --cache runs/rendition_v1_cache --pairs 110
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from parakeet.data.dataset import LatentShardDataset


def per_dim_rho(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-dimension correlation over the overlapping frames."""
    length = min(a.shape[1], b.shape[1])
    scores = []
    for dim in range(a.shape[0]):
        x, y = a[dim, :length].float(), b[dim, :length].float()
        if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
            continue
        scores.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
    return sum(scores) / max(1, len(scores))


def best_shift_rho(a: torch.Tensor, b: torch.Tensor, window: int) -> tuple[float, int]:
    best, best_shift = -1.0, 0
    for shift in range(-window, window + 1):
        if shift >= 0:
            left, right = a[:, shift:], b[:, :b.shape[1] - shift]
        else:
            left, right = a[:, :a.shape[1] + shift], b[:, -shift:]
        if min(left.shape[1], right.shape[1]) < 20:
            continue
        value = per_dim_rho(left, right)
        if value > best:
            best, best_shift = value, shift
    return best, best_shift


def bootstrap_ci(values: list[float], draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    means = []
    for _ in range(draws):
        index = torch.randint(0, len(values), (len(values),), generator=generator)
        means.append(statistics.mean(values[int(i)] for i in index))
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Rendition reliability from near-identical-text pairs")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pairs", type=int, default=110, help="the second block starts at this index")
    ap.add_argument("--shift-window", type=int, default=12, help="frames; 12 at 93.75 Hz is 128 ms")
    ap.add_argument("--out", default="runs/rendition_reliability.json")
    args = ap.parse_args()

    dataset = LatentShardDataset(args.cache, apply_voice_norm=False)
    total = len(dataset)
    if total != 2 * args.pairs:
        print(f"note: cache holds {total} items against 2 x {args.pairs} = {2 * args.pairs} expected; "
              f"using min(pairs, total//2)")
    pairs = min(args.pairs, total // 2)

    zero, best, shifts, lengths = [], [], [], []
    for index in range(pairs):
        first = dataset[index]["latent"]
        second = dataset[index + args.pairs]["latent"]
        lengths.append(abs(first.shape[1] - second.shape[1]) / 93.75)
        zero.append(per_dim_rho(first, second))
        value, shift = best_shift_rho(first, second, args.shift_window)
        if value > -1:
            best.append(value)
            shifts.append(shift)

    mean_zero = statistics.mean(zero)
    mean_best = statistics.mean(best) if best else float("nan")
    lo_zero, hi_zero = bootstrap_ci(zero)
    lo_best, hi_best = bootstrap_ci(best) if best else (float("nan"), float("nan"))

    report = {
        "pairs": pairs,
        "zero_shift": {"mean": mean_zero, "ci95": [lo_zero, hi_zero],
                       "ceiling_sqrt": math.sqrt(max(mean_zero, 0.0))},
        "best_shift": {"mean": mean_best, "ci95": [lo_best, hi_best],
                       "ceiling_sqrt": math.sqrt(max(mean_best, 0.0)) if best else None,
                       "median_shift_frames": statistics.median(shifts) if shifts else None},
        "length_mismatch_seconds": {"mean": statistics.mean(lengths), "max": max(lengths)},
        "bar": {"recogniser_fails": 0.60, "reads_perfectly": 0.75},
    }
    print(f"rendition reliability over {pairs} same-voice near-identical-text pairs")
    print(f"  zero shift   rho {mean_zero:+.4f}  95% CI [{lo_zero:+.4f}, {hi_zero:+.4f}]"
          f"  -> ceiling {math.sqrt(max(mean_zero, 0.0)):.3f}")
    if best:
        print(f"  best shift   rho {mean_best:+.4f}  95% CI [{lo_best:+.4f}, {hi_best:+.4f}]"
              f"  -> ceiling {math.sqrt(max(mean_best, 0.0)):.3f}")
        print(f"  median |shift| {statistics.median(shifts):+.0f} frames")
    print(f"  length mismatch: mean {statistics.mean(lengths):.3f} s, max {max(lengths):.3f} s")
    print(f"\n  compare with the n=4 estimate this replaces: rho 0.445, 95% CI [-0.902, +0.985]")
    print(f"  the recogniser fails at rho 0.60, so a ceiling at or above 0.60 keeps the deterministic route open")
    ceiling = math.sqrt(max(mean_best, 0.0)) if best else math.sqrt(max(mean_zero, 0.0))
    verdict = ("OPEN: the conservative reliability implies a ceiling at or above the recogniser's fail point, so "
               "the deterministic route has room by this measure too." if ceiling >= 0.60 else
               "CLOSED by this measure: even the upper-bound alignment leaves the ceiling below the fail point.")
    report["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
