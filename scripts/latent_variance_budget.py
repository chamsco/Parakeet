"""How much of the acoustic target is predictable without the text at all?

This is the quantity behind the one untested hypothesis left (`docs/PLAN.md`, Run 3) and it has never been
computed. The project has correlations -- same-text agreement 0.171 cross-voice, 0.445 same-voice -- but no
variance decomposition, and the decomposition is what says how large the text's share of the problem can possibly
be.

Predictors are added in the order a text-blind model would learn them, and the R^2 after each is the fraction of
the target's per-dimension variance already explained:

  1. nothing (the zero predictor, by definition 0)
  2. the global per-dimension mean                 -- what the signal "looks like" on average
  3. + the per-voice per-dimension mean            -- who is speaking
  4. + the per-utterance per-dimension mean        -- this recording's own DC level
  5. + a per-dimension AR(1) fit to the utterance  -- the temporal smoothness the flow learns first

Whatever survives step 5 is the part the text has to supply. If it is small, the flow's 2-7 % text dependence is
not a training failure but a statement about the target; if it is large, the text's share is there to be learned
and the failure is elsewhere.

    python scripts/latent_variance_budget.py --cache runs/expanded_v3 [--items 600]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


import torch

from parakeet.data.dataset import LatentShardDataset


def main() -> int:
    ap = argparse.ArgumentParser(description="Variance budget of the frame-latent target")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--items", type=int, default=600, help="how many utterances to sample")
    ap.add_argument("--max-frames", type=int, default=400, help="cap per utterance for speed")
    ap.add_argument("--no-voice-norm", action="store_true",
                    help="read RAW latents: per-voice normalisation changes what the per-voice term can explain, so comparisons between caches must use the same state")
    ap.add_argument("--compress", type=int, default=6,
                    help="temporal fold the flow applies (flow.compress); the flow models the folded form")
    args = ap.parse_args()
    compress = args.compress

    dataset = LatentShardDataset(args.cache, indices=list(range(min(args.items, len(LatentShardDataset(args.cache))))),
                                apply_voice_norm=not args.no_voice_norm)
    items = []
    for index in range(len(dataset)):
        item = dataset[index]
        latent = item["latent"][:, : args.max_frames].float()
        if latent.shape[-1] >= 8:
            items.append((int(item["voice"]), latent))
    if not items:
        raise SystemExit("no usable items")

    total_energy = sum(float(latent.pow(2).sum()) for _, latent in items)
    global_mean = sum(latent.mean(dim=-1) for _, latent in items) / len(items)          # (24,)
    per_voice: dict = defaultdict(list)
    for voice, latent in items:
        per_voice[voice].append(latent.mean(dim=-1))
    voice_mean = {voice: sum(vs) / len(vs) for voice, vs in per_voice.items()}

    def residual_energy(predict) -> float:
        return sum(float((latent - predict(voice, latent)).pow(2).sum()) for voice, latent in items)

    # 1. zero predictor: explained = 0 by construction
    steps = [("nothing", 0.0)]
    # 2. global mean
    r2_global = 1.0 - residual_energy(lambda v, x: global_mean[:, None]) / total_energy
    steps.append(("global per-dim mean", r2_global))
    # 3. per-voice mean
    r2_voice = 1.0 - residual_energy(lambda v, x: voice_mean[v][:, None]) / total_energy
    steps.append(("+ per-voice mean", r2_voice))
    # 4. per-utterance DC
    r2_utt = 1.0 - residual_energy(lambda v, x: x.mean(dim=-1, keepdim=True).expand_as(x)) / total_energy
    steps.append(("+ per-utterance DC", r2_utt))

    # 5. per-utterance AR(1) per dimension: x_t = a + b * x_{t-1}
    def ar1(x: torch.Tensor) -> torch.Tensor:
        previous, current = x[:, :-1], x[:, 1:]
        prev_mean = previous.mean(dim=-1, keepdim=True)
        cur_mean = current.mean(dim=-1, keepdim=True)
        covariance = ((previous - prev_mean) * (current - cur_mean)).mean(dim=-1, keepdim=True)
        variance = (previous - prev_mean).pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8)
        slope = covariance / variance
        prediction = cur_mean + slope * (previous - prev_mean)
        return torch.cat([x[:, :1], prediction], dim=-1)

    r2_ar = 1.0 - residual_energy(lambda v, x: ar1(x)) / total_energy
    steps.append(("+ AR(1) smoothness", r2_ar))

    print(f"cache {args.cache}: {len(items)} utterances, {total_energy:.0f} total energy")
    print(f"\n{'predictor':>24s} {'R^2':>8s} {'added':>8s}")
    previous = 0.0
    for label, value in steps:
        print(f"{label:>24s} {value:8.3f} {value - previous:+8.3f}")
        previous = value
    residual = 1.0 - r2_ar
    print(f"\nresidual after all five: {residual:.3f} -- the share of the target the text must supply")
    if residual < 0.15:
        print("  -> the target is almost entirely predictable without text: a low text dependence is expected,")
        print("     and the flow's 2-7% is close to what is available.")
    else:
        print("  -> there is substantially more variance left than the flow's text dependence accounts for;")
        print("     the text's share exists to be learned, and the failure is not the target's nature.")

    # The fold matters: the flow models compress:6 folded latents, where the per-dimension AR(1) that dominates
    # the unfolded form is redistributed across axes.  The number that applies to the flow is the folded one.
    from parakeet.models.flow import fold_time

    folded = [(voice, fold_time(latent[None], compress)[0]) for voice, latent in items]
    folded = [(voice, latent) for voice, latent in folded if latent.shape[-1] >= 8]
    if folded:
        folded_energy = sum(float(latent.pow(2).sum()) for _, latent in folded)
        global_mean_f = sum(latent.mean(dim=-1) for _, latent in folded) / len(folded)
        per_voice_f: dict = defaultdict(list)
        for voice, latent in folded:
            per_voice_f[voice].append(latent.mean(dim=-1))
        voice_mean_f = {voice: sum(vs) / len(vs) for voice, vs in per_voice_f.items()}

        def residual_energy_f(predict) -> float:
            return sum(float((latent - predict(voice, latent)).pow(2).sum()) for voice, latent in folded)

        def ar1_f(x: torch.Tensor) -> torch.Tensor:
            previous, current = x[:, :-1], x[:, 1:]
            prev_mean = previous.mean(dim=-1, keepdim=True)
            cur_mean = current.mean(dim=-1, keepdim=True)
            covariance = ((previous - prev_mean) * (current - cur_mean)).mean(dim=-1, keepdim=True)
            variance = (previous - prev_mean).pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8)
            slope = covariance / variance
            return torch.cat([x[:, :1], cur_mean + slope * (previous - prev_mean)], dim=-1)

        r2_f_voice = 1.0 - residual_energy_f(lambda v, x: voice_mean_f[v][:, None]) / folded_energy
        r2_f_ar = 1.0 - residual_energy_f(lambda v, x: ar1_f(x)) / folded_energy
        print(f"\nON THE FOLDED TARGET the flow actually models ({folded[0][1].shape[0]} dims):")
        print(f"  per-voice mean      R^2 {r2_f_voice:6.3f}")
        print(f"  + AR(1) smoothness  R^2 {r2_f_ar:6.3f}")
        print(f"  residual for the text: {1.0 - r2_f_ar:.3f}")

        folded_report = {"dims": int(folded[0][1].shape[0]),
                         "per_voice_r2": r2_f_voice,
                         "plus_ar1_r2": r2_f_ar,
                         "residual_for_text": 1.0 - r2_f_ar}
    else:
        # Every item folded thinner than the 8-frame floor. Report the unfolded chain with folded null instead of
        # indexing folded[0] after all the measurement work has already been done.
        folded_report = None

    # The folded 144-dim target is what the flow models, so its residual is the share of the objective the text has
    # to explain. The report carries it whenever it exists.
    Path("runs/latent_variance_budget.json").write_text(
        json.dumps({"items": len(items), **{label: value for label, value in steps},
                    "residual": residual,
                    "folded": folded_report}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
