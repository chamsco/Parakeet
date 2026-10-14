"""ρ against classifier-free guidance weight, with an error bar.

Two things this tool exists to do:

* **CFG sweep.** The standard remedy for weak conditioning. The field already has the text at every step and CFG
  is implemented (config default 1.5), so the question is: if the text direction is merely under-weighted,
  extrapolating along it should raise the correlation with the target. Round 99 answered no -- ρ falls
  monotonically from 1.0 to 8.0 while the output gets 2.5× louder -- and this is the tool that measured it.
* **ρ with a spread, not a point.** Round 100 found a single reading carries no information at these magnitudes:
  arm A reported +0.0882 at step 2000 and continuing from that same checkpoint measured −0.0277. Five probe seeds
  give a mean and a range, and `straddles_zero` says whether the number is distinguishable from nothing at all --
  which, for every flow checkpoint in this project, it is not.

    python scripts/cfg_rho_sweep.py --run runs/ab_fresh --checkpoint flow_step2000.pt \
        --cache runs/expanded_v3 [--weights 1,1.5,2,4,8] [--seeds 0,1,2,3,4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


import torch

from parakeet.config import load_config
from parakeet.data.dataset import LatentShardDataset
from parakeet.eval.gates import sampled_rho_with_error
from parakeet.models import build_model
from parakeet.train.common import (
    apply_checkpoint_geometry,
    derive_n_voices_from_cache,
    load_latent_norm_from_cache,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="rho vs guidance weight, with a seed spread")
    ap.add_argument("--run", required=True, help="directory holding the checkpoint")
    ap.add_argument("--checkpoint", default="flow_last.pt")
    ap.add_argument("--config", default="configs/parakeet_flow_plan.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--items", type=int, default=64)
    ap.add_argument("--probe-items", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=4, help="NFE")
    ap.add_argument("--weights", default="1,1.5,2,4,8")
    args = ap.parse_args()

    path = Path(args.run) / args.checkpoint
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path}")

    cfg = load_config(args.config)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    apply_checkpoint_geometry(cfg, state)
    cfg.n_voices = derive_n_voices_from_cache(args.cache) or 1
    model = build_model(cfg)
    current = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items()
                           if k in current and tuple(current[k].shape) == tuple(v.shape)}, strict=False)
    load_latent_norm_from_cache(model, args.cache)

    dataset = LatentShardDataset(args.cache, indices=list(range(args.items)))
    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    print(f"checkpoint {path} (step {payload.get('step')}) | NFE {args.steps} | {len(dataset)} items "
          f"| seeds {seeds}")
    print(f"{'cfg':>6s} {'rho mean':>10s} {'std':>8s} {'range':>20s}  verdict")

    rows = []
    for weight in weights:
        cfg.flow.cfg_scale = weight
        result = sampled_rho_with_error(model, cfg, dataset, items=args.probe_items, seeds=seeds,
                                        nfe=args.steps)
        verdict = "indistinguishable from zero" if result["straddles_zero"] else "clear of zero"
        rows.append({"cfg": weight,
                     **{k: result[k] for k in ("mean", "std", "min", "max", "n_seeds")},
                     "straddles_zero": result["straddles_zero"]})
        print(f"{weight:6.2f} {result['mean']:+10.4f} {result['std']:8.4f} "
              f"[{result['min']:+.4f}, {result['max']:+.4f}]  {verdict}", flush=True)

    Path("runs/cfg_rho_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    best = max(rows, key=lambda r: r["mean"])
    print(f"\nbest mean rho {best['mean']:+.4f} at cfg {best['cfg']} "
          f"(target 0.75 to be intelligible; every value here is inside the seed spread)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
