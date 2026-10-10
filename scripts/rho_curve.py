"""rho against training steps, on both the trained items and items never trained on.

The flow's progress metric is the correlation between its sampled latents and the teacher's (rho >= 0.75 is
what the recogniser needs, round 37).  A single end-of-run number cannot say whether rho is still climbing
-- and with GPU time being the scarce resource, that is the question that decides how big a run to buy.

This walks a run directory's checkpoints and reports rho for the trained items and for a held-out range,
so the curve can be extrapolated instead of guessed at.

    python scripts/rho_curve.py --run runs/flow_general --items 200 --holdout 32
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from parakeet.data.dataset import LatentShardDataset
from overfit_flow import sampled_rho  # the same measurement the training harness reports


def step_of(path: Path) -> int:
    match = re.search(r"step(\d+)", path.name)
    return int(match.group(1)) if match else 10**9


def main() -> int:
    ap = argparse.ArgumentParser(description="rho vs steps, trained and held out")
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", default="runs/mixed_v2/latent_cache")
    ap.add_argument("--config", default="configs/parakeet_flow.yaml")
    ap.add_argument("--items", type=int, default=200, help="trained range is [0, items)")
    ap.add_argument("--holdout", type=int, default=32, help="held-out range is [items, items+holdout)")
    ap.add_argument("--probe-items", type=int, default=12, help="how many of each range to score")
    args = ap.parse_args()

    from parakeet.config import load_config
    from parakeet.models import build_model
    from parakeet.train.common import (
        apply_checkpoint_geometry,
        derive_n_voices_from_cache,
        load_latent_norm_from_cache,
    )

    checkpoints = sorted(Path(args.run).glob("flow_*.pt"), key=step_of)
    if not checkpoints:
        print(f"no checkpoints in {args.run}")
        return 2

    trained = LatentShardDataset(args.cache, indices=list(range(args.items)))
    holdout = LatentShardDataset(
        args.cache, indices=list(range(args.items, args.items + args.holdout))
    )

    rows = []
    print(f"{'step':>7s} {'rho trained':>12s} {'rho held-out':>13s}")
    for checkpoint in checkpoints:
        cfg = load_config(args.config)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = (payload.get("ema") or {}).get("shadow") or payload["model"]
        apply_checkpoint_geometry(cfg, state)
        cfg.n_voices = derive_n_voices_from_cache(args.cache)
        model = build_model(cfg)
        current = model.state_dict()
        model.load_state_dict({k: v for k, v in state.items()
                               if k in current and tuple(current[k].shape) == tuple(v.shape)},
                              strict=False)
        load_latent_norm_from_cache(model, args.cache)
        model.eval()
        step = step_of(checkpoint)
        train_rho = sampled_rho(model, cfg, trained, args.probe_items)
        hold_rho = sampled_rho(model, cfg, holdout, args.probe_items) if len(holdout) else float("nan")
        rows.append({"step": step, "rho_trained": train_rho, "rho_holdout": hold_rho})
        print(f"{step:7d} {train_rho:12.4f} {hold_rho:13.4f}")

    Path("runs/rho_curve.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\ntarget 0.75; written to runs/rho_curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
