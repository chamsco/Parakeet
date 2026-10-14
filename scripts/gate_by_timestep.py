"""Where in the flow's timestep does the text signal live?

The gate averages over t, so it reports a single 2-7 % for the whole trajectory. That is enough to say the field
barely uses the text, but not enough to say whether a *weighted* loss could help: if the text's contribution is
concentrated at low t -- where the coarse structure is decided -- then reweighting the objective toward low t is a
cheap intervention worth trying. If it is uniformly near zero at every t, no reweighting can invent it.

This replicates `ParakeetFlow.flow_loss` exactly (fold_time, make_xt, v_target = x1c - x0, MSE against v_pred) but
with t supplied explicitly, and reports the right-vs-shuffled difference per t bin, plus the absolute loss so the
shape of the objective is visible too.

    python scripts/gate_by_timestep.py --run runs/flow_v3_long --checkpoint flow_step4000.pt \
        --cache runs/expanded_v3 [--items 32] [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


import torch

from parakeet.config import load_config
from parakeet.data.dataset import LatentShardBatchSource, LatentShardDataset
from parakeet.models import build_model
from parakeet.models.flow import fold_time, make_xt
from parakeet.train.common import (
    apply_checkpoint_geometry,
    derive_n_voices_from_cache,
    load_latent_norm_from_cache,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Flow-term text dependence per timestep bin")
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="flow_last.pt")
    ap.add_argument("--config", default="configs/parakeet_flow_plan.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--bins", default="0.05,0.2,0.4,0.6,0.8,0.95")
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
    model.eval()

    dataset = LatentShardDataset(args.cache, indices=list(range(args.items)))
    source = LatentShardBatchSource(dataset, batch_size=args.batch_size, shuffle=True, seed=0,
                                    self_reference=True, pair_references=False)
    timesteps = [float(x) for x in args.bins.split(",") if x.strip()]
    key = cfg.flow.compress
    print(f"checkpoint {path} (step {payload.get('step')}) | compress {key} | "
          f"{len(dataset)} items | {args.repeats} repeats per bin")

    def flow_term(ids, mask, batch, t_value: float) -> float:
        """One batch's flow-matching loss at an explicit t, matching flow_loss without the plan term."""
        latent = batch["latent"]
        x1c = fold_time(latent, key)
        generator = torch.Generator().manual_seed(1234)          # same x0 for right and shuffled
        x0 = torch.randn(x1c.shape, generator=generator)
        t = torch.full((x1c.shape[0],), t_value)
        x_t = make_xt(x1c, x0, t)
        memory, memory_mask, _ = model.conditions(ids, mask, voice=batch.get("voice"))
        with torch.no_grad():
            v_pred = model.vf(x_t, t, memory, memory_mask)
        return float((v_pred - (x1c - x0)).pow(2).mean())

    print(f"\n{'t':>6s} {'loss right':>12s} {'loss shuffled':>14s} {'difference':>12s} {'share':>8s}")
    rows = []
    for t_value in timesteps:
        right, shuffled = [], []
        for _ in range(args.repeats):
            batch = source()
            ids, mask = batch["ids"], batch.get("text_mask")
            right.append(flow_term(ids, mask, batch, t_value))
            order = torch.roll(torch.arange(ids.shape[0]), shifts=1)
            shuffled.append(flow_term(ids[order], None if mask is None else mask[order], batch, t_value))
        mean = lambda v: sum(v) / len(v)  # noqa: E731
        r, s = mean(right), mean(shuffled)
        share = 100.0 * (s - r) / max(abs(r), 1e-9)
        rows.append({"t": t_value, "right": r, "shuffled": s, "share": share})
        print(f"{t_value:6.2f} {r:12.4f} {s:14.4f} {s - r:+12.4f} {share:+7.1f}%")

    best = max(rows, key=lambda row: row["share"])
    print(f"\nlargest text share {best['share']:+.1f}% at t={best['t']}")
    if best["share"] < 5.0:
        print("  -> the text contributes ~nothing at EVERY timestep: no loss reweighting can create it, and the "
              "next intervention has to change what the field is asked to do, not how hard it is pushed")
    else:
        print("  -> the text contribution is concentrated somewhere in the trajectory; a loss weighted toward "
              "that region is a cheap intervention worth testing")

    Path("runs/gate_by_timestep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
