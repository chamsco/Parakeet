"""What is the plan worth, in frame-ρ terms?

The residual form (`flow.plan_residual`) makes the flow model `x1 - plan` instead of `x1`, so the *samples* inherit
the plan's text conditioning whatever the velocity field does. That is the one structural change this project has
implemented but never judged by the metric that applies to a sampler: the gate is the wrong readout for it (moving
the text signal into the plan *reduces* the field's loss dependence by construction), and the earlier tests read it
by ρ on 2 items, which round 64-65 showed cannot discriminate anything.

Before spending CPU on training it, its ceiling is free to measure: unfold the plan to frames and correlate it with
the target. If that is near zero the whole route is dead; if it is substantial, the residual form has a floor worth
building on, and it is a floor the field can only add to.

    python scripts/plan_ceiling.py --run runs/flow_v3_long --checkpoint flow_step4000.pt --cache runs/expanded_v3
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


import torch

from parakeet.config import load_config
from parakeet.data.dataset import LatentShardDataset
from parakeet.models import build_model
from parakeet.models.flow import unfold_time
from parakeet.train.common import (
    apply_checkpoint_geometry,
    derive_n_voices_from_cache,
    load_latent_norm_from_cache,
)


def frame_rho(a: torch.Tensor, b: torch.Tensor) -> float:
    length = min(a.shape[-1], b.shape[-1])
    scores = []
    for dim in range(a.shape[0]):
        x, y = a[dim, :length].float(), b[dim, :length].float()
        if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
            continue
        scores.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
    return sum(scores) / max(1, len(scores))


def token_rho(a: torch.Tensor, b: torch.Tensor) -> float:
    length = min(a.shape[0], b.shape[0])
    scores = []
    for dim in range(a.shape[1]):
        x, y = a[:length, dim].float(), b[:length, dim].float()
        if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
            continue
        scores.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
    return sum(scores) / max(1, len(scores))


def main() -> int:
    ap = argparse.ArgumentParser(description="The plan's own frame/token rho against the target")
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="flow_last.pt")
    ap.add_argument("--config", default="configs/parakeet_flow_plan.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--items", type=int, default=24)
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

    if not getattr(model, "use_plan", False):
        raise SystemExit("this checkpoint has no plan head (flow.use_plan false): nothing to measure")
    if not hasattr(model, "plan_from_tokens"):
        raise SystemExit("model has no plan_from_tokens: the plan's ceiling cannot be read from this build")

    dataset = LatentShardDataset(args.cache, indices=list(range(args.items)))
    frames, tokens = [], []
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            ids = item["ids"][None]
            mask = item.get("text_mask")
            mask = mask[None] if mask is not None else None
            voice = item.get("voice")
            voice = voice.reshape(1) if voice is not None else None
            plan = model.plan_from_tokens(model.text(ids, mask))
            tokens.append(token_rho(plan[0], item["latent_token"]))
            if hasattr(model, "upsample_plan"):
                frames.append(frame_rho(model.upsample_plan(plan, item["latent"].shape[-1])[0], item["latent"]))
            else:
                unfolded = unfold_time(plan, cfg.flow.compress, t_out=item["latent"].shape[-1])
                frames.append(frame_rho(unfolded[0], item["latent"]))

    mean = lambda v: sum(v) / len(v)  # noqa: E731
    report = {"items": len(dataset), "token_rho": mean(tokens), "frame_rho": mean(frames),
              "frame_rho_median": statistics.median(frames), "frame_rho_max": max(frames)}
    print(f"checkpoint {path} (step {payload.get('step')}) | {len(dataset)} items")
    print(f"  plan vs target tokens : rho {report['token_rho']:+.4f}")
    print(f"  plan vs target frames : rho {report['frame_rho']:+.4f} "
          f"(median {report['frame_rho_median']:+.4f}, max {report['frame_rho_max']:+.4f})")
    print("\n  This is the floor the residual form would sample at, and the field could only add to it.")
    if report["frame_rho"] > 0.20:
        print("  -> substantial: the residual route has a real floor, and a training run is worth its CPU.")
    else:
        print("  -> near zero: the residual form inherits nothing, so it is not the structural change that was "
              "missing and should not be trained.")

    Path("runs/plan_ceiling.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
