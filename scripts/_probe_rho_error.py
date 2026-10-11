"""Dev: does a rho interval straddle zero on the checkpoint that produced two contradictory readings?"""

import sys

sys.path.insert(0, ".")

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

cfg = load_config("configs/parakeet_flow_plan.yaml")
payload = torch.load("runs/ab_fresh/flow_step2000.pt", map_location="cpu", weights_only=False)
state = (payload.get("ema") or {}).get("shadow") or payload["model"]
apply_checkpoint_geometry(cfg, state)
cfg.n_voices = derive_n_voices_from_cache("runs/expanded_v3") or 1
model = build_model(cfg)
current = model.state_dict()
model.load_state_dict(
    {k: v for k, v in state.items() if k in current and tuple(current[k].shape) == tuple(v.shape)},
    strict=False,
)
load_latent_norm_from_cache(model, "runs/expanded_v3")

dataset = LatentShardDataset("runs/expanded_v3", indices=list(range(64)))
result = sampled_rho_with_error(model, cfg, dataset, items=16, seeds=(0, 1, 2, 3, 4))
print(f"mean {result['mean']:+.4f} | std {result['std']:.4f} | "
      f"range [{result['min']:+.4f}, {result['max']:+.4f}] | seeds {result['n_seeds']}")
print(f"per-seed: {[round(s, 4) for s in result['per_seed']]}")
print(f"straddles zero: {result['straddles_zero']}")
print()
print("arm A reported +0.0882 at this step and the continuation from it measured -0.0277.")
print("If the interval straddles zero, both readings are draws from the same noise and neither is evidence.")
