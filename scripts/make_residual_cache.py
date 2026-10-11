"""Build a cache whose frame targets are `x1 - token_route(ids)`.

The residual route established in rounds 116-117 and given a measured floor of 0.406:

    plan        = token_route(ids)                    # rho 0.406 frames -- the only thing here that learns
    x1_residual = x1 - plan                           # what the flow is asked to model instead
    sample      = plan + flow_field(x0 -> residual)   # starts at 0.406; the field can only add

Writing the residual as a *cache* means the existing flow training applies unchanged -- no model changes, no new
loss terms, no new sampling path.  The plan is not stored: it is deterministic from the token route's checkpoint, so
the evaluation recomputes it rather than carrying a second copy of the corpus on disk.

    python scripts/make_residual_cache.py --cache runs/expanded_v3 \
        --token-route runs/text_continue/distill-text_step5000.pt --out runs/residual_v3 [--limit 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from parakeet.config import load_config
from parakeet.data.dataset import LatentShardDataset
from parakeet.models import build_model
from parakeet.train.common import (
    apply_checkpoint_geometry,
    derive_n_voices_from_cache,
    load_latent_norm_from_cache,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache of x1 - token_route(ids)")
    ap.add_argument("--cache", required=True, help="source corpus cache")
    ap.add_argument("--token-route", required=True, help="distill-text checkpoint providing the plan")
    ap.add_argument("--token-config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=2000, help="how many items to residualise")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fit-items", type=int, default=128,
                    help="items used to fit the per-dimension regression coefficient alpha")
    args = ap.parse_args()

    source = Path(args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.token_config)
    payload = torch.load(args.token_route, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    apply_checkpoint_geometry(cfg, state)
    cfg.n_voices = derive_n_voices_from_cache(args.cache) or 1
    plan_model = build_model(cfg)
    current = plan_model.state_dict()
    plan_model.load_state_dict({k: v for k, v in state.items()
                                if k in current and tuple(current[k].shape) == tuple(v.shape)}, strict=False)
    load_latent_norm_from_cache(plan_model, args.cache)
    plan_model.eval()

    dataset = LatentShardDataset(source, indices=list(range(args.limit)))
    meta_in = json.loads((source / "cache_meta.json").read_text(encoding="utf-8"))

    def plan_for(item):
        ids = item["ids"][None]
        mask = item.get("text_mask")
        mask = mask[None] if mask is not None else None
        voice = item.get("voice")
        voice = voice.reshape(1) if voice is not None else None
        side = plan_model.text_side(ids, mask, voice)
        planned = plan_model.decoder_latent_from_tokens(
            side["latent_token"], item.get("durations"), f0=side.get("f0"), energy=side.get("energy"),
        )
        return planned[0] if isinstance(planned, tuple) else planned

    # Fit the per-dimension regression coefficient before writing anything.  Subtracting the plan *unscaled*
    # is wrong in a way that is easy to miss: energy(x - p) = energy(x) - 2cov + energy(p), so a predictor with
    # rho 0.406 and comparable variance makes the residual LARGER than the target (the trial measured 2.16x).
    # Using alpha = cov(x, p) / var(p) minimises the residual energy, and it leaves the floor's rho unchanged
    # because correlation is scale-invariant -- so the sample still starts at 0.406, just from a
    # well-conditioned residual.
    fit_n = min(args.fit_items, len(dataset))
    sum_xp = sum_pp = None
    with torch.no_grad():
        for index in range(fit_n):
            item = dataset[index]
            planned = plan_for(item)
            target = item["latent"]
            length = min(planned.shape[-1], target.shape[-1])
            x = target[..., :length].float()
            p = planned[0, :, :length].float()
            xp = (x * p).sum(dim=-1)
            pp = (p * p).sum(dim=-1)
            sum_xp = xp if sum_xp is None else sum_xp + xp
            sum_pp = pp if sum_pp is None else sum_pp + pp
    alpha = (sum_xp / sum_pp.clamp_min(1e-8)).clamp(0.0, 2.0)
    print(f"fitted alpha on {fit_n} items: mean {float(alpha.mean()):.3f} "
          f"range [{float(alpha.min()):.3f}, {float(alpha.max()):.3f}]", flush=True)

    items_out = []
    total_before = total_after = 0.0
    with torch.no_grad():
        for start in range(0, len(dataset), args.batch_size):
            for index in range(start, min(start + args.batch_size, len(dataset))):
                item = dataset[index]
                planned = plan_for(item)
                target = item["latent"]
                length = min(planned.shape[-1], target.shape[-1])
                residual = target[..., :length] - alpha.view(-1, 1) * planned[0, :, :length]
                total_before += float(target[..., :length].pow(2).sum())
                total_after += float(residual.pow(2).sum())
                record = dict(item)
                record["latent"] = residual.contiguous()
                items_out.append(record)
            if (start // args.batch_size) % 20 == 0:
                print(f"  {len(items_out)}/{len(dataset)} residualised", flush=True)

    torch.save({"items": items_out}, out / "shard_00000.pt")
    (out / "index.json").write_text(json.dumps([{"path": "shard_00000.pt", "n": len(items_out)}]), "utf-8")
    meta = dict(meta_in)
    meta["residual_of"] = str(source)
    meta["residual_plan"] = str(args.token_route)
    meta["items"] = len(items_out)
    meta["residual_alpha"] = [float(v) for v in alpha]
    (out / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nwrote {len(items_out)} residual items -> {out}")
    print(f"  target energy before {total_before:.0f}, after {total_after:.0f} "
          f"({100.0 * (1.0 - total_after / max(total_before, 1e-9)):.1f}% removed by the plan)")
    print("  The plan explains ~16% of the variance (rho 0.406), so the residual still carries ~84%: the field's")
    print("  job is smaller but not small, and the sample starts at 0.406 rather than at zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
