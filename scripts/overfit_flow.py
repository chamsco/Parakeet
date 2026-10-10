"""Can the flow learn the text-to-latent mapping *at all*?  Overfit a dozen utterances.

The evidence: the flow-matching loss falls steadily (2.00 -> 1.48 -> 0.89 over 1 000 steps) while the
correlation between its *sampled* latents and the teacher's stays at ~0.00-0.02.  That is the signature of
a model learning the marginal velocity field -- the easy part -- and not the text-conditioned part.

Capacity, data scale and optimisation schedule all confound the full-corpus runs, so remove them: train
on twelve utterances only, where a working architecture must be able to fit.  If rho on those same twelve
climbs toward the 0.75 that intelligibility needs, the architecture is sound and the problem is scale; if
it stays at zero while the loss falls, the conditional path is still not being learned and no amount of
compute will fix it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from parakeet.config import load_config
from parakeet.data.dataset import LatentShardBatchSource, LatentShardDataset
from parakeet.models import build_model
from parakeet.models.flow import consistency_sample, unfold_time
from parakeet.train.common import (
    apply_checkpoint_geometry,
    derive_n_voices_from_cache,
    load_latent_norm_from_cache,
)
from parakeet.train.stages import run_stage


def sampled_rho(model, cfg, dataset, items: int = 12) -> float:
    scores = []
    with torch.no_grad():
        for index in range(min(items, len(dataset))):
            item = dataset[index]
            ids = item["ids"][None]
            mask = item.get("text_mask")
            mask = mask[None] if mask is not None else None
            voice = item.get("voice")
            voice = voice.reshape(1) if voice is not None else None
            memory, memory_mask, cond = model.conditions(ids, mask, voice=voice)
            frames = int(model.predict_latent_frames(model.text(ids, mask), mask, cond).item())
            tc = model.compressed_frames(frames)
            generator = torch.Generator().manual_seed(0)
            x0 = torch.randn(1, cfg.flow.latent_dim * cfg.flow.compress, tc, generator=generator)
            x1c = consistency_sample(
                model.vf, memory, memory_mask,
                (1, cfg.flow.latent_dim * cfg.flow.compress, tc),
                steps=4, device=torch.device("cpu"), cfg_scale=cfg.flow.cfg_scale, x0=x0,
            )
            sampled = unfold_time(x1c, cfg.flow.compress, t_out=frames)
            target = item["latent"][None]
            length = min(sampled.shape[-1], target.shape[-1])
            per_dim = []
            for dim in range(sampled.shape[1]):
                x, y = sampled[0, dim, :length], target[0, dim, :length]
                if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
                    continue
                per_dim.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
            if per_dim:
                scores.append(sum(per_dim) / len(per_dim))
    return sum(scores) / max(1, len(scores)) if scores else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="Overfit the flow on a handful of utterances")
    ap.add_argument("--cache", default="runs/mixed_v2/latent_cache")
    ap.add_argument("--config", default="configs/parakeet_flow.yaml")
    ap.add_argument("--autoencoder", default="runs/ae_scaled/adversarial/autoencoder_last.pt")
    ap.add_argument("--items", type=int, default=12)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="runs/flow_overfit")
    ap.add_argument("--cross-gain", type=float, default=None,
                    help="override the cross-attention gain after building; a large value tests whether "
                         "conditioning *strength* is what limits learning the mapping")
    ap.add_argument("--tag", default="", help="suffix for the report file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = LatentShardDataset(args.cache, indices=list(range(args.items)))
    payload = torch.load(args.autoencoder, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    apply_checkpoint_geometry(cfg, state)
    # the *cache* is authoritative for the voice table: the autoencoder checkpoint predates the corpus
    # growing and carries a narrower embedding, which would make voice indices out of range
    cfg.n_voices = derive_n_voices_from_cache(args.cache) or max(
        1, max(int(dataset[i]["voice"]) for i in range(len(dataset))) + 1
    )
    model = build_model(cfg)
    current = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items()
                           if k in current and tuple(current[k].shape) == tuple(v.shape)},
                          strict=False)
    load_latent_norm_from_cache(model, args.cache)
    if args.cross_gain is not None:
        with torch.no_grad():
            for block in model.vf.blocks:
                block.cross_gain.fill_(args.cross_gain)
        print(f"cross-attention gain overridden to {args.cross_gain}")

    before = sampled_rho(model, cfg, dataset, args.items)
    print(f"before training: sampled latent rho on the {len(dataset)} training utterances = {before:+.4f}")

    source = LatentShardBatchSource(
        dataset, batch_size=args.batch_size, shuffle=True, seed=0,
        pair_references=False, self_reference=True,
    )
    # Align the schedule with the run, exactly as `train.py --steps` does: the config's warmup is 1000
    # steps, and calling `run_stage` directly with a shorter budget leaves the learning rate at ~0 for the
    # whole run -- which silently invalidates the test (the first attempt at this measured a model that
    # had barely been trained, not one that could not learn).
    cfg.train.max_steps = args.steps
    cfg.train.warmup_steps = max(10, min(cfg.train.warmup_steps, args.steps // 20))
    print(f"schedule: {args.steps} steps with {cfg.train.warmup_steps} warmup")
    logs = run_stage(
        "flow", cfg, model=model, batches=source, max_steps=args.steps, out_dir=args.out,
        device="cpu", log_fn=lambda l: print("  ", {k: round(v, 4) if isinstance(v, float) else v
                                                   for k, v in l.items()}),
    )
    after = sampled_rho(model, cfg, dataset, args.items)
    print(f"\nafter {args.steps} steps on {len(dataset)} utterances: rho = {after:+.4f} "
          f"(target 0.75 to be intelligible)")
    Path("runs/flow_overfit.json").write_text(json.dumps({
        "items": len(dataset), "steps": args.steps,
        "rho_before": before, "rho_after": after, "logs": logs,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
