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
    ap.add_argument("--lr", type=float, default=None,
                    help="override the learning rate; a memorisation test wants a bigger one than a "
                         "full-corpus run")
    ap.add_argument("--crop-frames", type=int, default=None,
                    help="train on random aligned crops of this many latent frames.  Text is cropped with "
                         "the audio, so the pairing stays consistent, and each step sees far more variety "
                         "for the same compute (the papers' own recipe)")
    ap.add_argument("--warm-start", default=None,
                    help="checkpoint whose matching weights to load before training.  The text encoder "
                         "(dim 256, 4 layers) is shared with the Tiny variant, so a text side trained to "
                         "regress latents gives the flow's cross-attention a meaningful memory from step 0 "
                         "instead of a random one")
    ap.add_argument("--t-sampling", default=None, choices=["uniform", "logit_normal"],                    help="timestep distribution for flow matching; `logit_normal` concentrates t around "
                         "0.5 (the SD3 trick) instead of spending most samples near the noise end")
    ap.add_argument("--holdout", type=int, default=0,
                    help="evaluate rho on this many cache items *after* the training range, giving the "
                         "flow's first train/held-out comparison -- memorisation on the training items "
                         "says nothing about generalisation to unseen text")
    ap.add_argument("--save-every", type=int, default=0,
                    help="checkpoint interval.  Default is a quarter of the run, which is useless for a "
                         "multi-day run: a long run needs regular checkpoints so rho can be read as it "
                         "goes and so an interruption does not lose hours")
    ap.add_argument("--plan-residual", action="store_true",
                    help="model the RESIDUAL x1 - upsample(plan) instead of using the plan as conditioning.  The textual QUESTION is whether the velocity field's own objective then depends on the text -- which the decomposed text_dependence.py can now measure.")
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
    if args.warm_start:
        warm = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        warm_state = (warm.get("ema") or {}).get("shadow") or warm["model"]
        current = model.state_dict()
        usable = {k: v for k, v in warm_state.items()
                  if k in current and tuple(current[k].shape) == tuple(v.shape)}
        model.load_state_dict(usable, strict=False)
        text_keys = [k for k in usable if k.startswith("text.")]
        print(f"warm start: loaded {len(usable)} tensors, {len(text_keys)} of them from the text encoder "
              f"({args.warm_start})")
    if args.plan_residual:
        cfg.flow.plan_residual = True
        print("plan mode: RESIDUAL (the field models x1 - upsample(plan))")
    if args.t_sampling:
        cfg.flow.t_sampling = args.t_sampling
        print(f"timestep sampling: {args.t_sampling}")
    if args.cross_gain is not None:
        with torch.no_grad():
            for block in model.vf.blocks:
                block.cross_gain.fill_(args.cross_gain)
        print(f"cross-attention gain overridden to {args.cross_gain}")

    # score a bounded number of items: the point is a stable estimate, and scoring 200 of them costs
    # more than a minute of sampling for a number that twelve items pin down to ~0.01
    probe = min(args.items, 24)
    before = sampled_rho(model, cfg, dataset, probe)
    print(f"before training: sampled latent rho on {probe} of the {len(dataset)} training utterances "
          f"= {before:+.4f}")

    holdout = None
    if args.holdout:
        holdout = LatentShardDataset(
            args.cache, indices=list(range(args.items, args.items + args.holdout))
        )
        print(f"holdout: {len(holdout)} utterances never trained on "
              f"(rho before = {sampled_rho(model, cfg, holdout, min(args.holdout, 24)):+.4f})")

    source = LatentShardBatchSource(
        dataset, batch_size=args.batch_size, shuffle=True, seed=0,
        pair_references=False, self_reference=True, crop_frames=args.crop_frames,
    )
    # Align the schedule with the run, exactly as `train.py --steps` does: the config's warmup is 1000
    # steps, and calling `run_stage` directly with a shorter budget leaves the learning rate at ~0 for the
    # whole run -- which silently invalidates the test (the first attempt at this measured a model that
    # had barely been trained, not one that could not learn).
    cfg.train.max_steps = args.steps
    cfg.train.warmup_steps = max(10, min(cfg.train.warmup_steps, args.steps // 20))
    cfg.train.save_every = args.save_every or max(1, args.steps // 4)
    if args.lr is not None:
        cfg.train.lr = args.lr
    print(f"schedule: {args.steps} steps with {cfg.train.warmup_steps} warmup at lr {cfg.train.lr}")
    logs = run_stage(
        "flow", cfg, model=model, batches=source, max_steps=args.steps, out_dir=args.out,
        device="cpu", log_fn=lambda l: print("  ", {k: round(v, 4) if isinstance(v, float) else v
                                                   for k, v in l.items()}),
    )
    after = sampled_rho(model, cfg, dataset, probe)
    holdout_after = (
        sampled_rho(model, cfg, holdout, min(args.holdout, 24)) if holdout is not None else None
    )
    print(f"\nafter {args.steps} steps on {len(dataset)} utterances: rho = {after:+.4f} "
          f"(target 0.75 to be intelligible)")
    if holdout is not None:
        print(f"held-out rho (never trained on): {holdout_after:+.4f}  <- the number that generalises")
    Path("runs/flow_overfit.json").write_text(json.dumps({
        "items": len(dataset), "steps": args.steps, "holdout": args.holdout,
        "rho_before": before, "rho_after": after, "rho_holdout": holdout_after,
        "crop_frames": args.crop_frames, "lr": cfg.train.lr, "logs": logs,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
