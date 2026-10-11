"""Can the text side memorise a handful of utterances?

Validation rho plateaus at ~0.37 with training rho no higher (round 60): the model cannot even fit the
training set. That has three possible causes -- insufficient capacity, an optimisation problem, or a task
whose text->acoustic mapping is intrinsically high-entropy -- and they call for different spending.

Memorisation separates them cheaply. If the text side can drive per-dimension correlation on eight
utterances to ~0.9, capacity and optimisation are adequate for fitting and the plateau is about the mapping
itself. If it cannot, no amount of data will help.

    python scripts/overfit_text.py --items 8 --steps 3000 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from parakeet.config import load_config
from parakeet.data.dataset import LatentShardBatchSource, LatentShardDataset
from parakeet.train.common import derive_n_voices_from_cache, load_checkpoint_into, load_latent_norm_from_cache
from parakeet.train.stages import run_stage


def token_rho(model, dataset, items: int) -> float:
    """Mean per-dimension correlation between predicted and target *token* latents.

    Tokens, not frames: that is what the stage's loss is computed on, so it is the right measure of "has the
    model fitted these items".
    """
    scores = []
    with torch.no_grad():
        for index in range(min(items, len(dataset))):
            item = dataset[index]
            ids = item["ids"][None]
            mask = item.get("text_mask")
            mask = mask[None] if mask is not None else None
            voice = item.get("voice")
            voice = voice.reshape(1) if voice is not None else None
            predicted = model.text_side(ids, mask, voice)["latent_token"][0]
            target = item["latent_token"]
            length = min(predicted.shape[0], target.shape[0])
            per_dim = []
            for dim in range(predicted.shape[1]):
                x, y = predicted[:length, dim], target[:length, dim]
                if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
                    continue
                per_dim.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
            if per_dim:
                scores.append(sum(per_dim) / len(per_dim))
    return sum(scores) / max(1, len(scores)) if scores else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="Memorisation test for the tiny text side")
    ap.add_argument("--cache", default="runs/mixed_v2/latent_cache")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--warm-start", default="runs/text_continue/distill-text_step5000.pt")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/overfit_text")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.text.dim = 512
    cfg.text.n_layers = 6
    cfg.flow.text_dim = 512
    cfg.flow.cond_dim = 512
    cfg.n_voices = derive_n_voices_from_cache(args.cache) or 1
    model, applied, _payload = load_checkpoint_into(cfg, args.warm_start)
    load_latent_norm_from_cache(model, args.cache)
    print(f"warm start {args.warm_start} | geometry {applied} | voices {cfg.n_voices}")

    dataset = LatentShardDataset(args.cache, indices=list(range(args.items)))
    before = token_rho(model, dataset, args.items)
    print(f"before: token rho on {len(dataset)} utterances = {before:+.4f}")

    source = LatentShardBatchSource(
        dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True, seed=0
    )
    cfg.train.max_steps = args.steps
    cfg.train.warmup_steps = max(10, min(cfg.train.warmup_steps, args.steps // 40))
    cfg.train.lr = args.lr
    cfg.train.save_every = 10**9  # no checkpoints: this is a question, not a deliverable
    print(f"schedule: {args.steps} steps, warmup {cfg.train.warmup_steps}, lr {cfg.train.lr}")
    logs = run_stage(
        "distill-text", cfg, model=model, batches=source, max_steps=args.steps, out_dir=args.out,
        device="cpu",
        log_fn=lambda entry: print("  ", {k: (round(v, 4) if isinstance(v, float) else v)
                                          for k, v in entry.items()}),
    )
    after = token_rho(model, dataset, args.items)
    print(f"\nafter {args.steps} steps: token rho on the same {len(dataset)} utterances = {after:+.4f}")
    verdict = (
        "CAN memorise -- capacity and optimisation are adequate for fitting"
        if after > 0.8
        else "CANNOT memorise -- capacity or optimisation is the wall, not the data"
        if after < 0.5
        else "partially memorises"
    )
    print(f"  {verdict}")
    Path("runs/overfit_text.json").write_text(
        json.dumps({
            "items": len(dataset), "steps": args.steps, "lr": args.lr,
            "rho_before": before, "rho_after": after,
            "last_logs": (logs[-3:] if isinstance(logs, list) else logs),
        }, indent=2, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
