"""Train one Parakeet stage.

    # real run
    python scripts/train.py --config configs/parakeet_tiny.yaml --stage autoencoder \
        --cache data/latent_cache --steps 50000

    # dry run (no corpus, synthetic batches) -- useful to validate a config or a machine
    python scripts/train.py --config configs/parakeet_small.yaml --stage flow --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.data.dataset import LatentShardBatchSource, LatentShardDataset, SyntheticBatchSource  # noqa: E402
from parakeet.models import build_model, count_parameters  # noqa: E402
from parakeet.train.stages import STAGE_STEPS, run_stage  # noqa: E402

STAGES = list(STAGE_STEPS)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a Parakeet stage")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--cache", default=None, help="latent shard cache dir (from build_latent_cache)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true", help="use synthetic batches (no data needed)")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.steps:
        cfg.train.max_steps = args.steps
    if args.out:
        cfg.train.out_dir = args.out

    model = build_model(cfg)
    if args.resume:
        from parakeet.train.common import load_checkpoint

        load_checkpoint(args.resume, model)
        print(f"resumed from {args.resume}")

    print(f"stage={args.stage} variant={cfg.variant} params={count_parameters(model)/1e6:.3f}M")

    if args.dry_run or not args.cache:
        if not args.dry_run:
            print("no --cache given: falling back to --dry-run synthetic batches")
        source = SyntheticBatchSource(cfg, args.stage, batch_size=cfg.train.batch_size)
    else:
        dataset = LatentShardDataset(args.cache)
        source = LatentShardBatchSource(dataset, batch_size=cfg.train.batch_size, seed=cfg.train.seed)

    def log_fn(logs):
        printable = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items()
        )
        print(f"step {int(logs.get('step', 0)):>8d} {printable}")

    logs = run_stage(
        args.stage,
        cfg,
        model=model,
        batches=source,
        max_steps=cfg.train.max_steps,
        out_dir=cfg.train.out_dir,
        device=args.device,
        log_fn=log_fn,
    )
    Path(cfg.train.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.train.out_dir) / f"{args.stage}_final.json").write_text(json.dumps(logs, indent=2))
    print("done:", json.dumps(logs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
