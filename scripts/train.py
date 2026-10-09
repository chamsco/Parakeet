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
from parakeet.data.dataset import make_batch_source  # noqa: E402
from parakeet.models import build_model, count_parameters  # noqa: E402
from parakeet.train.stages import STAGE_STEPS, run_stage  # noqa: E402

STAGES = list(STAGE_STEPS)


def _cache_provenance(cache: str | None) -> dict:
    """Which data produced this run: teacher mixture, voices and a content hash of the cache index.

    Recorded in ``<out_dir>/run.json`` so a checkpoint can be traced back to its corpus, which is the
    point of provenance for a distillation pipeline built on someone else's voices.
    """
    if not cache:
        return {}
    from parakeet.train.common import file_fingerprint

    cache_dir = Path(cache)
    meta: dict = {}
    meta_path = cache_dir / "cache_meta.json"
    if meta_path.exists():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = {
            "teachers": payload.get("teacher_names"),
            "teacher_weights": payload.get("teacher_weights"),
            "voices": payload.get("voice_names"),
            "n_shards": payload.get("n_shards"),
        }
    index = cache_dir / "index.json"
    if index.exists():
        meta["cache_index_sha256"] = file_fingerprint(index)
    return meta


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
    ap.add_argument("--no-pair-references", action="store_true",
                    help="condition the flow stage on each utterance's own mel instead of a "
                         "different utterance of the same voice (PilotTTS pairing is the default)")
    ap.add_argument("--max-ref-frames", type=int, default=None,
                    help="cap the reference prompt length (default: train.max_ref_frames)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.steps:
        cfg.train.max_steps = args.steps
    if args.out:
        cfg.train.out_dir = args.out
    if args.max_ref_frames:
        cfg.train.max_ref_frames = args.max_ref_frames

    model = build_model(cfg)
    if args.resume:
        from parakeet.train.common import load_checkpoint

        load_checkpoint(args.resume, model)
        print(f"resumed from {args.resume}")

    print(f"stage={args.stage} variant={cfg.variant} params={count_parameters(model)/1e6:.3f}M")

    # the batch-source decision lives in the library so it is shared and tested (it was neither
    # when it lived here: the CLI bypassed cross-sample pairing for the flow stage entirely)
    if args.dry_run or not args.cache:
        if not args.dry_run:
            print("no --cache given: falling back to --dry-run synthetic batches")
        source = make_batch_source(cfg, args.stage, None, batch_size=cfg.train.batch_size)
    else:
        source = make_batch_source(
            cfg,
            args.stage,
            args.cache,
            batch_size=cfg.train.batch_size,
            pair_references=not args.no_pair_references,
            max_ref_frames=cfg.train.max_ref_frames,
        )
        if args.stage == "flow":
            print(
                f"flow conditioning: pair_references={not args.no_pair_references} "
                f"max_ref_frames={cfg.train.max_ref_frames}"
            )

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
        run_metadata=_cache_provenance(args.cache),
    )
    Path(cfg.train.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.train.out_dir) / f"{args.stage}_final.json").write_text(json.dumps(logs, indent=2))
    print("done:", json.dumps(logs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
