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
from parakeet.train.common import derive_n_voices_from_cache  # noqa: E402
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
    ap.add_argument("--set", action="append", metavar="KEY.PATH=VALUE",
                    help="override a config field, e.g. autoencoder.decoder_uses_token_latents=false "
                         "(repeatable)")
    ap.add_argument("--no-pair-references", action="store_true",
                    help="condition the flow stage on each utterance's own mel instead of a "
                         "different utterance of the same voice (PilotTTS pairing is the default)")
    ap.add_argument("--max-ref-frames", type=int, default=None,
                    help="cap the reference prompt length (default: train.max_ref_frames)")
    ap.add_argument("--warm-start", default=None, metavar="CKPT",
                    help="load *weights only* from a checkpoint, with this stage's optimizer and "
                         "schedule starting fresh.  Use --resume to continue the same run; use this "
                         "to fine-tune a new stage from an existing one, which --resume cannot "
                         "express because it restores the previous stage's optimizer state")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for override in args.set or []:
        # `--set autoencoder.decoder_uses_token_latents=false` -- a run-time switch keeps an A/B an
        # A/B, instead of two config files that drift apart
        target, _, raw = override.partition("=")
        if not raw:
            raise SystemExit(f"--set expects key.path=value, got {override!r}")
        parts = target.strip().split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        field = parts[-1]
        if not hasattr(obj, field):
            raise SystemExit(f"--set {target}: no such config field")
        current = getattr(obj, field)
        if isinstance(current, bool):
            value = raw.strip().lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw
        setattr(obj, field, value)
        print(f"[train] override {target} = {value!r}")
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.steps:
        cfg.train.max_steps = args.steps
    if args.out:
        cfg.train.out_dir = args.out
    if args.max_ref_frames:
        cfg.train.max_ref_frames = args.max_ref_frames

    # the corpus decides how wide the voice table must be -- and a mismatch cannot be loaded even with
    # strict=False, so this has to happen before build_model / resume
    derived_voices = derive_n_voices_from_cache(args.cache)
    if derived_voices is not None and derived_voices != cfg.n_voices:
        print(f"[train] n_voices {cfg.n_voices} -> {derived_voices} (from the cache's voice names)")
        cfg.n_voices = derived_voices

    model = build_model(cfg)
    if args.warm_start:
        payload = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        ema_shadow = (payload.get("ema") or {}).get("shadow")
        state = ema_shadow or payload["model"]
        # Skip keys whose *shape* differs instead of failing the run.  `strict=False` tolerates missing
        # and unexpected keys but not a size mismatch, which is what makes a cross-variant warm start
        # awkward: an autoencoder checkpoint trained before the corpus grew carries a 4-voice embedding
        # against a 12-voice table, and one stale key should not cost the whole run.  Reported, not
        # silent, so a genuinely wrong checkpoint still shows up.
        current = model.state_dict()
        usable, skipped = {}, []
        for key, value in state.items():
            if key not in current:
                continue
            if tuple(current[key].shape) != tuple(value.shape):
                skipped.append(f"{key} {tuple(value.shape)}->{tuple(current[key].shape)}")
                continue
            usable[key] = value
        info = model.load_state_dict(usable, strict=False)
        print(f"[train] warm start from {args.warm_start} at step {payload.get('step')} "
              f"({'EMA' if ema_shadow else 'raw'} weights, {len(info.missing_keys)} missing / "
              f"{len(info.unexpected_keys)} unexpected keys); optimizer and schedule start fresh")
        if skipped:
            print(f"[train]   skipped {len(skipped)} key(s) whose shape did not match: "
                  f"{', '.join(skipped[:3])}{' ...' if len(skipped) > 3 else ''}")
    if args.resume:
        # NOTE: the model is *not* loaded here.  run_stage does the full restore -- optimizer, EMA,
        # discriminator, LR schedule position, RNG and batch order -- so that resuming continues the
        # run instead of restarting it with warm weights.  Loading only the model here (as this
        # script used to) silently discarded all of that.
        print(f"will resume from {args.resume} (full state: optimizer, EMA, discriminator, schedule)")

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
        resume_from=args.resume,
    )
    Path(cfg.train.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.train.out_dir) / f"{args.stage}_final.json").write_text(json.dumps(logs, indent=2))
    print("done:", json.dumps(logs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
