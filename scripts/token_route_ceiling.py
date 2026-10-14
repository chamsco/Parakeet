"""What frame-ρ would a residual built on the TOKEN ROUTE inherit?

Round 116 proposed `x1_residual = x1 - unfold(token_route(ids))` and quoted a floor of 0.37 — but 0.37 is the token
route's **token** ρ. What a residual route actually inherits is the token route's **frame** ρ, after its decoder
expands tokens to frames, and that had never been measured. If the samples came out as the plan alone (the field
adding nothing, which is what fields have done here), the frame ρ of the token route's own output is the floor.

Two minutes, versus the 3-9 CPU-hours of training the route would otherwise cost before finding out.

    python scripts/token_route_ceiling.py --checkpoint runs/text_continue/distill-text_step5000.pt \
        --config configs/parakeet_tiny.yaml --cache runs/expanded_v3 [--items 16]
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
    ap = argparse.ArgumentParser(description="The token route's frame rho against the target")
    ap.add_argument("--checkpoint", required=True, help="a distill-text (token route) checkpoint")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--items", type=int, default=16)
    args = ap.parse_args()

    path = Path(args.checkpoint)
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

    # A cache whose global normaliser is (near) identity is the symptom of the defect recorded in
    # docs/05-VERIFICATION.md section 144: runs/holdout_cache carried an identity global normaliser while the training
    # caches share one fitted mean and variance, and the correction lived only in that cache's metadata, which is
    # gitignored. A cache rebuild would restore the identity silently and move every reading by about 0.019 — the size
    # of the instrument's own resolution. So the instrument reports it rather than depending on a note.
    import json as _json

    _meta_path = Path(args.cache) / "cache_meta.json"
    if _meta_path.exists():
        _norm = (_json.loads(_meta_path.read_text(encoding="utf-8")).get("latent_norm") or {})
        _mean = _norm.get("mean")
        if _mean is not None and max(abs(float(m)) for m in _mean) < 1e-6:
            print("  WARNING: this cache's global latent normaliser is an IDENTITY (all-zero mean).")
            print("  The training caches carry a fitted mean, so every reading here is on a different convention")
            print("  than the models were trained in. See docs/05-VERIFICATION.md section 144.")

    if not hasattr(model, "decoder_latent_from_tokens"):
        raise SystemExit("this build has no decoder_latent_from_tokens: cannot expand tokens to frames")

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
            side = model.text_side(ids, mask, voice)
            durations = item.get("durations")
            # the TRUE durations, so this measures whether the predicted latents track the target frames rather
            # than whether the duration head is accurate; the residual route would use predicted durations, which
            # can only make its floor worse, never better.
            frames_out = model.decoder_latent_from_tokens(
                side["latent_token"], durations, f0=side.get("f0"), energy=side.get("energy")
            )
            predicted_frames = frames_out[0] if isinstance(frames_out, tuple) else frames_out
            frames.append(frame_rho(predicted_frames[0], item["latent"]))
            tokens.append(token_rho(side["latent_token"][0], item["latent_token"]))

    mean = lambda v: sum(v) / len(v)  # noqa: E731
    report = {
        "items": len(dataset),
        "frame_rho": mean(frames),
        "frame_rho_median": statistics.median(frames),
        "frame_rho_max": max(frames),
        "token_rho": mean(tokens) if tokens else None,
    }
    print(f"checkpoint {path} (step {payload.get('step')}) | {len(dataset)} items")
    print(f"  token route vs target FRAMES : rho {report['frame_rho']:+.4f} "
          f"(median {report['frame_rho_median']:+.4f}, max {report['frame_rho_max']:+.4f})")
    if report["token_rho"] is not None:
        print(f"  token route vs target TOKENS : rho {report['token_rho']:+.4f}")
    print("\n  A residual route inherits this as its floor: it is what the samples would score if the flow's")
    print("  field added nothing, which is what fields have done in this project so far.")
    if report["frame_rho"] > 0.20:
        print("  -> substantial: the residual route has a real floor and is worth building and training.")
    else:
        print("  -> near zero: the token route's own frames do not track the target, so a residual built on it")
        print("     inherits nothing. Do not build it.")

    Path("runs/token_route_ceiling.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
