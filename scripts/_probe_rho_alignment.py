"""Is the rho measurement fair?  A time shift would look exactly like "the model ignores the text".

Every rho measurement in rounds 37-40 sampled at the *predicted* length and compared against the teacher's
frames, truncating to the shorter of the two.  If the predicted length is offset -- and lengths here are
1.03-1.8x the target -- the sampled sequence and the target describe different spans, and a **time shift
destroys per-dimension correlation** regardless of whether the content is right.

This recomputes rho three ways on the same samples:
  * as measured before (truncate to the shorter);
  * with the sampling length forced to the target's;
  * with a shift search, taking the best correlation over a window of offsets.

If rho jumps with alignment, the flow's failure has been mis-measured and the conclusion changes.
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")

import torch

from parakeet.config import load_config
from parakeet.data.dataset import LatentShardDataset
from parakeet.models import build_model
from parakeet.models.flow import consistency_sample, unfold_time
from parakeet.train.common import (
    apply_checkpoint_geometry,
    derive_n_voices_from_cache,
    load_latent_norm_from_cache,
)

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "runs/flow_overfit2b/flow_last.pt"
ITEMS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CACHE = "runs/mixed_v2/latent_cache"

cfg = load_config("configs/parakeet_flow.yaml")
dataset = LatentShardDataset(CACHE, indices=list(range(ITEMS)))
payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
state = (payload.get("ema") or {}).get("shadow") or payload["model"]
apply_checkpoint_geometry(cfg, state)
cfg.n_voices = derive_n_voices_from_cache(CACHE)
model = build_model(cfg)
current = model.state_dict()
model.load_state_dict({k: v for k, v in state.items()
                       if k in current and tuple(current[k].shape) == tuple(v.shape)}, strict=False)
load_latent_norm_from_cache(model, CACHE)
model.eval()
print(f"{CHECKPOINT} on {ITEMS} training utterances\n")


def correlate(a: torch.Tensor, b: torch.Tensor, shift: int = 0) -> float:
    """Mean per-dim correlation between two (C, T) latents, with ``b`` shifted by ``shift`` frames."""
    if shift >= 0:
        a2, b2 = a[:, shift:], b[:, :b.shape[-1] - shift]
    else:
        a2, b2 = a[:, :a.shape[-1] + shift], b[:, -shift:]
    length = min(a2.shape[-1], b2.shape[-1])
    if length < 8:
        return float("nan")
    a2, b2 = a2[:, :length], b2[:, :length]
    scores = []
    for dim in range(a2.shape[0]):
        x, y = a2[dim], b2[dim]
        if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
            continue
        scores.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
    return sum(scores) / max(1, len(scores))


truncated, forced, best_shift = [], [], []
with torch.no_grad():
    for index in range(ITEMS):
        item = dataset[index]
        target = item["latent"][None]
        target_frames = target.shape[-1]
        ids = item["ids"][None]
        mask = item.get("text_mask")
        mask = mask[None] if mask is not None else None
        voice = item.get("voice")
        voice = voice.reshape(1) if voice is not None else None
        memory, memory_mask, cond = model.conditions(ids, mask, voice=voice)

        predicted = int(model.predict_latent_frames(model.text(ids, mask), mask, cond).item())
        # (1) as measured before: sample at the predicted length
        tc_pred = model.compressed_frames(predicted)
        generator = torch.Generator().manual_seed(0)
        x0 = torch.randn(1, cfg.flow.latent_dim * cfg.flow.compress, tc_pred, generator=generator)
        sampled = unfold_time(
            consistency_sample(model.vf, memory, memory_mask,
                               (1, cfg.flow.latent_dim * cfg.flow.compress, tc_pred),
                               steps=4, device=torch.device("cpu"), cfg_scale=cfg.flow.cfg_scale, x0=x0),
            cfg.flow.compress, t_out=predicted,
        )
        truncated.append(correlate(sampled[0], target[0]))

        # (2) force the sampling to the target's length, so both cover the same span
        tc_target = model.compressed_frames(target_frames)
        generator = torch.Generator().manual_seed(0)
        x0 = torch.randn(1, cfg.flow.latent_dim * cfg.flow.compress, tc_target, generator=generator)
        sampled_t = unfold_time(
            consistency_sample(model.vf, memory, memory_mask,
                               (1, cfg.flow.latent_dim * cfg.flow.compress, tc_target),
                               steps=4, device=torch.device("cpu"), cfg_scale=cfg.flow.cfg_scale, x0=x0),
            cfg.flow.compress, t_out=target_frames,
        )
        forced.append(correlate(sampled_t[0], target[0]))

        # (3) shift search on the forced-length sample
        window = 60
        scores = {shift: correlate(sampled_t[0], target[0], shift) for shift in range(-window, window + 1, 5)}
        scores = {k: v for k, v in scores.items() if v == v}
        best = max(scores.items(), key=lambda kv: kv[1]) if scores else (0, float("nan"))
        best_shift.append(best[1])
        print(f"  item {index}: predicted {predicted} vs target {target_frames} frames | "
              f"rho truncated {truncated[-1]:+.3f} | forced {forced[-1]:+.3f} | "
              f"best-shift {best[1]:+.3f} at {best[0]:+d}")


def mean(values):
    values = [v for v in values if v == v]
    return sum(values) / max(1, len(values))


print(f"\n  mean rho as measured before : {mean(truncated):+.4f}")
print(f"  mean rho, length forced     : {mean(forced):+.4f}")
print(f"  mean rho, best shift        : {mean(best_shift):+.4f}")
print(f"  target for intelligibility  : +0.75")
Path("runs/rho_alignment_check.json").write_text(
    __import__("json").dumps({"truncated": truncated, "forced": forced, "best_shift": best_shift}, indent=2),
    encoding="utf-8",
)
