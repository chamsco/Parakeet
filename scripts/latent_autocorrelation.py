"""Is the flow's sampled latent temporally white while the teacher's is smooth?

Measured (round 49): the flow's samples are NOT white.  Their temporal autocorrelation tracks the
teacher's closely -- lag 1 0.885 against 0.901, lag 4 0.617 against 0.485, lag 8 0.399 against 0.205 -- so
the field has learned the marginal *temporal process* of real latents.  What it has not learned is which
latent belongs to which text (rho ~ 0).  This script exists to keep that distinction measurable: rho is
agreement of values per dimension, and it says nothing about temporal structure.  An earlier guess that the
frame-rate buzz (median F0 93.7486 Hz = the 93.75 Hz frame rate) came from white samples was refuted by
exactly this measurement.

rho measures agreement of *values* per dimension; it says nothing about temporal structure.  If the samples
are white and the targets are not, that is a concrete, actionable description of the failure -- and it also
explains why the flow-matching loss falls: "white noise with the right variance" is a decent marginal
solution and learns nothing about the text.
"""

import sys

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

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "runs/plan_ab/flow_last.pt"
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "configs/parakeet_flow_plan.yaml"
CACHE = "runs/mixed_v2/latent_cache"

cfg = load_config(CONFIG)
dataset = LatentShardDataset(CACHE, indices=[0, 1, 2, 3])
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


def autocorrelation(latent: torch.Tensor, lags=(1, 2, 4, 8, 16)) -> dict:
    """Mean per-dimension autocorrelation at each lag, over the batch's items."""
    out = {}
    for lag in lags:
        values = []
        for item in range(latent.shape[0]):
            x = latent[item]
            if x.shape[-1] <= lag + 8:
                continue
            a, b = x[:, :-lag], x[:, lag:]
            for dim in range(a.shape[0]):
                p, q = a[dim], b[dim]
                if float(p.std()) < 1e-6 or float(q.std()) < 1e-6:
                    continue
                values.append(float(torch.corrcoef(torch.stack([p, q]))[0, 1]))
        out[lag] = sum(values) / max(1, len(values))
    return out


targets, samples = [], []
with torch.no_grad():
    for index in range(len(dataset)):
        item = dataset[index]
        target = item["latent"][None]
        frames = int(target.shape[-1])
        tc = model.compressed_frames(frames)
        ids = item["ids"][None]
        mask = item.get("text_mask")
        mask = mask[None] if mask is not None else None
        voice = item.get("voice")
        voice = voice.reshape(1) if voice is not None else None
        memory, memory_mask, _cond = model.conditions(ids, mask, voice=voice)
        generator = torch.Generator().manual_seed(0)
        x0 = torch.randn(1, cfg.flow.latent_dim * cfg.flow.compress, tc, generator=generator)
        x1c = consistency_sample(model.vf, memory, memory_mask,
                                 (1, cfg.flow.latent_dim * cfg.flow.compress, tc),
                                 steps=4, device=torch.device("cpu"), cfg_scale=cfg.flow.cfg_scale, x0=x0)
        sampled = unfold_time(x1c, cfg.flow.compress, t_out=frames)
        length = min(sampled.shape[-1], frames)
        targets.append(target[0, :, :length])
        samples.append(sampled[0, :, :length])

print(f"latent: {targets[0].shape[0]} dims x {targets[0].shape[-1]} frames, {len(targets)} utterances\n")
print(f"{'lag':>4s} {'teacher target':>15s} {'flow sample':>13s}")
common = min(t.shape[-1] for t in targets)
target_ac = autocorrelation(torch.stack([t[:, :common] for t in targets]))
sample_ac = autocorrelation(torch.stack([s[:, :common] for s in samples]))
for lag in sorted(target_ac):
    print(f"{lag:4d} {target_ac[lag]:15.4f} {sample_ac[lag]:13.4f}")
print(f"\nstd: target {float(torch.stack([t[:, :common] for t in targets]).std()):.3f} | sample {float(torch.stack([s[:, :common] for s in samples]).std()):.3f}")
print("\n  A smooth teacher and a white sample at lag 1 is the frame-rate buzz: the decoder is being fed")
print("  values in the right range with no temporal structure, so it emits energy at the 93.75 Hz frame rate.")
