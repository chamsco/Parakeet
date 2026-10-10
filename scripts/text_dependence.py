"""Does the flow's objective depend on the text at all?  A shuffled-text control, at the loss level.

Round 39 measured an *untrained* field being indifferent to which text it was given, and round 49 measured
p(latent) learned while p(latent | text) is not.  Both were inferred from samples.  This asks the sharper
question directly on the training objective: pair each utterance's acoustics with a **different utterance's
text** and see whether the flow-matching loss moves.

If the loss is the same with shuffled text as with the right text, the conditioning is not reaching the
objective at all -- and no amount of training fixes that, because there is nothing to learn.  If the loss
degrades with shuffled text, the conditioning is in the objective and the problem is optimisation or scale.

Gradient norms per module group are reported alongside, because a conditioning path that receives no
gradient is a different bug from one that receives gradient and is not yet learned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from parakeet.config import load_config
from parakeet.data.dataset import LatentShardBatchSource, LatentShardDataset
from parakeet.models import build_model
from parakeet.train.common import (
    apply_checkpoint_geometry,
    derive_n_voices_from_cache,
    load_latent_norm_from_cache,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Is the text in the flow's objective?")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/parakeet_flow_plan.yaml")
    ap.add_argument("--cache", default="runs/mixed_v2/latent_cache")
    ap.add_argument("--items", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5, help="gradient-probe steps")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = LatentShardDataset(args.cache, indices=list(range(args.items)))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    apply_checkpoint_geometry(cfg, state)
    cfg.n_voices = derive_n_voices_from_cache(args.cache) or 1
    model = build_model(cfg)
    current = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items()
                           if k in current and tuple(current[k].shape) == tuple(v.shape)}, strict=False)
    load_latent_norm_from_cache(model, args.cache)
    model.train()
    print(f"checkpoint step: {payload.get('step')} | use_plan={model.use_plan} "
          f"| plan_residual={getattr(cfg.flow, 'plan_residual', False)}")

    source = LatentShardBatchSource(dataset, batch_size=args.batch_size, shuffle=False, self_reference=True)


    def loss_with(text_mode: str) -> float:
        """Mean flow-matching loss over a few batches; `text_mode` is `right` or `shuffled`."""
        values = []
        for _ in range(3):
            batch = source()
            ids, mask = batch["ids"], batch.get("text_mask")
            if text_mode == "shuffled":
                # a different utterance's text, same batch slot count: this is the control
                order = torch.roll(torch.arange(ids.shape[0]), shifts=1)
                ids = ids[order]
                mask = None if mask is None else mask[order]
            torch.manual_seed(0)
            loss, _aux = model.flow_loss(
                ids, mask, batch["latent"], voice=batch.get("voice"),
                latent_token=batch.get("latent_token"),
            )
            values.append(float(loss))
        return sum(values) / len(values)


    right = loss_with("right")
    shuffled = loss_with("shuffled")
    print(f"\nflow loss with the RIGHT text     : {right:.4f}")
    print(f"flow loss with a SHUFFLED text    : {shuffled:.4f}")
    delta = shuffled - right
    print(f"difference                        : {delta:+.4f} "
          f"({100 * delta / max(right, 1e-9):+.1f}% of the loss)")
    if abs(delta) < 0.01 * right:
        print("  -> the objective is TEXT-INDEPENDENT: there is nothing for training to learn here")
    else:
        print("  -> the objective does depend on the text; the gap is optimisation or scale")

    # one backward is enough to see which paths receive gradient at all
    groups = {
        "text encoder": "text.",
        "plan head": "plan_",
        "velocity field": "vf.",
        "speaker/style": ("speaker.", "cond_proj.", "voice_embed."),
        "length predictor": "length_predictor.",
    }
    model.zero_grad(set_to_none=True)
    batch = source()
    loss, _aux = model.flow_loss(
        batch["ids"], batch.get("text_mask"), batch["latent"], voice=batch.get("voice"),
        latent_token=batch.get("latent_token"),
    )
    loss.backward()
    # NB: `flow_loss` carries the flow and plan terms only; the length term lives in `stage_flow`, so a zero`n    # for the length predictor here is the probe's scope, not a model finding`n    print("\ngradient norm by module group (one backward):")
    for name, prefixes in groups.items():
        if isinstance(prefixes, str):
            prefixes = (prefixes,)
        squared = 0.0
        for key, parameter in model.named_parameters():
            if parameter.grad is None or not key.startswith(prefixes):
                continue
            squared += float(parameter.grad.pow(2).sum())
        print(f"  {name:18s} {squared ** 0.5:12.5f}" + ("" if squared else "   <- NO GRADIENT"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
