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


    def loss_with(text_mode: str) -> tuple:
        """Mean losses; `text_mode` is `right` or `shuffled`.  Returns ``(total, flow_only, plan)``.

        The decomposition matters: with a plan head the total includes a *supervised* text -> token
        regression, which is text-dependent by construction.  Reporting the total alone would credit the
        velocity field with the plan's work -- and it did, until this was separated.
        """
        totals, flows, plans = [], [], []
        for _ in range(3):
            batch = source()
            ids, mask = batch["ids"], batch.get("text_mask")
            if text_mode == "shuffled":
                # a different utterance's text, same batch slot count: this is the control
                order = torch.roll(torch.arange(ids.shape[0]), shifts=1)
                ids = ids[order]
                mask = None if mask is None else mask[order]
            torch.manual_seed(0)
            loss, aux = model.flow_loss(
                ids, mask, batch["latent"], voice=batch.get("voice"),
                latent_token=batch.get("latent_token"),
            )
            plan = aux.get("plan")
            plan_value = float(plan) if plan is not None else 0.0
            totals.append(float(loss))
            plans.append(plan_value)
            flows.append(float(loss) - plan_value)
        mean = lambda v: sum(v) / len(v)
        return mean(totals), mean(flows), mean(plans)


    right_total, right_flow, right_plan = loss_with("right")
    shuffled_total, shuffled_flow, shuffled_plan = loss_with("shuffled")
    print(f"\n{'':24s} {'right':>10s} {'shuffled':>10s} {'difference':>11s} {'share':>8s}")
    for label, right, shuffled in (
        ("total (flow + plan)", right_total, shuffled_total),
        ("flow term only", right_flow, shuffled_flow),
        ("plan term (supervised)", right_plan, shuffled_plan),
    ):
        delta = shuffled - right
        print(f"{label:24s} {right:10.4f} {shuffled:10.4f} {delta:+11.4f} "
              f"{100 * delta / max(right, 1e-9):+7.1f}%")
    delta = shuffled_flow - right_flow
    print(f"\n  the velocity field's OWN text dependence: "
          f"{100 * delta / max(right_flow, 1e-9):+.1f}% (the plan's gradient is excluded)")
    if abs(delta) < 0.01 * right_flow:
        print("  -> the velocity field's objective is TEXT-INDEPENDENT: the plan was carrying the signal")
    else:
        print("  -> the velocity field itself uses the text; the gap is optimisation or scale")

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
