"""The conditioning gate: how much of the flow's objective actually comes from the text.

The number this project judges conditioning changes by. Written after the measurement was miscounted once: with
a plan head, the *total* flow-matching loss includes a **supervised** text -> token regression, which is
text-dependent by construction, so a total that looked like 37 % was almost entirely the plan while the velocity
field's own dependence was 2 %. Anything that reports conditioning must decompose it.

    from parakeet.eval.gates import text_dependence

    result = text_dependence(model, batches)
    result["flow_share"]   # the number to move: the velocity field's own text dependence, in percent

The control is a shuffled text: the same batch, with each utterance's text replaced by another's. That is the
only control that distinguishes "the model uses the text" from "the model fits the marginal", because the
marginal term is identical under both.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch


def text_dependence(
    model: torch.nn.Module,
    batches: Callable[[], Dict[str, torch.Tensor]],
    *,
    repeats: int = 3,
    seed: int = 0,
) -> Dict[str, float]:
    """Flow term vs plan term, with the right text and with a shuffled text.

    Returns ``right``/``shuffled``/``difference`` for each of ``total``, ``flow`` and ``plan``, plus the
    ``*_share`` percentages.  ``flow_share`` is the one to quote: the velocity field's own text dependence.
    """

    # Measure in eval mode.  `flow_loss` applies 10 % classifier-free-guidance conditioning dropout when the
    # model is *training*, so a gate measured in train mode compares right-text (polluted by dropped
    # conditioning on one batch in ten) against shuffled-text and reports an inflated share.  Round 114 found
    # this by measuring the same thing at fixed timesteps in eval mode: 0.3-1.3 %, against the 2-7 % this
    # function had been reporting.  Both say "small", but the number quoted all over this project's docs was
    # the inflated one.
    was_training = model.training
    model.eval()

    def measure(shuffle: bool) -> Dict[str, float]:
        totals, flows, plans = [], [], []
        for _ in range(repeats):
            batch = batches()
            ids, mask = batch["ids"], batch.get("text_mask")
            if shuffle:
                order = torch.roll(torch.arange(ids.shape[0]), shifts=1)
                ids = ids[order]
                mask = None if mask is None else mask[order]
            torch.manual_seed(seed)
            loss, aux = model.flow_loss(
                ids, mask, batch["latent"], voice=batch.get("voice"),
                latent_token=batch.get("latent_token"),
            )
            plan = aux.get("plan")
            plan_value = float(plan) if plan is not None else 0.0
            totals.append(float(loss))
            plans.append(plan_value)
            flows.append(float(loss) - plan_value)
        mean = lambda values: sum(values) / len(values)  # noqa: E731
        return {"total": mean(totals), "flow": mean(flows), "plan": mean(plans)}

    right, shuffled = measure(False), measure(True)
    model.train(was_training)
    out: Dict[str, float] = {}
    for key in ("total", "flow", "plan"):
        out[f"right_{key}"] = right[key]
        out[f"shuffled_{key}"] = shuffled[key]
        out[f"difference_{key}"] = shuffled[key] - right[key]
        out[f"{key}_share"] = 100.0 * (shuffled[key] - right[key]) / max(right[key], 1e-9)
    out["flow_share"] = out["flow_share"]
    out["measured_in_eval_mode"] = True
    return out


def sampled_rho_with_error(
    model,
    cfg,
    dataset,
    items: int = 24,
    *,
    seeds: tuple = (0, 1, 2, 3, 4),
    nfe: int = 4,
) -> Dict[str, float]:
    """Per-dimension correlation between sampled and target latents, **with an error bar**.

    Round 100 found this readout is noisy at the magnitudes this project produces: arm A reported +0.0882 at step
    2000, and continuing from that same checkpoint measured −0.0277 — same model, same cache, different probe
    draw. A single reading near zero therefore carries no information, which is exactly where every flow result
    here has landed. Reporting the spread across probe seeds is what makes the number usable: an interval that
    straddles zero says "unmeasured", and only an interval clear of zero is a positive.

    Returns ``mean``, ``std``, ``min``, ``max`` and ``n_seeds`` over per-seed ρ values.
    """
    import torch

    from parakeet.models.flow import consistency_sample
    from parakeet.models.parakeet import unfold_time

    was_training = model.training
    model.eval()
    scores = []
    try:
        with torch.no_grad():
            for seed in seeds:
                per_item = []
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
                    generator = torch.Generator().manual_seed(seed)
                    x0 = torch.randn(1, cfg.flow.latent_dim * cfg.flow.compress, tc, generator=generator)
                    x1c = consistency_sample(
                        model.vf, memory, memory_mask,
                        (1, cfg.flow.latent_dim * cfg.flow.compress, tc),
                        steps=nfe, device=torch.device("cpu"),
                        cfg_scale=getattr(cfg.flow, "cfg_scale", 1.0), x0=x0,
                    )
                    sampled = unfold_time(x1c, cfg.flow.compress, t_out=frames)
                    target = item["latent"][None]
                    length = min(sampled.shape[-1], target.shape[-1])
                    dims = []
                    for dim in range(sampled.shape[1]):
                        x, y = sampled[0, dim, :length], target[0, dim, :length]
                        if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
                            continue
                        dims.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
                    if dims:
                        per_item.append(sum(dims) / len(dims))
                if per_item:
                    scores.append(sum(per_item) / len(per_item))
    finally:
        model.train(was_training)

    if not scores:
        # Degenerate output (every dimension constant, e.g. a model that emits zeros) carries no information,
        # so it is reported the same way as noise around zero -- and crucially WITH the key callers read.
        # The early return used to omit `straddles_zero`, which would have raised KeyError in any caller that
        # branches on it rather than returning "cannot tell".
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"),
                "n_seeds": 0, "per_seed": [], "straddles_zero": True}
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
    return {
        "mean": mean,
        "std": variance ** 0.5,
        "min": min(scores),
        "max": max(scores),
        "n_seeds": len(scores),
        "per_seed": scores,
        "straddles_zero": min(scores) <= 0.0 <= max(scores),
    }
