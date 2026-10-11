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
    out: Dict[str, float] = {}
    for key in ("total", "flow", "plan"):
        out[f"right_{key}"] = right[key]
        out[f"shuffled_{key}"] = shuffled[key]
        out[f"difference_{key}"] = shuffled[key] - right[key]
        out[f"{key}_share"] = 100.0 * (shuffled[key] - right[key]) / max(right[key], 1e-9)
    out["flow_share"] = out["flow_share"]
    return out
