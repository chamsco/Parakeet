"""Diagnose fit: what has the text side learned, on the data it trained on and on unseen prompts?

    python scripts/fit_diagnosis.py --checkpoint runs/mixed_audio/distill-audio_last.pt

Round 28's fit check found the student at WER 1.000 on its **own training prompts** -- it never fit the
data, which reframed three rounds of A/Bs as comparisons of undertrained models.  WER only says "not
intelligible"; this says *which* of the text side's outputs is wrong, which is what decides the next
fix:

* per-token **durations**: mean absolute error in frames and the ratio of predicted to true total
  (a collapsed duration head shows up as a ratio far from 1, as it did in round 19 at 0.29x);
* per-token **F0 and energy**;
* the **latent tokens**: per-dimension correlation between prediction and target, and their cosine
  similarity -- the quantity that decides whether the decoder gets a usable latent.

Both splits are reported, so the fit/generalisation gap is visible instead of inferred: a high
train correlation with a low validation one is a generalisation problem, and both low is underfitting.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.data.dataset import LatentShardDataset  # noqa: E402
from parakeet.data.text import TextTokenizer  # noqa: E402
from parakeet.models import build_model  # noqa: E402
from parakeet.models.duration import normalized_to_durations  # noqa: E402
from parakeet.train.common import infer_model_geometry  # noqa: E402


def collect(
    model, dataset: LatentShardDataset, indices: List[int], tokenizer: TextTokenizer, cfg
) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for index in indices:
            item = dataset[index]
            ids = item["ids"][None]
            side = model.text_side(ids)
            target_durations = item["durations"].float()
            predicted_durations = normalized_to_durations(side["log_duration"])[0].float()
            n = min(predicted_durations.numel(), target_durations.numel())
            frames_true = float(target_durations[:n].sum())
            frames_pred = float(predicted_durations[:n].sum())
            target_latent = item["latent_token"].float()
            predicted_latent = side["latent_token"][0].float()
            m = min(predicted_latent.shape[0], target_latent.shape[0])
            a = predicted_latent[:m].reshape(m, -1)
            b = target_latent[:m].reshape(m, -1)
            cosine = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
            per_dim = []
            for dim in range(min(a.shape[-1], b.shape[-1])):
                x, y = a[:, dim], b[:, dim]
                if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
                    continue
                per_dim.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
            rows.append(
                {
                    "duration_mae_frames": float((predicted_durations[:n] - target_durations[:n]).abs().mean()),
                    "duration_ratio": frames_pred / max(1.0, frames_true),
                    "f0_mae": float((side["f0"][0, :n] - item["f0"][:n].float()).abs().mean()),
                    "energy_mae": float((side["energy"][0, :n] - item["energy"][:n].float()).abs().mean()),
                    "latent_cosine": cosine,
                    "latent_dim_correlation": sum(per_dim) / max(1, len(per_dim)),
                }
            )
    if not rows:
        return {}
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit diagnosis for the text side")
    ap.add_argument("--checkpoint", default="runs/mixed_audio/distill-audio_last.pt")
    ap.add_argument("--cache", default="runs/mixed_train/latent_cache")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--text-mode", default=None, choices=["char", "phoneme"])
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--out", default="runs/fit_diagnosis.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.text_mode:
        cfg.text.mode = args.text_mode
    cache = Path(args.cache)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    geometry = infer_model_geometry(state)
    if "n_voices" in geometry:
        cfg.n_voices = geometry["n_voices"]
    if "latent_head_width" in geometry:
        cfg.autoencoder.latent_rate = max(
            1, geometry["latent_head_width"] // int(cfg.autoencoder.latent_dim)
        )
    model = build_model(cfg)
    model.load_state_dict(state, strict=False)
    model.eval()
    dataset = LatentShardDataset(cache)
    tokenizer = TextTokenizer(mode=cfg.text.mode)

    n = min(args.samples, len(dataset))
    # the cache is written in manifest order, so the first entries are training items; hold out the
    # tail as a proxy for unseen text when there is no val cache at hand
    train_indices = list(range(n))
    validation_indices = list(range(max(0, len(dataset) - n), len(dataset)))
    train = collect(model, dataset, train_indices, tokenizer, cfg)
    validation = collect(model, dataset, validation_indices, tokenizer, cfg)

    checks = {
        "durations_are_not_collapsed": 0.6 < train["duration_ratio"] < 1.6,
        "latent_tokens_correlate_with_the_target": train["latent_cosine"] > 0.5,
        "the_two_splits_are_measurable": bool(train and validation),
    }
    report = {
        "checkpoint": args.checkpoint,
        "cache": cache.as_posix(),
        "text_mode": cfg.text.mode,
        "latent_rate": cfg.autoencoder.latent_rate,
        "samples": n,
        "train_items": train,
        "validation_items": validation,
        "gap": {
            key: train[key] - validation[key] for key in train
        },
        "note": (
            "the train items are the corpus's own utterances; the validation items are the tail of the "
            "cache (a proxy -- a proper val cache would be better).  Both low means underfitting; high "
            "train with low validation means the model memorised."
        ),
        "checks": checks,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"  checkpoint {args.checkpoint} | mode {cfg.text.mode} "
          f"| rate {cfg.autoencoder.latent_rate}")
    header = f"  {'split':10s} | dur MAE | dur ratio | f0 MAE | energy MAE | latent cos | dim corr"
    print(header)
    for name, values in (("train", train), ("validation", validation)):
        print(f"  {name:10s} | {values['duration_mae_frames']:7.2f} | {values['duration_ratio']:9.3f} | "
              f"{values['f0_mae']:6.3f} | {values['energy_mae']:10.3f} | "
              f"{values['latent_cosine']:10.3f} | {values['latent_dim_correlation']:8.3f}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
