"""Mixture sweep: does the teacher mixture actually steer what the student learns?

    python scripts/mixture_demo.py --quick     # ~30 s
    python scripts/mixture_demo.py             # ~2 min

The objective's central claim is *mix training*: distil several teachers into one student, weighted
by the mixture.  A weighting that is only a config value nothing reads would look identical in every
other measurement in this repo, so this demo isolates it.

Two synthetic "teachers" with very different pitch (a low-pitched and a high-pitched voice) provide
the same text; the Tiny text side is trained from an identical initialisation, on identical data,
for an identical number of steps, with only the per-sample mixture weight changing.  If the weights
reach the gradient, the predicted F0 must move monotonically with the mixture share.

Nothing here speaks to audio quality -- it answers a narrower question: is the mixture a mechanism
or a decoration?
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.audio.f0 import normalized_to_f0  # noqa: E402
from parakeet.config import ParakeetConfig, load_config  # noqa: E402
from parakeet.data.features import token_targets_from_corpus  # noqa: E402
from parakeet.data.synthetic import make_corpus  # noqa: E402
from parakeet.data.text import TextTokenizer  # noqa: E402
from parakeet.models import build_model  # noqa: E402
from parakeet.train.stages import tiny_text_step  # noqa: E402


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def tiny_cfg(base: ParakeetConfig, utterances: int) -> ParakeetConfig:
    """A small but real Tiny model, so the sweep is cheap enough to run five times."""
    cfg = copy.deepcopy(base)
    cfg.autoencoder.encoder_dims = [48, 64, 96]
    cfg.autoencoder.encoder_blocks = [1, 2, 2]
    cfg.autoencoder.decoder_dim = 128
    cfg.autoencoder.decoder_blocks = 3
    cfg.text.dim = 128
    cfg.text.n_layers = 3
    cfg.duration.hidden = 128
    cfg.train.lr = 2e-3
    # text.dim is coupled to the flow module's memory width by validation
    cfg.flow.text_dim = cfg.text.dim
    cfg.flow.cond_dim = cfg.text.dim
    cfg.flow.dim = 128
    cfg.flow.n_heads = 4
    return cfg.validate()


def _targets(model, cfg, corpora, weights, tokenizer):
    out = []
    for corpus, weight in zip(corpora, weights):
        for target in token_targets_from_corpus(model, corpus, cfg, tokenizer):
            target["teacher_weight"] = torch.tensor(float(weight))
            out.append(target)
    return out


def _batch(targets, indices, device=None):
    picked = [targets[i] for i in indices]
    ids = torch.stack([t["ids"] for t in picked])
    batch = {
        "ids": ids,
        "text_mask": torch.ones_like(ids, dtype=torch.bool),
        "durations": torch.stack([t["durations"] for t in picked]),
        "f0": torch.stack([t["f0"] for t in picked]),
        "energy": torch.stack([t["energy"] for t in picked]),
        "latent_token": torch.stack([t["latent_token"] for t in picked]),
        "teacher_weight": torch.tensor([float(t["teacher_weight"]) for t in picked]),
    }
    if device:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


@torch.no_grad()
def predicted_f0_hz(model, targets) -> float:
    values = []
    for t in targets:
        ids = t["ids"][None]
        side = model.text_side(ids)
        values.append(normalized_to_f0(side["f0"][0]).mean())
    return float(torch.stack(values).mean().item())


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep the teacher mixture and watch the student follow")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--utterances", type=int, default=4)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="mixture share given to the HIGH-pitched teacher")
    ap.add_argument("--out", default="runs/mixture_demo")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps = 30
        args.weights = [0.0, 0.5, 1.0]

    cfg = tiny_cfg(load_config(args.config), args.utterances)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = TextTokenizer(mode=cfg.text.mode)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    low = make_corpus(args.utterances, cfg.audio, seed=11, f0_range=(80.0, 100.0))
    high = make_corpus(args.utterances, cfg.audio, seed=12, f0_range=(200.0, 230.0))
    low_truth = sum(u.token_f0[0] for u in low) / len(low)
    high_truth = sum(u.token_f0[0] for u in high) / len(high)
    _banner(f"two synthetic teachers: low {low_truth:.0f} Hz vs high {high_truth:.0f} Hz | "
            f"{args.steps} steps/run | {cfg.text.dim}-dim text side")

    runs: List[Dict[str, float]] = []
    t_start = time.perf_counter()
    for share in args.weights:
        torch.manual_seed(cfg.train.seed)  # identical initialisation for every run
        model = build_model(cfg)
        targets = _targets(model, cfg, [low, high], [1.0 - share, share], tokenizer)
        probe_idx = list(range(min(args.batch_size, len(targets))))
        probe_note = "low-pitch samples only (the first corpus, in order)"
        with torch.no_grad():
            initial_loss, _ = tiny_text_step(cfg, model, _batch(targets, probe_idx))
        generator = torch.Generator().manual_seed(0)
        for _ in range(args.steps):
            idx = torch.randint(0, len(targets), (args.batch_size,), generator=generator).tolist()
            loss, _ = tiny_text_step(cfg, model, _batch(targets, idx))
            loss.backward()
            # plain SGD so the comparison does not depend on optimiser state
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p -= cfg.train.lr * p.grad
                        p.grad = None
        with torch.no_grad():
            final_loss, _ = tiny_text_step(cfg, model, _batch(targets, probe_idx))
        f0 = predicted_f0_hz(model, targets)
        runs.append(
            {
                "high_share": float(share),
                "predicted_f0_hz": f0,
                "probe_low_initial_loss": float(initial_loss),
                "probe_low_loss": float(final_loss),
            }
        )
        print(f"  high share {share:4.2f} -> predicted F0 {f0:6.1f} Hz | low-pitch probe loss "
              f"{float(initial_loss):.4f} -> {float(final_loss):.4f}")

    predictions = [r["predicted_f0_hz"] for r in runs]
    losses = [r["probe_low_loss"] for r in runs]
    monotone = all(b >= a - 0.5 for a, b in zip(predictions, predictions[1:]))
    span = predictions[-1] - predictions[0]
    # The probe batch holds *low-pitch* samples, and its loss is weighted by the same mixture.  If the
    # weights reach the gradient, fitting the high teacher must cost fit on the low one, so this loss
    # rises with the high share -- an independent confirmation of the F0 result, not a restatement.
    probe_tracks = all(b >= a - 0.05 for a, b in zip(losses, losses[1:]))
    checks = {
        "mixture_steers_monotonically": monotone,
        "mixture_span_is_meaningful": span > 5.0,
        "every_run_learned": all(r["probe_low_loss"] < r["probe_low_initial_loss"] for r in runs),
        "probe_loss_tracks_mixture": probe_tracks,
    }
    report = {
        "config": args.config,
        "params": sum(p.numel() for p in model.parameters()),
        "teachers": {"low_hz": low_truth, "high_hz": high_truth, "utterances_each": args.utterances},
        "steps_per_run": args.steps,
        "runs": runs,
        "predicted_f0_span_hz": span,
        "checks": checks,
        "seconds_total": time.perf_counter() - t_start,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _banner("RESULT")
    print(f"  predicted F0 across the mixture: "
          f"{' -> '.join(f'{p:.0f}Hz' for p in predictions)}  (span {span:.1f} Hz)")
    print(f"  teachers sit at {low_truth:.0f} Hz (low) and {high_truth:.0f} Hz (high)")
    print(f"  low-pitch probe loss rises with the high share: "
          f"{' -> '.join(f'{l:.2f}' for l in losses)}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nreport -> {out/'report.json'}")
    print("MIXTURE DEMO " + ("PASSED" if all(checks.values()) else "FAILED"))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
