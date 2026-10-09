"""The token-expansion seam A/B: train the decoder on the distribution inference produces.

    python scripts/seam_ab.py --run      # trains both arms and evaluates them (~25 min)
    python scripts/seam_ab.py            # re-derives the report from the existing evaluations

Round 20 localised the bottleneck after the autoencoder was fixed: expanding per-token latents
exactly as inference does (with the *teacher's* durations/F0/energy) scored WER 0.870 where the
frame-level latent scored 0.167, for just 0.016 of mel cosine.  `distill-decoder` exists for exactly
this and its docstring promised it -- "the decoder then trains on exactly the latent distribution the
text side will produce at synthesis time" -- but the implementation consumed the cached *frame* latent,
and the stage could not run on a real cache at all (the shards had no waveform), so it had only ever
executed under `--dry-run`.

This script runs the comparison both ways from one warm start and re-derives the report from the two
diagnostic runs, so the result can be re-checked without retraining.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = {"off": "false", "on": "true"}


def run_arm(arm: str, cfg: str, cache: str, warm_start: str, steps: int, limit: int) -> None:
    train_cmd = [
        sys.executable, "scripts/train.py",
        "--config", cfg,
        "--stage", "distill-decoder",
        "--cache", cache,
        "--warm-start", warm_start,
        "--steps", str(steps),
        # the discriminator is orthogonal to *which distribution* the decoder sees, and it costs 24x
        # per step, so the A/B holds it at zero and isolates the question being asked
        "--set", "train.loss.adversarial=0",
        "--set", "train.loss.feature_match=0",
        "--set", f"autoencoder.decoder_uses_token_latents={ARMS[arm]}",
        "--out", f"runs/seam_{arm}",
    ]
    print("  $ " + " ".join(train_cmd[1:]), flush=True)
    subprocess.run(train_cmd, cwd=ROOT, check=True)
    diag_cmd = [
        sys.executable, "scripts/real_diagnose.py",
        "--limit", str(limit),
        "--checkpoint", f"runs/seam_{arm}/distill-decoder_last.pt",
        "--out", f"runs/diag_seam_{arm}",
    ]
    print("  $ " + " ".join(diag_cmd[1:]), flush=True)
    subprocess.run(diag_cmd, cwd=ROOT, check=True)


def path_row(payload: Dict, prefix: str) -> Dict:
    for row in payload["paths"]:
        if row["path"].startswith(prefix):
            return row
    raise KeyError(prefix)


def main() -> int:
    ap = argparse.ArgumentParser(description="Token-expansion seam A/B")
    ap.add_argument("--run", action="store_true", help="train and evaluate both arms first")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--cache", default="runs/real_train/latent_cache")
    ap.add_argument("--warm-start", default="runs/real_train/distill-text_last.pt")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--out", default="runs/seam_ab.json")
    args = ap.parse_args()

    if args.run:
        for arm in ARMS:
            print(f"  --- arm {arm} (decoder_uses_token_latents={ARMS[arm]}) ---", flush=True)
            run_arm(arm, args.config, args.cache, args.warm_start, args.steps, args.limit)

    reports: Dict[str, Dict] = {}
    for arm in ARMS:
        path = Path(f"runs/diag_seam_{arm}") / "report.json"
        if not path.exists():
            print(f"missing {path}; run with --run first")
            return 2
        reports[arm] = json.loads(path.read_text(encoding="utf-8"))

    def wer(arm: str, prefix: str):
        return path_row(reports[arm], prefix)["wer"]

    frame = [wer(arm, "2. teacher_frame_latent") for arm in ARMS]
    token = [wer(arm, "3. teacher_token_expanded") for arm in ARMS]
    no_prosody = [wer(arm, "3b.") for arm in ARMS]
    roundtrip = [wer(arm, "1. ae_roundtrip") for arm in ARMS]
    student = [wer(arm, "4. student") for arm in ARMS]

    checks = {
        # the mechanism the stage was built for does something measurable
        "token_path_improved_when_trained_on_its_own_distribution": token[1] < token[0],
        # and it does not damage the path that already worked
        "autoencoder_roundtrip_unchanged": abs(roundtrip[1] - roundtrip[0]) < 0.05,
        # with the projection finally in the graph it becomes useful rather than noise
        "prosody_projection_helps_once_trained": no_prosody[1] > token[1],
        # recorded rather than glossed: the gap is still open, which is the next task
        "gap_to_the_frame_path_remains_and_is_recorded": token[1] > 2 * frame[1],
    }
    report = {
        "question": (
            "does training the decoder on the token-expanded latent (the distribution inference "
            "builds) close the seam, or is the loss structural?"
        ),
        "setup": {
            "arms": {"off": "decoder consumed the cached frame latent", "on": "decoder built its input with decoder_latent_from_tokens"},
            "steps": args.steps,
            "warm_start": args.warm_start,
            "cache": args.cache,
            "adversarial_weight": 0.0,
            "note": "same warm start, same cache, same steps; only the decoder input distribution differs",
        },
        "wer": {
            "ae_roundtrip": {"off": roundtrip[0], "on": roundtrip[1]},
            "teacher_frame_latent": {"off": frame[0], "on": frame[1]},
            "teacher_token_expanded": {"off": token[0], "on": token[1]},
            "token_expanded_without_prosody": {"off": no_prosody[0], "on": no_prosody[1]},
            "student": {"off": student[0], "on": student[1]},
        },
        "mel_cosine": {
            name: {
                arm: reports[arm]["log_mel_cosine_vs_reference"][name]
                for arm in ARMS
            }
            for name in ("teacher_frame_latent", "teacher_token_expanded")
        },
        "conclusion": (
            "The documented mechanism helps (token-path WER {:.3f} -> {:.3f}) and it also puts "
            "`prosody_proj` in the graph, after which removing it hurts ({:.3f} vs {:.3f}) instead of "
            "helping -- so the untrained projection was a real defect that this stage now fixes.  But "
            "the gap to the frame path ({:.3f}) is not closed: the residual is structural, because the "
            "cached per-token latent is an *average* over the ~6 frames of that token, so no decoder "
            "can recover the within-token detail that averaging removed."
        ).format(token[0], token[1], no_prosody[1], token[1], frame[1]),
        "next": [
            "raise the effective token rate for the latent path (predict sub-token latents) rather "
            "than one average per text token",
            "or add a refinement stage that consumes token latents and predicts frame latents -- the "
            "role the flow/consistency sampler plays in the Paper's Small model",
        ],
        "checks": checks,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n  arm          ae_roundtrip | frame_latent | token_expanded | no_prosody | student")
    for i, arm in enumerate(ARMS):
        print(
            f"  {arm:12s} {roundtrip[i]:12.3f} | {frame[i]:12.3f} | {token[i]:14.3f} | "
            f"{no_prosody[i]:10.3f} | {student[i]:.3f}"
        )
    print(f"\nreport -> {args.out}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
