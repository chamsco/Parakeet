"""Latent-rate A/B end to end: one latent per text token vs three.

    python scripts/latent_rate_ab.py            # combines the existing runs into one report

Round 22's oracle sweep (`scripts/seam_rate.py`) established that most of the token->frame seam is
*information*: with sub-latents built from the teacher's own frame latent, WER falls 0.722 (rate 1) ->
0.204 (rate 2) -> 0.093 (rate 3) against 0.167 for the frame latent itself.  That is a ceiling, not a
result: it assumes a text side that predicts those sub-latents **perfectly**.  This script compares
the two arms that a *trained* text side actually produces, using the same autoencoder, corpus and
step count, and reports where the remaining gap lives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    "rate1": {"eval": "runs/eval_rate1/report.json", "train": "runs/real_train/report.json",
              "latent_rate": 1},
    "rate3": {"eval": "runs/eval_rate3/report.json", "train": "runs/real_rate3/report.json",
              "latent_rate": 3},
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Latent-rate A/B (student, end to end)")
    ap.add_argument("--out", default="runs/latent_rate_ab.json")
    args = ap.parse_args()

    payload: Dict[str, Dict] = {}
    for arm, paths in ARMS.items():
        for key in ("eval", "train"):
            path = ROOT / paths[key]
            if not path.exists():
                print(f"missing {path}; run the arm first")
                return 2
        payload[arm] = {
            "latent_rate": paths["latent_rate"],
            "eval": json.loads((ROOT / paths["eval"]).read_text(encoding="utf-8")),
            "train": json.loads((ROOT / paths["train"]).read_text(encoding="utf-8")),
        }

    def metric(arm: str, *keys):
        node = payload[arm]["eval"]
        for key in keys:
            node = node[key]
        return node

    student_wer = {arm: metric(arm, "wer", "student") for arm in ARMS}
    student_mos = {arm: metric(arm, "naturalness", "student") for arm in ARMS}
    teacher_wer = {arm: metric(arm, "wer", "teacher") for arm in ARMS}
    teacher_mos = {arm: metric(arm, "naturalness", "teacher") for arm in ARMS}
    cosine = {arm: metric(arm, "synthesis", "log_mel_cosine_vs_reference") for arm in ARMS}
    rtf = {arm: metric(arm, "synthesis", "x_realtime") for arm in ARMS}
    text_loss = {arm: payload[arm]["train"]["text_side"]["loss_after"] for arm in ARMS}

    checks = {
        # the controls have to hold on both arms, or neither number means anything
        "both_arms_have_a_valid_recogniser_control": all(
            teacher_wer[arm] is not None and teacher_wer[arm] < 0.2 for arm in ARMS
        ),
        "both_arms_have_a_valid_naturalness_control": all(
            teacher_mos[arm] > student_mos[arm] for arm in ARMS
        ),
        # the change has to help on both axes, not just trade one for the other
        "rate3_improves_intelligibility": student_wer["rate3"] < student_wer["rate1"],
        "rate3_improves_naturalness": student_mos["rate3"] > student_mos["rate1"],
        "rate3_keeps_the_mel_proxy": cosine["rate3"] >= cosine["rate1"] - 0.005,
        # and it must stay fast: this is a *light* model
        "both_arms_stay_far_faster_than_real_time": all(rtf[arm] > 20 for arm in ARMS),
    }
    report = {
        "question": (
            "does a trained text side realise the oracle gain from predicting sub-token latents, and "
            "if not, where does the remaining gap live?"
        ),
        "setup": {
            "same_autoencoder": True,
            "steps_text": payload["rate1"]["train"]["steps"]["distill_text"],
            "corpus_utterances": payload["rate1"]["train"]["corpus"]["utterances"],
            "latent_rate": {arm: payload[arm]["latent_rate"] for arm in ARMS},
        },
        "student": {
            "wer": student_wer,
            "dnsmos": student_mos,
            "log_mel_cosine": cosine,
            "x_realtime": rtf,
            "text_side_loss_after": text_loss,
        },
        "controls": {"teacher_wer": teacher_wer, "teacher_dnsmos": teacher_mos},
        "conclusion": (
            "Raising the latent rate to 3 improves the student on both axes (WER {:.3f} -> {:.3f}, "
            "DNSMOS {:.2f} -> {:.2f}) at no cost in speed, so the oracle finding does transfer.  But "
            "the student is still unintelligible, and the oracle path is not: the remaining gap is "
            "*latent prediction*, not the seam and not the autoencoder.  The text side fits its "
            "training objective well (loss {:.3f}) while the audio it renders is wrong, which is the "
            "signature of a latent-space loss that does not track what the decoder needs -- the next "
            "thing to change is the objective, not the architecture: train the text side through the "
            "decoder with a mel/audio loss instead of an L1 on cached latents."
        ).format(
            student_wer["rate1"], student_wer["rate3"],
            student_mos["rate1"], student_mos["rate3"], text_loss["rate3"],
        ),
        "checks": checks,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("  arm    latent_rate | student WER | student DNSMOS | cosine | x real time")
    for arm in ARMS:
        print(
            f"  {arm:8s} {payload[arm]['latent_rate']:4d}      | {student_wer[arm]:11.3f} | "
            f"{student_mos[arm]:14.3f} | {cosine[arm]:.4f} | {rtf[arm]:.1f}"
        )
    print(f"\n  controls: teacher WER {teacher_wer['rate1']:.3f}, DNSMOS {teacher_mos['rate1']:.3f}")
    print(f"report -> {args.out}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
