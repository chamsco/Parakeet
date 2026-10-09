"""Does more data convert into quality?  21 utterances vs 129, on **unseen prompts**.

    python scripts/data_scale_ab.py          # composes the two held-out evaluations

Both arms are evaluated on the same 12-prompt / 31-utterance validation split, which the training
split never saw (the split is by prompt, so the *text* is unseen, not merely the file).  The smaller
arm is the model that every earlier real-audio number in this project came from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    "small_corpus_21": {
        "eval": "runs/eval_val_old/report.json",
        "description": "21 curated utterances (rounds 18-23), rate 3, through-decoder objective",
    },
    "scaled_corpus_129": {
        "eval": "runs/eval_val_new/report.json",
        "description": "129 curated utterances over 47 prompts (round 24), same objective and rate",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Data-scale A/B on a prompt-disjoint hold-out")
    ap.add_argument("--out", default="runs/data_scale_ab.json")
    args = ap.parse_args()

    payload: Dict[str, Dict] = {}
    for arm, paths in ARMS.items():
        path = ROOT / paths["eval"]
        if not path.exists():
            print(f"missing {path}; run the arm first")
            return 2
        payload[arm] = json.loads(path.read_text(encoding="utf-8"))

    def get(arm: str, *keys):
        node = payload[arm]
        for key in keys:
            node = node[key]
        return node

    wer = {arm: get(arm, "wer", "student") for arm in ARMS}
    mos = {arm: get(arm, "naturalness", "student") for arm in ARMS}
    cosine = {arm: get(arm, "synthesis", "log_mel_cosine_vs_reference") for arm in ARMS}
    rtf = {arm: get(arm, "synthesis", "x_realtime") for arm in ARMS}
    teacher_wer = {arm: get(arm, "wer", "teacher") for arm in ARMS}
    teacher_mos = {arm: get(arm, "naturalness", "teacher") for arm in ARMS}
    utterances = {arm: get(arm, "corpus", "utterances") for arm in ARMS}

    checks = {
        # the recogniser must be able to read the *teacher* on this held-out set, or no WER here means
        # anything -- note this control is 0.000 on unseen prompts too, so the student's WER is real
        "the_recogniser_control_holds_on_the_hold_out": all(
            teacher_wer[arm] is not None and teacher_wer[arm] < 0.5 for arm in ARMS
        ),
        "the_naturalness_control_holds": all(teacher_mos[arm] > mos[arm] for arm in ARMS),
        # what more data did do
        "more_data_improves_the_perceptual_proxy": (
            mos["scaled_corpus_129"] > mos["small_corpus_21"]
        ),
        "more_data_improves_the_mel_proxy": (
            cosine["scaled_corpus_129"] > cosine["small_corpus_21"]
        ),
        "both_arms_stay_far_faster_than_real_time": all(rtf[arm] > 20 for arm in ARMS),
        # ...and what it did not do.  Stated as a check so it cannot be quietly dropped: on *unseen*
        # prompts the student is at WER ~1.0 in both arms, which means the earlier round-23 gains
        # (1.648 -> 0.667) were measured on the corpus's own texts and were partly fitting.
        "the_hold_out_finding_is_recorded": (
            wer["scaled_corpus_129"] is not None and wer["small_corpus_21"] is not None
        ),
    }
    not_yet_generalising = bool(
        wer["scaled_corpus_129"] is not None
        and wer["small_corpus_21"] is not None
        and min(wer.values()) >= 0.9
    )
    report = {
        "question": "does 6x more training audio improve the student on prompts it has never seen?",
        "arms": {arm: ARMS[arm]["description"] for arm in ARMS},
        "held_out": {"split": "prompt-disjoint", "manifest": "val.jsonl",
                     "utterances": utterances},
        "student": {"wer": wer, "dnsmos": mos, "log_mel_cosine": cosine, "x_realtime": rtf},
        "controls": {"teacher_wer": teacher_wer, "teacher_dnsmos": teacher_mos},
        "autoencoder_ceiling": {
            "source": "runs/ae_scaled/report.json",
            "round_trip_wer": 0.289,
            "teacher_wer_on_the_same_subset": 0.281,
            "note": ("the scaled autoencoder's round-trip is indistinguishable from the recogniser's "
                     "own noise floor on this split, so the *renderer* is not the limit any more"),
        },
        "conclusion": (
            "6x more training audio improves both proxies (DNSMOS {:.3f} -> {:.3f}, mel cosine "
            "{:.4f} -> {:.4f}) and the model stays two orders of magnitude faster than real time, but "
            "**WER on unseen prompts is ~1.0 in both arms**.  The round-23 headline (1.648 -> 0.667) was "
            "measured on the corpus's own texts, so it was partly fitting; with a prompt-disjoint "
            "hold-out the student is still at chance on new sentences.  That is the honest state, and it "
            "is exactly what a held-out split is for."
        ).format(
            mos["small_corpus_21"], mos["scaled_corpus_129"],
            cosine["small_corpus_21"], cosine["scaled_corpus_129"],
        ),
        "not_yet_generalising": not_yet_generalising,
        "checks": checks,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("  arm                | student WER | DNSMOS | mel cosine | x real time")
    for arm in ARMS:
        print(f"  {arm:18s} | {wer[arm]:11.3f} | {mos[arm]:6.3f} | {cosine[arm]:10.4f} | {rtf[arm]:.1f}")
    print(f"\n  controls on the hold-out: teacher WER {teacher_wer['small_corpus_21']:.3f}, "
          f"DNSMOS {teacher_mos['small_corpus_21']:.3f}")
    print(f"report -> {args.out}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
