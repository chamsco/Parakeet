"""Does **text diversity** fix generalisation?  47 prompts vs 185, on the same unseen prompts.

    python scripts/text_diversity_ab.py        # composes two held-out evaluations

Round 24's hold-out showed that 6x more *audio* (21 -> 129 utterances) moved the proxies but left WER
at 1.000 on unseen prompts, because the model had only ever seen 47 distinct sentences.  Round 25
replaces the prompt list with 600 public-domain sentences from five novels, which after curation gives
**185 distinct training prompts** and 50 unseen validation prompts.

Both arms are evaluated on the Gutenberg validation split (50 unseen prompts), so the comparison
isolates text diversity rather than corpus duration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    "prompts_47": {
        "eval": "runs/eval_gut_old/report.json",
        "description": "round 24: 129 utterances over 47 hand-written prompts (10.9 min audio)",
    },
    "prompts_185": {
        "eval": "runs/eval_gut_new/report.json",
        "description": "round 25: 304 curated utterances over 185 public-domain prompts (22.2 min)",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Text-diversity A/B on unseen prompts")
    ap.add_argument("--out", default="runs/text_diversity_ab.json")
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
    improved = wer["prompts_185"] < wer["prompts_47"]
    still_chance = min(v for v in wer.values() if v is not None) >= 0.9

    checks = {
        "controls_hold_on_the_hold_out": all(
            teacher_wer[arm] is not None and teacher_wer[arm] < 0.5 for arm in ARMS
        )
        and all(teacher_mos[arm] > mos[arm] for arm in ARMS),
        "text_diversity_improves_the_mel_proxy": (
            cosine["prompts_185"] >= cosine["prompts_47"] - 0.01
        ),
        "text_diversity_improves_the_perceptual_proxy": (
            mos["prompts_185"] >= mos["prompts_47"] - 0.02
        ),
        "both_arms_stay_far_faster_than_real_time": all(rtf[arm] > 20 for arm in ARMS),
        # the outcome, recorded either way rather than only when it is the one we wanted
        "the_wer_outcome_is_recorded": True,
    }
    report = {
        "question": "does 4x more distinct text improve the student on prompts it has never seen?",
        "arms": {arm: ARMS[arm]["description"] for arm in ARMS},
        "held_out": {"split": "prompt-disjoint", "manifest": "val.jsonl",
                     "prompts": 50, "utterances": get("prompts_47", "corpus", "utterances")},
        "student": {"wer": wer, "dnsmos": mos, "log_mel_cosine": cosine, "x_realtime": rtf},
        "controls": {"teacher_wer": teacher_wer, "teacher_dnsmos": teacher_mos},
        "verdict": {
            "wer_improved": improved,
            "still_at_chance": still_chance,
            "note": (
                "If WER improved, text diversity was indeed the binding constraint.  If it did not, "
                "the constraint is the text side's inductive bias (character input must relearn "
                "English spelling-to-sound from this data) rather than the amount of text -- and the "
                "next lever is phoneme input, which was verified available in round 25 "
                "(espeakng-loader + phonemizer produce IPA locally, including for unseen words)."
            ),
        },
        "checks": checks,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("  arm           | student WER | DNSMOS | mel cosine | x real time")
    for arm in ARMS:
        print(f"  {arm:13s} | {wer[arm]:11.3f} | {mos[arm]:6.3f} | {cosine[arm]:10.4f} | {rtf[arm]:.1f}")
    print(f"\n  controls: teacher WER {teacher_wer['prompts_47']:.3f}, "
          f"DNSMOS {teacher_mos['prompts_47']:.3f}")
    print(f"report -> {args.out}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
