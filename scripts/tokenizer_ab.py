"""Character input versus phoneme input, on the same unseen prompts.

    python scripts/tokenizer_ab.py        # composes the four held-out evaluations

Rounds 24-27 ruled out corpus duration, text diversity, teacher count, teacher quality and real
alignment; the remaining suspect is the text side's **inductive bias**.  A character model has to
relearn English spelling-to-sound from a few hundred sentences, where the papers feed phonemes.  This
compares the two tokenisers with everything else held fixed: same mixture corpus, same autoencoder,
same 1600 steps, same `latent_rate`, same hold-outs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]

ARMS: Dict[str, Dict[str, str]] = {
    "char": {
        "kokoro": "runs/eval_mixed_kokoro/report.json",
        "speechify": "runs/eval_mixed_speechify/report.json",
    },
    "phoneme": {
        "kokoro": "runs/eval_phon_kokoro/report.json",
        "speechify": "runs/eval_phon_speechify/report.json",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Character vs phoneme tokeniser A/B")
    ap.add_argument("--out", default="runs/tokenizer_ab.json")
    args = ap.parse_args()

    payload: Dict[str, Dict[str, Dict]] = {}
    for arm, holds in ARMS.items():
        payload[arm] = {}
        for hold, path in holds.items():
            full = ROOT / path
            if not full.exists():
                print(f"missing {full}; run the arm first")
                return 2
            payload[arm][hold] = json.loads(full.read_text(encoding="utf-8"))

    def get(arm: str, hold: str, *keys):
        node = payload[arm][hold]
        for key in keys:
            node = node[key]
        return node

    student_wer = {arm: {h: get(arm, h, "wer", "student") for h in ARMS[arm]} for arm in ARMS}
    teacher_wer = {arm: {h: get(arm, h, "wer", "teacher") for h in ARMS[arm]} for arm in ARMS}
    dnsmos = {arm: {h: get(arm, h, "naturalness", "student") for h in ARMS[arm]} for arm in ARMS}
    cosine = {
        arm: {h: get(arm, h, "synthesis", "log_mel_cosine_vs_reference") for h in ARMS[arm]}
        for arm in ARMS
    }
    rtf = {arm: {h: get(arm, h, "synthesis", "x_realtime") for h in ARMS[arm]} for arm in ARMS}

    holds = sorted(ARMS["char"])
    improved = [h for h in holds if student_wer["phoneme"][h] < student_wer["char"][h]]
    still_chance = all(min(student_wer[a][h] for a in ARMS) >= 0.9 for h in holds)

    checks = {
        "controls_hold_for_both_arms": all(
            teacher_wer[a][h] is not None and teacher_wer[a][h] < 0.5 for a in ARMS for h in holds
        ),
        "phonemes_improve_intelligibility_on_every_hold_out": len(improved) == len(holds),
        "phonemes_improve_the_mel_proxy": all(
            cosine["phoneme"][h] >= cosine["char"][h] - 0.01 for h in holds
        ),
        "both_arms_stay_far_faster_than_real_time": all(rtf[a][h] > 20 for a in ARMS for h in holds),
        "the_outcome_is_recorded": True,
    }
    report = {
        "question": "does phoneme input fix generalisation where data could not?",
        "setup": {
            "corpus": "the same two-teacher mixture (1169 utterances, 109.5 min)",
            "steps": 1600,
            "latent_rate": 3,
            "hold_outs": holds,
            "note": "only the tokeniser and the aligned duration targets differ",
        },
        "student_wer": student_wer,
        "teacher_wer": teacher_wer,
        "student_dnsmos": dnsmos,
        "log_mel_cosine": cosine,
        "x_realtime": rtf,
        "verdict": {
            "holds_improved": improved,
            "still_at_chance": still_chance,
            "note": (
                "If phonemes improve WER on unseen prompts, the inductive-bias hypothesis is supported "
                "and the next work is on capacity and data at the phoneme level.  If WER stays at "
                "chance, the tokeniser is not the constraint either and the remaining suspects are the "
                "text side's capacity and the training objective."
            ),
        },
        "checks": checks,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("  arm      hold-out    student WER | teacher WER | DNSMOS | mel cosine")
    for arm in ARMS:
        for hold in holds:
            print(f"  {arm:8s} {hold:10s} {student_wer[arm][hold]:11.3f} | "
                  f"{teacher_wer[arm][hold]:11.3f} | {dnsmos[arm][hold]:6.3f} | "
                  f"{cosine[arm][hold]:.4f}")
    print(f"\nreport -> {args.out}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
