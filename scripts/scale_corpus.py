"""Scale the real corpus, and split it by **prompt** so validation text is unseen.

    python scripts/scale_corpus.py --voices 4 --out data/scaled_corpus

Every previous real-audio measurement in this project trained and evaluated on the same 21
utterances, so "WER 0.167 for the autoencoder round-trip" was partly memorisation and the val split
was whatever the code happened to hold.  With the objective finally in the right place (round 23:
train the text side through the decoder), the binding constraint is data.

This script synthesises a larger Kokoro corpus, curates it, and writes **prompt-disjoint**
`train.jsonl` / `val.jsonl` manifests: a prompt goes entirely to one split, because the same sentence
in four voices across a split boundary leaks the text and makes validation meaningless.

Kokoro-82M is Apache-2.0 and runs faster than real time on this CPU, so this is minutes of work, not a
dataset campaign.  All voices in the ``kokoro-en-v0_19`` bundle are ``af_*``; the corpus is therefore
single-gender, which limits what the multi-voice conditioning can be shown to do (recorded, not hidden).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.data.curate import CurateConfig, curate_manifest  # noqa: E402
from parakeet.data.teacher import (  # noqa: E402
    SPEECHIFY_ENGLISH_VOICES,
    build_backend,
    check_teacher,
    synthesize_corpus,
)

#: 60 prompts, deliberately varied: length (8-24 words), clause structure, punctuation, digit-heavy
#: and consonant-heavy sentences.  All comfortably above the curation gate's 3 s minimum so the
#: corpus is not silently filtered down to the short ones.
PROMPTS: List[str] = [
    "The quick brown fox jumps over the lazy dog while the sun sets behind the hills.",
    "A small model can still speak clearly and naturally if it learns the right things.",
    "Every measurement in this project used to be synthetic, and that hid three real defects.",
    "Real speech changes what the quality gates actually see, as the last few weeks showed.",
    "Distilling several teachers into one voice takes patience, careful bookkeeping, and time.",
    "The student learns prosody from cached teacher signals rather than from raw waveforms.",
    "Phase locking is a post filter, not a trained model, which keeps the parameter count low.",
    "Curation decides which utterances are worth training on, so its thresholds had better be right.",
    "Please remember to bring the blue folder, the signed form, and a pen to the meeting tomorrow.",
    "She said she would call back at half past seven, but the phone never rang that evening.",
    "After the storm passed, the village rebuilt the bridge stronger than it had been before.",
    "Numbers matter here: three thousand two hundred and forty one samples were rejected outright.",
    "The engineer adjusted the gain, checked the meter, and wrote the reading in the notebook.",
    "Nobody expected the small model to beat the larger one, and yet the numbers said otherwise.",
    "Between the river and the railway line there is a narrow path that few people ever notice.",
    "He counted the steps out loud, one, two, three, four, five, six, seven, eight, and stopped.",
    "Learning to listen carefully is harder than learning to speak, especially in a noisy room.",
    "The recipe calls for two cups of flour, a pinch of salt, and a spoonful of honey.",
    "Weather like this makes the harbour quiet, and the boats stay tied to their moorings.",
    "A good test fails for the reason you expect, and says so in a way you can act on.",
    "The library closes at six, so we should return these books before the afternoon ends.",
    "Somewhere in the archive there is a photograph of the square before the buildings changed.",
    "She prefers tea to coffee, though she will drink either if the conversation is good.",
    "The machine hums quietly in the corner, doing the same thing a thousand times a second.",
    "Careful measurement beats confident guessing, even when the guessing feels more productive.",
    "They walked through the market, past the fruit stalls, and out towards the old station.",
    "The instructions were clear: measure twice, cut once, and label every piece before moving on.",
    "Rain fell steadily for three days, and by the fourth the fields had turned a deep green.",
    "His handwriting was small and neat, the kind that takes effort to read quickly.",
    "The committee met on Tuesday, argued for two hours, and then agreed to meet again next week.",
    "There is a particular silence in a workshop when everyone is concentrating at once.",
    "She learned the piece by heart, and played it without looking at the pages once.",
    "The train was late, so we bought sandwiches and waited on the cold platform together.",
    "Every spring the swallows return to the same eaves, and every autumn they leave again.",
    "The report was four pages long, but the important part was a single sentence in the middle.",
    "He repaired the clock with a pair of tweezers and a patience that surprised everyone.",
    "A narrow beam of light crossed the room and landed on the dusty piano in the corner.",
    "They agreed that the scheme was clever, impractical, and far too expensive to build.",
    "The map was old, the ink faded, but the coastline was still recognisable after all those years.",
    "Whatever you decide, tell me before Friday, because the order has to be placed on Monday.",
    "The children built a fort out of chairs and blankets and defended it until dinner time.",
    "Fresh snow covered the garden, hiding the paths and softening every edge in sight.",
    "He read the letter twice, folded it carefully, and put it back inside the envelope.",
    "The harbour lights came on one by one as the last of the daylight drained from the sky.",
    "A single wrong number in the table made the whole conclusion look stronger than it was.",
    "She kept a list of things to do, and a second list of things she had already finished.",
    "The audience waited, the curtains moved, and then the orchestra began to play quietly.",
    "Nothing in the manual explained what to do when the indicator flashed twice and stopped.",
    "Long before sunrise, the fishermen were already loading their nets onto the boats.",
    "The kettle boiled, the toast burned slightly, and the morning began in its usual way.",
    "They planted the row of trees along the fence, and watered them through the dry summer.",
    "His argument was simple: the evidence was thin, and the claim was far too broad.",
    "A small dog followed us for a mile and then turned back at the edge of the village.",
    "The tape held, the box survived the journey, and everything inside arrived undamaged.",
    "She wrote the address on the back of her hand so she would not forget it again.",
    "By the time the bell rang, the classroom had gone completely quiet and still.",
    "The old radio crackled, and a voice announced the weather for the following day.",
    "He measured the room, sketched the shelves, and ordered the wood the same afternoon.",
    "Trust the process, check the result, and write down what you changed and why.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a larger prompt-disjoint Kokoro corpus")
    ap.add_argument("--voices", type=int, default=4, help="how many voices to use")
    ap.add_argument("--teacher", default="kokoro", help="teacher name (kokoro, speechify, ...)")
    ap.add_argument("--val-prompts", type=int, default=12, help="prompts held out for validation")
    ap.add_argument("--out", default="data/scaled_corpus")
    ap.add_argument("--prompt-limit", type=int, default=None)
    ap.add_argument("--prompts-file", default=None,
                    help="one prompt per line (e.g. from scripts/build_prompts.py).  Text diversity is "
                         "the measured binding constraint, so a large public-domain list is preferred "
                         "over the built-in 60")
    ap.add_argument("--reuse-audio", action="store_true",
                    help="if the corpus already has a manifest, re-curate the existing wavs instead of "
                         "synthesising again: curation is cheap and synthesis is not (and may be paid)")
    ap.add_argument("--min-bandwidth-hz", type=float, default=None,
                    help="override CurateConfig.min_bandwidth_hz.  Worth overriding with a "
                         "*measurement*: the CosyVoice 5 kHz floor was calibrated on a 24 kHz "
                         "synthesiser and rejects 71%% of a 48 kHz teacher whose DNSMOS is higher")
    ap.add_argument("--min-dnsmos", type=float, default=None,
                    help="override the calibrated DNSMOS floor (default 2.0, measured in round 24)")
    args = ap.parse_args()

    spec = check_teacher(args.teacher)
    backend = build_backend(args.teacher)
    if args.teacher == "speechify":
        available = list(SPEECHIFY_ENGLISH_VOICES)
    else:
        available = list(getattr(backend, "voices", ()))
    voices = available[: args.voices]
    out = Path(args.out)
    corpus = out / "corpus"
    out.mkdir(parents=True, exist_ok=True)

    prompts = PROMPTS[: args.prompt_limit] if args.prompt_limit else PROMPTS
    prompt_source = "built-in list (60 sentences)"
    if args.prompts_file:
        prompts = [
            line.strip()
            for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.prompt_limit:
            prompts = prompts[: args.prompt_limit]
        prompt_source = f"{args.prompts_file} ({len(prompts)} prompts)"
    texts = [t for t in prompts for _ in voices]
    voice_list = [v for _ in prompts for v in voices]
    print(f"teacher {spec.name} ({spec.weights_license}, training allowed: {spec.allows_training})")
    print(f"{len(prompts)} prompts x {len(voices)} voices = {len(texts)} utterances "
          f"({len(set(voices))} voices, all-{voices[0][:2]}*)")

    existing = out if os.path.isdir(corpus) else None
    manifest_path = corpus / "manifest.jsonl"
    if args.reuse_audio and manifest_path.exists():
        print(f"reusing {manifest_path} (no synthesis; curation only)")
        manifest = manifest_path
        synth_seconds = 0.0
    else:
        started = time.perf_counter()
        manifest = synthesize_corpus(
            texts, corpus,
            mix={args.teacher: 1.0},
            voices={args.teacher: voice_list},
            backends={args.teacher: backend},
        )
        synth_seconds = time.perf_counter() - started
    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    audio_seconds = sum(r["duration_s"] for r in records)
    print(f"  synthesised {len(records)} utterances, {audio_seconds / 60:.1f} min of audio "
          f"in {synth_seconds / 60:.1f} min (RTF {synth_seconds / max(audio_seconds, 1e-9):.2f})")

    import soundfile as sf

    def load_wav(rel: str):
        wav, sr = sf.read(str(corpus / rel), dtype="float32")
        return __import__("torch").from_numpy(wav), sr

    curate_cfg = CurateConfig()


    if args.min_bandwidth_hz is not None:


        curate_cfg.min_bandwidth_hz = args.min_bandwidth_hz


    if args.min_dnsmos is not None:


        curate_cfg.min_dnsmos = args.min_dnsmos


    report = curate_manifest(records, load_wav, corpus / "curated", curate_cfg, normalize=True)
    kept = [
        json.loads(l)
        for l in (corpus / "curated" / "kept.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"  curation kept {report.n_kept}/{report.n_total} ({dict(report.reason_counts)})")

    # split by PROMPT: the same sentence in four voices must not straddle the boundary
    by_prompt: Dict[str, List[dict]] = {}
    for record in kept:
        by_prompt.setdefault(record["text"], []).append(record)
    prompt_keys = sorted(by_prompt)
    val_keys = set(prompt_keys[: args.val_prompts])
    train = [r for key in prompt_keys if key not in val_keys for r in by_prompt[key]]
    val = [r for key in prompt_keys if key in val_keys for r in by_prompt[key]]

    for name, rows in (("train", train), ("val", val)):
        (corpus / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

    payload = {
        "teacher": {"name": spec.name, "license": spec.weights_license, "runtime": type(backend).__name__},
        "prompts_source": prompt_source,
        "voices": voices,
        "prompts": len(prompts),
        "synthesised": len(records),
        "kept": report.n_kept,
        "curation_reasons": report.reason_counts,
        "audio_minutes": audio_seconds / 60,
        "synthesis_rtf": synth_seconds / max(audio_seconds, 1e-9),
        "split": {
            "by": "prompt (text), so validation text is unseen",
            "train_utterances": len(train),
            "train_prompts": len(prompt_keys) - len(val_keys),
            "val_utterances": len(val),
            "val_prompts": len(val_keys),
            "train_minutes": sum(r["duration_s"] for r in train) / 60,
            "val_minutes": sum(r["duration_s"] for r in val) / 60,
        },
        "caveats": [
            "all 11 voices in this bundle are af_*: the corpus is single-gender",
            "synthesised speech, so it inherits Kokoro's own artefacts as the ceiling",
        ],
    }
    (out / "corpus_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  split: {payload['split']['train_utterances']} train "
          f"({payload['split']['train_minutes']:.1f} min, {payload['split']['train_prompts']} prompts) | "
          f"{payload['split']['val_utterances']} val "
          f"({payload['split']['val_minutes']:.1f} min, {payload['split']['val_prompts']} prompts)")
    print(f"report -> {out/'corpus_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
