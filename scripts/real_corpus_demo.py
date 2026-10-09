"""Real-teacher corpus: the documented pipeline, on real speech, for the first time.

    python scripts/real_corpus_demo.py --utterances 24

Every previous measurement in this project used synthetic fixtures, because the teachers were
assumed to need a GPU and 6 GB of weights.  That assumption was wrong: **Kokoro-82M is Apache-2.0,
runs faster than real time on this CPU, and installs through sherpa-onnx** (the official `kokoro`
package cannot install here -- it needs ``misaki[en]`` -> spacy -> blis, which has no wheel for this
Python and no Rust toolchain).  Same weights, same licence, different runtime.

So this script runs the real chain: teacher -> corpus -> curation -> latent cache, and reports what
real speech does to each stage.  It is deliberately structural, not a quality claim: no autoencoder
training happens here, and the duration targets are the *uniform fallback* because the sherpa
runtime does not expose Kokoro's per-token timings (the `kokoro` pip pipeline does; see
``SherpaKokoroBackend.durations``).  What it establishes is that the data path works on real audio,
and what real audio does to thresholds that were previously only ever exercised on formants.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.audio import estimate_f0  # noqa: E402
from parakeet.config import load_config  # noqa: E402
from parakeet.data.curate import CurateConfig, curate_manifest  # noqa: E402
from parakeet.data.features import cache_teacher_corpus  # noqa: E402
from parakeet.data.teacher import build_backend, check_teacher, synthesize_corpus  # noqa: E402
from parakeet.models import build_model  # noqa: E402

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A small model can still speak clearly and naturally.",
    "Every measurement in this project used to be synthetic.",
    "Real speech changes what the quality gates actually see.",
    "Distilling several teachers into one voice takes patience.",
    "The student learns prosody from cached teacher signals.",
    "Phase locking is a post filter, not a trained model.",
    "Curation decides which utterances are worth training on.",
]
VOICES = ["af_heart", "af_bella", "af_sky", "af_sarah"]


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the real teacher data path end to end")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--voices", type=int, default=4)
    ap.add_argument("--out", default="data/real_corpus")
    ap.add_argument("--report", default="runs/real_corpus_report.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t_start = time.perf_counter()
    strict: Dict[str, bool] = {}

    # ---------------- 1. teacher ----------------
    spec = check_teacher("kokoro")
    backend = build_backend("kokoro")
    _banner(f"1/4 real teacher: {spec.name} ({spec.kind}) via {type(backend).__name__}")
    print(f"  licence: {spec.weights_license} | allows_training={spec.allows_training}")
    print(f"  sample_rate={backend.sample_rate} | voices={len(getattr(backend, 'voices', ()))}")
    strict["teacher_is_permitted"] = bool(spec.allows_training and not getattr(spec, "restricted", False))
    strict["teacher_is_real_not_a_fixture"] = spec.kind != "local_fixture"

    prompts = PROMPTS[: args.prompts]
    voices = VOICES[: args.voices]
    _banner(f"2/4 corpus: {len(prompts)} prompts x {len(voices)} voices, real 24 kHz speech")
    t0 = time.perf_counter()
    manifest = synthesize_corpus(
        [t for t in prompts for _ in voices],
        out / "corpus",
        mix={"kokoro": 1.0},
        voices={"kokoro": voices},
        backends={"kokoro": backend},
    )
    synth_s = time.perf_counter() - t0
    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    total_audio = sum(r["duration_s"] for r in records)
    print(f"  -> {len(records)} utterances, {total_audio:.1f}s audio in {synth_s:.1f}s "
          f"(RTF {synth_s / max(total_audio, 1e-9):.2f})")
    strict["corpus_has_real_audio"] = len(records) > 0 and total_audio > 5.0

    # ---------------- 2. signal statistics on real speech ----------------
    import soundfile as sf

    wavs = [torch.from_numpy(sf.read(str(out / "corpus" / r["wav_path"]), dtype="float32")[0])
            for r in records]
    _banner("3/4 what real speech does to the pitch tracker (round-17 threshold fix)")
    per_utterance: List[Dict[str, float]] = []
    for record, wav in zip(records, wavs):
        wav_t = wav.reshape(1, -1)
        f0, voiced, conf = estimate_f0(wav_t, backend.sample_rate, hop_length=256, frame_length=2048)
        f0_old, voiced_old, _ = estimate_f0(
            wav_t, backend.sample_rate, hop_length=256, frame_length=2048, threshold=0.25
        )
        auto, auto_voiced, _ = estimate_f0(
            wav_t, backend.sample_rate, hop_length=256, frame_length=2048, method="autocorr"
        )
        median_f0 = float(f0[voiced].median()) if bool(voiced.any()) else float("nan")
        per_utterance.append(
            {
                "utt_id": record["utt_id"],
                "voice": record["voice"],
                "seconds": record["duration_s"],
                "voiced_default": float(voiced.float().mean()),
                "voiced_old_threshold": float(voiced_old.float().mean()),
                "voiced_autocorr": float(auto_voiced.float().mean()),
                "median_f0_hz": median_f0,
                "median_f0_old_threshold": (
                    float(f0_old[voiced_old].median()) if bool(voiced_old.any()) else float("nan")
                ),
                "confidence": float(conf[voiced].median()) if bool(voiced.any()) else float("nan"),
            }
        )
    voiced_default = statistics.mean(p["voiced_default"] for p in per_utterance)
    voiced_old = statistics.mean(p["voiced_old_threshold"] for p in per_utterance)
    voiced_auto = statistics.mean(p["voiced_autocorr"] for p in per_utterance)
    f0s = [p["median_f0_hz"] for p in per_utterance if p["median_f0_hz"] == p["median_f0_hz"]]
    print(f"  voiced frames: default threshold {voiced_default:.2f} | old 0.25 {voiced_old:.2f} "
          f"| autocorr {voiced_auto:.2f}")
    print(f"  median F0 across {len(f0s)} utterances: {min(f0s):.0f}-{max(f0s):.0f} Hz "
          f"(median {statistics.median(f0s):.0f} Hz)")
    strict["real_speech_is_detected_as_voiced"] = voiced_default > 0.4
    strict["threshold_fix_helps_real_speech"] = voiced_default > voiced_old
    strict["f0_is_plausible_for_speech"] = 70.0 < statistics.median(f0s) < 350.0

    # ---------------- 3. curation with the published gates ----------------
    _banner("4/4 curation (published thresholds) + latent cache")

    def load_wav(rel: str):
        wav, sr = sf.read(str(out / "corpus" / rel), dtype="float32")
        return torch.from_numpy(wav), sr

    # curation output goes *inside* the corpus directory, which is the layout the corpus builder
    # uses and what cache_teacher_corpus looks for -- otherwise the cache silently trains on the
    # rejected utterances instead of the kept ones
    report = curate_manifest(records, load_wav, out / "corpus" / "curated", CurateConfig(), normalize=True)
    kept = [
        json.loads(l)
        for l in (out / "corpus" / "curated" / "kept.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    rejected = [
        json.loads(l)
        for l in (out / "corpus" / "curated" / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"  kept {report.n_kept}/{report.n_total} | rejected {report.n_rejected} "
          f"| reasons {dict(report.reason_counts)}")
    if rejected:
        example = rejected[0]
        print(f"  example rejection: {example['utt_id']} {example['quality']['reasons']}")
    if kept:
        rms = statistics.mean(r["quality"]["rms_db"] for r in kept)
        bandwidth = statistics.mean(r["quality"]["bandwidth99_hz"] for r in kept)
        snrs = [r["quality"]["snr_db"] for r in kept if r["quality"]["snr_db"] is not None]
        snr_text = f"{statistics.mean(snrs):.1f} dB" if snrs else "unevaluable on this audio"
        print(f"  kept-set rms {rms:.1f} dB | bandwidth99 {bandwidth:.0f} Hz | snr {snr_text}")
    strict["curation_ran_on_real_speech"] = report.n_total == len(records)

    cfg = load_config(args.config)
    cfg.n_voices = len(voices)
    model = build_model(cfg)
    cache = cache_teacher_corpus(out / "corpus", out / "latent_cache", cfg, model.autoencoder)
    meta = json.loads((cache / "cache_meta.json").read_text(encoding="utf-8"))
    from parakeet.data.dataset import LatentShardDataset
    from parakeet.audio.f0 import normalized_to_f0

    dataset = LatentShardDataset(cache)
    target_f0 = [float(normalized_to_f0(dataset[i]["f0"]).median()) for i in range(len(dataset))]
    frames = [int(dataset[i]["n_frames"]) for i in range(len(dataset))]
    print(f"  cache: {len(dataset)} items (from the curated manifest, {report.n_kept} kept of "
          f"{report.n_total}) | teachers {meta['teacher_names']} | voices {meta['voice_names']}")
    print(f"  per-token F0 targets: {min(target_f0):.0f}-{max(target_f0):.0f} Hz | "
          f"latent frames {min(frames)}-{max(frames)}")
    strict["cache_built_from_real_audio"] = len(dataset) > 0 and bool(meta["voice_names"])
    strict["cache_uses_the_curated_set"] = len(dataset) == report.n_kept
    strict["targets_are_plausible"] = all(60.0 < v < 400.0 for v in target_f0)

    payload = {
        "teacher": {
            "name": spec.name,
            "kind": spec.kind,
            "runtime": type(backend).__name__,
            "weights_license": spec.weights_license,
            "allows_training": spec.allows_training,
        },
        "corpus": {
            "utterances": len(records),
            "audio_seconds": total_audio,
            "voices": voices,
            "synthesis_rtf": synth_s / max(total_audio, 1e-9),
        },
        "pitch_tracking": {
            "voiced_default_threshold": voiced_default,
            "voiced_old_threshold_0_25": voiced_old,
            "voiced_autocorr": voiced_auto,
            "median_f0_range_hz": [min(f0s), max(f0s)],
            "per_utterance": per_utterance,
        },
        "curation": {
            "n_total": report.n_total,
            "n_kept": report.n_kept,
            "n_rejected": report.n_rejected,
            "reason_counts": report.reason_counts,
            "note": "published thresholds, first time applied to real speech rather than fixtures",
        },
        "cache": {
            "items": len(dataset),
            "teacher_names": meta["teacher_names"],
            "voice_names": meta["voice_names"],
            "per_token_f0_hz_range": [min(target_f0), max(target_f0)],
            "latent_frames_range": [min(frames), max(frames)],
        },
        "duration_targets": (
            "UNIFORM FALLBACK: the sherpa-onnx runtime does not expose Kokoro's per-token timings, so "
            "duration targets here are an even split, not the teacher's own timing.  The `kokoro` pip "
            "pipeline supplies them but cannot be installed in this environment (misaki->spacy->blis)."
        ),
        "checks": strict,
        "caveat": (
            "Structural, not a quality claim: no autoencoder training happens here and the real "
            "corpus is small.  It shows that real 24 kHz speech flows through the documented data "
            "path, and what real speech does to thresholds previously exercised only on formants."
        ),
        "seconds_total": time.perf_counter() - t_start,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _banner("RESULT")
    for name, ok in strict.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nreport -> {report_path}")
    print("REAL CORPUS " + ("PASSED" if all(strict.values()) else "FAILED"))
    return 0 if all(strict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
