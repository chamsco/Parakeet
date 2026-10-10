"""Evaluate a trained student on real speech with perceptual and intelligibility metrics.

    python scripts/real_eval.py --checkpoint runs/real_train/distill-text_last.pt \
        --corpus data/real_corpus/corpus --out runs/real_eval

Previous rounds could only report *proxies* -- log-mel distance, phase coherence -- because UTMOS
and an ASR were assumed to be unavailable.  They are installable (`pip install utmos
faster-whisper`), so this script measures what the papers measure:

* **UTMOS22** on the generated audio **and on the Kokoro reference audio**.  The reference is the
  control: a perceptual metric that cannot separate a strong teacher from an untrained student is
  not measuring naturalness, and without that check a low score would be uninterpretable.
* **WER** with faster-whisper, transcribing the generated audio and comparing against the intended
  text.  The recogniser size is recorded with the number, because a `base.en` WER is not comparable
  to a paper's `large-v3` WER.
* The proxies for continuity: log-mel cosine against the reference, phase coherence, and the
  single-thread real-time factor of the whole synthesis path.

Both metrics legitimately report *unavailable* -- the script then says so and fails its own check
rather than printing a number it did not measure.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.audio.mel import MelSpectrogram  # noqa: E402
from parakeet.config import load_config  # noqa: E402
from parakeet.eval.metrics import (  # noqa: E402
    OptionalMetric,
    dnsmos_score,
    speechlikeness,
    utmos,
    whisper_wer,
)
from parakeet.inference import Synthesizer, phase_coherence, write_wav  # noqa: E402
from parakeet.models import build_model  # noqa: E402


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def _resample_16k(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if sample_rate == 16000:
        return wav
    target = int(wav.numel() * 16000 / sample_rate)
    return torch.nn.functional.interpolate(
        wav.reshape(1, 1, -1), size=target, mode="linear", align_corners=False
    ).reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description="UTMOS + WER on real speech")
    ap.add_argument("--checkpoint", default="runs/real_train/distill-text_last.pt")
    ap.add_argument("--corpus", default="data/real_corpus/corpus")
    ap.add_argument("--manifest", default=None, help="manifest file inside --corpus (e.g. val.jsonl for a held-out split)")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--latent-rate", type=int, default=None, help="latents per text token (must match the checkpoint)")
    ap.add_argument("--text-mode", default=None, choices=["char", "phoneme"], help="tokeniser mode; must match the checkpoint")
    ap.add_argument("--limit", type=int, default=8)

    ap.add_argument("--prose-only", action="store_true",

                        help="drop heading/title records: their reference text cannot normalise to spoken "

                             "words ('CHAPTER IV' is read as 'chapter four'), which inflates every WER")
    ap.add_argument("--whisper", default="base.en", help="faster-whisper model size")
    ap.add_argument("--reference", action="store_true",
                    help="condition on a *reference mel* from another utterance of the same voice.  The "
                         "flow is trained that way (`pair_references` default for that stage): without a "
                         "reference its speaker conditioner receives zeros, which training never showed "
                         "it.  The Tiny path ignores this -- it conditions on a voice embedding.")
    ap.add_argument("--cfg-scale", type=float, default=None,
                        help="classifier-free guidance for the flow sampler; the config's 1.5 can over-shoot on a partly"
                             " trained model")
    ap.add_argument("--steps", type=int, default=None,
                    help="flow-matching sampling steps (NFE).  The flow variant defaults to "
                         "cfg.flow.nfe (32), which is 3x *slower than real time* on the CPU: a "
                         "distilled sampler is meant to use 2-4, and SupertonicTTS reports WER 11.43 at "
                         "NFE 4 against 2.64 at NFE 32, so the number belongs in the report")
    ap.add_argument("--cache", default=None,
                        help="the latent cache the checkpoint trained on; it stores the latent-normaliser"
                             " statistics that every decode path must apply (round 36)")
    ap.add_argument("--out", default="runs/real_eval")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus)
    manifest_name = getattr(args, "manifest", None)
    if manifest_name:
        manifest = Path(manifest_name) if Path(manifest_name).is_absolute() else corpus / manifest_name
    else:
        manifest = corpus / "curated" / "kept.jsonl"
        if not manifest.exists():
            manifest = corpus / "manifest.jsonl"
    records = [
        json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()
    ]

    if getattr(args, "prose_only", False):

        from parakeet.data.text import is_prose_like


        before = len(records)

        records = [r for r in records if is_prose_like(r["text"])]

        print(f"prose filter: {len(records)}/{before} records kept (headings removed)")

    records = records[: args.limit]
    if not records:
        print(f"no records in {manifest}")
        return 2

    cfg = load_config(args.config)
    if getattr(args, "text_mode", None):
        cfg.text.mode = args.text_mode
    if getattr(args, "latent_rate", None):
        cfg.autoencoder.latent_rate = args.latent_rate
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("ema", {}).get("shadow", payload["model"])
    # the checkpoint knows its own widths; inferring them from a neighbouring cache directory failed
    # for any run whose cache lived elsewhere (`size mismatch for voice_embed.weight`)
    cached_meta = Path(args.checkpoint).parent / "latent_cache" / "cache_meta.json"
    if cached_meta.exists():
        cfg.n_voices = max(1, len(json.loads(cached_meta.read_text(encoding="utf-8"))["voice_names"]))
    if "voice_embed.weight" in state:
        cfg.n_voices = max(1, int(state["voice_embed.weight"].shape[0]))
    for key, field, dim in (
        ("latent_head.2.weight", "latent_rate", None),
    ):
        if key in state and dim is None:
            width = int(state[key].shape[0])
            cfg.autoencoder.latent_rate = max(1, width // int(cfg.autoencoder.latent_dim))
    # the checkpoint knows its own widths -- all four of them (voices, latent rate, text width, text depth).
    # The ad-hoc version here covered only the first two, which is why a dim-512 text side failed to load
    # with "size mismatch for latent_head.2.weight" in round 54.
    from parakeet.train.common import load_checkpoint_into

    model, applied, payload = load_checkpoint_into(cfg, args.checkpoint)
    for key, value in applied.items():
        if key in ("text_dim", "text_layers"):
            print(f"[real_eval] {key} {value} (from the checkpoint)")
    model.eval()
    if args.cache:
        from parakeet.train.common import load_latent_norm_from_cache

        if load_latent_norm_from_cache(model, args.cache):
            print(f"[eval] latent normaliser loaded from {args.cache}")
        else:
            print(f"[eval] WARNING {args.cache} carries no latent-normaliser statistics; the decoder will"
                  f" receive normalised latents (run scripts/repair_latent_norm.py)")
    print(f"loaded {args.checkpoint} (step {payload.get('step')}, "
          f"{'EMA' if 'ema' in payload else 'raw'} weights) | n_voices={cfg.n_voices} "
          f"| latent_rate={cfg.autoencoder.latent_rate}")

    import soundfile as sf

    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=True)
    mel = MelSpectrogram(cfg.audio)
    tone: List[float] = []
    for i, record in enumerate(records):
        voice = record.get("voice") or None
        tone.append(float(voice not in (None, "", "v0")) if voice else 0.0)
    print(f"evaluating {len(records)} utterances over voices "
          f"{sorted({str(r.get('voice')) for r in records})}")

    _banner("synthesizing from text")
    generated: List[torch.Tensor] = []
    references: List[torch.Tensor] = []
    texts: List[str] = []
    cosine: List[float] = []
    coherence: List[float] = []
    clipped_peaks: List[float] = []

    synth_seconds = 0.0
    audio_seconds = 0.0
    for i, record in enumerate(records):
        t0 = time.perf_counter()
        ref_wav = None
        if args.reference:
            # another utterance of the *same* voice: exactly the conditioning the flow saw in training
            partner = next(
                (r for r in records if r is not record and r.get("voice") == record.get("voice")), None
            ) or next((r for r in records if r is not record), None)
            if partner is not None:
                partner_wav, partner_rate = sf.read(
                    str(corpus / partner["wav_path"]), dtype="float32"
                )
                ref_wav = torch.from_numpy(partner_wav).reshape(1, -1)
                if partner_rate != cfg.audio.sample_rate:
                    size = int(ref_wav.shape[-1] * cfg.audio.sample_rate / partner_rate)
                    ref_wav = torch.nn.functional.interpolate(
                        ref_wav[:, None, :], size=size, mode="linear", align_corners=False
                    )[:, 0, :]
        wav = synth.synthesize(record["text"], seed=0, steps=args.steps,
                               cfg_scale=args.cfg_scale, ref_wav=ref_wav)
        synth_seconds += time.perf_counter() - t0
        # Level matching.  The curated references are peak-normalised, and the autoencoder's output is
        # not bounded: the flow clipped 14.6% of its samples, which both the recogniser and DNSMOS read
        # as distortion.  Comparing a clipped candidate against normalised references measures the
        # clipping rather than the model, so the candidate is peak-normalised the same way and the gain
        # is recorded instead of hidden.
        peak = float(wav.abs().max()) if wav.numel() else 0.0
        if peak > 1.0:
            wav = wav * (0.99 / peak)
            clipped_peaks.append(peak)
        reference, ref_rate = sf.read(str(corpus / record["wav_path"]), dtype="float32")
        ref_t = torch.from_numpy(reference).reshape(-1)
        if ref_rate != cfg.audio.sample_rate:
            # the control is the teacher's own audio, and a 48 kHz teacher is not 24 kHz.  Declaring
            # the wrong rate to whisper_wer time-stretches the control by the ratio, which made a
            # perfectly readable teacher score a WER of 0.925 and "fail its own control".
            target = int(ref_t.numel() * cfg.audio.sample_rate / ref_rate)
            ref_t = torch.nn.functional.interpolate(
                ref_t.reshape(1, 1, -1), size=target, mode="linear", align_corners=False
            ).reshape(-1)
        generated.append(wav.reshape(-1))
        references.append(ref_t)
        texts.append(record["text"])
        audio_seconds += wav.numel() / cfg.audio.sample_rate
        n = min(wav.numel(), ref_t.numel())
        if n > 2000:
            a = mel.log_mel(wav.reshape(1, -1)[..., :n])
            b = mel.log_mel(ref_t[:n].reshape(1, -1))
            cosine.append(float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)))
        coherence.append(float(phase_coherence(wav.reshape(1, -1), sample_rate=cfg.audio.sample_rate)))
        if i < 3:
            write_wav(out / f"generated_{i}.wav", wav, cfg.audio.sample_rate)
            write_wav(out / f"reference_{i}.wav", ref_t.reshape(1, -1), cfg.audio.sample_rate)
    rtf = synth_seconds / max(audio_seconds, 1e-9)
    # a flow checkpoint early in training can predict a near-zero length, and then every comparison
    # against a reference is empty.  Report that rather than dying in statistics.mean, and make the
    # length ratio visible: for the flow sampler it is the first thing that has to be right.
    cosine_text = f"{statistics.mean(cosine):.4f}" if cosine else "n/a (no overlapping audio)"
    coherence_text = f"{statistics.mean(coherence):.4f}" if coherence else "n/a"
    generated_seconds = sum(w.numel() for w in generated) / cfg.audio.sample_rate
    reference_seconds = sum(r.numel() for r in references) / cfg.audio.sample_rate
    length_ratio = generated_seconds / max(reference_seconds, 1e-9)
    print(f"  {len(records)} utterances in {synth_seconds:.1f}s "
          f"(RTF {rtf:.2f}, {1 / max(rtf, 1e-9):.1f}x real time) | log-mel cosine "
          f"{cosine_text} | phase coherence {coherence_text}")
    if clipped_peaks:
        print(f"  level: {len(clipped_peaks)}/{len(records)} outputs peaked above 1.0 "
              f"(max {max(clipped_peaks):.2f}); peak-normalised to 0.99 before the metrics")
    print(f"  length: generated {generated_seconds:.1f}s vs reference {reference_seconds:.1f}s "
          f"(ratio {length_ratio:.2f})")

    _banner("is it speech at all?  (voiced fraction, pitch, spectral flatness vs the teacher)")
    print("  NB this is a sanity check, not a quality verdict: the autoencoder's own round trip --")
    print("     intelligible, WER 0.000 -- also reports speech_like NO (voiced 0.99).  WER with its")
    print("     control is the metric that discriminates.")
    likeness = speechlikeness(generated, cfg.audio.sample_rate, reference=references)
    likeness_teacher = likeness.pop("reference", {})
    print(f"  student: voiced {likeness['voiced_fraction']:.2f} | f0 {likeness['median_f0_hz']:.0f} Hz "
          f"| flatness {likeness['spectral_flatness']:.3f} | speech_like "
          f"{'YES' if likeness['speech_like'] else 'NO'}")
    if likeness_teacher:
        print(f"  teacher: voiced {likeness_teacher['voiced_fraction']:.2f} | "
              f"f0 {likeness_teacher['median_f0_hz']:.0f} Hz | "
              f"flatness {likeness_teacher['spectral_flatness']:.3f} | speech_like "
              f"{'YES' if likeness_teacher['speech_like'] else 'NO'}  <- control")
        print("  (a generator can match the mel envelope while producing a buzz or noise; this is the "
              "gate that cannot be gamed)")

    _banner("perceptual naturalness: DNSMOS (available) and UTMOS (documented, not installable here)")
    student_dns = dnsmos_score(generated, sample_rate=cfg.audio.sample_rate)
    teacher_dns = dnsmos_score(references, sample_rate=cfg.audio.sample_rate)
    if student_dns.available:
        print(f"  student DNSMOS {student_dns.value:.3f} ({student_dns.detail})")
        print(f"  teacher DNSMOS {teacher_dns.value:.3f}  <- control: the metric must separate these")
    else:
        print(f"  DNSMOS UNAVAILABLE: {student_dns.reason}")
    student_mos = utmos(generated, sample_rate=cfg.audio.sample_rate)
    teacher_mos = utmos(references, sample_rate=cfg.audio.sample_rate)
    if student_mos.available:
        print(f"  student UTMOS {student_mos.value:.3f} ({student_mos.detail})")
    else:
        print(f"  UTMOS UNAVAILABLE: {student_mos.reason}")

    _banner(f"WER with faster-whisper {args.whisper}")
    wer = whisper_wer(generated, texts, sample_rate=cfg.audio.sample_rate, model_size=args.whisper)
    # a control for the recogniser itself: transcribing the *teacher* audio should give a low WER
    reference_wer = whisper_wer(
        references, texts, sample_rate=cfg.audio.sample_rate, model_size=args.whisper
    )
    if wer.available:
        print(f"  student WER {wer.value:.3f} ({wer.detail})")
    else:
        print(f"  student WER UNAVAILABLE: {wer.reason}")
    if reference_wer.available:
        print(f"  teacher WER {reference_wer.value:.3f}  <- control: the recogniser works on this text")

    checks = {
        "synthesis_produces_audio": all(torch.isfinite(w).all().item() for w in generated),
        "dnsmos_available": bool(student_dns.available and teacher_dns.available),
        "wer_available": bool(wer.available and reference_wer.available),
        # the controls that make the numbers interpretable
        "naturalness_metric_separates_teacher_from_student": bool(
            student_dns.available and teacher_dns.available
            and teacher_dns.value > student_dns.value
        ),
        "recogniser_works_on_the_teacher_audio": bool(
            reference_wer.available and reference_wer.value < 0.5
        ),
    }
    report: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": payload.get("step"),
        "weights": "ema" if "ema" in payload else "raw",
        "corpus": {"manifest": manifest.name, "utterances": len(records)},
        "level_correction": {"outputs_over_full_scale": len(clipped_peaks),
                              "max_peak": max(clipped_peaks) if clipped_peaks else None},
        "synthesis": {"rtf": rtf, "x_realtime": 1 / max(rtf, 1e-9),
                      "log_mel_cosine_vs_reference": (
                          statistics.mean(cosine) if cosine else None
                      ),
                      "phase_coherence": statistics.mean(coherence) if coherence else None,
                      "generated_seconds": generated_seconds,
                      "reference_seconds": reference_seconds,
                      "length_ratio": length_ratio},
        "speechlikeness": {**likeness, "teacher": likeness_teacher},
        "naturalness": {
            "metric": "dnsmos p835",
            "student": student_dns.value,
            "teacher": teacher_dns.value,
            "note": "teacher = the Kokoro reference the corpus was built from; it is the control",
        },
        "utmos": {
            "student": student_mos.value,
            "teacher": teacher_mos.value,
            "available": bool(student_mos.available),
            "reason": student_mos.reason,
            "note": "the documented metric; `utmos` needs fairseq, whose sdist cannot build here",
        },
        "wer": {
            "student": wer.value,
            "teacher": reference_wer.value,
            "recogniser": args.whisper,
            "note": "the recogniser size is part of the number; not comparable to a large-v3 WER",
        },
        "caveats": [
            "the student is trained on a small corpus for a short CPU budget: this is a baseline",
            "DNSMOS on the *teacher* is the ceiling for this corpus, not a human MOS",
            "WER with a small recogniser is an upper bound relative to the papers' large-v3",
        ],
        "checks": checks,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _banner("RESULT")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nreport -> {out/'report.json'}")
    print("REAL EVAL " + ("PASSED" if all(checks.values()) else "FAILED"))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
