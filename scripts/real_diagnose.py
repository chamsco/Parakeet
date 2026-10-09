"""Where does the text -> audio chain actually break?  Four paths, measured.

    python scripts/real_diagnose.py --limit 6

Round 18 trained a student on real speech and measured WER 1.00 against a teacher that scores 0.00
with the same recogniser: the output is unintelligible.  "The model is bad" is not a diagnosis.  The
chain is

    text -> text side -> per-token (duration, f0, energy, latent) -> frame latent -> decoder -> wav

and each seam can be exercised with *teacher* inputs, which localises the failure:

===============================  ==========================================================
path                             what it isolates
===============================  ==========================================================
1. ae_roundtrip                  the autoencoder's own ceiling: encode real audio, decode it
2. teacher_frame_latent          the cached frame-level latent -> decoder, no token path
3. teacher_token_expanded        per-token latents expanded exactly as inference does, with
                                 the teacher's own durations/f0/energy -> isolates the
                                 expansion + prosody + denormalisation seam
4. student                       the real thing: text in, audio out
===============================  ==========================================================

If (1) is unintelligible the autoencoder is the bottleneck and no amount of text-side work helps.
If (3) is bad but (2) is good, the seam between the halves is broken.  If (4) is bad but (3) is good,
it is prediction error.  The report also prints the duration and latent statistics that would
explain each case.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.audio.mel import MelSpectrogram  # noqa: E402
from parakeet.config import load_config  # noqa: E402
from parakeet.data.dataset import LatentShardDataset  # noqa: E402
from parakeet.eval.metrics import dnsmos_score, whisper_wer  # noqa: E402
from parakeet.inference import Synthesizer, write_wav  # noqa: E402
from parakeet.models import build_model  # noqa: E402
from parakeet.train.common import infer_model_geometry  # noqa: E402
from parakeet.models.duration import normalized_to_durations  # noqa: E402


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def _mel_cosine(mel: MelSpectrogram, a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(a.numel(), b.numel())
    if n < 2000:
        return float("nan")
    return float(
        torch.nn.functional.cosine_similarity(
            mel.log_mel(a.reshape(1, -1)[..., :n]).flatten(),
            mel.log_mel(b.reshape(1, -1)[..., :n]).flatten(),
            dim=0,
        )
    )


def measure(path_name: str, audios: List[torch.Tensor], texts: Sequence[str], cfg, whisper: str) -> Dict:
    wer = whisper_wer(audios, texts, sample_rate=cfg.audio.sample_rate, model_size=whisper)
    mos = dnsmos_score(audios, sample_rate=cfg.audio.sample_rate)
    return {
        "path": path_name,
        "wer": wer.value,
        "wer_available": wer.available,
        "dnsmos": mos.value,
        "dnsmos_available": mos.available,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Localise the text->audio failure with teacher inputs")
    ap.add_argument("--checkpoint", default="runs/real_train/distill-text_last.pt")
    ap.add_argument("--cache", default="runs/real_train/latent_cache")
    ap.add_argument("--corpus", default="data/real_corpus/corpus")
    ap.add_argument("--manifest", default=None, help="manifest file inside --corpus (e.g. val.jsonl for a held-out split)")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--latent-rate", type=int, default=None, help="latents per text token (must match the checkpoint)")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--whisper", default="base.en")
    ap.add_argument("--out", default="runs/real_diagnose")
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
    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    dataset = LatentShardDataset(args.cache)
    meta = json.loads((Path(args.cache) / "cache_meta.json").read_text(encoding="utf-8"))
    cfg = load_config(args.config)
    if getattr(args, "latent_rate", None):
        cfg.autoencoder.latent_rate = args.latent_rate
    cfg.n_voices = max(1, len(meta["voice_names"]))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    # build the model to match the checkpoint, not the config: `strict=False` does not tolerate a
    # shape mismatch, and this is the second script to fail on that (see infer_model_geometry)
    geometry = infer_model_geometry(state)
    if "n_voices" in geometry:
        cfg.n_voices = geometry["n_voices"]
    if "latent_head_width" in geometry and not getattr(args, "latent_rate", None):
        cfg.autoencoder.latent_rate = max(
            1, geometry["latent_head_width"] // int(cfg.autoencoder.latent_dim)
        )
    model = build_model(cfg)
    model.load_state_dict(state, strict=False)
    model.eval()
    mel = MelSpectrogram(cfg.audio)
    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=True)
    print(f"checkpoint step {payload.get('step')} | {len(dataset)} cached items | "
          f"{len(records)} corpus records | voices {meta['voice_names']}")

    import soundfile as sf

    ae_audio: List[torch.Tensor] = []
    frame_audio: List[torch.Tensor] = []
    token_audio: List[torch.Tensor] = []
    token_plain_audio: List[torch.Tensor] = []
    student_audio: List[torch.Tensor] = []
    references: List[torch.Tensor] = []
    texts: List[str] = []
    duration_rows: List[Dict[str, float]] = []
    latent_rows: List[Dict[str, float]] = []

    n = min(args.limit, len(dataset), len(records))
    with torch.no_grad():
        for i in range(n):
            item = dataset[i]
            reference = torch.from_numpy(
                sf.read(str(corpus / records[i]["wav_path"]), dtype="float32")[0]
            ).reshape(-1)
            n_frames = int(item["n_frames"])

            # 1. autoencoder round trip on the real waveform
            log_mel = mel.log_mel(reference.reshape(1, -1))
            latent = model.autoencoder.encode(log_mel)
            ae_wav = model.autoencoder.decode(latent).reshape(-1)

            # 2. the cached frame-level latent (normalised) -> decoder
            frame_latent = model.latent_norm.denormalize(item["latent"][None, :, :n_frames])
            frame_wav = model.autoencoder.decode(frame_latent).reshape(-1)

            # 3. per-token latents expanded exactly as inference does, with TEACHER durations
            tok, _ = model.decoder_latent_from_tokens(
                item["latent_token"][None],
                item["durations"][None],
                item["f0"][None],
                item["energy"][None],
            )
            token_wav = model.autoencoder.decode(tok).reshape(-1)

            # 3b. the same expansion *without* the prosody projection.  `latent_from_tokens` adds
            # `prosody_proj(f0, energy)` to the latent, and no training stage ever calls that function
            # (the autoencoder stage never builds a latent from tokens, `distill-text` only compares
            # token-level predictions, and `distill-decoder` consumes the *cached frame* latent) --
            # so the projection is an untrained random module applied at inference only.  This path
            # isolates its cost.
            tok_plain, _ = model.decoder_latent_from_tokens(
                item["latent_token"][None], item["durations"][None]
            )
            token_plain_wav = model.autoencoder.decode(tok_plain).reshape(-1)

            # 4. the actual student path
            student_wav = synth.synthesize(records[i]["text"], seed=0).reshape(-1)

            ae_audio.append(ae_wav)
            frame_audio.append(frame_wav)
            token_audio.append(token_wav)
            token_plain_audio.append(token_plain_wav)
            student_audio.append(student_wav)
            references.append(reference)
            texts.append(records[i]["text"])

            predicted_frames = int(
                normalized_to_durations(model.text_side(item["ids"][None])["log_duration"]).sum()
            )
            duration_rows.append(
                {
                    "utt": records[i]["utt_id"],
                    "reference_frames": n_frames,
                    "teacher_duration_frames": int(item["durations"].sum()),
                    "predicted_frames": predicted_frames,
                }
            )
            latent_rows.append(
                {
                    "utt": records[i]["utt_id"],
                    "teacher_latent_std": float(item["latent"][:, :n_frames].std()),
                    "teacher_token_std": float(item["latent_token"].std()),
                    "decoded_frames": int(tok.shape[-1]),
                }
            )
            if i < 2:
                for name, audio in (
                    ("ae_roundtrip", ae_wav), ("frame_latent", frame_wav),
                    ("token_expanded", token_wav), ("student", student_wav),
                    ("reference", reference),
                ):
                    write_wav(out / f"{name}_{i}.wav", audio.reshape(1, -1), cfg.audio.sample_rate)

    _banner("paths through the same decoder, scored by the same recogniser")
    results = [
        measure("reference (Kokoro, the ceiling)", references, texts, cfg, args.whisper),
        measure("1. ae_roundtrip", ae_audio, texts, cfg, args.whisper),
        measure("2. teacher_frame_latent", frame_audio, texts, cfg, args.whisper),
        measure("3. teacher_token_expanded", token_audio, texts, cfg, args.whisper),
        measure("3b. token_expanded_no_prosody", token_plain_audio, texts, cfg, args.whisper),
        measure("4. student (text in)", student_audio, texts, cfg, args.whisper),
    ]
    for row in results:
        wer = f"{row['wer']:.3f}" if row["wer_available"] else "n/a"
        mos = f"{row['dnsmos']:.2f}" if row["dnsmos_available"] else "n/a"
        print(f"  {row['path']:32s} WER {wer:>6s} | DNSMOS {mos:>5s}")

    cosines = {
        "ae_roundtrip": [_mel_cosine(mel, a, r) for a, r in zip(ae_audio, references)],
        "teacher_frame_latent": [_mel_cosine(mel, a, r) for a, r in zip(frame_audio, references)],
        "teacher_token_expanded": [_mel_cosine(mel, a, r) for a, r in zip(token_audio, references)],
        "student": [_mel_cosine(mel, a, r) for a, r in zip(student_audio, references)],
    }
    print("\n  log-mel cosine against the real reference:")
    for name, values in cosines.items():
        finite = [v for v in values if v == v]
        print(f"    {name:26s} {statistics.mean(finite):.4f}")

    by_path = {row["path"]: row for row in results}
    frame_wer = by_path["2. teacher_frame_latent"]["wer"]
    token_wer = by_path["3. teacher_token_expanded"]["wer"]
    ae_wer = by_path["1. ae_roundtrip"]["wer"]
    student_wer = by_path["4. student (text in)"]["wer"]
    reference_wer = by_path["reference (Kokoro, the ceiling)"]["wer"]
    predicted_total = statistics.mean(r["predicted_frames"] for r in duration_rows)
    reference_total = statistics.mean(r["reference_frames"] for r in duration_rows)

    # waveform-level fidelity of the round-trip: a mel distance alone can look acceptable while the
    # output is *uncorrelated* with the input, which is the state that produces WER 1.0
    fidelity: Dict[str, Dict[str, float]] = {}
    for name, audios in (
        ("ae_roundtrip", ae_audio), ("teacher_frame_latent", frame_audio),
        ("teacher_token_expanded", token_audio), ("student", student_audio),
    ):
        snrs, wave_cosines = [], []
        for produced, ref in zip(audios, references):
            n = min(produced.numel(), ref.numel())
            if n < 2000:
                continue
            a, b = produced[:n], ref[:n]
            snrs.append(float(10 * torch.log10(b.pow(2).mean() / (a - b).pow(2).mean().clamp_min(1e-12))))
            wave_cosines.append(float(torch.nn.functional.cosine_similarity(a, b, dim=0)))
        fidelity[name] = {
            "snr_db": statistics.mean(snrs) if snrs else float("nan"),
            "waveform_cosine": statistics.mean(wave_cosines) if wave_cosines else float("nan"),
            "log_mel_cosine": statistics.mean([v for v in cosines[name] if v == v]),
        }
    print("\n  fidelity against the real reference (waveform level):")
    for name, row in fidelity.items():
        print(f"    {name:26s} SNR {row['snr_db']:6.2f} dB | waveform cosine "
              f"{row['waveform_cosine']:+.3f} | mel cosine {row['log_mel_cosine']:.4f}")

    # `checks` are the *diagnostic's* validity -- did the measurement work at all?  The state of the
    # system is `findings`, so this script's exit code means "the diagnosis ran correctly" rather
    # than "the model is good" (it is not).
    checks = {
        "recogniser_works_on_the_reference": bool(reference_wer is not None and reference_wer < 0.2),
        "every_path_produced_audio": all(
            torch.isfinite(a).all().item() for a in ae_audio + frame_audio + token_audio + student_audio
        ),
        "the_reference_scores_best": bool(
            reference_wer is not None
            and all(
                reference_wer <= by_path[p]["wer"]
                for p in by_path
                if by_path[p]["wer"] is not None
            )
        ),
    }
    findings = {
        "ae_roundtrip_is_intelligible": bool(ae_wer is not None and ae_wer < 0.5),
        "ae_roundtrip_is_correlated_with_its_input": bool(
            fidelity["ae_roundtrip"]["waveform_cosine"] > 0.5
        ),
        "token_expansion_is_not_the_bottleneck": bool(
            token_wer is not None and frame_wer is not None and token_wer <= frame_wer + 0.15
        ),
        "student_is_competitive_with_the_token_path": bool(
            student_wer is not None and token_wer is not None and student_wer <= token_wer + 0.15
        ),
        "predicted_duration_is_in_the_right_ballpark": (
            0.5 * reference_total < predicted_total < 2.0 * reference_total
        ),
    }
    # walk the chain: the first stage that loses intelligibility is the bottleneck.  Saying "none
    # detected" while a teacher-input path sits at WER 0.87 is how a bottleneck stays hidden -- the
    # round-19 failure looked identical until the autoencoder was fixed and this seam appeared.
    if not findings["ae_roundtrip_is_intelligible"]:
        bottleneck = "autoencoder"
    elif not findings["token_expansion_is_not_the_bottleneck"]:
        bottleneck = "token expansion (per-token latents -> frame latents)"
    elif not findings["student_is_competitive_with_the_token_path"]:
        bottleneck = "text side (prediction error on top of an intelligible path)"
    else:
        bottleneck = "none detected"
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": payload.get("step"),
        "utterances": n,
        "recogniser": args.whisper,
        "paths": results,
        "log_mel_cosine_vs_reference": {k: statistics.mean([v for v in vs if v == v]) for k, vs in cosines.items()},
        "fidelity": fidelity,
        "durations": duration_rows,
        "durations_summary": {
            "reference_frames_mean": reference_total,
            "predicted_frames_mean": predicted_total,
            "ratio": predicted_total / max(reference_total, 1e-9),
        },
        "latents": latent_rows,
        "diagnosis": {
            "bottleneck": bottleneck,
            "ceiling": "the autoencoder round-trip is the best this system can do",
            "ordering": "reference >= ae_roundtrip >= teacher_frame_latent >= teacher_token_expanded >= student",
            "note": (
                "Whichever step first loses intelligibility is the bottleneck: if ae_roundtrip is "
                "already bad, no text-side work helps; if only the student is bad, it is prediction "
                "error and capacity/data, not the seam."
            ),
        },
        "checks": checks,
        "findings": findings,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _banner("RESULT")
    print(f"  durations: reference {reference_total:.0f} frames vs predicted "
          f"{predicted_total:.0f} ({predicted_total / max(reference_total, 1e-9):.2f}x)")
    print(f"  diagnosed bottleneck: {bottleneck}")
    print("\n  diagnostic validity:")
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    print("\n  system health (a FAIL here is the finding, not an error):")
    for name, ok in findings.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nreport -> {out/'report.json'}")
    print("REAL DIAGNOSE " + ("PASSED" if all(checks.values()) else "FAILED"))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
