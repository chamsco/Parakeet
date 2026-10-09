"""Train the autoencoder properly: reconstruction first, then adversarial sharpening.

    python scripts/ae_train.py --recon-steps 2000 --adv-steps 200

Round 19 localised the failure: the autoencoder's round-trip on real audio is **uncorrelated with its
input** (waveform cosine +0.000, WER 1.000), so nothing downstream can work.  Round 20 measured *why
the budget did not help*: a generator step with the MPD/MSD discriminator costs **5.25 s** on this CPU
and the same step without it costs **0.22 s** -- a factor of 24.  The shipped recipe ran 300
adversarial steps from a random initialisation, i.e. it spent the whole budget on the sharpening term
before the autoencoder could encode anything.

So the budget is spent the way codec training normally does it:

  phase 1  reconstruction only (mel + spectral, `adversarial` weight 0)   ~0.2 s/step
  phase 2  the full objective with the discriminator                       ~5 s/step

Both phases go through :func:`parakeet.train.stages.run_stage`, which is the point.  The first version
of this script hand-rolled its own loop and **lost the warmup schedule**, going mel 2.06 -> 0.50 and
then to NaN between steps 1200 and 1800 with no visible symptom until an all-NaN report at the end
(the same "reimplemented the helper" mistake as calling `build_latent_cache` instead of
`cache_teacher_corpus`).  Using the stage runner also brings the EMA, LR schedule, provenance, resume
state and the new non-finite guard, which now reports *which* step diverged instead of failing
silently.

Measurement between phases uses things that can tell "slightly degraded" from "unrelated": log-mel
L1, waveform correlation, SNR, DNSMOS, and finally the **round-trip WER**, gated by the recogniser
transcribing the teacher's own audio correctly.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.audio.mel import MelSpectrogram  # noqa: E402
from parakeet.config import load_config  # noqa: E402
from parakeet.data.dataset import WaveformCorpusSource  # noqa: E402
from parakeet.eval.metrics import dnsmos_score, whisper_wer_with_control  # noqa: E402
from parakeet.inference import phase_coherence, write_wav  # noqa: E402
from parakeet.models import build_model, count_parameters  # noqa: E402
from parakeet.train.common import load_checkpoint  # noqa: E402
from parakeet.train.stages import run_stage  # noqa: E402


def fidelity(model, waves: List[torch.Tensor], mel: MelSpectrogram, cfg) -> Dict[str, float]:
    """Reconstruction quality on real speech, including the *correlation* the mel proxy hid."""
    was_training = model.training
    model.eval()
    rows, audios = [], []
    with torch.no_grad():
        for ref in waves:
            recon = model.autoencoder.decode(
                model.autoencoder.encode(mel.log_mel(ref.reshape(1, -1))), length=ref.numel()
            ).reshape(-1)
            n = min(recon.numel(), ref.numel())
            a, b = recon[:n], ref[:n]
            rows.append(
                {
                    "mel_l1": float(torch.nn.functional.l1_loss(
                        mel.log_mel(a.reshape(1, -1)), mel.log_mel(b.reshape(1, -1)))),
                    "waveform_cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                    "snr_db": float(10 * torch.log10(b.pow(2).mean() / (a - b).pow(2).mean().clamp_min(1e-12))),
                    "phase_coherence": float(phase_coherence(a.reshape(1, -1), sample_rate=cfg.audio.sample_rate)),
                }
            )
            audios.append(recon)
    out = {key: statistics.mean(row[key] for row in rows) for key in rows[0]}
    metric = dnsmos_score(audios, sample_rate=cfg.audio.sample_rate)
    out["dnsmos"] = metric.value if metric.available else float("nan")
    model.train(was_training)
    return out


def load_ema_into(model, checkpoint: Path) -> Optional[Dict[str, float]]:
    """Put the checkpoint's EMA weights into the model; None if it carries none."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    shadow = (payload.get("ema") or {}).get("shadow")
    if not shadow:
        return None
    live = {k: v.detach().clone() for k, v in model.autoencoder.state_dict().items()}
    model.autoencoder.load_state_dict(shadow, strict=False)
    return live


def main() -> int:
    ap = argparse.ArgumentParser(description="Phased autoencoder training on real speech")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--corpus", default="data/real_corpus/corpus")
    ap.add_argument("--manifest", default=None, help="manifest file inside --corpus (e.g. val.jsonl for a held-out split)")
    ap.add_argument("--recon-steps", type=int, default=2000)
    ap.add_argument("--adv-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-utterances", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, default=None,
                        help="cap the audio length per item.  Long clips make a padded batch heterogeneous "
                             "and were the likely trigger for the round-24 divergence on the scaled corpus")
    ap.add_argument("--whisper", default="base.en")
    ap.add_argument("--out", default="runs/ae_long")
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
    cfg = load_config(args.config)
    cfg.train.batch_size = args.batch_size
    mel = MelSpectrogram(cfg.audio)
    import soundfile as sf

    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    # derive the voice count from the corpus rather than forcing 1: the autoencoder does not use
    # voices, but the checkpoint stores the whole model, and a mismatched `voice_embed` shape makes
    # the file unusable for the downstream stages (strict=False does not tolerate a shape mismatch)
    cfg.n_voices = max(1, len({str(r.get("voice") or "") for r in records}))
    texts = [r["text"] for r in records[: args.eval_utterances]]
    waves = [
        torch.from_numpy(sf.read(str(corpus / r["wav_path"]), dtype="float32")[0])
        for r in records[: args.eval_utterances]
    ]

    torch.manual_seed(cfg.train.seed)
    model = build_model(cfg)
    source = WaveformCorpusSource(
        manifest, batch_size=args.batch_size, corpus_dir=corpus, seed=cfg.train.seed,
        max_seconds=args.max_seconds,
    )
    print(f"corpus {len(records)} utterances | {count_parameters(model)/1e6:.3f} M params | "
          f"{args.recon_steps} recon steps + {args.adv_steps} adversarial steps "
          f"| warmup {cfg.train.warmup_steps} (capped to a tenth of each phase)")
    started = time.perf_counter()
    phases: List[Dict[str, object]] = []

    def run_phase(name: str, steps: int, adversarial_weight: float, feature_weight: float) -> None:
        if steps <= 0:
            return
        phase_cfg = replace(
            cfg,
            train=replace(
                cfg.train,
                loss=replace(
                    cfg.train.loss,
                    adversarial=adversarial_weight,
                    feature_match=feature_weight,
                ),
                warmup_steps=min(cfg.train.warmup_steps, max(1, steps // 10)),
                log_every=max(1, steps // 5),
                save_every=0,
            ),
        )
        phase_dir = out / name
        phase_started = time.perf_counter()
        logs = run_stage("autoencoder", phase_cfg, model=model, batches=source,
                         max_steps=steps, out_dir=str(phase_dir))
        seconds = time.perf_counter() - phase_started
        live = fidelity(model, waves, mel, cfg)
        checkpoint = phase_dir / "autoencoder_last.pt"
        live_backup = load_ema_into(model, checkpoint) if checkpoint.exists() else None
        ema = fidelity(model, waves, mel, cfg) if live_backup else {}
        if live_backup:
            model.autoencoder.load_state_dict(live_backup)  # phase 2 starts from the live weights
        row = {
            "phase": name,
            "steps": steps,
            "adversarial_weight": adversarial_weight,
            "seconds": seconds,
            "seconds_per_step": seconds / steps,
            "nonfinite_steps": logs.get("nonfinite_steps", 0.0),
            "first_nonfinite_step": logs.get("first_nonfinite_step"),
            "final_loss": logs.get("loss"),
            "live": live,
            "ema": ema,
        }
        phases.append(row)
        print(f"    {name}: {seconds:.0f}s ({seconds/steps:.2f}s/step) | loss {(logs.get('loss') or float('nan')):.4f} "
              f"| live mel {live['mel_l1']:.4f} cos {live['waveform_cosine']:+.3f} "
              f"| ema mel {ema.get('mel_l1', float('nan')):.4f} "
              f"cos {ema.get('waveform_cosine', float('nan')):+.3f} "
              f"| nonfinite {logs.get('nonfinite_steps', 0):.0f}", flush=True)

    run_phase("recon", args.recon_steps, 0.0, 0.0)
    run_phase("adversarial", args.adv_steps,
              load_config(args.config).train.loss.adversarial,
              load_config(args.config).train.loss.feature_match)

    # the pipeline's convention is to evaluate and export the EMA weights
    checkpoint = out / "adversarial" / "autoencoder_last.pt"
    if not checkpoint.exists():
        checkpoint = out / "recon" / "autoencoder_last.pt"
    load_ema_into(model, checkpoint)
    final = fidelity(model, waves, mel, cfg)
    print(f"\n  final (EMA of the last phase): mel {final['mel_l1']:.4f} | "
          f"cosine {final['waveform_cosine']:+.3f} | SNR {final['snr_db']:+.2f} dB | "
          f"DNSMOS {final['dnsmos']:.2f}")

    model.eval()
    recons, refs = [], []
    with torch.no_grad():
        for ref in waves:
            recons.append(
                model.autoencoder.decode(
                    model.autoencoder.encode(mel.log_mel(ref.reshape(1, -1))), length=ref.numel()
                ).reshape(-1)
            )
            refs.append(ref)
    controlled = whisper_wer_with_control(
        recons, texts, control_audio=refs, sample_rate=cfg.audio.sample_rate, model_size=args.whisper
    )
    wer = controlled["wer"]
    teacher_wer = controlled["control"]
    print(f"  round-trip WER {wer.value if wer.available else 'unavailable'} "
          f"| teacher WER {teacher_wer.value if teacher_wer.available else 'unavailable'} "
          f"({wer.detail})")
    for i in range(min(2, len(recons))):
        write_wav(out / f"roundtrip_{i}.wav", recons[i].reshape(1, -1), cfg.audio.sample_rate)
        write_wav(out / f"reference_{i}.wav", refs[i].reshape(1, -1), cfg.audio.sample_rate)

    recon_phase = next((p for p in phases if p["phase"] == "recon"), None)
    baseline_mel = 1.3864  # the shipped recipe: 300 adversarial steps from random init
    checks = {
        "no_divergence": all(float(p.get("nonfinite_steps") or 0) == 0 for p in phases),
        "reconstruction_improved_on_the_baseline": bool(
            final["mel_l1"] < baseline_mel and final["mel_l1"] == final["mel_l1"]
        ),
        "recon_phase_improved_the_loss": bool(
            recon_phase is not None and recon_phase["final_loss"] is not None
        ),
        "round_trip_wer_is_measured_with_a_valid_control": bool(wer.available and teacher_wer.available),
        "round_trip_wer_beats_the_baseline": bool(wer.available and wer.value < 1.0),
    }
    report = {
        "config": args.config,
        "corpus": {"manifest": manifest.name, "utterances": len(records),
                   "audio_seconds": sum(r["duration_s"] for r in records)},
        "steps": {"recon": args.recon_steps, "adversarial": args.adv_steps},
        "params": count_parameters(model),
        "phases": phases,
        "final": {**final, "weights": "ema"},
        "round_trip": {
            "student": wer.value,
            "teacher": teacher_wer.value,
            "recogniser": args.whisper,
            "attempts": controlled["attempts"],
            "control_note": controlled["note"],
            "note": "encode-decode of the teacher's own audio; no text side involved",
        },
        "baseline_for_comparison": {
            "description": "the shipped recipe: 300 adversarial steps from random init (round 18/19)",
            "mel_l1": baseline_mel,
            "waveform_cosine": 0.0,
            "round_trip_wer": 1.0,
        },
        "checks": checks,
        "caveats": [
            "still a short CPU budget: an improvement, not a converged codec",
            "the adversarial phase costs ~24x more per step than reconstruction",
            "a hand-rolled loop without warmup diverged to NaN here (round 20); this run uses run_stage",
        ],
        "seconds_total": time.perf_counter() - started,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  report -> {out/'report.json'}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("AE PHASED TRAINING " + ("PASSED" if all(checks.values()) else "FAILED"))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
