"""Train the student on **real** teacher speech: autoencoder, then text side, then measure.

    python scripts/real_train_demo.py --steps-ae 300 --steps-text 400

Round 17 unblocked real data (Kokoro-82M, Apache-2.0, faster than real time on CPU).  This script
uses it for what the objective actually needs: a trained autoencoder and a text side that predict
real prosody, with the numbers measured rather than asserted.

Stages, in the documented order:

1. **autoencoder** on real 24 kHz waveforms (the only stage that trains on audio);
2. **latent normalizer + cache** from the *curated* corpus, using that trained autoencoder;
3. **distill-text** on the cached real teacher signals;
4. evaluation: reconstruction, teacher-signal fit, and text -> audio rendered by the trained
   decoder, compared against the real reference audio for the same text.

Honest limits, stated up front: this is a small corpus and a short CPU budget, so nothing here is a
converged model.  The duration targets come from the unaligned fallback (the sherpa runtime does not
expose Kokoro's token timings), and there is still no perceptual metric -- UTMOS is not installed.
What this establishes is the first real-audio training evidence and the numbers that go with it.
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

from parakeet.config import load_config  # noqa: E402
from parakeet.data.dataset import LatentShardBatchSource, LatentShardDataset, WaveformCorpusSource  # noqa: E402
from parakeet.data.features import cache_teacher_corpus, fit_latent_normalizer  # noqa: E402
from parakeet.inference import Synthesizer, write_wav  # noqa: E402
from parakeet.audio.mel import MelSpectrogram  # noqa: E402
from parakeet.eval import ae_reconstruction_l1, teacher_signal_loss  # noqa: E402
from parakeet.models import build_model, count_parameters  # noqa: E402
from parakeet.train.stages import run_stage  # noqa: E402


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train and evaluate on real teacher speech")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--corpus", default="data/real_corpus/corpus")
    ap.add_argument("--out", default="runs/real_train")
    ap.add_argument("--steps-ae", type=int, default=300)
    ap.add_argument("--steps-text", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--reuse-ae", action="store_true",
                    help="load the autoencoder checkpoint from --out instead of retraining it "
                         "(the AE takes ~25 min on CPU; the text side takes ~20 s)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus)
    manifest_name = "curated/kept.jsonl" if (corpus / "curated" / "kept.jsonl").exists() else "manifest.jsonl"
    manifest = corpus / manifest_name
    if not manifest.exists():
        print(f"no corpus at {manifest}; run scripts/real_corpus_demo.py first")
        return 2
    records = [
        json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    voices = sorted({str(r.get("voice") or "") for r in records})
    cfg = load_config(args.config)
    cfg.n_voices = max(1, len(voices))
    cfg.train.batch_size = args.batch_size
    cfg.train.log_every = max(1, args.steps_ae // 6)
    cfg.train.save_every = 0
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t_start = time.perf_counter()
    strict: Dict[str, bool] = {}

    _banner(f"real corpus: {len(records)} utterances from {manifest_name} | voices {voices}")
    source = WaveformCorpusSource(
        manifest, batch_size=args.batch_size, corpus_dir=corpus, seed=cfg.train.seed
    )
    seconds = sum(r["duration_s"] for r in records)
    print(f"  {seconds:.1f}s of real speech | {count_parameters(build_model(cfg))/1e6:.3f} M model")
    strict["corpus_is_real_speech"] = len(records) > 0 and seconds > 10.0

    model = build_model(cfg)
    import soundfile as sf

    waves = [
        torch.from_numpy(sf.read(str(corpus / r["wav_path"]), dtype="float32")[0]) for r in records
    ]

    # ---------------- 1. autoencoder on real speech ----------------
    ae_checkpoint = out / "autoencoder_last.pt"
    reusing = bool(args.reuse_ae and ae_checkpoint.exists())
    if reusing:
        _banner(f"1/4 autoencoder: reusing {ae_checkpoint.name} (no retraining)")
        payload = torch.load(ae_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("ema", {}).get("shadow", payload["model"]), strict=False)
        # the comparison stays honest: the baseline is a freshly initialised model, so "improved"
        # still means "better than untrained" even though no training happened this run
        ae_before = ae_reconstruction_l1(build_model(cfg), waves, cfg)
        ae_after = ae_reconstruction_l1(model, waves, cfg)
        ae_seconds = 0.0
    else:
        _banner(f"1/4 autoencoder on real speech | {args.steps_ae} steps")
        ae_before = ae_reconstruction_l1(model, waves, cfg)
        t0 = time.perf_counter()
        run_stage("autoencoder", cfg, model=model, batches=source, max_steps=args.steps_ae,
                  out_dir=str(out))
        ae_seconds = time.perf_counter() - t0
        ae_after = ae_reconstruction_l1(model, waves, cfg)
    print(f"  reconstruction log-mel L1 {ae_before:.4f} -> {ae_after:.4f} "
          f"({100 * (ae_before - ae_after) / ae_before:+.1f}%) "
          f"{'(reused checkpoint, baseline is a fresh model)' if reusing else f'in {ae_seconds:.0f}s ({ae_seconds / max(1, args.steps_ae):.2f}s/step)'}")
    strict["autoencoder_improved_on_real_speech"] = ae_after < ae_before
    strict["autoencoder_materially_improved"] = ae_after < 0.95 * ae_before

    # ---------------- 2. cache from the trained autoencoder ----------------
    _banner("2/4 latent cache from the trained autoencoder")
    fit_latent_normalizer(model.latent_norm, model.autoencoder, source, cfg, max_batches=4)
    # via the library helper, not build_latent_cache directly: it resolves record paths against the
    # corpus root (the curated manifest sits in a subdirectory), picks the curated manifest and
    # carries the mixture.  Calling the lower-level function here reproduced the round-10 path bug.
    cache = cache_teacher_corpus(
        corpus, out / "latent_cache", cfg, model.autoencoder,
        teacher_latent_norm=model.latent_norm,
    )
    dataset = LatentShardDataset(cache)
    meta = json.loads((cache / "cache_meta.json").read_text(encoding="utf-8"))
    print(f"  {len(dataset)} items | voices {meta['voice_names']} | "
          f"latent {meta['latent_dim']}d")
    strict["cache_built_from_trained_autoencoder"] = len(dataset) > 0

    # ---------------- 3. text side on real cached signals ----------------
    _banner(f"3/4 text side on real teacher signals | {args.steps_text} steps")
    probes = [dataset[i] for i in range(len(dataset))]
    text_before = teacher_signal_loss(model, probes, cfg)
    loader = LatentShardBatchSource(dataset, batch_size=args.batch_size, seed=cfg.train.seed)
    cfg.train.log_every = max(1, args.steps_text // 6)
    t0 = time.perf_counter()
    run_stage("distill-text", cfg, model=model, batches=loader, max_steps=args.steps_text,
              out_dir=str(out))
    text_seconds = time.perf_counter() - t0
    text_after = teacher_signal_loss(model, probes, cfg)
    print(f"  teacher-signal loss {text_before:.4f} -> {text_after:.4f} "
          f"({100 * (text_before - text_after) / text_before:+.1f}%) in {text_seconds:.0f}s")
    strict["text_side_learned_real_signals"] = text_after < text_before

    # ---------------- 4. text -> audio against the real reference ----------------
    _banner("4/4 text -> audio, rendered by the trained decoder")
    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=True)
    mel = MelSpectrogram(cfg.audio)
    cosine: List[float] = []
    mel_l1: List[float] = []
    for record, reference in zip(records, waves):
        generated = synth.synthesize(record["text"], seed=0)
        n = min(generated.shape[-1], reference.numel())
        if n < 2000:
            continue
        a = mel.log_mel(generated[..., :n])
        b = mel.log_mel(reference[:n].reshape(1, -1))
        cosine.append(float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)))
        mel_l1.append(float(torch.nn.functional.l1_loss(a, b)))
    write_wav(out / "real_synthesis.wav", generated, cfg.audio.sample_rate)
    mean_cosine = statistics.mean(cosine) if cosine else float("nan")
    mean_mel = statistics.mean(mel_l1) if mel_l1 else float("nan")
    print(f"  {len(cosine)} utterances | generated vs real reference: log-mel cosine "
          f"{mean_cosine:.4f} | L1 {mean_mel:.4f}")
    strict["synthesis_produces_audio"] = bool(cosine) and torch.isfinite(generated).all().item()

    report = {
        "config": args.config,
        "corpus": {"manifest": manifest_name, "utterances": len(records),
                   "audio_seconds": seconds, "voices": voices},
        "steps": {"autoencoder": args.steps_ae, "distill_text": args.steps_text},
        "params": count_parameters(build_model(cfg)),
        "autoencoder": {"recon_before": ae_before, "recon_after": ae_after,
                        "improvement_pct": 100 * (ae_before - ae_after) / ae_before,
                        "seconds": ae_seconds},
        "text_side": {"loss_before": text_before, "loss_after": text_after,
                      "improvement_pct": 100 * (text_before - text_after) / text_before,
                      "seconds": text_seconds},
        "text_to_audio": {"n_utterances": len(cosine), "log_mel_cosine": mean_cosine,
                          "log_mel_l1": mean_mel},
        "caveats": [
            "small corpus and a short CPU budget: this is not a converged model",
            "duration targets are the unaligned fallback (sherpa does not expose Kokoro timings)",
            "no perceptual metric: UTMOS is not installed, so naturalness is unmeasured",
            "log-mel cosine vs the reference is a crude proxy, not intelligibility",
        ],
        "checks": strict,
        "seconds_total": time.perf_counter() - t_start,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _banner("RESULT")
    for name, ok in strict.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nreport -> {out/'report.json'}\naudio  -> {out/'real_synthesis.wav'}")
    print("REAL TRAINING " + ("PASSED" if all(strict.values()) else "FAILED"))
    return 0 if all(strict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
