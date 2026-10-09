"""Learn-demo: does the training code actually *learn*?  CPU only, no data, no GPU, no network.

    python scripts/learn_demo.py                 # ~8 minutes on 8 CPU cores
    python scripts/learn_demo.py --quick         # ~1 minute version of the same experiment

The smoke test proves the pipeline *runs*.  This proves it **learns and composes**, which is the
strongest claim available without a corpus and a GPU:

1. train the autoencoder on structured synthetic utterances; measure mel-reconstruction error
   before vs after (an independent metric, not the training loss);
2. fit the latent normaliser;
3. build exact teacher-signal targets (durations, normalised F0, energy, per-token latent) from the
   *trained* autoencoder;
4. train the Tiny text side to regress them; measure the distillation loss before vs after;
5. train the decoder on the teacher-shaped latents -- the very tensors the text side predicts;
6. synthesise from text end to end and compare against a baseline with the identical rendering
   stack but an untrained text side.

Pass/fail is printed and the exit code reflects it.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.data.features import (  # noqa: E402
    TokenTargetBatchSource,
    fit_latent_normalizer,
    token_targets_from_corpus,
)
from parakeet.data.synthetic import SyntheticSpeechBatchSource, make_corpus  # noqa: E402
from parakeet.data.text import TextTokenizer  # noqa: E402
from parakeet.eval import (  # noqa: E402
    ae_reconstruction_l1,
    duration_error_frames,
    end_to_end_mel_l1,
    latent_normalizer_summary,
    teacher_signal_loss,
)
from parakeet.inference import phase_coherence, phase_lock, write_wav  # noqa: E402
from parakeet.models import build_model, count_parameters  # noqa: E402
from parakeet.train.stages import run_stage  # noqa: E402

CRITERIA_DOC = """
  autoencoder_learned   ae_recon_after  < 0.90 x ae_recon_before   representation fits
  text_side_learned     text_loss_after < 0.50 x text_loss_before  cached teacher signals fit
  end_to_end_improved   e2e_mel_after   < 0.90 x e2e_mel_before    text->audio beats untrained
  durations_fit         duration MAE    <= 2.0 frames              predicted timing is usable
""".strip()


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Prove the Parakeet stages learn (CPU only)")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--utterances", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps-ae", type=int, default=250)
    ap.add_argument("--steps-text", type=int, default=400)
    ap.add_argument("--steps-decoder", type=int, default=120)
    ap.add_argument("--out", default="runs/learn_demo")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps_ae, args.steps_text, args.steps_decoder = 30, 120, 30

    cfg = load_config(args.config)
    cfg.train.log_every = max(1, args.steps_ae)
    cfg.train.save_every = 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.train.seed)

    corpus = make_corpus(args.utterances, cfg.audio, seed=cfg.train.seed)
    tokenizer = TextTokenizer(mode=cfg.text.mode)
    seconds = sum(u.wav.numel() for u in corpus) / cfg.audio.sample_rate
    _banner(f"corpus: {len(corpus)} synthetic utterances | {seconds:.1f}s audio | "
            f"layouts {sorted({u.layout for u in corpus})} | tokens {corpus[0].n_tokens}")

    model = build_model(cfg)
    print(f"model: {cfg.name} | {count_parameters(model)/1e6:.3f}M params")
    t0 = time.perf_counter()

    # ---------------- 1. autoencoder ----------------
    wave_source = SyntheticSpeechBatchSource(corpus, batch_size=args.batch_size, seed=cfg.train.seed)
    ae_before = ae_reconstruction_l1(model, [u.wav for u in corpus], cfg)
    run_stage("autoencoder", cfg, model=model, batches=wave_source,
              max_steps=args.steps_ae, out_dir=str(out))
    fit_latent_normalizer(model.latent_norm, model.autoencoder, wave_source, cfg, max_batches=8)
    ae_after = ae_reconstruction_l1(model, [u.wav for u in corpus], cfg)
    stats = latent_normalizer_summary(model)
    print(f"[autoencoder]   recon mel L1 {ae_before:.4f} -> {ae_after:.4f} "
          f"({100 * (1 - ae_after / ae_before):.1f}% better) | {time.perf_counter() - t0:.1f}s")
    print(f"                latent normaliser: |mu|={stats['abs_mean']:.3f} "
          f"sigma={stats['mean_sigma']:.3f} updates={stats['updates']}")

    # snapshot: lets the end-to-end comparison isolate the *distilled* halves
    post_ae_state = copy.deepcopy(model.state_dict())

    # ---------------- 2. text side ----------------
    targets = token_targets_from_corpus(model, corpus, cfg, tokenizer)
    text_before = teacher_signal_loss(model, targets, cfg)
    text_logs = run_stage(
        "distill-text", cfg, model=model,
        batches=TokenTargetBatchSource(targets, "distill-text", batch_size=args.batch_size, seed=cfg.train.seed),
        max_steps=args.steps_text, out_dir=str(out),
    )
    text_after = teacher_signal_loss(model, targets, cfg)
    print(f"[distill-text]  teacher-signal loss {text_before:.4f} -> {text_after:.4f} "
          f"({100 * (1 - text_after / text_before):.1f}% better)")

    # ---------------- 3. decoder on teacher-shaped latents ----------------
    decoder_logs = run_stage(
        "distill-decoder", cfg, model=model,
        batches=TokenTargetBatchSource(targets, "distill-decoder", batch_size=args.batch_size,
                                       seed=cfg.train.seed, model=model),
        max_steps=args.steps_decoder, out_dir=str(out),
    )
    print(f"[distill-decod.]final mel loss {float(decoder_logs.get('mel', float('nan'))):.4f} "
          f"spec {float(decoder_logs.get('spectral', float('nan'))):.4f} "
          f"adv {float(decoder_logs.get('adv', float('nan'))):.4f}")

    # ---------------- 4. end-to-end ----------------
    baseline = build_model(cfg)
    baseline.load_state_dict(post_ae_state)
    baseline.eval()
    e2e_before = end_to_end_mel_l1(baseline, targets, cfg)
    e2e_after = end_to_end_mel_l1(model, targets, cfg)
    dur_mae = duration_error_frames(model, targets)
    print(f"[end-to-end]    mel L1 vs target {e2e_before:.4f} -> {e2e_after:.4f} "
          f"({100 * (1 - e2e_after / e2e_before):.1f}% better)")
    print(f"[durations]     mean |pred - true| = {dur_mae:.2f} frames "
          f"({dur_mae * cfg.audio.hop_length / cfg.audio.sample_rate * 1000:.0f} ms)")

    # ---------------- 5. audio + phase lock ----------------
    ids = targets[0]["ids"][None]
    mask = torch.ones_like(ids, dtype=torch.bool)
    wav_gen = model.synthesize(ids, mask)
    wav_locked = phase_lock(wav_gen, sample_rate=cfg.audio.sample_rate,
                            n_fft=cfg.audio.n_fft, hop_length=cfg.audio.hop_length)
    coh_raw = float(phase_coherence(wav_gen, sample_rate=cfg.audio.sample_rate).item())
    coh_locked = float(phase_coherence(wav_locked, sample_rate=cfg.audio.sample_rate).item())
    write_wav(out / "target.wav", targets[0]["wav"], cfg.audio.sample_rate)
    write_wav(out / "generated.wav", wav_gen, cfg.audio.sample_rate)
    write_wav(out / "generated_phase_locked.wav", wav_locked, cfg.audio.sample_rate)
    write_wav(out / "baseline_untrained_text.wav", baseline.synthesize(ids, mask), cfg.audio.sample_rate)
    print(f"[phase-lock]    coherence 2-8k {coh_raw:.4f} -> {coh_locked:.4f}")

    # ---------------- 6. report ----------------
    checks = {
        "autoencoder_learned": ae_after < 0.90 * ae_before,
        "text_side_learned": text_after < 0.50 * text_before,
        "end_to_end_improved": e2e_after < 0.90 * e2e_before,
        "durations_fit": dur_mae <= 2.0,
    }
    results = {
        "config": args.config,
        "params": count_parameters(model),
        "corpus": {"utterances": len(corpus), "audio_seconds": seconds,
                   "tokens_per_utterance": corpus[0].n_tokens, "layouts": sorted({u.layout for u in corpus})},
        "steps": {"ae": args.steps_ae, "text": args.steps_text, "decoder": args.steps_decoder},
        "ae_recon_before": ae_before,
        "ae_recon_after": ae_after,
        "text_loss_before": text_before,
        "text_loss_after": text_after,
        "e2e_mel_before": e2e_before,
        "e2e_mel_after": e2e_after,
        "duration_mae_frames": dur_mae,
        "phase_coherence_before": coh_raw,
        "phase_coherence_after": coh_locked,
        "latent_normalizer": stats,
        "stage_logs": {
            "distill_text": {k: float(v) for k, v in text_logs.items() if isinstance(v, (int, float))},
            "distill_decoder": {k: float(v) for k, v in decoder_logs.items() if isinstance(v, (int, float))},
        },
        "seconds_total": time.perf_counter() - t0,
        "checks": checks,
    }
    (out / "report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    _banner("RESULT")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\ncriteria:\n" + CRITERIA_DOC)
    print(f"\nreport -> {out / 'report.json'}\naudio  -> {out}/*.wav")
    print("LEARN DEMO " + ("PASSED" if all(checks.values()) else "FAILED"))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
