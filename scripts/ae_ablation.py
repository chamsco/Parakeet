"""Does the autoencoder's adversarial term earn its 4.8 s/step at this budget?

    python scripts/ae_ablation.py --steps 150

The autoencoder is the measured bottleneck (round 19: its round-trip on real audio is uncorrelated
with its input, WER 1.000).  Before spending hours of CPU on a longer run it is worth knowing *what*
to spend it on, because a generator step here is not one loss but five:

    mel + 3.0 * spectral + 1.0 * adversarial + 2.0 * feature_match + 0.05 * phase_lock

plus a separate discriminator step.  Adversarial training is standard for neural codecs and is what
makes decoded audio sharp rather than smooth, but it is also the classic way to spend a small budget
fighting a discriminator instead of learning to reconstruct.  This script trains the same model, from
the same seed, on the same real speech, for the same number of steps, under three objectives and
reports both the *quality* and the *wall clock*:

  recon_only     mel + spectral            (no discriminator at all)
  adversarial    the shipped default (all five terms + discriminator step)
  recon_warm     two-phase: recon_only, then the last third with the adversarial terms
                 (the standard codec recipe: learn the signal, then sharpen it)

Quality is measured on real speech by things that can distinguish "slightly degraded" from
"unrelated audio" -- the round-19 lesson: log-mel L1 alone could not.

    mel_l1              log-mel L1 against the input (lower is better)
    waveform_cosine     correlation of the reconstruction with the input (higher; ~0 means unrelated)
    snr_db              10*log10(signal / error)
    phase_coherence     the project's own phase metric
    dnsmos              reference-free naturalness
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
from parakeet.data.dataset import WaveformCorpusSource  # noqa: E402
from parakeet.eval.metrics import dnsmos_score  # noqa: E402
from parakeet.inference import phase_coherence  # noqa: E402
from parakeet.models import build_model, count_parameters  # noqa: E402
from parakeet.train.losses import build_loss_bundle  # noqa: E402
from parakeet.train.stages import autoencoder_step, discriminator_step  # noqa: E402


def evaluate(model, waves: List[torch.Tensor], mel: MelSpectrogram, cfg, want_dnsmos: bool = True) -> Dict[str, float]:
    model.eval()
    rows = []
    audios = []
    with torch.no_grad():
        for ref in waves:
            recon = model.autoencoder.decode(
                model.autoencoder.encode(mel.log_mel(ref.reshape(1, -1))), length=ref.numel()
            ).reshape(-1)
            n = min(recon.numel(), ref.numel())
            a, b = recon[:n], ref[:n]
            rows.append(
                {
                    "mel_l1": float(torch.nn.functional.l1_loss(mel.log_mel(a.reshape(1, -1)), mel.log_mel(b.reshape(1, -1)))),
                    "waveform_cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                    "snr_db": float(10 * torch.log10(b.pow(2).mean() / (a - b).pow(2).mean().clamp_min(1e-12))),
                    "phase_coherence": float(phase_coherence(a.reshape(1, -1), sample_rate=cfg.audio.sample_rate)),
                }
            )
            audios.append(recon)
    out = {key: statistics.mean(row[key] for row in rows) for key in rows[0]}
    if want_dnsmos:
        metric = dnsmos_score(audios, sample_rate=cfg.audio.sample_rate)
        out["dnsmos"] = metric.value if metric.available else float("nan")
    model.train()
    return out


def train_variant(
    name: str,
    cfg,
    waves: List[torch.Tensor],
    source,
    steps: int,
    mel: MelSpectrogram,
    log_every: int,
    adversarial_from: int,
) -> Dict[str, object]:
    torch.manual_seed(cfg.train.seed)
    model = build_model(cfg)
    losses = build_loss_bundle(cfg)          # the same objective set run_stage uses
    losses["mel_module"] = mel               # autoencoder_step reads the spectrogram from here
    adversary = losses["adversarial"]
    optimiser = torch.optim.AdamW(model.autoencoder.parameters(), lr=cfg.train.lr)
    disc_optimiser = torch.optim.AdamW(adversary.parameters(), lr=cfg.train.lr)
    before = evaluate(model, waves, mel, cfg, want_dnsmos=False)

    def enabled(step: int) -> bool:
        return name == "adversarial" or (name == "recon_warm" and step >= adversarial_from)

    started = time.perf_counter()
    history: List[Dict[str, float]] = []
    for step in range(1, steps + 1):
        batch = source()
        use_adv = enabled(step)
        total, logs, recon = autoencoder_step(
            cfg, model, batch["wav"], losses, spectral_weight=cfg.train.loss.spectral,
            adversarially=use_adv,
        )
        optimiser.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.autoencoder.parameters(), 1.0)
        optimiser.step()
        if use_adv:
            for _ in range(1):
                d_loss = discriminator_step(losses, batch["wav"], recon.detach())
                disc_optimiser.zero_grad(set_to_none=True)
                d_loss.backward()
                disc_optimiser.step()
        if step % log_every == 0 or step == steps:
            history.append(
                {
                    "step": step,
                    **{k: float(v) for k, v in logs.items() if k in {"mel", "spectral", "adv"}},
                }
            )
            last = history[-1]
            print(
                f"    {name} step {step:4d} mel {last['mel']:.4f} "
                f"spectral {last['spectral']:.4f} "
                f"adv {last.get('adv', float('nan')):.3f}",
                flush=True,
            )
    seconds = time.perf_counter() - started
    after = evaluate(model, waves, mel, cfg)
    model.train()
    return {
        "variant": name,
        "steps": steps,
        "adversarial_from": adversarial_from if name == "recon_warm" else (0 if name == "adversarial" else None),
        "seconds": seconds,
        "seconds_per_step": seconds / steps,
        "before": before,
        "after": after,
        "history": history,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Ablate the autoencoder objective at a fixed budget")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--corpus", default="data/real_corpus/corpus")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-utterances", type=int, default=6)
    ap.add_argument("--variants", default="recon_only,adversarial,recon_warm")
    ap.add_argument("--out", default="runs/ae_ablation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    corpus = Path(args.corpus)
    manifest = corpus / "curated" / "kept.jsonl"
    if not manifest.exists():
        manifest = corpus / "manifest.jsonl"
    source = WaveformCorpusSource(manifest, batch_size=args.batch_size, corpus_dir=corpus, seed=cfg.train.seed)
    cfg.train.batch_size = args.batch_size
    cfg.n_voices = 1
    mel = MelSpectrogram(cfg.audio)
    import soundfile as sf

    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    waves = [
        torch.from_numpy(sf.read(str(corpus / r["wav_path"]), dtype="float32")[0])
        for r in records[: args.eval_utterances]
    ]
    print(f"corpus: {len(records)} utterances | evaluating on {len(waves)} | "
          f"{count_parameters(build_model(cfg)) / 1e6:.3f} M params | {args.steps} steps per variant")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    results = []
    for name in variants:
        print(f"\n  --- {name} ---", flush=True)
        results.append(
            train_variant(
                name, cfg, waves, source, args.steps, mel,
                log_every=max(1, args.steps // 4),
                adversarial_from=int(0.66 * args.steps),
            )
        )

    print("\n  variant            s/step | mel L1 (before -> after) | wav cosine | SNR dB | DNSMOS | phase coh")
    for row in results:
        b, a = row["before"], row["after"]
        print(
            f"  {row['variant']:16s} {row['seconds_per_step']:6.2f} | "
            f"{b['mel_l1']:.4f} -> {a['mel_l1']:.4f} | {a['waveform_cosine']:+.3f} | "
            f"{a['snr_db']:6.2f} | {a['dnsmos']:5.2f} | {a['phase_coherence']:.3f}"
        )

    best = max(results, key=lambda r: (r["after"]["waveform_cosine"], -r["after"]["mel_l1"]))
    payload = {
        "config": args.config,
        "steps": args.steps,
        "utterances": len(records),
        "variants": results,
        "best_by_waveform_correlation": best["variant"],
        "notes": [
            "equal steps and equal seed per variant; wall clock differs because of the discriminator",
            "waveform correlation is the decisive metric: the round-19 failure was an uncorrelated "
            "round-trip that the mel proxy rated merely poor",
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  best by waveform correlation: {best['variant']}")
    print(f"report -> {out/'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
