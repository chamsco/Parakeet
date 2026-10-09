"""End-to-end recipe dry run -- the whole pipeline, offline, with no teacher dependencies.

    python scripts/recipe_dry_run.py --quick      # ~1 minute
    python scripts/recipe_dry_run.py              # ~4 minutes on 8 CPU cores

Every other demo in this repo tests a *part*: `learn_demo` the student halves,
`reflow_demo` the sampler, `streaming_demo` the streaming path, `mixture_demo` the mixture.
This one runs the actual recipe in the order the documentation claims, using two synthetic
"teachers" so no network, no API key and no 3B checkpoint is needed:

    prompts
      -> synthesize_corpus        (two fixture teachers, interleaved by the mixture)
      -> manifest.jsonl           (per-record provenance: teacher, voice, license, hash)
      -> build_latent_cache       (frozen autoencoder; per-record teacher_weight)
      -> LatentShardDataset       (the real dataset + collation path)
      -> run_stage("distill-text")  with the mixture weights reaching the loss
      -> Synthesizer.synthesize   (text -> audio, phase lock, int8 available)

The point is composition: a break anywhere in this chain is invisible to the per-part demos.
The fixtures are *not* real teachers -- a model trained here is a plumbing test, never a speech
model -- which is exactly why they are excluded from the default mixture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.audio.mel import MelSpectrogram  # noqa: E402
from parakeet.config import load_config  # noqa: E402
from parakeet.data.dataset import LatentShardBatchSource, LatentShardDataset  # noqa: E402
from parakeet.data.features import build_latent_cache, fit_latent_normalizer  # noqa: E402
from parakeet.data.teacher import build_backend, resolve_mix, synthesize_corpus  # noqa: E402
from parakeet.data.text import TextTokenizer  # noqa: E402
from parakeet.eval import ae_reconstruction_l1, teacher_signal_loss  # noqa: E402
from parakeet.inference import Synthesizer, phase_coherence, write_wav  # noqa: E402
from parakeet.models import build_model  # noqa: E402
from parakeet.train.stages import run_stage  # noqa: E402

PROMPTS = [
    "the quick brown fox jumps over the lazy dog",
    "parakeet distills a mixture of teachers into one small voice",
    "a dry run exercises the whole recipe without any teacher weights",
    "every stage writes provenance so the mixture can be audited later",
    "streaming synthesis starts speaking before the sentence is finished",
    "the autoencoder turns audio into a continuous latent representation",
]


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the whole Parakeet recipe offline")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--teachers", default="stub_low=0.6,stub_high=0.4")
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--steps-ae", type=int, default=120)
    ap.add_argument("--steps-text", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="runs/recipe_dry_run")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps_ae, args.steps_text, args.prompts = 25, 60, 4

    cfg = load_config(args.config)
    cfg.train.save_every = 0
    cfg.train.log_every = max(1, args.steps_ae)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = TextTokenizer(mode=cfg.text.mode)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t_start = time.perf_counter()
    strict: dict[str, bool] = {}

    # ---------------- 1. teacher corpus ----------------
    mix = resolve_mix(args.teachers.split(","))
    prompts = PROMPTS[: args.prompts]
    _banner(f"1/6 teacher corpus | mixture {mix} | {len(prompts)} prompts")
    backends = {name: build_backend(name) for name in mix}
    for name, backend in backends.items():
        print(f"  {name}: {backend.spec.kind} | {backend.spec.notes.splitlines()[0][:70]}")
    manifest = synthesize_corpus(
        prompts, out / "corpus", mix=mix, backends=backends, max_utts=len(prompts)
    )
    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    corpus_meta = json.loads((out / "corpus" / "corpus_meta.json").read_text(encoding="utf-8"))
    hours = corpus_meta["hours"]
    print(f"  -> {len(records)} utterances, {hours * 3600:.1f}s audio, "
          f"{len({r['teacher'] for r in records})} teachers in the manifest")
    strict["corpus_uses_both_teachers"] = len({r["teacher"] for r in records}) == len(mix)

    # ---------------- 2. autoencoder on the corpus ----------------
    _banner(f"2/6 autoencoder | {args.steps_ae} steps on the corpus audio")
    import soundfile as sf

    wavs = [torch.from_numpy(sf.read(str(out / "corpus" / r["wav_path"]), dtype="float32")[0])
            for r in records]

    class _CorpusSource:
        def __init__(self, waves, batch_size, seed=0):
            self.waves = waves
            self.batch_size = batch_size
            self.generator = torch.Generator().manual_seed(seed)

        def __call__(self):
            idx = torch.randint(len(self.waves), (self.batch_size,), generator=self.generator)
            batch = [self.waves[int(i)] for i in idx]
            width = max(b.numel() for b in batch)
            out_batch = torch.zeros(len(batch), width)
            for i, b in enumerate(batch):
                out_batch[i, : b.numel()] = b
            return {"wav": out_batch}

    source = _CorpusSource(wavs, args.batch_size, seed=cfg.train.seed)
    model = build_model(cfg)
    ae_before = ae_reconstruction_l1(model, wavs, cfg)
    run_stage("autoencoder", cfg, model=model, batches=source, max_steps=args.steps_ae, out_dir=str(out))
    fit_latent_normalizer(model.latent_norm, model.autoencoder, source, cfg, max_batches=4)
    ae_after = ae_reconstruction_l1(model, wavs, cfg)
    print(f"  -> reconstruction log-mel L1 {ae_before:.4f} -> {ae_after:.4f}")
    strict["autoencoder_improved"] = ae_after < ae_before

    # ---------------- 3. latent cache with mixture provenance ----------------
    _banner("3/6 latent cache (frozen autoencoder, per-record teacher_weight)")
    cache_dir = build_latent_cache(
        manifest,
        out / "latent_cache",
        cfg,
        model.autoencoder,
        tokenizer=tokenizer,
        teacher_latent_norm=model.latent_norm,
        teacher_weights=mix,
    )
    cache_meta = json.loads((cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
    print(f"  -> {cache_meta['n_shards']} shard(s), teachers {cache_meta['teacher_names']}, "
          f"weights {cache_meta['teacher_weights']}")
    strict["cache_records_mixture"] = cache_meta["teacher_weights"] == mix

    dataset = LatentShardDataset(cache_dir)
    loader = LatentShardBatchSource(dataset, batch_size=args.batch_size, shuffle=False, seed=0)
    batch = loader()
    weights = batch.get("teacher_weight")
    print(f"  -> dataset {len(dataset)} items | batch teacher_weight "
          f"{[round(float(w), 3) for w in weights] if weights is not None else None}")
    strict["weights_reach_the_batch"] = weights is not None and len(set(weights.tolist())) > 1

    # ---------------- 4. distillation with the mixture ----------------
    _banner(f"4/6 distill-text | {args.steps_text} steps on cached teacher signals")
    text_before = teacher_signal_loss(model, [dataset[i] for i in range(len(dataset))], cfg)
    logs = run_stage(
        "distill-text", cfg, model=model, batches=loader, max_steps=args.steps_text, out_dir=str(out)
    )
    text_after = teacher_signal_loss(model, [dataset[i] for i in range(len(dataset))], cfg)
    print(f"  -> teacher-signal loss {text_before:.4f} -> {text_after:.4f} | "
          f"final stage loss {float(logs.get('loss', float('nan'))):.4f}")
    strict["distillation_improved"] = text_after < text_before

    # ---------------- 5. synthesis ----------------
    _banner("5/6 synthesis from text")
    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=True)
    text = prompts[0]
    wav = synth.synthesize(text, seed=0)
    write_wav(out / "synthesized.wav", wav, cfg.audio.sample_rate)
    reference = wavs[0]
    mel = MelSpectrogram(cfg.audio)
    n = min(wav.shape[-1], reference.numel())
    mel_l1 = float(
        torch.nn.functional.l1_loss(
            mel.log_mel(wav[..., :n]), mel.log_mel(reference[:n].reshape(1, -1))
        )
    )
    coherence = float(phase_coherence(wav, sample_rate=cfg.audio.sample_rate).item())
    print(f"  -> {wav.shape[-1] / cfg.audio.sample_rate:.2f}s audio | mel L1 vs corpus audio "
          f"{mel_l1:.4f} | phase coherence 2-8k {coherence:.4f}")
    strict["synthesis_produces_audio"] = wav.shape[-1] > 0 and torch.isfinite(wav).all().item()

    # ---------------- 6. report ----------------
    _banner("6/6 summary")
    report = {
        "config": args.config,
        "mixture": mix,
        "prompts": len(prompts),
        "corpus": {"utterances": len(records), "audio_seconds": hours * 3600,
                   "teachers": sorted({r["teacher"] for r in records})},
        "steps": {"autoencoder": args.steps_ae, "distill_text": args.steps_text},
        "ae_recon_before": ae_before,
        "ae_recon_after": ae_after,
        "teacher_signal_loss_before": text_before,
        "teacher_signal_loss_after": text_after,
        "batch_teacher_weights": [float(w) for w in weights] if weights is not None else None,
        "synthesis": {"mel_l1_vs_corpus": mel_l1, "phase_coherence": coherence,
                      "seconds": wav.shape[-1] / cfg.audio.sample_rate},
        "checks": strict,
        "seconds_total": time.perf_counter() - t_start,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, ok in strict.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nreport -> {out/'report.json'}\naudio  -> {out/'synthesized.wav'}")
    print("RECIPE DRY RUN " + ("PASSED" if all(strict.values()) else "FAILED"))
    return 0 if all(strict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
