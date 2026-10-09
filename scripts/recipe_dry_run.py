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
from parakeet.train.stages import flow_step, run_stage  # noqa: E402

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
    ap.add_argument("--stage", default="distill-text", choices=["distill-text", "flow"],
                    help="which student stage to exercise: Tiny's distillation or Small's flow")
    ap.add_argument("--teachers", default="stub_low=0.6,stub_high=0.4")
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--steps-ae", type=int, default=120)
    ap.add_argument("--steps-student", type=int, default=None,
                    help="steps for --stage (defaults to 250, or 120 for flow)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-ref-frames", type=int, default=1500,
                    help="cap the conditioning reference (PilotTTS truncates the prompt at 15 s)")
    ap.add_argument("--out", default="runs/recipe_dry_run")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps_ae = 25
        args.prompts = 4
    if args.steps_student is None:
        args.steps_student = 120 if args.stage == "flow" else 250
        if args.quick:
            args.steps_student = 40 if args.stage == "flow" else 60
    if args.batch_size is None:
        args.batch_size = 4
    if args.stage == "flow":
        # the shipped Small config (45 M) is far too slow for a dry run: shrink the dims that matter
        # for *shape*, keep the architecture, and make sure the flow model exists
        args.config = args.config if args.config != "configs/parakeet_tiny.yaml" else "configs/parakeet_small.yaml"

    cfg = load_config(args.config)
    if args.stage == "flow":
        from dataclasses import replace as _replace

        cfg.variant = "small"
        cfg.voice_mode = "reference"
        cfg.n_voices = 3  # overridden below by the number of voices actually in the corpus
        cfg.autoencoder.encoder_dims = [32, 48, 64]
        cfg.autoencoder.encoder_blocks = [1, 2, 2]
        cfg.autoencoder.decoder_dim = 64
        cfg.autoencoder.decoder_blocks = 2
        cfg.text.dim = 64
        cfg.text.n_layers = 2
        cfg.text.n_heads = 4
        cfg.flow.dim = 64
        cfg.flow.depth = 2
        cfg.flow.n_heads = 4
        cfg.flow.text_dim = 64
        cfg.flow.cond_dim = 64
        cfg.speaker.channels = [16, 24]
        cfg.speaker.emb_dim = 32
        cfg.speaker.style_dim = 32
        cfg.speaker.n_query = 4
        cfg.duration.hidden = 64
        cfg = cfg.validate()
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
        prompts,
        out / "corpus",
        mix=mix,
        backends=backends,
        max_utts=len(prompts),
        # multiple voices per teacher: cross-sample pairing needs a *different* utterance of the
        # same voice, and the style-separation term needs a different voice
        voices={name: ["v0", "v1"] for name in mix},
    )
    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    corpus_meta = json.loads((out / "corpus" / "corpus_meta.json").read_text(encoding="utf-8"))
    hours = corpus_meta["hours"]
    print(f"  -> {len(records)} utterances, {hours * 3600:.1f}s audio, "
          f"{len({r['teacher'] for r in records})} teachers in the manifest")
    strict["corpus_uses_both_teachers"] = len({r["teacher"] for r in records}) == len(mix)

    # the corpus decides how many voices the student needs; build_latent_cache refuses a mismatch
    # rather than letting the voice embedding index out of range mid-training
    corpus_voices = sorted({str(r.get("voice") or "") for r in records})
    cfg.n_voices = max(1, len(corpus_voices))
    cfg = cfg.validate()
    print(f"  -> voices present: {corpus_voices} (n_voices={cfg.n_voices})")

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

    # ---------------- 3. curation + latent cache ----------------
    _banner("3/6 curation (P1 gates) + latent cache")
    import soundfile as sf

    from parakeet.data.curate import CurateConfig, curate_manifest
    from parakeet.data.features import cache_teacher_corpus

    def _load_wav(rel: str):
        wav, sr = sf.read(str(Path(args.out) / "corpus" / rel), dtype="float32")
        return torch.from_numpy(wav), sr

    # The published thresholds are calibrated for real 24 kHz speech (CosyVoice: >= 3 s; >= 5 kHz
    # bandwidth).  The fixtures are ~0.3 s band-limited synthetic stacks, so the published gates
    # reject every one of them -- a property of the fixture, not a defect.  Both runs are reported,
    # and the fixture-appropriate config is the one used, because otherwise there would be no data.
    published = curate_manifest(records, _load_wav, out / "corpus" / "curated_published",
                                CurateConfig(), normalize=True)
    relaxed_cfg = CurateConfig(min_duration_s=0.1, min_bandwidth_hz=0.0, min_snr_db=0.0)
    report = curate_manifest(records, _load_wav, out / "corpus" / "curated", relaxed_cfg,
                             normalize=True)
    print(f"  published gates : kept {published.n_kept}/{published.n_total} "
          f"{dict(published.reason_counts)}")
    print(f"  fixture gates   : kept {report.n_kept}/{report.n_total}, "
          f"rejected {report.n_rejected} {dict(report.reason_counts)}")
    print("  (fixtures are ~0.3 s band-limited stacks; the published duration/bandwidth gates reject "
          "100 % of them. Level, clipping and silence gates are NOT relaxed.)")
    strict["curation_ran"] = report.n_total == len(records)
    strict["curation_kept_usable_data"] = report.n_kept >= 1

    cache_dir = cache_teacher_corpus(
        out / "corpus",
        out / "latent_cache",
        cfg,
        model.autoencoder,
        tokenizer=tokenizer,
        teacher_latent_norm=model.latent_norm,
    )
    cache_meta = json.loads((cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
    print(f"  -> {cache_meta['n_shards']} shard(s) from the curated manifest | "
          f"teachers {cache_meta['teacher_names']} | weights {cache_meta['teacher_weights']} | "
          f"voices {cache_meta['voice_names']}")
    strict["cache_uses_the_curated_manifest"] = len(
        json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    ) >= 1 and (out / "corpus" / "curated" / "kept.jsonl").exists()
    strict["cache_records_mixture"] = cache_meta["teacher_weights"] == mix

    dataset = LatentShardDataset(cache_dir)
    # pair_references: PilotTTS cross-sample paired training -- the speaker/style reference is a
    # *different* utterance of the same voice, never the target itself
    loader = LatentShardBatchSource(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.stage != "flow",
        seed=0,
        pair_references=args.stage == "flow",
        max_ref_frames=args.max_ref_frames,
    )
    batch = loader()
    weights = batch.get("teacher_weight")
    print(f"  -> dataset {len(dataset)} items | batch teacher_weight "
          f"{[round(float(w), 3) for w in weights] if weights is not None else None}")
    strict["weights_reach_the_batch"] = weights is not None and len(set(weights.tolist())) > 1
    if args.stage == "flow":
        print(f"  -> references: ref_mel {tuple(batch['ref_mel'].shape)} "
              f"({int(batch['ref_mask'][0].sum())} frames valid), "
              f"negative reference {'present' if 'ref_mel_neg' in batch else 'MISSING'}")
        strict["references_reach_the_flow_stage"] = (
            "ref_mel" in batch and "ref_mel_neg" in batch and bool(batch["ref_mask"].any())
        )

    # ---------------- 4. the student stage ----------------
    if args.stage == "flow":
        _banner(f"4/6 flow | {args.steps_student} steps with paired references")
        fixed = loader()
        torch.manual_seed(0)
        before, _ = flow_step(cfg, model, fixed)
        logs = run_stage("flow", cfg, model=model, batches=loader,
                         max_steps=args.steps_student, out_dir=str(out))
        torch.manual_seed(0)
        after, after_logs = flow_step(cfg, model, fixed)
        print(f"  -> flow loss {float(before):.4f} -> {float(after):.4f} | "
              f"style separation {float(after_logs.get('style_separation', float('nan'))):.4f}")
        strict["student_improved"] = float(after) < float(before)
        strict["style_separation_is_active"] = "style_separation" in after_logs
    else:
        _banner(f"4/6 distill-text | {args.steps_student} steps on cached teacher signals")
        text_before = teacher_signal_loss(model, [dataset[i] for i in range(len(dataset))], cfg)
        logs = run_stage("distill-text", cfg, model=model, batches=loader,
                         max_steps=args.steps_student, out_dir=str(out))
        text_after = teacher_signal_loss(model, [dataset[i] for i in range(len(dataset))], cfg)
        print(f"  -> teacher-signal loss {text_before:.4f} -> {text_after:.4f} | "
              f"final stage loss {float(logs.get('loss', float('nan'))):.4f}")
        strict["student_improved"] = text_after < text_before

    # ---------------- 5. synthesis ----------------
    _banner("5/6 synthesis from text")
    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=True)
    text = prompts[0]
    flow_mode = args.stage == "flow"
    # an untrained length predictor picks an arbitrary (often 1-frame) duration, so pin the length
    # when exercising the flow path; the Tiny path derives it from predicted durations
    n_latent = int(dataset[0]["latent"].shape[-1]) if flow_mode else None
    wav = synth.synthesize(
        text, seed=0, ref_wav=wavs[0] if flow_mode else None, n_latent_frames=n_latent
    )
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

    # Conditioning sensitivity, measured where it is actually observable on an untrained model.
    # The acoustic comparison is useless here and we say so: the flow estimator's residual branches
    # start at layer_scale 1e-6, so its output is dominated by x0 and two different references give
    # near-identical audio (~1e-6 relative).  The structural measurement -- does the reference change
    # the conditioning the sampler consumes? -- is the meaningful one at this stage.
    if flow_mode and len(wavs) > 1:
        other = synth.synthesize(text, seed=0, ref_wav=wavs[-1], n_latent_frames=n_latent)
        m = min(wav.shape[-1], other.shape[-1])
        cosine = float(
            torch.nn.functional.cosine_similarity(
                wav[..., :m].flatten(), other[..., :m].flatten(), dim=0
            )
        )
        mask_ids = torch.ones(1, len(tokenizer.encode(text, add_special=False)), dtype=torch.bool)
        ids_t = tokenizer.encode(text, add_special=False)[None]
        ref_a, mask_a = synth._reference_tensors(wavs[0], None)
        ref_b, mask_b = synth._reference_tensors(wavs[-1], None)
        with torch.no_grad():
            _m1, _k1, cond_a = model.conditions(ids_t, mask_ids, ref_a, mask_a)
            _m2, _k2, cond_b = model.conditions(ids_t, mask_ids, ref_b, mask_b)
            _m3, _k3, cond_none = model.conditions(ids_t, mask_ids, None, None)
        cond_delta = float((cond_a - cond_b).abs().max())
        cond_vs_null = float((cond_a - cond_none).abs().max())
        print(f"  -> reference sensitivity: conditioning differs by {cond_delta:.4f} between two "
              f"references, {cond_vs_null:.4f} against the null fallback")
        print(f"  -> (acoustic cosine {cosine:.6f} is uninformative here: an untrained flow "
              f"estimator is dominated by x0 -- layer_scale_init 1e-6)")
        strict["reference_changes_the_conditioning"] = cond_delta > 1e-3 and cond_vs_null > 1e-3

    # ---------------- 6. report ----------------
    _banner("6/6 summary")
    student_before, student_after = (
        (float(before), float(after)) if flow_mode else (text_before, text_after)
    )
    report = {
        "config": args.config,
        "stage": args.stage,
        "mixture": mix,
        "prompts": len(prompts),
        "corpus": {"utterances": len(records), "audio_seconds": hours * 3600,
                   "teachers": sorted({r["teacher"] for r in records})},
        "steps": {"autoencoder": args.steps_ae, args.stage: args.steps_student},
        "ae_recon_before": ae_before,
        "ae_recon_after": ae_after,
        "student_loss_before": student_before,
        "student_loss_after": student_after,
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
