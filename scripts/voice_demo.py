"""Multi-voice distillation: does the voice embedding actually reach the student?

    python scripts/voice_demo.py --quick     # ~1 minute
    python scripts/voice_demo.py             # ~3 minutes

Same story as the teacher mixture in round 6: the config had ``n_voices``, the model had a
``voice_embed``, and the manifest had carried a ``voice`` per record since the first round -- but
nothing in the cache, collation or training loop ever produced a voice tensor, so every sample was
trained as voice 0.  This demo pins the mechanism:

* three fixture voices (pitch multipliers 0.85 / 1.0 / 1.6) render *the same texts*, so the only
  difference between samples is the voice;
* the Tiny text side is trained twice from an identical initialisation -- once with the voice index
  reaching the model, once with it forced to 0 (exactly what the code did before the fix);
* if voice conditioning works, the first run must separate the voices' predicted F0 for a fixed
  text, and the control must not.

The fixtures are not real teachers: this measures wiring, not speech quality.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.audio.f0 import normalized_to_f0  # noqa: E402
from parakeet.config import ParakeetConfig, load_config  # noqa: E402
from parakeet.data.dataset import LatentShardBatchSource, LatentShardDataset  # noqa: E402
from parakeet.data.features import build_latent_cache, fit_latent_normalizer  # noqa: E402
from parakeet.data.teacher import build_backend, synthesize_corpus  # noqa: E402
from parakeet.data.text import TextTokenizer  # noqa: E402
from parakeet.inference import Synthesizer, write_wav  # noqa: E402
from parakeet.models import build_model  # noqa: E402
from parakeet.train.stages import run_stage  # noqa: E402

TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "a small model can still speak with several distinct voices",
]
VOICES = ["low", "mid", "high"]


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def tiny_cfg(base: ParakeetConfig) -> ParakeetConfig:
    cfg = copy.deepcopy(base)
    cfg.n_voices = len(VOICES)
    cfg.autoencoder.encoder_dims = [48, 64, 96]
    cfg.autoencoder.encoder_blocks = [1, 2, 2]
    cfg.autoencoder.decoder_dim = 128
    cfg.autoencoder.decoder_blocks = 3
    cfg.text.dim = 128
    cfg.text.n_layers = 3
    cfg.duration.hidden = 128
    cfg.train.lr = 2e-3
    # text.dim is coupled to the flow module's memory width by validation
    cfg.flow.text_dim = cfg.text.dim
    cfg.flow.cond_dim = cfg.text.dim
    cfg.flow.dim = 128
    cfg.flow.n_heads = 4
    return cfg.validate()


def _corpus(cfg, out: Path, prompts: List[str]):
    """Each text is emitted once per voice, so voice is the only systematic difference."""
    repeated = [t for t in prompts for _ in VOICES]
    backends = {"stub_low": build_backend("stub_low")}
    return synthesize_corpus(
        repeated,
        out,
        mix={"stub_low": 1.0},
        voices={"stub_low": VOICES},
        backends=backends,
        max_utts=len(repeated),
    )


def train(cfg, cache_dir, steps: int, batch_size: int, use_voice: bool):
    """Train the text side; the control zeroes the voice index (the pre-fix behaviour)."""
    torch.manual_seed(cfg.train.seed)
    model = build_model(cfg)
    dataset = LatentShardDataset(cache_dir)
    loader = LatentShardBatchSource(dataset, batch_size=batch_size, shuffle=True, seed=0)
    probe = [dataset[i] for i in range(len(dataset))]
    from parakeet.eval import teacher_signal_loss

    before = teacher_signal_loss(model, probe, cfg)
    if use_voice:
        run_stage("distill-text", cfg, model=model, batches=loader, max_steps=steps,
                  out_dir=str(cache_dir.parent / "train"))
    else:
        def control_loader():
            batch = loader()
            batch["voice"] = torch.zeros_like(batch["voice"])
            return batch

        run_stage("distill-text", cfg, model=model, batches=control_loader, max_steps=steps,
                  out_dir=str(cache_dir.parent / "train_control"))
    after = teacher_signal_loss(model, probe, cfg)
    return model, before, after


@torch.no_grad()
def per_voice_f0(model, ids: torch.Tensor, n_voices: int) -> List[float]:
    """Predicted mean F0 for one fixed text, conditioned on each voice index."""
    values = []
    for v in range(n_voices):
        side = model.text_side(ids[None], voice=torch.tensor([v]))
        values.append(float(normalized_to_f0(side["f0"][0]).mean().item()))
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify multi-voice voice conditioning")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--steps-ae", type=int, default=80)
    ap.add_argument("--steps-text", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="runs/voice_demo")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps_ae, args.steps_text = 20, 60

    base = load_config(args.config)
    cfg = tiny_cfg(base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = TextTokenizer(mode=cfg.text.mode)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t_start = time.perf_counter()

    # ---------------- corpus: same texts, three voices ----------------
    _banner(f"corpus: {len(TEXTS)} texts x {len(VOICES)} voices = {len(TEXTS) * len(VOICES)} utterances")
    manifest = _corpus(cfg, out / "corpus", TEXTS)
    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    manifest_voices = {r["voice"] for r in records}
    print(f"  -> {len(records)} records | voices {sorted(manifest_voices)}")

    # ---------------- autoencoder fixture + cache ----------------
    import soundfile as sf

    wavs = [torch.from_numpy(sf.read(str(out / "corpus" / r["wav_path"]), dtype="float32")[0])
            for r in records]

    class _Source:
        def __init__(self, waves, batch_size, seed=0):
            self.waves, self.batch_size = waves, batch_size
            self.generator = torch.Generator().manual_seed(seed)

        def __call__(self):
            idx = torch.randint(len(self.waves), (self.batch_size,), generator=self.generator)
            picked = [self.waves[int(i)] for i in idx]
            width = max(p.numel() for p in picked)
            batch = torch.zeros(len(picked), width)
            for i, p in enumerate(picked):
                batch[i, : p.numel()] = p
            return {"wav": batch}

    source = _Source(wavs, args.batch_size, seed=cfg.train.seed)
    model = build_model(cfg)
    run_stage("autoencoder", cfg, model=model, batches=source, max_steps=args.steps_ae,
              out_dir=str(out))
    fit_latent_normalizer(model.latent_norm, model.autoencoder, source, cfg, max_batches=4)

    cache = build_latent_cache(
        manifest, out / "latent_cache", cfg, model.autoencoder, tokenizer=tokenizer,
        teacher_latent_norm=model.latent_norm, teacher_weights={"stub_low": 1.0},
    )
    meta = json.loads((cache / "cache_meta.json").read_text(encoding="utf-8"))
    print(f"  -> cache voice_names {meta['voice_names']}")

    # ---------------- train with and without voice conditioning ----------------
    _banner(f"train the text side twice from the same init ({args.steps_text} steps each)")
    model_voice, before_v, after_v = train(cfg, cache, args.steps_text, args.batch_size, True)
    model_ctrl, before_c, after_c = train(cfg, cache, args.steps_text, args.batch_size, False)
    print(f"  with voice : teacher-signal loss {before_v:.4f} -> {after_v:.4f}")
    print(f"  control    : teacher-signal loss {before_c:.4f} -> {after_c:.4f}")

    # ---------------- per-voice pitch for one fixed text ----------------
    _banner("predicted F0 for the SAME text under each voice index")
    ids = torch.tensor(tokenizer.encode(TEXTS[0], add_special=False))
    f0_voice = per_voice_f0(model_voice, ids, len(VOICES))
    f0_ctrl = per_voice_f0(model_ctrl, ids, len(VOICES))
    for i, name in enumerate(VOICES):
        print(f"  voice {i} ({name:4s}): with conditioning {f0_voice[i]:6.1f} Hz | "
              f"control {f0_ctrl[i]:6.1f} Hz")
    span_voice = max(f0_voice) - min(f0_voice)
    span_ctrl = max(f0_ctrl) - min(f0_ctrl)
    print(f"  span across voices: with conditioning {span_voice:.1f} Hz | control {span_ctrl:.1f} Hz")

    # ---------------- audio per voice ----------------
    synth = Synthesizer(model_voice, cfg, device="cpu", apply_phase_lock=True)
    for i, name in enumerate(VOICES):
        wav = synth.synthesize(TEXTS[0], voice=i, seed=0)
        write_wav(out / f"voice_{i}_{name}.wav", wav, cfg.audio.sample_rate)
    print(f"  -> wrote {len(VOICES)} wavs: {', '.join(f'voice_{i}_{n}.wav' for i, n in enumerate(VOICES))}")

    checks = {
        "cache_carries_all_voices": len(meta["voice_names"]) == len(VOICES),
        # the student must reproduce the fixtures' *relative* pitch, not merely differ: an inverted
        # ordering is not "voice conditioning working"
        "voice_f0_ordering_matches_fixture": f0_voice[0] < f0_voice[1] < f0_voice[2],
        # and it must fit the per-voice targets better than a model that never sees the voice index
        "conditioning_fits_better_than_control": after_v < after_c,
        "both_runs_learned": after_v < before_v and after_c < before_c,
    }
    report = {
        "config": args.config,
        "n_voices": cfg.n_voices,
        "manifest_voices": sorted(manifest_voices),
        "cache_voice_names": meta["voice_names"],
        "steps": {"ae": args.steps_ae, "text": args.steps_text},
        "loss": {"with_voice": [before_v, after_v], "control": [before_c, after_c]},
        "predicted_f0_hz": {"with_voice": f0_voice, "control": f0_ctrl},
        "span_hz": {"with_voice": span_voice, "control": span_ctrl},
        "checks": checks,
        "seconds_total": time.perf_counter() - t_start,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _banner("RESULT")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nreport -> {out/'report.json'}")
    print("VOICE DEMO " + ("PASSED" if all(checks.values()) else "FAILED"))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
