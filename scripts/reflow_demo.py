"""Few-step sampler validation: does Reflow actually make NFE 2 usable?

    python scripts/reflow_demo.py --quick        # ~2 minutes
    python scripts/reflow_demo.py                # ~15 minutes on 8 CPU cores

This is the experiment behind the "lightning fast" claim for Parakeet-Small.  SupertonicTTS needs
NFE 32 and cutting steps naively collapses quality (their WER degrades to 11.43 at NFE 4 versus
2.64 at NFE 32), so our recipe includes a Reflow (2-rectified-flow) stage whose job is to make a
2-step sampler good.  That claim is testable on CPU without any corpus:

1. train the flow-matching estimator on latents from a briefly-trained autoencoder
   (the autoencoder is a *fixture* here -- the sampler is what is under test);
2. snapshot the pre-Reflow student, and take the EMA model's **NFE-32 endpoint** as the reference;
3. compare, from *identical noise and conditioning*:
     * naive cut      -- pre-Reflow student at NFE 2
     * reflowed       -- post-Reflow student at NFE 2
     * teacher@16 and teacher@8 -- to show how much error is sampler discretisation
4. score latent MSE against the NFE-32 endpoint, and audio log-mel L1 against the target audio;
5. report the wall-clock cost of NFE 32 vs NFE 2.

Pass criteria: the reflowed 2-step sampler must be **closer to the teacher endpoint** than the
naive cut, in both latent and audio space.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.audio.mel import MelSpectrogram  # noqa: E402
from parakeet.config import load_config  # noqa: E402
from parakeet.data.features import fit_latent_normalizer  # noqa: E402
from parakeet.data.synthetic import SyntheticSpeechBatchSource, make_corpus  # noqa: E402
from parakeet.models import build_model  # noqa: E402
from parakeet.models.flow import consistency_sample, fold_time, unfold_time  # noqa: E402
from parakeet.train.common import EMAModel, build_optimizer, seed_everything  # noqa: E402
from parakeet.train.stages import flow_step, reflow_step, run_stage  # noqa: E402


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


class LatentFlowSource:
    """Batches of (text, reference mel, normalised latent) for flow-matching training."""

    def __init__(self, model, corpus, cfg, tokenizer, batch_size: int = 2, seed: int = 0) -> None:
        self.model = model
        self.cfg = cfg
        self.mel = MelSpectrogram(cfg.audio)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.groups: Dict[str, list] = {}
        for utt in corpus:
            self.groups.setdefault(utt.layout, []).append(utt)

    def __call__(self) -> Dict[str, torch.Tensor]:
        layouts = sorted(self.groups)
        choice = int(torch.randint(len(layouts), (1,), generator=self.generator).item())
        items = self.groups[layouts[choice]]
        idx = [
            int(torch.randint(len(items), (1,), generator=self.generator).item())
            for _ in range(self.batch_size)
        ]
        wavs = torch.stack([items[i].wav for i in idx], dim=0)
        log_mel = self.mel.log_mel(wavs)
        with torch.no_grad():
            latent = self.model.latent_norm.normalize(self.model.autoencoder.encode(log_mel))
        ids = torch.stack(
            [self.tokenizer.encode(items[i].text, add_special=False) for i in idx], dim=0
        )
        return {
            "ids": ids,
            "text_mask": torch.ones_like(ids, dtype=torch.bool),
            "latent": latent,
            "ref_mel": log_mel,
            "ref_mask": torch.ones(log_mel.shape[0], log_mel.shape[-1], dtype=torch.bool),
            "wav": wavs,
        }


@torch.no_grad()
def endpoint(
    model,
    batch: Dict[str, torch.Tensor],
    steps: int,
    x0: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a full endpoint from (optionally shared) noise at a given NFE."""
    memory, memory_mask, _ = model.conditions(
        batch["ids"], batch["text_mask"], batch["ref_mel"], batch["ref_mask"]
    )
    b, _, t_latent = batch["latent"].shape
    tc = model.compressed_frames(t_latent)
    shape = (b, model.cfg.flow.latent_dim * model.cfg.flow.compress, tc)
    if x0 is None:
        x0 = torch.randn(shape)
    x1 = consistency_sample(model.vf, memory, memory_mask, shape, steps=steps, device=x0.device, cfg_scale=1.0)
    return x1, x0


@torch.no_grad()
def decode_endpoint(model, batch: Dict[str, torch.Tensor], x1c: torch.Tensor) -> torch.Tensor:
    latent = unfold_time(x1c, model.cfg.flow.compress, t_out=batch["latent"].shape[-1])
    latent = model.latent_norm.denormalize(latent)
    return model.autoencoder.decode(latent)


def mel_l1(mel: MelSpectrogram, wav_a: torch.Tensor, wav_b: torch.Tensor) -> float:
    a, b = mel.log_mel(wav_a), mel.log_mel(wav_b)
    n = min(a.shape[-1], b.shape[-1])
    return float(F.l1_loss(a[..., :n], b[..., :n]).item())


@torch.no_grad()
def ae_floor_mel_l1(model, mel: MelSpectrogram, batch: Dict[str, torch.Tensor]) -> float:
    """The AE's own round-trip error: the floor below which no sampler comparison can discriminate."""
    latent = model.latent_norm.denormalize(batch["latent"])
    recon = model.autoencoder.decode(latent)
    return mel_l1(mel, recon, batch["wav"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate few-step (Reflow) sampling")
    ap.add_argument("--config", default="configs/parakeet_small.yaml")
    ap.add_argument("--utterances", type=int, default=16)
    ap.add_argument("--steps-ae", type=int, default=80)
    ap.add_argument("--steps-flow", type=int, default=300)
    ap.add_argument("--steps-reflow", type=int, default=150)
    ap.add_argument("--nfe-teacher", type=int, default=16, help="NFE used to build Reflow targets")
    ap.add_argument("--nfe-reference", type=int, default=32, help="NFE of the reference endpoint")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--eval-items", type=int, default=4)
    ap.add_argument("--out", default="runs/reflow_demo")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps_ae, args.steps_flow, args.steps_reflow = 10, 40, 20
        args.nfe_teacher, args.nfe_reference, args.eval_items = 4, 8, 2

    cfg = load_config(args.config)
    cfg.train.save_every = 0
    cfg.train.log_every = 10**9
    seed_everything(cfg.train.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    corpus = make_corpus(args.utterances, cfg.audio, seed=cfg.train.seed)
    seconds = sum(u.wav.numel() for u in corpus) / cfg.audio.sample_rate
    _banner(f"{cfg.name}: 45M-class flow model | corpus {len(corpus)} utts / {seconds:.1f}s synthetic")
    t_start = time.perf_counter()

    model = build_model(cfg)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M | "
          f"VF {sum(p.numel() for p in model.vf.parameters())/1e6:.3f}M")

    # ---------------- autoencoder fixture + latent targets ----------------
    wave_source = SyntheticSpeechBatchSource(corpus, batch_size=args.batch_size, seed=cfg.train.seed)
    run_stage("autoencoder", cfg, model=model, batches=wave_source, max_steps=args.steps_ae, out_dir=str(out))
    fit_latent_normalizer(model.latent_norm, model.autoencoder, wave_source, cfg, max_batches=6)
    from parakeet.data.text import TextTokenizer

    source = LatentFlowSource(model, corpus, cfg, TextTokenizer(mode=cfg.text.mode),
                              batch_size=args.batch_size, seed=cfg.train.seed)
    print(f"[fixture] autoencoder trained {args.steps_ae} steps, normaliser fitted "
          f"({time.perf_counter() - t_start:.0f}s elapsed)")

    # ---------------- flow matching ----------------
    opt = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    ema = EMAModel(model, cfg.train.ema_decay)
    flow_losses: List[float] = []
    for step in range(args.steps_flow):
        batch = source()
        loss, _ = flow_step(cfg, model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        ema.update(model)
        flow_losses.append(float(loss.detach()))
    print(f"[flow] {args.steps_flow} steps | loss {flow_losses[0]:.4f} -> "
          f"{sum(flow_losses[-10:]) / max(1, len(flow_losses[-10:])):.4f} | "
          f"{time.perf_counter() - t_start:.0f}s elapsed")

    # teacher = EMA snapshot; student-prior = snapshot before Reflow
    teacher = build_model(cfg)
    ema.copy_to(teacher)
    teacher.eval()
    naive = build_model(cfg)
    naive.load_state_dict(copy.deepcopy(model.state_dict()))
    naive.eval()

    # ---------------- reflow ----------------
    reflow_losses: List[float] = []
    for step in range(args.steps_reflow):
        batch = source()
        loss, _ = reflow_step(cfg, model, batch, teacher_model=teacher, teacher_steps=args.nfe_teacher)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        reflow_losses.append(float(loss.detach()))
    print(f"[reflow] {args.steps_reflow} steps @ teacher NFE {args.nfe_teacher} | "
          f"loss {reflow_losses[0]:.4f} -> {sum(reflow_losses[-10:]) / max(1, len(reflow_losses[-10:])):.4f} "
          f"| {time.perf_counter() - t_start:.0f}s elapsed")

    # ---------------- evaluation on fixed batches ----------------
    _banner("comparison from identical noise and conditioning")
    mel = MelSpectrogram(cfg.audio)
    eval_batches = []
    eval_source = LatentFlowSource(model, corpus, cfg, TextTokenizer(mode=cfg.text.mode),
                                   batch_size=min(args.batch_size, args.eval_items), seed=1234)
    for _ in range(args.eval_items):
        eval_batches.append(eval_source())

    rows: List[Dict[str, float]] = []
    floors = [ae_floor_mel_l1(model, mel, b) for b in eval_batches]
    for i, batch in enumerate(eval_batches):
        x_ref, x0 = endpoint(teacher, batch, args.nfe_reference)
        x_t16, _ = endpoint(teacher, batch, args.nfe_teacher, x0=x0)
        x_naive2, _ = endpoint(naive, batch, 2, x0=x0)
        x_reflow2, _ = endpoint(model, batch, 2, x0=x0)
        target_wav = batch["wav"]
        target_latent = fold_time(batch["latent"], cfg.flow.compress)
        # audio decoded from the *reference* endpoint: comparing both samplers against it puts the
        # autoencoder's systematic error in common mode, so the difference is sampler error rather
        # than AE reconstruction error
        wav_ref = decode_endpoint(model, batch, x_ref)
        row = {
            "item": float(i),
            "mse_teacher16_vs_ref": float(F.mse_loss(x_t16, x_ref).item()),
            "mse_naive2_vs_ref": float(F.mse_loss(x_naive2, x_ref).item()),
            "mse_reflow2_vs_ref": float(F.mse_loss(x_reflow2, x_ref).item()),
            "mse_naive2_vs_data": float(F.mse_loss(x_naive2, target_latent).item()),
            "mse_reflow2_vs_data": float(F.mse_loss(x_reflow2, target_latent).item()),
            "mse_ref_vs_data": float(F.mse_loss(x_ref, target_latent).item()),
            "mel_naive2_vs_refaudio": mel_l1(mel, decode_endpoint(model, batch, x_naive2), wav_ref),
            "mel_reflow2_vs_refaudio": mel_l1(mel, decode_endpoint(model, batch, x_reflow2), wav_ref),
            "mel_naive2": mel_l1(mel, decode_endpoint(model, batch, x_naive2), target_wav),
            "mel_reflow2": mel_l1(mel, decode_endpoint(model, batch, x_reflow2), target_wav),
            "mel_reference": mel_l1(mel, wav_ref, target_wav),
            "mel_ae_floor": floors[i],
        }
        rows.append(row)
        print(f"  item {i}: MSE vs teacher@{args.nfe_reference}  "
              f"teacher@{args.nfe_teacher}={row['mse_teacher16_vs_ref']:.4f}  "
              f"naive2={row['mse_naive2_vs_ref']:.4f}  reflow2={row['mse_reflow2_vs_ref']:.4f} | "
              f"vs data {row['mse_naive2_vs_data']:.4f}/{row['mse_reflow2_vs_data']:.4f} | "
              f"mel vs ref-audio naive={row['mel_naive2_vs_refaudio']:.4f} "
              f"reflow={row['mel_reflow2_vs_refaudio']:.4f}")

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / max(1, len(rows))

    # ---------------- timing ----------------
    timing_batch = eval_batches[0]
    t0 = time.perf_counter()
    endpoint(teacher, timing_batch, args.nfe_reference)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    endpoint(model, timing_batch, 2)
    t_2 = time.perf_counter() - t0
    print(f"\n[timing] NFE {args.nfe_reference}: {t_ref*1000:.0f} ms | NFE 2: {t_2*1000:.0f} ms | "
          f"speed-up {t_ref/max(t_2, 1e-9):.1f}x (VF passes {args.nfe_reference/2:.0f}x fewer)")

    summary = {
        "config": args.config,
        "params_total": sum(p.numel() for p in model.parameters()),
        "corpus": {"utterances": len(corpus), "audio_seconds": seconds},
        "steps": {"ae": args.steps_ae, "flow": args.steps_flow, "reflow": args.steps_reflow},
        "nfe": {"teacher": args.nfe_teacher, "reference": args.nfe_reference},
        "flow_loss_first": flow_losses[0] if flow_losses else None,
        "flow_loss_last": sum(flow_losses[-10:]) / max(1, len(flow_losses[-10:])) if flow_losses else None,
        "reflow_loss_first": reflow_losses[0] if reflow_losses else None,
        "reflow_loss_last": sum(reflow_losses[-10:]) / max(1, len(reflow_losses[-10:])) if reflow_losses else None,
        "mean": {
            "mse_teacher16_vs_ref": mean("mse_teacher16_vs_ref"),
            "mse_naive2_vs_ref": mean("mse_naive2_vs_ref"),
            "mse_reflow2_vs_ref": mean("mse_reflow2_vs_ref"),
            "mse_naive2_vs_data": mean("mse_naive2_vs_data"),
            "mse_reflow2_vs_data": mean("mse_reflow2_vs_data"),
            "mse_ref_vs_data": mean("mse_ref_vs_data"),
            "mel_naive2": mean("mel_naive2"),
            "mel_reflow2": mean("mel_reflow2"),
            "mel_reference": mean("mel_reference"),
            "mel_ae_floor": mean("mel_ae_floor"),
            "mel_naive2_vs_refaudio": mean("mel_naive2_vs_refaudio"),
            "mel_reflow2_vs_refaudio": mean("mel_reflow2_vs_refaudio"),
        },
        "timing_ms": {"nfe_reference": t_ref * 1000, "nfe_2": t_2 * 1000},
        "items": rows,
        "seconds_total": time.perf_counter() - t_start,
    }
    # ---------------------------------------------------------------------------------
    # What this experiment can and cannot conclude.
    #
    # It validates the *sampler*: from identical noise and conditioning, does a distilled 2-step
    # student land where the high-NFE teacher lands?  That is the Reflow claim, and it is measured
    # in latent space (not masked by the autoencoder) and in audio space against the reference
    # decoded through the same autoencoder (so AE error is common-mode).
    #
    # It cannot say anything about *model* quality.  With 15 s of synthetic audio and a few hundred
    # steps the flow is nowhere near converged -- the NFE-32 endpoint itself sits far from the data
    # latent, so "closer to the data" measures model error, not sampler error, and reflow faithfully
    # reproduces a teacher that is still wrong.  Those numbers are reported as diagnostics.
    # ---------------------------------------------------------------------------------
    checks = {
        "reflow_beats_naive_vs_teacher": mean("mse_reflow2_vs_ref") < mean("mse_naive2_vs_ref"),
        "reflow_matches_teacher_discretization": mean("mse_reflow2_vs_ref")
        <= mean("mse_teacher16_vs_ref"),
        "reflow_beats_naive_in_audio_space": mean("mel_reflow2_vs_refaudio")
        < mean("mel_naive2_vs_refaudio"),
    }
    summary["checks"] = checks
    summary["diagnostics"] = {
        "note": (
            "model-quality limited, not sampler limited: the reference NFE-32 endpoint is far from "
            "the data latent at this training budget, and mel-vs-target is dominated by the AE floor"
        ),
        "mse_ref_vs_data": mean("mse_ref_vs_data"),
        "mse_naive2_vs_data": mean("mse_naive2_vs_data"),
        "mse_reflow2_vs_data": mean("mse_reflow2_vs_data"),
        "mel_ae_floor": mean("mel_ae_floor"),
        "audio_dynamic_range_ratio": mean("mel_ae_floor")
        / max(abs(mean("mel_reflow2") - mean("mel_naive2")), 1e-9),
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _banner("RESULT")
    print(f"  latent MSE vs teacher@{args.nfe_reference}:  "
          f"teacher@{args.nfe_teacher} {mean('mse_teacher16_vs_ref'):.4f} | "
          f"naive@2 {mean('mse_naive2_vs_ref'):.4f} | reflow@2 {mean('mse_reflow2_vs_ref'):.4f}")
    print(f"  audio log-mel L1 vs decoded reference: naive@2 {mean('mel_naive2_vs_refaudio'):.4f} | "
          f"reflow@2 {mean('mel_reflow2_vs_refaudio'):.4f}")
    print(f"  [diagnostics, model-limited] latent MSE vs data: reference "
          f"{mean('mse_ref_vs_data'):.4f} | naive@2 {mean('mse_naive2_vs_data'):.4f} | reflow@2 "
          f"{mean('mse_reflow2_vs_data'):.4f}")
    print(f"  [diagnostics] mel vs target: naive {mean('mel_naive2'):.4f} | reflow "
          f"{mean('mel_reflow2'):.4f} | AE floor {mean('mel_ae_floor'):.4f} -> the AE floor is "
          f"{summary['diagnostics']['audio_dynamic_range_ratio']:.0f}x the sampler difference")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nreport -> {out/'report.json'}")
    print("REFLOW DEMO " + ("PASSED" if all(checks.values()) else "FAILED"))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
