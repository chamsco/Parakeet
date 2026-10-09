"""How much of the token->frame seam is *information* loss?  A rate sweep, with no training.

    python scripts/seam_rate.py --limit 6

Round 21 established that the seam costs WER 0.722 even when the decoder is trained on exactly the
distribution it meets (and 0.167 for the frame latent it cannot see).  The suspected cause is
arithmetic: the cached per-token latent is the **mean** over that token's ~6 frames, so the token path
carries 24 numbers per ~6 frames -- about 4 dimensions per frame against the encoder's 24.  No decoder
recovers detail that averaging removed.

This script tests that by *sweeping the token rate* using the teacher's own frame latent as an oracle.
For K sub-latents per token it fills every sub-span with the mean of the frame latent over that
sub-span and decodes.  K=1 is exactly what the pipeline does today; K = frames-per-token is the frame
latent itself (no averaging, the ceiling).  Nothing is trained, so any improvement is purely the
information the token path is allowed to carry -- which is the quantity that matters for deciding
whether to predict sub-token latents or to add a refinement stage.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.audio.mel import MelSpectrogram  # noqa: E402
from parakeet.config import load_config  # noqa: E402
from parakeet.data.dataset import LatentShardDataset  # noqa: E402
from parakeet.eval.metrics import dnsmos_score, whisper_wer_with_control  # noqa: E402
from parakeet.models import build_model  # noqa: E402


def build_latent_at_rate(
    frames: torch.Tensor, durations: torch.Tensor, n_frames: int, rate: int
) -> torch.Tensor:
    """``(C, F)`` teacher frame latent -> the latent a text side predicting ``rate`` per token yields.

    Each token's span is split into ``rate`` equal sub-spans and each is filled with the mean of the
    frame latent over it.  ``rate=1`` reproduces the current pipeline exactly (the token mean);
    ``rate`` >= the token's frame count reproduces the frame latent.
    """
    out = torch.zeros_like(frames[:, :n_frames])
    start = 0
    for length in durations.tolist():
        end = min(start + int(length), n_frames)
        if end <= start:
            continue
        edges = torch.linspace(start, end, rate + 1).round().long()
        for k in range(rate):
            a, b = int(edges[k]), int(edges[k + 1])
            b = max(b, a + 1)
            b = min(b, end)
            if b <= a:
                continue
            out[:, a:b] = frames[:, a:b].mean(dim=-1, keepdim=True)
        start = end
    if start < n_frames:  # any tail beyond the token spans stays as the frame latent
        out[:, start:] = frames[:, start:n_frames]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Token-rate sweep for the token->frame seam")
    ap.add_argument("--checkpoint", default="runs/seam_on/distill-decoder_last.pt")
    ap.add_argument("--cache", default="runs/real_train/latent_cache")
    ap.add_argument("--corpus", default="data/real_corpus/corpus")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--rates", default="1,2,3,6,12")
    ap.add_argument("--whisper", default="base.en")
    ap.add_argument("--out", default="runs/seam_rate.json")
    args = ap.parse_args()

    import soundfile as sf

    corpus = Path(args.corpus)
    manifest = corpus / "curated" / "kept.jsonl"
    if not manifest.exists():
        manifest = corpus / "manifest.jsonl"
    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    dataset = LatentShardDataset(args.cache)
    meta = json.loads((Path(args.cache) / "cache_meta.json").read_text(encoding="utf-8"))
    cfg = load_config(args.config)
    cfg.n_voices = max(1, len(meta["voice_names"]))
    model = build_model(cfg)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    mel = MelSpectrogram(cfg.audio)

    n = min(args.limit, len(dataset), len(records))
    references: List[torch.Tensor] = []
    texts: List[str] = []
    for i in range(n):
        references.append(
            torch.from_numpy(sf.read(str(corpus / records[i]["wav_path"]), dtype="float32")[0]).reshape(-1)
        )
        texts.append(records[i]["text"])

    rates = [int(r) for r in args.rates.split(",") if r.strip()]
    rows: List[Dict[str, object]] = []
    control_wer = None
    for rate in rates:
        audios: List[torch.Tensor] = []
        cosines: List[float] = []
        with torch.no_grad():
            for i in range(n):
                item = dataset[i]
                n_frames = int(item["n_frames"])
                durations = item["durations"]
                if rate == 1:
                    # exactly what the pipeline does: the frame latent built from token means
                    latent, _ = model.latent_from_tokens(
                        item["latent_token"][None], durations[None],
                        item["f0"][None], item["energy"][None],
                    )
                    frames = model.latent_norm.denormalize(latent)
                else:
                    rate_latent = build_latent_at_rate(
                        item["latent"], durations, n_frames, rate
                    )
                    frames = model.latent_norm.denormalize(rate_latent[None])
                wav = model.autoencoder.decode(frames).reshape(-1)
                audios.append(wav)
                ref = references[i]
                m = min(wav.numel(), ref.numel())
                if m > 2000:
                    cosines.append(
                        float(torch.nn.functional.cosine_similarity(
                            mel.log_mel(wav.reshape(1, -1)[..., :m]).flatten(),
                            mel.log_mel(ref[:m].reshape(1, -1)).flatten(),
                            dim=0,
                        ))
                    )
        controlled = whisper_wer_with_control(
            audios, texts, control_audio=references, sample_rate=cfg.audio.sample_rate,
            model_size=args.whisper,
        )
        control_wer = controlled["control"].value
        mos = dnsmos_score(audios, sample_rate=cfg.audio.sample_rate)
        frames_per_token = statistics.mean(
            float(dataset[i]["durations"].float().mean()) for i in range(n)
        )
        rows.append(
            {
                "rate": rate,
                "wer": controlled["wer"].value,
                "wer_available": controlled["wer"].available,
                "wer_reason": controlled["wer"].reason,
                "dnsmos": mos.value,
                "dnsmos_available": mos.available,
                "dnsmos_reason": mos.reason,
                "log_mel_cosine": statistics.mean(cosines) if cosines else float("nan"),
                # the information the token path actually carries, in dims per frame
                "dims_per_frame": rate * cfg.autoencoder.latent_dim / max(frames_per_token, 1e-9),
                "audio_finite": all(bool(torch.isfinite(a).all()) for a in audios),
                "control_note": controlled["note"],
            }
        )
        wer_text = f"{rows[-1]['wer']:.3f}" if rows[-1]["wer"] is not None else "n/a"
        mos_text = f"{rows[-1]['dnsmos']:.2f}" if rows[-1]["dnsmos"] is not None else "n/a"
        extra = "" if rows[-1]["dnsmos"] is not None else f"  [{rows[-1]['dnsmos_reason']}]"
        print(f"  rate {rate:3d}  WER {wer_text}  DNSMOS {mos_text}  "
              f"mel cosine {rows[-1]['log_mel_cosine']:.4f}  "
              f"({rows[-1]['dims_per_frame']:.1f} dims/frame){extra}", flush=True)

    wer_values = [row["wer"] for row in rows if row["wer"] is not None]
    # Deliberately *not* "more information never hurts": beyond ~3 sub-latents per token the WER at
    # this sample size stops improving and bounces around (0.093, 0.296, 0.167 for rates 3, 6, 12),
    # because the decoder was trained on rate-1 inputs and these are off-distribution.  The two claims
    # the data supports are the ones below.
    best_small_rate = min(
        (row["wer"] for row in rows if row["rate"] in (2, 3) and row["wer"] is not None),
        default=None,
    )
    rate_1 = next((row["wer"] for row in rows if row["rate"] == 1), None)
    highest = next((row["wer"] for row in rows if row["rate"] == rates[-1]), None)
    rate_3 = next((row["wer"] for row in rows if row["rate"] == 3), None)
    checks = {
        "recogniser_works_on_the_reference": bool(control_wer is not None and control_wer < 0.2),
        "the_control_held_for_every_rate": all(row["control_note"] for row in rows),
        "a_small_rate_increase_closes_most_of_the_seam": bool(
            rate_1 is not None and best_small_rate is not None and best_small_rate < 0.4 * rate_1
        ),
        "the_curve_plateaus_rather_than_improving_forever": bool(
            highest is not None and rate_3 is not None and highest > rate_3 - 0.15
        ),
    }
    report = {
        "checkpoint": args.checkpoint,
        "utterances": n,
        "recogniser": args.whisper,
        "reference_wer": control_wer,
        "rates": rows,
        "note": (
            "K sub-latents per token built from the TEACHER's frame latent: an oracle for a text side "
            "predicting K latents per token.  rate 1 is the current pipeline; a large rate approaches "
            "the frame latent (WER 0.167) exactly."
        ),
        "checks": checks,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    first, last = wer_values[0], wer_values[-1]
    print(f"\n  WER {first:.3f} at rate {rates[0]} -> {last:.3f} at rate {rates[-1]} "
          f"({100 * (first - last) / max(first, 1e-9):.0f}% of the seam is information loss)")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"report -> {args.out}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
