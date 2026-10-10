"""Recover a cache's latent-normaliser statistics and write them into its metadata.

The cache stores **normalised** latents: `real_train_demo.py` fits a `LatentNormalizer` on the corpus and
`cache_teacher_corpus` applies `normalize`.  Every decode site is supposed to undo that --
`decoder_latent_from_tokens` does call `denormalize`, and `ParakeetFlow.synthesize` calls it too -- but
the fitted statistics were **never persisted**: the normaliser that builds the cache lives for one
process, and anything loading a checkpoint later gets `mean 0 / var 1`, which makes `denormalize` the
identity.  The decoder therefore receives normalised latents.

Measured on the cached latent of one utterance (round 36):

    encoder(teacher audio)   -> decode  WER 0.000   speech-like
    cached latent            -> decode  WER 1.000   speech-like by the gate, wrong words
    recovered inverse transform -> decode  WER 0.000

with per-dimension correlation(raw, cached) = 1.000 and a **zero residual** after the affine map, so the
transform is exactly `raw = cached * sqrt(var + eps) + mean`.

This recovers those statistics from the data itself (encode the corpus with the trained autoencoder and
take per-dimension mean/variance, which is what `fit_latent_normalizer` computed) and merges them into
`cache_meta.json`, so every consumer can load them instead of silently using the identity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def recover(
    autoencoder, corpus: Path, manifest: Path, cfg, limit: int = 32, device: str = "cpu"
) -> Optional[dict]:
    import soundfile as sf

    from parakeet.audio.mel import MelSpectrogram

    mel = MelSpectrogram(cfg.audio).to(device)
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:limit]
    latents = []
    with torch.no_grad():
        for row in rows:
            wav, rate = sf.read(str(corpus / row["wav_path"]), dtype="float32")
            wav_t = torch.from_numpy(wav).reshape(1, -1).to(device)
            if rate != cfg.audio.sample_rate:
                wav_t = torch.nn.functional.interpolate(
                    wav_t[:, None, :],
                    size=int(wav_t.shape[-1] * cfg.audio.sample_rate / rate),
                    mode="linear", align_corners=False,
                )[:, 0, :]
            latents.append(autoencoder.encode(mel.log_mel(wav_t)).cpu())
    if not latents:
        return None
    stacked = torch.cat(latents, dim=-1)
    return {
        "mean": stacked.mean(dim=-1)[0].tolist(),
        "var": stacked.var(dim=-1, unbiased=False)[0].tolist(),
        "samples": len(latents),
        "source": str(manifest),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Persist a cache's latent-normaliser statistics")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--corpus", required=True, help="corpus directory the cache was built from")
    ap.add_argument("--manifest", default="curated/kept.jsonl")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--autoencoder", default="runs/ae_scaled/adversarial/autoencoder_last.pt")
    ap.add_argument("--limit", type=int, default=32)
    args = ap.parse_args()

    from parakeet.config import load_config
    from parakeet.models import build_model
    from parakeet.train.common import derive_n_voices_from_cache

    cfg = load_config(args.config)
    voices = derive_n_voices_from_cache(args.cache)
    if voices:
        cfg.n_voices = voices
    payload = torch.load(args.autoencoder, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    model = build_model(cfg)
    current = model.state_dict()
    usable = {k: v for k, v in state.items()
              if k in current and tuple(current[k].shape) == tuple(v.shape)}
    model.load_state_dict(usable, strict=False)
    model.eval()

    corpus = Path(args.corpus)
    manifest = corpus / args.manifest
    if not manifest.exists():
        manifest = corpus / "manifest.jsonl"
    stats = recover(model.autoencoder, corpus, manifest, cfg, limit=args.limit)
    if stats is None:
        print("could not recover statistics (no records)")
        return 2

    meta_path = Path(args.cache) / "cache_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["latent_norm"] = stats
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    mean = torch.tensor(stats["mean"])
    var = torch.tensor(stats["var"])
    print(f"wrote latent normaliser statistics to {meta_path}")
    print(f"  mean: {float(mean.mean()):+.4f} (per-dim {float(mean.min()):+.3f}..{float(mean.max()):+.3f})")
    print(f"  var : {float(var.mean()):.4f} (per-dim {float(var.min()):.4f}..{float(var.max()):.4f})")
    print(f"  from {stats['samples']} utterances of {stats['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
