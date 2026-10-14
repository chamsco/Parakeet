"""Build a cache whose frame targets are the RENDITION-INDEPENDENT part of the signal.

Round 136 validated the idea on 110 pairs: average the two waveforms of a same-voice, same-words pair, take the
log-mel, re-encode through the frozen autoencoder, and the result decodes at WER **0.368** where this project's
models score **1.000**.  The target is real; this makes it available for training.

For each variant record (which carries `variant_of`, pointing at a base-corpus record) this writes one cache item
holding that pair's shared latent, keeping the ids, voice, durations, f0 and energy from the original so the item is
trainable exactly as before.  The token-level target is rebuilt from the SHARED frame latent by averaging over each
token's span, so both targets describe the same signal -- mixing a shared frame target with an original token target
would mean the two halves of the loss disagree about what is being predicted.

Validation path, and it is the reason the geometry here is worth trusting: run it on the 110-pair rendition corpus
with `--pairs-offset`, and the WER it implies must match the 0.368 measured directly.

    python scripts/make_shared_target_cache.py --base runs/mixed_v2/latent_cache \
        --variants data/rendition_base_v1/manifest.jsonl --out runs/shared_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


import soundfile as sf
import torch

from parakeet.audio.mel import MelSpectrogram
from parakeet.config import load_config
from parakeet.data.dataset import LatentShardDataset
from parakeet.models import build_model
from parakeet.train.common import apply_checkpoint_geometry, load_latent_norm_from_cache


def span_average(frames: torch.Tensor, durations: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Average a (C, T) frame series over each token's frame span -> (n_tokens, C)."""
    out = torch.zeros(n_tokens, frames.shape[0], dtype=frames.dtype)
    position = 0
    for token in range(n_tokens):
        span = int(durations[token]) if token < len(durations) else 0
        span = max(1, span)
        end = min(position + span, frames.shape[1])
        if position >= frames.shape[1]:
            break
        out[token] = frames[:, position:end].mean(dim=-1)
        position = end
    if position < frames.shape[1] and n_tokens:
        out[n_tokens - 1] = frames[:, position:].mean(dim=-1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache of rendition-independent (shared) frame targets")
    ap.add_argument("--base", required=True, help="the corpus cache whose items the variants correspond to")
    ap.add_argument("--base-manifest", default="data/mixed_corpus/train_v2/manifest.jsonl",
                    help="the manifest the base cache was written from, in order")
    ap.add_argument("--variants", default="", help="variant manifest carrying variant_of -> base utt_id")
    ap.add_argument("--pairs-offset", type=int, default=0,
                    help="instead of variant_of, pair item j with item j+offset (the 110-pair layout)")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--checkpoint", default="runs/ae_scaled/adversarial/autoencoder_last.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--originals-only", action="store_true",
                    help="write the SAME items and order but keep each item's ORIGINAL targets. This is the "
                         "control arm of the matched experiment: identical items, so the target is the only "
                         "variable between the two caches.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    apply_checkpoint_geometry(cfg, state)
    model = build_model(cfg)
    current = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items()
                           if k in current and tuple(current[k].shape) == tuple(v.shape)}, strict=False)
    load_latent_norm_from_cache(model, args.base)
    model.eval()
    mel = MelSpectrogram(cfg.audio).eval()

    dataset = LatentShardDataset(args.base, apply_voice_norm=False)
    base_manifest = [json.loads(line) for line in Path(args.base_manifest).read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    if len(base_manifest) != len(dataset):
        raise SystemExit(f"base manifest has {len(base_manifest)} records for {len(dataset)} cache items: "
                         "the item order cannot be trusted, so pairs cannot be formed safely")
    index_of = {str(record.get("utt_id")): i for i, record in enumerate(base_manifest)}

    def encode_shared(wav_a: torch.Tensor, wav_b: torch.Tensor) -> torch.Tensor:
        length = min(wav_a.shape[-1], wav_b.shape[-1])
        waveform = (wav_a[..., :length] + wav_b[..., :length]) / 2.0
        with torch.no_grad():
            return model.autoencoder.encode(mel.log_mel(waveform[None]))[0]

    pairs: list[tuple[int, dict | None]] = []
    if args.pairs_offset:
        for j in range(len(dataset) - args.pairs_offset):
            pairs.append((j, {"__positional__": j + args.pairs_offset}))
    else:
        for variant in [json.loads(line) for line in
                        Path(args.variants).read_text(encoding="utf-8").splitlines() if line.strip()]:
            origin = index_of.get(str(variant.get("variant_of")))
            if origin is None:
                continue
            # the variant RECORD travels with the pair.  Carrying only an index into the variant list desynced
            # whenever a variant was skipped, silently pairing an original with somebody else's variant.
            pairs.append((origin, variant))
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        raise SystemExit("no pairs formed")

    corpus_root = Path(args.variants).parent if args.variants else None
    items = []
    for origin_index, variant in pairs:
        origin = dataset[origin_index]
        if variant is not None and "__positional__" in variant:
            variant_wav = dataset[int(variant["__positional__"])]["wav"]
        else:
            path = Path(str(variant.get("wav_path") or ""))
            if not path.is_absolute():
                path = corpus_root / path
            audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
            variant_wav = torch.from_numpy(audio.mean(axis=1))
            if rate != int(cfg.audio.sample_rate):
                import torch.nn.functional as F
                variant_wav = F.interpolate(variant_wav[None, None],
                                            size=int(round(variant_wav.shape[0] * cfg.audio.sample_rate / rate)),
                                            mode="linear", align_corners=False)[0, 0]

        shared = encode_shared(origin["wav"], variant_wav)
        durations = origin["durations"]
        n_tokens = int(durations.shape[0])
        record = dict(origin)
        if args.originals_only:
            # control arm: the same item, untouched.  Pairing, order and metadata are identical to the shared
            # cache by construction, so any difference between arms is the target and nothing else.
            record["source_index"] = origin_index
            items.append(record)
            if len(items) % 100 == 0:
                print(f"  {len(items)}/{len(pairs)} built (originals)", flush=True)
            continue
        record["latent"] = shared.contiguous()
        record["latent_token"] = span_average(shared, durations, n_tokens).contiguous()
        record["n_frames"] = shared.shape[-1]
        record["log_mel"] = mel.log_mel(origin["wav"][None])[0][..., :shared.shape[-1]].contiguous()
        record["shared_of"] = origin_index
        items.append(record)
        if len(items) % 100 == 0:
            print(f"  {len(items)}/{len(pairs)} built", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"items": items}, out / "shard_00000.pt")
    (out / "index.json").write_text(json.dumps([{"path": "shard_00000.pt", "n": len(items)}]), "utf-8")
    meta_src = json.loads((Path(args.base) / "cache_meta.json").read_text(encoding="utf-8"))
    meta_src["shared_target_of"] = str(args.base)
    meta_src["items"] = len(items)
    meta_src["note"] = "frame targets are the rendition-independent part: encode(log-mel(mean(two same-voice " \
                       "renditions of the same words))). Validated at WER 0.368 against 1.000 for the originals."
    (out / "cache_meta.json").write_text(json.dumps(meta_src, indent=2), encoding="utf-8")
    print(f"\nwrote {len(items)} shared-target items -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
