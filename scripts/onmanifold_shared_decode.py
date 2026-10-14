"""The ON-MANIFOLD shared component: encode the mean WAVEFORM, not the mean latent.

Round 135 measured the predictable part of a rendition pair by averaging the two latents, and found it decodes
worse than either rendition (WER 0.512 against 0.209) because the mean of two latents is not a latent -- it sits off
the autoencoder's manifold.

The fix is one step upstream: average the WAVEFORMS, take the mel, and let the autoencoder's own ENCODER produce the
shared latent. Then the target is by construction a point the encoder can produce and the decoder can render, and
the question becomes whether the rendition-independent part of the signal survives that route.

Self-check, and it is the important one: encoding rendition A's waveform must reproduce the CACHED latent for item A
(a high correlation). If it does not, the encode path here does not match the convention the cache and the model
were trained in, and every number below is meaningless. That check is the round-122 lesson applied in advance.

    python scripts/onmanifold_shared_decode.py --cache runs/rendition_v1_cache --pairs 110 --limit 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import torch

from parakeet.audio.mel import MelSpectrogram
from parakeet.config import load_config
from parakeet.data.dataset import LatentShardDataset
from parakeet.eval.metrics import whisper_wer_with_control
from parakeet.models import build_model
from parakeet.train.common import apply_checkpoint_geometry, load_latent_norm_from_cache


def per_dim_rho(a: torch.Tensor, b: torch.Tensor) -> float:
    length = min(a.shape[-1], b.shape[-1])
    scores = []
    for dim in range(a.shape[0]):
        x, y = a[dim, :length].float(), b[dim, :length].float()
        if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
            continue
        scores.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
    return sum(scores) / max(1, len(scores))


def value_of(metric):
    if metric is None:
        return None
    if isinstance(metric, (int, float)):
        return float(metric)
    return getattr(metric, "value", None)


def main() -> int:
    ap = argparse.ArgumentParser(description="Decode the on-manifold shared component")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pairs", type=int, default=110)
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--checkpoint", default="runs/ae_scaled/adversarial/autoencoder_last.pt")
    ap.add_argument("--manifest", default="data/rendition_v1/manifest.jsonl")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--out", default="runs/onmanifold_shared_decode.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    apply_checkpoint_geometry(cfg, state)
    model = build_model(cfg)
    current = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items()
                           if k in current and tuple(current[k].shape) == tuple(v.shape)}, strict=False)
    load_latent_norm_from_cache(model, args.cache)
    model.eval()

    dataset = LatentShardDataset(args.cache)
    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    corpus_root = Path(args.manifest).parent
    mel = MelSpectrogram(cfg.audio).eval()
    sample_rate = int(getattr(cfg.audio, "sample_rate", 24000))

    def load_wav(record) -> torch.Tensor:
        path = Path(str(record.get("wav_path") or ""))
        if not path.is_absolute():
            path = corpus_root / path
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.mean(axis=1))
        if rate != sample_rate:
            import torch.nn.functional as F
            waveform = F.interpolate(waveform[None, None],
                                     size=int(round(waveform.shape[0] * sample_rate / rate)),
                                     mode="linear", align_corners=False)[0, 0]
        return waveform

    def encode(waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.autoencoder.encode(mel.log_mel(waveform[None]))[0]

    def decode(latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.autoencoder.decode(latent[None]).reshape(-1).float()

    count = min(args.limit, args.pairs)
    consistency, shared_audio, texts = [], [], []
    for index in range(count):
        record_a = manifest[index]
        record_b = manifest[index + args.pairs]
        texts.append(str(record_a.get("text", "")))
        wav_a, wav_b = load_wav(record_a), load_wav(record_b)
        length = min(wav_a.shape[0], wav_b.shape[0])

        # The cache stores latents in the NORMALISED convention -- per-voice statistics on top of the model's own
        # latent_norm -- and the decoder expects that convention. So an encoded latent has to be put through the
        # same per-voice transform before it can be compared with a cached one or decoded. Without this the
        # self-check reads 0.32 and the guard stops the run, which is what happened first.
        voice = int(dataset[index]["voice"])
        stats = dataset.voice_norm.get(voice)
        normalise = (lambda z: (z - stats["mean"]) / stats["std"]) if stats is not None else (lambda z: z)

        latent_a = normalise(encode(wav_a[:length]))
        shared_audio.append(decode(normalise(encode((wav_a[:length] + wav_b[:length]) / 2.0))))
        consistency.append(per_dim_rho(latent_a, dataset[index]["latent"][..., :latent_a.shape[-1]]))

    control = [load_wav(record) for record in manifest[:count]]
    report: dict = {"items": count, "encode_matches_cache_rho": sum(consistency) / len(consistency)}
    print(f"self-check: encoding rendition A reproduces the cached latent at rho "
          f"{report['encode_matches_cache_rho']:+.4f}")
    if report["encode_matches_cache_rho"] < 0.90:
        raise SystemExit("  the encode path does NOT match the cache convention -- numbers below would be "
                         "meaningless, so stopping")

    scored = whisper_wer_with_control(shared_audio, texts, control_audio=control,
                                     sample_rate=sample_rate, model_size="base.en")
    report["shared_mean_waveform"] = {"wer": value_of(scored.get("wer")),
                                      "control": value_of(scored.get("control"))}
    print(f"  on-manifold shared  WER {report['shared_mean_waveform']['wer']} "
          f"(teacher control {report['shared_mean_waveform']['control']})")
    print(f"  compare: latent-average 0.512 | rendition A 0.209 | rendition B 0.395 | models here WER 1.000")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
