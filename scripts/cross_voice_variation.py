"""Is rho >= 0.75 even attainable?  Measure how much a latent varies across *renditions* of one text.

The text side plateaus at 0.387 train / 0.251 validation against the target of 0.75 measured from the
correlation-threshold experiment.  Before blaming capacity, the target itself has to be checked: text does
not determine acoustics uniquely -- pitch, timing and timbre depend on the speaker and the performance --
so a text-only predictor may be chasing a one-to-many mapping whose ceiling is well below 0.75.

The operator's eight takes are ideal material: the *same paragraph* read by eight Speechify voices,
already segmented and curated.  This encodes them with the trained autoencoder and, for pairs of voices
carrying the same text, measures the per-dimension correlation between their latents.  That correlation
is the acoustic variance the text cannot predict, and it is an upper bound on what any text-only model
can reach for that text.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, ".")

import soundfile as sf
import torch

from parakeet.audio.mel import MelSpectrogram
from parakeet.config import load_config
from parakeet.models import build_model
from parakeet.train.common import derive_n_voices_from_cache, load_latent_norm_from_cache

CORPUS = Path("data/downloads_corpus/corpus")


def normalise_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def per_dim_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-dimension correlation between two (C, T) latents over their shared frames."""
    frames = min(a.shape[-1], b.shape[-1])
    a, b = a[:, :frames], b[:, :frames]
    correlations = []
    for dim in range(a.shape[0]):
        x, y = a[dim], b[dim]
        if float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
            continue
        correlations.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
    return sum(correlations) / max(1, len(correlations))


def main() -> int:
    cfg = load_config("configs/parakeet_tiny.yaml")
    cfg.n_voices = derive_n_voices_from_cache("runs/mixed_v2/latent_cache")
    payload = torch.load("runs/ae_scaled/adversarial/autoencoder_last.pt", map_location="cpu",
                         weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    model = build_model(cfg)
    current = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items()
                           if k in current and tuple(current[k].shape) == tuple(v.shape)},
                          strict=False)
    load_latent_norm_from_cache(model, "runs/mixed_v2/latent_cache")
    model.eval()
    mel = MelSpectrogram(cfg.audio)

    rows = [json.loads(line) for line in (CORPUS / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        groups.setdefault(normalise_text(row["text"]), []).append(row)
    matched = {text: items for text, items in groups.items() if len(items) >= 2}
    print(f"{len(rows)} segments | {len(groups)} distinct texts | {len(matched)} read by 2+ voices")

    latents: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for row in rows:
            key = f"{row['voice']}|{normalise_text(row['text'])[:40]}"
            if key in latents:
                continue
            wav, rate = sf.read(str(CORPUS / row["wav_path"]), dtype="float32")
            wav_t = torch.from_numpy(wav).reshape(1, -1)
            if rate != cfg.audio.sample_rate:
                wav_t = torch.nn.functional.interpolate(
                    wav_t[:, None, :], size=int(wav_t.shape[-1] * cfg.audio.sample_rate / rate),
                    mode="linear", align_corners=False,
                )[:, 0, :]
            latents[key] = model.autoencoder.encode(mel.log_mel(wav_t))[0]  # (C, T) raw

    cross_voice, within_voice = [], []
    for text, items in matched.items():
        keys = [f"{r['voice']}|{text[:40]}" for r in items]
        keys = [k for k in keys if k in latents]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                voice_i = keys[i].split("|")[0]
                voice_j = keys[j].split("|")[0]
                correlate = per_dim_correlation(latents[keys[i]], latents[keys[j]])
                (cross_voice if voice_i != voice_j else within_voice).append(correlate)

    # control: *different* texts in the same voice (should be near zero if the encoder tracks content)
    different_text = []
    keys = list(latents)
    for i in range(0, min(len(keys), 40), 2):
        for j in range(1, min(len(keys), 40), 2):
            if keys[i].split("|")[0] == keys[j].split("|")[0] and keys[i] != keys[j]:
                different_text.append(per_dim_correlation(latents[keys[i]], latents[keys[j]]))
    different_text = different_text[:60]

    def mean(values):
        return sum(values) / max(1, len(values))

    print(f"\n  same text, DIFFERENT voice: {mean(cross_voice):+.3f} "
          f"(n={len(cross_voice)}, range {min(cross_voice) if cross_voice else 0:.2f}.."
          f"{max(cross_voice) if cross_voice else 0:.2f})")
    print(f"  different text, same voice: {mean(different_text):+.3f} (n={len(different_text)})")
    print(f"\n  the target for intelligibility measured elsewhere: rho >= 0.75")

    report = {
        "segments": len(rows),
        "distinct_texts": len(groups),
        "texts_with_multiple_voices": len(matched),
        "cross_voice_correlation": mean(cross_voice),
        "cross_voice_n": len(cross_voice),
        "different_text_correlation": mean(different_text),
        "target": 0.75,
        "reading": (
            "the correlation between two voices reading the SAME text is the part of the latent that "
            "text cannot determine, so a text-only predictor cannot exceed it for that text"
        ),
    }
    Path("runs/cross_voice_variation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("report -> runs/cross_voice_variation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
