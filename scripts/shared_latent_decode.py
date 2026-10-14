"""Does the PREDICTABLE part of a rendition still decode to intelligible speech?

The reliability measurement (round 134) says two renditions of the same words agree at rho 0.714 in this latent
space, so the shared component -- their mean -- is the part any text-conditioned model can aim at, and the only part
it could ever predict. Its square root, 0.845, is the ceiling for such a model.

That ceiling is a correlation, not a verdict on the audio. This asks the question directly:

    decode(latent_A)              -> is one rendition intelligible? (the control)
    decode(latent_B)              -> the other rendition
    decode(mean(A, B))            -> the PREDICTABLE component, i.e. the target a model would aim at

If the mean decodes as intelligibly as the renditions, then the text->latent task has an intelligible target and the
ceiling is not the wall. If it collapses, then averaging destroys the signal -- the mean of two latents need not be a
latent -- and the "train on the shared component" idea fails immediately, for a reason worth knowing before
spending CPU on it.

All three are decoded through the same frozen autoencoder, in the same latent convention, and scored by the same
recogniser with the rendition itself as the control.

    python scripts/shared_latent_decode.py --cache runs/rendition_v1_cache --pairs 110 --limit 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


import torch
import soundfile as sf

from parakeet.config import load_config
from parakeet.data.dataset import LatentShardDataset
from parakeet.eval.metrics import whisper_wer_with_control
from parakeet.models import build_model
from parakeet.train.common import apply_checkpoint_geometry


def main() -> int:
    ap = argparse.ArgumentParser(description="Decode the predictable (shared) component of a rendition pair")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pairs", type=int, default=110, help="the second block starts at this index")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml",
                    help="the AE checkpoint records its own geometry and apply_checkpoint_geometry reads it from "
                         "the state dict, so this only needs to be the right VARIANT")
    ap.add_argument("--checkpoint", default="runs/ae_scaled/adversarial/autoencoder_last.pt")
    ap.add_argument("--manifest", default="data/rendition_v1/manifest.jsonl",
                    help="cache items carry 'wav' but not 'text', so the text comes from the corpus manifest, in "
                         "the same order the cache was written")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out", default="runs/shared_latent_decode.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    apply_checkpoint_geometry(cfg, state)
    model = build_model(cfg)
    current = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items()
                           if k in current and tuple(current[k].shape) == tuple(v.shape)}, strict=False)
    model.eval()

    dataset = LatentShardDataset(args.cache)
    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if len(manifest) < len(dataset):
        raise SystemExit(f"manifest has {len(manifest)} records for {len(dataset)} cache items: cannot align texts")
    texts = [str(record.get("text", "")) for record in manifest[:args.limit]]
    if not all(texts):
        raise SystemExit("manifest records carry no 'text'")

    def decode(latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.autoencoder.decode(latent[None]).reshape(-1).float()

    first, second, shared = [], [], []
    for index in range(min(args.limit, args.pairs)):
        a = dataset[index]["latent"]
        b = dataset[index + args.pairs]["latent"]
        length = min(a.shape[-1], b.shape[-1])
        first.append(decode(a[..., :length]))
        second.append(decode(b[..., :length]))
        shared.append(decode(((a[..., :length] + b[..., :length]) / 2.0)))

    sample_rate = int(getattr(cfg.audio, "sample_rate", 24000))

    #: A REAL control: the teacher's own audio for the same text, straight from the manifest.  Passing each clip as
    #: its own control -- which the first version of this script did -- makes the control gate vacuous: it compares
    #: the clip with itself, so the recogniser reports "failed its own control" whenever a clip is poor and HIDES
    #: the number instead of measuring it.
    control_audio = []
    corpus_root = Path(args.manifest).parent
    for record in manifest[:len(first)]:
        # the manifest records 'wav_path' RELATIVE to the corpus directory
        path = Path(str(record.get("wav_path") or record.get("wav") or ""))
        if not path.is_absolute():
            path = corpus_root / path
        if not path.exists():
            raise SystemExit(f"teacher wav missing for the control: {path}")
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.mean(axis=1))
        if rate != sample_rate:
            import torch.nn.functional as F
            ratio = sample_rate / rate
            target = int(round(waveform.shape[0] * ratio))
            waveform = F.interpolate(waveform[None, None], size=target, mode="linear",
                                     align_corners=False)[0, 0]
        control_audio.append(waveform.float())

    def value_of(metric):
        """whisper_wer_with_control returns {'wer', 'control'} where each may be a bare float or an OptionalMetric."""
        if metric is None:
            return None
        if isinstance(metric, (int, float)):
            return float(metric)
        return getattr(metric, "value", None)

    report: dict = {"items": len(first), "sample_rate": sample_rate}
    for name, audios in (("rendition_A", first), ("rendition_B", second), ("shared_mean", shared)):
        scored = whisper_wer_with_control(audios, texts, control_audio=control_audio,
                                         sample_rate=sample_rate, model_size="base.en")
        report[name] = {"wer": value_of(scored.get("wer")), "control": value_of(scored.get("control")),
                        "reason": getattr(scored.get("wer"), "reason", None)}
        print(f"  {name:14s} WER {report[name]['wer']} (control {report[name]['control']})"
              f"{'' if report[name]['wer'] is not None else '  reason: ' + str(report[name]['reason'])}")

    student = [report[k]["wer"] for k in ("rendition_A", "rendition_B", "shared_mean")]
    if all(v is not None for v in student):
        best_rendition = min(student[0], student[1])
        mean_wer = student[2]
        report["best_rendition_wer"] = best_rendition
        if mean_wer <= best_rendition + 0.05:
            verdict = ("OPEN: the shared component decodes as well as a rendition, so the exact target a "
                       "text-conditioned model would aim at is intelligible.")
        else:
            # Averaging latents moves them off the autoencoder's manifold, so the target is worse than a
            # rendition -- but "worse than a rendition" is not "useless": compare it with what the model
            # actually produces today (WER 1.000) before calling the method closed.
            verdict = (f"QUALIFIED: the shared component decodes at WER {mean_wer:.3f} against {best_rendition:.3f} "
                       f"for the better rendition, so averaging latents DEGRADES the target -- it moves off the "
                       f"decoder's manifold. The method is therefore not a free win, but it is not worthless "
                       f"either: that target is still far more intelligible than the WER 1.000 this project's "
                       f"models produce, so it sets a real floor for a route that aims at the shared component.")
        report["verdict"] = verdict
        print(f"\n  VERDICT: {verdict}")
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
