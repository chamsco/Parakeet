"""Build a teacher corpus (the "mix training" data stage).

    # permissive teachers only (default) -- Orpheus 60% / Kokoro 40%
    python scripts/make_teacher_corpus.py --texts data/prompts.txt --out data/teacher_corpus \
        --teachers orpheus=0.6,kokoro=0.4 --limit 200

    # then cache latents + teacher signals for the two student halves
    python scripts/make_teacher_corpus.py --cache-only --corpus data/teacher_corpus \
        --config configs/parakeet_tiny.yaml --ae-checkpoint runs/parakeet-tiny/autoencoder_last.pt

MiniMax is refused unless you explicitly acknowledge its terms (`--acknowledge-restricted`
plus `PARAKEET_ACCEPT_TEACHER_TOS=1`, or `--i-have-written-permission`).  See docs/LEGAL.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.data.curate import curate_manifest  # noqa: E402
from parakeet.data.features import cache_teacher_corpus  # noqa: E402
from parakeet.data.teacher import (  # noqa: E402
    DEFAULT_MIX,
    TeacherLicenseError,
    build_backend,
    check_teacher,
    resolve_mix,
    synthesize_corpus,
)
from parakeet.data.text import TextTokenizer  # noqa: E402
from parakeet.models import build_model  # noqa: E402

DEFAULT_VOICES = {
    "orpheus": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
    # the 11 speakers in the kokoro-en-v0_19 bundle (the 53/103-speaker multi-lang bundles add more;
    # SherpaKokoroBackend raises on a name it does not have rather than using the wrong speaker)
    "kokoro": [
        "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_nova",
        "af_river", "af_alloy", "af_aoede", "af_jessica", "af_kore",
    ],
    "minimax": ["English_expressive_narrator"],
    # fixtures: distinct pitches, so a multi-voice fixture corpus exercises voice conditioning
    "stub_low": ["low", "mid", "high"],
    "stub_high": ["mid", "high"],
}


def read_texts(path: Path, limit: int | None) -> list[str]:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[:limit] if limit else lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Parakeet teacher corpus")
    ap.add_argument("--texts", default=None, help="text prompts, one per line")
    ap.add_argument("--out", default="data/teacher_corpus")
    ap.add_argument("--teachers", default=",".join(f"{k}={v}" for k, v in DEFAULT_MIX.items()))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--acknowledge-restricted", action="store_true")
    ap.add_argument("--i-have-written-permission", action="store_true")
    # cache-only mode
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--corpus", default=None, help="corpus dir containing manifest.jsonl")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--ae-checkpoint", default=None)
    ap.add_argument("--cache-out", default="data/latent_cache")
    ap.add_argument("--no-curate", action="store_true",
                    help="skip the P1 quality gates (loudness/clipping/SNR/punctuation).  Curation "
                         "is ON by default: it is the documented pipeline, and curate_manifest() "
                         "used to be dead code that no entry point ever called")
    ap.add_argument("--curate-out", default=None, help="where kept.jsonl/rejected.jsonl go")
    ap.add_argument("--cache-curated-only", action="store_true", default=True,
                    help="cache the curated (kept.jsonl) manifest rather than the raw one")
    args = ap.parse_args()

    mix = resolve_mix(args.teachers.split(","))
    acknowledge = args.acknowledge_restricted or args.i_have_written_permission

    for name in mix:
        try:
            spec = check_teacher(name, acknowledge_restricted=acknowledge)
        except TeacherLicenseError as exc:
            print(f"REFUSED: {exc}")
            return 2
        print(f"teacher {name}: {spec.kind} | weights: {spec.weights_license}")

    if args.cache_only:
        corpus = Path(args.corpus or args.out)
        cfg = load_config(args.config)
        model = build_model(cfg)
        if args.ae_checkpoint:
            payload = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
            state = payload.get("ema", {}).get("shadow", payload["model"])
            model.load_state_dict(state, strict=False)
            print(f"loaded autoencoder from {args.ae_checkpoint}")
        else:
            print("WARNING: no --ae-checkpoint, latents come from a randomly-initialised encoder")
        # the mixture is read from the corpus provenance, so the cache cannot silently lose it
        out = cache_teacher_corpus(
            corpus,
            args.cache_out,
            cfg,
            model.autoencoder,
            tokenizer=TextTokenizer(mode=cfg.text.mode),
            limit=args.limit,
            teacher_latent_norm=model.latent_norm,
        )
        meta = json.loads((Path(args.cache_out) / "cache_meta.json").read_text(encoding="utf-8"))
        print(f"wrote latent cache -> {out}")
        print(f"  teachers {meta['teacher_names']} | mixture {meta['teacher_weights']} | "
              f"voices {meta['voice_names']}")
        return 0

    if not args.texts:
        print("--texts is required unless --cache-only")
        return 2
    texts = read_texts(Path(args.texts), args.limit)
    print(f"{len(texts)} prompts; mixture {mix}")

    backends = {}
    for name in mix:
        if name == "minimax":
            backends[name] = build_backend(name, acknowledge_restricted=acknowledge)
        else:
            backends[name] = build_backend(name)
    manifest = synthesize_corpus(
        texts,
        args.out,
        mix=mix,
        voices=DEFAULT_VOICES,
        backends=backends,
        acknowledge_restricted=acknowledge,
        max_utts=args.limit,
    )
    print(f"manifest -> {manifest}")
    for b in backends.values():
        b.close()

    if not args.no_curate:
        import soundfile as sf

        records = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        def load_wav(rel: str):
            wav, sr = sf.read(str(Path(args.out) / rel), dtype="float32")
            return torch.from_numpy(wav), sr

        curate_out = Path(args.curate_out) if args.curate_out else Path(args.out) / "curated"
        report = curate_manifest(records, load_wav, curate_out, normalize=True)
        print(
            f"curation -> kept {report.n_kept}/{report.n_total} "
            f"({report.hours_kept * 3600:.1f}s), rejected {report.n_rejected} "
            f"{dict(report.reason_counts)}"
        )
        print(f"  kept -> {curate_out/'kept.jsonl'}\n  rejected -> {curate_out/'rejected.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
