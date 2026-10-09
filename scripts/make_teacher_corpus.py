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
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.data.features import build_latent_cache  # noqa: E402
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
    "kokoro": ["af_heart", "af_bella", "am_michael", "bf_emma"],
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
        out = build_latent_cache(
            corpus / "manifest.jsonl",
            args.cache_out,
            cfg,
            model.autoencoder,
            tokenizer=TextTokenizer(mode=cfg.text.mode),
            limit=args.limit,
        )
        print(f"wrote latent cache -> {out}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
