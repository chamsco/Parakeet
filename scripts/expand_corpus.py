"""Cache, merge and normalise a new corpus, verifying the traps that have each cost this project a round.

Every step here exists because a previous expansion failed at it:

* `--latent-rate 3` — the first expansion's cache was built at rate 1 (24-wide token targets) against the
  base cache's 3 (72-wide); the merge would have produced 770 items the model could not read among 1 207 it
  could (round 56).  The merge now *refuses* that, but the cache is built correctly to begin with;
* `merge_caches` remaps voice indices and keeps the base cache's latent normaliser — the table is per-cache,
  and refitting the normaliser would move the latent space under a resumed checkpoint (round 56);
* `fit_voice_norm` — a voice's latent scale is a property of that voice (corpus Kokoro pools at 0.83-0.85,
  other Kokoro voices at 0.44-0.52), which is why two expansions landed 41 % away and measured *worse* when
  mixed in (rounds 58-62);
* the verification pass fails loudly on a width mismatch, a missing voice table, or a half whose pooled scale
  is still out of line — the three symptoms that otherwise surface much later as an IndexError in a batch
  source or as quieter, unexplained audio.

    python scripts/expand_corpus.py --corpus data/kokoro_v3 --cache runs/kokoro_v3_cache \
        --base runs/mixed_v2/latent_cache --out runs/expanded_v3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from parakeet.data.dataset import LatentShardDataset  # noqa: E402
from fit_voice_norm import fit_voice_norm  # noqa: E402
from merge_caches import merge_caches  # noqa: E402


def build_cache(corpus: Path, cache: Path, config: str, ae: str, latent_rate: int) -> None:
    command = [
        sys.executable, "scripts/make_teacher_corpus.py",
        "--corpus", str(corpus), "--cache-only", "--config", config,
        "--ae-checkpoint", ae, "--cache-out", str(cache), "--latent-rate", str(latent_rate),
    ]
    print(f"$ {' '.join(command[1:])}", flush=True)
    done = subprocess.run(command, cwd=str(ROOT))
    if done.returncode:
        raise SystemExit(f"cache build failed with exit code {done.returncode}")


def verify(cache: Path, base_items: int, min_scale_gap: float = 0.15) -> dict:
    """Fail loudly on the three symptoms that have each cost a round."""
    dataset = LatentShardDataset(cache)
    meta = json.loads((cache / "cache_meta.json").read_text(encoding="utf-8"))
    voices = list(meta.get("voice_names") or [])

    widths = {int(dataset[i]["latent_token"].shape[-1]) for i in range(0, len(dataset), 37)}
    if widths != {72}:
        raise SystemExit(f"token widths are {widths}, expected {{72}} -- the caches cannot be mixed")

    voice_norm = meta.get("voice_norm") or {}
    if not voice_norm:
        raise SystemExit("the merged cache has no voice_norm: per-voice scale will not be normalised")

    # every fitted voice should pool near 1.0 after normalisation; the *new* half is what to check
    scales: dict = {}
    for index in range(0, len(dataset), 13):
        item = dataset[index]
        key = int(item["voice"])
        scales.setdefault(key, []).append(item["latent"])
    pooled = {v: float(torch.cat(x, dim=-1).std()) for v, x in scales.items()}
    finished = {v: s for v, s in pooled.items() if voices[v] in voice_norm}
    worst = max((abs(s - 1.0) for s in finished.values()), default=0.0)

    report = {
        "items": len(dataset),
        "base_items": base_items,
        "new_items": len(dataset) - base_items,
        "voices": len(voices),
        "voices_with_normalisation": len(voice_norm),
        "token_widths": sorted(widths),
        "worst_pooled_scale_error": round(worst, 4),
        "hint": "the expansion's own voices are the last ones in the voice table",
    }
    if worst > min_scale_gap:
        raise SystemExit(
            f"a fitted voice pools at {worst:.3f} from unit scale (> {min_scale_gap}): "
            f"per-voice normalisation is not doing its job"
        )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache, merge and normalise a corpus expansion")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cache", required=True, help="where the new corpus's cache goes")
    ap.add_argument("--base", default="runs/mixed_v2/latent_cache")
    ap.add_argument("--out", required=True, help="the merged cache")
    ap.add_argument("--latent-rate", type=int, default=3)
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--ae-checkpoint", default="runs/ae_scaled/adversarial/autoencoder_last.pt")
    ap.add_argument("--skip-cache", action="store_true", help="the per-corpus cache already exists")
    args = ap.parse_args()

    corpus, cache, base, out = Path(args.corpus), Path(args.cache), Path(args.base), Path(args.out)
    base_items = len(LatentShardDataset(base))

    if not args.skip_cache or not (cache / "index.json").exists():
        build_cache(corpus, cache, args.config, args.ae_checkpoint, args.latent_rate)

    counts = merge_caches(base, cache, out)
    print(f"merged: {json.dumps(counts)}")

    fitted = fit_voice_norm(out)
    meta = json.loads((out / "cache_meta.json").read_text(encoding="utf-8"))
    meta["voice_norm"] = fitted
    (out / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"fitted per-voice statistics for {len(fitted)} voices")

    report = verify(out, base_items)
    Path("runs/expand_corpus_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
