"""Do the expansion's wavs sit at a different level than the corpus's Speechify wavs?

The expansion's latents are 41 % narrower (token std 0.643 against 1.096). The cache stores the wav too, so
the question "is this a level difference or a different voice?" can be answered from the audio rather than
guessed at -- and it decides whether a simple gain fixes the expansion or whether the data is genuinely
different speech.
"""

from __future__ import annotations

import array
import json
import math
import random
import statistics
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

random.seed(0)


def stats(path: Path) -> tuple[float, float, float]:
    """Return (rms_db, peak_db, crest_db) for a 16-bit wav."""
    with wave.open(str(path)) as handle:
        raw = handle.readframes(handle.getnframes())
        width = handle.getsampwidth()
        rate = handle.getframerate()
    if width != 2:
        return (0.0, 0.0, 0.0)
    samples = array.array("h")
    samples.frombytes(raw)
    if not samples:
        return (0.0, 0.0, 0.0)
    peak = max(abs(min(samples)), abs(max(samples))) / 32768.0
    rms = math.sqrt(sum(float(s) * s for s in samples) / len(samples)) / 32768.0
    rms = max(rms, 1e-9)
    peak = max(peak, 1e-9)
    return (20 * math.log10(rms), 20 * math.log10(peak), 20 * math.log10(peak / rms))


def collect(root: Path, limit: int = 60) -> dict:
    wavs = sorted(root.rglob("*.wav"))
    return summarise(wavs, limit)


def collect_from_manifest(manifest: Path, limit: int = 60, teacher: str | None = None) -> dict:
    """The corpus manifest carries `corpus_root` + `wav_path`; the wavs are not under the manifest."""
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if teacher:
        rows = [r for r in rows if r.get("teacher") == teacher]
    paths = []
    for row in rows:
        root = row.get("corpus_root") or manifest.parent
        candidate = Path(root) / row.get("wav_path", "")
        if candidate.exists():
            paths.append(candidate)
    return summarise(paths, limit)


def summarise(wavs: list[Path], limit: int) -> dict:
    if len(wavs) > limit:
        wavs = random.sample(wavs, limit)
    rms, peak, crest = [], [], []
    for path in wavs:
        try:
            values = stats(path)
        except Exception:  # noqa: BLE001
            continue
        rms.append(values[0])
        peak.append(values[1])
        crest.append(values[2])
    median = lambda v: statistics.median(v) if v else float("nan")
    return {"n": len(rms), "rms_db": median(rms), "peak_db": median(peak), "crest_db": median(crest)}


base_stats = collect_from_manifest(
    Path("data/mixed_corpus/train_v2/manifest.jsonl"), teacher="speechify"
)
expansion_stats = collect(Path("data/speechify_v2/wav"))
print(f"base corpus      : n={base_stats['n']:3d} rms {base_stats['rms_db']:7.2f} dB  peak {base_stats['peak_db']:7.2f} dB  crest {base_stats['crest_db']:6.2f} dB")
print(f"expansion        : n={expansion_stats['n']:3d} rms {expansion_stats['rms_db']:7.2f} dB  peak {expansion_stats['peak_db']:7.2f} dB  crest {expansion_stats['crest_db']:6.2f} dB")
print(f"\ndifference       : rms {base_stats['rms_db'] - expansion_stats['rms_db']:+.2f} dB | "
      f"crest {base_stats['crest_db'] - expansion_stats['crest_db']:+.2f} dB")
print("\n  A large rms gap is a level difference, fixable with a gain before encoding.")
print("  A similar rms with a different crest factor is genuinely different speech (dynamics),")
print("  and no gain will make its latents match.")
