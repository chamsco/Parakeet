"""Are newly generated wavs sane before hours of them go into the corpus?

    python scripts/generated_audio_qa.py --dir data/kokoro_v3/wav

A misconfigured TTS run is quiet, clipped or short, and none of that shows up until training behaves oddly.
This measures level, crest factor and duration on a sample of the new corpus and compares with the corpus's
existing Kokoro items -- the same teacher, so the statistics should be close.
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

random.seed(0)


def stats(path: Path) -> tuple:
    with wave.open(str(path)) as handle:
        raw = handle.readframes(handle.getnframes())
        width, rate, frames = handle.getsampwidth(), handle.getframerate(), handle.getnframes()
    if width != 2 or not frames:
        return (0.0, 0.0, 0.0, 0.0, rate)
    samples = array.array("h")
    samples.frombytes(raw)
    peak = max(abs(min(samples)), abs(max(samples))) / 32768.0
    rms = math.sqrt(sum(float(s) * s for s in samples) / len(samples)) / 32768.0
    rms = max(rms, 1e-9)
    peak = max(peak, 1e-9)
    return (20 * math.log10(rms), 20 * math.log10(peak), 20 * math.log10(peak / rms),
            frames / rate, rate)


def summarise(paths: list, label: str) -> dict:
    rows = [stats(p) for p in paths]
    median = lambda i: statistics.median([r[i] for r in rows])
    print(f"{label:28s} n={len(rows):4d} | rms {median(0):7.2f} dB | peak {median(1):7.2f} dB | "
          f"crest {median(2):6.2f} dB | duration {median(3):5.2f} s | rate {int(median(4))}")
    return {"n": len(rows), "rms_db": median(0), "peak_db": median(1), "crest_db": median(2),
            "duration_s": median(3), "rate": int(median(4))}


import argparse

ap = argparse.ArgumentParser(description="Level/clipping/duration sanity for generated audio")
ap.add_argument("--dir", default="data/kokoro_v3/wav")
ap.add_argument("--limit", type=int, default=40)
args = ap.parse_args()
new_dir = Path(args.dir)
new_wavs = sorted(new_dir.glob("*.wav"))
if not new_wavs:
    raise SystemExit("no generated wavs yet")
sample = random.sample(new_wavs, min(args.limit, len(new_wavs)))

manifest = Path("data/kokoro_v2/manifest.jsonl")
old_wavs = []
if manifest.exists():
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            candidate = Path(record.get("corpus_root", "")) / record.get("wav_path", "")
            if candidate.exists():
                old_wavs.append(candidate)
old_sample = random.sample(old_wavs, min(40, len(old_wavs))) if old_wavs else []

print(f"generated so far: {len(new_wavs)} wavs\n")
summarise(old_sample, "previous Kokoro expansion") if old_sample else None
new = summarise(sample, "NEW generation (kokoro_v3)")

# the two checks that would catch a broken run
problems = []
if new["rms_db"] < -35 or new["rms_db"] > -10:
    problems.append(f"level out of range: {new['rms_db']:.1f} dB rms")
if new["peak_db"] > -0.1:
    problems.append(f"clipping: peak {new['peak_db']:.2f} dB")
if new["duration_s"] < 1.5:
    problems.append(f"utterances too short: {new['duration_s']:.2f} s median")
if new["rate"] != 24000:
    problems.append(f"unexpected sample rate: {new['rate']}")
print("\n" + ("PROBLEMS: " + "; ".join(problems) if problems else
              "all checks passed: level, clipping, duration and sample rate are in range"))
