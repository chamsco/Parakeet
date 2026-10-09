"""Ingest arbitrary user-provided recordings into the teacher-corpus format.

    python scripts/ingest_audio.py --out data/downloads_corpus --teacher speechify \
        --file belinda.wav --file benjamin.wav ...

A recording arrives as audio and nothing else, while training needs (audio, text) pairs -- and long
files exceed the curation window.  This transcribes each file with word timestamps, cuts it into
utterances at the silences (`parakeet.data.segment`), writes them out with a manifest, and records the
provenance: the teacher, its licence from the spec, the source filename, and the fact that the *text*
was recovered by ASR rather than supplied by the teacher.

That last note matters.  A teacher's own transcript is ground truth; an ASR transcript is a
measurement, and it carries the recogniser's errors into the training pairs.  The manifest says which
is which so a later reader is not misled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.data.segment import split_long_segment  # noqa: E402
from parakeet.data.segment import segment_by_words  # noqa: E402
from parakeet.data.teacher import TEACHERS, check_teacher  # noqa: E402


def transcribe_with_words(path: Path, model_size: str) -> tuple:
    import numpy as np
    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel

    wav, rate = sf.read(str(path), dtype="float32")
    mono = wav if wav.ndim == 1 else wav.mean(axis=1)
    samples = np.asarray(mono, dtype="float32")
    if rate != 16000:
        target = max(1, int(samples.size * 16000 / rate))
        samples = torch.nn.functional.interpolate(
            torch.from_numpy(samples)[None, None, :], size=target, mode="linear", align_corners=False
        ).reshape(-1).numpy()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(samples, language="en", word_timestamps=True)
    words: List[Dict] = []
    for segment in segments:
        for word in getattr(segment, "words", None) or []:
            words.append({"start": float(word.start), "end": float(word.end), "text": word.word})
    return words, samples.size / 16000.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest recordings into corpus format")
    ap.add_argument("--file", action="append", required=True, help="a recording (repeatable)")
    ap.add_argument("--voice", action="append", default=None,
                    help="voice name per --file, in order (default: the filename stem)")
    ap.add_argument("--teacher", default="speechify")
    ap.add_argument("--out", default="data/downloads_corpus")
    ap.add_argument("--min-seconds", type=float, default=3.0)
    ap.add_argument("--max-seconds", type=float, default=14.0)
    ap.add_argument("--gap-seconds", type=float, default=0.30)
    ap.add_argument("--whisper", default="base.en")
    ap.add_argument("--min-bandwidth-hz", type=float, default=None,
                    help="override the CosyVoice 5 kHz floor with a *measured* one; round 26 showed it "
                         "mismeasures teachers other than the one it was calibrated on")
    ap.add_argument("--min-dnsmos", type=float, default=None,
                    help="override the calibrated DNSMOS floor (default 2.0)")
    args = ap.parse_args()

    spec = check_teacher(args.teacher)
    out = Path(args.out)
    corpus = out / "corpus"
    (corpus / "wav").mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    records: List[Dict] = []
    per_file: List[Dict] = []
    for index, name in enumerate(args.file):
        path = Path(name)
        if not path.exists():
            print(f"  !! missing {path}")
            continue
        voice = (args.voice[index] if args.voice and index < len(args.voice)
                 else path.stem.split("_")[0])
        words, total_seconds = transcribe_with_words(path, args.whisper)
        segments = segment_by_words(words, args.min_seconds, args.max_seconds, args.gap_seconds)
        # anything still over the cap is split on its own word timings
        pieces = []
        for segment in segments:
            pieces.extend(split_long_segment(segment, args.max_seconds, words))
        pieces = [p for p in pieces if p.duration >= args.min_seconds and p.text.strip()]

        audio, rate = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        for piece in pieces:
            start = int(piece.start * rate)
            stop = int(piece.end * rate)
            chunk = audio[start:stop]
            if chunk.size == 0:
                continue
            utt_id = f"{args.teacher}_{voice}_{len(records):06d}"
            wav_path = corpus / "wav" / f"{utt_id}.wav"
            sf.write(str(wav_path), chunk, rate)
            records.append(
                {
                    "utt_id": utt_id,
                    "text": piece.text,
                    "teacher": args.teacher,
                    "voice": voice,
                    "wav_path": str(wav_path.relative_to(corpus)),
                    "sample_rate": rate,
                    "duration_s": float(chunk.size / rate),
                    "tags": [],
                    "license": spec.weights_license,
                    "hash": "",
                    "source_file": path.name,
                    "text_source": "asr_transcript(faster-whisper "
                                   f"{args.whisper}, word_timestamps=True)",
                }
            )
        per_file.append(
            {
                "file": path.name,
                "voice": voice,
                "seconds": round(total_seconds, 1),
                "asr_words": len(words),
                "segments": len(pieces),
            }
        )
        print(f"  {path.name}: {total_seconds:5.1f}s -> {len(pieces)} segments "
              f"(voice {voice})", flush=True)

    if not records:
        print("nothing ingested")
        return 2

    # curation runs here rather than later because the gates are the only thing that decides whether a
    # segment is usable, and a corpus directory that has never been curated trains on everything
    from parakeet.data.curate import CurateConfig, curate_manifest

    def load_wav(relative: str):
        import torch

        wav, rate = sf.read(str(corpus / relative), dtype="float32")
        return torch.from_numpy(wav), rate

    curate_cfg = CurateConfig()
    if args.min_bandwidth_hz is not None:
        curate_cfg.min_bandwidth_hz = args.min_bandwidth_hz
    if args.min_dnsmos is not None:
        curate_cfg.min_dnsmos = args.min_dnsmos
    report = curate_manifest(records, load_wav, corpus / "curated", curate_cfg, normalize=True)
    kept = [
        json.loads(line)
        for line in (corpus / "curated" / "kept.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"  curation kept {report.n_kept}/{report.n_total} ({dict(report.reason_counts)})")
    records = kept
    if not records:
        print(f"nothing survived curation: {dict(report.reason_counts)}")
        return 2
    (corpus / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    (corpus / "corpus_meta.json").write_text(
        json.dumps(
            {
                "mix": {args.teacher: 1.0},
                "teachers": {args.teacher: json.loads(json.dumps(TEACHERS[args.teacher].__dict__))},
                "n_utterances": len(records),
                "ingested": per_file,
                "curation": {"kept": len(records), "reasons": dict(report.reason_counts)},
                "text_provenance": (
                    "text recovered by ASR from the audio (the recordings arrived without transcripts), "
                    "so the pairs carry the recogniser's errors -- recorded because a teacher's own "
                    "transcript would be ground truth and this is a measurement"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    total = sum(r["duration_s"] for r in records)
    print(f"\ningested {len(records)} utterances, {total / 60:.1f} min, "
          f"{len({r['voice'] for r in records})} voices -> {corpus/'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
