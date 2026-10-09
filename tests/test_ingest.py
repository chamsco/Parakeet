"""Ingesting arbitrary recordings: silence segmentation, and the text that must come with it.

Round 29 arrived with eight 39-48 s Speechify takes -- 5.6 minutes of speech in eight voices -- and
**every one failed curation on `max_duration_s`**, because the gate rejected long audio instead of
segmenting it.  That would reject any real recording (a chapter, a podcast), so the pipeline gained a
segmenter.  These tests pin its two contracts: segments land inside the duration window, and no words
are lost or duplicated when the text is rebuilt from the word timings.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from parakeet.data.segment import segment_by_words, split_long_segment

ROOT = Path(__file__).resolve().parents[1]


def _words(spec):
    """``spec`` is a list of (start, end, text)."""
    return [{"start": s, "end": e, "word": t} for s, e, t in spec]


def test_segments_respect_the_duration_window_and_keep_every_word():
    # a 40 s "recording": 10 groups of 4 words with clear pauses between the groups
    spec = []
    clock = 0.0
    expected = []
    for group in range(10):
        for word in range(4):
            spec.append((clock, clock + 0.7, f"w{group}_{word}"))
            clock += 0.75
        expected.extend(f"w{group}_{w}" for w in range(4))
        clock += 0.5  # a pause between groups
    words = _words(spec)

    segments = segment_by_words(words, min_seconds=3.0, max_seconds=14.0, gap_seconds=0.3)
    assert len(segments) > 1, "a 40 s recording must not come back as one segment"
    for segment in segments:
        assert segment.duration <= 14.0 + 1e-6, segment.duration
    # merging pulls short neighbours together, so the window is not a hard lower bound on every piece,
    # but nothing may be lost
    rebuilt = " ".join(s.text for s in segments).split()
    assert rebuilt == expected, (len(rebuilt), len(expected))


def test_a_recording_with_no_pauses_is_still_cut_to_the_cap():
    """One 40 s stretch with no usable pause must come back as several pieces."""
    words = _words([(i * 0.5, i * 0.5 + 0.45, f"w{i}") for i in range(80)])  # 40 s, no gaps
    segments = segment_by_words(words, min_seconds=3.0, max_seconds=14.0, gap_seconds=0.3)
    assert len(segments) >= 3, "the packer must cut on the duration cap, not only on pauses"
    for segment in segments:
        assert segment.duration <= 14.0 + 1e-6, segment.duration
    assert " ".join(s.text for s in segments).split() == [f"w{i}" for i in range(80)]

    # and an explicitly oversized segment is split on its own word timings
    pieces = split_long_segment(segments[0], 6.0, words)
    assert all(p.duration <= 6.0 + 1e-6 for p in pieces)
    assert len(pieces) >= 2


def test_no_words_means_no_segments():
    assert segment_by_words([]) == []
    assert segment_by_words([{"start": 1.0, "end": 1.0, "word": "x"}]) == []


def test_mixer_accepts_a_per_corpus_manifest(tmp_path):
    """Generated corpora write train.jsonl; an ingested one writes curated/kept.jsonl."""
    import numpy as np
    import soundfile as sf

    for name, manifest_name in (("gen", "train.jsonl"), ("ingested", "curated/kept.jsonl")):
        corpus = tmp_path / name / "corpus"
        (corpus / "wav").mkdir(parents=True, exist_ok=True)
        (corpus / manifest_name).parent.mkdir(parents=True, exist_ok=True)
        wave = (np.sin(np.arange(2400) * 0.05) * 0.2).astype(np.float32)
        sf.write(str(corpus / "wav" / "a.wav"), wave, 24000)
        (corpus / manifest_name).write_text(
            json.dumps({
                "utt_id": "a", "text": "a short sentence for the mixer test", "teacher": name,
                "voice": "v0", "wav_path": "wav/a.wav", "sample_rate": 24000, "duration_s": 0.1,
                "license": f"{name} licence",
            }) + "\n",
            encoding="utf-8",
        )
        (corpus / "corpus_meta.json").write_text(
            json.dumps({"teachers": {name: {"weights_license": f"{name} licence"}}}), encoding="utf-8"
        )
    out = tmp_path / "mixed"
    result = subprocess.run(
        [sys.executable, "scripts/mix_corpora.py",
         "--corpus", str(tmp_path / "gen" / "corpus") + ":train.jsonl",
         "--corpus", str(tmp_path / "ingested" / "corpus") + ":curated/kept.jsonl",
         "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["teacher"] for r in rows} == {"gen", "ingested"}
