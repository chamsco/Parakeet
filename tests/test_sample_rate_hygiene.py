"""The 48 kHz teacher exposed three sample-rate bugs; these tests keep them fixed.

Round 26/27: `whisper_wer` assumed 16 kHz input, `real_eval.py` declared the config rate for a
teacher's reference audio, and `WaveformCorpusSource` fed raw 48 kHz audio to a model whose mel
filterbank assumes 24 kHz.  Each was invisible while every teacher was 24 kHz, and each produced a
*wrong number* rather than an error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from parakeet.data.dataset import WaveformCorpusSource
from parakeet.data.text import is_prose_like


def _write(path: Path, samples: int, rate: int) -> None:
    wave = (np.sin(np.arange(samples) * 0.05) * 0.2).astype(np.float32)
    sf.write(str(path), wave, rate)


def test_waveform_source_resamples_a_teacher_to_the_student_rate(tmp_path):
    """A 48 kHz teacher must arrive at the model's rate, not at half speed."""
    (tmp_path / "wav").mkdir(parents=True, exist_ok=True)
    _write(tmp_path / "wav" / "high.wav", 48000, 48000)
    _write(tmp_path / "wav" / "low.wav", 24000, 24000)
    lines = [
        json.dumps({"utt_id": "high", "text": "a", "teacher": "t", "wav_path": "wav/high.wav",
                    "sample_rate": 48000, "duration_s": 1.0}),
        json.dumps({"utt_id": "low", "text": "b", "teacher": "t", "wav_path": "wav/low.wav",
                    "sample_rate": 24000, "duration_s": 1.0}),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw = WaveformCorpusSource(manifest, batch_size=2, seed=0, shuffle=False)
    batch_raw = raw()
    assert batch_raw["wav_lengths"].tolist() == [48000, 24000], "without a target rate, unchanged"

    resampled = WaveformCorpusSource(manifest, batch_size=2, seed=0, shuffle=False, sample_rate=24000)
    batch = resampled()
    assert batch["wav_lengths"].tolist() == [24000, 24000], (
        "one second of audio must be one second of audio at the model's rate"
    )


def test_prose_filter_rejects_headings_that_cannot_be_transcribed():
    """A heading's written form never matches speech, so any WER over it is inflated."""
    assert not is_prose_like("A Caucus-Race and a Long Tale CHAPTER IV.")
    assert not is_prose_like("CHAPTER XII. The Man in the Coach.")
    assert not is_prose_like("PRIDE AND PREJUDICE")
    assert is_prose_like("A maid rushed across and threw open the window.")
    assert is_prose_like("After a while, finding that nothing more happened, she decided to go on.")
