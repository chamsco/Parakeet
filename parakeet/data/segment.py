"""Cut a long recording into training utterances at silence boundaries.

Every corpus builder in this project assumed the teacher returns one utterance.  Round 29's
user-provided files are 39-48 s each -- 5.6 minutes of speech in eight voices -- and **every one failed
curation on `max_duration_s`**, because the gate rejects long audio instead of segmenting it.  The same
gap would reject any real recording (an audiobook chapter, a podcast) that a future teacher provides.

Word timings come from the ASR that transcribed the file (`faster_whisper` with `word_timestamps=True`),
which is also how the *text* for each segment is recovered: audio without text cannot train a TTS model.
The segments are packed greedily so each one is within the duration window, preferring to cut at the
largest silences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class Segment:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def segment_by_words(
    words: Sequence[Dict],
    min_seconds: float = 3.0,
    max_seconds: float = 15.0,
    gap_seconds: float = 0.30,
    total_seconds: float | None = None,
) -> List[Segment]:
    """Pack timestamped words into utterances of ``min_seconds .. max_seconds``.

    A segment closes when the next word is separated by more than ``gap_seconds`` and the segment is
    already long enough, or when adding the next word would exceed ``max_seconds``.  Trailing time up
    to ``total_seconds`` is attached to the final segment so the audio is not silently truncated.
    """
    cleaned: List[Dict] = []
    for word in words:
        try:
            start, end = float(word["start"]), float(word["end"])
            text = str(word.get("word") or word.get("text") or "").strip()
        except (KeyError, TypeError, ValueError):
            continue
        if text and end > start:
            cleaned.append({"start": start, "end": end, "text": text})
    if not cleaned:
        return []

    segments: List[Segment] = []
    current: List[Dict] = []

    def flush() -> None:
        if not current:
            return
        segments.append(
            Segment(
                start=current[0]["start"],
                end=current[-1]["end"],
                text=" ".join(w["text"] for w in current),
            )
        )
        current.clear()

    for index, word in enumerate(cleaned):
        if current:
            gap = word["start"] - current[-1]["end"]
            would_run = word["end"] - current[0]["start"]
            if (gap >= gap_seconds and (current[-1]["end"] - current[0]["start"]) >= min_seconds) or (
                would_run > max_seconds
            ):
                flush()
        current.append(word)
    flush()

    # a segment shorter than the floor is merged into its neighbour rather than dropped: the words are
    # real speech and the text belongs with the audio
    merged: List[Segment] = []
    for segment in segments:
        if merged and segment.duration < min_seconds:
            previous = merged.pop()
            merged.append(
                Segment(previous.start, segment.end, f"{previous.text} {segment.text}".strip())
            )
        else:
            merged.append(segment)
    if total_seconds is not None and merged:
        last = merged[-1]
        merged[-1] = Segment(last.start, min(last.end, total_seconds) if last.end <= total_seconds
                             else last.end, last.text)
    return merged


def split_long_segment(segment: Segment, max_seconds: float, words: Sequence[Dict]) -> List[Segment]:
    """Split a segment that no packing can bring under the cap (e.g. one unbroken 40 s sentence).

    Uses the words inside the span to find the least-bad cut, and returns the pieces; a single word
    longer than the cap is returned unchanged rather than cut mid-word.
    """
    if segment.duration <= max_seconds:
        return [segment]
    inside = [w for w in words if segment.start - 1e-6 <= float(w["start"]) < segment.end]
    if len(inside) < 2:
        return [segment]
    pieces = segment_by_words(inside, min_seconds=max_seconds / 4, max_seconds=max_seconds,
                              gap_seconds=1e9)  # force cuts on the duration cap only
    if len(pieces) < 2:
        count = max(2, int(math.ceil(segment.duration / max_seconds)))
        step = segment.duration / count
        return [
            Segment(segment.start + i * step, segment.start + (i + 1) * step,
                    " ".join(w["text"] for w in inside if segment.start + i * step <= float(w["start"])
                             < segment.start + (i + 1) * step) or segment.text)
            for i in range(count)
        ]
    return pieces
