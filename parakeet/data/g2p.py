"""Grapheme-to-phoneme, locally, with the espeak-ng that `espeakng-loader` ships.

The papers feed phonemes and this project has always fed characters -- which is the prime suspect for
its generalisation failure on unseen text (rounds 24-27 ruled out corpus duration, text diversity,
teacher count, teacher quality and real alignment, leaving the text side's inductive bias).  A
character model has to relearn English spelling-to-sound from a few hundred sentences; a phoneme model
gets the sound directly.

The obvious route (`misaki` -> spacy -> blis) does not build in this environment, but `espeakng-loader`
ships the espeak-ng DLL and data and `phonemizer` can drive it.  Everything here is optional: when the
phonemiser is missing, callers fall back to characters and the cache records which was used.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence

#: relative duration weights for splitting a *word's* measured duration across its phonemes.  Words and
#: their timings come from the teacher's marks; the within-word split is an approximation, documented as
#: such, because preserving the word boundary exactly is what matters for prosody.
_PHONEME_WEIGHTS: Dict[str, float] = {
    "vowel": 2.0,
    "consonant": 1.0,
    "silence": 0.5,
}


def phonemize_available() -> bool:
    """Is a local phonemiser importable and usable?"""
    try:
        import espeakng_loader  # type: ignore  # noqa: F401
        import phonemizer  # type: ignore  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - optional dependency
        return False


@lru_cache(maxsize=1)
def _phonemizer():
    """Build the espeak-ng-backed phonemizer once, with the library paths in the environment."""
    import espeakng_loader
    from phonemizer import phonemize

    os.environ.setdefault("ESPEAK_DATA_PATH", str(espeakng_loader.get_data_path()))
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(espeakng_loader.get_library_path()))
    return phonemize


@lru_cache(maxsize=8192)
def text_to_phonemes(text: str, language: str = "en-us") -> str:
    """Text -> IPA.  Cached: the phonemiser is fast per call but the cache build calls it thousands of
    times on a corpus that repeats prompts across voices."""
    phonemize = _phonemizer()
    result = phonemize([text], language=language, backend="espeak", strip=True,
                       preserve_punctuation=False)
    return result[0] if result else ""


def _weight(symbol: str) -> float:
    if symbol.isspace():
        return _PHONEME_WEIGHTS["silence"]
    # vowels in espeak's en-us inventory; everything else counts as a consonant
    if symbol in "əɪʊɛæɑɔʌɜɐᵻiueoaɚɝ":
        return _PHONEME_WEIGHTS["vowel"]
    if symbol in "ˈˌː":
        return 0.0  # stress and length marks attach to a neighbouring phoneme
    return _PHONEME_WEIGHTS["consonant"]


def phoneme_frames_from_char_frames(
    text: str,
    char_frames: Sequence[int],
    phonemize: Callable[[str], str] = text_to_phonemes,
) -> Optional[List[int]]:
    """Per-**character** frame counts (the teacher's marks) -> per-**phoneme** frame counts.

    The marks time *words*, which is the strongest signal a duration head can get short of a forced
    aligner.  This keeps every word's total exactly and splits it across that word's phonemes by a
    fixed weight table, so the within-word distribution is an approximation while the word boundaries --
    which is what the prosody depends on -- are the teacher's own.
    """
    if not char_frames:
        return None
    words: List[tuple] = []  # (word_text, total_frames, trailing_space_frames)
    current = ""
    total = 0
    trailing = 0
    for index, character in enumerate(text):
        frames = int(char_frames[index]) if index < len(char_frames) else 0
        if character.isspace():
            if current:
                trailing += frames
                words.append((current, total, trailing))
                current, total, trailing = "", 0, 0
            else:
                trailing += frames
            continue
        current += character
        total += frames
    if current:
        words.append((current, total, trailing))

    out: List[int] = []
    for word, word_frames, space_frames in words:
        phonemes = phonemize(word)
        symbols = [s for s in phonemes if not s.isspace()]
        if not symbols:
            out.extend([max(1, word_frames)] if word_frames else [1])
            out.extend([max(1, space_frames)] if space_frames else [])
            continue
        weights = [_weight(s) for s in symbols]
        if sum(weights) <= 0:
            weights = [1.0] * len(symbols)
        budget = max(len(symbols), word_frames)
        allocated = [max(1, int(round(budget * w / sum(weights)))) for w in weights]
        # keep the word's measured total: adjust the longest phoneme by the rounding difference
        difference = word_frames - sum(allocated)
        if allocated:
            longest = max(range(len(allocated)), key=lambda i: allocated[i])
            allocated[longest] = max(1, allocated[longest] + difference)
        out.extend(allocated)
        if space_frames:
            out.append(max(1, space_frames))
    return out or None
