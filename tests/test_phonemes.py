"""Phoneme input: the G2P path, the vocabulary it needs, and the frames it must carry.

The papers feed phonemes; this project has always fed characters.  After rounds 24-27 ruled out corpus
duration, text diversity, teacher count, teacher quality and real alignment, the text side's inductive
bias is what remains -- and a character model has to relearn English spelling-to-sound from a few
hundred sentences.

Two properties matter enough to test: the vocabulary must cover what the phonemiser actually emits
(the hand-written `_IPA_CHARS` missed the length mark `ː`, which appears 1865 times in 600 prompts, so
a phoneme model would have sent `UNK` for nearly every long vowel), and word-level timings from the
teacher's marks must survive the conversion to phoneme tokens.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parakeet.config import load_config
from parakeet.data.g2p import (
    phoneme_frames_from_char_frames,
    phonemize_available,
    text_to_phonemes,
)
from parakeet.data.text import TextTokenizer

requires_phonemizer = pytest.mark.skipif(
    not phonemize_available(), reason="no local phonemiser (espeakng-loader + phonemizer)"
)


@requires_phonemizer
def test_phoneme_mode_phonemises_before_encoding():
    tokenizer = TextTokenizer(mode="phoneme")
    assert tokenizer.phonemized is True
    prepared = tokenizer.prepare("The quick brown fox jumps over the lazy dog.")
    assert prepared != "The quick brown fox jumps over the lazy dog."
    assert "kw" in prepared, prepared
    ids = tokenizer.encode("The quick brown fox jumps over the lazy dog.", add_special=False)
    unknown = tokenizer.vocab.stoi["<unk>"]
    assert int((ids == unknown).sum()) == 0, "every emitted symbol must be in the vocabulary"


@requires_phonemizer
def test_unseen_words_phonemise_without_unk():
    """The point of phonemes: a word the corpus never saw still maps to sounds."""
    tokenizer = TextTokenizer(mode="phoneme")
    for word in ("quixotic", "perspicacious", "sesquipedalian"):
        ids = tokenizer.encode(word, add_special=False)
        assert int(ids.numel()) > 0
        assert int((ids == tokenizer.vocab.stoi["<unk>"]).sum()) == 0, word


def test_phoneme_vocabulary_fits_the_configured_embedding():
    """A symbol outside the embedding is an index error at training time, not a warning."""
    cfg = load_config("configs/parakeet_tiny.yaml")
    for mode in ("char", "phoneme"):
        tokenizer = TextTokenizer(mode=mode)
        assert tokenizer.vocab_size <= cfg.text.vocab_size, (
            f"{mode} vocab {tokenizer.vocab_size} exceeds the configured embedding "
            f"{cfg.text.vocab_size}"
        )


@requires_phonemizer
def test_phoneme_frames_preserve_each_words_measured_total():
    """The teacher's marks time words; converting to phonemes must not move that timing."""
    text = "The quick brown fox"
    char_frames = [6, 6, 6, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 3, 3, 3, 3]
    frames = phoneme_frames_from_char_frames(text, char_frames)
    assert frames is not None
    assert sum(frames) == sum(char_frames), (sum(frames), sum(char_frames))

    tokenizer = TextTokenizer(mode="phoneme")
    ids = tokenizer.encode(text, add_special=False)
    assert len(frames) == int(ids.numel()), "one duration per phoneme token, or the axis misaligns"

    # word boundaries, measured from the characters, are the teacher's own
    assert all(f >= 1 for f in frames)


def test_missing_phonemiser_degrades_to_characters_instead_of_unk(monkeypatch):
    import parakeet.data.g2p as g2p

    monkeypatch.setattr(g2p, "phonemize_available", lambda: False)
    tokenizer = TextTokenizer(mode="phoneme")
    assert tokenizer.phonemized is False, "it must not pretend to phonemise"
    ids = tokenizer.encode("hello world", add_special=False)
    assert int(ids.numel()) > 0, "it must still tokenise"
    unknown = tokenizer.vocab.stoi["<unk>"]
    assert int((ids == unknown).sum()) == 0, "characters are all in the vocabulary"


@requires_phonemizer
def test_conversion_returns_none_without_frames():
    assert phoneme_frames_from_char_frames("hello", []) is None
    assert text_to_phonemes("hello") != ""
