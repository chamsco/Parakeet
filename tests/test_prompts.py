"""The public-domain prompt builder: licence provenance and usable sentences.

The round-24 hold-out showed that *text diversity*, not corpus duration, is the binding constraint
(47 distinct sentences left the text side at chance on unseen prompts).  These tests pin the two
properties that make a prompt list usable: it must record where every sentence came from, and it must
not pass through text the synthesiser mangles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))

from build_prompts import SOURCES, sentences, strip_gutenberg  # noqa: E402


def test_every_source_records_a_licence():
    assert SOURCES, "a prompt list with no provenance is not usable"
    for source in SOURCES:
        assert source["licence"].startswith("public domain"), source
        assert source["url"].startswith("https://")
        assert source["title"] and source["author"]


def test_gutenberg_boilerplate_is_stripped():
    text = (
        "Header junk nobody wants to synth.\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "This is a real sentence inside the book.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "Licence boilerplate that must not be synthesised.\n"
    )
    body = strip_gutenberg(text)
    assert "This is a real sentence" in body
    assert "Header junk" not in body
    assert "Licence boilerplate" not in body


def test_sentence_filter_keeps_prose_and_drops_junk():
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "No. "
        "CHAPTER XII. "
        "She walked to the window and looked out at the rain falling steadily. "
        "SUPERCALIFRAGILISTICEXPIALIDOCIOUS words are far too long to synthesise reliably here. "
        "He said, 'I shall return before the evening, so do not wait for me.' "
    )
    kept = sentences(text, min_words=8, max_words=22)
    assert any("walked to the window" in s for s in kept)
    assert all(8 <= len(s.split()) <= 22 for s in kept), kept
    assert not any("SUPERCALIFRAGILISTIC" in s for s in kept), "very long words are filtered"
    assert all(s[0].isalpha() for s in kept)
    assert all(s.endswith((".", "!", "?")) for s in kept)
