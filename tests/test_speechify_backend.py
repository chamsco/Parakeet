"""The Speechify teacher: request shape, decoding, retries, timings -- and its licence.

The backend talks to a *paid* hosted service, so every test here stubs the transport: a test suite
that spends the operator's characters is a bug in itself.  What is asserted is the part that matters
for a long unattended corpus build -- the request shape, base64/WAV decoding, fast failure on auth
errors, retry on transient ones, honest character accounting, and that word timings become the
per-character duration targets the pipeline has never had.
"""

from __future__ import annotations

import base64
import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from parakeet.data.teacher import (
    TEACHERS,
    SpeechifyBackend,
    check_teacher,
    marks_to_token_frames,
)

ROOT = Path(__file__).resolve().parents[1]


def _wav_bytes(rate: int = 48000, seconds: float = 0.5, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    samples = (np.sin(np.arange(int(rate * seconds)) * 0.05) * 8000).astype(np.int16)
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def _fake_response(payload: dict):
    class _Response:
        status = 200

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Response()


@pytest.fixture()
def backend(monkeypatch):
    monkeypatch.setenv("SPEECHIFY_API_KEY", "sk_test_key_not_real")
    monkeypatch.setattr(SpeechifyBackend, "_read_secret_file", staticmethod(lambda: None))
    return SpeechifyBackend(attempts=2)


def test_spec_records_the_licence_distinction():
    spec = check_teacher("speechify")
    assert spec.allows_training is True
    assert spec.sample_rate == 48000
    # the permission covers demonstration/quantization; publishing weights is broader, and the spec
    # must say so rather than implying a general training licence
    assert "demonstration" in spec.notes
    assert "derivative" in spec.weights_license or "derivative" in spec.notes
    assert "quantization" in spec.weights_license or "quantization" in spec.notes


def test_backend_refuses_to_run_without_a_key(monkeypatch):
    monkeypatch.delenv("SPEECHIFY_API_KEY", raising=False)
    monkeypatch.setattr(SpeechifyBackend, "_read_secret_file", staticmethod(lambda: None))
    with pytest.raises(RuntimeError, match="SPEECHIFY_API_KEY"):
        SpeechifyBackend()


def test_marks_become_per_character_frames():
    marks = {
        "chunks": [
            {"type": "word", "start": 0, "end": 3, "start_time": 0, "end_time": 256, "value": "The"},
            {"type": "word", "start": 4, "end": 9, "start_time": 256, "end_time": 512, "value": "quick"},
            {"type": "word", "start": 10, "end": 15, "start_time": 512, "end_time": 811, "value": "brown"},
            {"type": "word", "start": 16, "end": 19, "start_time": 811, "end_time": 1280, "value": "fox"},
        ]
    }
    text = "The quick brown fox"
    frames = marks_to_token_frames(marks, text, 24000, 256)
    assert frames is not None and len(frames) == len(text), "one duration per character, or the char tokenizer misaligns"
    assert all(f >= 1 for f in frames)
    total = sum(frames)
    expected = 1.280 * 24000 / 256
    assert abs(total - expected) / expected < 0.1, (total, expected)


def test_marks_absent_means_no_alignment():
    assert marks_to_token_frames(None, "hello", 24000, 256) is None
    assert marks_to_token_frames({"chunks": []}, "hello", 24000, 256) is None
    assert marks_to_token_frames({"chunks": [{"type": "word"}]}, "hello", 24000, 256) is None


def test_synthesize_decodes_wav_and_counts_characters(backend, monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(request, timeout=0):
        calls["n"] += 1
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["voice_id"] == "geffen_32"
        assert payload["audio_format"] == "wav"
        assert request.headers["Authorization"].startswith("Bearer ")
        return _fake_response(
            {
                "audio_data": base64.b64encode(_wav_bytes()).decode("ascii"),
                "audio_format": "wav",
                "billable_characters_count": len(payload["input"]),
                "speech_marks": {"type": "sentence", "chunks": [
                    {"type": "word", "start": 0, "end": 5, "start_time": 0, "end_time": 400, "value": "Hello"}
                ]},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    wav, rate, marks = backend.synthesize_with_marks("Hello there friend", "geffen_32")
    assert rate == 48000
    assert wav.ndim == 1 and wav.size > 0 and np.isfinite(wav).all()
    assert abs(float(np.abs(wav).max()) - 8000 / 32768) < 0.05, "int16 scaling is applied"
    assert marks and marks["chunks"][0]["value"] == "Hello"
    assert backend.characters_sent == len("Hello there friend")
    assert backend.requests == 1 and calls["n"] == 1


def test_transient_failure_is_retried_then_succeeds(backend, monkeypatch):
    calls = {"n": 0}

    def flaky_urlopen(request, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("network flake")
        return _fake_response(
            {"audio_data": base64.b64encode(_wav_bytes()).decode("ascii"), "audio_format": "wav"}
        )

    monkeypatch.setattr("urllib.request.urlopen", flaky_urlopen)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    wav, rate, _marks = backend.synthesize_with_marks("retry me")
    assert calls["n"] == 2 and wav.size > 0 and backend.failures == 1


def test_auth_failure_is_not_retried(backend, monkeypatch):
    import urllib.error

    calls = {"n": 0}

    def unauthorized(request, timeout=0):
        calls["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr("urllib.request.urlopen", unauthorized)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="401"):
        backend.synthesize_with_marks("no")
    assert calls["n"] == 1, "a bad key will not become good by asking again"


def test_the_key_never_reaches_a_corpus_file(backend, monkeypatch, tmp_path):
    """A corpus manifest records the licence, not the credential."""
    from parakeet.data.teacher import synthesize_corpus

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=0: _fake_response(
            {
                "audio_data": base64.b64encode(_wav_bytes()).decode("ascii"),
                "audio_format": "wav",
                "speech_marks": {"chunks": [
                    {"type": "word", "start": 0, "end": 5, "start_time": 0, "end_time": 300, "value": "Hello"}
                ]},
            }
        ),
    )
    manifest = synthesize_corpus(
        ["Hello there"], tmp_path, mix={"speechify": 1.0},
        voices={"speechify": ["geffen_32"]}, backends={"speechify": backend},
    )
    written = manifest.read_text(encoding="utf-8") + (tmp_path / "corpus_meta.json").read_text(encoding="utf-8")
    assert backend.api_key not in written
    assert "sk_" not in written
    record = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert record["token_frames"] and len(record["token_frames"]) == len("Hello there"), (
        "the teacher's own alignment must travel with the record"
    )
    assert TEACHERS["speechify"].weights_license[:20] in record["license"]
