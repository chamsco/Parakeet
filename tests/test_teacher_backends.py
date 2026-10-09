"""Contract tests for the three *real* teacher backends, with injected stubs.

These are the only code paths in the repository that had never been executed by anything: they need
a 3B checkpoint, a 82M model or a paid API, so no demo, test or CI job ever ran a line of them.  The
working tree could therefore be confidently wrong about them -- and it was.

`OrpheusBackend` grouped SNAC codebooks contiguously when mapping the 7 codes per super-frame to the
three SNAC levels.  The published Orpheus decoder assigns codebooks ``{0}``, ``{1, 4}``,
``{2, 3, 5, 6}``.  Contiguous grouping looks plausible, passes every shape check, and decodes to
noise: every corpus built from Orpheus would have been garbage.

So the central test here does not check "shapes look fine" -- it reimplements the published
algorithm and asserts element-wise equality against the implementation, on a token stream with
distinguishable codes.
"""

import io
import json
import wave

import numpy as np
import pytest
import torch

from parakeet.data.teacher import (
    KokoroBackend,
    MiniMaxBackend,
    OrpheusBackend,
    TeacherLicenseError,
)

# --------------------------------------------------------------------------------------
# Orpheus
# --------------------------------------------------------------------------------------
N_CODEBOOKS = 7
CODEBOOK_SIZE = 4096
AUDIO_BASE = 128266


def published_redistribute(code_list):
    """The reference implementation from the Orpheus decoder (`redistribute_codes`).

    ``code_list`` is the flat list of audio-token ids already offset by ``AUDIO_BASE``.
    Returns ``(layer_1, layer_2, layer_3)`` as flat lists.

    Cross-checked against two independent copies of the published decoder:
    https://github.com/CrispStrobe/CrispTTS/blob/main/decoder.py and the canopyai Orpheus-TTS
    decoder (see also https://deepwiki.com/canopyai/Orpheus-TTS/2.2-audio-decoding).  Both assign
    codebooks ``{0}``, ``{1,4}``, ``{2,3,5,6}`` to the three SNAC levels and flatten each level
    frame-major -- which is what this reproduces.
    """
    layer_1, layer_2, layer_3 = [], [], []
    for i in range((len(code_list) + 7 - 1) // 7):
        layer_1.append(code_list[7 * i])
        layer_2.append(code_list[7 * i + 1] - 4096)
        layer_3.append(code_list[7 * i + 2] - (2 * 4096))
        layer_3.append(code_list[7 * i + 3] - (3 * 4096))
        layer_2.append(code_list[7 * i + 4] - (4 * 4096))
        layer_3.append(code_list[7 * i + 5] - (5 * 4096))
        layer_3.append(code_list[7 * i + 6] - (6 * 4096))
    return layer_1, layer_2, layer_3


class _Ids:
    def __init__(self, ids):
        self.input_ids = torch.tensor([ids], dtype=torch.long)


class _StubTokenizer:
    def convert_tokens_to_ids(self, token):
        return OrpheusBackend.START_OF_HUMAN

    def __call__(self, prompt, return_tensors="pt"):
        # a few text ids followed by the delimiter tokens the backend adds
        return _Ids([10, 11, 12])


class _StubModel:
    def __init__(self, generated):
        self.generated = generated
        self.seen = None

    def generate(self, wrap, **kwargs):
        self.seen = {"wrap": wrap.tolist(), "kwargs": kwargs}
        return torch.tensor([list(self.generated)], dtype=torch.long)


class _RecordingSNAC:
    def __init__(self):
        self.levels = None

    def decode(self, codes):
        assert isinstance(codes, list) and len(codes) == 3, "SNAC takes three levels"
        self.levels = [c.clone() for c in codes]
        # return (B, 1, N) like the real decoder
        return torch.zeros(codes[0].shape[0], 1, 24000)


def _orpheus_backend(n_frames=5):
    """An OrpheusBackend with the model/tokenizer/SNAC replaced by recording stubs."""
    audio = []
    for frame in range(n_frames):
        for cb in range(N_CODEBOOKS):
            audio.append(AUDIO_BASE + cb * CODEBOOK_SIZE + (frame * 7 + cb) % CODEBOOK_SIZE)
    generated = [7, 8] + audio + [OrpheusBackend.END_OF_SPEECH, 99]
    backend = OrpheusBackend.__new__(OrpheusBackend)
    backend.device = "cpu"
    backend.tokenizer = _StubTokenizer()
    backend.model = _StubModel(generated)
    backend.snac = _RecordingSNAC()
    return backend, generated


def test_orpheus_codebook_to_level_mapping_matches_the_published_decoder():
    """The regression that matters: contiguous grouping decodes to noise."""
    backend, generated = _orpheus_backend(n_frames=5)
    wav, sr = backend.synthesize("hello <laugh>", voice="tara")

    audio_tokens = [t for t in generated if AUDIO_BASE <= t < AUDIO_BASE + N_CODEBOOKS * CODEBOOK_SIZE]
    expected_1, expected_2, expected_3 = published_redistribute([t - AUDIO_BASE for t in audio_tokens])
    got_1, got_2, got_3 = backend.snac.levels

    assert got_1.reshape(-1).tolist() == expected_1, "level 1 is codebook 0 (one code per frame)"
    assert got_2.reshape(-1).tolist() == expected_2, "level 2 must be codebooks {1, 4}"
    assert got_3.reshape(-1).tolist() == expected_3, "level 3 must be codebooks {2, 3, 5, 6}"
    assert sr == 24000
    assert wav.shape[-1] == 24000


def test_contiguous_grouping_would_fail_the_mapping_check():
    """Positive control: the buggy grouping cannot satisfy the published mapping.

    Without this, the mapping test above would prove nothing about whether it has teeth.
    """
    generated = []
    for frame in range(4):
        for cb in range(N_CODEBOOKS):
            generated.append(AUDIO_BASE + cb * CODEBOOK_SIZE + (frame * 7 + cb) % CODEBOOK_SIZE)
    flat = [t - AUDIO_BASE for t in generated]
    _expected_1, expected_2, expected_3 = published_redistribute(flat)

    # what the previous implementation produced: contiguous groups {1,2} and {3,4,5,6}
    contiguous_2 = [
        value % CODEBOOK_SIZE for i in range(4) for value in (flat[7 * i + 1], flat[7 * i + 2])
    ]
    contiguous_3 = [
        value % CODEBOOK_SIZE
        for i in range(4)
        for value in (flat[7 * i + 3], flat[7 * i + 4], flat[7 * i + 5], flat[7 * i + 6])
    ]
    assert contiguous_2 != expected_2, "{1,2} must not equal the published level-2 codes"
    assert contiguous_3 != expected_3, "{3..6} must not equal the published level-3 codes"
    assert sorted(expected_2[:2]) != sorted(contiguous_2[:2]), (
        "level 2 uses codebooks {1,4}, not {1,2}"
    )


def test_orpheus_level_shapes_follow_the_stride_pattern():
    backend, _ = _orpheus_backend(n_frames=6)
    backend.synthesize("hello")
    l1, l2, l3 = backend.snac.levels
    assert l1.shape == (1, 6), "one coarse code per super-frame"
    assert l2.shape == (1, 12), "two codes per super-frame"
    assert l3.shape == (1, 24), "four codes per super-frame"


def test_orpheus_prompt_wrapping_and_generation_settings():
    """The chat wrapper and sampling settings are part of the teacher's contract."""
    backend, _ = _orpheus_backend(n_frames=2)
    backend.synthesize("hello world", voice="leo")
    wrap = backend.model.seen["wrap"][0]
    assert wrap[0] == OrpheusBackend.START_OF_HUMAN
    assert wrap[-2:] == [OrpheusBackend.END_OF_TEXT, OrpheusBackend.END_OF_HUMAN]
    assert backend.model.seen["kwargs"]["do_sample"] is True
    assert backend.model.seen["kwargs"]["temperature"] == pytest.approx(0.6)
    assert backend.model.seen["kwargs"]["repetition_penalty"] == pytest.approx(1.1)


def test_orpheus_ignores_partial_superframes():
    """A trailing partial group of 7 must be dropped, not reshaped into a bogus frame."""
    backend, generated = _orpheus_backend(n_frames=3)
    generated = generated[:-2] + [AUDIO_BASE + 5]  # 3*7 + 1 audio tokens
    backend.model.generated = generated
    backend.snac = _RecordingSNAC()
    backend.synthesize("hello")
    assert backend.snac.levels[0].shape == (1, 3), "the partial super-frame is dropped"


def test_orpheus_non_audio_tokens_are_filtered_out():
    backend, _ = _orpheus_backend(n_frames=4)
    backend.model.generated = [1, 2, AUDIO_BASE - 1, AUDIO_BASE + 7 * CODEBOOK_SIZE, 5] + [
        AUDIO_BASE + cb for cb in range(N_CODEBOOKS)
    ]
    backend.snac = _RecordingSNAC()
    backend.synthesize("hello")
    assert backend.snac.levels[0].shape == (1, 1), "only in-range audio tokens count"


# --------------------------------------------------------------------------------------
# Kokoro
# --------------------------------------------------------------------------------------
class _StubKokoroPipeline:
    """Mimics `KPipeline.__call__`, which yields (graphemes, phonemes, audio) triples."""

    def __init__(self, chunks=(3, 4)):
        self.chunks = chunks
        self.calls = []

    def __call__(self, text, voice=None, speed=1.0, return_durations=False, **kwargs):
        self.calls.append({"text": text, "voice": voice, "speed": speed,
                           "return_durations": return_durations})
        if return_durations:
            yield ("g", "p", [0.1, 0.2, 0.3])
            return
        for n in self.chunks:
            yield ("g", "p", torch.arange(n, dtype=torch.float32) / 10.0)


def test_kokoro_concatenates_chunks_and_reports_its_rate():
    backend = KokoroBackend.__new__(KokoroBackend)
    backend.pipeline = _StubKokoroPipeline(chunks=(3, 4))
    wav, sr = backend.synthesize("hello world", voice="af_heart")
    assert wav.shape == (7,), "chunks must be concatenated, not stacked or dropped"
    assert sr == backend.spec.sample_rate
    assert backend.pipeline.calls[0]["voice"] == "af_heart"
    assert backend.pipeline.calls[0]["speed"] == 1.0


def test_kokoro_empty_output_is_an_empty_array_not_a_crash():
    backend = KokoroBackend.__new__(KokoroBackend)
    backend.pipeline = _StubKokoroPipeline(chunks=())
    wav, sr = backend.synthesize("", voice="af_heart")
    assert wav.shape[0] == 0
    assert sr == backend.spec.sample_rate


def test_kokoro_durations_pass_through_and_degrade_to_none():
    """Kokoro is the one teacher that exposes its own durations (Paradee's teacher signals)."""
    backend = KokoroBackend.__new__(KokoroBackend)
    backend.pipeline = _StubKokoroPipeline()
    assert backend.durations("hello") == [0.1, 0.2, 0.3]

    class _NoDurations(_StubKokoroPipeline):
        def __call__(self, text, **kwargs):
            raise TypeError("return_durations unsupported")

    backend.pipeline = _NoDurations()
    assert backend.durations("hello") is None, "must degrade rather than raise"


# --------------------------------------------------------------------------------------
# MiniMax (legally gated; the code path still has to be correct)
# --------------------------------------------------------------------------------------
def _wav_bytes(samples, sr=32000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return buffer.getvalue()


def test_minimax_is_refused_without_acknowledgement(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    with pytest.raises(TeacherLicenseError):
        MiniMaxBackend(api_key="test-key")


def test_minimax_request_payload_and_hex_audio_decode(monkeypatch):
    """Verifies the request shape and the response decode without calling the API."""
    import urllib.request

    raw = _wav_bytes([0, 16384, -16384, 32767])
    body = {"data": {"audio": raw.hex()}, "base_resp": {"status_code": 0}}
    seen = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(body).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.headers)
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    backend = MiniMaxBackend(api_key="test-key", acknowledge_restricted=True)
    wav, sr = backend.synthesize("hello", voice="English_expressive_narrator")

    assert seen["url"] == MiniMaxBackend.DEFAULT_URL
    assert seen["headers"].get("Authorization") == "Bearer test-key"
    assert seen["payload"]["model"] == "speech-2.8-turbo"
    assert seen["payload"]["text"] == "hello"
    assert seen["payload"]["voice_setting"]["voice_id"] == "English_expressive_narrator"
    assert seen["payload"]["audio_setting"]["sample_rate"] == 32000
    assert seen["payload"]["audio_setting"]["format"] == "wav"
    assert seen["payload"]["stream"] is False

    assert sr == 32000
    assert wav.shape == (4,)
    assert wav[1] == pytest.approx(0.5, abs=1e-3), "int16 must be scaled to [-1, 1]"


def test_minimax_missing_audio_raises_with_the_response(monkeypatch):
    import urllib.request

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"base_resp": {"status_code": 1004, "status_msg": "auth failed"}}).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Response())
    backend = MiniMaxBackend(api_key="k", acknowledge_restricted=True)
    with pytest.raises(RuntimeError, match="no audio"):
        backend.synthesize("hello")


def test_minimax_requires_a_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
        MiniMaxBackend(api_key=None, acknowledge_restricted=True)
