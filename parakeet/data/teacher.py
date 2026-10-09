"""Teacher corpus construction -- the "mix training" data stage.

Two teachers, two very different signals, one student grid
----------------------------------------------------------
* **Orpheus** (Canopy Labs, Llama-3.2-3B + SNAC 24 kHz) is an autoregressive codec LM: it can
  emit its own discrete codes, and it exposes expressive inline tags.  Apache-2.0 code, gated
  weights over a Llama-3.2 base.
* **MiniMax speech-2.8-turbo** is a closed HTTP API.  It returns **audio only** -- no logits,
  no codes, no intermediate features -- and its own terms bar using MiniMax Voice to develop
  foundation models.  It is therefore *refused by default* (:class:`TeacherLicenseError`) and
  can only be enabled with an explicit acknowledgement flag; see ``docs/LEGAL.md``.
* **Kokoro-82M** (Apache-2.0) is included as the fully permissive third teacher -- it is what
  Paradee distilled, and it keeps the default recipe reproducible without any licence risk.

The mixture is resolved at the *audio* level: every teacher's waveform is re-encoded into
Parakeet's own 24-dim continuous latent by the frozen autoencoder.  That is what makes the
frame-rate mismatch (Orpheus/SNAC 12 Hz superframes with 7 codebooks vs Kokoro 40 Hz frames vs
an arbitrary MiniMax rate) a non-issue -- and it is why we can mix teachers at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import ParakeetConfig
from .text import TAGS, normalize_text


class TeacherLicenseError(RuntimeError):
    """Raised when a teacher may not be used for training data under its terms."""


@dataclass
class TeacherSpec:
    name: str
    kind: str  # local_hf | local_pkg | http_api
    code_license: str
    weights_license: str
    #: may the teacher's *output audio* be used to train a student model?
    allows_training: bool
    sample_rate: int
    supports_tags: Sequence[str] = field(default_factory=tuple)
    notes: str = ""
    #: requires an explicit acknowledgement flag + env var to be used
    restricted: bool = False
    model_id: Optional[str] = None


ORPHEUS = TeacherSpec(
    name="orpheus",
    kind="local_hf",
    code_license="Apache-2.0",
    weights_license="apache-2.0 tag, gated download, Llama-3.2-3B-Instruct base -> Meta Llama 3.2 Community License",
    allows_training=True,
    sample_rate=24000,
    supports_tags=(
        "<laugh>", "<chuckle>", "<sigh>", "<cough>", "<sniffle>", "<groan>", "<yawn>", "<gasp>",
    ),
    notes=(
        "Autoregressive over SNAC: 7 codebooks, 1 super-frame per 12 Hz (2048 samples, 83.3 ms), "
        "~84 audio tokens/s of speech, 8 English voices. Canopy warns that training purely on "
        "synthetic audio degrades codebook utilisation -- so teacher audio is used for "
        "capability/render distillation, not for bulk pretraining of a code predictor."
    ),
    model_id="canopylabs/orpheus-3b-0.1-ft",
)

KOKORO = TeacherSpec(
    name="kokoro",
    kind="local_pkg",
    code_license="Apache-2.0",
    weights_license="Apache-2.0",
    allows_training=True,
    sample_rate=24000,
    supports_tags=(),
    notes=(
        "StyleTTS2 + ISTFTNet, 82M, 54 voices, 40 Hz frame rate. The teacher Paradee distilled "
        "(af_heart, 12k WikiText-103 sentences -> 23.9 h). Needs espeak-ng/misaki for G2P."
    ),
    model_id="hexgrad/Kokoro-82M",
)

MINIMAX = TeacherSpec(
    name="minimax",
    kind="http_api",
    code_license="proprietary (API only)",
    weights_license="proprietary; outputs are user-owned but granted back for service improvement",
    allows_training=False,
    sample_rate=32000,
    supports_tags=(
        "<laugh>", "<chuckle>", "<sigh>", "<cough>", "<sniffle>", "<groan>", "<yawn>",
        "<gasp>", "<breath>", "<whisper>",
    ),
    notes=(
        "MiniMax Voice terms bar 'developing foundation models using MiniMax Voice' without prior "
        "written permission, and bar creating derivative works of the service. Its 19 interjection "
        "tags are still usable as a *taxonomy* for our own labelling. Audio-only: no codes/logits."
    ),
    restricted=True,
    model_id="speech-2.8-turbo",
)

# --------------------------------------------------------------------------------------
# Synthetic fixtures
#
# These are NOT teachers.  They render deterministic, speech-like audio from the text so that the
# whole data path -- corpus -> manifest -> latent cache -> training -> synthesis -- can be exercised
# with no network access, no teacher weights and no heavy dependencies.  They are deliberately
# absent from DEFAULT_MIX: a model trained on them is a plumbing test, never a speech model.
# --------------------------------------------------------------------------------------
STUB_LOW = TeacherSpec(
    name="stub_low",
    kind="local_fixture",
    code_license="n/a -- synthetic fixture, not a real teacher",
    weights_license="n/a -- synthetic fixture, not a real teacher",
    allows_training=True,
    sample_rate=24000,
    notes=(
        "SYNTHETIC FIXTURE (low pitch). Renders formant-shaped harmonic stacks from the text so the "
        "data path can be tested offline. Never ship a model trained on this as a real TTS system."
    ),
)

STUB_HIGH = TeacherSpec(
    name="stub_high",
    kind="local_fixture",
    code_license="n/a -- synthetic fixture, not a real teacher",
    weights_license="n/a -- synthetic fixture, not a real teacher",
    allows_training=True,
    sample_rate=24000,
    notes=(
        "SYNTHETIC FIXTURE (high pitch). Same purpose as stub_low; the pair exists so a *mixture* "
        "can be exercised end to end without any external teacher."
    ),
)

SPEECHIFY = TeacherSpec(
    name="speechify",
    kind="http_api",
    code_license="proprietary service (HTTP API)",
    weights_license=(
        "proprietary hosted service.  Used under permission the operator obtained in writing from the "
        "provider, stated to cover demonstration/quantization purposes for this task"
    ),
    allows_training=True,
    sample_rate=48000,
    notes=(
        "Speechify speech API (`simba-3.2`), 48 kHz WAV, and -- uniquely among the teachers wired up "
        "here -- it returns **speech marks**: per-word character offsets with millisecond timings.  "
        "That is the alignment this project has otherwise never had: every duration target so far came "
        "from the unaligned fallback, which rounds 19 and 22 measured to be a real quality limit.  "
        "LICENCE DISTINCTION, recorded rather than smoothed over: the operator's written permission is "
        "stated for *demonstration/quantization*; publishing derivative weights trained on this audio "
        "is a broader use than that wording establishes, so the model card lists it explicitly and "
        "docs/LEGAL.md should be revisited before any release.  Measured DNSMOS on one sample: 3.19 "
        "(P.835 overall) against Kokoro's 2.86 corpus mean."
    ),
    model_id="simba-3.2",
)

TEACHERS: Dict[str, TeacherSpec] = {
    t.name: t for t in (ORPHEUS, KOKORO, MINIMAX, SPEECHIFY, STUB_LOW, STUB_HIGH)
}

#: the English voices this workspace can reach, from ``GET /v1/voices`` (round 26).  Recorded rather
#: than guessed: the API rejects an unknown id with ``voice_not_found`` and points at that endpoint.
#: They are mixed-gender and two locales (en-GB + en-US), which the all-``af_*`` Kokoro bundle is not.
SPEECHIFY_ENGLISH_VOICES: Sequence[str] = (
    "alfonso", "alicia", "alec", "alton", "amon", "geffen",
)

#: fixtures are excluded from the default mixture on purpose
DEFAULT_MIX: Dict[str, float] = {"orpheus": 0.6, "kokoro": 0.4}

#: voice presets for the fixtures (so a mixture of two fixtures is a one-word change)
STUB_PRESETS: Dict[str, Dict[str, float]] = {
    "stub_low": {"f0": 95.0},
    "stub_high": {"f0": 205.0},
}

#: per-voice pitch multipliers for the fixtures: a multi-voice fixture corpus needs voices that are
#: actually distinguishable, otherwise voice conditioning cannot be tested
STUB_VOICE_PITCH: Dict[str, float] = {
    "low": 0.85,
    "mid": 1.0,
    "high": 1.6,
    "v0": 0.9,
    "v1": 1.0,
    "v2": 1.35,
}


def resolve_mix(spec: Optional[Sequence[str]] = None) -> Dict[str, float]:
    if not spec:
        return dict(DEFAULT_MIX)
    weights: Dict[str, float] = {}
    for item in spec:
        if "=" in item:
            name, w = item.split("=", 1)
            weights[name] = float(w)
        else:
            weights[item] = 1.0
    return weights


def check_teacher(name: str, acknowledge_restricted: bool = False) -> TeacherSpec:
    """Gate a teacher by licence.  Raises :class:`TeacherLicenseError` when disallowed."""
    if name not in TEACHERS:
        raise KeyError(f"unknown teacher {name!r}; known: {sorted(TEACHERS)}")
    spec = TEACHERS[name]
    if spec.restricted and not (
        acknowledge_restricted or os.environ.get("PARAKEET_ACCEPT_TEACHER_TOS") == "1"
    ):
        raise TeacherLicenseError(
            f"Teacher {name!r} is refused by default: {spec.notes} "
            "Pass acknowledge_restricted=True (or set PARAKEET_ACCEPT_TEACHER_TOS=1) only if you "
            "have written permission from the provider. See docs/LEGAL.md."
        )
    return spec


# --------------------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------------------
class TeacherBackend:
    """Interface every teacher implements."""

    spec: TeacherSpec

    def synthesize(self, text: str, voice: Optional[str] = None) -> Tuple[np.ndarray, int]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class StubTeacherBackend(TeacherBackend):
    """Deterministic, dependency-free speech-*like* renderer, for plumbing tests only.

    One formant-shaped harmonic token per character.  Duration and pitch come from a hash of the
    text, so a corpus is reproducible without any RNG state, and the same text always renders the
    same audio.  Different ``f0`` presets give different "voices", which is what makes a mixture of
    two stubs meaningful for testing the mixture machinery.
    """

    spec = STUB_LOW  # overridden per registered name in BACKENDS

    def __init__(
        self,
        f0: float = 110.0,
        sample_rate: int = 24000,
        frames_per_token: Tuple[int, int] = (6, 12),
        seed: int = 0,
        voice_pitch: Optional[Dict[str, float]] = None,
    ) -> None:
        self.f0 = float(f0)
        self.sample_rate = int(sample_rate)
        self.frames_per_token = frames_per_token
        self.seed = int(seed)
        #: pitch multiplier per voice name, so a multi-voice fixture corpus is genuinely
        #: multi-voice (which is what makes voice conditioning testable)
        self.voice_pitch = dict(voice_pitch or STUB_VOICE_PITCH)

    def pitch_for_voice(self, voice: Optional[str]) -> float:
        if not voice:
            return 1.0
        if voice in self.voice_pitch:
            return float(self.voice_pitch[voice])
        # unknown voice names still get a stable, distinct pitch rather than silently collapsing
        return 1.0 + 0.2 * (self._stable_hash(voice) % 5)

    def _stable_hash(self, value: str) -> int:
        """``hash()`` is randomised per process for strings, which would make a corpus
        irreproducible across runs; crc32 is stable."""
        import zlib

        return zlib.crc32(f"{self.seed}:{value}".encode("utf-8"))

    def _token_frames(self, char: str) -> int:
        lo, hi = self.frames_per_token
        return lo + (self._stable_hash(char) % (hi - lo + 1))

    def synthesize(self, text: str, voice: Optional[str] = None) -> Tuple[np.ndarray, int]:
        from .synthetic import FORMANTS, render_token

        from ..config import AudioConfig

        hop = AudioConfig(sample_rate=self.sample_rate).hop_length
        text = text.strip() or "a"
        tokens = [c for c in text.lower() if not c.isspace()]
        generator = torch.Generator().manual_seed(self._stable_hash(text) % (2**31))
        base_f0 = self.f0 * self.pitch_for_voice(voice)
        chunks: List[torch.Tensor] = []
        for i, char in enumerate(tokens):
            # gentle, *bounded* declination: an unbounded 2%/token over a 40-character sentence
            # sweeps down to 0.16x the base pitch (24 Hz for the high voice), which is outside any
            # pitch tracker's range and silently corrupts every F0 target in the corpus
            declination = 1.0 - min(0.15, 0.005 * i)
            f0 = base_f0 * declination
            f1, f2 = FORMANTS[(ord(char) + i) % len(FORMANTS)]
            n_samples = self._token_frames(char) * hop
            chunks.append(
                render_token(f0, f1, f2, n_samples, self.sample_rate, generator=generator)
            )
        wav = torch.cat(chunks) if chunks else torch.zeros(hop, dtype=torch.float32)
        peak = wav.abs().max().clamp_min(1e-6)
        return (wav / peak * 0.3).numpy().astype(np.float32), self.sample_rate


class SherpaKokoroBackend(TeacherBackend):
    """Kokoro-82M through **sherpa-onnx** -- the runtime that actually installs on this machine.

    The `kokoro` pip package depends on ``misaki[en]`` -> ``spacy`` -> ``blis``, which has no wheel
    for this Python and no Rust toolchain to build from source, so that path is closed here.
    sherpa-onnx ships prebuilt wheels, carries its own espeak-ng G2P, and runs the *same*
    Apache-2.0 Kokoro-82M weights (``hexgrad/Kokoro-82M``), so the teacher and its licence are
    unchanged -- only the runtime is.

    Voices are the 11 English speaker embeddings in the ``kokoro-en-v0_19`` bundle; the id order is
    the canonical one documented at
    https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html (the first 11 entries of
    the 53-speaker map).  A name outside the bundle raises with the available list rather than
    silently synthesising with the wrong speaker.
    """

    spec = KOKORO
    EN_V0_19_VOICES: Tuple[str, ...] = (
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
        "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    )
    DEFAULT_DIR = "data/teachers/kokoro-en-v0_19"
    DOWNLOAD_HINT = (
        "download the Apache-2.0 bundle with:\n"
        "  curl -L -o kokoro-en-v0_19.tar.bz2 "
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2\n"
        "  tar xf kokoro-en-v0_19.tar.bz2 -C data/teachers/\n"
        "or point PARAKEET_KOKORO_DIR at an extracted bundle"
    )

    def __init__(
        self,
        model_dir: Optional[str] = None,
        num_threads: int = 2,
        provider: str = "cpu",
        voices: Optional[Sequence[str]] = None,
    ) -> None:
        import os

        import sherpa_onnx  # type: ignore

        directory = Path(
            model_dir or os.environ.get("PARAKEET_KOKORO_DIR") or self.DEFAULT_DIR
        )
        required = {
            "model.onnx": directory / "model.onnx",
            "voices.bin": directory / "voices.bin",
            "tokens.txt": directory / "tokens.txt",
            "espeak-ng-data": directory / "espeak-ng-data",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Kokoro bundle incomplete at {directory}: missing {missing}.\n{self.DOWNLOAD_HINT}"
            )
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(required["model.onnx"]),
                    voices=str(required["voices.bin"]),
                    tokens=str(required["tokens.txt"]),
                    data_dir=str(required["espeak-ng-data"]),
                ),
                num_threads=num_threads,
                provider=provider,
                debug=False,
            ),
            max_num_sentences=1,
        )
        self.tts = sherpa_onnx.OfflineTts(config)
        self.sample_rate = int(self.tts.sample_rate)
        self.model_dir = directory
        self.voices = tuple(voices or self.EN_V0_19_VOICES)

    def _speaker_id(self, voice: Optional[str]) -> int:
        if not voice:
            return 0
        if voice in self.voices:
            return self.voices.index(voice)
        raise ValueError(
            f"voice {voice!r} is not in this bundle (available: {list(self.voices)}).  The "
            "54-speaker kokoro-multi-lang bundles have more; check the speaker map before adding one."
        )

    def synthesize(self, text: str, voice: str = "af_heart") -> Tuple[np.ndarray, int]:
        sid = self._speaker_id(voice)
        audio = self.tts.generate(text=normalize_text(text, keep_tags=True), sid=sid, speed=1.0)
        wav = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
        return wav, int(audio.sample_rate)

    def durations(self, text: str, voice: str = "af_heart") -> Optional[List[float]]:
        """sherpa-onnx does not expose Kokoro's per-token durations; reported as unavailable.

        The ``kokoro`` pip pipeline does, which is why :meth:`KokoroBackend.durations` exists -- but
        returning invented timings would be worse than returning None, and the cache falls back to
        the uniform split and says so.
        """
        return None


def marks_to_token_frames(
    marks: Optional[Dict], text: str, sample_rate: int, hop_length: int, n_tokens: Optional[int] = None
) -> Optional[List[int]]:
    """Word timings from Speechify's ``speech_marks`` -> **per-token frame counts**.

    This is the alignment the pipeline never had.  Every duration target before this came from
    :func:`extract_signals`'s unaligned fallback (an even split, or an energy-weighted blend), which
    rounds 19 and 22 both measured to be a real limit on quality -- the student was being taught
    durations that had nothing to do with how the teacher pronounced the sentence.

    ``marks`` carries word entries with character ``start``/``end`` offsets and millisecond
    ``start_time``/``end_time``.  Each word's span is extended to the midpoint of the silence on either
    side (so pauses belong to someone), then the word's frames are split across its characters in
    proportion to how many of the *given* token positions fall inside it, so the counts always sum to
    the true number of frames and the token axis keeps its length.
    """
    if not marks:
        return None
    words = marks.get("chunks") if isinstance(marks, dict) else None
    if not words:
        return None
    frames_per_second = sample_rate / max(1, hop_length)
    entries = []
    for word in words:
        try:
            start_char, end_char = int(word["start"]), int(word["end"])
            start_ms, end_ms = float(word["start_time"]), float(word["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_char <= start_char or end_ms <= start_ms:
            continue
        entries.append((start_char, end_char, start_ms, end_ms))
    if not entries:
        return None
    entries.sort(key=lambda e: e[0])

    # one duration per **character** of the text (spaces included): the char tokenizer produces exactly
    # `len(text)` tokens, and a mismatch would silently mis-assign every duration after it
    token_positions = list(range(min(len(text), n_tokens) if n_tokens is not None else len(text)))
    if not token_positions:
        return None

    # boundaries between words: the midpoint of the gap between them
    spans = []
    for index, (start_char, end_char, start_ms, end_ms) in enumerate(entries):
        previous_end = entries[index - 1][3] if index > 0 else 0.0
        next_start = entries[index + 1][2] if index + 1 < len(entries) else end_ms
        left = start_ms if index == 0 else 0.5 * (previous_end + start_ms)
        right = end_ms if index == len(entries) - 1 else 0.5 * (end_ms + next_start)
        spans.append((start_char, end_char, left, right))

    counts: List[int] = []
    for position in token_positions:
        # a character (or the space after a word) belongs to the last word that starts at or before it
        chosen = spans[0]
        for span in spans:
            if span[0] <= position:
                chosen = span
            else:
                break
        counts.append(chosen)
    # convert each token's chosen (left, right) window to frames, sharing the word's frames across its
    # characters so the total matches the audio
    token_windows = [(span[2], span[3]) for span in counts]
    per_word_totals: Dict[Tuple[float, float], int] = {}
    for window in set(token_windows):
        span_ms = max(1.0, window[1] - window[0])
        per_word_totals[window] = max(1, int(round(span_ms / 1000.0 * frames_per_second)))
    per_word_counts: Dict[Tuple[float, float], int] = {}
    for window in token_windows:
        per_word_counts[window] = per_word_counts.get(window, 0) + 1
    token_frames: List[int] = []
    for window in token_windows:
        share = max(1, per_word_counts[window])
        token_frames.append(max(1, int(round(per_word_totals[window] / share))))
    if n_tokens is not None:
        token_frames = token_frames[:n_tokens]
        while len(token_frames) < n_tokens:
            token_frames.append(1)
    return token_frames


class SpeechifyBackend(TeacherBackend):
    """Speechify speech API (``simba-3.2``) -- 48 kHz WAV **plus word-level speech marks**.

    The key is read from ``SPEECHIFY_API_KEY`` or ``.secrets/speechify.key``; it is never written into
    a corpus, a config or a commit (the hygiene test scans tracked files for key-shaped strings).

    Rate limiting and quota matter here: the service bills per character, so the backend records how
    many characters it has sent and reports it, and transient failures are retried with backoff rather
    than aborting a long corpus build.
    """

    spec = SPEECHIFY
    DEFAULT_URL = "https://api.speechify.ai/v1/audio/speech"

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "geffen_32",
        model: str = "simba-3.2",
        url: Optional[str] = None,
        audio_format: str = "wav",
        attempts: int = 3,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("SPEECHIFY_API_KEY") or self._read_secret_file()
        if not self.api_key:
            raise RuntimeError(
                "no Speechify key: set SPEECHIFY_API_KEY or write .secrets/speechify.key "
                "(gitignored).  The service requires the operator's own permission."
            )
        self.default_voice = voice
        self.model = model
        self.url = url or self.DEFAULT_URL
        self.audio_format = audio_format
        self.attempts = max(1, int(attempts))
        self.timeout = timeout
        self.characters_sent = 0
        self.requests = 0
        self.failures = 0
        #: marks for the most recent synthesis (`synthesize_with_marks` returns them explicitly)
        self.last_marks: Optional[Dict] = None

    @staticmethod
    def _read_secret_file() -> Optional[str]:
        path = Path(".secrets/speechify.key")
        if path.exists():
            return path.read_text(encoding="utf-8").strip() or None
        return None

    def _post(self, text: str, voice: Optional[str]) -> Dict:
        import time
        import urllib.error
        import urllib.request

        payload = {
            "input": text,
            "voice_id": voice or self.default_voice,
            "model": self.model,
            "audio_format": self.audio_format,
        }
        last_error: Optional[Exception] = None
        for attempt in range(1, self.attempts + 1):
            request = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                self.requests += 1
                self.characters_sent += int(body.get("billable_characters_count") or len(text))
                return body
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:200]
                last_error = RuntimeError(f"HTTP {exc.code} from Speechify: {detail!r}")
                # 4xx other than rate limiting will not improve by retrying
                if exc.code < 500 and exc.code != 429:
                    break
            except Exception as exc:  # noqa: BLE001 - network flakes are expected on a long build
                last_error = exc
            self.failures += 1
            if attempt < self.attempts:
                time.sleep(min(8.0, 1.5 * attempt))
        raise RuntimeError(f"Speechify request failed after {self.attempts} attempts: {last_error}")

    def synthesize_with_marks(self, text: str, voice: Optional[str] = None) -> Tuple[np.ndarray, int, Optional[Dict]]:
        """Like :meth:`synthesize`, but also returns the word timings when the service provides them."""
        import base64
        import io
        import wave

        body = self._post(text, voice)
        encoded = body.get("audio_data") or body.get("audio")
        if not encoded:
            raise RuntimeError(f"Speechify returned no audio: {str(body)[:300]}")
        raw = base64.b64decode(encoded)
        marks = body.get("speech_marks")
        self.last_marks = marks if isinstance(marks, dict) else None
        fmt = str(body.get("audio_format") or self.audio_format).lower()
        if raw[:4] == b"RIFF" or fmt.startswith("wav"):
            with wave.open(io.BytesIO(raw), "rb") as handle:
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                rate = handle.getframerate()
                frames = handle.readframes(handle.getnframes())
            dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
            data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            scale = float(np.iinfo(np.int16).max if width <= 2 else np.iinfo(np.int32).max)
            data = data / scale
            if channels > 1:
                data = data.reshape(-1, channels).mean(axis=1)
            return data, rate, self.last_marks
        raise RuntimeError(
            f"unsupported Speechify audio format {fmt!r} (ask for wav; mp3 would need a decoder)"
        )

    def synthesize(self, text: str, voice: str = "geffen_32") -> Tuple[np.ndarray, int]:
        wav, rate, _marks = self.synthesize_with_marks(text, voice)
        return wav, rate

    def durations(self, text: str, voice: str = "geffen_32") -> Optional[List[float]]:
        """Not available without a synthesis; the timings come out of :meth:`synthesize_with_marks`."""
        return None


class OrpheusBackend(TeacherBackend):
    """Local Orpheus (Llama-3.2-3B + SNAC 24 kHz) inference.

    Requires ``transformers`` and ``snac``.  Token-id constants below come from the model card;
    :meth:`verify` checks them against the loaded tokenizer and fails loudly rather than
    silently producing noise.
    """

    spec = ORPHEUS
    START_OF_HUMAN = 128259
    END_OF_TEXT = 128009
    END_OF_HUMAN = 128260
    START_OF_SPEECH = 128257
    END_OF_SPEECH = 128258
    AUDIO_BASE = 128266
    N_CODEBOOKS = 7
    CODEBOOK_SIZE = 4096

    def __init__(self, model_id: Optional[str] = None, snac_id: str = "hubertsiuzdak/snac_24khz", device: str = "cpu") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        from snac import SNAC  # noqa: F401

        model_id = model_id or self.spec.model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
        self.snac = SNAC.from_pretrained(snac_id).to(device).eval()
        self.verify()

    def verify(self) -> None:
        for name, value in [
            ("<custom_token_0>", self.START_OF_HUMAN),
        ]:
            try:
                got = self.tokenizer.convert_tokens_to_ids(name)
            except Exception:  # pragma: no cover - depends on tokenizer version
                continue
            if got != value:
                raise RuntimeError(
                    f"Orpheus token layout changed: {name} -> {got}, expected {value}. "
                    "Re-derive the constants from the model card before trusting output."
                )

    @torch.no_grad()
    def synthesize(self, text: str, voice: str = "tara") -> Tuple[np.ndarray, int]:
        prompt = f"{voice}: {normalize_text(text, keep_tags=True)}"
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        wrap = torch.tensor(
            [[self.START_OF_HUMAN, *ids[0].tolist(), self.END_OF_TEXT, self.END_OF_HUMAN]],
            device=self.device,
        )
        out = self.model.generate(
            wrap,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        audio_tokens = [
            t
            for t in out[0].tolist()
            if self.AUDIO_BASE <= t < self.AUDIO_BASE + self.N_CODEBOOKS * self.CODEBOOK_SIZE
        ]
        # 7 codes per super-frame.  The codebook -> SNAC-level mapping is NOT contiguous: the
        # published Orpheus decoder assigns codebooks {0}, {1, 4}, {2, 3, 5, 6} to levels 1, 2, 3
        # (one, two and four codes per super-frame).  Grouping them contiguously ({1,2}, {3..6})
        # looks right and decodes to noise, and nothing in this repo could catch it: the real
        # backend needs a 3B checkpoint, so the path had never been executed.
        n = (len(audio_tokens) // self.N_CODEBOOKS) * self.N_CODEBOOKS
        codes = np.array(audio_tokens[:n], dtype=np.int64).reshape(-1, self.N_CODEBOOKS) - self.AUDIO_BASE
        by_level = [
            torch.tensor(
                codes[:, i] % self.CODEBOOK_SIZE, device=self.device, dtype=torch.int32
            )
            for i in range(self.N_CODEBOOKS)
        ]
        l1 = by_level[0][None]
        l2 = torch.stack([by_level[1], by_level[4]], dim=-1).reshape(1, -1)
        l3 = torch.stack(
            [by_level[2], by_level[3], by_level[5], by_level[6]], dim=-1
        ).reshape(1, -1)
        wav = self.snac.decode([l1, l2, l3])[0].squeeze().float().cpu().numpy()
        return wav, self.spec.sample_rate


class KokoroBackend(TeacherBackend):
    """Local Kokoro-82M (Apache-2.0).  Requires ``kokoro`` (+ ``misaki`` for G2P)."""

    spec = KOKORO

    def __init__(self, lang_code: str = "a", repo_id: Optional[str] = None) -> None:
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code=lang_code, repo_id=repo_id or self.spec.model_id)

    def synthesize(self, text: str, voice: str = "af_heart") -> Tuple[np.ndarray, int]:
        chunks: List[np.ndarray] = []
        for _, _, audio in self.pipeline(normalize_text(text), voice=voice, speed=1.0):
            chunks.append(np.asarray(audio.detach().cpu().numpy()).reshape(-1))
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.spec.sample_rate
        return np.concatenate(chunks), self.spec.sample_rate

    def durations(self, text: str, voice: str = "af_heart") -> Optional[List[float]]:
        """Kokoro exposes its own predicted durations -- Paradee saves these as teacher signals."""
        try:
            out = list(self.pipeline(normalize_text(text), voice=voice, return_durations=True))
        except TypeError:
            return None
        if not out:
            return None
        _, _, durations = out[0][:3]
        return list(durations)


class MiniMaxBackend(TeacherBackend):
    """MiniMax speech-2.8-turbo HTTP API.  Restricted: refuses unless acknowledged.

    Audio-only output means this teacher can only supervise the *acoustic* half (audio/latent
    distribution) -- there are no codes or hidden states to distil from.  See docs/LEGAL.md
    before enabling it.
    """

    spec = MINIMAX
    DEFAULT_URL = "https://api.minimax.io/v1/t2a_v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "speech-2.8-turbo",
        url: Optional[str] = None,
        acknowledge_restricted: bool = False,
    ) -> None:
        check_teacher("minimax", acknowledge_restricted=acknowledge_restricted)
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not self.api_key:
            raise RuntimeError("set MINIMAX_API_KEY (and obtain written permission) to use this backend")
        self.model = model
        self.url = url or self.DEFAULT_URL

    def synthesize(self, text: str, voice: str = "English_expressive_narrator") -> Tuple[np.ndarray, int]:
        import urllib.request

        payload = {
            "model": self.model,
            "text": text,
            "stream": False,
            "voice_setting": {"voice_id": voice, "speed": 1.0, "vol": 1.0, "pitch": 0},
            "audio_setting": {"sample_rate": self.spec.sample_rate, "format": "wav", "channel": 1},
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        hex_audio = body.get("data", {}).get("audio")
        if not hex_audio:
            raise RuntimeError(f"MiniMax returned no audio: {str(body)[:400]}")
        import io
        import wave

        raw = bytes.fromhex(hex_audio)
        with wave.open(io.BytesIO(raw)) as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            wav = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return wav, sr


def _kokoro_runtime() -> type:
    """Prefer the runtime that imports.  Same teacher, same licence, different packaging.

    ``kokoro`` (the official pip package) needs ``misaki[en]``, which needs spacy/blis wheels that do
    not exist for every Python; sherpa-onnx ships prebuilt wheels and bundles its own G2P.  Either
    way the weights are hexgrad/Kokoro-82M under Apache-2.0.
    """
    try:
        import kokoro  # noqa: F401

        return KokoroBackend
    except Exception:  # noqa: BLE001 - any import failure means "not usable here"
        return SherpaKokoroBackend


BACKENDS: Dict[str, type] = {
    "orpheus": OrpheusBackend,
    "speechify": SpeechifyBackend,
    "kokoro": _kokoro_runtime(),
    "minimax": MiniMaxBackend,
    "stub_low": StubTeacherBackend,
    "stub_high": StubTeacherBackend,
}


def build_backend(name: str, **kwargs) -> TeacherBackend:
    check_teacher(name, acknowledge_restricted=kwargs.pop("acknowledge_restricted", False))
    if name in STUB_PRESETS:
        for key, value in STUB_PRESETS[name].items():
            kwargs.setdefault(key, value)
    return BACKENDS[name](**kwargs)


# --------------------------------------------------------------------------------------
# corpus writing
# --------------------------------------------------------------------------------------
def sha1_of_array(wav) -> str:
    """Content hash of a waveform, for corpus provenance (integrity, not security)."""
    import hashlib

    import numpy as np

    array = np.ascontiguousarray(np.asarray(wav, dtype=np.float32))
    return hashlib.sha1(array.tobytes()).hexdigest()


@dataclass
class CorpusRecord:
    utt_id: str
    text: str
    teacher: str
    voice: str
    wav_path: str
    sample_rate: int
    duration_s: float
    tags: List[str] = field(default_factory=list)
    license: str = ""
    quality: Optional[float] = None
    hash: str = ""
    #: per-character frame counts from the teacher's own timings (Speechify speech marks).  When
    #: present, the cache uses them instead of the unaligned fallback -- the first real alignment in
    #: this project.
    token_frames: Optional[List[int]] = None


def synthesize_corpus(
    texts: Iterable[str],
    out_dir: str | Path,
    mix: Optional[Dict[str, float]] = None,
    voices: Optional[Dict[str, Sequence[str]]] = None,
    backends: Optional[Dict[str, TeacherBackend]] = None,
    acknowledge_restricted: bool = False,
    max_utts: Optional[int] = None,
    token_sample_rate: int = 24000,
    token_hop_length: int = 256,
) -> Path:
    """Render a paired teacher corpus and write ``manifest.jsonl`` + wavs.

    ``mix`` maps teacher -> weight; the mixture is realised deterministically by interleaving
    so that every shard of the corpus contains the full mixture (important: shard-local
    balance avoids long stretches of gradient from a single teacher).

    When a backend can report **word timings** (Speechify speech marks), the record carries
    ``token_frames`` -- per-character frame counts at the *student's* rate and hop.  The cache then
    uses the teacher's own alignment instead of the unaligned fallback; ``token_sample_rate`` and
    ``token_hop_length`` are parameters rather than derived from the teacher because the student
    resamples everything to 24 kHz anyway.
    """
    import soundfile as sf

    out_dir = Path(out_dir)
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    mix = mix or DEFAULT_MIX
    backends = backends or {}
    lines: List[str] = []
    teachers = list(mix.keys())
    for name in teachers:
        check_teacher(name, acknowledge_restricted=acknowledge_restricted)
    weights = np.array([mix[t] for t in teachers], dtype=np.float64)
    weights = weights / weights.sum()

    keep_cache: Dict[str, Tuple[np.ndarray, int]] = {}
    for i, text in enumerate(texts):
        if max_utts is not None and i >= max_utts:
            break
        teacher = teachers[int(np.argmax(np.cumsum(weights) > ((i * 0.61803398875) % 1.0)))]
        backend = backends.get(teacher)
        if backend is None:
            continue
        voice_list = (voices or {}).get(teacher) or [None]
        voice = voice_list[i % len(voice_list)]
        marks: Optional[Dict] = None
        try:
            if hasattr(backend, "synthesize_with_marks"):
                # a backend that can align uses it: the timings are the point of asking this teacher
                if voice:
                    wav, sr, marks = backend.synthesize_with_marks(text, voice)
                else:
                    wav, sr, marks = backend.synthesize_with_marks(text)
            else:
                wav, sr = backend.synthesize(text, voice) if voice else backend.synthesize(text)
        except Exception as exc:  # keep the corpus build resumable
            print(f"[teacher] {teacher} failed on {i}: {exc}")
            continue
        if wav.size == 0:
            continue
        utt_id = f"{teacher}_{i:07d}"
        path = out_dir / "wav" / f"{utt_id}.wav"
        sf.write(str(path), wav, sr)
        tags = [t for t in TAGS if t in text.lower()]
        rec = CorpusRecord(
            utt_id=utt_id,
            text=text,
            teacher=teacher,
            voice=voice or "",
            wav_path=str(path.relative_to(out_dir)),
            sample_rate=sr,
            duration_s=float(wav.shape[0] / sr),
            tags=tags,
            license=TEACHERS[teacher].weights_license,
            hash=sha1_of_array(wav),
            token_frames=marks_to_token_frames(
                marks, text, token_sample_rate, token_hop_length
            ),
        )
        lines.append(json.dumps(asdict(rec), ensure_ascii=False))
        keep_cache[utt_id] = (wav, sr)
        if (i + 1) % 100 == 0:
            print(f"[teacher] {i+1} utterances, {sum(r['duration_s'] for r in map(json.loads, lines))/3600:.2f} h")

    manifest = out_dir / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    meta = {
        "mix": mix,
        "teachers": {k: asdict(TEACHERS[k]) for k in mix},
        "n_utterances": len(lines),
        "hours": sum(json.loads(l)["duration_s"] for l in lines) / 3600.0 if lines else 0.0,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "corpus_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return manifest
