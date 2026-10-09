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

TEACHERS: Dict[str, TeacherSpec] = {
    t.name: t for t in (ORPHEUS, KOKORO, MINIMAX, STUB_LOW, STUB_HIGH)
}

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


BACKENDS: Dict[str, type] = {
    "orpheus": OrpheusBackend,
    "kokoro": KokoroBackend,
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


def synthesize_corpus(
    texts: Iterable[str],
    out_dir: str | Path,
    mix: Optional[Dict[str, float]] = None,
    voices: Optional[Dict[str, Sequence[str]]] = None,
    backends: Optional[Dict[str, TeacherBackend]] = None,
    acknowledge_restricted: bool = False,
    max_utts: Optional[int] = None,
) -> Path:
    """Render a paired teacher corpus and write ``manifest.jsonl`` + wavs.

    ``mix`` maps teacher -> weight; the mixture is realised deterministically by interleaving
    so that every shard of the corpus contains the full mixture (important: shard-local
    balance avoids long stretches of gradient from a single teacher).
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
        try:
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
