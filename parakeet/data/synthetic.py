"""Structured synthetic utterances — a fixture generator for CPU learning experiments.

This is **not data**. It exists so that the question "does this training code actually learn?"
can be answered on a laptop with no corpus, no GPU and no network.

The signals are deliberately speech-*like* and fully structured: per-character tokens with known
frame counts, a known F0 contour and measured energy. That means the cached teacher-signal
targets used by the Tiny distillation stages are exact rather than estimated by an aligner, which
is what makes an end-to-end learning assertion meaningful instead of noisy.

Signal model per token: a harmonic stack at a token-specific F0, shaped by two formant
resonances and a gentle spectral tilt, with a raised-cosine attack/release envelope. F0 declines
slightly across the utterance, as in real speech.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from ..audio.f0 import frame_energy_db
from ..config import AudioConfig

#: (F1, F2) pairs roughly at the vowels /a/, /e/, /i/, /o/, /u/, /ə/
FORMANTS: Tuple[Tuple[float, float], ...] = (
    (730.0, 1090.0),
    (530.0, 1840.0),
    (270.0, 2290.0),
    (570.0, 840.0),
    (300.0, 870.0),
    (500.0, 1500.0),
)

#: two duration layouts so that per-token duration prediction is actually exercised
LAYOUTS: Dict[str, Tuple[int, ...]] = {
    "short": (10, 8, 12, 9, 11, 8, 10, 12),
    "long": (12, 10, 14, 11, 13, 10, 12, 14),
}


@dataclass
class SyntheticUtterance:
    """One synthetic utterance with exact token-level ground truth."""

    text: str
    layout: str
    wav: torch.Tensor  # (N,)
    token_frames: List[int]
    token_f0: List[float]
    token_energy_db: List[float]

    @property
    def n_frames(self) -> int:
        return int(sum(self.token_frames))

    @property
    def n_tokens(self) -> int:
        return len(self.token_frames)


def _formant_gain(freqs: torch.Tensor, f1: float, f2: float, bw: float = 110.0) -> torch.Tensor:
    g1 = torch.exp(-0.5 * ((freqs - f1) / bw) ** 2)
    g2 = torch.exp(-0.5 * ((freqs - f2) / (bw * 1.4)) ** 2)
    return 0.7 * g1 + 0.5 * g2 + 0.03


def render_token(
    f0: float,
    f1: float,
    f2: float,
    n_samples: int,
    sample_rate: int,
    amplitude: float = 0.3,
    n_harmonics: int = 40,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Render one voiced token as a formant-shaped harmonic stack."""
    t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
    k = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
    freqs = k * f0
    gain = _formant_gain(freqs, f1, f2) * torch.exp(-freqs / 6000.0)
    phases = torch.rand(n_harmonics, generator=generator) * 2 * torch.pi
    sig = (gain[:, None] * torch.sin(2 * torch.pi * freqs[:, None] * t[None] + phases[:, None])).sum(0)

    env = torch.ones_like(t)
    attack = max(1, int(0.015 * sample_rate))
    release = max(1, int(0.025 * sample_rate))
    attack = min(attack, n_samples)
    release = min(release, n_samples)
    env[:attack] = 0.5 - 0.5 * torch.cos(torch.pi * torch.arange(attack, dtype=torch.float32) / attack)
    env[-release:] = 0.5 + 0.5 * torch.cos(
        torch.pi * torch.arange(release, dtype=torch.float32) / release
    )
    return amplitude * env * sig / 3.0


def make_utterance(
    audio: AudioConfig,
    layout: str = "short",
    letters: Optional[str] = None,
    f0: float = 120.0,
    vowel_cycle: Sequence[int] = (0, 1, 2, 3),
    seed: int = 0,
) -> SyntheticUtterance:
    """Build one utterance: one character token per character of ``letters``."""
    generator = torch.Generator().manual_seed(seed)
    frames = LAYOUTS[layout]
    if letters is None:
        rng = random.Random(seed)
        letters = "".join(rng.sample(string.ascii_lowercase, len(frames)))
    letters = letters[: len(frames)]
    hop = audio.hop_length

    wavs: List[torch.Tensor] = []
    energy: List[float] = []
    f0s: List[float] = []
    for i, n_frames in enumerate(frames):
        token_f0 = f0 * (1.0 - 0.02 * i)  # slight declination across the utterance
        f1, f2 = FORMANTS[vowel_cycle[i % len(vowel_cycle)]]
        n_samples = n_frames * hop
        token = render_token(token_f0, f1, f2, n_samples, audio.sample_rate, generator=generator)
        wavs.append(token)
        energy.append(
            float(frame_energy_db(token[None], audio.n_fft, audio.hop_length).mean().item())
        )
        f0s.append(float(token_f0))

    wav = torch.cat(wavs)
    peak = wav.abs().max().clamp_min(1e-6)
    wav = wav / peak * 0.3
    return SyntheticUtterance(
        text=letters,
        layout=layout,
        wav=wav,
        token_frames=list(frames),
        token_f0=f0s,
        token_energy_db=energy,
    )


def make_corpus(
    n: int = 16,
    audio: Optional[AudioConfig] = None,
    seed: int = 0,
    f0_range: Tuple[float, float] = (95.0, 210.0),
) -> List[SyntheticUtterance]:
    """A deterministic corpus of utterances spread over both layouts and a range of voices."""
    audio = audio or AudioConfig()
    rng = random.Random(seed)
    out: List[SyntheticUtterance] = []
    for i in range(n):
        layout = "short" if i % 2 == 0 else "long"
        f0 = rng.uniform(*f0_range)
        out.append(
            make_utterance(
                audio,
                layout=layout,
                f0=f0,
                vowel_cycle=(i % 6, (i + 1) % 6, (i + 2) % 6, (i + 3) % 6),
                seed=seed + i,
            )
        )
    return out


class SyntheticSpeechBatchSource:
    """Infinite waveform batches for the autoencoder stage (one layout per batch)."""

    def __init__(
        self,
        corpus: Sequence[SyntheticUtterance],
        batch_size: int = 4,
        seed: int = 0,
        device: Optional[str] = None,
    ) -> None:
        self.groups: Dict[str, List[SyntheticUtterance]] = {}
        for utt in corpus:
            self.groups.setdefault(utt.layout, []).append(utt)
        if not self.groups:
            raise ValueError("empty corpus")
        self.batch_size = batch_size
        self.device = device
        self.generator = torch.Generator().manual_seed(seed)
        self.cursor: Dict[str, int] = {k: 0 for k in self.groups}

    def __call__(self) -> Dict[str, torch.Tensor]:
        layouts = sorted(self.groups)
        choice = int(torch.randint(len(layouts), (1,), generator=self.generator).item())
        layout = layouts[choice]
        items = self.groups[layout]
        idx = [
            int(torch.randint(len(items), (1,), generator=self.generator).item())
            for _ in range(self.batch_size)
        ]
        wav = torch.stack([items[i].wav for i in idx], dim=0)
        if self.device:
            wav = wav.to(self.device)
        return {"wav": wav}


def corpus_hours(corpus: Iterable[SyntheticUtterance], sample_rate: int) -> float:
    total = sum(u.wav.numel() for u in corpus)
    return total / sample_rate / 3600.0
