"""Audio DSP primitives (dependency-free: pure torch).

Everything Parakeet needs at training/inference time is implemented here so that the
package runs with nothing but ``torch`` + ``numpy``.  Heavier reference implementations
(librosa, parselmouth, torchaudio) are only used by the optional *teacher-signal* tooling.
"""

from .mel import MelSpectrogram, hz_to_mel, mel_filterbank, mel_to_hz  # noqa: F401
from .f0 import estimate_f0, frame_energy_db, f0_to_bins, bins_to_f0  # noqa: F401
from .istft import OLAISTFT, StreamingOLA  # noqa: F401

__all__ = [
    "MelSpectrogram",
    "mel_filterbank",
    "hz_to_mel",
    "mel_to_hz",
    "estimate_f0",
    "frame_energy_db",
    "f0_to_bins",
    "bins_to_f0",
    "OLAISTFT",
    "StreamingOLA",
]
