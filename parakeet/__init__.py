"""Parakeet: a tiny, lightning-fast, high-realism text-to-speech model.

Parakeet is built by distilling a *mixture* of teachers (Canopy Labs' Orpheus TTS,
MiniMax speech-2.8-turbo and, as a fully permissive fallback, Kokoro-82M) into a
small continuous-latent ConvNeXt + flow-matching student.

Design lineage
--------------
* Paradee (arXiv 2610.06817)  -> two-halves, separately-trained distillation; teacher-signal
  caching; low spectral weight + adversarial decoder training; int8; phase-locking filter.
* SupertonicTTS (arXiv 2503.23108) -> 24-dim continuous latent, ConvNeXt blocks, character-level
  text with implicit cross-attention alignment, temporal compression, utterance duration
  predictor, context-sharing batch expansion.
* PilotTTS (arXiv 2605.27258) -> disciplined data pipeline and Q-Former conditioning that
  factorises speaker identity from speaking style.
"""

from .config import ParakeetConfig, load_config, save_config  # noqa: F401

__version__ = "0.1.0"
__all__ = ["ParakeetConfig", "load_config", "save_config", "__version__"]
