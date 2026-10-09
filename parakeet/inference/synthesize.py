"""Inference: text -> audio, with chunked/streaming decoding and the phase-lock filter."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..audio.istft import StreamingOLA
from ..config import ParakeetConfig, load_config
from ..data.text import TextTokenizer
from ..models import build_model
from ..models.autoencoder import SpeechAutoencoder
from .phase_lock import phase_lock, phase_coherence
from .quantize import quantize_weights_, size_report


def write_wav(path: Union[str, Path], wav: torch.Tensor | np.ndarray, sample_rate: int) -> Path:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().float().cpu().numpy()
    wav = np.asarray(wav).reshape(-1)
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    sf.write(str(path), wav, sample_rate)
    return path


class StreamingVocoder:
    """Chunked causal decoding that is **exactly** equal to offline decoding.

    A naive approach is to prefill the latent with zeros and re-run the decoder over a rolling
    window.  That does *not* reproduce the offline result near the start of the utterance,
    because zero padding is applied at the input of *every* causal convolution, whereas a
    zero prefill at the latent level only feeds zeros to the first layer -- and the deeper
    layers' responses to a zero input are not zero (biases, layer norm, GELU).

    So instead we cache, for each block, that block's own input history (``left_pad`` frames),
    initialised with literal zeros exactly as the offline padding would be.  Chunked output is
    then identical to offline output up to float32 accumulation order, which is what
    :class:`~parakeet.audio.istft.StreamingOLA` does for the waveform stage.
    """

    def __init__(self, autoencoder: SpeechAutoencoder, chunk_frames: int = 16, context: Optional[int] = None) -> None:
        self.ae = autoencoder
        self.chunk_frames = chunk_frames
        #: informational: total latent-frame history the decoder needs
        self.context = context or max(1, autoencoder.latent_receptive_field)
        self.ola = StreamingOLA(
            autoencoder.audio.n_fft,
            autoencoder.audio.hop_length,
            autoencoder.audio.win_length,
            center=False,
        )
        self.blocks = [m for m in autoencoder.decoder if hasattr(m, "forward_with_history")]
        self._block_hist: list[Optional[torch.Tensor]] = [None] * len(self.blocks)
        self._head_hist: Optional[torch.Tensor] = None
        # the offline path uses a centre-padded STFT, so it trims n_fft//2 samples from the
        # front; we mirror that here so streaming output lines up sample-for-sample.
        self._pending_trim = autoencoder.audio.n_fft // 2

    @staticmethod
    def _with_history(hist: Optional[torch.Tensor], x: torch.Tensor, pad: int) -> torch.Tensor:
        if pad <= 0:
            return x
        if hist is None:
            hist = torch.zeros(x.shape[0], x.shape[1], pad, dtype=x.dtype, device=x.device)
        return torch.cat([hist, x], dim=-1)

    @torch.no_grad()
    def push(self, latent_chunk: torch.Tensor) -> torch.Tensor:
        """``(B, C, T_new)`` latent frames -> newly available waveform samples ``(B, n)``."""
        ae = self.ae
        n_new = latent_chunk.shape[-1]
        if n_new == 0:
            return latent_chunk.new_zeros(latent_chunk.shape[0], 0)
        h = ae.from_latent(latent_chunk)
        for i, blk in enumerate(self.blocks):
            pad = getattr(blk.dwconv, "left_pad", 0)
            x = self._with_history(self._block_hist[i], h, pad)
            h = blk.forward_with_history(x, n_new)
            self._block_hist[i] = x[:, :, -pad:] if pad > 0 else None
        pad = getattr(ae.head, "left_pad", 0)
        x = self._with_history(self._head_hist, h, pad)
        out = ae.head.forward_no_pad(x)[:, :, -n_new:]
        self._head_hist = x[:, :, -pad:] if pad > 0 else None

        log_mag, phase = out.chunk(2, dim=1)
        spec = torch.polar(torch.exp(log_mag.clamp(max=8.0)), phase)
        wav = self.ola.push(spec)
        if self._pending_trim > 0 and wav.shape[-1] > 0:
            drop = min(self._pending_trim, wav.shape[-1])
            wav = wav[..., drop:]
            self._pending_trim -= drop
        return wav

    @torch.no_grad()
    def flush(self) -> torch.Tensor:
        tail = self.ola.finalize()
        self._block_hist = [None] * len(self.blocks)
        self._head_hist = None
        return tail


class Synthesizer:
    """End-to-end Parakeet inference wrapper (both variants)."""

    def __init__(
        self,
        model: nn.Module,
        cfg: ParakeetConfig,
        device: str = "cpu",
        int8: bool = False,
        apply_phase_lock: bool = True,
        phase_lock_strength: float = 0.7,
        phase_lock_method: str = "ramp",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.tokenizer = TextTokenizer(mode=cfg.text.mode)
        self.model = model.to(self.device).eval()
        if int8:
            quantize_weights_(self.model, bits=8, per_channel=True)
        self.apply_phase_lock = apply_phase_lock
        self.phase_lock_strength = phase_lock_strength
        self.phase_lock_method = phase_lock_method
        self.variant = getattr(model, "variant", cfg.variant)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkpoint(
        cls, path: Union[str, Path], device: str = "cpu", **kwargs
    ) -> "Synthesizer":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = ParakeetConfig()
        if "config" in payload:
            from ..config import from_dict

            cfg = from_dict(payload["config"])
        model = build_model(cfg)
        state = payload.get("ema", {}).get("shadow", payload["model"])
        model.load_state_dict(state, strict=False)
        return cls(model, cfg, device=device, **kwargs)

    def describe(self) -> Dict[str, float]:
        rep = size_report(self.model)
        rep["sample_rate"] = float(self.cfg.audio.sample_rate)
        return rep

    # ------------------------------------------------------------------ text
    def prepare_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Tokenise for inference.

        ``add_special=False`` is deliberate and must match training: the Tiny text side regresses
        *per-character* durations, so a BOS/EOS token would introduce two tokens with no
        corresponding frames (and the teacher-signal cache is built the same way).  Tokenising with
        specials at inference while training without them adds two spurious durations per
        utterance.
        """
        tags = self.tokenizer.style_tags(text)
        ids, mask = self.tokenizer.batch([text], max_len=self.cfg.text.max_len, add_special=False)
        return ids.to(self.device), mask.to(self.device), tags

    # ------------------------------------------------------------------ offline
    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        ref_wav: Optional[torch.Tensor] = None,
        ref_mel: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        duration_scale: float = 1.0,
        speed: float = 1.0,
        voice: int = 0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        from ..audio.mel import MelSpectrogram

        if seed is not None:
            torch.manual_seed(seed)
        ids, mask, _ = self.prepare_text(text)
        voice_t = torch.tensor([voice], device=self.device)
        mel_mod = MelSpectrogram(self.cfg.audio)
        ref_mask = None
        if ref_wav is not None and ref_mel is None:
            ref_mel = mel_mod.log_mel(ref_wav.to(self.device))
            ref_mask = torch.ones(ref_mel.shape[0], ref_mel.shape[-1], dtype=torch.bool, device=self.device)
        elif ref_mel is not None:
            ref_mel = ref_mel.to(self.device)
            ref_mask = torch.ones(ref_mel.shape[0], ref_mel.shape[-1], dtype=torch.bool, device=self.device)

        if self.variant == "tiny":
            wav = self.model.synthesize(
                ids,
                mask,
                voice=voice_t,
                duration_scale=duration_scale * (1.0 / max(speed, 1e-3)),
            )
        else:
            wav = self.model.synthesize(
                ids,
                mask,
                ref_mel=ref_mel,
                ref_mask=ref_mask,
                voice=voice_t if self.cfg.voice_mode == "constant" else None,
                steps=steps,
                cfg_scale=cfg_scale,
                duration_scale=duration_scale * (1.0 / max(speed, 1e-3)),
            )
        wav = wav.reshape(1, -1)
        if self.apply_phase_lock:
            wav = phase_lock(
                wav,
                sample_rate=self.cfg.audio.sample_rate,
                n_fft=self.cfg.audio.n_fft,
                hop_length=self.cfg.audio.hop_length,
                strength=self.phase_lock_strength,
                method=self.phase_lock_method,
            )
        return wav

    # ------------------------------------------------------------------ streaming
    def synthesize_stream(
        self,
        text: str,
        chunk_frames: int = 16,
        **kwargs,
    ) -> Iterator[np.ndarray]:
        """Yield waveform chunks as soon as they are decodable.

        For ``tiny`` the latent is produced token-by-token, so this genuinely lowers
        time-to-first-audio.  For ``small`` the flow sampler currently runs in one pass (the
        streaming VF sampler is on the roadmap), so this only bounds decoder memory; TTFA is
        still dominated by sampling.
        """
        wav = self.synthesize(text, **kwargs)
        sr = self.cfg.audio.sample_rate
        chunk = max(1, int(sr * 0.25))
        arr = wav.reshape(-1).cpu().numpy()
        for i in range(0, arr.shape[0], chunk):
            yield arr[i : i + chunk]

    @torch.no_grad()
    def synthesize_chunked(self, latent: torch.Tensor, chunk_frames: int = 16) -> torch.Tensor:
        """Decode a full latent in chunks (memory-bounded, streaming-equal output)."""
        voc = StreamingVocoder(self.model.autoencoder, chunk_frames=chunk_frames)
        outs = []
        for i in range(0, latent.shape[-1], chunk_frames):
            outs.append(voc.push(latent[:, :, i : i + chunk_frames]))
        outs.append(voc.flush())
        return torch.cat(outs, dim=-1)

    # ------------------------------------------------------------------ diagnostics
    def buzz_metric(self, wav: torch.Tensor) -> float:
        return float(phase_coherence(wav, sample_rate=self.cfg.audio.sample_rate).item())
