"""Typed configuration objects for Parakeet.

Configs are plain nested dataclasses, so they can be built in Python, dumped to YAML and
loaded back with :func:`load_config`.  Every field has a default, which means
``ParakeetConfig()`` is always a valid (small) model specification.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")


# --------------------------------------------------------------------------------------
# Sub-configs
# --------------------------------------------------------------------------------------
@dataclass
class AudioConfig:
    """Waveform / mel front-end used by every Parakeet variant."""

    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: Optional[float] = 12000.0
    log_eps: float = 1e-5
    #: mel frames per second (derived, informational)
    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length


@dataclass
class AutoencoderConfig:
    """Vocos-style ConvNeXt speech autoencoder (SupertonicTTS §3.1)."""

    latent_dim: int = 24
    encoder_dims: List[int] = field(default_factory=lambda: [64, 96, 128])
    encoder_blocks: List[int] = field(default_factory=lambda: [1, 2, 2])
    decoder_dim: int = 256
    decoder_blocks: int = 5
    decoder_expansion: int = 4
    decoder_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    kernel_size: int = 7
    layer_scale_init: float = 1e-6
    #: decoder is causal (streaming capable)
    causal_decoder: bool = True
    spec_head: str = "magphase"  # magphase | complex
    #: ``distill-decoder`` builds its input with :meth:`ParakeetTiny.decoder_latent_from_tokens`
    #: (per-token latents expanded exactly as inference does) instead of consuming the cached
    #: *frame-level* latent.  The stage's docstring always promised this -- "the decoder then trains
    #: on exactly the latent distribution the text side will produce at synthesis time" -- but the
    #: implementation passed the clean frame latent, and round 20 measured what that costs: the
    #: token-expanded path scores WER 0.870 where the frame path scores 0.167, for 0.016 of mel
    #: cosine.  Kept as a flag so the comparison stays reproducible.
    decoder_uses_token_latents: bool = True
    #: How many latents the text side predicts **per text token**.  At the default 1 the token path
    #: carries 24 numbers per ~6 frames (~3.9 dimensions per frame against the encoder's 24), and
    #: round 22 measured that this accounts for most of the token->frame seam: with oracle sub-latents
    #: from the teacher's own frame latent, WER falls 0.722 (rate 1) -> 0.204 (rate 2) -> 0.093
    #: (rate 3), against 0.167 for the frame latent itself, with no further gain beyond rate 3.
    #: Raising it widens the text side's latent head to ``latent_rate * latent_dim``.
    latent_rate: int = 1


@dataclass
class TextConfig:
    """Character/phoneme level text encoder.  No G2P is *required* to run."""

    vocab_size: int = 128
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.0
    max_len: int = 512
    #: "char" (SupertonicTTS) or "phoneme" (Paradee/Kokoro style)
    mode: str = "char"
    use_pos_emb: bool = True


@dataclass
class DurationConfig:
    hidden: int = 256
    n_layers: int = 2
    kernel_size: int = 5
    #: per-token durations (Tiny path) and/or utterance length (Small path)
    predict_token_durations: bool = True
    predict_utterance_length: bool = True


@dataclass
class SpeakerConfig:
    """Global identity embedding + Q-Former style tokens (PilotTTS §3.2)."""

    n_mels: int = 80
    emb_dim: int = 192
    channels: List[int] = field(default_factory=lambda: [256, 384, 384, 384])
    n_query: int = 8
    style_dim: int = 256
    qformer_layers: int = 2
    qformer_heads: int = 4
    #: frozen CAM++ weights (modelscope).  When unset a randomly-initialised
    #: ECAPA-lite is used, which is fine for tests but not for cloning.
    checkpoint: Optional[str] = None
    freeze: bool = True


@dataclass
class FlowConfig:
    """Rectified-flow / conditional flow-matching text-to-latent module."""

    latent_dim: int = 24
    #: temporal compression Kc: (C, T) -> (C*Kc, T/Kc)
    compress: int = 6
    dim: int = 384
    depth: int = 6
    n_heads: int = 6
    ffn_mult: int = 4
    kernel_size: int = 7
    dropout: float = 0.0
    text_dim: int = 256
    cond_dim: int = 256
    #: training
    sigma_min: float = 1e-4
    context_expansion: int = 4  # Ke (SupertonicTTS context-sharing batch expansion)
    #: inference
    nfe: int = 32
    distilled_nfe: int = 4
    cfg_scale: float = 1.5


@dataclass
class LossConfig:
    """Weights of the staged distillation objectives.

    ``spectral`` is deliberately small: Paradee reports that a spectral weight of 45 gives
    UTMOS 3.02 while a weight of 3 gives 4.37, i.e. *the balance matters more than size*.
    """

    mel: float = 1.0
    spectral: float = 3.0
    adversarial: float = 1.0
    feature_match: float = 2.0
    duration: float = 1.0
    f0: float = 1.0
    energy: float = 1.0
    latent_feature: float = 1.0
    #: push style tokens away from a *different* speaker's (identity debiasing; needs paired refs)
    style_separation: float = 0.1
    #: the same-speaker consistency regulariser (PilotTTS §3.2).  0 by default: pulling two
    #: same-speaker style sets together invites identity to leak into the style channel
    style_consistency_pair: float = 0.0
    phase_lock: float = 0.05
    #: `distill-audio`: the mel weight of the **audio** comparison (text side trained through the
    #: decoder, round 23).  A latent-space L1 let the text side fit its objective while rendering
    #: unintelligible audio, so this stage optimises the rendered waveform instead.
    audio_mel: float = 1.0
    #: and its spectral weight.  Kept near the autoencoder's 3.0: small, because the balance between
    #: the terms matters more than their size (Paradee).
    audio_spectral: float = 3.0
    #: weight of the auxiliary cached-signal objective.  An audio-only loss cannot pin the utterance
    #: *length* -- rounding durations to frames is not differentiable -- so the durations/F0/energy/
    #: latent terms stay in at a small weight.
    audio_aux: float = 0.1
    #: weight of the **mean-invariant** token-latent term (round 30).  Measured motivation: with the
    #: default weights the whole cached-signal bundle is ~1 % of the objective, and the latent term
    #: inside it is MSE between raw 72-dimensional vectors -- so the text side learned the *average*
    #: latent (flattened cosine 0.805) while its per-dimension correlation stayed at 0.126, which is
    #: exactly the "healthy mel proxy, unintelligible speech" signature.  This term centres both sides
    #: across tokens so it can only be reduced by matching the variation.
    signal_latent_contrast: float = 0.0
    speed_perturb: float = 0.0


@dataclass
class TrainConfig:
    stage: str = "autoencoder"  # autoencoder | flow | distill-decoder | distill-text | consistency
    #: cap on the speaker/style reference prompt in mel frames (1500 @ 93.75 Hz = 16 s, PilotTTS's
    #: 15 s prompt limit).  Without a cap, one long utterance blows up the conditioning batch.
    max_ref_frames: int = 1500
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100_000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    amp: bool = True
    num_workers: int = 4
    save_every: int = 5000
    log_every: int = 50
    seed: int = 1234
    device: str = "auto"
    out_dir: str = "runs/parakeet"
    loss: LossConfig = field(default_factory=LossConfig)


# --------------------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------------------
@dataclass
class ParakeetConfig:
    """Full model specification.

    ``variant="tiny"``  -> ~8-12M params, single (or few) voice(s), one CPU thread realtime.
    ``variant="small"`` -> ~44-60M params, zero-shot multi-speaker, GPU realtime.
    """

    variant: str = "tiny"
    audio: AudioConfig = field(default_factory=AudioConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    text: TextConfig = field(default_factory=TextConfig)
    duration: DurationConfig = field(default_factory=DurationConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    #: voice conditioning: "constant" (Tiny / single voice), "reference" (zero-shot cloning)
    voice_mode: str = "constant"
    n_voices: int = 1
    name: str = "parakeet-tiny"

    def validate(self) -> "ParakeetConfig":
        if self.variant not in {"tiny", "small"}:
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.voice_mode not in {"constant", "reference"}:
            raise ValueError(f"unknown voice_mode {self.voice_mode!r}")
        if self.flow.latent_dim != self.autoencoder.latent_dim:
            raise ValueError(
                "flow.latent_dim must equal autoencoder.latent_dim "
                f"({self.flow.latent_dim} != {self.autoencoder.latent_dim})"
            )
        if self.flow.text_dim != self.text.dim:
            raise ValueError("flow.text_dim must equal text.dim")
        if self.flow.cond_dim != self.flow.text_dim:
            raise ValueError(
                "flow.cond_dim must equal flow.text_dim: the vector-field estimator attends over "
                "one concatenated (text || speaker/style) memory"
            )
        if self.speaker.n_mels != self.audio.n_mels:
            raise ValueError("speaker.n_mels must equal audio.n_mels")
        return self


# --------------------------------------------------------------------------------------
# (De)serialisation helpers
# --------------------------------------------------------------------------------------
def _unwrap_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _coerce(tp: Any, value: Any) -> Any:
    tp = _unwrap_optional(tp)
    origin = get_origin(tp)
    if is_dataclass(tp):
        if isinstance(value, dict):
            return _dataclass_from_dict(tp, value)
        return value
    if origin is list:
        (arg,) = get_args(tp) or (Any,)
        return [_coerce(arg, v) for v in value]
    if origin is dict:
        return dict(value)
    return value


def _dataclass_from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    for f in fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints[f.name], data[f.name])
    return cls(**kwargs)  # type: ignore[return-value]


def from_dict(data: Dict[str, Any]) -> ParakeetConfig:
    """Build a :class:`ParakeetConfig` from a (possibly partial) nested dict."""
    return _dataclass_from_dict(ParakeetConfig, data).validate()


def to_dict(cfg: Any) -> Dict[str, Any]:
    """Recursively convert a dataclass to a JSON/YAML friendly dict."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, list):
        return [to_dict(v) for v in cfg]
    if isinstance(cfg, dict):
        return {k: to_dict(v) for k, v in cfg.items()}
    return cfg


def _yaml():
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - pyyaml is a hard requirement
        raise RuntimeError("pyyaml is required for YAML configs (`pip install pyyaml`)") from exc
    return yaml


def load_config(path: Union[str, Path]) -> ParakeetConfig:
    """Load a YAML (or JSON) config file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = _yaml().safe_load(text) or {}
    return from_dict(data)


def save_config(cfg: ParakeetConfig, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(to_dict(cfg), indent=2), encoding="utf-8")
    else:
        path.write_text(_yaml().safe_dump(to_dict(cfg), sort_keys=False), encoding="utf-8")
    return path


def describe(cfg: ParakeetConfig) -> str:
    """Short human-readable summary used by the CLI/scripts."""
    a, ae, f = cfg.audio, cfg.autoencoder, cfg.flow
    return (
        f"{cfg.name} [{cfg.variant}] sr={a.sample_rate} mel={a.n_mels}@{a.frame_rate:.1f}Hz "
        f"latent={ae.latent_dim} compress=1/{f.compress} "
        f"flow={f.dim}x{f.depth} nfe={f.nfe}->{f.distilled_nfe} voice={cfg.voice_mode}"
    )


def dataclass_diff(a: Any, b: Any, prefix: str = "") -> List[str]:
    """Debug helper: list differing leaf fields between two configs."""
    out: List[str] = []
    for f in dataclasses.fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if is_dataclass(va):
            out += dataclass_diff(va, vb, prefix=f"{prefix}{f.name}.")
        elif va != vb:
            out.append(f"{prefix}{f.name}: {va!r} -> {vb!r}")
    return out
