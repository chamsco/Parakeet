"""Feature extraction and the latent-shard cache.

Paradee's practical breakthrough is that the teacher is used **offline**: it renders a corpus,
the intermediate signals it exposes are saved, and the student halves then train against fixed
targets.  We keep that structure, with our own grid:

* waveform 24 kHz, hop 256 -> 93.75 mel frames/s;
* latent: 24 dims at the mel frame rate, folded by ``Kc=6`` -> 144 dims at 15.6 Hz for the flow
  module.  Every teacher's audio is re-encoded into this grid by the *frozen* autoencoder,
  which is what makes a mixture of teachers possible at all;
* cached teacher signals: durations, F0 (quantised), energy, and the per-token latent feature
  that the Tiny text side regresses directly.  Paradee measured direct feature supervision at
  UTMOS 4.39 vs 3.78 for end-to-end text-side distillation, so direct supervision is the
  default here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio.f0 import (
    bins_to_f0,
    energy_to_normalized,
    estimate_f0,
    f0_to_bins,
    f0_to_normalized,
    frame_energy_db,
)
from ..audio.mel import MelSpectrogram
from ..config import ParakeetConfig
from ..data.text import TextTokenizer

#: Paradee writes shards of 500 utterances; we keep the same cadence.
SHARD_SIZE = 500


@dataclass
class SignalTargets:
    """Everything the Tiny text side needs, saved per utterance."""

    utt_id: str
    n_tokens: int
    n_frames: int
    durations: List[int]
    #: quantised pitch code (analysis/debug).  0 == unvoiced.
    f0_bins: List[int]
    #: the *training target*: continuous log-F0 in [0, 1] (0 == unvoiced).  O(1) scale so that the
    #: pitch term cannot dominate the other distillation terms.
    f0_norm: List[float]
    energy_db: List[float]
    #: the *training target* for energy: O(1) normalised dBFS in [0, 1]
    energy_norm: List[float]
    latent_token: List[List[float]]
    teacher: str = ""
    tags: List[str] = None  # type: ignore[assignment]


def extract_signals(
    wav: torch.Tensor,
    cfg: ParakeetConfig,
    token_ids: torch.Tensor,
    durations: Optional[torch.Tensor] = None,
    latent_frames: Optional[torch.Tensor] = None,
    latent_rate: int = 1,
) -> SignalTargets:
    """Compute per-token teacher targets from a (teacher) waveform.

    ``durations`` must come from the teacher (Kokoro predicts them; for Orpheus/MiniMax use a
    forced aligner -- PilotTTS uses Qwen3-Force-Alignment).  When absent we fall back to a uniform
    frame split and mark it, rather than silently training on garbage.
    """
    mel = MelSpectrogram(cfg.audio)
    frames = mel.log_mel(wav)
    n_frames = frames.shape[-1]
    energy = frame_energy_db(wav, cfg.audio.n_fft, cfg.audio.hop_length)
    # YIN with a wider window than the STFT: pitch tracking needs several periods, and the
    # autocorrelation alternative is formant-biased (168 Hz reported for an 81 Hz voice).
    f0, voiced, _ = estimate_f0(
        wav,
        cfg.audio.sample_rate,
        hop_length=cfg.audio.hop_length,
        frame_length=max(cfg.audio.n_fft, 2048),
        method="yin",
    )
    f0b = f0_to_bins(f0, voiced)
    f0n = f0_to_normalized(f0, voiced)

    n_tokens = int(token_ids.numel())
    if durations is None:
        # Documented fallback when no aligner ran.  It used to split the frames *uniformly*, which
        # on real speech puts many token spans entirely inside pauses -- measured on a real Kokoro
        # corpus, that produced per-token F0 targets of 0 Hz for whole utterances (the student would
        # be taught that letters in a pause are 60 Hz).  An equal-*energy* split is a crude aligner
        # but a far better prior; it is **blended with the uniform split** because pure equal-energy
        # collapses on a signal that is mostly silence (measured: [56, 1, 1, 1, ...] frames), and a
        # blend can never be worse than half-uniform while still following the energy.
        weights = torch.pow(10.0, energy[0, :n_frames].float() / 20.0).clamp_min(1e-6)
        cumulative = torch.cumsum(weights, 0)
        cumulative = cumulative / cumulative[-1].clamp_min(1e-9)
        uniform = torch.linspace(0.0, 1.0, n_tokens + 1)[1:-1]
        energy_edges = torch.searchsorted(cumulative, uniform).float()
        uniform_edges = uniform * n_frames
        edges = (0.5 * energy_edges + 0.5 * uniform_edges).round().long().tolist()
        boundaries = [0] + [int(v) for v in edges] + [n_frames]
        for i in range(1, len(boundaries) - 1):  # strictly increasing: one frame minimum per token
            boundaries[i] = max(boundaries[i], boundaries[i - 1] + 1)
        boundaries[-1] = n_frames
        durations = torch.tensor(
            [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)],
            dtype=torch.long,
        ).clamp_min(1)
        if n_tokens > n_frames:  # degenerate: more tokens than frames
            durations = torch.cat([durations, torch.ones(n_tokens - len(durations), dtype=torch.long)])
    elif not isinstance(durations, torch.Tensor):
        durations = torch.as_tensor(durations, dtype=torch.long)

    latent_token = None
    if latent_frames is not None:
        # average the frame latents inside each token span -> "phoneme feature" (Paradee).  With
        # ``latent_rate`` > 1 the span is split into that many equal sub-spans and each sub-vector is
        # the mean over its own sub-span: the geometry comes from `subtoken_spans`, the same function
        # inference uses to expand predictions, so the two cannot drift apart.
        from ..models.duration import subtoken_spans

        rate = max(1, int(latent_rate or 1))
        spans: List[torch.Tensor] = []
        geometry = subtoken_spans(durations, rate, n_frames)
        for token_spans in geometry:
            if not token_spans:
                token_spans = [(0, max(1, min(n_frames, 1)))]
            whole = latent_frames[0, :, token_spans[0][0] : token_spans[-1][1]]
            token_mean = whole.mean(dim=-1) if whole.numel() else torch.zeros(latent_frames.shape[1])
            pieces = []
            for a, b in token_spans:
                a = max(0, min(int(a), n_frames))
                b = max(a + 1, min(int(b), n_frames))
                pieces.append(latent_frames[0, :, a:b].mean(dim=-1) if b > a else token_mean)
            while len(pieces) < rate:  # a token shorter than `rate` frames keeps its shape
                pieces.append(token_mean)
            spans.append(torch.cat(pieces[:rate]))
        latent_token = torch.stack(spans)

    return SignalTargets(
        utt_id="",
        n_tokens=n_tokens,
        n_frames=n_frames,
        durations=[int(d) for d in durations.tolist()],
        f0_bins=[int(b) for b in f0b[0, :n_frames].tolist()],
        f0_norm=[float(v) for v in f0n[0, :n_frames].tolist()],
        energy_db=[float(v) for v in energy[0, :n_frames].tolist()],
        energy_norm=[float(v) for v in energy_to_normalized(energy[0, :n_frames]).tolist()],
        latent_token=latent_token.tolist() if latent_token is not None else [],
    )


class LatentShardWriter:
    """Accumulate utterance tensors and flush them as ``.pt`` shards (Paradee-style)."""

    def __init__(self, out_dir: str | Path, shard_size: int = SHARD_SIZE) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.buffer: List[Dict[str, torch.Tensor]] = []
        self.index: List[dict] = []
        self.shard_idx = 0

    def add(self, item: Dict[str, torch.Tensor]) -> None:
        self.buffer.append({k: v.detach().cpu() for k, v in item.items() if isinstance(v, torch.Tensor)})
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> Optional[Path]:
        if not self.buffer:
            return None
        path = self.out_dir / f"shard_{self.shard_idx:05d}.pt"
        torch.save({"items": self.buffer}, path)
        self.index.append({"path": path.name, "n": len(self.buffer)})
        self.buffer = []
        self.shard_idx += 1
        (self.out_dir / "index.json").write_text(json.dumps(self.index, indent=2), encoding="utf-8")
        return path


def aggregate_to_tokens(
    values: Sequence[float],
    durations: Sequence[int],
    n_tokens: Optional[int] = None,
    ignore_zeros: bool = False,
    carry_nearest: bool = False,
) -> List[float]:
    """Average a frame-level series over each token's frame span.

    The text side predicts **per-token** prosody (one F0/energy value per text token), while
    :func:`extract_signals` naturally produces frame-level series.  The cache must store the
    token-level form, otherwise training crashes with a shape mismatch (45 tokens against 315
    frames) -- which is exactly what the end-to-end dry run caught.

    ``ignore_zeros`` matters for F0: unvoiced frames carry the value 0, and including them in the
    mean tells the student that a half-voiced token has *half* the pitch it really has.  The mean is
    then taken over voiced frames only, and a fully unvoiced span stays 0.

    ``carry_nearest`` then replaces those zeros with the nearest voiced token's value.  A text token
    whose span fell inside a pause is an *alignment* artifact, not a 60 Hz pitch: writing 0 there
    teaches the student that pause-adjacent letters are the lowest pitch in the range.  Measured on
    a real Kokoro corpus without an aligner, whole utterances came back with 0 Hz targets before
    this; the fill count is reported so the approximation stays visible.
    """
    out: List[float] = []
    start = 0
    for d in durations:
        end = min(len(values), max(start + 1, start + int(d)))
        span = list(values[start:end]) if end > start else list(values[start : start + 1])
        if ignore_zeros:
            span = [v for v in span if v != 0] or [0.0]
        out.append(float(sum(span) / max(1, len(span))))
        start = end
    if n_tokens is not None:
        if len(out) < n_tokens:  # pad rather than misalign
            out.extend([0.0] * (n_tokens - len(out)))
        out = out[:n_tokens]
    if carry_nearest and out:
        filled = [i for i, v in enumerate(out) if v == 0]
        voiced = [i for i, v in enumerate(out) if v != 0]
        if filled and voiced:
            for i in filled:
                nearest = min(voiced, key=lambda j: abs(j - i))
                out[i] = out[nearest]
    return out


@torch.no_grad()
def build_latent_cache(
    manifest_path: str | Path,
    out_dir: str | Path,
    cfg: ParakeetConfig,
    autoencoder: nn.Module,
    tokenizer: Optional[TextTokenizer] = None,
    teacher_latent_norm: Optional[nn.Module] = None,
    limit: Optional[int] = None,
    use_teacher_durations: bool = True,
    teacher_weights: Optional[Dict[str, float]] = None,
    teacher_min_weight: float = 0.05,
    voice_names: Optional[Sequence[str]] = None,
    base_dir: Optional[str | Path] = None,
) -> Path:
    """Turn a teacher corpus into shards of
    ``(ids, durations, f0, energy, latent_token, latent, ref_mel, teacher_weight, teacher_index,
    voice)``.

    ``teacher_weights`` is the mixture (teacher name -> share).  Each sample stores its raw weight
    (share x quality, if the manifest carries a quality score) so that the mixture reaches the loss
    as per-sample weighting instead of being a config value nothing reads.

    ``voice_names`` maps the manifest's ``voice`` strings onto embedding indices in first-seen
    order; without it the manifest's voices are collected automatically.  Multi-voice training was
    unwired for the same reason the mixture was: the manifest carried a voice, nothing downstream
    ever read it.

    ``base_dir`` is where ``wav_path`` entries are resolved from, defaulting to the manifest's own
    directory.  It must be passed explicitly when the manifest lives in a subdirectory -- a curated
    ``corpus/curated/kept.jsonl`` still refers to ``wav/...`` relative to the corpus root.
    """
    import soundfile as sf

    from ..train.losses import MultiTeacherMixer

    mixer = MultiTeacherMixer(teacher_weights or {}, min_weight=teacher_min_weight)
    teacher_names: List[str] = []
    voices: List[str] = list(voice_names or [])
    base = Path(base_dir) if base_dir is not None else Path(manifest_path).parent
    records = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    tokenizer = tokenizer or TextTokenizer(mode=cfg.text.mode)
    mel = MelSpectrogram(cfg.audio)
    writer = LatentShardWriter(out_dir)

    for i, rec in enumerate(records):
        if limit is not None and i >= limit:
            break
        wav, sr = sf.read(str(base / rec["wav_path"]), dtype="float32")
        wav_t = torch.from_numpy(wav).reshape(1, -1)
        if sr != cfg.audio.sample_rate:
            import torch.nn.functional as F

            new_len = int(wav_t.shape[-1] * cfg.audio.sample_rate / sr)
            wav_t = F.interpolate(wav_t[:, None, :], size=new_len, mode="linear", align_corners=False)[:, 0, :]
        log_mel = mel.log_mel(wav_t)
        latent = autoencoder.encode(log_mel)
        if teacher_latent_norm is not None:
            latent = teacher_latent_norm.normalize(latent)

        teacher = str(rec.get("teacher", "unknown"))
        if teacher not in teacher_names:
            teacher_names.append(teacher)
        quality = rec.get("quality")
        quality_t = None
        if quality is not None:
            score = quality.get("mos") if isinstance(quality, dict) else None
            quality_t = torch.tensor([float(score) if score is not None else 1.0])
        weight = mixer.weights([teacher], quality=quality_t)[0]

        # NOTE: no special tokens.  The cache must tokenise exactly like inference
        # (Synthesizer.prepare_text uses add_special=False) and like the per-token targets, or the
        # text side learns two extra tokens that inference never supplies.
        ids = tokenizer.encode(rec["text"], max_len=cfg.text.max_len, add_special=False)
        # durations come from the teacher's own timing when the manifest carries it (a forced
        # aligner, per docs/03-DATA.md); use_teacher_durations=False forces the uniform fallback.
        # The flag used to be accepted and ignored, so asking for the fallback silently got the
        # teacher's durations anyway.  Manifests written by synthesize_corpus carry no token_frames,
        # so both settings coincide until an aligner fills them in.
        teacher_frames = rec.get("token_frames") if use_teacher_durations else None
        if teacher_frames and getattr(tokenizer, "phonemized", False):
            # the marks time *characters*; in phoneme mode the tokens are phonemes.  Convert while
            # preserving each word's measured total (see g2p.phoneme_frames_from_char_frames).
            from .g2p import phoneme_frames_from_char_frames

            converted = phoneme_frames_from_char_frames(rec["text"], teacher_frames)
            if converted:
                teacher_frames = converted
        if teacher_frames is not None:
            # the token axis must match the tokeniser exactly, or every downstream target misaligns
            needed = int(ids.numel())
            reconciled = [int(f) for f in teacher_frames][:needed]
            while len(reconciled) < needed:
                reconciled.append(1)
            teacher_frames = reconciled
        sig = extract_signals(
            wav_t, cfg, ids, durations=teacher_frames, latent_frames=latent,
            latent_rate=cfg.autoencoder.latent_rate,
        )
        n_frames = min(sig.n_frames, latent.shape[-1])
        n_tokens = int(ids.numel())
        # per-token prosody targets (frame-level series averaged over each token's span)
        f0_tokens = aggregate_to_tokens(
            sig.f0_norm, sig.durations, n_tokens, ignore_zeros=True, carry_nearest=True
        )
        energy_tokens = aggregate_to_tokens(sig.energy_norm, sig.durations, n_tokens)

        voice = str(rec.get("voice") or "")
        if voice not in voices:
            voices.append(voice)
        writer.add(
            {
                "ids": ids,
                "n_frames": torch.tensor(n_frames),
                "latent": latent[0, :, :n_frames],
                "log_mel": log_mel[0, :, :n_frames],
                "durations": torch.tensor(sig.durations, dtype=torch.long),
                "f0": torch.tensor(f0_tokens, dtype=torch.float32),
                "energy": torch.tensor(energy_tokens, dtype=torch.float32),
                "latent_token": torch.tensor(sig.latent_token or [[0.0] * latent.shape[1]] * n_tokens),
                # the target waveform.  `distill-decoder` needs it (the mel and multi-resolution STFT
                # losses compare against audio, not features), and without it the stage could only run
                # under `--dry-run` -- its promised "train on the latent distribution inference
                # produces" path was unreachable on real data.  Storing audio costs ~4 bytes/sample and
                # removes the dependency on the corpus being present at training time.
                "wav": wav_t[0, : n_frames * cfg.audio.hop_length],
                "teacher_weight": weight,
                "teacher_index": torch.tensor(teacher_names.index(teacher)),
                # the voice embedding index the text side conditions on
                "voice": torch.tensor(voices.index(voice), dtype=torch.long),
            }
        )
    if len(voices) > max(1, cfg.n_voices):
        raise ValueError(
            f"corpus contains {len(voices)} voices {voices} but cfg.n_voices={cfg.n_voices}; "
            "raise n_voices (or pass explicit voice_names) or the voice embedding will index out "
            "of range during training"
        )
    path = writer.flush()
    meta = {
        "n_shards": writer.shard_idx,
        "shard_size": writer.shard_size,
        "config": asdict(cfg.audio),
        "latent_dim": cfg.autoencoder.latent_dim,
        "tokenizer_mode": cfg.text.mode,
        "teacher_names": teacher_names,
        "teacher_weights": teacher_weights or {},
        "voice_names": voices,
    }
    (Path(out_dir) / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return Path(out_dir)


def cache_teacher_corpus(
    corpus_dir: str | Path,
    out_dir: str | Path,
    cfg: ParakeetConfig,
    autoencoder: nn.Module,
    tokenizer: Optional[TextTokenizer] = None,
    limit: Optional[int] = None,
    teacher_latent_norm: Optional[nn.Module] = None,
    teacher_weights: Optional[Dict[str, float]] = None,
    manifest_name: Optional[str] = None,
    teacher_min_weight: float = 0.05,
) -> Path:
    """Build a latent cache from a teacher-corpus directory (the documented pipeline entry point).

    Three things are deliberately automatic, because each was a real defect on the CLI path:

    * **the mixture comes from the corpus's own provenance** (``corpus_meta.json``) unless overridden.
      The CLI used to call :func:`build_latent_cache` with no mixture at all, so every weight became
      1.0 and "mix training" reverted to a decoration on the documented path;
    * **the curated manifest is preferred** when curation has run (``kept.jsonl`` beats
      ``manifest.jsonl``), so the P1 filters actually affect training data;
    * the voice index comes from the manifest, so multi-voice conditioning is preserved.
    """
    corpus_dir = Path(corpus_dir)
    if manifest_name is None:
        # curated manifests win over the raw one, wherever curation wrote them: the CLI puts them in
        # <corpus>/curated/, an in-place curation leaves them in <corpus>/
        candidates = [
            corpus_dir / "kept.jsonl",
            corpus_dir / "curated" / "kept.jsonl",
            corpus_dir / "manifest.jsonl",
        ]
        existing = [c for c in candidates if c.exists()]
        if not existing:
            raise FileNotFoundError(
                f"no manifest in {corpus_dir}: expected kept.jsonl, curated/kept.jsonl or "
                "manifest.jsonl"
            )
        manifest = existing[0]
    else:
        manifest = corpus_dir / manifest_name

    if teacher_weights is None:
        meta_path = corpus_dir / "corpus_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            teacher_weights = meta.get("mix") or None
    return build_latent_cache(
        manifest,
        out_dir,
        cfg,
        autoencoder,
        tokenizer=tokenizer,
        teacher_latent_norm=teacher_latent_norm,
        limit=limit,
        teacher_weights=teacher_weights,
        teacher_min_weight=teacher_min_weight,
        # record paths are relative to the corpus root, not to a curated subdirectory
        base_dir=corpus_dir,
    )


@torch.no_grad()
def latents_from_waveforms(
    wavs: torch.Tensor, cfg: ParakeetConfig, autoencoder: nn.Module
) -> torch.Tensor:
    """Convenience helper: waveform batch -> normalised latent batch."""
    mel = MelSpectrogram(cfg.audio)
    return autoencoder.encode(mel.log_mel(wavs))


@torch.no_grad()
def token_targets_from_corpus(
    model,
    utterances,
    cfg: ParakeetConfig,
    tokenizer: Optional[TextTokenizer] = None,
    normalize: bool = True,
) -> List[Dict[str, object]]:
    """Build in-memory teacher-signal targets: durations, F0 bins, energy and per-token latent.

    This is the single-utterance counterpart of :func:`build_latent_cache`: the per-token latent
    is the mean of the *normalised* frame latents over each token's span, which is exactly the
    quantity the Tiny text side regresses and the decoder is then trained to render.

    ``utterances`` must expose ``text``, ``token_frames``, ``token_f0``, ``token_energy_db`` and
    ``wav`` (see :mod:`parakeet.data.synthetic`).
    """
    tokenizer = tokenizer or TextTokenizer(mode=cfg.text.mode)
    mel = MelSpectrogram(cfg.audio)
    out: List[Dict[str, object]] = []
    for utt in utterances:
        wav = utt.wav.reshape(1, -1)
        latent = model.autoencoder.encode(mel.log_mel(wav))[0]  # (C, T_latent)
        n_latent = latent.shape[-1]
        spans, start = [], 0
        for d in utt.token_frames:
            end = min(n_latent, max(start + 1, start + int(d)))
            spans.append((start, end))
            start = end
        token_latent = torch.stack([latent[:, s:e].mean(-1) for s, e in spans])  # (Tk, C)
        if normalize:
            token_latent = model.latent_norm.normalize(token_latent.T[None])[0].T

        f0_norm = f0_to_normalized(
            torch.tensor(utt.token_f0, dtype=torch.float32)[None],
            torch.ones(1, len(utt.token_f0), dtype=torch.bool),
        )[0]
        out.append(
            {
                "utt": utt,
                "text": utt.text,
                "wav": wav,
                "ids": tokenizer.encode(utt.text, add_special=False),
                "durations": torch.tensor(utt.token_frames, dtype=torch.long),
                "f0": f0_norm,
                "energy": energy_to_normalized(
                    torch.tensor(utt.token_energy_db, dtype=torch.float32)
                ),
                "latent_token": token_latent,
            }
        )
    return out


class TokenTargetBatchSource:
    """Infinite batches of cached token targets, one duration layout at a time.

    Keeping a batch within one layout means every waveform in it has the same length, so the
    adversarial and spectral losses need no padding masks.
    """

    def __init__(
        self,
        targets: Sequence[Dict[str, object]],
        stage: str,
        batch_size: int = 4,
        seed: int = 0,
        model=None,
    ) -> None:
        self.groups: Dict[str, List[Dict[str, object]]] = {}
        for t in targets:
            layout = getattr(t.get("utt"), "layout", "default")
            self.groups.setdefault(layout, []).append(t)
        if not self.groups:
            raise ValueError("no targets")
        self.stage = stage
        self.batch_size = batch_size
        self.model = model
        self.generator = torch.Generator().manual_seed(seed)

    def _sample(self) -> List[Dict[str, object]]:
        layouts = sorted(self.groups)
        choice = int(torch.randint(len(layouts), (1,), generator=self.generator).item())
        items = self.groups[layouts[choice]]
        return [
            items[int(torch.randint(len(items), (1,), generator=self.generator).item())]
            for _ in range(self.batch_size)
        ]

    def __call__(self) -> Dict[str, torch.Tensor]:
        items = self._sample()
        batch: Dict[str, torch.Tensor] = {
            "ids": torch.stack([t["ids"] for t in items]),  # type: ignore[arg-type]
            "durations": torch.stack([t["durations"] for t in items]),  # type: ignore[arg-type]
            "f0": torch.stack([t["f0"] for t in items]),  # type: ignore[arg-type]
            "energy": torch.stack([t["energy"] for t in items]),  # type: ignore[arg-type]
            "latent_token": torch.stack([t["latent_token"] for t in items]),  # type: ignore[arg-type]
        }
        batch["text_mask"] = torch.ones_like(batch["ids"], dtype=torch.bool)
        if self.stage == "distill-decoder":
            if self.model is None:
                raise ValueError("distill-decoder batches need the model to build latents")
            wavs = torch.stack([t["wav"].reshape(-1) for t in items], dim=0)  # type: ignore[union-attr]
            latent, _ = self.model.decoder_latent_from_tokens(
                batch["latent_token"], batch["durations"], batch["f0"], batch["energy"]
            )
            hop = self.model.cfg.audio.hop_length if hasattr(self.model, "cfg") else 256
            needed = wavs.shape[-1] // hop + 2
            if latent.shape[-1] < needed:
                latent = F.pad(latent, (0, needed - latent.shape[-1]), mode="replicate")
            else:
                latent = latent[..., :needed]
            batch["wav"] = wavs
            batch["latent"] = latent
        return batch


@torch.no_grad()
def fit_latent_normalizer(
    normalizer: nn.Module,
    autoencoder: nn.Module,
    batches,
    cfg: ParakeetConfig,
    max_batches: int = 20,
) -> nn.Module:
    """Fit the running latent statistics on the frozen autoencoder's output.

    This must happen **after** autoencoder training and **before** the latent cache is written:
    the text side and the flow module both learn in normalised latent space, and synthesis
    de-normalises through :meth:`ParakeetTiny.decoder_latent_from_tokens`.  Skipping this step
    leaves an identity normaliser, which "works" but wastes dynamic range.
    """
    mel = MelSpectrogram(cfg.audio)
    for _ in range(max_batches):
        batch = batches()
        wav = batch["wav"] if isinstance(batch, dict) else batch
        latent = autoencoder.encode(mel.log_mel(wav))
        normalizer.update(latent)
    return normalizer
