"""Datasets and batch sources.

Two sources:

* :class:`LatentShardDataset` -- the real training set, reading the shards written by
  :mod:`parakeet.data.features`.
* :class:`SyntheticBatchSource` -- a *shape-faithful* random batch generator.  It exists so the
  full curriculum (all five stages) can be smoke-tested on a CPU with no corpus and no
  network, and so CI can catch shape/regression bugs immediately.  It is a test double, not a
  training strategy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from ..config import ParakeetConfig


def collate(
    items: Sequence[Dict[str, torch.Tensor]], max_ref_frames: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """Pad variable-length token/frame tensors and build masks.

    ``ref_mel``/``ref_mask`` are the **speaker/style conditioning reference**.  They were missing
    from the cached path entirely (``log_mel`` was silently dropped by this function), which meant
    the Small/flow model trained from a cache with ``ref_mel=None`` -- a zero speaker embedding and
    no style tokens at all.  Every demo that exercised conditioning built its own ``ref_mel`` by
    hand, so nothing noticed.
    """
    out: Dict[str, torch.Tensor] = {}
    b = len(items)
    if any("wav" in it for it in items):
        # the target waveform, needed by the decoder stage (mel / multi-resolution STFT losses compare
        # against audio).  `distill-decoder` could only ever run under --dry-run without it.
        max_wav = max(int(it["wav"].numel()) for it in items)
        wav = torch.zeros(b, max_wav)
        for i, it in enumerate(items):
            if "wav" in it:
                wav[i, : it["wav"].numel()] = it["wav"]
        out["wav"] = wav
    max_tok = max(int(it["ids"].numel()) for it in items)
    max_frame = max(int(it["latent"].shape[-1]) for it in items)
    ids = torch.zeros(b, max_tok, dtype=torch.long)
    text_mask = torch.zeros(b, max_tok, dtype=torch.bool)
    latent = torch.zeros(b, items[0]["latent"].shape[0], max_frame)
    frame_mask = torch.zeros(b, max_frame, dtype=torch.bool)
    for i, it in enumerate(items):
        n = int(it["ids"].numel())
        ids[i, :n] = it["ids"]
        text_mask[i, :n] = True
        t = int(it["latent"].shape[-1])
        latent[i, :, :t] = it["latent"]
        frame_mask[i, :t] = True
    out.update(ids=ids, text_mask=text_mask, latent=latent, frame_mask=frame_mask)
    # reference mel(s) for the speaker/style conditioner, padded or truncated to max_ref_frames.
    # A loader may override ref_mel to implement cross-sample pairing; otherwise the item's own
    # log_mel is the reference, so a plain cache collates into a usable conditioning batch.
    for key, mask_key in (("ref_mel", "ref_mask"), ("ref_mel_neg", "ref_mask_neg")):
        source = [it.get(key, it.get("log_mel") if key == "ref_mel" else None) for it in items]
        if all(s is None for s in source):
            continue
        n_mels = next(s.shape[0] for s in source if s is not None)
        width = max(int(s.shape[-1]) for s in source if s is not None)
        if max_ref_frames is not None:
            width = min(width, int(max_ref_frames))
        ref = torch.zeros(b, n_mels, width)
        mask = torch.zeros(b, width, dtype=torch.bool)
        for i, s in enumerate(source):
            if s is None:
                continue
            n = min(width, int(s.shape[-1]))
            ref[i, :, :n] = s[:, :n]
            mask[i, :n] = True
        out[key] = ref
        out[mask_key] = mask
    for key in ("durations", "f0", "energy", "latent_token"):
        if key not in items[0]:
            continue
        dims = items[0][key].shape[1:]
        pad = torch.zeros(b, max_tok, *dims, dtype=items[0][key].dtype)
        for i, it in enumerate(items):
            n = min(max_tok, it[key].shape[0])
            pad[i, :n] = it[key][:n]
        out[key] = pad
    # per-sample metadata must survive collation, otherwise it never reaches the loss: the teacher
    # mixture (round 6) and the voice index (round 8) were both silently dropped here
    for key in ("teacher_weight", "teacher_index", "voice"):
        if key in items[0]:
            out[key] = torch.stack([torch.as_tensor(it[key]).reshape(()) for it in items])
    return out


class LatentShardDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        max_frames: Optional[int] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        """Latent-shard dataset.

        ``indices`` selects a subset (train/val splits).  Without it there is no held-out
        evaluation anywhere in the project, which is how a capacity comparison can appear to show
        that a smaller model "fits as well" when it is merely overfitting less.
        """
        self.cache_dir = Path(cache_dir)
        index = json.loads((self.cache_dir / "index.json").read_text(encoding="utf-8"))
        self.shards = [index[i : i + 1] for i in range(len(index))]
        self._items: List[Dict[str, torch.Tensor]] = []
        for entry in index:
            payload = torch.load(self.cache_dir / entry["path"], map_location="cpu", weights_only=False)
            self._items.extend(payload["items"])
        self.max_frames = max_frames
        self.indices = list(indices) if indices is not None else None

    def __len__(self) -> int:
        return len(self._items) if self.indices is None else len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real = self._items[idx] if self.indices is None else self._items[self.indices[idx]]
        item = dict(real)
        if self.max_frames is not None:
            item["latent"] = item["latent"][:, : self.max_frames]
            # NOTE: log_mel is the *reference prompt*, not a target, so it is deliberately not
            # truncated here -- reference length is capped separately at collation
        return item


def crop_item_to_tokens(
    item: Dict[str, torch.Tensor], start_frame: int, n_frames: int
) -> Optional[Dict[str, torch.Tensor]]:
    """Crop a cached item to a frame window **and** the text tokens that cover it.

    Training a flow on whole utterances means every step sees ~140 compressed frames of *one* text: the
    per-step cost scales with sequence length while the thing that has to be learned — which latent belongs
    to which text — needs *variety* per step instead.  The papers train on random aligned crops for exactly
    that reason, and this project can too because every cache item carries per-token frame counts
    (`durations`), so a frame window maps to a token span.

    Text and acoustics must be cropped *together*: keeping the whole sentence while cutting the audio
    teaches the model to predict an arbitrary part of the utterance from all of it.  Returns ``None`` when
    the item cannot be cropped (no durations, or fewer than two tokens in the window).
    """
    durations = item.get("durations")
    latent = item.get("latent")
    if durations is None or latent is None or durations.numel() < 2:
        return None
    total_frames = int(latent.shape[-1])
    n_frames = int(min(n_frames, total_frames))
    if n_frames < 8 or total_frames < 16:
        return None
    start_frame = int(max(0, min(start_frame, total_frames - n_frames)))

    cumulative = torch.cumsum(durations.float(), dim=0)
    first = int(torch.searchsorted(cumulative, torch.tensor(float(start_frame))).item())
    last = int(torch.searchsorted(cumulative, torch.tensor(float(start_frame + n_frames - 1))).item()) + 1
    first = max(0, min(first, durations.numel() - 1))
    last = max(first + 1, min(last, durations.numel()))
    if last - first < 2:
        return None

    cropped = dict(item)
    cropped["latent"] = latent[..., start_frame : start_frame + n_frames]
    if item.get("log_mel") is not None:
        cropped["log_mel"] = item["log_mel"][..., start_frame : start_frame + n_frames]
    # per-token tensors are indexed along their *first* axis: `ids` and `durations` are (T,), while
    # `latent_token` is (T, latent_dim * rate) -- slicing the last axis there would cut the features
    for key in ("ids", "durations", "f0", "energy", "latent_token"):
        value = item.get(key)
        if value is not None:
            cropped[key] = value[first:last]
    cropped.pop("text_mask", None)  # `collate` rebuilds it from the token lengths
    cropped["crop"] = torch.tensor([start_frame, n_frames, first, last])
    return cropped


class LatentShardBatchSource:
    """Infinite batch iterator over a :class:`LatentShardDataset` (a callable for ``run_stage``).

    ``pair_references`` implements PilotTTS's **cross-sample paired training**: the reference used
    for speaker/style conditioning is a *different utterance of the same voice*, never the target
    utterance itself.  Training with the target's own mel teaches the conditioner to copy the
    answer; using a mismatched same-speaker prompt is what forces identity (speaker embedding) and
    style (style tokens) to be separately useful.  When enabled, a reference from a *different*
    voice is also supplied so the style representation can be pushed away from speaker identity.
    """

    def __init__(
        self,
        dataset: LatentShardDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int = 0,
        device: Optional[str] = None,
        pair_references: bool = False,
        max_ref_frames: Optional[int] = None,
        self_reference: bool = True,
        crop_frames: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.order: List[int] = []
        self.pos = 0
        self.device = device
        self.pair_references = pair_references
        self.max_ref_frames = max_ref_frames
        #: random aligned crops, in frames: every step then sees *variety* rather than one long sequence.
        #: `crop_item_to_tokens` cuts text and acoustics together, so the conditioning stays consistent.
        self.crop_frames = crop_frames
        #: False means "only paired references, skip items that have no partner" (a strict
        #: pairing curriculum).  True falls back to the item's own mel when no partner exists.
        self.self_reference = self_reference
        self._groups: Dict[int, List[int]] = {}
        if pair_references:
            for i in range(len(dataset)):
                voice = dataset[i].get("voice")
                self._groups.setdefault(int(voice) if voice is not None else -1, []).append(i)

    def _next_indices(self) -> List[int]:
        if self.pos + self.batch_size > len(self.order):
            self.order = torch.randperm(len(self.dataset), generator=self.generator).tolist()
            self.pos = 0
        idx = self.order[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size
        return idx

    def state_dict(self) -> Dict[str, object]:
        """Batch-order state, so a resumed run sees the same samples in the same order.

        Without this, resuming a run silently reshuffles: the model, optimizer and schedule are
        restored but the *data* sequence restarts, which is enough to make the resumed trajectory
        diverge from an uninterrupted one.
        """
        return {
            "generator": self.generator.get_state(),
            "order": list(self.order),
            "pos": int(self.pos),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if not state:
            return
        if state.get("generator") is not None:
            self.generator.set_state(state["generator"])
        self.order = list(state.get("order") or [])
        self.pos = int(state.get("pos") or 0)

    def _partner(self, index: int, groups: Sequence[Sequence[int]]) -> Optional[int]:
        """Another utterance from one of ``groups``, never ``index`` itself."""
        for group in groups:
            candidates = [j for j in group if j != index]
            if candidates:
                pick = torch.randint(len(candidates), (1,), generator=self.generator).item()
                return int(candidates[pick])
        return None

    def _attach_references(self, indices: List[int], items: List[Dict[str, torch.Tensor]]):
        for item, index in zip(items, indices):
            voice = item.get("voice")
            voice = int(voice) if voice is not None else -1
            same = [self._groups.get(voice, [])]
            different = [idxs for v, idxs in self._groups.items() if v != voice and idxs]
            positive = self._partner(index, same) if self.pair_references else None
            negative = self._partner(index, different) if self.pair_references else None
            if positive is not None:
                item["ref_mel"] = self.dataset[positive]["log_mel"]
            elif self.self_reference:
                item["ref_mel"] = item["log_mel"]
            if negative is not None:
                item["ref_mel_neg"] = self.dataset[negative]["log_mel"]

    def __call__(self) -> Dict[str, torch.Tensor]:
        indices = self._next_indices()
        items = [self.dataset[i] for i in indices]
        if self.crop_frames:
            cropped = []
            for item in items:
                total = int(item["latent"].shape[-1])
                window = min(self.crop_frames, total)
                if total <= window:
                    cropped.append(item)
                    continue
                start = int(
                    torch.randint(0, total - window + 1, (1,), generator=self.generator).item()
                )
                piece = crop_item_to_tokens(item, start, window)
                cropped.append(piece if piece is not None else item)
            items = cropped
        if self.pair_references:
            self._attach_references(indices, items)
        elif self.self_reference:
            for item in items:
                item["ref_mel"] = item["log_mel"]
        batch = collate(items, max_ref_frames=self.max_ref_frames)
        if self.device:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch


def make_batch_source(
    cfg: ParakeetConfig,
    stage: str,
    cache: Optional[str | Path] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    pair_references: Optional[bool] = None,
    max_ref_frames: Optional[int] = None,
    shuffle: bool = True,
):
    """The batch source for a training stage -- the single place this decision is made.

    This exists because the decision used to live in ``scripts/train.py``, which meant it was
    neither tested nor shared: the CLI constructed ``LatentShardBatchSource`` *without*
    ``pair_references``, so a user running the documented ``--stage flow`` command silently trained
    on each utterance's own mel -- bypassing PilotTTS cross-sample pairing entirely -- and without a
    reference-length cap.

    ``pair_references`` defaults to true for the flow stage (the only stage that consumes a
    reference) and false otherwise; pass it explicitly to override.
    """
    if cache is None:
        return SyntheticBatchSource(
            cfg, stage, batch_size=batch_size or cfg.train.batch_size, seed=seed
        )
    dataset = LatentShardDataset(cache)
    if pair_references is None:
        pair_references = stage == "flow"
    if max_ref_frames is None:
        max_ref_frames = getattr(cfg.train, "max_ref_frames", None)
    return LatentShardBatchSource(
        dataset,
        batch_size=batch_size or cfg.train.batch_size,
        shuffle=shuffle,
        seed=seed if seed is not None else cfg.train.seed,
        device=device,
        pair_references=pair_references,
        max_ref_frames=max_ref_frames,
    )


class WaveformCorpusSource:
    """Batches of **raw waveforms** from a teacher corpus (what the autoencoder stage trains on).

    Declared late but needed early: the autoencoder is the one stage that trains on audio, and until
    now every demo hand-rolled its own padding source for it -- there was no library path from a
    corpus manifest to a waveform batch, which is why no real-audio autoencoder training had ever
    happened.  Items are padded to the longest in the batch and returned with a length mask.
    """

    def __init__(
        self,
        manifest: str | Path,
        batch_size: int = 4,
        corpus_dir: Optional[str | Path] = None,
        max_seconds: Optional[float] = None,
        seed: int = 0,
        limit: Optional[int] = None,
        shuffle: bool = True,
        sample_rate: Optional[int] = None,
    ) -> None:
        import soundfile as sf  # noqa: F401  (imported for its side effect on error messages)

        self.manifest = Path(manifest)
        self.corpus_dir = Path(corpus_dir) if corpus_dir else self.manifest.parent
        self.records = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit is not None:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f"no records in {self.manifest}")
        self.batch_size = batch_size
        self.max_seconds = max_seconds
        #: resample to this rate (the *student's*).  Without it a 48 kHz teacher is fed to a model whose
        #: mel filterbank assumes 24 kHz: the audio is read at half speed and the autoencoder learns a
        #: different voice.  `build_latent_cache` always resampled; this path did not.
        self.sample_rate = sample_rate
        self.generator = torch.Generator().manual_seed(seed)
        self.shuffle = shuffle
        self.order: List[int] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, index: int) -> torch.Tensor:
        import soundfile as sf

        record = self.records[index]
        wav, sample_rate = sf.read(str(self.corpus_dir / record["wav_path"]), dtype="float32")
        tensor = torch.from_numpy(wav).reshape(-1)
        if self.sample_rate is not None and sample_rate != self.sample_rate:
            target = int(tensor.numel() * self.sample_rate / sample_rate)
            tensor = torch.nn.functional.interpolate(
                tensor.reshape(1, 1, -1), size=target, mode="linear", align_corners=False
            ).reshape(-1)
            sample_rate = self.sample_rate
        if self.max_seconds is not None:
            tensor = tensor[: int(self.max_seconds * sample_rate)]
        return tensor

    def __call__(self) -> Dict[str, torch.Tensor]:
        if self.pos + self.batch_size > len(self.order):
            self.order = (
                torch.randperm(len(self.records), generator=self.generator).tolist()
                if self.shuffle
                else list(range(len(self.records)))
            )
            self.pos = 0
        indices = self.order[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size
        waves = [self._load(i) for i in indices]
        width = max(w.numel() for w in waves)
        batch = torch.zeros(len(waves), width)
        lengths = torch.zeros(len(waves), dtype=torch.long)
        for i, w in enumerate(waves):
            batch[i, : w.numel()] = w
            lengths[i] = w.numel()
        return {"wav": batch, "wav_lengths": lengths}

    def state_dict(self) -> Dict[str, object]:
        return {"generator": self.generator.get_state(), "order": list(self.order),
                "pos": int(self.pos)}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if not state:
            return
        if state.get("generator") is not None:
            self.generator.set_state(state["generator"])
        self.order = list(state.get("order") or [])
        self.pos = int(state.get("pos") or 0)


class SyntheticBatchSource:
    """Random but *shape-correct* batches for dry runs and CPU smoke tests.

    ``stage`` selects which keys are produced, matching exactly what
    :mod:`parakeet.train.stages` consumes for that stage.
    """

    def __init__(
        self,
        cfg: ParakeetConfig,
        stage: str,
        batch_size: int = 2,
        n_frames: int = 64,
        n_tokens: int = 24,
        vocab_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.stage = stage
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.n_tokens = n_tokens
        self.vocab_size = vocab_size or cfg.text.vocab_size
        self.generator = torch.Generator().manual_seed(seed if seed is not None else cfg.train.seed)

    def rand(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, generator=self.generator)

    def state_dict(self) -> Dict[str, object]:
        """Batch-order state (see :meth:`LatentShardBatchSource.state_dict`)."""
        return {"generator": self.generator.get_state(), "order": [], "pos": 0}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if state and state.get("generator") is not None:
            self.generator.set_state(state["generator"])

    def __call__(self) -> Dict[str, torch.Tensor]:
        b = self.batch_size
        audio = self.cfg.audio
        token_width = self.cfg.autoencoder.latent_dim * max(
            1, int(self.cfg.autoencoder.latent_rate)
        )
        if self.stage == "autoencoder":
            n = self.n_frames * audio.hop_length
            return {"wav": 0.1 * self.rand(b, n)}
        if self.stage == "distill-decoder":
            # a dry run must exercise what a real run does.  This used to return *only* a waveform, so
            # the stage never took its documented token-expanded path under --dry-run -- which is part
            # of why the mismatch between that path and the cached frame latent went unnoticed.
            durations = torch.full((b, self.n_tokens), 2, dtype=torch.long)
            return {
                "wav": 0.1 * self.rand(b, 2 * self.n_tokens * audio.hop_length),
                "ids": torch.randint(1, self.vocab_size, (b, self.n_tokens), generator=self.generator),
                "latent": self.rand(b, self.cfg.autoencoder.latent_dim, 2 * self.n_tokens),
                "latent_token": self.rand(b, self.n_tokens, token_width),
                "durations": durations,
                "f0": self.rand(b, self.n_tokens),
                "energy": self.rand(b, self.n_tokens),
            }
        if self.stage == "distill-audio":
            # the text side trained through the decoder: it needs the target audio, the token signals
            # (the auxiliary objective) and the text
            return {
                "wav": 0.1 * self.rand(b, 2 * self.n_tokens * audio.hop_length),
                "ids": torch.randint(1, self.vocab_size, (b, self.n_tokens), generator=self.generator),
                "text_mask": torch.ones(b, self.n_tokens, dtype=torch.bool),
                "voice": torch.zeros(b, dtype=torch.long),
                "teacher_weight": torch.ones(b),
                "durations": torch.full((b, self.n_tokens), 2, dtype=torch.long),
                "f0": self.rand(b, self.n_tokens),
                "energy": self.rand(b, self.n_tokens),
                "latent_token": self.rand(b, self.n_tokens, token_width),
            }
        ids = torch.randint(1, self.vocab_size, (b, self.n_tokens), generator=self.generator)
        text_mask = torch.ones(b, self.n_tokens, dtype=torch.bool)
        if self.stage in {"flow", "reflow"}:
            latent = self.rand(b, self.cfg.autoencoder.latent_dim, self.n_frames)
            ref_mel = self.rand(b, audio.n_mels, 100)
            return {
                "ids": ids,
                "text_mask": text_mask,
                "latent": latent,
                "ref_mel": ref_mel,
                "ref_mask": torch.ones(b, 100, dtype=torch.bool),
            }
        if self.stage == "distill-text":
            return {
                "ids": ids,
                "text_mask": text_mask,
                "voice": torch.zeros(b, dtype=torch.long),
                "durations": torch.randint(2, 8, (b, self.n_tokens), generator=self.generator),
                "f0": self.rand(b, self.n_tokens),
                "energy": self.rand(b, self.n_tokens),
                "latent_token": self.rand(b, self.n_tokens, token_width),
            }
        raise ValueError(f"unknown stage {self.stage!r}")


def infinite(loader: Iterator[Dict[str, torch.Tensor]]) -> Callable[[], Dict[str, torch.Tensor]]:
    """Wrap a DataLoader so ``run_stage`` can pull batches forever."""
    it = iter(loader)

    def _next() -> Dict[str, torch.Tensor]:
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    return _next
