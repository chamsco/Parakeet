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
    def __init__(self, cache_dir: str | Path, max_frames: Optional[int] = None) -> None:
        self.cache_dir = Path(cache_dir)
        index = json.loads((self.cache_dir / "index.json").read_text(encoding="utf-8"))
        self.shards = [index[i : i + 1] for i in range(len(index))]
        self._items: List[Dict[str, torch.Tensor]] = []
        for entry in index:
            payload = torch.load(self.cache_dir / entry["path"], map_location="cpu", weights_only=False)
            self._items.extend(payload["items"])
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = dict(self._items[idx])
        if self.max_frames is not None:
            item["latent"] = item["latent"][:, : self.max_frames]
            # NOTE: log_mel is the *reference prompt*, not a target, so it is deliberately not
            # truncated here -- reference length is capped separately at collation
        return item


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
        if self.stage in {"autoencoder", "distill-decoder"}:
            n = self.n_frames * audio.hop_length
            return {"wav": 0.1 * self.rand(b, n)}
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
                "latent_token": self.rand(b, self.n_tokens, self.cfg.autoencoder.latent_dim),
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
