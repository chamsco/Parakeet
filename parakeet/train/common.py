"""Training utilities: EMA, optimizers, schedulers, checkpointing, device handling."""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn

from ..config import ParakeetConfig, to_dict


def resolve_device(spec: str = "auto") -> torch.device:
    if spec and spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMAModel:
    """Exponential moving average of parameters (used as the distillation teacher)."""

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 0) -> None:
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = self.decay
        if self.warmup:
            d = min(d, (1 + self.num_updates) / (self.warmup + self.num_updates))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    def state_dict(self) -> Dict[str, Any]:
        return {"shadow": self.shadow, "num_updates": self.num_updates, "decay": self.decay}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
        self.num_updates = int(state.get("num_updates", 0))
        self.decay = float(state.get("decay", self.decay))


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01, betas=(0.8, 0.99)) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "norm" in name.lower() or "gamma" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.05
):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, progress)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def _rng_state() -> Dict[str, Any]:
    """Every RNG that affects a training step, so a resumed run can be bit-identical."""
    import numpy as np

    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _restore_rng(state: Optional[Dict[str, Any]]) -> bool:
    if not state:
        return False
    import numpy as np

    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    return True


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    cfg: Optional[ParakeetConfig] = None,
    ema: Optional[EMAModel] = None,
    discriminator: Optional[nn.Module] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a **resumable** checkpoint: model, optimizer, EMA, discriminator, step, RNG.

    Saving only the weights is the classic way to lose a long run: on resume the optimizer moments,
    the EMA (which is the teacher for Reflow and the better-quality final weights), the critic and
    the RNG state are all gone, so the loss jumps and the schedule restarts.  ``run_stage`` already
    loaded all of them; it just never wrote them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
        "rng": _rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if cfg is not None:
        payload["config"] = to_dict(cfg)
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if discriminator is not None:
        payload["discriminator"] = discriminator.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[EMAModel] = None,
    discriminator: Optional[nn.Module] = None,
    map_location: Optional[str] = None,
    strict: bool = True,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """Restore everything :func:`save_checkpoint` wrote; returns the payload."""
    payload = torch.load(Path(path), map_location=map_location or "cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    if discriminator is not None and "discriminator" in payload:
        discriminator.load_state_dict(payload["discriminator"])
    payload["rng_restored"] = bool(_restore_rng(payload.get("rng"))) if restore_rng else False
    return payload


class Meter:
    """Tiny running-average logger."""

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.count = 0

    def update(self, values: Dict[str, float]) -> None:
        self.count += 1
        for k, v in values.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)

    def mean(self) -> Dict[str, float]:
        return {k: v / max(1, self.count) for k, v in self.sums.items()}

    def reset(self) -> None:
        self.sums.clear()
        self.count = 0


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_(module: nn.Module, frozen: bool = True) -> None:
    for p in module.parameters():
        p.requires_grad = not frozen


def git_revision(short: bool = True) -> Optional[Dict[str, Optional[str]]]:
    """``{"rev": ..., "dirty": ...}`` for the checkout this module lives in, or None.

    Best effort by design: a wheel installed outside a git checkout, or a machine without git, must
    not stop a run -- provenance is recorded when it is available, and its absence is visible.
    """
    import subprocess

    root = Path(__file__).resolve().parents[2]
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        if rev.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10
        )
        return {
            "rev": rev.stdout.strip(),
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        }
    except Exception:  # noqa: BLE001 - provenance must never break a run
        return None


def config_fingerprint(cfg: ParakeetConfig) -> str:
    """Stable hash of the full config, so a checkpoint can be tied to the exact recipe."""
    import hashlib

    payload = json.dumps(to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path) -> Optional[str]:
    """SHA-256 of a file's bytes (corpus manifests, cache indexes), or None if unreadable."""
    import hashlib

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def derive_n_voices_from_cache(cache_dir: Optional[str | Path]) -> Optional[int]:
    """How many voices the cache actually holds, or None if it cannot be determined.

    The corpus decides ``n_voices``: the latent cache builder refuses a corpus with more voices than
    the config, and a model built with the wrong width cannot load a checkpoint trained with the right
    one (``strict=False`` does not tolerate a *shape* mismatch).  Every demo derived this by hand and
    the training CLI did not derive it at all, which is how ``--resume`` on a 3-voice checkpoint failed
    against a config saying ``n_voices: 1``.
    """
    if not cache_dir:
        return None
    meta_path = Path(cache_dir) / "cache_meta.json"
    if not meta_path.exists():
        return None
    try:
        voices = json.loads(meta_path.read_text(encoding="utf-8")).get("voice_names") or []
    except (json.JSONDecodeError, OSError):
        return None
    return max(1, len(voices)) if voices else None


def derive_latent_rate_from_cache(
    cache_dir: Optional[str | Path], latent_dim: int = 24
) -> Optional[int]:
    """How many sub-latents per text token the cache holds, read from the data itself.

    The cache builder takes ``--latent-rate``, so a cache can hold 24-dim or 72-dim ``latent_token``
    targets depending on how it was built, while the model's head width comes from the config.  Getting
    them out of step produces a shape error deep inside the loss (``tensor a (24) must match tensor b
    (72)``) that says nothing about the cause -- measured twice in this project, once in the round-23
    objective change and once in round 33.  Reading the width off the first cache item removes the whole
    class of mistake, the same way the voice table is derived.
    """
    if not cache_dir:
        return None
    root = Path(cache_dir)
    width = None
    index_path = root / "index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            entries = index if isinstance(index, list) else (
                index.get("items") or index.get("shards") or []
            )
            first = entries[0] if entries else None
            if isinstance(first, dict):
                width = first.get("latent_token_width") or first.get("token_dim")
        except (json.JSONDecodeError, OSError, TypeError):
            width = None
    if width is None:
        # fall back to reading one shard's tensor shape; the index does not have to carry the width
        try:
            import torch

            shards = sorted(root.glob("*.pt"))
            if not shards:
                return None
            payload = torch.load(shards[0], map_location="cpu", weights_only=False)
            items = payload.get("items") if isinstance(payload, dict) else payload
            if items:
                token = items[0].get("latent_token")
                width = int(token.shape[-1]) if token is not None else None
        except (OSError, KeyError, IndexError, TypeError, RuntimeError, AttributeError):
            return None
    if not width:
        return None
    rate = int(round(int(width) / max(1, int(latent_dim))))
    return rate if rate >= 1 else None


def infer_model_geometry(state: Dict[str, Any]) -> Dict[str, int]:
    """Read the *widths* a checkpoint implies, so a model can be built to match it.

    Two scripts have now failed on this: a checkpoint knows how many voices it has and how wide its
    latent head is, but the code built the model from the config first and then hit a shape mismatch
    that ``strict=False`` does not tolerate.  Round 23 fixed it inline in ``real_eval.py`` and round 26
    hit it again in ``real_diagnose.py``, which is exactly why it lives here now.
    """
    geometry: Dict[str, int] = {}
    voice = state.get("voice_embed.weight")
    if voice is not None and hasattr(voice, "shape") and len(voice.shape) == 2:
        geometry["n_voices"] = max(1, int(voice.shape[0]))
    head = state.get("latent_head.2.weight")
    if head is not None and hasattr(head, "shape") and len(head.shape) == 2:
        geometry["latent_head_width"] = int(head.shape[0])
    return geometry


def calibrate_head_scale(
    model, batch: Dict[str, Any], target_key: str = "latent_token", head_name: str = "latent_head"
) -> Dict[str, float]:
    """Rescale a regression head so its *initial* output matches the target's scale.

    Round 33 measured why this matters.  AdamW moves a parameter by roughly the learning rate per step
    regardless of gradient size, so the distance from the initialisation to the target sets a floor on
    the steps needed.  The text side's `latent_head` starts with an output standard deviation of
    **0.249** against a target of **0.897** -- a factor of 3.6, i.e. ~18 000 steps at lr 2e-4 just to
    reach the right *magnitude*, before any structure can be learned.  Runs of 1 600-2 400 steps
    therefore learned the mean (cheap) and not the variation (expensive), which is exactly the round-30
    fit diagnosis (flattened cosine 0.805, per-dimension correlation 0.126).

    Multiplying the head's last linear weight *and* bias by the ratio scales its output by exactly that
    ratio, so training starts where it needs to end up.  Returns the measurement for the log.
    """
    head = getattr(model, head_name, None)
    if head is None or not hasattr(head, "__getitem__"):
        return {}
    with torch.no_grad():
        side = model.text_side(batch["ids"], batch.get("text_mask"), batch.get("voice"))
        predicted = side[target_key]
        target = batch[target_key]
        predicted_std = float(predicted.std())
        target_std = float(target.std())
        if predicted_std < 1e-8 or target_std < 1e-8:
            return {"predicted_std": predicted_std, "target_std": target_std, "ratio": 1.0}
        ratio = target_std / predicted_std
        last = head[-1]
        for parameter in (last.weight, last.bias):
            if parameter is not None:
                parameter.mul_(ratio)
        after = float(
            model.text_side(batch["ids"], batch.get("text_mask"), batch.get("voice"))[target_key].std()
        )
    return {
        "predicted_std": predicted_std,
        "target_std": target_std,
        "ratio": ratio,
        "predicted_std_after": after,
    }


def write_run_metadata(
    out_dir: str | Path,
    cfg: ParakeetConfig,
    extra: Optional[Dict[str, Any]] = None,
    stage: Optional[str] = None,
) -> Path:
    """Write ``run.json``: what code, what config and what data produced this run.

    This is the provenance record that makes a distillation result auditable -- which git revision,
    which config (by hash), which teacher mixture and which corpus the weights came from.  It was
    dead code until round 11, so no run had ever written one.
    """
    import platform

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "git": git_revision(),
        "config_sha256": config_fingerprint(cfg),
        "config": to_dict(cfg),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": getattr(torch, "__version__", None),
            "numpy": getattr(__import__("numpy"), "__version__", None),
        },
    }
    if extra:
        meta["extra"] = extra
    path = out / "run.json"
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path
