"""Training utilities: EMA, optimizers, schedulers, checkpointing, device handling."""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
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
) -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location or "cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    if discriminator is not None and "discriminator" in payload:
        discriminator.load_state_dict(payload["discriminator"])
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


@dataclass
class StageStats:
    step: int = 0
    loss: float = 0.0
    extras: Dict[str, float] = field(default_factory=dict)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_(module: nn.Module, frozen: bool = True) -> None:
    for p in module.parameters():
        p.requires_grad = not frozen


def write_run_metadata(out_dir: str | Path, cfg: ParakeetConfig, extra: Optional[Dict[str, Any]] = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config": to_dict(cfg)}
    if extra:
        meta["extra"] = extra
    path = out / "run.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
