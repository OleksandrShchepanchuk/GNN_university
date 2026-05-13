"""Train / validate loops with early stopping and checkpoint saving."""

from __future__ import annotations

from typing import Any


def train_one_epoch(
    model: Any, loader: Any, optimizer: Any, loss_fn: Any, device: str
) -> dict[str, float]:
    """Single training epoch. Returns ``{loss, ...}``."""
    raise NotImplementedError("training.loops.train_one_epoch is not implemented yet")


def validate(model: Any, loader: Any, loss_fn: Any, device: str) -> dict[str, float]:
    """Validation pass. Returns metric dict."""
    raise NotImplementedError("training.loops.validate is not implemented yet")


def fit(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    optimizer: Any,
    loss_fn: Any,
    epochs: int,
    early_stopping_patience: int,
    device: str,
    checkpoint_dir: Any,
    run_name: str,
) -> dict[str, list[float]]:
    """Full training procedure with early stopping. Returns the history dict."""
    raise NotImplementedError("training.loops.fit is not implemented yet")


__all__ = ["fit", "train_one_epoch", "validate"]
