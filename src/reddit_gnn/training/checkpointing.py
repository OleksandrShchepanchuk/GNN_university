"""Checkpoint save / load utilities.

A checkpoint is a single ``.pt`` file containing:

* ``model_state_dict``     — ``model.state_dict()``;
* ``optimizer_state_dict`` — ``optimizer.state_dict()`` (``None`` when only a
  weights snapshot is being saved, e.g. final-evaluation-only checkpoints);
* ``cfg``                  — the merged YAML config that produced the model;
* ``val_metric_at_save``   — the validation metric value at the moment the
  checkpoint was written (e.g. best PR-AUC seen so far).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    cfg: dict[str, Any] | None = None,
    val_metric: float | None = None,
) -> Path:
    """Persist model + (optionally) optimizer state along with cfg/metric metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "cfg": cfg,
        "val_metric_at_save": val_metric,
    }
    torch.save(state, path)
    log.info("Saved checkpoint to %s (val_metric=%s)", path, val_metric)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model (and optionally optimizer) weights in place. Returns the
    full state dict (without the heavyweight tensors), useful for the cfg /
    ``val_metric_at_save`` metadata."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return {
        "cfg": state.get("cfg"),
        "val_metric_at_save": state.get("val_metric_at_save"),
    }


__all__ = ["load_checkpoint", "save_checkpoint"]
