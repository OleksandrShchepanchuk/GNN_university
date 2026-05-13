"""Checkpoint save / load helpers.

Checkpoints land under ``models/checkpoints/<run_name>/`` with ``best.pt`` and
``last.pt`` files plus a sidecar ``meta.json`` describing the merged config,
selected metric, and git SHA (when available).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(model: Any, path: str | Path, meta: dict[str, Any] | None = None) -> None:
    """Persist ``model.state_dict()`` plus optional metadata sidecar."""
    raise NotImplementedError("training.checkpointing.save_checkpoint is not implemented yet")


def load_checkpoint(model: Any, path: str | Path) -> Any:
    """Load weights into ``model`` in place. Returns the metadata dict."""
    raise NotImplementedError("training.checkpointing.load_checkpoint is not implemented yet")


def latest_checkpoint(run_dir: str | Path) -> Path:
    """Return the most recent ``.pt`` file inside ``run_dir``."""
    raise NotImplementedError("training.checkpointing.latest_checkpoint is not implemented yet")


__all__ = ["latest_checkpoint", "load_checkpoint", "save_checkpoint"]
