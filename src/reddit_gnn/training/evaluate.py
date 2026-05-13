"""Re-evaluate a saved checkpoint against any subset of loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from reddit_gnn.training.checkpointing import load_checkpoint
from reddit_gnn.training.loops import evaluate as run_evaluate
from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    loaders: dict[str, Any],
    device: torch.device | str = "cpu",
) -> dict[str, dict[str, Any]]:
    """Load weights from ``checkpoint_path`` into ``model``; evaluate every loader."""
    model = model.to(device)
    meta = load_checkpoint(checkpoint_path, model, optimizer=None, map_location=device)
    log.info(
        "evaluate_checkpoint: loaded %s (val_metric_at_save=%s)",
        checkpoint_path,
        meta.get("val_metric_at_save"),
    )

    out: dict[str, dict[str, Any]] = {}
    for split_name, loader in loaders.items():
        out[split_name] = run_evaluate(model, loader, device, loss_fn=None)
        log.info(
            "evaluate_checkpoint: %s pr_auc(neg)=%.4f | f1_macro=%.4f",
            split_name,
            float(out[split_name]["metrics"]["pr_auc"]),
            float(out[split_name]["metrics"]["f1_macro"]),
        )
    return out


__all__ = ["evaluate_checkpoint"]
