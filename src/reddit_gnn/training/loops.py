"""Train / validate loops with early stopping, scheduler, and checkpointing.

The loop is GNN-flavoured: each loader batch carries a sub-sampled
message-passing graph (``edge_index``), a supervision subset
(``edge_label_index``, ``edge_label``), and ``batch.input_id`` — the indices
into the *full* per-split edge feature tensor — which we use to look up
``edge_label_attr`` for each supervision batch without storing it inside the
loader.

Early stopping watches the validation **PR-AUC for the negative class**
(the headline metric for this imbalanced task).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from reddit_gnn.tracking import log_metrics as tracking_log_metrics
from reddit_gnn.training.checkpointing import save_checkpoint
from reddit_gnn.training.losses import compute_pos_weight, weighted_bce_with_logits
from reddit_gnn.training.metrics import classification_metrics
from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-batch / per-loader helpers
# ---------------------------------------------------------------------------


def _prepare_loader_for_device(loader, device: torch.device) -> None:
    """Pre-move per-split per-edge feature tensors to ``device`` once.

    Avoids CPU<->GPU round-trips inside the inner loop when a CUDA device is
    in use; on CPU this is a near no-op (PyTorch ``.to('cpu')`` returns the
    same tensor when already on CPU).
    """
    data = getattr(loader, "data", None)
    if data is None:
        return
    for attr in (
        "x",
        "edge_index",
        "edge_attr",
        "edge_time",
        "edge_label_attr",
        "edge_label_index",
        "edge_label",
        "edge_label_time",
    ):
        t = getattr(data, attr, None)
        if isinstance(t, torch.Tensor) and t.device != device:
            setattr(data, attr, t.to(device, non_blocking=True))


def _batch_edge_label_attr(loader, batch, device: torch.device) -> torch.Tensor | None:
    """Look up the engineered edge features for the supervision edges in this batch.

    ``loader.data.edge_label_attr`` is the full ``[S_total, F_e]`` tensor for
    the split; ``batch.input_id`` indexes into it. Both are expected to be on
    the same device after ``_prepare_loader_for_device``.
    """
    full = getattr(loader.data, "edge_label_attr", None)
    if full is None:
        return None
    input_id = getattr(batch, "input_id", None)
    if input_id is None:
        return None
    return full.index_select(0, input_id.to(full.device)).to(device)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    clip_value: float | None = 1.0,
) -> float:
    """Run one training epoch. Returns the average loss weighted by batch sup-edge count."""
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        edge_label_attr = _batch_edge_label_attr(loader, batch, device)
        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_label_index,
            edge_label_attr,
        )
        target = batch.edge_label.to(logits.dtype)
        loss = loss_fn(logits, target)
        loss.backward()
        if clip_value is not None and clip_value > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        n = int(batch.edge_label.numel())
        total_loss += float(loss.detach()) * n
        total += n
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn=None,
) -> dict[str, Any]:
    """Run inference over ``loader`` and return metrics + concatenated y_true/y_score."""
    model.eval()
    y_true_chunks: list[np.ndarray] = []
    y_score_chunks: list[np.ndarray] = []
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        edge_label_attr = _batch_edge_label_attr(loader, batch, device)
        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_label_index,
            edge_label_attr,
        )
        scores = torch.sigmoid(logits)
        y_true_chunks.append(batch.edge_label.detach().cpu().numpy())
        y_score_chunks.append(scores.detach().cpu().numpy())
        if loss_fn is not None:
            target = batch.edge_label.to(logits.dtype)
            n = int(batch.edge_label.numel())
            total_loss += float(loss_fn(logits, target).detach()) * n
            total += n

    y_true = np.concatenate(y_true_chunks) if y_true_chunks else np.array([], dtype=np.int64)
    y_score = np.concatenate(y_score_chunks) if y_score_chunks else np.array([], dtype=np.float32)
    metrics = classification_metrics(y_true, y_score)
    loss = (total_loss / max(total, 1)) if (loss_fn is not None and total > 0) else float("nan")
    return {"metrics": metrics, "y_true": y_true, "y_score": y_score, "loss": loss}


# ---------------------------------------------------------------------------
# Full training procedure
# ---------------------------------------------------------------------------


def fit(
    model: nn.Module,
    loaders: dict[str, Any],
    cfg: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    history_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train ``model`` with AdamW + ReduceLROnPlateau + early stopping.

    Watches val PR-AUC for the negative class. Best checkpoint (max val_pr_auc)
    goes to ``checkpoint_path``; a per-epoch history CSV (epoch, lr, train_loss,
    val_loss, val_pr_auc, val_f1_macro, val_f1_negative_class, elapsed_s) lands
    at ``history_path`` if provided.

    Returns a dict with:
        * ``history``                  — list[dict], one per epoch.
        * ``best_epoch`` / ``best_val_pr_auc``.
        * ``checkpoint_path``.
    """
    train_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    lr = float(train_cfg.get("lr", 5e-3))
    weight_decay = float(train_cfg.get("weight_decay", 5e-4))
    epochs = int(train_cfg.get("epochs", 100))
    patience = int(train_cfg.get("early_stopping_patience", 10))
    clip_value = float(train_cfg.get("grad_clip", 1.0))
    device = torch.device(train_cfg.get("device", "cpu"))

    model = model.to(device)
    for loader in loaders.values():
        _prepare_loader_for_device(loader, device)

    # Default loss: weighted BCE using train supervision labels.
    train_loader = loaders["train"]
    full_train_labels = train_loader.data.edge_label
    pos_weight = compute_pos_weight(full_train_labels)
    log.info("fit: pos_weight=%.4f (computed on TRAIN sup labels only)", pos_weight)

    def loss_fn(logits, target):
        return weighted_bce_with_logits(logits, target, pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(patience // 2, 1),
        min_lr=1e-6,
    )

    val_loader = loaders.get("val")
    if val_loader is None:
        raise KeyError("loaders dict is missing a 'val' entry")

    best_val: float = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, clip_value=clip_value
        )
        val_out = evaluate(model, val_loader, device, loss_fn=loss_fn)
        val_metrics = val_out["metrics"]
        val_pr_auc = float(val_metrics["pr_auc"])
        val_pr_auc_safe = -float("inf") if np.isnan(val_pr_auc) else val_pr_auc

        scheduler.step(val_pr_auc_safe)
        lr_now = float(optimizer.param_groups[0]["lr"])

        elapsed = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "val_loss": val_out["loss"],
            "val_pr_auc": val_pr_auc,
            "val_pr_auc_positive": float(val_metrics.get("pr_auc_positive", float("nan"))),
            "val_roc_auc": float(val_metrics.get("roc_auc", float("nan"))),
            "val_f1_macro": float(val_metrics.get("f1_macro", float("nan"))),
            "val_f1_negative_class": float(val_metrics.get("f1_negative_class", float("nan"))),
            "val_balanced_accuracy": float(val_metrics.get("balanced_accuracy", float("nan"))),
            "elapsed_s": elapsed,
        }
        history.append(row)

        # MLflow: when no run is active this is a no-op.
        tracking_log_metrics(
            {k: v for k, v in row.items() if k != "epoch"},
            step=epoch,
        )

        log.info(
            "epoch %d/%d | lr=%.4g | train_loss=%.4f | val_loss=%.4f | "
            "val_pr_auc(neg)=%.4f | val_f1_macro=%.4f | %.2fs",
            epoch,
            epochs,
            lr_now,
            train_loss,
            row["val_loss"],
            val_pr_auc,
            row["val_f1_macro"],
            elapsed,
        )

        if val_pr_auc_safe > best_val:
            best_val = val_pr_auc_safe
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer=optimizer,
                cfg=cfg,
                val_metric=best_val,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                log.info(
                    "Early stopping at epoch %d (no improvement in %d epochs; best=%.4f@%d)",
                    epoch,
                    patience,
                    best_val,
                    best_epoch,
                )
                break

    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
        log.info("Wrote training history to %s", history_path)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val,
        "checkpoint_path": Path(checkpoint_path),
    }


__all__ = ["evaluate", "fit", "train_one_epoch"]
