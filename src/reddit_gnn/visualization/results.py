"""Training-curve, evaluation, and qualitative-error plots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from reddit_gnn.visualization import NEGATIVE_COLOR, POSITIVE_COLOR, _maybe_save
from reddit_gnn.visualization.subgraphs import _build_signed_digraph

_CLASS_NAMES_DEFAULT = ("negative (0)", "neutral/positive (1)")


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
    normalize: bool = False,
    save_path: str | Path | None = None,
) -> Figure:
    """Annotated confusion matrix (2x2 by default)."""
    cm = np.asarray(cm)
    if class_names is None:
        class_names = list(_CLASS_NAMES_DEFAULT[: cm.shape[0]])
    display = cm.astype(float)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(display, cmap="Blues", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = display[i, j]
            text = f"{value:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > display.max() / 2 else "black",
            )
    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_pr_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    save_path: str | Path | None = None,
) -> Figure:
    """Side-by-side Precision-Recall and ROC curves."""
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    axes[0].plot(recall, precision, color="#4c72b0", lw=2, label=f"AP = {ap:.3f}")
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title("Precision-Recall")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(True, linestyle=":", alpha=0.4)
    axes[0].legend(loc="lower left")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    axes[1].plot(fpr, tpr, color=POSITIVE_COLOR, lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], color="black", linestyle="--", lw=1, alpha=0.5)
    axes[1].set_xlabel("false positive rate")
    axes[1].set_ylabel("true positive rate")
    axes[1].set_title("ROC")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(True, linestyle=":", alpha=0.4)
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_training_curves(
    history: Mapping[str, Iterable[float]],
    *,
    metric: str = "f1_macro",
    save_path: str | Path | None = None,
) -> Figure:
    """Loss + a primary metric over epochs.

    ``history`` is a dict like ``{"train_loss": [...], "val_loss": [...],
    "train_<metric>": [...], "val_<metric>": [...]}``. Missing keys are skipped.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    train_loss = list(history.get("train_loss", []))
    val_loss = list(history.get("val_loss", []))
    epochs_loss = range(1, max(len(train_loss), len(val_loss)) + 1)
    if train_loss:
        axes[0].plot(
            list(epochs_loss)[: len(train_loss)], train_loss, label="train", color="#4c72b0"
        )
    if val_loss:
        axes[0].plot(list(epochs_loss)[: len(val_loss)], val_loss, label="val", color="#dd8452")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, linestyle=":", alpha=0.4)
    if train_loss or val_loss:
        axes[0].legend()

    tr = list(history.get(f"train_{metric}", []))
    va = list(history.get(f"val_{metric}", []))
    if tr:
        axes[1].plot(range(1, len(tr) + 1), tr, label="train", color="#4c72b0")
    if va:
        axes[1].plot(range(1, len(va) + 1), va, label="val", color="#dd8452")
    axes[1].set_title(metric)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(metric)
    axes[1].grid(True, linestyle=":", alpha=0.4)
    if tr or va:
        axes[1].legend()

    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_model_comparison(
    results: pd.DataFrame,
    *,
    metric: str = "f1_macro",
    model_col: str = "model",
    sort: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Horizontal bar chart comparing models on a single metric."""
    if metric not in results.columns:
        raise KeyError(f"plot_model_comparison: metric {metric!r} not in DataFrame columns")
    if model_col not in results.columns:
        raise KeyError(f"plot_model_comparison: column {model_col!r} not in DataFrame columns")
    table = results[[model_col, metric]].copy()
    if sort:
        table = table.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(table) + 1.5))
    ax.barh(table[model_col], table[metric], color="#4c72b0", edgecolor="black")
    for i, value in enumerate(table[metric].to_numpy()):
        ax.text(value, i, f" {value:.3f}", va="center", fontsize=9)
    ax.set_xlabel(metric)
    ax.set_title(f"Model comparison ({metric})")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_error_by_degree_bin(
    degree: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = 10,
    use_log_bins: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Error rate as a function of node-level degree (binned).

    ``degree`` is a per-row array (same length as ``y_true`` / ``y_pred``)
    expressing the relevant node degree — e.g. the source-node out-degree —
    on the training graph.
    """
    deg = np.asarray(degree, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    err = (y_true != y_pred).astype(float)

    if use_log_bins and np.any(deg > 0):
        max_d = float(deg.max())
        edges = np.unique(np.logspace(0, np.log10(max(max_d, 2) + 1), n_bins + 1))
        edges[0] = 0
    else:
        edges = np.linspace(deg.min(), max(deg.max(), 1), n_bins + 1)

    bin_idx = np.clip(np.digitize(deg, edges) - 1, 0, len(edges) - 2)
    bin_err = np.zeros(len(edges) - 1)
    bin_n = np.zeros(len(edges) - 1, dtype=int)
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        bin_n[b] = int(mask.sum())
        if bin_n[b] > 0:
            bin_err[b] = err[mask].mean()

    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(
        range(len(centers)),
        bin_err,
        color="#4c72b0",
        edgecolor="black",
        alpha=0.85,
    )
    for bar, n in zip(bars, bin_n, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    from itertools import pairwise

    ax.set_xticks(range(len(centers)))
    ax.set_xticklabels(
        [f"{lo:.0f}-{hi:.0f}" for lo, hi in pairwise(edges)],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("degree bin")
    ax.set_ylabel("error rate")
    ax.set_title("Error rate by degree bin")
    ax.set_ylim(0, max(bin_err.max() * 1.25, 0.05))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_predicted_subgraph(
    df_subset: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> Figure:
    """Render a small subgraph colored by (true vs predicted) correctness.

    Edge styles:
        * solid green   — true positive (pred & label both 1).
        * solid red     — true negative (pred & label both 0).
        * dashed black  — misclassified.
    """
    import networkx as nx
    from matplotlib.lines import Line2D

    if len(df_subset) != len(y_true) or len(df_subset) != len(y_pred):
        raise ValueError("df_subset, y_true, y_pred must have equal length")

    G = _build_signed_digraph(df_subset.assign(label_binary=np.asarray(y_true).astype(int)))
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    pos = nx.spring_layout(G, seed=seed)

    fig, ax = plt.subplots(figsize=(8, 7))
    if G.number_of_nodes() == 0:
        ax.set_title("Predicted subgraph (empty)")
        ax.axis("off")
        return _maybe_save(fig, save_path)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=220, node_color="#cccccc", edgecolors="black")

    correct_pos: list[tuple[str, str]] = []
    correct_neg: list[tuple[str, str]] = []
    wrong: list[tuple[str, str]] = []
    for (u, v, _label), yt, yp in zip(
        df_subset[["source_subreddit_norm", "target_subreddit_norm", "label_binary"]].itertuples(
            index=False, name=None
        ),
        y_true,
        y_pred,
        strict=True,
    ):
        edge = (str(u), str(v))
        if yt == yp == 1:
            correct_pos.append(edge)
        elif yt == yp == 0:
            correct_neg.append(edge)
        else:
            wrong.append(edge)

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=correct_pos,
        ax=ax,
        edge_color=POSITIVE_COLOR,
        arrows=True,
        width=1.0,
        alpha=0.85,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=correct_neg,
        ax=ax,
        edge_color=NEGATIVE_COLOR,
        arrows=True,
        width=1.0,
        alpha=0.85,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=wrong,
        ax=ax,
        edge_color="black",
        style="dashed",
        arrows=True,
        width=1.2,
        alpha=0.9,
    )

    deg = dict(G.degree())
    top_nodes = sorted(deg, key=deg.get, reverse=True)[: min(20, len(deg))]
    nx.draw_networkx_labels(G, pos, labels={n: n for n in top_nodes}, ax=ax, font_size=7)

    handles = [
        Line2D([0], [0], color=POSITIVE_COLOR, lw=2, label="correct positive"),
        Line2D([0], [0], color=NEGATIVE_COLOR, lw=2, label="correct negative"),
        Line2D([0], [0], color="black", lw=2, linestyle="dashed", label="misclassified"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_title(
        f"Predicted subgraph ({len(correct_pos) + len(correct_neg)} correct, {len(wrong)} wrong)"
    )
    ax.axis("off")
    fig.tight_layout()
    return _maybe_save(fig, save_path)


__all__ = [
    "plot_confusion_matrix",
    "plot_error_by_degree_bin",
    "plot_model_comparison",
    "plot_pr_roc",
    "plot_predicted_subgraph",
    "plot_training_curves",
]
