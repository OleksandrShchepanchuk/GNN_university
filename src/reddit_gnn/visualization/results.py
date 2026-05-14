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
    max_edges: int = 100,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> Figure:
    """Render a sample of (up to ``max_edges``) edges with **two colors per edge**.

    Each directed edge is drawn twice:
        * a thick outer line in the **true-sign** color (red for negative,
          green for positive);
        * a thinner inner line in the **predicted-sign** color.

    When the model is correct, the inner color covers the outer one and the
    edge looks uniformly green or red. When it is wrong, the outer color
    shows up as a halo around the inner color — disagreements jump out
    visually even on dense subgraphs.

    Default ``max_edges=100`` matches the project spec; the sampling is
    deterministic given ``seed``.
    """
    import networkx as nx
    from matplotlib.lines import Line2D

    if len(df_subset) != len(y_true) or len(df_subset) != len(y_pred):
        raise ValueError("df_subset, y_true, y_pred must have equal length")

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if len(df_subset) > max_edges:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df_subset), size=max_edges, replace=False)
        df_subset = df_subset.iloc[idx].reset_index(drop=True)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    if len(df_subset) == 0:
        ax.set_title("Predicted subgraph (empty)")
        ax.axis("off")
        return _maybe_save(fig, save_path)

    G = _build_signed_digraph(df_subset.assign(label_binary=y_true))
    pos = nx.spring_layout(G, seed=seed)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=220, node_color="#cccccc", edgecolors="black")

    src_col = "source_subreddit_norm"
    tgt_col = "target_subreddit_norm"
    triples = list(
        zip(
            df_subset[src_col].astype(str).tolist(),
            df_subset[tgt_col].astype(str).tolist(),
            y_true.tolist(),
            y_pred.tolist(),
            strict=True,
        )
    )

    def _color(label: int) -> str:
        return POSITIVE_COLOR if label == 1 else NEGATIVE_COLOR

    by_true: dict[int, list[tuple[str, str]]] = {0: [], 1: []}
    by_pred: dict[int, list[tuple[str, str]]] = {0: [], 1: []}
    n_correct = 0
    for u, v, yt, yp in triples:
        by_true[yt].append((u, v))
        by_pred[yp].append((u, v))
        if yt == yp:
            n_correct += 1

    # Pass 1: thick outer line — TRUE sign.
    for label, edges in by_true.items():
        if not edges:
            continue
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            ax=ax,
            edge_color=_color(label),
            arrows=True,
            width=4.5,
            alpha=0.95,
            connectionstyle="arc3,rad=0.05",
        )
    # Pass 2: thinner inner line — PREDICTED sign.
    for label, edges in by_pred.items():
        if not edges:
            continue
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            ax=ax,
            edge_color=_color(label),
            arrows=True,
            width=1.8,
            alpha=1.0,
            connectionstyle="arc3,rad=0.05",
        )

    deg = dict(G.degree())
    top_nodes = sorted(deg, key=deg.get, reverse=True)[: min(20, len(deg))]
    nx.draw_networkx_labels(G, pos, labels={n: n for n in top_nodes}, ax=ax, font_size=7)

    handles = [
        Line2D([0], [0], color=POSITIVE_COLOR, lw=4, label="positive (label/pred = 1)"),
        Line2D([0], [0], color=NEGATIVE_COLOR, lw=4, label="negative (label/pred = 0)"),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=POSITIVE_COLOR,
            markeredgecolor=NEGATIVE_COLOR,
            markeredgewidth=2.0,
            markersize=10,
            label="disagreement (outer halo)",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.set_title(
        f"Predicted subgraph (n={len(triples)}; {n_correct} correct, {len(triples) - n_correct} wrong)"
    )
    ax.axis("off")
    fig.tight_layout()
    return _maybe_save(fig, save_path)


__all__ = [
    "plot_confusion_matrix",
    "plot_cross_metric_comparison",
    "plot_cross_model_confusion_grid",
    "plot_error_by_degree_bin",
    "plot_model_comparison",
    "plot_pr_roc",
    "plot_predicted_subgraph",
    "plot_training_curves",
]


def plot_cross_metric_comparison(
    comparison_df: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
) -> Figure:
    """2x3 grid of per-model bar charts across six metrics.

    The headline subplot (PR-AUC neg) gets a dashed horizontal class-prior line
    and lift-annotation labels above each bar; other subplots get the standard
    bar chart with mean ± std error bars from the multi-seed columns.

    ``comparison_df`` must have the columns produced by the export aggregator
    (``test_pr_auc_neg``, ``test_pr_auc_neg_std``, ``test_pr_auc_lift``,
    ``test_roc_auc``, ``test_balanced_accuracy``, ``test_mcc``,
    ``test_f1_macro``, ``test_f1_negative_class``, ``class_prior_negative``).
    """
    # Panel layout is organised around the headline "negative class" story:
    # the top row is negative-class-specific metrics (headline PR-AUC, F1,
    # precision, recall — all interpreted with class 0 as the positive case),
    # the bottom row is symmetric metrics that don't care which class is which
    # (ROC-AUC, MCC) plus the supporting F1-macro for cross-checking.
    panel = [
        ("test_pr_auc_neg", "PR-AUC (negative class)", "Headline + lift over prior"),
        ("test_f1_negative_class", "F1 on negative class", "Companion headline metric"),
        ("test_precision_negative", "Precision (negative class)", "At threshold = 0.5"),
        ("test_recall_negative", "Recall (negative class)", "At threshold = 0.5"),
        ("test_roc_auc", "ROC-AUC", "Symmetric in class label"),
        ("test_mcc", "MCC", "Symmetric correlation [-1, 1]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()

    df = comparison_df.sort_values("test_pr_auc_neg", ascending=False).reset_index(drop=True)
    class_prior = (
        float(df["class_prior_negative"].dropna().iloc[0])
        if "class_prior_negative" in df.columns and not df["class_prior_negative"].dropna().empty
        else float("nan")
    )

    for ax, (col, title, subtitle) in zip(axes_flat, panel, strict=True):
        if col not in df.columns:
            ax.axis("off")
            ax.set_title(f"{title}\n(column missing)")
            continue
        std_col = f"{col}_std"
        yerr = df[std_col].to_numpy() if std_col in df.columns else None
        bars = ax.bar(
            df["model"],
            df[col],
            yerr=yerr,
            capsize=4,
            color="#4c72b0",
            edgecolor="black",
            alpha=0.85,
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        # Headline subplot: class-prior baseline + lift annotations.
        if col == "test_pr_auc_neg":
            if np.isfinite(class_prior):
                ax.axhline(
                    class_prior,
                    color="black",
                    linestyle="--",
                    lw=1,
                    alpha=0.6,
                    label=f"class prior = {class_prior:.3f}",
                )
                ax.legend(loc="upper right", fontsize=8)
            if "test_pr_auc_lift" in df.columns:
                for bar, lift in zip(bars, df["test_pr_auc_lift"].to_numpy(), strict=True):
                    if not np.isfinite(lift):
                        continue
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (yerr[list(bars).index(bar)] if yerr is not None else 0),
                        f"{lift:.1f}x",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

    fig.suptitle(
        "Per-model test metrics (mean ± std over 3 seeds)",
        fontsize=12,
        y=1.00,
    )
    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_cross_model_confusion_grid(
    predictions_dir: Path,
    comparison_df: pd.DataFrame,
    *,
    seed: int = 0,
    save_path: str | Path | None = None,
    pred_paths_by_model: Mapping[str, Path] | None = None,
) -> Figure:
    """1xN grid of test-split confusion matrices, one per model.

    Reads each model's seed-`seed` predictions CSV. By default looks for
    ``predictions_dir / <model>-seed{seed}.csv``; pass ``pred_paths_by_model``
    to override the file mapping (used when the leaderboard row for a model
    came from a hotfix-tagged run, e.g. ``sage-h64-seed0.csv``). Each subplot
    is titled with the model name and its ``test_pr_auc_neg`` from
    ``comparison_df``. Cells are annotated with raw count + percent.
    """
    models = comparison_df["model"].tolist()
    cms: dict[str, np.ndarray] = {}
    pr_aucs: dict[str, float] = {}
    for m in models:
        if pred_paths_by_model is not None and m in pred_paths_by_model:
            csv = Path(pred_paths_by_model[m])
        else:
            csv = Path(predictions_dir) / f"{m}-seed{seed}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        sub = df[df["split"] == "test"]
        if sub.empty:
            continue
        y_true = sub["y_true"].to_numpy().astype(int)
        y_pred = sub["y_pred"].to_numpy().astype(int)
        cm = np.array(
            [
                [
                    int(((y_true == 0) & (y_pred == 0)).sum()),
                    int(((y_true == 0) & (y_pred == 1)).sum()),
                ],
                [
                    int(((y_true == 1) & (y_pred == 0)).sum()),
                    int(((y_true == 1) & (y_pred == 1)).sum()),
                ],
            ]
        )
        cms[m] = cm
        row = comparison_df[comparison_df["model"] == m]
        pr_aucs[m] = float(row["test_pr_auc_neg"].iloc[0]) if not row.empty else float("nan")

    n = len(cms)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no seed0 predictions found", ha="center", va="center")
        ax.axis("off")
        return _maybe_save(fig, save_path)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]
    vmax = max(cm.max() for cm in cms.values())

    for ax, m in zip(axes, cms.keys(), strict=True):
        cm = cms[m]
        total = int(cm.sum())
        im = ax.imshow(cm, cmap="Blues", aspect="equal", vmin=0, vmax=vmax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"], fontsize=8)
        ax.set_yticklabels(["true 0", "true 1"], fontsize=8)
        ax.set_xlabel("predicted", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("true", fontsize=8)
        ax.set_title(f"{m}\nPR-AUC neg = {pr_aucs[m]:.3f}", fontsize=9)
        for i in range(2):
            for j in range(2):
                val = int(cm[i, j])
                pct = (val / total * 100.0) if total else 0.0
                ax.text(
                    j,
                    i,
                    f"{val}\n{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if val > vmax * 0.6 else "black",
                )
    fig.colorbar(im, ax=axes, fraction=0.035, pad=0.04)
    fig.suptitle("Test-split confusion matrices (seed 0 of each model)", fontsize=11, y=1.02)
    return _maybe_save(fig, save_path)


def plot_cross_model_pr_curves(
    predictions_dir: Path,
    comparison_df: pd.DataFrame,
    *,
    seed: int = 0,
    save_path: str | Path | None = None,
    pred_paths_by_model: Mapping[str, Path] | None = None,
) -> Figure:
    """Overlay PR curves for the negative class across all models on one axis.

    Reads each model's seed-`seed` predictions CSV, computes PR on the negative
    class (score = 1 - p_class1), and overlays the curves with each model's
    test PR-AUC-neg in the legend. The class-prior baseline gets a dashed line.
    Pass ``pred_paths_by_model`` to override the default filename lookup (used
    when the leaderboard row for a model is sourced from a hotfix-tagged run).
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    models = comparison_df["model"].tolist()
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(models), 3)))
    prior_neg: float | None = None

    for color, m in zip(palette, models, strict=False):
        if pred_paths_by_model is not None and m in pred_paths_by_model:
            csv = Path(pred_paths_by_model[m])
        else:
            csv = Path(predictions_dir) / f"{m}-seed{seed}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        sub = df[df["split"] == "test"]
        if sub.empty:
            continue
        y_true = sub["y_true"].to_numpy().astype(int)
        y_score = sub["y_score"].to_numpy().astype(float)
        y_neg_true = (y_true == 0).astype(int)
        y_neg_score = 1.0 - y_score
        precision, recall, _ = precision_recall_curve(y_neg_true, y_neg_score)
        ap = average_precision_score(y_neg_true, y_neg_score)
        ax.plot(recall, precision, lw=2, color=color, label=f"{m}: AP={ap:.3f}")
        if prior_neg is None:
            prior_neg = float(y_neg_true.mean())

    if prior_neg is not None:
        ax.axhline(
            prior_neg,
            color="grey",
            linestyle="--",
            lw=1.2,
            label=f"class prior = {prior_neg:.3f}",
        )

    ax.set_xlabel("recall (negative class)")
    ax.set_ylabel("precision (negative class)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"PR curves on the negative class (seed = {seed})")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_cross_model_roc_curves(
    predictions_dir: Path,
    comparison_df: pd.DataFrame,
    *,
    seed: int = 0,
    save_path: str | Path | None = None,
    pred_paths_by_model: Mapping[str, Path] | None = None,
) -> Figure:
    """Overlay ROC curves across all models on one axis.

    Pass ``pred_paths_by_model`` to override the default filename lookup.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    models = comparison_df["model"].tolist()
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(models), 3)))

    for color, m in zip(palette, models, strict=False):
        if pred_paths_by_model is not None and m in pred_paths_by_model:
            csv = Path(pred_paths_by_model[m])
        else:
            csv = Path(predictions_dir) / f"{m}-seed{seed}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        sub = df[df["split"] == "test"]
        if sub.empty:
            continue
        y_true = sub["y_true"].to_numpy().astype(int)
        y_score = sub["y_score"].to_numpy().astype(float)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{m}: AUC={auc:.3f}")

    ax.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1, alpha=0.5, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"ROC curves (seed = {seed})")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_threshold_tradeoff(
    predictions_dir: Path,
    model: str,
    *,
    seed: int = 0,
    save_path: str | Path | None = None,
    predictions_csv: Path | None = None,
) -> Figure:
    """Precision / recall / F1 on the negative class as a function of threshold.

    The reader can pick an operating point by tracing a vertical line at any
    threshold. The default 0.5 threshold gets a vertical marker. Pass
    ``predictions_csv`` to override the default ``predictions_dir/<model>-seed<seed>.csv``
    lookup (e.g. when sourcing from a hotfix-tagged run).
    """
    from sklearn.metrics import precision_recall_curve

    csv = (
        Path(predictions_csv)
        if predictions_csv is not None
        else Path(predictions_dir) / f"{model}-seed{seed}.csv"
    )
    if not csv.exists():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"predictions for {model}-seed{seed} not found", ha="center")
        ax.axis("off")
        return _maybe_save(fig, save_path)
    df = pd.read_csv(csv)
    sub = df[df["split"] == "test"]
    y_true = sub["y_true"].to_numpy().astype(int)
    y_score = sub["y_score"].to_numpy().astype(float)
    y_neg_true = (y_true == 0).astype(int)
    y_neg_score = 1.0 - y_score
    precision, recall, thr = precision_recall_curve(y_neg_true, y_neg_score)
    # precision_recall_curve returns one fewer threshold than precision/recall
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    # thr is "score >= thr predicts positive in PR convention" — convert back
    # to the original threshold on p_class1: decision_threshold_on_p1 = 1 - thr.
    decision_thresh = 1.0 - thr

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    order = np.argsort(decision_thresh)
    ax.plot(
        decision_thresh[order], precision[order], lw=2, label="precision (neg)", color="#4c72b0"
    )
    ax.plot(decision_thresh[order], recall[order], lw=2, label="recall (neg)", color="#dd8452")
    ax.plot(decision_thresh[order], f1[order], lw=2, label="F1 (neg)", color="#55a868")
    ax.axvline(0.5, color="grey", linestyle="--", lw=1, alpha=0.7, label="threshold = 0.5")

    ax.set_xlabel("decision threshold on p(class=1)  —  predict 0 if p < threshold")
    ax.set_ylabel("metric value on the negative class")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Negative-class precision / recall / F1 vs. threshold — {model}-seed{seed}")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_seed_trajectories(
    history_paths: Mapping[str, Path],
    *,
    metric: str = "val_pr_auc",
    save_path: str | Path | None = None,
) -> Figure:
    """Overlay per-epoch trajectories of `metric` across seeds for one model.

    `history_paths` maps a label (e.g. "seed 0") to the path of its
    `history_*.csv`. Useful for showing seed-level convergence/divergence
    side-by-side (e.g. before-vs-after the warmup fix).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(history_paths), 3)))
    for color, (label, path) in zip(palette, history_paths.items(), strict=False):
        path = Path(path)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if metric not in df.columns:
            continue
        ax.plot(df["epoch"], df[metric], lw=1.5, color=color, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(f"Per-epoch {metric} across seeds")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return _maybe_save(fig, save_path)
