"""Regenerate every figure and table under ``reports/`` from saved artifacts.

This is the **deterministic** companion to ``scripts/run_experiment.py``:
re-running it never trains a model. It walks the existing
``models/predictions/*.csv`` and ``reports/tables/metrics_*.json`` files,
recomputes the per-run error analyses, redraws the per-run figures, and
emits a cross-model ``reports/tables/comparison.csv`` aggregated across
seeds. Safe to re-run; idempotent.

Outputs (per run):
    reports/figures/confusion_<run>.png
    reports/figures/pr_roc_<run>.png
    reports/figures/error_by_degree_<run>.png
    reports/figures/error_by_time_<run>.png
    reports/figures/predicted_subgraph_<run>.png
    reports/tables/errors_by_degree_<run>.csv
    reports/tables/errors_by_time_<run>.csv
    reports/tables/errors_by_subreddit_<run>.csv
    reports/tables/confusion_examples_<run>.csv

Outputs (cross-model):
    reports/tables/comparison.csv
    reports/tables/model_agreement_agreement.csv
    reports/tables/model_agreement_kappa.csv
    reports/figures/model_agreement.png
    reports/figures/model_comparison_<metric>.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from rich.console import Console

from reddit_gnn.paths import PATHS
from reddit_gnn.training.error_analysis import (
    confusion_examples,
    errors_by_degree_bin,
    errors_by_subreddit,
    errors_by_time_bin,
    model_agreement,
)
from reddit_gnn.training.metrics import classification_metrics
from reddit_gnn.utils.logging import get_logger
from reddit_gnn.visualization.results import (
    plot_confusion_matrix,
    plot_error_by_degree_bin,
    plot_model_comparison,
    plot_pr_roc,
    plot_predicted_subgraph,
)

app = typer.Typer(add_completion=False, help=__doc__)
log = get_logger(__name__)
console = Console()


COMPARISON_COLUMNS = [
    "model",
    "hp_summary",
    "train_f1",
    "val_f1",
    "test_f1",
    "test_pr_auc_neg",
    "test_roc_auc",
    "test_balanced_acc",
    "test_mcc",
    "n_params",
    "mean_seed",
    "std_seed",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_run(metrics_file: Path) -> dict[str, Any] | None:
    """Read a metrics JSON; return None if it isn't a recognisable run record."""
    try:
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("skipping unreadable metrics file %s: %s", metrics_file, exc)
        return None
    if "run_name" not in data or "metrics_per_split" not in data:
        return None
    return data


def _find_run_files(metrics_dir: Path, predictions_dir: Path) -> list[dict[str, Any]]:
    """Pair each ``metrics_<run>.json`` with its predictions / config / history."""
    runs: list[dict[str, Any]] = []
    for metrics_path in sorted(metrics_dir.glob("metrics_*.json")):
        data = _load_run(metrics_path)
        if data is None:
            continue
        run_name = data["run_name"]
        runs.append(
            {
                "run_name": run_name,
                "model_type": data.get("model_type", "unknown"),
                "n_params": int(data.get("num_params", 0)),
                "metrics": data["metrics_per_split"],
                "metrics_path": metrics_path,
                "predictions_path": predictions_dir / f"{run_name}.csv",
                "config_path": metrics_dir / f"config_{run_name}.json",
                "history_path": metrics_dir / f"history_{run_name}.csv",
            }
        )
    return runs


def _hp_summary(cfg: dict[str, Any] | None) -> str:
    """One-line hyperparameter summary suitable for the comparison CSV."""
    if not cfg:
        return ""
    model = cfg.get("model", {})
    enc = model.get("encoder", {})
    training = cfg.get("training", {})
    bits: list[str] = []
    if "hidden_channels" in enc:
        bits.append(f"h={enc['hidden_channels']}")
    if "num_layers" in enc:
        bits.append(f"L={enc['num_layers']}")
    if "heads" in enc:
        bits.append(f"heads={enc['heads']}")
    if "aggr" in enc:
        bits.append(f"aggr={enc['aggr']}")
    if "lr" in training:
        bits.append(f"lr={training['lr']}")
    if "weight_decay" in training:
        bits.append(f"wd={training['weight_decay']}")
    if "hidden" in model:  # MLP baseline
        bits.append(f"h={model['hidden']}")
    return ",".join(bits)


# ---------------------------------------------------------------------------
# Per-run artifact regeneration
# ---------------------------------------------------------------------------


def _process_run(run: dict[str, Any], tables_dir: Path, figures_dir: Path) -> None:
    """Regenerate every per-run figure + table from saved artifacts."""
    run_name = run["run_name"]
    preds_path = run["predictions_path"]
    if not preds_path.exists():
        log.warning("predictions CSV missing for %s: %s", run_name, preds_path)
        return

    preds = pd.read_csv(preds_path)
    test = preds[preds["split"] == "test"].reset_index(drop=True)
    if test.empty:
        log.warning("no test rows for %s; skipping", run_name)
        return

    y_true = test["y_true"].to_numpy().astype(int)
    y_pred = test["y_pred"].to_numpy().astype(int)
    y_score = test["y_score"].to_numpy().astype(float)

    # Provide a subreddit_norm column from the source/target ids so the
    # plot/error helpers don't need the parquet.
    test_with_names = test.copy()
    test_with_names["source_subreddit_norm"] = "sr_" + test_with_names["source_id"].astype(str)
    test_with_names["target_subreddit_norm"] = "sr_" + test_with_names["target_id"].astype(str)

    # --- Tables
    src_degree = test.groupby("source_id")["source_id"].transform("size").to_numpy()
    deg_df = errors_by_degree_bin(test, y_true, y_pred, src_degree, n_bins=10)
    time_df = errors_by_time_bin(test, y_true, y_pred, n_bins=10, time_col="timestamp")
    subreddit_buckets = errors_by_subreddit(test_with_names, y_true, y_pred, top_k=20)
    examples = confusion_examples(test_with_names, y_true, y_pred, n=10)

    deg_df.to_csv(tables_dir / f"errors_by_degree_{run_name}.csv", index=False)
    time_df.to_csv(tables_dir / f"errors_by_time_{run_name}.csv", index=False)
    subreddit_buckets["top_fp_subreddits"].assign(kind="fp").pipe(
        lambda d: pd.concat([d, subreddit_buckets["top_fn_subreddits"].assign(kind="fn")])
    ).to_csv(tables_dir / f"errors_by_subreddit_{run_name}.csv", index=False)
    pd.concat([df.assign(cell=cell) for cell, df in examples.items()], ignore_index=True).to_csv(
        tables_dir / f"confusion_examples_{run_name}.csv", index=False
    )

    # --- Figures
    metrics = classification_metrics(y_true, y_score)
    cm = metrics["confusion_matrix"]
    plot_confusion_matrix(cm, normalize=False, save_path=figures_dir / f"confusion_{run_name}.png")
    plt.close("all")

    if len(np.unique(y_true)) == 2:
        plot_pr_roc(y_true, y_score, save_path=figures_dir / f"pr_roc_{run_name}.png")
        plt.close("all")

    plot_error_by_degree_bin(
        src_degree,
        y_true,
        y_pred,
        n_bins=10,
        save_path=figures_dir / f"error_by_degree_{run_name}.png",
    )
    plt.close("all")

    # Error-by-time as a simple bar chart.
    if not time_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(time_df)), time_df["error_rate"], color="#4c72b0", edgecolor="black")
        ax.set_xticks(range(len(time_df)))
        ax.set_xticklabels(
            [f"{pd.Timestamp(t).date()}" for t in time_df["time_min"]], rotation=45, ha="right"
        )
        ax.set_ylabel("error rate")
        ax.set_title(f"Error rate by time bin — {run_name}")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(figures_dir / f"error_by_time_{run_name}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    # Predicted subgraph (deterministic sample of 100 edges).
    plot_predicted_subgraph(
        test_with_names,
        y_true,
        y_pred,
        max_edges=100,
        seed=42,
        save_path=figures_dir / f"predicted_subgraph_{run_name}.png",
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# Cross-model aggregation: comparison.csv
# ---------------------------------------------------------------------------


def _read_config(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _build_comparison(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-run records into one row per model.

    Point estimates (``train_f1``, ``test_pr_auc_neg``, …) are averaged across
    **every** run of the model_type so the table includes information from
    tuning sweeps too. ``mean_seed`` / ``std_seed`` are computed from the
    subset of runs whose ``run_name`` ends in ``-seed<int>`` — these are the
    multi-seed retrain runs that share a single fixed configuration. When no
    seed-tagged runs exist for a model_type, we fall back to all of its runs.
    """
    if not runs:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    import re as _re

    seed_pattern = _re.compile(r"-seed\d+$")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        grouped[r["model_type"]].append(r)

    rows: list[dict[str, Any]] = []
    for model_type, group in grouped.items():
        seed_only = [r for r in group if seed_pattern.search(r["run_name"])]
        seed_group = seed_only if seed_only else group
        seed_pr_auc = [g["metrics"].get("test", {}).get("pr_auc", np.nan) for g in seed_group]
        # Extract per-seed point estimates of every column.
        train_f1s = [g["metrics"].get("train", {}).get("f1_macro", np.nan) for g in group]
        val_f1s = [g["metrics"].get("val", {}).get("f1_macro", np.nan) for g in group]
        test_f1s = [g["metrics"].get("test", {}).get("f1_macro", np.nan) for g in group]
        test_pr_auc = [g["metrics"].get("test", {}).get("pr_auc", np.nan) for g in group]
        test_roc_auc = [g["metrics"].get("test", {}).get("roc_auc", np.nan) for g in group]
        test_bacc = [g["metrics"].get("test", {}).get("balanced_accuracy", np.nan) for g in group]
        test_mcc = [g["metrics"].get("test", {}).get("mcc", np.nan) for g in group]
        n_params = [int(g.get("n_params", 0)) for g in group]
        configs = [_read_config(g["config_path"]) for g in group]

        rows.append(
            {
                "model": model_type,
                "hp_summary": _hp_summary(configs[0]) if configs else "",
                "train_f1": float(np.nanmean(train_f1s)),
                "val_f1": float(np.nanmean(val_f1s)),
                "test_f1": float(np.nanmean(test_f1s)),
                "test_pr_auc_neg": float(np.nanmean(test_pr_auc)),
                "test_roc_auc": float(np.nanmean(test_roc_auc)),
                "test_balanced_acc": float(np.nanmean(test_bacc)),
                "test_mcc": float(np.nanmean(test_mcc)),
                "n_params": int(np.nanmean(n_params)),
                "mean_seed": float(np.nanmean(seed_pr_auc)),
                "std_seed": (
                    float(np.nanstd(seed_pr_auc, ddof=0)) if len(seed_pr_auc) > 1 else 0.0
                ),
            }
        )

    df = pd.DataFrame(rows)
    df = df[COMPARISON_COLUMNS]
    # Sort by primary metric, highest first.
    return df.sort_values("test_pr_auc_neg", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-model: agreement + bar chart of test metric
# ---------------------------------------------------------------------------


def _build_model_agreement_artifacts(
    runs: list[dict[str, Any]], tables_dir: Path, figures_dir: Path
) -> None:
    """Compute pairwise agreement / kappa across one representative run per model."""
    if not runs:
        return

    representatives: dict[str, dict[str, Any]] = {}
    for r in runs:
        # Pick the run with the highest test PR-AUC per model as the representative.
        mt = r["model_type"]
        score = r["metrics"].get("test", {}).get("pr_auc", float("-inf"))
        if mt not in representatives or score > representatives[mt]["score"]:
            representatives[mt] = {"score": score, "run": r}

    preds_dict: dict[str, np.ndarray] = {}
    for mt, repr_ in representatives.items():
        preds_path = repr_["run"]["predictions_path"]
        if not preds_path.exists():
            continue
        df = pd.read_csv(preds_path)
        test = (
            df[df["split"] == "test"]
            .sort_values(["source_id", "target_id", "timestamp"])
            .reset_index(drop=True)
        )
        preds_dict[mt] = test["y_pred"].to_numpy().astype(int)

    if len(preds_dict) < 2:
        return

    # Trim to the minimum length so all vectors align (in case different runs
    # produced different test splits — they shouldn't, with the deterministic
    # chronological split, but be defensive).
    n_min = min(v.size for v in preds_dict.values())
    preds_dict = {k: v[:n_min] for k, v in preds_dict.items()}

    out = model_agreement(preds_dict)
    out["agreement"].to_csv(tables_dir / "model_agreement_agreement.csv")
    out["kappa"].to_csv(tables_dir / "model_agreement_kappa.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(out["kappa"].to_numpy(), cmap="RdBu", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(out["kappa"].columns)))
    ax.set_yticks(range(len(out["kappa"].index)))
    ax.set_xticklabels(out["kappa"].columns, rotation=45, ha="right")
    ax.set_yticklabels(out["kappa"].index)
    ax.set_title("Pairwise Cohen's κ on test predictions")
    for i in range(out["kappa"].shape[0]):
        for j in range(out["kappa"].shape[1]):
            val = float(out["kappa"].iat[i, j])
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if abs(val) > 0.5 else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(figures_dir / "model_agreement.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    predictions_dir: Path = typer.Option(
        PATHS.predictions, "--predictions-dir", help="Directory with per-run predictions CSVs."
    ),
    metrics_dir: Path = typer.Option(
        PATHS.reports_tables, "--metrics-dir", help="Directory with per-run metrics JSONs."
    ),
    figures_dir: Path = typer.Option(
        PATHS.reports_figures, "--figures-dir", help="Where to write report figures."
    ),
    tables_dir: Path = typer.Option(
        PATHS.reports_tables, "--tables-dir", help="Where to write report tables."
    ),
) -> None:
    """Rebuild every figure and table in ``reports/`` from saved run artifacts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    runs = _find_run_files(metrics_dir, predictions_dir)
    log.info("Found %d run(s) to process", len(runs))

    # Per-run regeneration.
    for r in runs:
        log.info("Processing %s (%s)", r["run_name"], r["model_type"])
        try:
            _process_run(r, tables_dir, figures_dir)
        except Exception as exc:  # pragma: no cover — surface the run name but keep going
            log.warning("Failed to process %s: %s", r["run_name"], exc)

    # Cross-model aggregation.
    comparison = _build_comparison(runs)
    comparison_path = tables_dir / "comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    log.info("Wrote %s (%d row(s))", comparison_path, len(comparison))

    if not comparison.empty:
        plot_model_comparison(
            comparison.rename(columns={"test_pr_auc_neg": "pr_auc_neg"}),
            metric="pr_auc_neg",
            save_path=figures_dir / "model_comparison_pr_auc_neg.png",
        )
        plt.close("all")
        plot_model_comparison(
            comparison.rename(columns={"test_f1": "f1_macro"}),
            metric="f1_macro",
            save_path=figures_dir / "model_comparison_f1_macro.png",
        )
        plt.close("all")
        _build_model_agreement_artifacts(runs, tables_dir, figures_dir)

    console.print(f"[bold green]Done.[/bold green] {len(runs)} run(s) processed.")
    console.print(f"  comparison : {comparison_path}")
    console.print(f"  figures    : {figures_dir}")
    console.print(f"  tables     : {tables_dir}")


if __name__ == "__main__":
    app()
