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
    plot_cross_metric_comparison,
    plot_cross_model_confusion_grid,
    plot_cross_model_pr_curves,
    plot_cross_model_roc_curves,
    plot_error_by_degree_bin,
    plot_model_comparison,
    plot_pr_roc,
    plot_predicted_subgraph,
    plot_threshold_tradeoff,
)

app = typer.Typer(add_completion=False, help=__doc__)
log = get_logger(__name__)
console = Console()


COMPARISON_COLUMNS = [
    "model",
    "hp_summary",
    "n_seeds",
    "n_params",
    "test_pr_auc_neg",
    "test_pr_auc_neg_std",
    "test_pr_auc_lift",
    "test_pr_auc_lift_std",
    "test_roc_auc",
    "test_roc_auc_std",
    "test_balanced_accuracy",
    "test_balanced_accuracy_std",
    "test_mcc",
    "test_mcc_std",
    "test_f1_macro",
    "test_f1_macro_std",
    "test_f1_negative_class",
    "test_f1_negative_class_std",
    "test_precision_negative",
    "test_precision_negative_std",
    "test_recall_negative",
    "test_recall_negative_std",
    "test_accuracy",
    "test_accuracy_std",
    # Validation side
    "val_pr_auc_neg",
    "val_pr_auc_neg_std",
    "val_f1_macro",
    "val_f1_macro_std",
    "class_prior_negative",
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
    """Aggregate per-run records into one row per model_type.

    ONLY the multi-seed retrain runs (``<model_type>-seed{0,1,2}``) feed the
    mean ± std columns. Single-shot runs (defaults, smoke tests, hp tuning
    grid) are ignored. This keeps the headline comparison clean: every row
    represents the same fixed configuration trained with 3 different model
    init seeds against the *same* (frozen) MP partition.

    Models with fewer than 1 seed-tagged run are omitted entirely — the
    table cannot honestly report mean ± std otherwise.
    """
    if not runs:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    import re as _re

    seed_pattern = _re.compile(r"-seed\d+$")

    # When a model has both the original `<model>-seed{0,1,2}` series AND a
    # newer hotfix series (e.g. `sage-h64-seed{0,1,2}` after the seed-stability
    # fix), prefer the hotfix series — the bug investigation that produced
    # them is the authoritative configuration we want reported in the
    # leaderboard. Listed in priority order: leftmost tag wins per model_type.
    PREFERRED_TAGS = {
        "sage": ["-h64-"],
        "gat": ["-fix-"],
        # GATv1's warmup=25 series escaped the lottery-init pattern; report it
        # as a separate leaderboard row from the GATv2 entry so the static-vs-
        # dynamic-attention comparison sits inside the same table.
        "gat_v1": ["-warmup25-", "-fix-"],
        "signed_gcn": ["-warmup-"],
        "baseline_mlp": ["-warmup-"],
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        if seed_pattern.search(r["run_name"]):
            grouped[r["model_type"]].append(r)

    for mt, tags in PREFERRED_TAGS.items():
        if mt not in grouped:
            continue
        for tag in tags:
            preferred = [g for g in grouped[mt] if tag in g["run_name"]]
            if preferred:
                grouped[mt] = preferred
                break

    def _seed_arr(group, split_name: str, key: str) -> np.ndarray:
        return np.array(
            [g["metrics"].get(split_name, {}).get(key, np.nan) for g in group],
            dtype=float,
        )

    def _mean(a: np.ndarray) -> float:
        return float(np.nanmean(a)) if a.size else float("nan")

    def _std(a: np.ndarray) -> float:
        return float(np.nanstd(a, ddof=0)) if a.size > 1 else 0.0

    rows: list[dict[str, Any]] = []
    for model_type, group in grouped.items():
        if not group:
            continue
        # Per-seed arrays.
        test_pr_auc = _seed_arr(group, "test", "pr_auc")
        test_pr_auc_lift = _seed_arr(group, "test", "pr_auc_lift")
        test_roc = _seed_arr(group, "test", "roc_auc")
        test_bacc = _seed_arr(group, "test", "balanced_accuracy")
        test_mcc = _seed_arr(group, "test", "mcc")
        test_f1 = _seed_arr(group, "test", "f1_macro")
        test_f1_neg = _seed_arr(group, "test", "f1_negative_class")
        test_prec_neg = _seed_arr(group, "test", "precision_negative")
        test_rec_neg = _seed_arr(group, "test", "recall_negative")
        test_acc = _seed_arr(group, "test", "accuracy")
        val_pr_auc = _seed_arr(group, "val", "pr_auc")
        val_f1 = _seed_arr(group, "val", "f1_macro")
        class_prior = _seed_arr(group, "test", "class_prior_negative")
        n_params_arr = np.array([int(g.get("n_params", 0)) for g in group])
        configs = [_read_config(g["config_path"]) for g in group]

        rows.append(
            {
                "model": model_type,
                "hp_summary": _hp_summary(configs[0]) if configs else "",
                "n_seeds": len(group),
                "n_params": int(np.nanmean(n_params_arr)),
                "test_pr_auc_neg": _mean(test_pr_auc),
                "test_pr_auc_neg_std": _std(test_pr_auc),
                "test_pr_auc_lift": _mean(test_pr_auc_lift),
                "test_pr_auc_lift_std": _std(test_pr_auc_lift),
                "test_roc_auc": _mean(test_roc),
                "test_roc_auc_std": _std(test_roc),
                "test_balanced_accuracy": _mean(test_bacc),
                "test_balanced_accuracy_std": _std(test_bacc),
                "test_mcc": _mean(test_mcc),
                "test_mcc_std": _std(test_mcc),
                "test_f1_macro": _mean(test_f1),
                "test_f1_macro_std": _std(test_f1),
                "test_f1_negative_class": _mean(test_f1_neg),
                "test_f1_negative_class_std": _std(test_f1_neg),
                "test_precision_negative": _mean(test_prec_neg),
                "test_precision_negative_std": _std(test_prec_neg),
                "test_recall_negative": _mean(test_rec_neg),
                "test_recall_negative_std": _std(test_rec_neg),
                "test_accuracy": _mean(test_acc),
                "test_accuracy_std": _std(test_acc),
                "val_pr_auc_neg": _mean(val_pr_auc),
                "val_pr_auc_neg_std": _std(val_pr_auc),
                "val_f1_macro": _mean(val_f1),
                "val_f1_macro_std": _std(val_f1),
                "class_prior_negative": _mean(class_prior),
            }
        )

    df = pd.DataFrame(rows)
    df = df[COMPARISON_COLUMNS]
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

    # Build a {model_type -> seed-0 predictions CSV path} mapping that respects
    # the same preferred-tag filter `_build_comparison` applied (so confusion
    # / PR / ROC plots use the hotfix-tagged runs when present).
    import re as _re2

    _seed_pat = _re2.compile(r"-seed(\d+)$")
    pred_paths_by_model: dict[str, Path] = {}
    for r in runs:
        m = _seed_pat.search(r["run_name"])
        if not m or int(m.group(1)) != 0:
            continue
        mt = r["model_type"]
        if mt in comparison["model"].tolist():
            # Prefer the same hotfix tag the comparison row was built from.
            # Heuristic: pick the run whose name appears in any predictions
            # CSV that exists; otherwise fall back to bare seed0.
            pred = predictions_dir / f"{r['run_name']}.csv"
            if pred.exists():
                # If this is a hotfix run, it wins; otherwise keep whatever we have
                # unless nothing is set yet.
                is_hotfix = any(t.strip("-") in r["run_name"] for t in ["h64", "warmup", "fix"])
                if is_hotfix or mt not in pred_paths_by_model:
                    pred_paths_by_model[mt] = pred

    if not comparison.empty:
        plot_model_comparison(
            comparison.rename(columns={"test_pr_auc_neg": "pr_auc_neg"}),
            metric="pr_auc_neg",
            save_path=figures_dir / "model_comparison_pr_auc_neg.png",
        )
        plt.close("all")
        plot_model_comparison(
            comparison.rename(columns={"test_f1_macro": "f1_macro"}),
            metric="f1_macro",
            save_path=figures_dir / "model_comparison_f1_macro.png",
        )
        plt.close("all")

        # 2x3 cross-metric panel with class-prior baseline and lift annotations.
        plot_cross_metric_comparison(
            comparison,
            save_path=figures_dir / "cross_metric_comparison.png",
        )
        plt.close("all")

        # 1xN confusion-matrix grid (one representative seed per model).
        plot_cross_model_confusion_grid(
            predictions_dir,
            comparison,
            seed=0,
            pred_paths_by_model=pred_paths_by_model,
            save_path=figures_dir / "confusion_grid_all_models.png",
        )
        plt.close("all")

        # Cross-model PR curves on the negative class (overlay).
        plot_cross_model_pr_curves(
            predictions_dir,
            comparison,
            seed=0,
            pred_paths_by_model=pred_paths_by_model,
            save_path=figures_dir / "cross_model_pr_curves.png",
        )
        plt.close("all")

        # Cross-model ROC curves (overlay).
        plot_cross_model_roc_curves(
            predictions_dir,
            comparison,
            seed=0,
            pred_paths_by_model=pred_paths_by_model,
            save_path=figures_dir / "cross_model_roc_curves.png",
        )
        plt.close("all")

        # Threshold trade-off plot for the winning model.
        winner = comparison.iloc[0]["model"]
        plot_threshold_tradeoff(
            predictions_dir,
            winner,
            seed=0,
            predictions_csv=pred_paths_by_model.get(winner),
            save_path=figures_dir / f"threshold_tradeoff_{winner}.png",
        )
        plt.close("all")

        _build_model_agreement_artifacts(runs, tables_dir, figures_dir)

    console.print(f"[bold green]Done.[/bold green] {len(runs)} run(s) processed.")
    console.print(f"  comparison : {comparison_path}")
    console.print(f"  figures    : {figures_dir}")
    console.print(f"  tables     : {tables_dir}")


if __name__ == "__main__":
    app()
