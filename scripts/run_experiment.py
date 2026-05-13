"""Run a single training experiment from a YAML config.

Usage:
    python scripts/run_experiment.py --config configs/gcn.yaml

End-to-end:
    1. Load + merge YAML configs.
    2. Seed everything; init MLflow tracking (unless ``--no-tracking``).
    3. Load processed parquet + node mapping.
    4. Fit FeatureBuilder on TRAIN df only; transform per-split.
    5. chronological_edge_split -> build_message_passing_split -> build_pyg_data_per_split.
    6. Build the torch model from cfg.
    7. Build LinkNeighborLoaders.
    8. ``fit`` with AdamW + ReduceLROnPlateau + early stopping.
    9. Evaluate train / val / test on the best checkpoint.
    10. Save metrics JSON, predictions CSV, training-curve PNG, checkpoint, config.
    11. Print a rich summary table.

Notes:
    * Sklearn ``baseline_logreg`` and ``signed_gcn`` follow a different path
      from the standard mini-batched torch loop; they are documented as
      unsupported in this script for now and raise ``NotImplementedError``
      with a clear message.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from rich.console import Console
from rich.table import Table

from reddit_gnn.config import Paths
from reddit_gnn.data.features import FeatureBuilder
from reddit_gnn.data.preprocess import save_processed_dataset  # noqa: F401 — re-exported usage
from reddit_gnn.data.pyg_dataset import build_pyg_data_per_split
from reddit_gnn.data.splits import (
    build_message_passing_split,
    chronological_edge_split,
)
from reddit_gnn.models.edge_classifier import build_torch_model, parameter_count
from reddit_gnn.paths import PATHS
from reddit_gnn.seed import set_global_seed
from reddit_gnn.tracking import (
    ExperimentContext,
    init_tracking,
    log_params,
    log_text,
    set_tags,
)
from reddit_gnn.training.evaluate import evaluate_checkpoint
from reddit_gnn.training.loaders import make_link_loaders
from reddit_gnn.training.loops import fit
from reddit_gnn.utils.io import (
    load_json,
    load_yaml,
    save_json,
    save_metrics_json,
    save_predictions_csv,
)
from reddit_gnn.utils.logging import get_logger
from reddit_gnn.visualization.results import plot_training_curves

app = typer.Typer(add_completion=False, help=__doc__)
log = get_logger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Right-biased deep merge. Mutates a copy of ``base``."""
    out = {**base}
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_merged_config(path: Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    defaults = cfg.pop("defaults", None)
    if defaults:
        base_path = (
            Paths().project_root / defaults if not Path(defaults).is_absolute() else Path(defaults)
        )
        base = load_yaml(base_path)
        cfg = _deep_merge(base, cfg)
    return cfg


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _flatten_for_params(d: dict, *, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_for_params(v, prefix=key))
        else:
            out[key] = v
    return out


def _print_summary(model_type: str, n_params: int, metrics_per_split: dict[str, dict[str, float]]):
    table = Table(title=f"Run summary — {model_type}  ({n_params:,} params)")
    table.add_column("metric", style="bold")
    for split_name in ("train", "val", "test"):
        table.add_column(split_name, justify="right")
    for metric in (
        "pr_auc",
        "pr_auc_positive",
        "roc_auc",
        "f1_macro",
        "f1_negative_class",
        "balanced_accuracy",
        "mcc",
        "precision_negative",
        "recall_negative",
    ):
        row = [metric]
        for split in ("train", "val", "test"):
            value = metrics_per_split.get(split, {}).get(metric)
            row.append(
                f"{value:.4f}" if isinstance(value, (int, float)) and value is not None else "—"
            )
        table.add_row(*row)
    console.print(table)


def _predictions_dataframe(
    split_name: str,
    data,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> pd.DataFrame:
    src = data.edge_label_index[0].cpu().numpy()
    tgt = data.edge_label_index[1].cpu().numpy()
    times_ns = data.edge_label_time.cpu().numpy()
    times = pd.to_datetime(times_ns)
    return pd.DataFrame(
        {
            "split": split_name,
            "source_id": src,
            "target_id": tgt,
            "timestamp": times,
            "y_true": y_true.astype(int),
            "y_score": y_score.astype(float),
            "y_pred": (y_score >= 0.5).astype(int),
        }
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False, help="Path to YAML."),
    no_tracking: bool = typer.Option(
        False, "--no-tracking", help="Disable MLflow for this run (overrides config)."
    ),
    seed_override: int | None = typer.Option(None, "--seed", help="Override cfg seed."),
    device_override: str | None = typer.Option(None, "--device", help="Override device."),
) -> None:
    """Entry point — see module docstring."""
    cfg = _load_merged_config(config)

    seed = int(seed_override or cfg.get("training", {}).get("seed", 42))
    cfg.setdefault("training", {})["seed"] = seed
    if device_override is not None:
        cfg["training"]["device"] = device_override
    device = torch.device(cfg["training"].get("device", "cpu"))

    set_global_seed(seed)

    # ---- Tracking
    tracking_cfg = cfg.get("tracking", {})
    enabled = bool(tracking_cfg.get("enabled", True)) and not no_tracking
    init_tracking(
        tracking_uri=tracking_cfg.get("tracking_uri"),
        experiment_name=tracking_cfg.get("experiment_name", "reddit_signed"),
        enabled=enabled,
    )

    run_name_template = tracking_cfg.get("run_name_template", "{model_type}-{timestamp}")
    model_type = cfg["model"]["type"]
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_name = run_name_template.format(model_type=model_type, timestamp=timestamp)

    # Output paths
    PATHS.ensure()
    checkpoint_path = PATHS.checkpoints / f"{run_name}.pt"
    metrics_path = PATHS.reports_tables / f"metrics_{run_name}.json"
    predictions_path = PATHS.predictions / f"{run_name}.csv"
    history_path = PATHS.reports_tables / f"history_{run_name}.csv"
    curve_path = PATHS.reports_figures / f"curve_{run_name}.png"
    config_dump = PATHS.reports_tables / f"config_{run_name}.yaml"

    log.info("run_name=%s | model=%s | seed=%d | device=%s", run_name, model_type, seed, device)
    save_json(cfg, PATHS.reports_tables / f"config_{run_name}.json")
    Path(config_dump).write_text(json.dumps(cfg, indent=2, default=str))

    # ---- Load data
    edges_path = PATHS.data_processed / "edges.parquet"
    n2i_path = PATHS.data_processed / "node_to_id.json"
    if not edges_path.exists() or not n2i_path.exists():
        raise FileNotFoundError(
            "Processed dataset is missing. Run `make data` first to produce "
            "data/processed/edges.parquet and node_to_id.json."
        )
    df = pd.read_parquet(edges_path)
    node_to_id = load_json(n2i_path)
    num_nodes = int(max(node_to_id.values())) + 1
    log.info("Loaded %d edges, %d nodes", len(df), num_nodes)

    # ---- Splits (chronological)
    split = chronological_edge_split(df)
    splits = build_message_passing_split(df, split, seed=seed)

    # ---- Features (fit on TRAIN df only)
    train_df = df.iloc[split.train_idx]
    raw_dir = PATHS.data_raw
    embeddings_path = raw_dir / "web-redditEmbeddings-subreddits.csv"
    use_snap = embeddings_path.exists()

    fb = FeatureBuilder(use_snap_embeddings=use_snap)
    fb.fit(train_df, num_nodes=num_nodes, embeddings_path=embeddings_path if use_snap else None)
    node_features = fb.transform_node_features(train_df, num_nodes=num_nodes)
    edge_features = fb.transform_edge_features(df)
    log.info(
        "Features: x=%s, edge_features=%s (snap=%s)",
        tuple(node_features.shape),
        tuple(edge_features.shape),
        use_snap,
    )

    # ---- PyG Data per split
    data_per_split = build_pyg_data_per_split(df, node_features, edge_features, splits)

    # ---- Model
    if model_type in {"baseline_logreg", "signed_gcn"}:
        raise NotImplementedError(
            f"Model type {model_type!r} is not yet wired into scripts/run_experiment.py. "
            "Use one of: gcn, sage, gat, baseline_mlp."
        )

    model = build_torch_model(
        cfg,
        node_feature_dim=int(node_features.shape[1]),
        edge_feature_dim=int(edge_features.shape[1]),
    )
    n_params = parameter_count(model)
    log.info("Built %s with %d trainable params", model_type, n_params)

    # ---- Loaders
    sampler_cfg = cfg.get("training", {}).get("neighbor_sampler", {}) or {}
    loaders = make_link_loaders(
        data_per_split,
        num_neighbors=sampler_cfg.get("num_neighbors", [15, 10]),
        batch_size=int(sampler_cfg.get("batch_size", cfg["training"].get("batch_size", 2048))),
    )

    # ---- Tags / params
    tags = {
        "model_type": model_type,
        "seed": str(seed),
        "device": str(device),
        "run_name": run_name,
    }
    git_sha = _git_sha()
    if git_sha:
        tags["git_sha"] = git_sha

    flat_params = _flatten_for_params({"model": cfg["model"], "training": cfg["training"]})
    flat_params["num_params"] = n_params

    metrics_per_split: dict[str, dict[str, float]] = {}
    final_y: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    with ExperimentContext(run_name, tags=tags) as ctx:
        set_tags(tags)
        log_params(flat_params)
        log_text(json.dumps(cfg, indent=2, default=str), filename="config.json")

        result = fit(
            model,
            loaders,
            cfg,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
        )
        log.info(
            "fit complete: best epoch=%d, best val pr_auc(neg)=%.4f",
            result["best_epoch"],
            result["best_val_pr_auc"],
        )

        # Reload best checkpoint and evaluate every split.
        eval_results = evaluate_checkpoint(checkpoint_path, model, loaders, device=device)
        for split_name, out in eval_results.items():
            metrics_per_split[split_name] = {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in out["metrics"].items()
                if k != "confusion_matrix"
            }
            cm = out["metrics"].get("confusion_matrix")
            metrics_per_split[split_name]["confusion_matrix"] = (
                cm.tolist() if hasattr(cm, "tolist") else cm
            )
            final_y[split_name] = (out["y_true"], out["y_score"])

        # Mirror final metrics into MLflow as scalars (alongside per-epoch).
        from reddit_gnn.tracking import log_metrics as tracking_log_metrics

        scalar_metrics = {}
        for split_name, m in metrics_per_split.items():
            for k, v in m.items():
                if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                    scalar_metrics[f"{split_name}_{k}"] = float(v)
        tracking_log_metrics(scalar_metrics)

        # Save metrics JSON locally (mirrors to MLflow via save_metrics_json).
        save_metrics_json(
            {
                "run_name": run_name,
                "model_type": model_type,
                "num_params": n_params,
                "best_epoch": result["best_epoch"],
                "best_val_pr_auc": float(result["best_val_pr_auc"]),
                "metrics_per_split": metrics_per_split,
            },
            metrics_path,
        )

        # Predictions CSV (all splits stacked).
        pred_frames: list[pd.DataFrame] = []
        for split_name, (y_true, y_score) in final_y.items():
            pred_frames.append(
                _predictions_dataframe(split_name, data_per_split[split_name], y_true, y_score)
            )
        save_predictions_csv(pd.concat(pred_frames, ignore_index=True), predictions_path)

        # Training-curve figure.
        history = result["history"]
        history_dict = {
            "train_loss": [r["train_loss"] for r in history],
            "val_loss": [r["val_loss"] for r in history],
            "train_pr_auc": [],  # not tracked per-batch; left empty by design
            "val_pr_auc": [r["val_pr_auc"] for r in history],
        }
        try:
            fig = plot_training_curves(history_dict, metric="pr_auc", save_path=curve_path)
            from reddit_gnn.tracking import log_figure as tracking_log_figure

            tracking_log_figure(fig, filename=f"curve_{run_name}.png", artifact_path="figures")
        except Exception as exc:  # pragma: no cover — figure rendering shouldn't fail training
            log.warning("Could not render training curve: %s", exc)

        # Mirror the checkpoint to MLflow too.
        from reddit_gnn.tracking import log_model_state

        log_model_state(checkpoint_path, artifact_path="model")
        del ctx  # explicit; the with-statement will close it

    _print_summary(model_type, n_params, metrics_per_split)
    console.print(f"[bold green]Done.[/bold green] Run: [cyan]{run_name}[/cyan]")
    console.print(f"  checkpoint  : {checkpoint_path}")
    console.print(f"  metrics     : {metrics_path}")
    console.print(f"  predictions : {predictions_path}")
    console.print(f"  history     : {history_path}")
    console.print(f"  curve       : {curve_path}")


if __name__ == "__main__":
    app()
