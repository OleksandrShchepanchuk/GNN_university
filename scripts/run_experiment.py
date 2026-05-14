"""Run a single training experiment from a YAML config.

Usage:
    python scripts/run_experiment.py --config configs/gcn.yaml
    python scripts/run_experiment.py --config configs/gcn.yaml \
        --override training.lr=0.001 --override run.name=gcn-lr0.001

End-to-end:
    1. Load + merge YAML configs; apply --override flags.
    2. Seed everything; init MLflow tracking (unless ``--no-tracking``).
    3. Load processed parquet + node mapping.
    4. Fit FeatureBuilder on TRAIN df only; transform per-split.
    5. chronological_edge_split -> build_message_passing_split -> build_pyg_data_per_split.
    6. Dispatch on cfg["model"]["type"]:
         * ``baseline_logreg`` -> sklearn pipeline (one-shot fit + evaluate).
         * everything else     -> torch GNN / MLP loop with AdamW + early stopping.
    7. Save metrics JSON, predictions CSV, history CSV, training-curve PNG (GNN
       path) or model pickle (sklearn path), checkpoint, config.
    8. Print a rich summary table.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import time
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
from reddit_gnn.data.pyg_dataset import build_pyg_data_per_split
from reddit_gnn.data.splits import (
    build_message_passing_split,
    chronological_edge_split,
)
from reddit_gnn.models.baselines import (
    LogisticRegressionBaseline,
    LogisticRegressionWithNodeFeats,
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
from reddit_gnn.training.metrics import classification_metrics
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
# Config helpers (merge + override)
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Right-biased deep merge."""
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


def _parse_override_value(raw: str) -> Any:
    """Parse a ``--override KEY=VALUE`` RHS as int -> float -> bool -> str."""
    s = raw.strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    if s.lower() in {"null", "none"}:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _apply_override(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set ``cfg`` at ``dotted_key`` (e.g. ``training.lr``) to ``value`` in place."""
    parts = dotted_key.split(".")
    node = cfg
    for k in parts[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[parts[-1]] = value


def _apply_overrides(cfg: dict, overrides: list[str]) -> set[str]:
    """Apply every ``KEY=VALUE`` override; return the set of dotted keys touched."""
    touched: set[str] = set()
    for ov in overrides or []:
        if "=" not in ov:
            raise typer.BadParameter(f"--override must be KEY=VALUE; got {ov!r}")
        key, raw = ov.split("=", 1)
        key = key.strip()
        value = _parse_override_value(raw)
        _apply_override(cfg, key, value)
        touched.add(key)
        log.info("override: %s = %r", key, value)
    return touched


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
    src_ids: np.ndarray,
    tgt_ids: np.ndarray,
    times_ns: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split_name,
            "source_id": src_ids,
            "target_id": tgt_ids,
            "timestamp": pd.to_datetime(times_ns),
            "y_true": y_true.astype(int),
            "y_score": y_score.astype(float),
            "y_pred": (y_score >= 0.5).astype(int),
        }
    )


def _metrics_no_cm(metrics: dict[str, Any]) -> dict[str, Any]:
    """Drop the confusion matrix (it goes in a separate column) for the JSON dump."""
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if k == "confusion_matrix":
            out[k] = v.tolist() if hasattr(v, "tolist") else v
        elif isinstance(v, (int, float, np.floating)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Sklearn LogReg path
# ---------------------------------------------------------------------------


def _concat_endpoint_blocks(
    edge_attrs: np.ndarray,
    x: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """``[edge_attrs || x_src || x_tgt || |x_src - x_tgt| || x_src * x_tgt]``."""
    src = edge_index[0]
    tgt = edge_index[1]
    x_src = x[src]
    x_tgt = x[tgt]
    return np.concatenate([edge_attrs, x_src, x_tgt, np.abs(x_src - x_tgt), x_src * x_tgt], axis=1)


def _run_sklearn_logreg(
    cfg: dict,
    df: pd.DataFrame,
    splits: dict[str, dict[str, torch.Tensor]],
    feature_builder: FeatureBuilder,
    num_nodes: int,
    run_name: str,
    *,
    checkpoint_path: Path,
    metrics_path: Path,
    predictions_path: Path,
    history_path: Path,
    tags: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """sklearn LogisticRegression baseline. Fits once on TRAIN sup rows; evaluates all splits.

    * ``use_node_features=True``: concatenates endpoint-derived blocks
      (``LogisticRegressionWithNodeFeats``-style features).
    * ``use_node_features=False``: raw 93-D edge features only.

    Saves the same artifact shape as the torch path: metrics JSON,
    predictions CSV (all splits stacked), one-row history CSV with fit time,
    and a sklearn-model pickle in place of the checkpoint.
    """
    model_cfg = cfg.get("model", {})
    sklearn_kwargs = dict(model_cfg.get("sklearn", {}))
    use_node = bool(model_cfg.get("use_node_features", True))

    edge_features_full = feature_builder.transform_edge_features(df).numpy().astype(np.float32)
    node_features_full = (
        feature_builder.transform_node_features(
            df.iloc[splits["train"]["mp_idx"].cpu().numpy()], num_nodes=num_nodes
        )
        .numpy()
        .astype(np.float32)
    )

    src_full = df["source_id"].to_numpy().astype(np.int64)
    tgt_full = df["target_id"].to_numpy().astype(np.int64)
    label_full = df["label_binary"].to_numpy().astype(np.int64)
    time_full = df["TIMESTAMP"].astype("int64").to_numpy()

    train_sup_idx = splits["train"]["sup_idx"].cpu().numpy()
    val_sup_idx = splits["val"]["sup_idx"].cpu().numpy()
    test_sup_idx = splits["test"]["sup_idx"].cpu().numpy()

    def _features_for(idx: np.ndarray) -> np.ndarray:
        edge_attrs = edge_features_full[idx]
        if not use_node:
            return edge_attrs
        edge_index = np.stack([src_full[idx], tgt_full[idx]], axis=0)
        return _concat_endpoint_blocks(edge_attrs, node_features_full, edge_index)

    X_train = _features_for(train_sup_idx)
    y_train = label_full[train_sup_idx]
    X_val = _features_for(val_sup_idx)
    y_val = label_full[val_sup_idx]
    X_test = _features_for(test_sup_idx)
    y_test = label_full[test_sup_idx]

    log.info(
        "baseline_logreg: feature dims X_train=%s, X_val=%s, X_test=%s (use_node=%s)",
        X_train.shape,
        X_val.shape,
        X_test.shape,
        use_node,
    )

    if use_node:
        model = LogisticRegressionWithNodeFeats(**sklearn_kwargs)
        # We've already prepared the concatenated features ourselves above; the
        # class API also rebuilds them internally, so feed it raw edge_attrs +
        # the same x/edge_index it expects.
        edge_index_train = np.stack([src_full[train_sup_idx], tgt_full[train_sup_idx]], axis=0)
        t0 = time.perf_counter()
        model.fit(edge_features_full[train_sup_idx], node_features_full, edge_index_train, y_train)
        fit_seconds = time.perf_counter() - t0

        def _proba(idx: np.ndarray) -> np.ndarray:
            ei = np.stack([src_full[idx], tgt_full[idx]], axis=0)
            return model.predict_proba(edge_features_full[idx], node_features_full, ei)[:, 1]
    else:
        model = LogisticRegressionBaseline(**sklearn_kwargs)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        def _proba(idx: np.ndarray) -> np.ndarray:
            return model.predict_proba(edge_features_full[idx])[:, 1]

    log.info("baseline_logreg fit time: %.2fs", fit_seconds)

    metrics_per_split: dict[str, dict[str, Any]] = {}
    pred_frames: list[pd.DataFrame] = []
    for split_name, idx, y in [
        ("train", train_sup_idx, y_train),
        ("val", val_sup_idx, y_val),
        ("test", test_sup_idx, y_test),
    ]:
        scores = _proba(idx)
        m = classification_metrics(y, scores)
        metrics_per_split[split_name] = _metrics_no_cm(m)
        pred_frames.append(
            _predictions_dataframe(
                split_name, src_full[idx], tgt_full[idx], time_full[idx], y, scores
            )
        )

    # Save sklearn model as a pickle (sibling of where the .pt checkpoint
    # would land for torch models).
    pickle_path = checkpoint_path.with_suffix(".pkl")
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("wb") as f:
        pickle.dump(model, f)
    log.info("Saved sklearn model to %s", pickle_path)

    # Trivial one-row history (no epoch loop for sklearn).
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "fit_seconds": fit_seconds,
                "train_pr_auc": metrics_per_split["train"]["pr_auc"],
                "val_pr_auc": metrics_per_split["val"]["pr_auc"],
                "test_pr_auc": metrics_per_split["test"]["pr_auc"],
            }
        ]
    ).to_csv(history_path, index=False)

    # Metrics + predictions (matches GNN path's keys for the aggregator).
    save_metrics_json(
        {
            "run_name": run_name,
            "model_type": "baseline_logreg",
            "num_params": int(X_train.shape[1]) + 1,  # weights + bias
            "best_epoch": 1,
            "best_val_pr_auc": float(metrics_per_split["val"]["pr_auc"]),
            "fit_seconds": float(fit_seconds),
            "metrics_per_split": metrics_per_split,
        },
        metrics_path,
    )
    save_predictions_csv(pd.concat(pred_frames, ignore_index=True), predictions_path)

    # MLflow scalars + artifact.
    from reddit_gnn.tracking import log_artifact
    from reddit_gnn.tracking import log_metrics as tracking_log_metrics

    scalar_metrics = {}
    for split_name, m in metrics_per_split.items():
        for k, v in m.items():
            if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                scalar_metrics[f"{split_name}_{k}"] = float(v)
    tracking_log_metrics(scalar_metrics)
    log_artifact(pickle_path, artifact_path="model")
    del tags  # tags already set by the caller's ExperimentContext

    return metrics_per_split


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False, help="Path to YAML."),
    overrides: list[str] = typer.Option(
        [],
        "--override",
        help="KEY=VALUE config overrides (dotted keys; repeatable). Example: --override training.lr=0.001",
    ),
    no_tracking: bool = typer.Option(
        False, "--no-tracking", help="Disable MLflow for this run (overrides config)."
    ),
    seed_override: int | None = typer.Option(None, "--seed", help="Override cfg seed."),
    device_override: str | None = typer.Option(None, "--device", help="Override device."),
) -> None:
    """Entry point — see module docstring."""
    cfg = _load_merged_config(config)
    touched = _apply_overrides(cfg, overrides)

    seed = int(seed_override or cfg.get("training", {}).get("seed", 42))
    cfg.setdefault("training", {})["seed"] = seed
    if device_override is not None:
        cfg["training"]["device"] = device_override
    device = torch.device(cfg["training"].get("device", "cpu"))

    set_global_seed(seed)

    tracking_cfg = cfg.get("tracking", {})
    enabled = bool(tracking_cfg.get("enabled", True)) and not no_tracking
    init_tracking(
        tracking_uri=tracking_cfg.get("tracking_uri"),
        experiment_name=tracking_cfg.get("experiment_name", "reddit_signed"),
        enabled=enabled,
    )

    model_type = cfg["model"]["type"]
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    if "run.name" in touched and cfg.get("run", {}).get("name"):
        run_name = str(cfg["run"]["name"])
    else:
        template = tracking_cfg.get("run_name_template", "{model_type}-{timestamp}")
        run_name = template.format(model_type=model_type, timestamp=timestamp)

    PATHS.ensure()
    checkpoint_path = PATHS.checkpoints / f"{run_name}.pt"
    metrics_path = PATHS.reports_tables / f"metrics_{run_name}.json"
    predictions_path = PATHS.predictions / f"{run_name}.csv"
    history_path = PATHS.reports_tables / f"history_{run_name}.csv"
    curve_path = PATHS.reports_figures / f"curve_{run_name}.png"

    log.info("run_name=%s | model=%s | seed=%d | device=%s", run_name, model_type, seed, device)
    save_json(cfg, PATHS.reports_tables / f"config_{run_name}.json")

    # ---- Load processed parquet
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

    # ---- Splits + features (train-only fit)
    # NOTE: partition_seed is intentionally decoupled from training.seed. The
    # MP/sup random partition is part of the *data*, not part of the model, so
    # multi-seed retrains must see the same graph; otherwise different seeds
    # also sample different splits and the resulting variance conflates
    # init-noise with graph-structure-noise. See decision #7 in the README.
    partition_seed = int(cfg.get("data", {}).get("partition_seed", 42))
    split = chronological_edge_split(df)
    splits = build_message_passing_split(df, split, seed=partition_seed)
    log.info(
        "splits: partition_seed=%d  training.seed=%d (decoupled)",
        partition_seed,
        seed,
    )
    train_df = df.iloc[split.train_idx]
    raw_dir = PATHS.data_raw
    embeddings_path = raw_dir / "web-redditEmbeddings-subreddits.csv"
    use_snap = embeddings_path.exists()

    fb = FeatureBuilder(use_snap_embeddings=use_snap)
    fb.fit(train_df, num_nodes=num_nodes, embeddings_path=embeddings_path if use_snap else None)

    # ---- Common tags / params
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

    metrics_per_split: dict[str, dict[str, Any]] = {}
    n_params = 0

    with ExperimentContext(run_name, tags=tags) as ctx:
        set_tags(tags)
        log_params(flat_params)
        log_text(json.dumps(cfg, indent=2, default=str), filename="config.json")

        if model_type == "baseline_logreg":
            metrics_per_split = _run_sklearn_logreg(
                cfg,
                df,
                splits,
                fb,
                num_nodes,
                run_name,
                checkpoint_path=checkpoint_path,
                metrics_path=metrics_path,
                predictions_path=predictions_path,
                history_path=history_path,
                tags=tags,
            )
            n_params = int(
                fb.transform_edge_features(df.head(1)).shape[1]
                * (5 if cfg["model"].get("use_node_features", True) else 1)
            )
        elif model_type == "signed_gcn":
            raise NotImplementedError(
                "signed_gcn end-to-end is not yet wired into scripts/run_experiment.py. "
                "Use one of: baseline_logreg, baseline_mlp, gcn, sage, gat."
            )
        else:
            metrics_per_split, n_params = _run_torch_path(
                cfg=cfg,
                df=df,
                splits=splits,
                feature_builder=fb,
                num_nodes=num_nodes,
                model_type=model_type,
                device=device,
                run_name=run_name,
                checkpoint_path=checkpoint_path,
                metrics_path=metrics_path,
                predictions_path=predictions_path,
                history_path=history_path,
                curve_path=curve_path,
            )
        del ctx  # explicit close happens via __exit__

    _print_summary(model_type, n_params, metrics_per_split)
    console.print(f"[bold green]Done.[/bold green] Run: [cyan]{run_name}[/cyan]")
    console.print(f"  checkpoint  : {checkpoint_path}")
    console.print(f"  metrics     : {metrics_path}")
    console.print(f"  predictions : {predictions_path}")
    console.print(f"  history     : {history_path}")
    console.print(f"  curve       : {curve_path}")


# ---------------------------------------------------------------------------
# Torch GNN / MLP path
# ---------------------------------------------------------------------------


def _run_torch_path(
    *,
    cfg: dict,
    df: pd.DataFrame,
    splits: dict[str, dict[str, torch.Tensor]],
    feature_builder: FeatureBuilder,
    num_nodes: int,
    model_type: str,
    device: torch.device,
    run_name: str,
    checkpoint_path: Path,
    metrics_path: Path,
    predictions_path: Path,
    history_path: Path,
    curve_path: Path,
) -> tuple[dict[str, dict[str, Any]], int]:
    node_features = feature_builder.transform_node_features(
        df.iloc[splits["train"]["mp_idx"].cpu().numpy()], num_nodes=num_nodes
    )
    edge_features = feature_builder.transform_edge_features(df)
    log.info(
        "Features: x=%s, edge_features=%s",
        tuple(node_features.shape),
        tuple(edge_features.shape),
    )

    data_per_split = build_pyg_data_per_split(df, node_features, edge_features, splits)
    model = build_torch_model(
        cfg,
        node_feature_dim=int(node_features.shape[1]),
        edge_feature_dim=int(edge_features.shape[1]),
    )
    n_params = parameter_count(model)
    log.info("Built %s with %d trainable params", model_type, n_params)

    sampler_cfg = cfg.get("training", {}).get("neighbor_sampler", {}) or {}
    loaders = make_link_loaders(
        data_per_split,
        num_neighbors=sampler_cfg.get("num_neighbors", [15, 10]),
        batch_size=int(sampler_cfg.get("batch_size", cfg["training"].get("batch_size", 2048))),
    )

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

    eval_results = evaluate_checkpoint(checkpoint_path, model, loaders, device=device)
    metrics_per_split: dict[str, dict[str, Any]] = {}
    pred_frames: list[pd.DataFrame] = []
    for split_name, out in eval_results.items():
        metrics_per_split[split_name] = _metrics_no_cm(out["metrics"])
        data = data_per_split[split_name]
        pred_frames.append(
            _predictions_dataframe(
                split_name,
                data.edge_label_index[0].cpu().numpy(),
                data.edge_label_index[1].cpu().numpy(),
                data.edge_label_time.cpu().numpy(),
                out["y_true"],
                out["y_score"],
            )
        )

    from reddit_gnn.tracking import (
        log_figure as tracking_log_figure,
    )
    from reddit_gnn.tracking import (
        log_metrics as tracking_log_metrics,
    )
    from reddit_gnn.tracking import (
        log_model_state,
    )

    scalar_metrics = {}
    for split_name, m in metrics_per_split.items():
        for k, v in m.items():
            if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                scalar_metrics[f"{split_name}_{k}"] = float(v)
    tracking_log_metrics(scalar_metrics)

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
    save_predictions_csv(pd.concat(pred_frames, ignore_index=True), predictions_path)

    history = result["history"]
    history_dict = {
        "train_loss": [r["train_loss"] for r in history],
        "val_loss": [r["val_loss"] for r in history],
        "train_pr_auc": [],
        "val_pr_auc": [r["val_pr_auc"] for r in history],
    }
    try:
        fig = plot_training_curves(history_dict, metric="pr_auc", save_path=curve_path)
        tracking_log_figure(fig, filename=f"curve_{run_name}.png", artifact_path="figures")
    except Exception as exc:  # pragma: no cover
        log.warning("Could not render training curve: %s", exc)

    log_model_state(checkpoint_path, artifact_path="model")
    return metrics_per_split, n_params


if __name__ == "__main__":
    app()
