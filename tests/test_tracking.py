"""Tests for ``reddit_gnn.tracking`` (the MLflow backend).

Each test points MLflow at its own ``tmp_path / mlruns`` directory so tests
don't pollute each other's run history and don't depend on whatever URI the
previous test left configured on the MLflow client.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pytest

from reddit_gnn.tracking import (
    ExperimentContext,
    active_run_id,
    end_run,
    init_tracking,
    log_artifact,
    log_figure,
    log_metrics,
    log_params,
    log_text,
    mlflow_backend,
    set_tags,
    start_run,
)


@pytest.fixture(autouse=True)
def _reset_tracking_state():
    """End any leftover mlflow run; reset the module-level enabled flag."""
    yield
    while mlflow.active_run() is not None:
        try:
            mlflow.end_run()
        except Exception:
            break
    mlflow_backend._ENABLED = False
    mlflow_backend._INITIALIZED = False


def _experiment_dir(mlruns_root: Path, experiment_name: str) -> Path:
    """Resolve the on-disk experiment directory by name (id is auto-assigned)."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    assert exp is not None, f"experiment {experiment_name!r} was not created"
    return mlruns_root / exp.experiment_id


# ---------------------------------------------------------------------------
# 1. Disabled tracking is a no-op
# ---------------------------------------------------------------------------


def test_disabled_tracking_is_noop(tmp_path, monkeypatch):
    """With ``enabled=False`` every helper must return without raising and
    without touching ``mlruns/`` under the current working directory."""
    monkeypatch.chdir(tmp_path)

    init_tracking(enabled=False)
    assert active_run_id() is None

    # The context manager and every helper should be safe to call.
    with ExperimentContext("disabled-smoke") as ctx:
        log_params({"a": 1, "b": "x"})
        log_metrics({"loss": 0.5}, step=0)
        log_text("hello", "notes.txt")
        set_tags({"role": "test"})
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        log_figure(fig, "noop.png")
        assert ctx.run_id is None

    # Nothing was created under cwd.
    assert not (tmp_path / "mlruns").exists()
    # And no run remains active.
    assert mlflow.active_run() is None


# ---------------------------------------------------------------------------
# 2. Local file store round-trip
# ---------------------------------------------------------------------------


def test_local_file_store(tmp_path):
    mlruns = tmp_path / "mlruns"
    init_tracking(tracking_uri=f"file:{mlruns}", experiment_name="reddit_signed")

    with ExperimentContext("smoke") as ctx:
        log_params({"a": 1, "model_type": "gcn"})
        log_metrics({"loss": 0.5}, step=0)
        run_id = ctx.run_id

    assert run_id is not None

    exp_dir = _experiment_dir(mlruns, "reddit_signed")
    params_file = exp_dir / run_id / "params" / "a"
    metrics_file = exp_dir / run_id / "metrics" / "loss"

    assert params_file.exists(), f"missing param file: {params_file}"
    assert params_file.read_text().strip() == "1"

    assert metrics_file.exists(), f"missing metric file: {metrics_file}"
    parts = metrics_file.read_text().strip().split()
    # File format is "<unix_timestamp_ms> <value> <step>".
    assert len(parts) == 3
    assert float(parts[1]) == pytest.approx(0.5)
    assert int(parts[2]) == 0

    # Cross-check via the MLflow API too.
    run = mlflow.get_run(run_id)
    assert run.data.params["a"] == "1"
    assert run.data.metrics["loss"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 3. log_figure produces an artifact
# ---------------------------------------------------------------------------


def test_log_figure(tmp_path):
    mlruns = tmp_path / "mlruns"
    init_tracking(tracking_uri=f"file:{mlruns}", experiment_name="reddit_signed")

    with ExperimentContext("fig") as ctx:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        log_figure(fig, "curve.png", artifact_path="figures")
        run_id = ctx.run_id

    assert run_id is not None
    artifact = (
        _experiment_dir(mlruns, "reddit_signed") / run_id / "artifacts" / "figures" / "curve.png"
    )
    assert artifact.exists(), f"figure artifact not written: {artifact}"
    assert artifact.stat().st_size > 0


# ---------------------------------------------------------------------------
# 4. Non-finite metric values are dropped, not raised
# ---------------------------------------------------------------------------


def test_log_metrics_drops_non_finite(tmp_path, caplog):
    mlruns = tmp_path / "mlruns"
    init_tracking(tracking_uri=f"file:{mlruns}", experiment_name="reddit_signed")

    with (
        caplog.at_level(logging.WARNING, logger="reddit_gnn.tracking.mlflow_backend"),
        ExperimentContext("nan-run") as ctx,
    ):
        log_metrics(
            {
                "good": 0.5,
                "bad_nan": float("nan"),
                "bad_inf": float("inf"),
                "bad_neg_inf": float("-inf"),
            }
        )
        run_id = ctx.run_id

    assert run_id is not None
    run = mlflow.get_run(run_id)
    assert "good" in run.data.metrics
    assert "bad_nan" not in run.data.metrics
    assert "bad_inf" not in run.data.metrics
    assert "bad_neg_inf" not in run.data.metrics

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("non-finite" in w.lower() for w in warnings), warnings


# ---------------------------------------------------------------------------
# 5. Nested runs and their parent/child link
# ---------------------------------------------------------------------------


def test_nested_run(tmp_path):
    mlruns = tmp_path / "mlruns"
    init_tracking(tracking_uri=f"file:{mlruns}", experiment_name="reddit_signed")

    parent_id: str | None = None
    child_id: str | None = None

    with ExperimentContext("parent") as parent_ctx:
        parent_id = parent_ctx.run_id
        with ExperimentContext("child", nested=True) as child_ctx:
            child_id = child_ctx.run_id
            log_params({"trial": 1})

    assert parent_id is not None
    assert child_id is not None
    assert parent_id != child_id

    child = mlflow.get_run(child_id)
    parent = mlflow.get_run(parent_id)
    assert child.data.tags.get("mlflow.parentRunId") == parent_id
    assert parent.data.tags.get("mlflow.parentRunId") is None
    # Sanity: both runs landed in the same experiment.
    assert child.info.experiment_id == parent.info.experiment_id


# ---------------------------------------------------------------------------
# 6. Bonus: io.save_metrics_json mirrors to MLflow when a run is active
# ---------------------------------------------------------------------------


def test_save_metrics_json_mirrors_to_mlflow(tmp_path):
    """The local write is the source of truth; MLflow gets a copy."""
    from reddit_gnn.utils.io import save_metrics_json

    mlruns = tmp_path / "mlruns"
    init_tracking(tracking_uri=f"file:{mlruns}", experiment_name="reddit_signed")

    local_path = tmp_path / "metrics.json"
    with ExperimentContext("io") as ctx:
        save_metrics_json({"f1": 0.8, "auc": 0.9}, local_path)
        run_id = ctx.run_id

    # Local file present
    assert local_path.exists()
    # And mirrored to MLflow
    assert run_id is not None
    artifact = (
        _experiment_dir(mlruns, "reddit_signed") / run_id / "artifacts" / "metrics" / "metrics.json"
    )
    assert artifact.exists()


# ---------------------------------------------------------------------------
# 7. Bonus: helpers outside an active run are quiet no-ops, not errors
# ---------------------------------------------------------------------------


def test_helpers_outside_active_run_are_silent(tmp_path):
    mlruns = tmp_path / "mlruns"
    init_tracking(tracking_uri=f"file:{mlruns}", experiment_name="reddit_signed")

    # No active run: every helper should return cleanly.
    log_params({"a": 1})
    log_metrics({"x": 1.0})
    log_text("note", "note.txt")
    set_tags({"k": "v"})
    log_artifact(tmp_path)  # path doesn't matter; helper short-circuits
    # And start_run / end_run still work afterwards.
    started = start_run("after-noop")
    assert started is not None
    end_run()
