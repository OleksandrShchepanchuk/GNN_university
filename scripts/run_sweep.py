"""Run an Optuna hyperparameter sweep over ``configs/sweep.yaml``.

Usage:
    python scripts/run_sweep.py --config configs/sweep.yaml
"""

from __future__ import annotations

from pathlib import Path

import typer

from reddit_gnn.utils.logging import get_logger

app = typer.Typer(add_completion=False, help=__doc__)
log = get_logger(__name__)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/sweep.yaml"), help="Sweep YAML."),
    n_trials: int | None = typer.Option(None, help="Override study.n_trials."),
) -> None:
    """Entry point. Stub until the sweep glue is implemented."""
    log.info("scripts/run_sweep.py is a stub; sweep not implemented yet.")
    raise NotImplementedError(
        "scripts.run_sweep is not implemented yet. Hook into reddit_gnn.training.tune.run_study."
    )


if __name__ == "__main__":
    app()
