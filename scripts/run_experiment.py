"""Run a single training experiment from a YAML config.

Usage:
    python scripts/run_experiment.py --config configs/gcn.yaml

Merges ``configs/base.yaml`` under the chosen config (the chosen config wins
on conflicts), seeds everything, builds the dataset/model, fits, evaluates
on the held-out test split, and writes predictions + metrics under
``models/predictions/<run_name>/``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from reddit_gnn.utils.logging import get_logger

app = typer.Typer(add_completion=False, help=__doc__)
log = get_logger(__name__)


@app.command()
def main(
    config: Path = typer.Option(..., exists=False, help="Path to the experiment YAML."),
    seed: int | None = typer.Option(None, help="Override seed from config."),
    device: str | None = typer.Option(None, help="Override device ('cpu' / 'cuda')."),
) -> None:
    """Entry point. Stub until the training pipeline is implemented."""
    log.info("scripts/run_experiment.py is a stub; training not implemented yet.")
    raise NotImplementedError(
        "scripts.run_experiment is not implemented yet. "
        "Hook into reddit_gnn.training.loops.fit + reddit_gnn.training.evaluate."
    )


if __name__ == "__main__":
    app()
