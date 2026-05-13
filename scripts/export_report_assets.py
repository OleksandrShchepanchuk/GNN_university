"""Render figures and tables for the final report from a trained run.

Usage:
    python scripts/export_report_assets.py --run gcn

Reads ``models/predictions/<run>/`` and writes plots + LaTeX-ready tables to
``reports/figures/`` and ``reports/tables/``.
"""

from __future__ import annotations

import typer

from reddit_gnn.utils.logging import get_logger

app = typer.Typer(add_completion=False, help=__doc__)
log = get_logger(__name__)


@app.command()
def main(
    run: str = typer.Option(..., help="Name of the training run to export."),
) -> None:
    """Entry point. Stub until report exporters are implemented."""
    log.info("scripts/export_report_assets.py is a stub; nothing exported yet.")
    raise NotImplementedError(
        "scripts.export_report_assets is not implemented yet. "
        "Use reddit_gnn.visualization.* + reddit_gnn.training.evaluate."
    )


if __name__ == "__main__":
    app()
