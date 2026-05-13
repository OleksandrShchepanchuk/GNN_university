"""Download SNAP files and run the full preprocessing pipeline.

Usage:
    python scripts/prepare_data.py [--force-download]

This is a thin Typer CLI wrapper around :mod:`reddit_gnn.data`. All real work
lives in the library so notebooks can reuse the same code paths.
"""

from __future__ import annotations

import typer

from reddit_gnn.utils.logging import get_logger

app = typer.Typer(add_completion=False, help=__doc__)
log = get_logger(__name__)


@app.command()
def main(
    force_download: bool = typer.Option(
        False, "--force-download", help="Re-download even if files already exist."
    ),
    drop_missing_embeddings: bool = typer.Option(
        False, help="Drop edges whose endpoints lack LIWC embeddings."
    ),
) -> None:
    """Entry point. Currently a stub — wires up to data pipeline once implemented."""
    log.info("scripts/prepare_data.py is a stub; data pipeline not implemented yet.")
    raise NotImplementedError(
        "scripts.prepare_data is not implemented yet. "
        "Wire it up to reddit_gnn.data.{download, load, preprocess, features, pyg_dataset}."
    )


if __name__ == "__main__":
    app()
