"""Download SNAP files, load them, and persist the cleaned + indexed edge table.

Usage:
    python scripts/prepare_data.py [--network-type body|title|both] \\
        [--raw-dir DATA/RAW] [--processed-dir DATA/PROCESSED] \\
        [--sample-edges N] [--seed 42]

The pipeline runs in three steps:
    1. :func:`reddit_gnn.data.download.ensure_raw_files` — idempotent download.
    2. :func:`reddit_gnn.data.load.load_reddit_dataset`  — parse TSVs into a frame.
    3. :func:`reddit_gnn.data.preprocess.preprocess_dataset` — clean, index, save.

``--sample-edges`` applies *after* cleaning, before the node mapping is built.
It is a debugging knob — the resulting parquet must not be used for reported
metrics. A loud warning is logged whenever it is active.
"""

from __future__ import annotations

from pathlib import Path

import typer

from reddit_gnn.data.download import ensure_raw_files
from reddit_gnn.data.load import load_reddit_dataset
from reddit_gnn.data.preprocess import preprocess_dataset
from reddit_gnn.paths import PATHS
from reddit_gnn.seed import set_global_seed
from reddit_gnn.utils.logging import get_logger

app = typer.Typer(add_completion=False, help=__doc__)
log = get_logger(__name__)


@app.command()
def main(
    network_type: str = typer.Option(
        "both",
        "--network-type",
        help="Which SNAP file(s) to load: 'body', 'title', or 'both'.",
    ),
    raw_dir: Path = typer.Option(
        PATHS.data_raw,
        "--raw-dir",
        help="Where SNAP files are downloaded / read from.",
    ),
    processed_dir: Path = typer.Option(
        PATHS.data_processed,
        "--processed-dir",
        help="Destination for the processed parquet + JSON sidecars.",
    ),
    sample_edges: int | None = typer.Option(
        None,
        "--sample-edges",
        help=(
            "DEBUG ONLY: keep only this many edges (uniform random on the "
            "cleaned frame). Do not use the resulting dataset for any reported "
            "metric."
        ),
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Global seed used by the sampler and downstream RNGs.",
    ),
) -> None:
    """Download, load, preprocess, and persist the SNAP edge table."""
    set_global_seed(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    log.info("Step 1/3: ensuring SNAP raw files in %s", raw_dir)
    paths = ensure_raw_files(raw_dir)
    for fname, p in paths.items():
        log.info("  %s -> %s (%d bytes)", fname, p, p.stat().st_size)

    log.info("Step 2/3: loading network_type=%s from %s", network_type, raw_dir)
    df = load_reddit_dataset(raw_dir, network_type=network_type)
    log.info("Loaded %d rows; columns=%d", len(df), df.shape[1])

    # Preview the raw frame, hiding the wide p* property columns.
    preview_cols = [c for c in df.columns if not (c.startswith("p") and c[1:].isdigit())]
    print(df[preview_cols].head().to_string())

    if sample_edges is not None and sample_edges > 0:
        log.warning(
            "DEBUG: --sample-edges=%d will subsample the cleaned frame; "
            "do NOT use the resulting parquet for the final report.",
            sample_edges,
        )

    log.info("Step 3/3: preprocessing -> %s", processed_dir)
    edges_path = preprocess_dataset(
        raw_dir,
        processed_dir,
        network_type=network_type,
        df=df,
        sample_edges=sample_edges,
        seed=seed,
    )
    log.info("Done. Edge parquet at %s", edges_path)


if __name__ == "__main__":
    app()
