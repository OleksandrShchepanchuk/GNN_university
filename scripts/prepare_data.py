"""Download the SNAP files and parse them into a single DataFrame.

Usage:
    python scripts/prepare_data.py [--network-type body|title|both] \\
        [--raw-dir DATA/RAW] [--processed-dir DATA/PROCESSED] \\
        [--sample-edges N] [--seed 42]

This iteration only chains *download + load + print head*; the heavier
preprocessing / feature engineering / PyG-Data construction land in follow-up
commits. ``--processed-dir`` is accepted but not yet written to.
"""

from __future__ import annotations

from pathlib import Path

import typer

from reddit_gnn.data.download import ensure_raw_files
from reddit_gnn.data.load import load_reddit_dataset
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
        help="Destination for processed artifacts (not yet written in this step).",
    ),
    sample_edges: int | None = typer.Option(
        None,
        "--sample-edges",
        help="If set, sub-sample this many edges (uniform, seeded) before printing.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Global seed used by the sampler and downstream RNGs.",
    ),
) -> None:
    """Download SNAP files, load them, and print the head of the merged frame."""
    set_global_seed(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    log.info("Ensuring SNAP raw files in %s", raw_dir)
    paths = ensure_raw_files(raw_dir)
    for fname, p in paths.items():
        log.info("  %s -> %s (%d bytes)", fname, p, p.stat().st_size)

    log.info("Loading network_type=%s from %s", network_type, raw_dir)
    df = load_reddit_dataset(raw_dir, network_type=network_type)
    log.info("Loaded %d rows; columns=%d", len(df), df.shape[1])

    if sample_edges is not None and sample_edges > 0 and sample_edges < len(df):
        log.info("Sub-sampling %d edges (seed=%d)", sample_edges, seed)
        df = df.sample(n=sample_edges, random_state=seed).reset_index(drop=True)

    label_counts = df["POST_LABEL"].value_counts(dropna=False).to_dict()
    log.info("POST_LABEL distribution: %s", label_counts)

    # Surface the first few rows (drop the wide p0..p85 columns from the head
    # preview so the terminal isn't unreadable).
    preview_cols = [c for c in df.columns if not c.startswith("p") or not c[1:].isdigit()]
    print(df[preview_cols].head().to_string())


if __name__ == "__main__":
    app()
