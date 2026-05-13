"""Parse the SNAP TSV/CSV files into pandas DataFrames.

The two hyperlink TSVs share the schema::

    SOURCE_SUBREDDIT \\t TARGET_SUBREDDIT \\t POST_ID \\t TIMESTAMP \\t POST_LABEL \\t POST_PROPERTIES

* ``POST_LABEL`` is ``-1`` or ``+1``; remapping to the trained ``{0, 1}`` is
  deferred to :mod:`reddit_gnn.data.preprocess` (so the raw label values stay
  visible at load time for sanity checks).
* ``POST_PROPERTIES`` is a comma-separated 86-D vector of LIWC/text features.
  This module always expands it into 86 numeric columns named ``p0`` … ``p85``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reddit_gnn.data.download import BODY_FILENAME, TITLE_FILENAME
from reddit_gnn.utils.logging import get_logger

log = get_logger(__name__)

POST_PROPERTIES_DIM = 86
EMBEDDING_DIM = 300

REQUIRED_COLUMNS = (
    "SOURCE_SUBREDDIT",
    "TARGET_SUBREDDIT",
    "POST_ID",
    "TIMESTAMP",
    "POST_LABEL",
    "POST_PROPERTIES",
)

# Some SNAP mirrors ship the body/title TSVs with renamed columns
# (``LINK_SENTIMENT`` instead of ``POST_LABEL``, ``PROPERTIES`` instead of
# ``POST_PROPERTIES``). We accept either name on the way in and normalize
# everything to the canonical schema documented above.
SNAP_COLUMN_ALIASES = {
    "LINK_SENTIMENT": "POST_LABEL",
    "PROPERTIES": "POST_PROPERTIES",
}

_PROPERTY_COLUMNS = tuple(f"p{i}" for i in range(POST_PROPERTIES_DIM))


def _expand_post_properties(series: pd.Series) -> tuple[pd.DataFrame, int]:
    """Vectorized parse of the ``POST_PROPERTIES`` column.

    Returns the ``(num_rows, 86)`` numeric DataFrame plus the count of cells
    that had to be coerced to ``0.0`` (invalid numerics, missing entries,
    or non-string input). Tolerates surrounding brackets / parentheses /
    whitespace and any value pandas can't parse as a float.
    """
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.strip("[]() \t")
    )
    expanded = cleaned.str.split(",", n=POST_PROPERTIES_DIM - 1, expand=True)

    if expanded.shape[1] < POST_PROPERTIES_DIM:
        for i in range(expanded.shape[1], POST_PROPERTIES_DIM):
            expanded[i] = np.nan
    expanded = expanded.iloc[:, :POST_PROPERTIES_DIM]
    expanded.columns = list(_PROPERTY_COLUMNS)

    numeric = expanded.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    invalid_mask = numeric.isna()
    n_invalid = int(invalid_mask.to_numpy().sum())
    numeric = numeric.fillna(0.0).astype(np.float32)
    return numeric, n_invalid


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case every column name (case-insensitive header)."""
    return df.rename(columns={c: c.strip().upper() for c in df.columns})


def parse_hyperlinks_tsv(path: str | Path, source_tag: str) -> pd.DataFrame:
    """Read one hyperlink TSV and return a tidy ``DataFrame``.

    Columns in the returned frame:
        * the six original (upper-cased) columns;
        * ``p0`` … ``p85`` — expanded ``POST_PROPERTIES``;
        * ``source_subreddit_norm`` / ``target_subreddit_norm`` — lowercased,
          stripped subreddit names (use these for graph indexing);
        * ``source_file`` — ``source_tag`` echoed onto every row.

    Rows with missing ``SOURCE_SUBREDDIT`` / ``TARGET_SUBREDDIT`` /
    ``TIMESTAMP`` / ``POST_LABEL`` are dropped.

    Invalid entries inside ``POST_PROPERTIES`` (non-numeric, NaN, ``[`` / ``]``
    wrapping, extra whitespace) are tolerated and replaced with ``0.0`` — a
    single summary warning is logged when any value had to be coerced.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing SNAP hyperlinks TSV: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    df = _normalize_columns(df)
    df = df.rename(columns=SNAP_COLUMN_ALIASES)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"TSV {path} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[list(REQUIRED_COLUMNS)].copy()

    props_df, n_invalid = _expand_post_properties(df["POST_PROPERTIES"])
    if n_invalid > 0:
        log.warning(
            "%s: %d POST_PROPERTIES cell(s) coerced to 0.0 across %d rows",
            path.name,
            n_invalid,
            len(df),
        )
    df = pd.concat([df.reset_index(drop=True), props_df.reset_index(drop=True)], axis=1)

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce").astype("datetime64[ns]")
    df["POST_LABEL"] = pd.to_numeric(df["POST_LABEL"], errors="coerce").astype("Int64")

    df["source_subreddit_norm"] = df["SOURCE_SUBREDDIT"].astype(str).str.strip().str.lower()
    df["target_subreddit_norm"] = df["TARGET_SUBREDDIT"].astype(str).str.strip().str.lower()
    df["source_file"] = source_tag

    before = len(df)
    df = df.dropna(subset=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "TIMESTAMP", "POST_LABEL"])
    df = df[
        (df["source_subreddit_norm"] != "")
        & (df["target_subreddit_norm"] != "")
    ]
    df = df.reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log.info("%s: dropped %d row(s) with missing essentials", path.name, dropped)

    return df


def load_reddit_dataset(
    raw_dir: str | Path,
    network_type: str = "both",
) -> pd.DataFrame:
    """Load body, title, or both TSVs as a single concatenated DataFrame.

    ``network_type`` must be one of ``{"body", "title", "both"}``. The returned
    frame always has a ``source_file`` column (``"body"`` or ``"title"``).
    """
    if network_type not in {"body", "title", "both"}:
        raise ValueError(
            f"network_type must be one of 'body', 'title', 'both'; got {network_type!r}"
        )
    raw = Path(raw_dir)
    paths = {"body": raw / BODY_FILENAME, "title": raw / TITLE_FILENAME}
    selected = ("body", "title") if network_type == "both" else (network_type,)

    frames: list[pd.DataFrame] = []
    for tag in selected:
        p = paths[tag]
        if not p.exists():
            raise FileNotFoundError(
                f"Missing SNAP file: {p}. "
                f"Run `python scripts/prepare_data.py` (or `make data`) to download it."
            )
        frames.append(parse_hyperlinks_tsv(p, source_tag=tag))

    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def parse_subreddit_embeddings(path: str | Path) -> tuple[dict[str, int], np.ndarray]:
    """Parse the LIWC subreddit embeddings CSV.

    Returns ``(name_to_idx, embedding_matrix)`` where:
        * ``name_to_idx`` maps lower-cased, stripped subreddit names to row
          indices in ``embedding_matrix``;
        * ``embedding_matrix`` has shape ``(n_subreddits, 300)``, dtype float32.

    Tolerates a leading header line (auto-detected by checking whether the
    second column of the first row is numeric).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing SNAP embeddings CSV: {path}")

    raw = pd.read_csv(path, header=None, dtype=str, low_memory=False)
    if raw.shape[1] < EMBEDDING_DIM + 1:
        raise ValueError(
            f"Embeddings file {path} has {raw.shape[1]} columns; expected at "
            f"least {EMBEDDING_DIM + 1} (subreddit + {EMBEDDING_DIM} floats)."
        )

    first_value = raw.iloc[0, 1]
    try:
        float(first_value)
        # First row is data.
    except (TypeError, ValueError):
        # First row is a header — drop it.
        raw = raw.iloc[1:].reset_index(drop=True)

    raw = raw.iloc[:, : EMBEDDING_DIM + 1]
    names = raw.iloc[:, 0].astype(str).str.strip().str.lower()
    values = (
        raw.iloc[:, 1 : EMBEDDING_DIM + 1]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
        .to_numpy()
    )

    # Deduplicate by name, keeping the first occurrence.
    seen: dict[str, int] = {}
    keep: list[int] = []
    for i, name in enumerate(names):
        if not name or name in seen:
            continue
        seen[name] = len(keep)
        keep.append(i)
    embedding_matrix = values[keep]
    name_to_idx = dict(seen)
    return name_to_idx, embedding_matrix


__all__ = [
    "EMBEDDING_DIM",
    "POST_PROPERTIES_DIM",
    "REQUIRED_COLUMNS",
    "SNAP_COLUMN_ALIASES",
    "load_reddit_dataset",
    "parse_hyperlinks_tsv",
    "parse_subreddit_embeddings",
]
