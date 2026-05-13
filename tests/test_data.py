"""Tests for ``reddit_gnn.data.download`` and ``reddit_gnn.data.load``.

These exercise the TSV parser against synthetic in-memory files; no network
access is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reddit_gnn.data.download import BODY_FILENAME, REDDIT_HYPERLINK_URLS, TITLE_FILENAME
from reddit_gnn.data.load import (
    POST_PROPERTIES_DIM,
    load_reddit_dataset,
    parse_hyperlinks_tsv,
    parse_subreddit_embeddings,
)
from reddit_gnn.data.preprocess import remap_post_label

_HEADER = "SOURCE_SUBREDDIT\tTARGET_SUBREDDIT\tPOST_ID\tTIMESTAMP\tPOST_LABEL\tPOST_PROPERTIES"


def _properties_csv(values: list[float]) -> str:
    """Render 86 floats as a comma-separated CSV string."""
    if len(values) != POST_PROPERTIES_DIM:
        raise ValueError("synthetic POST_PROPERTIES must have 86 entries")
    return ",".join(str(v) for v in values)


def _write_tsv(path: Path, rows: list[tuple[str, ...]]) -> Path:
    """Write a TSV with the canonical header + the given rows."""
    lines = [_HEADER]
    for r in rows:
        lines.append("\t".join(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _row(
    src: str,
    dst: str,
    post_id: str,
    ts: str,
    label: int,
    props: str | None = None,
) -> tuple[str, ...]:
    if props is None:
        props = _properties_csv([0.1] * POST_PROPERTIES_DIM)
    return (src, dst, post_id, ts, str(label), props)


def test_imports() -> None:
    """Every data module must import cleanly even without raw files present."""
    from reddit_gnn.data import (  # noqa: F401
        download,
        features,
        load,
        preprocess,
        pyg_dataset,
        splits,
    )


def test_label_remap_constants_present() -> None:
    """The label-remap function must be a stable public symbol."""
    # Contract: -1 -> 0 (negative), +1 -> 1 (neutral/positive).
    assert remap_post_label(-1) == 0
    assert remap_post_label(1) == 1
    with pytest.raises(ValueError):
        remap_post_label(0)


def test_download_url_table_is_correct() -> None:
    """The three SNAP filenames must be wired to the documented URLs."""
    assert BODY_FILENAME in REDDIT_HYPERLINK_URLS
    assert TITLE_FILENAME in REDDIT_HYPERLINK_URLS
    body_url = REDDIT_HYPERLINK_URLS[BODY_FILENAME]
    title_url = REDDIT_HYPERLINK_URLS[TITLE_FILENAME]
    assert body_url.endswith(BODY_FILENAME)
    assert title_url.endswith(TITLE_FILENAME)


def test_parse_hyperlinks_tsv_synthetic_five_rows(tmp_path: Path) -> None:
    """Synthetic 5-row TSV → expected columns / dtypes / label set."""
    rows = [
        _row("askreddit", "news", "p0", "2014-01-01 12:00:00", 1),
        _row("AskReddit", "politics", "p1", "2014-02-15 03:30:00", -1),
        _row("gaming", "movies", "p2", "2014-03-10 09:45:00", 1),
        _row("Music", "books", "p3", "2014-04-20 18:00:00", -1),
        _row("science", "space", "p4", "2014-05-05 06:15:00", 1),
    ]
    path = _write_tsv(tmp_path / "synthetic.tsv", rows)

    df = parse_hyperlinks_tsv(path, source_tag="body")

    assert len(df) == 5

    # Expanded property columns
    prop_cols = [f"p{i}" for i in range(POST_PROPERTIES_DIM)]
    for col in prop_cols:
        assert col in df.columns
    assert df[prop_cols].shape == (5, POST_PROPERTIES_DIM)

    # Original columns preserved (uppercased)
    for col in (
        "SOURCE_SUBREDDIT",
        "TARGET_SUBREDDIT",
        "POST_ID",
        "TIMESTAMP",
        "POST_LABEL",
        "POST_PROPERTIES",
    ):
        assert col in df.columns

    # Derived columns
    assert "source_subreddit_norm" in df.columns
    assert "target_subreddit_norm" in df.columns
    assert "source_file" in df.columns
    assert (df["source_file"] == "body").all()

    # Subreddit names are lowercased and stripped
    assert df["source_subreddit_norm"].str.islower().all()
    assert df.loc[1, "source_subreddit_norm"] == "askreddit"

    # Timestamp dtype
    assert pd.api.types.is_datetime64_ns_dtype(df["TIMESTAMP"])

    # Labels are -1 / +1 only
    labels = set(df["POST_LABEL"].dropna().astype(int).unique().tolist())
    assert labels <= {-1, 1}


def test_parse_hyperlinks_tsv_tolerates_malformed_properties(tmp_path: Path) -> None:
    """Brackets and NaN inside POST_PROPERTIES must parse to 0.0 without raising."""
    # Row 0: brackets around the whole vector
    bracketed = "[" + _properties_csv([0.5] * POST_PROPERTIES_DIM) + "]"

    # Row 1: NaN at indices 3 and 7; "bad" at index 11
    nan_vec = [0.2] * POST_PROPERTIES_DIM
    parts = [str(v) for v in nan_vec]
    parts[3] = "NaN"
    parts[7] = "nan"
    parts[11] = "not-a-number"
    nan_props = ",".join(parts)

    # Row 2: a clean control row
    clean_props = _properties_csv([0.1] * POST_PROPERTIES_DIM)

    rows = [
        _row("a", "b", "x0", "2015-01-01 00:00:00", 1, bracketed),
        _row("c", "d", "x1", "2015-02-01 00:00:00", -1, nan_props),
        _row("e", "f", "x2", "2015-03-01 00:00:00", 1, clean_props),
    ]
    path = _write_tsv(tmp_path / "malformed.tsv", rows)

    df = parse_hyperlinks_tsv(path, source_tag="body")

    assert len(df) == 3

    prop_cols = [f"p{i}" for i in range(POST_PROPERTIES_DIM)]
    prop_frame = df[prop_cols].astype(np.float32)

    # Brackets stripped → values intact at ~0.5
    np.testing.assert_allclose(prop_frame.iloc[0].to_numpy(), 0.5, atol=1e-5)

    # NaN-inside row: offending positions are 0.0, the rest are ~0.2
    nan_row = prop_frame.iloc[1].to_numpy()
    assert nan_row[3] == 0.0
    assert nan_row[7] == 0.0
    assert nan_row[11] == 0.0
    np.testing.assert_allclose(nan_row[0], 0.2, atol=1e-5)
    np.testing.assert_allclose(nan_row[-1], 0.2, atol=1e-5)

    # Clean control row unaffected
    np.testing.assert_allclose(prop_frame.iloc[2].to_numpy(), 0.1, atol=1e-5)


def test_load_reddit_dataset_both_mode_merges_and_tags(tmp_path: Path) -> None:
    """Body + title concat with correct ``source_file`` tagging."""
    body_rows = [
        _row("a", "b", "b0", "2014-06-01 10:00:00", 1),
        _row("c", "d", "b1", "2014-06-02 10:00:00", -1),
    ]
    title_rows = [
        _row("e", "f", "t0", "2014-06-03 10:00:00", 1),
        _row("g", "h", "t1", "2014-06-04 10:00:00", -1),
    ]
    _write_tsv(tmp_path / BODY_FILENAME, body_rows)
    _write_tsv(tmp_path / TITLE_FILENAME, title_rows)

    df = load_reddit_dataset(tmp_path, network_type="both")

    assert len(df) == 4
    assert set(df["source_file"].unique()) == {"body", "title"}
    assert (df.loc[df["source_file"] == "body", "POST_ID"].tolist()) == ["b0", "b1"]
    assert (df.loc[df["source_file"] == "title", "POST_ID"].tolist()) == ["t0", "t1"]


def test_load_reddit_dataset_validates_network_type(tmp_path: Path) -> None:
    """An unknown ``network_type`` must raise ``ValueError`` before any file I/O."""
    with pytest.raises(ValueError):
        load_reddit_dataset(tmp_path, network_type="bogus")


def test_load_reddit_dataset_missing_file_raises_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        load_reddit_dataset(tmp_path, network_type="body")
    assert "prepare_data" in str(excinfo.value)


def test_parse_subreddit_embeddings_roundtrip(tmp_path: Path) -> None:
    """Embeddings parser returns the right shape and lowercased keys."""
    n = 3
    dim = 300
    rng = np.random.default_rng(0)
    values = rng.standard_normal((n, dim)).astype(np.float32)
    names = ["AskReddit", "News", "Gaming"]
    csv_path = tmp_path / "embeddings.csv"
    lines = []
    for name, row in zip(names, values, strict=True):
        lines.append(",".join([name, *(f"{v:.6f}" for v in row)]))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    name_to_idx, mat = parse_subreddit_embeddings(csv_path)

    assert mat.shape == (n, dim)
    assert set(name_to_idx.keys()) == {n.lower() for n in names}
    # The 'askreddit' row should approximately match the first random vector.
    np.testing.assert_allclose(mat[name_to_idx["askreddit"]], values[0], atol=1e-5)
