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
from reddit_gnn.data.preprocess import (
    build_node_mapping,
    clean_edges,
    preprocess_dataset,
    remap_post_label,
    save_processed_dataset,
)

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


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


def _preprocess_input_frame(tmp_path: Path) -> pd.DataFrame:
    """Build a synthetic raw frame (post-load) that exercises clean_edges.

    Includes:
        * Self-loop row (must be dropped).
        * Two duplicate (src, dst, ts, post_id) rows (one must be dropped).
        * Body and title rows (for ``is_title`` tagging).
        * Out-of-order timestamps (must end up sorted).
    """
    body_rows = [
        _row("askreddit", "askreddit", "self0", "2014-03-01 00:00:00", 1),  # self-loop
        _row("askreddit", "news", "p1", "2014-02-01 00:00:00", 1),
        _row("AskReddit", "news", "p1", "2014-02-01 00:00:00", 1),  # dup after normalize
        _row("gaming", "movies", "p2", "2014-04-01 00:00:00", -1),
    ]
    title_rows = [
        _row("music", "books", "t0", "2014-01-15 00:00:00", -1),
        _row("science", "space", "t1", "2014-05-10 00:00:00", 1),
    ]
    _write_tsv(tmp_path / BODY_FILENAME, body_rows)
    _write_tsv(tmp_path / TITLE_FILENAME, title_rows)
    return load_reddit_dataset(tmp_path, network_type="both")


def test_label_remap(tmp_path: Path) -> None:
    """clean_edges remaps -1 -> 0, +1 -> 1, and leaves only {0, 1} in label_binary."""
    raw = _preprocess_input_frame(tmp_path)
    cleaned = clean_edges(raw)

    # Column shape: label_binary present, POST_LABEL dropped.
    assert "label_binary" in cleaned.columns
    assert "POST_LABEL" not in cleaned.columns
    assert cleaned["label_binary"].dtype == np.int8

    labels = set(cleaned["label_binary"].unique().tolist())
    assert labels <= {0, 1}

    # Spot-check the row-by-row remap: gaming->movies had POST_LABEL=-1 -> 0.
    gaming_row = cleaned[
        (cleaned["source_subreddit_norm"] == "gaming")
        & (cleaned["target_subreddit_norm"] == "movies")
    ]
    assert len(gaming_row) == 1
    assert int(gaming_row["label_binary"].iloc[0]) == 0

    # askreddit->news had POST_LABEL=+1 -> 1.
    ar_row = cleaned[
        (cleaned["source_subreddit_norm"] == "askreddit")
        & (cleaned["target_subreddit_norm"] == "news")
    ]
    assert len(ar_row) == 1
    assert int(ar_row["label_binary"].iloc[0]) == 1


def test_no_self_loops(tmp_path: Path) -> None:
    """clean_edges drops every row where source equals target."""
    raw = _preprocess_input_frame(tmp_path)
    cleaned = clean_edges(raw)
    assert (cleaned["source_subreddit_norm"] != cleaned["target_subreddit_norm"]).all()
    # Specifically, the askreddit -> askreddit synthetic self-loop is gone.
    assert (
        (cleaned["source_subreddit_norm"] == "askreddit")
        & (cleaned["target_subreddit_norm"] == "askreddit")
    ).sum() == 0


def test_chronological_order(tmp_path: Path) -> None:
    """TIMESTAMP is monotonically non-decreasing after clean_edges."""
    raw = _preprocess_input_frame(tmp_path)
    cleaned = clean_edges(raw)
    ts = cleaned["TIMESTAMP"].to_numpy()
    assert (np.diff(ts).astype("timedelta64[ns]").astype("int64") >= 0).all()
    # And the first row really is the earliest one in the input (music->books).
    assert cleaned.iloc[0]["source_subreddit_norm"] == "music"


def test_node_mapping_bijection(tmp_path: Path) -> None:
    """build_node_mapping produces a contiguous bijection over the union of names."""
    raw = _preprocess_input_frame(tmp_path)
    cleaned = clean_edges(raw)
    mapped, node_to_id, id_to_node = build_node_mapping(cleaned)

    # Bijection: same cardinality, inverse maps round-trip.
    assert len(node_to_id) == len(id_to_node)
    for name, idx in node_to_id.items():
        assert id_to_node[idx] == name
    for idx, name in id_to_node.items():
        assert node_to_id[name] == idx

    # Contiguous from 0.
    ids = sorted(node_to_id.values())
    assert ids == list(range(len(ids)))

    # Every subreddit appearing as src or dst is in the mapping.
    expected_names = set(cleaned["source_subreddit_norm"]) | set(cleaned["target_subreddit_norm"])
    assert set(node_to_id.keys()) == expected_names

    # source_id / target_id are int64 and consistent with the mapping.
    assert "source_id" in mapped.columns
    assert "target_id" in mapped.columns
    assert mapped["source_id"].dtype == np.int64
    assert mapped["target_id"].dtype == np.int64
    for _, row in mapped.iterrows():
        assert node_to_id[row["source_subreddit_norm"]] == row["source_id"]
        assert node_to_id[row["target_subreddit_norm"]] == row["target_id"]


def test_preprocess_dataset_writes_artifacts(tmp_path: Path) -> None:
    """End-to-end: preprocess_dataset writes parquet + JSON sidecars and is reloadable."""
    raw = _preprocess_input_frame(tmp_path)
    out_dir = tmp_path / "processed"
    edges_path = preprocess_dataset(
        raw_dir=tmp_path,
        processed_dir=out_dir,
        network_type="both",
        df=raw,
    )

    assert edges_path == out_dir / "edges.parquet"
    assert edges_path.exists()
    for sidecar in ("node_to_id.json", "id_to_node.json", "metadata.json"):
        assert (out_dir / sidecar).exists()

    reloaded = pd.read_parquet(edges_path)
    assert "label_binary" in reloaded.columns
    assert "source_id" in reloaded.columns
    assert "target_id" in reloaded.columns
    assert set(reloaded["label_binary"].unique().tolist()) <= {0, 1}


def test_save_processed_dataset_metadata_is_consistent(tmp_path: Path) -> None:
    """metadata.json reports counts that match the saved parquet."""
    import json

    raw = _preprocess_input_frame(tmp_path)
    cleaned = clean_edges(raw)
    mapped, n2i, i2n = build_node_mapping(cleaned)

    out_dir = tmp_path / "processed"
    save_processed_dataset(mapped, n2i, i2n, out_dir, network_type="both")

    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["num_nodes"] == len(n2i)
    assert metadata["num_edges"] == len(mapped)
    assert metadata["network_type"] == "both"
    assert set(metadata["label_distribution"].keys()) <= {"0", "1"}


# ---------------------------------------------------------------------------
# Analysis smoke test
# ---------------------------------------------------------------------------


def test_compute_basic_stats_smoke() -> None:
    """compute_basic_stats returns the expected counts on a hand-built tiny df."""
    from reddit_gnn.analysis.graph_stats import compute_basic_stats

    # 4 nodes (0..3), 5 edges, one of which is a duplicate of another (in terms
    # of (source_id, target_id)); no self-loops.
    df = pd.DataFrame(
        {
            "source_id": [0, 1, 2, 0, 3],
            "target_id": [1, 2, 3, 1, 0],  # row 3 duplicates (0->1) from row 0
            "source_subreddit_norm": ["a", "b", "c", "a", "d"],
            "target_subreddit_norm": ["b", "c", "d", "b", "a"],
            "label_binary": np.array([1, 0, 1, 1, 0], dtype=np.int8),
            "TIMESTAMP": pd.to_datetime(
                [
                    "2014-01-01",
                    "2014-02-01",
                    "2014-03-01",
                    "2014-04-01",
                    "2014-05-01",
                ]
            ),
        }
    )
    stats = compute_basic_stats(df)
    assert stats["num_nodes"] == 4
    assert stats["num_edges"] == 5
    assert stats["self_loop_count"] == 0
    assert stats["duplicate_edge_count"] == 1
    # Average degree = total edges / num nodes = 5/4 = 1.25 for both directions.
    assert stats["avg_in_degree"] == pytest.approx(1.25)
    assert stats["avg_out_degree"] == pytest.approx(1.25)
    # Node 1 receives edges from 0 (twice) and 2 -> max in-degree 2.
    assert stats["max_in_degree"] == 2
