# Reddit Hyperlink GNN — Edge Sign Classification

**Author:** Олександр Щепанчук
**Group:** ПМПм-12
**Dataset:** [SNAP Reddit Hyperlink Network](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)

## Task

**Edge sign classification on observed subreddit-to-subreddit hyperlinks.**

This is *not* ordinary link prediction. We never sample non-edges and never treat
`label == 0` as "no edge". Every example is a real hyperlink observed in the
SNAP body / title CSVs; the prediction target is the *sentiment label*
(`POST_LABEL`) attached to that hyperlink, remapped as:

| Raw `POST_LABEL` | Trained label | Meaning              |
|------------------|---------------|----------------------|
| `-1`             | `0`           | negative             |
| `+1`             | `1`           | neutral / positive   |

## Layout

```
configs/        YAML configs (base + per-model + sweep)
data/           raw/interim/processed (gitignored except README + .gitkeep)
notebooks/      EDA, main experiment, error analysis
src/reddit_gnn/ library code (data, analysis, viz, models, training, utils)
scripts/        prepare_data, run_experiment, run_sweep, export_report_assets
tests/          pytest suites
models/         checkpoints + predictions (gitignored)
reports/        figures + tables (gitignored)
```

## Setup

Python 3.12 is the target (`.python-version`); 3.11 also works.

```bash
# Preferred (uv)
make install         # -> `uv sync --all-extras`

# Fallback
pip install -e ".[dev]"
```

Then download the SNAP files (see [data/raw/README.md](data/raw/README.md)) and run:

```bash
make data            # download + preprocess
make train CONFIG=configs/gcn.yaml
make sweep
make test
make lint
```

## Reference

Style reference only (no code copied): https://github.com/SerbulovArtem/Graph_Machine_Learning_IND_1

## Status

Scaffold only. All domain logic raises `NotImplementedError` and is intentionally
deferred — this commit establishes the project skeleton, configs, and tooling.
