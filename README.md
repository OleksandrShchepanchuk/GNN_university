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

## Experiment tracking with MLflow

The project ships with a thin tracking layer at `src/reddit_gnn/tracking/` that
wraps MLflow. All training / sweep code routes through this module instead of
calling `mlflow.*` directly, so MLflow can be disabled (or swapped for another
backend) without touching training code.

**What is tracked:**
- **Params** — the merged YAML config (model_type, hidden_channels, lr, weight_decay, …).
- **Metrics per epoch** — train/val loss, F1-macro, PR-AUC, balanced accuracy, etc.
- **System metrics** — CPU / GPU / RAM utilisation, auto-logged.
- **Artifacts** — best checkpoint, training-curve PNG, predictions CSV (`reports/predictions/*`), confusion matrix, `metrics.json`.
- **Tags** — `model_type`, `seed`, `git_sha` when available, and `exception_type` if the run failed.

**Layout:** one experiment `reddit_signed` with one parent run per script
invocation. Sweeps create a parent run `sweep_{arch}` plus one nested child
run per Optuna trial.

**View the runs:**
```bash
make mlflow-ui
# -> open http://127.0.0.1:5000
```

**Disable tracking for one run:**
```bash
python scripts/run_experiment.py --config configs/gcn.yaml --no-tracking
```
or set `tracking.enabled: false` in the YAML config. With tracking disabled
every helper in `reddit_gnn.tracking` is a no-op — no `mlruns/` directory is
created.

**Wipe the local store:**
```bash
make mlflow-clean   # rm -rf ./mlruns; preserved by `make clean` on purpose
```

## Status

Scaffold + dataset pipeline + tracking are in place. Model code, training
loops, and the sweep entry points still raise `NotImplementedError` and will
land in subsequent commits.
