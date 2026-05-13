SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3
UV := $(shell command -v uv 2>/dev/null)

.PHONY: help install data train sweep evaluate notebook test lint clean all

help:
	@echo "Available targets:"
	@echo "  install   - Sync env (uv sync if available, else pip install -e .[dev])"
	@echo "  data      - Run scripts/prepare_data.py to fetch + preprocess SNAP files"
	@echo "  train     - Run scripts/run_experiment.py with a config (CONFIG=configs/gcn.yaml)"
	@echo "  sweep     - Run scripts/run_sweep.py (CONFIG=configs/sweep.yaml)"
	@echo "  evaluate  - Re-evaluate a trained checkpoint (RUN=run_name)"
	@echo "  notebook  - Launch JupyterLab in ./notebooks"
	@echo "  test      - Run pytest"
	@echo "  lint      - Run ruff check + ruff format --check"
	@echo "  clean     - Remove caches, build artifacts, and processed data"
	@echo "  all       - install + data + test"

install:
	@echo ">>> Installing project (dev extras)"
ifeq ($(UV),)
	@echo "    uv not found; falling back to pip install -e .[dev]"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
else
	@echo "    Using uv: $(UV)"
	$(UV) sync --all-extras
endif

data:
	@echo ">>> Preparing dataset (download + preprocess SNAP Reddit Hyperlinks)"
	$(PYTHON) scripts/prepare_data.py

CONFIG ?= configs/gcn.yaml
train:
	@echo ">>> Training with config: $(CONFIG)"
	$(PYTHON) scripts/run_experiment.py --config $(CONFIG)

SWEEP_CONFIG ?= configs/sweep.yaml
sweep:
	@echo ">>> Running hyperparameter sweep: $(SWEEP_CONFIG)"
	$(PYTHON) scripts/run_sweep.py --config $(SWEEP_CONFIG)

RUN ?= latest
evaluate:
	@echo ">>> Evaluating run: $(RUN)"
	$(PYTHON) scripts/export_report_assets.py --run $(RUN)

notebook:
	@echo ">>> Launching JupyterLab in notebooks/"
	$(PYTHON) -m jupyterlab --notebook-dir=notebooks

test:
	@echo ">>> Running pytest"
	$(PYTHON) -m pytest

lint:
	@echo ">>> Running ruff check + format --check"
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

clean:
	@echo ">>> Cleaning caches and build artifacts"
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf data/interim/* data/processed/*
	@touch data/interim/.gitkeep data/processed/.gitkeep

all: install data test
	@echo ">>> all done"
