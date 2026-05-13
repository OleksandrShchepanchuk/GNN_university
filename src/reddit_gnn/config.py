"""Configuration dataclasses.

Two dataclasses live here:

* :class:`Paths` — filesystem locations derived from the project root.
  Importantly, *no* absolute paths are hardcoded; the project root is computed
  from this file's location.
* :class:`TrainConfig` — training hyperparameters used by the script entry
  points. YAML configs in ``configs/`` are loaded and merged onto an instance
  of this dataclass by the training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root = two parents above this file:
#   src/reddit_gnn/config.py -> src/reddit_gnn -> src -> <project_root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Canonical filesystem layout for the project.

    All fields default to subdirectories of :data:`PROJECT_ROOT`.
    """

    project_root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_interim: Path = PROJECT_ROOT / "data" / "interim"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    reports_figures: Path = PROJECT_ROOT / "reports" / "figures"
    reports_tables: Path = PROJECT_ROOT / "reports" / "tables"
    checkpoints: Path = PROJECT_ROOT / "models" / "checkpoints"
    predictions: Path = PROJECT_ROOT / "models" / "predictions"

    def ensure(self) -> Paths:
        """Create every managed directory if it doesn't exist."""
        for p in (
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.reports_figures,
            self.reports_tables,
            self.checkpoints,
            self.predictions,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class TrainConfig:
    """Training hyperparameters consumed by the experiment scripts."""

    model_type: str = "gcn"
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    lr: float = 5e-3
    weight_decay: float = 5e-4
    epochs: int = 100
    batch_size: int = 4096
    early_stopping_patience: int = 15
    seed: int = 42
    device: str = "cpu"
    run_name: str = "run"
    extra: dict = field(default_factory=dict)


__all__ = ["PROJECT_ROOT", "Paths", "TrainConfig"]
