"""Early stopping and global best-checkpoint state for EMA validation metrics.

Pure module: no file I/O, no distributed calls, no torch import. The solver
layer (``DetSolver``) owns checkpoint writes and DDP coordination.

Two best values are tracked:
- ``best_observed_metric``: any strict improvement updates this and triggers a
  ``best.pth`` save (small improvements still become the delivered model).
- ``best_significant_metric``: only improvements beyond ``min_delta`` reset the
  patience counter (validation noise cannot extend training indefinitely).
"""
from __future__ import annotations

from dataclasses import dataclass

# Absolute tolerance for verifying the restored model's mAP50_95 against the
# recorded best during finalization.
RESTORED_METRIC_TOLERANCE = 1e-3


@dataclass
class EarlyStoppingConfig:
    """Parsed and validated ``early_stopping`` configuration."""

    enabled: bool = False
    metric: str = "mAP50_95"
    mode: str = "max"
    min_epochs: int = 100
    patience: int = 12
    min_delta: float = 0.001
    restore_best: bool = True

    @classmethod
    def from_yaml(cls, yaml_cfg: dict) -> "EarlyStoppingConfig":
        """Parse the ``early_stopping`` mapping from a resolved yaml_cfg.

        A missing or empty mapping yields ``enabled=False`` (current training
        behavior preserved). A present non-empty mapping is validated.
        """
        es = yaml_cfg.get("early_stopping") or {}
        if not es:
            return cls(enabled=False)
        cfg = cls(
            enabled=bool(es.get("enabled", True)),
            metric=str(es.get("metric", "mAP50_95")),
            mode=str(es.get("mode", "max")),
            min_epochs=int(es.get("min_epochs", 100)),
            patience=int(es.get("patience", 12)),
            min_delta=float(es.get("min_delta", 0.001)),
            restore_best=bool(es.get("restore_best", True)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode != "max":
            raise ValueError(f"early_stopping.mode must be 'max', got {self.mode!r}")
        if self.min_epochs < 0:
            raise ValueError(
                f"early_stopping.min_epochs must be >= 0, got {self.min_epochs}"
            )
        if self.patience < 1:
            raise ValueError(
                f"early_stopping.patience must be >= 1, got {self.patience}"
            )
        if self.min_delta < 0:
            raise ValueError(
                f"early_stopping.min_delta must be >= 0, got {self.min_delta}"
            )


@dataclass
class EarlyStoppingState:
    """Pure early-stopping state machine.

    ``update`` feeds one validated metric and returns whether a ``best.pth``
    save is warranted. ``should_stop`` applies the ``min_epochs`` floor and
    patience. Only the four fields below are persisted.
    """

    best_observed_metric: float = float("-inf")
    best_significant_metric: float = float("-inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0

    def update(self, current_metric: float, epoch: int, min_delta: float) -> bool:
        """Record one validated metric.

        Returns True when ``best.pth`` should be saved (strict observed
        improvement over the current global best).
        """
        improved = current_metric > self.best_observed_metric
        if improved:
            self.best_observed_metric = current_metric
            self.best_epoch = epoch
        if current_metric > self.best_significant_metric + min_delta:
            self.best_significant_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved

    def should_stop(self, epoch: int, min_epochs: int, patience: int) -> bool:
        return epoch >= min_epochs and self.epochs_without_improvement >= patience

    def reset_patience(self) -> None:
        self.epochs_without_improvement = 0

    def initialize_from_metric(self, metric: float, epoch: int) -> None:
        """Initialize state from a fresh validation (resume from old checkpoint)."""
        self.best_observed_metric = metric
        self.best_significant_metric = metric
        self.best_epoch = epoch
        self.epochs_without_improvement = 0

    def state_dict(self) -> dict:
        return {
            "best_observed_metric": self.best_observed_metric,
            "best_significant_metric": self.best_significant_metric,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_observed_metric = float(state["best_observed_metric"])
        self.best_significant_metric = float(state["best_significant_metric"])
        self.best_epoch = int(state["best_epoch"])
        self.epochs_without_improvement = int(state["epochs_without_improvement"])
