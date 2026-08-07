# EMA Early Stopping and Best Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add EMA `mAP50_95` early stopping with global `best.pth` recovery to the DEIMv2 OBB training loop, so the delivered model matches the training peak instead of the degraded final epoch.

**Architecture:** A pure, persistence-ready state machine (`engine/solver/early_stopping.py`) holds config + dual-metric patience state with no I/O or DDP. `DetSolver` owns the wiring: rank 0 updates state after each validation epoch, saves `best.pth` on any observed improvement, broadcasts state + stop decision via `broadcast_object_list`, and on normal exit restores `best.pth`, re-validates, and writes `final_run_meta.json`. `last.pth` is moved to write *after* validation so interruption-resume preserves the exact patience state.

**Tech Stack:** Python 3.10+, PyTorch 2.5.1, pytest 9.x, PyYAML. No new dependencies.

## Global Constraints

- **No git commits.** User constraint from the design session. Tasks end with pytest verification steps, never `git commit`.
- **Preserve existing uncommitted user changes.** Never `git add`/`git reset`/revert files outside this plan's file list.
- **Run all pytest commands from the repo root** `/mnt/d/cx/thired/deimv2_daod`. Existing tests use CWD-relative config paths and `python -m pytest` (which adds CWD to `sys.path`). Do not run from inside `test/`.
- `epoches: 150` remains the only hard training limit. Do not introduce `max_epochs`/`schedule_epochs`.
- ES-Base keeps the existing schedule verbatim: `warmup_iter: 15`, `flat_epoch: 75`, `no_aug_epoch: 15`, and the current augmentation policy (`policy.epoch`, `mixup_epochs`, `stop_epoch`).
- Approved early-stopping config (verbatim): `enabled: true`, `metric: mAP50_95`, `mode: max`, `min_epochs: 100`, `patience: 12`, `min_delta: 0.001`, `restore_best: true`.
- Do **not** modify: loss formulas, optimizer params, LR boundaries, augmentation policy, stage rollback logic, `best_stg1.pth`/`best_stg2.pth` semantics, precision settings (`amp_dtype`, `obb_geometry_fp32`, scaler).
- Do **not** add `early_stopping` to the abandoned C-group config (`sp_fz_rep0_nloss_fp16_obb_fp32.yml`) or to the legacy base configs. Only the two new `*_es.yml` configs carry it.
- `RESTORED_METRIC_TOLERANCE = 1e-3` (absolute) is the module-level verification tolerance. It is a code constant, not a config field.
- Pure module (`early_stopping.py`) must not import `torch`, do file I/O, or call DDP. All tests are CPU-only; no real multi-GPU required.
- Follow existing style: `from __future__ import annotations`, dataclasses, `dist_utils.save_on_master` for writes, `sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))` in test files.

---

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `engine/solver/early_stopping.py` | `EarlyStoppingConfig` (parse+validate), `EarlyStoppingState` (pure state machine + persistence), `RESTORED_METRIC_TOLERANCE`. No I/O, no DDP, no torch. | Create |
| `engine/solver/det_solver.py` | Wire ES into `fit()`: init before resume load, pre-resume init hook, stage-transition patience reset, per-epoch update + `best.pth`, moved `last.pth` ordering, DDP broadcast, early break, exit capture, `_finalize_training` restore + `final_run_meta.json`. | Modify |
| `test/test_early_stopping.py` | Pure tests for `EarlyStoppingConfig`, `EarlyStoppingState`, persistence. | Create |
| `test/test_det_solver_early_stopping.py` | Integration tests via mocked `fit()`: disabled behavior, best.pth, patience/min-epochs, checkpoint ordering, DDP broadcast, finalize/restore, resume, stage transition, diagnostic exit. | Create |
| `test/test_early_stopping_configs.py` | Config inheritance: ES blocks present only in the two new configs; schedule/precision preserved; C-group and legacy bases unpolluted. | Create |
| `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_amp_es.yml` | ES-Base FP16: includes `amp` base, own `output_dir`, adds approved `early_stopping` block. | Create |
| `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16_es.yml` | ES-Base BF16: includes `bf16` base, own `output_dir`, adds approved `early_stopping` block. | Create |

---

### Task 1: EarlyStoppingConfig (parse + validate)

**Files:**
- Create: `engine/solver/early_stopping.py`
- Test: `test/test_early_stopping.py`

**Interfaces:**
- Produces: `EarlyStoppingConfig` dataclass with `enabled: bool`, `metric: str`, `mode: str`, `min_epochs: int`, `patience: int`, `min_delta: float`, `restore_best: bool`; classmethod `from_yaml(yaml_cfg: dict) -> EarlyStoppingConfig`; `validate() -> None`. Missing/empty `early_stopping` key → `EarlyStoppingConfig(enabled=False)`.

- [ ] **Step 1: Write the failing test**

Create `test/test_early_stopping.py`:

```python
"""Pure early-stopping config + state-machine tests.

TDD: RED -> GREEN. Fails with ImportError until engine/solver/early_stopping.py
exists. CPU-only; no DDP, no file I/O inside the module under test.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from engine.solver.early_stopping import EarlyStoppingConfig

VALID_BLOCK = {
    "enabled": True,
    "metric": "mAP50_95",
    "mode": "max",
    "min_epochs": 100,
    "patience": 12,
    "min_delta": 0.001,
    "restore_best": True,
}


class TestEarlyStoppingConfig:

    def test_missing_config_disables(self):
        cfg = EarlyStoppingConfig.from_yaml({})
        assert cfg.enabled is False

    def test_empty_block_disables(self):
        cfg = EarlyStoppingConfig.from_yaml({"early_stopping": {}})
        assert cfg.enabled is False

    def test_approved_block_parses(self):
        cfg = EarlyStoppingConfig.from_yaml({"early_stopping": VALID_BLOCK})
        assert cfg.enabled is True
        assert cfg.metric == "mAP50_95"
        assert cfg.mode == "max"
        assert cfg.min_epochs == 100
        assert cfg.patience == 12
        assert cfg.min_delta == 0.001
        assert cfg.restore_best is True

    def test_partial_block_uses_defaults(self):
        cfg = EarlyStoppingConfig.from_yaml({"early_stopping": {"enabled": True}})
        assert cfg.enabled is True
        assert cfg.metric == "mAP50_95"
        assert cfg.mode == "max"
        assert cfg.min_epochs == 100
        assert cfg.patience == 12
        assert cfg.min_delta == 0.001
        assert cfg.restore_best is True

    def test_explicit_enabled_false(self):
        cfg = EarlyStoppingConfig.from_yaml(
            {"early_stopping": {"enabled": False, "patience": 7}}
        )
        assert cfg.enabled is False
        assert cfg.patience == 7

    @pytest.mark.parametrize(
        "block",
        [
            {"mode": "min"},
            {"patience": 0},
            {"patience": -3},
            {"min_epochs": -1},
            {"min_delta": -0.5},
        ],
    )
    def test_invalid_block_raises(self, block):
        with pytest.raises(ValueError):
            EarlyStoppingConfig.from_yaml({"early_stopping": block})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_early_stopping.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'engine.solver.early_stopping'`.

- [ ] **Step 3: Write minimal implementation**

Create `engine/solver/early_stopping.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_early_stopping.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: No commit (user constraint)** — verify `git status` shows only intended new files; proceed.

---

### Task 2: EarlyStoppingState (pure state machine + persistence)

**Files:**
- Modify: `engine/solver/early_stopping.py` (append)
- Test: `test/test_early_stopping.py` (append)

**Interfaces:**
- Consumes: `EarlyStoppingConfig` from Task 1.
- Produces: `EarlyStoppingState` dataclass with fields `best_observed_metric: float`, `best_significant_metric: float`, `best_epoch: int`, `epochs_without_improvement: int`; methods `update(current_metric, epoch, min_delta) -> bool`, `should_stop(epoch, min_epochs, patience) -> bool`, `reset_patience()`, `initialize_from_metric(metric, epoch)`, `state_dict() -> dict`, `load_state_dict(state)`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_early_stopping.py`:

```python
from engine.solver.early_stopping import EarlyStoppingState


class TestEarlyStoppingState:

    def test_first_metric_is_observed_best_and_resets_patience(self):
        st = EarlyStoppingState()
        assert st.update(0.30, epoch=10, min_delta=0.001) is True
        assert st.best_observed_metric == 0.30
        assert st.best_epoch == 10
        assert st.best_significant_metric == 0.30
        assert st.epochs_without_improvement == 0

    def test_strict_improvement_saves_observed_best(self):
        st = EarlyStoppingState()
        st.update(0.30, epoch=10, min_delta=0.001)
        assert st.update(0.32, epoch=11, min_delta=0.001) is True
        assert st.best_observed_metric == 0.32
        assert st.best_epoch == 11

    def test_small_improvement_saves_but_does_not_reset_patience(self):
        st = EarlyStoppingState()
        st.update(0.50, epoch=1, min_delta=0.001)
        st.epochs_without_improvement = 3
        st.best_significant_metric = 0.50
        # +0.0005 < min_delta: observed best updates, patience keeps counting
        assert st.update(0.5005, epoch=2, min_delta=0.001) is True
        assert st.best_observed_metric == 0.5005
        assert st.epochs_without_improvement == 4

    def test_significant_improvement_resets_patience(self):
        st = EarlyStoppingState()
        st.update(0.50, epoch=1, min_delta=0.001)
        st.epochs_without_improvement = 5
        # +0.005 >= min_delta: resets patience
        assert st.update(0.505, epoch=2, min_delta=0.001) is False
        assert st.best_significant_metric == 0.505
        assert st.epochs_without_improvement == 0

    def test_no_improvement_increments_patience(self):
        st = EarlyStoppingState()
        st.update(0.50, epoch=1, min_delta=0.001)
        assert st.update(0.49, epoch=2, min_delta=0.001) is False
        assert st.epochs_without_improvement == 1

    def test_should_stop_respects_min_epochs_floor(self):
        st = EarlyStoppingState()
        st.epochs_without_improvement = 12
        assert st.should_stop(epoch=99, min_epochs=100, patience=12) is False
        assert st.should_stop(epoch=100, min_epochs=100, patience=12) is True

    def test_should_stop_requires_patience_exhausted(self):
        st = EarlyStoppingState()
        st.epochs_without_improvement = 11
        assert st.should_stop(epoch=150, min_epochs=100, patience=12) is False

    def test_reset_patience_preserves_best_values(self):
        st = EarlyStoppingState()
        st.update(0.50, epoch=5, min_delta=0.001)
        st.epochs_without_improvement = 9
        st.reset_patience()
        assert st.epochs_without_improvement == 0
        assert st.best_observed_metric == 0.50
        assert st.best_epoch == 5

    def test_initialize_from_metric_sets_all_fields(self):
        st = EarlyStoppingState()
        st.initialize_from_metric(0.31, epoch=6)
        assert st.best_observed_metric == 0.31
        assert st.best_significant_metric == 0.31
        assert st.best_epoch == 6
        assert st.epochs_without_improvement == 0


class TestEarlyStoppingStatePersistence:

    def test_state_dict_round_trip(self):
        st = EarlyStoppingState()
        st.update(0.42, epoch=17, min_delta=0.001)
        st.epochs_without_improvement = 4
        d = st.state_dict()
        st2 = EarlyStoppingState()
        st2.load_state_dict(d)
        assert st2.best_observed_metric == 0.42
        assert st2.best_significant_metric == 0.42
        assert st2.best_epoch == 17
        assert st2.epochs_without_improvement == 4

    def test_state_dict_keys(self):
        st = EarlyStoppingState()
        d = st.state_dict()
        assert set(d) == {
            "best_observed_metric",
            "best_significant_metric",
            "best_epoch",
            "epochs_without_improvement",
        }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_early_stopping.py -v`
Expected: FAIL — `ImportError: cannot import name 'EarlyStoppingState'`.

- [ ] **Step 3: Write minimal implementation**

Append to `engine/solver/early_stopping.py` (after `EarlyStoppingConfig`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_early_stopping.py -v`
Expected: PASS (21 tests).

- [ ] **Step 5: No commit (user constraint)** — proceed.

---

### Task 3: Solver init wiring + disabled-behavior preservation + test harness

**Files:**
- Modify: `engine/solver/det_solver.py`
- Test: `test/test_det_solver_early_stopping.py` (create)

**Interfaces:**
- Consumes: `EarlyStoppingConfig`, `EarlyStoppingState` (Tasks 1–2).
- Produces: `DetSolver._init_early_stopping() -> None` (sets `self.early_stopping_config`, `self.early_stopping` (or `None`), `self._diagnostic_exit`). Called as the first statement of `fit()` — *before* `self.train()` so resume loading sees the attribute.
- Test harness produces: `_make_solver(tmp_path, *, epoches=3, stop_epoch=999, es=None, ema=False)` and `_run_fit(solver, metrics, monkeypatch, *, step_cap=False, restore_metric=None)`. `es=None` means the config carries no `early_stopping` key (legacy).

- [ ] **Step 1: Write the failing test**

Create `test/test_det_solver_early_stopping.py`:

```python
"""DetSolver early-stopping integration tests (mocked training flow).

Harness: DetSolver is built via __new__ (skips heavy _setup), the real
train()/load_resume_state are shadowed where noted, and det_solver module
functions (stats, train_one_epoch, evaluate) are monkeypatched. Mirrors the
mocking style of test_det_engine_diagnostics.py. CPU-only.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from engine.solver import det_solver
from engine.solver.det_solver import DetSolver
from engine.solver.early_stopping import EarlyStoppingState


def _es_yaml(**overrides):
    es = {
        "enabled": True,
        "metric": "mAP50_95",
        "mode": "max",
        "min_epochs": 2,
        "patience": 2,
        "min_delta": 0.001,
        "restore_best": True,
    }
    es.update(overrides)
    return es


def _make_solver(tmp_path, *, epoches=3, stop_epoch=999, es=None, ema=False):
    """Build a DetSolver with a stub graph; train() is shadowed to a no-op.

    es=None -> yaml_cfg has no early_stopping key (legacy behavior).
    ema=True -> self.ema is a stub with .module/.decay (needed by the
    stage-transition test).
    """
    model = nn.Linear(2, 2)
    yaml_cfg = {
        "max_optimizer_steps": None,
        "fail_on_zero_grad": False,
        "nan_max_events": 10,
        "amp_dtype": None,
        "obb_geometry_fp32": False,
    }
    if es is not None:
        yaml_cfg["early_stopping"] = es

    cfg = SimpleNamespace(
        lrsheduler=None,
        epoches=epoches,
        lr_gamma=0.1,
        warmup_iter=0,
        flat_epoch=1,
        no_aug_epoch=0,
        clip_max_norm=0.0,
        print_freq=1000,
        checkpoint_freq=1000,
        yaml_cfg=yaml_cfg,
        device="cpu",
        resume=None,
        use_ema=False,
        use_amp=False,
    )

    solver = DetSolver.__new__(DetSolver)
    solver.cfg = cfg
    solver.model = model
    solver.ema = (
        SimpleNamespace(module=model, decay=0.9999) if ema else None
    )
    solver.criterion = None
    solver.postprocessor = SimpleNamespace(box_mode="obb")
    solver.scaler = None
    solver.device = torch.device("cpu")
    solver.last_epoch = -1
    solver.output_dir = tmp_path
    solver.writer = None
    solver.optimizer = None
    solver.lr_scheduler = SimpleNamespace(step=lambda: None)
    solver.lr_warmup_scheduler = SimpleNamespace(finished=lambda: True)
    solver.self_lr_scheduler = False
    solver.train_dataloader = SimpleNamespace(
        set_epoch=lambda e: None,
        collate_fn=SimpleNamespace(stop_epoch=stop_epoch, ema_restart_decay=0.9999),
        sampler=SimpleNamespace(set_epoch=lambda e: None),
    )
    solver.val_dataloader = None
    solver.evaluator = None
    solver.train = lambda: None
    solver._comet_experiment = None
    return solver


def _run_fit(solver, metrics, monkeypatch, *, step_cap=False, restore_metric=None):
    """Monkeypatch engine fns and run fit().

    metrics: one value per training epoch (popped left-to-right).
    restore_metric: value returned by the final restore re-evaluation; defaults
    to 0.0 when the metrics list is exhausted.
    """
    monkeypatch.setattr(det_solver, "stats", lambda cfg: (0, ""))
    monkeypatch.setattr(
        det_solver,
        "train_one_epoch",
        lambda *a, **k: {"_step_cap_reached": True} if step_cap else {},
    )

    def fake_evaluate(*args, **kwargs):
        if metrics:
            return ({"mAP50_95": metrics.pop(0)}, None)
        return ({"mAP50_95": restore_metric if restore_metric is not None else 0.0}, None)

    monkeypatch.setattr(det_solver, "evaluate", fake_evaluate)
    solver.fit()


class TestDisabledBehavior:

    def test_missing_config_preserves_legacy_behavior(self, tmp_path, monkeypatch):
        """Design test 1: no ES key -> no best.pth, no meta, last.pth still written."""
        solver = _make_solver(tmp_path, epoches=3, es=None)
        _run_fit(solver, [0.10, 0.20, 0.30], monkeypatch)
        assert not (tmp_path / "best.pth").exists()
        assert not (tmp_path / "final_run_meta.json").exists()
        assert (tmp_path / "last.pth").exists()

    def test_explicit_disabled_block_is_legacy(self, tmp_path, monkeypatch):
        solver = _make_solver(
            tmp_path, epoches=3, es={"enabled": False}
        )
        _run_fit(solver, [0.10, 0.20, 0.30], monkeypatch)
        assert not (tmp_path / "best.pth").exists()
        assert not (tmp_path / "final_run_meta.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_det_solver_early_stopping.py -v`
Expected: FAIL — `AttributeError: 'DetSolver' object has no attribute '_init_early_stopping'`.

- [ ] **Step 3: Write minimal implementation**

Edit `engine/solver/det_solver.py`:

**Edit 3.1 — imports** (after line 18 `from ..optim.lr_scheduler import FlatCosineLRScheduler`):

```python
from .early_stopping import (
    EarlyStoppingConfig,
    EarlyStoppingState,
    RESTORED_METRIC_TOLERANCE,
)
```

**Edit 3.2 — `fit()` entry** (current lines 24–27):

```python
    def fit(
        self,
    ):
        self.train()
        args = self.cfg
```

replace with:

```python
    def fit(
        self,
    ):
        self._init_early_stopping()
        self.train()
        args = self.cfg
```

**Edit 3.3 — new method** (insert between `fit()` and `val()`, i.e. after the `print("Training time {}".format(total_time_str))` block and before `def val(self):`):

```python
    def _init_early_stopping(self):
        """Build early-stopping config and state from cfg.yaml_cfg.

        Runs before self.train() so resume loading can restore the persisted
        early-stopping state. Missing/disabled config leaves
        self.early_stopping = None (current training behavior preserved).
        """
        self.early_stopping_config = EarlyStoppingConfig.from_yaml(self.cfg.yaml_cfg)
        self.early_stopping = (
            EarlyStoppingState() if self.early_stopping_config.enabled else None
        )
        self._diagnostic_exit = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_det_solver_early_stopping.py -v`
Expected: PASS (2 tests). (With ES disabled, `fit()` runs the full legacy loop under the mocked harness; `last.pth` is still written by the pre-existing checkpoint block.)

- [ ] **Step 5: No commit (user constraint)** — proceed.

---

### Task 4: Epoch-loop ES wiring (update, best.pth, ordering, DDP, early break)

**Files:**
- Modify: `engine/solver/det_solver.py`
- Test: `test/test_det_solver_early_stopping.py` (append)

**Interfaces:**
- Consumes: `EarlyStoppingState.update/should_stop`, `EarlyStoppingConfig` fields.
- Produces:
  - `DetSolver._sync_early_stopping(stop_early: bool) -> bool` — DDP broadcast of state dict + stop decision; no-op passthrough when not distributed.
  - `DetSolver._update_early_stopping(metric: float, epoch: int) -> tuple[bool, bool]` — rank-0 update + `best.pth` save on observed improvement; returns `(should_save_best, should_stop)`.
  - Loop behavior: `stop_early` init before the epoch loop; stage-transition `reset_patience`; pre-resume init hook; `_diagnostic_exit` flag on step-cap break; `last.pth`/periodic checkpoint written *after* validation + ES update; `break` after log entry when `stop_early`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_det_solver_early_stopping.py`:

```python
def _load_ckpt(tmp_path, name):
    return torch.load(tmp_path / name, map_location="cpu", weights_only=False)


class TestBestCheckpointSaving:

    def test_strict_improvement_saves_best_pth(self, tmp_path, monkeypatch):
        """Design test 2: strict improvement writes a complete best.pth."""
        solver = _make_solver(tmp_path, epoches=3, es=_es_yaml())
        _run_fit(solver, [0.30, 0.32, 0.31], monkeypatch)
        best = _load_ckpt(tmp_path, "best.pth")
        es_state = best["early_stopping"]
        assert es_state["best_epoch"] == 1
        assert es_state["best_observed_metric"] == pytest.approx(0.32)
        assert "model" in best
        assert "last_epoch" in best

    def test_small_improvement_saves_best_but_keeps_patience(self, tmp_path, monkeypatch):
        """Design test 3 (integration): sub-min_delta gain still updates best.pth."""
        solver = _make_solver(
            tmp_path, epoches=4, es=_es_yaml(min_epochs=0, patience=10)
        )
        # best 0.30 at ep0; +0.0005 at ep1 is < min_delta
        _run_fit(solver, [0.30, 0.3005, 0.29, 0.28], monkeypatch)
        best = _load_ckpt(tmp_path, "best.pth")
        assert best["early_stopping"]["best_epoch"] == 1
        assert best["early_stopping"]["best_observed_metric"] == pytest.approx(0.3005)
        # patience kept counting: ep1 no reset -> ep2, ep3 increments
        last = _load_ckpt(tmp_path, "last.pth")
        assert last["early_stopping"]["epochs_without_improvement"] == 3


class TestPatienceStopping:

    def test_patience_exhausted_stops_after_min_epochs(self, tmp_path, monkeypatch):
        """Design test 5 (integration): stop fires at min_epochs, not before."""
        solver = _make_solver(
            tmp_path, epoches=10, es=_es_yaml(min_epochs=2, patience=2)
        )
        _run_fit(
            solver, [0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21],
            monkeypatch,
        )
        last = _load_ckpt(tmp_path, "last.pth")
        # ep0 improves; eps 1,2 no improvement -> counter hits 2 at epoch 2
        assert last["last_epoch"] == 2
        assert last["early_stopping"]["epochs_without_improvement"] == 2

    def test_no_stop_before_min_epochs(self, tmp_path, monkeypatch):
        """min_epochs=5 blocks the stop even though patience reaches 2 early."""
        solver = _make_solver(
            tmp_path, epoches=10, es=_es_yaml(min_epochs=5, patience=2)
        )
        # ep0 improves; flat afterwards -> counter hits patience at ep2 but
        # 5-epoch floor holds until epoch 5
        _run_fit(solver, [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30], monkeypatch)
        last = _load_ckpt(tmp_path, "last.pth")
        assert last["last_epoch"] == 5
        assert last["early_stopping"]["epochs_without_improvement"] == 5


class TestCheckpointOrdering:

    def test_last_pth_is_true_final_epoch(self, tmp_path, monkeypatch):
        """Design test 12 (partial): last.pth keeps the real final epoch."""
        solver = _make_solver(tmp_path, epoches=3, es=_es_yaml())
        _run_fit(solver, [0.30, 0.32, 0.31], monkeypatch)
        last = _load_ckpt(tmp_path, "last.pth")
        assert last["last_epoch"] == 2
        assert last["early_stopping"]["best_epoch"] == 1  # patience state current
        best = _load_ckpt(tmp_path, "best.pth")
        assert best["last_epoch"] == 1


class TestDDPBroadcast:

    def _init_solver(self, tmp_path):
        solver = _make_solver(tmp_path, es=_es_yaml())
        solver._init_early_stopping()
        return solver

    def test_main_rank_broadcasts_state_and_stop(self, tmp_path, monkeypatch):
        """Design test 6 (rank 0 side): state dict + stop flag are broadcast."""
        solver = self._init_solver(tmp_path)
        solver.early_stopping.update(0.5, epoch=1, min_delta=0.001)
        captured = {}

        monkeypatch.setattr(
            det_solver.dist_utils, "is_dist_available_and_initialized", lambda: True
        )
        monkeypatch.setattr(det_solver.dist_utils, "is_main_process", lambda: True)
        monkeypatch.setattr(
            torch.distributed,
            "broadcast_object_list",
            lambda payload, src=0: captured.update(payload=payload, src=src),
        )

        stop = solver._sync_early_stopping(True)
        assert stop is True
        assert captured["src"] == 0
        assert captured["payload"][1] is True
        assert captured["payload"][0]["best_epoch"] == 1

    def test_non_main_rank_receives_state_and_stop(self, tmp_path, monkeypatch):
        """Design test 6 (other ranks): receive state and stop from rank 0."""
        solver = self._init_solver(tmp_path)
        state = {
            "best_observed_metric": 0.5,
            "best_significant_metric": 0.5,
            "best_epoch": 7,
            "epochs_without_improvement": 3,
        }

        monkeypatch.setattr(
            det_solver.dist_utils, "is_dist_available_and_initialized", lambda: True
        )
        monkeypatch.setattr(det_solver.dist_utils, "is_main_process", lambda: False)
        monkeypatch.setattr(
            torch.distributed,
            "broadcast_object_list",
            lambda payload, src=0: payload.__setitem__(0, state)
            or payload.__setitem__(1, True),
        )

        stop = solver._sync_early_stopping(False)
        assert stop is True
        assert solver.early_stopping.best_epoch == 7
        assert solver.early_stopping.epochs_without_improvement == 3

    def test_non_distributed_is_passthrough(self, tmp_path, monkeypatch):
        solver = self._init_solver(tmp_path)
        assert solver._sync_early_stopping(True) is True
        assert solver._sync_early_stopping(False) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_det_solver_early_stopping.py -v`
Expected: FAIL — best.pth/checkpoint tests fail with `FileNotFoundError` (no `best.pth`/`last.pth` written yet because the ES loop hook does not exist); `TestDDPBroadcast` fails with `AttributeError: ... no attribute '_sync_early_stopping'`.

- [ ] **Step 3: Write minimal implementation**

Edit `engine/solver/det_solver.py`:

**Edit 4.1 — `stop_early` init** (current line ~64, `top1 = 0`):

```python
        top1 = 0
        best_stat = {
            "epoch": -1,
        }
```

replace with:

```python
        top1 = 0
        stop_early = False
        best_stat = {
            "epoch": -1,
        }
```

**Edit 4.2 — pre-resume init hook** (current lines 92–96, OBB branch of the pre-resume validation):

```python
            else:
                v = test_stats.get("mAP50_95", 0)
                best_stat["epoch"] = self.last_epoch
                best_stat["mAP50_95"] = v
                top1 = v
                print(f"best_stat: {best_stat}")
```

replace with:

```python
            else:
                v = test_stats.get("mAP50_95", 0)
                best_stat["epoch"] = self.last_epoch
                best_stat["mAP50_95"] = v
                top1 = v
                print(f"best_stat: {best_stat}")
                if (
                    self.early_stopping is not None
                    and self.early_stopping.best_epoch < 0
                ):
                    self.early_stopping.initialize_from_metric(
                        v, self.last_epoch
                    )
                    print(
                        f"Initialized early-stopping state from pre-resume "
                        f"EMA validation: best_mAP50_95={v:.4f} "
                        f"at epoch {self.last_epoch}"
                    )
```

**Edit 4.3 — stage-transition patience reset** (current lines 140–145):

```python
            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                saved_epoch = self.last_epoch
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                self.last_epoch = saved_epoch
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")
```

replace with:

```python
            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                saved_epoch = self.last_epoch
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                self.last_epoch = saved_epoch
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")
                if self.early_stopping is not None:
                    self.early_stopping.reset_patience()
                    print(
                        f"Early-stopping patience reset at stage transition "
                        f"(epoch {epoch}); global best preserved "
                        f"(best_epoch={self.early_stopping.best_epoch}, "
                        f"best_mAP50_95={self.early_stopping.best_observed_metric:.4f})"
                    )
```

**Edit 4.4 — diagnostic flag** (current lines 175–177):

```python
            if train_stats.pop("_step_cap_reached", False):
                print(f"[Diagnostic] step cap reached at epoch {epoch}. Stopping.")
                break
```

replace with:

```python
            if train_stats.pop("_step_cap_reached", False):
                print(f"[Diagnostic] step cap reached at epoch {epoch}. Stopping.")
                self._diagnostic_exit = True
                break
```

**Edit 4.5 — remove the pre-validation checkpoint block** (current lines 188–196):

```python
            if self.output_dir :
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(
                        self.output_dir / f"checkpoint{epoch:04}.pth"
                    )
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)
```

Delete this block entirely (it is re-inserted after validation in Edit 4.7).

**Edit 4.6 — ES update in the OBB branch** (current lines 256–259, the end of the `else` OBB branch, after the `best_stg1.pth` save):

```python
                    else:
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )
```

replace with:

```python
                    else:
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )

                # Early-stopping update (OBB): monitors EMA mAP50_95, saves
                # best.pth on observed improvement, computes the local stop
                # decision. The DDP broadcast of the decision happens AFTER the
                # epoch's log entry and checkpoint writes (see Edit 4.7).
                _, stop_early = self._update_early_stopping(v, epoch)
```

**Edit 4.7 — re-insert checkpoint block after validation + early-break** (after the `coco_evaluator` eval-file save block, before the loop body ends at the current line 313):

```python
                for name in filenames:
                    torch.save(
                        coco_evaluator.coco_eval["bbox"].eval,
                        self.output_dir / "eval" / name,
                    )

            # Checkpoint after validation so last.pth carries the exact
            # early-stopping state (interruption-resume continuity).
            if self.output_dir:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(
                        self.output_dir / f"checkpoint{epoch:04}.pth"
                    )
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            # Design ordering: the loop exits only after the epoch's log entry
            # and checkpoint writes have completed; rank 0 then broadcasts the
            # stop decision so every rank breaks together.
            stop_early = self._sync_early_stopping(stop_early)

            if stop_early:
                print(
                    f"[EarlyStopping] stopping at epoch {epoch} after "
                    f"{self.early_stopping_config.patience} epochs without "
                    f"significant improvement."
                )
                break
```

**Edit 4.8 — new methods** (insert next to `_init_early_stopping` from Task 3):

```python
    def _sync_early_stopping(self, stop_early):
        """Broadcast early-stopping state and stop decision to all ranks (DDP).

        Rank 0 already updated the state. Non-main ranks overwrite their local
        state from the broadcast payload. Returns the agreed stop decision.
        Safe when early stopping is disabled (returns stop_early unchanged).
        """
        if self.early_stopping is None:
            return stop_early
        if not dist_utils.is_dist_available_and_initialized():
            return stop_early
        if dist_utils.is_main_process():
            payload = [self.early_stopping.state_dict(), stop_early]
        else:
            payload = [None, False]
        torch.distributed.broadcast_object_list(payload, src=0)
        if not dist_utils.is_main_process():
            self.early_stopping.load_state_dict(payload[0])
            stop_early = bool(payload[1])
        return stop_early

    def _update_early_stopping(self, metric, epoch):
        """Update ES state on rank 0, save best.pth on observed improvement.

        Returns (should_save_best, should_stop) with only the local stop
        decision; the caller broadcasts it via ``_sync_early_stopping`` AFTER
        the epoch's log entry and checkpoint writes (design ordering).
        """
        if self.early_stopping is None:
            return False, False
        should_save_best = False
        should_stop = False
        if dist_utils.is_main_process():
            should_save_best = self.early_stopping.update(
                metric, epoch, self.early_stopping_config.min_delta
            )
            if should_save_best and self.output_dir:
                dist_utils.save_on_master(
                    self.state_dict(), self.output_dir / "best.pth"
                )
            should_stop = self.early_stopping.should_stop(
                epoch,
                self.early_stopping_config.min_epochs,
                self.early_stopping_config.patience,
            )
        return should_save_best, should_stop
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_det_solver_early_stopping.py -v`
Expected: PASS — TestDisabledBehavior (2), TestBestCheckpointSaving (2), TestPatienceStopping (2), TestCheckpointOrdering (1), TestDDPBroadcast (3).

- [ ] **Step 5: No commit (user constraint)** — proceed.

---

### Task 5: Finalize — restore best.pth, re-validate, metadata, resume/stage/diagnostic paths

**Files:**
- Modify: `engine/solver/det_solver.py`
- Test: `test/test_det_solver_early_stopping.py` (append)

**Interfaces:**
- Consumes: `_update_early_stopping`, `_diagnostic_exit`, `RESTORED_METRIC_TOLERANCE`.
- Produces: `DetSolver._finalize_training(stop_reason: str, stop_epoch: int, box_mode: str) -> None` — captures metadata *before* restore, restores `best.pth` for non-diagnostic exits, re-validates EMA model, writes `final_run_meta.json` (keys: `stop_reason`, `stop_epoch`, `best_epoch`, `best_mAP50_95`, `restored_mAP50_95`, `epochs_after_best`, `restore_skipped`, `restore_match`). Early-returns when `self.early_stopping is None`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_det_solver_early_stopping.py`:

```python
def _read_meta(tmp_path):
    return json.loads((tmp_path / "final_run_meta.json").read_text())


class TestFinalizeRestore:

    def test_max_epochs_restores_and_validates_best_pth(self, tmp_path, monkeypatch):
        """Design tests 7 + 11: epoch-limit exit restores best.pth, restored
        eval matches the saved best within tolerance."""
        solver = _make_solver(tmp_path, epoches=3, es=_es_yaml())
        _run_fit(
            solver, [0.30, 0.32, 0.31], monkeypatch, restore_metric=0.32
        )
        meta = _read_meta(tmp_path)
        assert meta["stop_reason"] == "max_epochs"
        assert meta["stop_epoch"] == 2
        assert meta["best_epoch"] == 1
        assert meta["best_mAP50_95"] == pytest.approx(0.32)
        assert meta["restored_mAP50_95"] == pytest.approx(0.32)
        assert meta["restore_match"] is True
        assert meta["restore_skipped"] is False
        # last.pth is NOT overwritten by the restored best state
        last = _load_ckpt(tmp_path, "last.pth")
        assert last["last_epoch"] == 2

    def test_early_stop_restores_best(self, tmp_path, monkeypatch):
        solver = _make_solver(tmp_path, epoches=10, es=_es_yaml())
        # Exactly 3 training metrics: consumed by epochs 0-2 before the stop at
        # epoch 2; the finalize restore-eval then falls through to
        # restore_metric=0.30 instead of a stale training value.
        _run_fit(
            solver, [0.30, 0.29, 0.28],
            monkeypatch,
            restore_metric=0.30,
        )
        meta = _read_meta(tmp_path)
        assert meta["stop_reason"] == "early_stopping"
        assert meta["stop_epoch"] == 2
        assert meta["best_epoch"] == 0
        assert meta["best_mAP50_95"] == pytest.approx(0.30)
        assert meta["restored_mAP50_95"] == pytest.approx(0.30)

    def test_diagnostic_exit_does_not_restore(self, tmp_path, monkeypatch):
        """Design test 13: max_optimizer_steps exit skips best.pth restore."""
        # A best.pth exists, but diagnostic exit must not load it.
        torch.save(
            {
                "last_epoch": 0,
                "model": nn.Linear(2, 2).state_dict(),
                "early_stopping": {
                    "best_observed_metric": 0.99,
                    "best_significant_metric": 0.99,
                    "best_epoch": 0,
                    "epochs_without_improvement": 0,
                },
            },
            tmp_path / "best.pth",
        )
        solver = _make_solver(tmp_path, epoches=3, es=_es_yaml())
        _run_fit(solver, [0.30], monkeypatch, step_cap=True)
        meta = _read_meta(tmp_path)
        assert meta["stop_reason"] == "diagnostic"
        assert meta["restore_skipped"] is True
        assert meta["restored_mAP50_95"] is None
        assert meta["best_mAP50_95"] is None  # -inf normalized to None


class TestResumeCompat:

    def test_new_last_pth_restores_full_es_state(self, tmp_path, monkeypatch):
        """Design test 8: resume from a new last.pth restores the patience window."""
        solver = _make_solver(tmp_path, epoches=3, es=_es_yaml())
        _run_fit(solver, [0.30, 0.32, 0.31], monkeypatch)

        fresh = DetSolver.__new__(DetSolver)
        fresh.model = nn.Linear(2, 2)
        fresh.ema = None
        fresh.early_stopping = EarlyStoppingState()
        fresh.last_epoch = -1
        fresh.load_resume_state(str(tmp_path / "last.pth"))
        assert fresh.last_epoch == 2
        assert fresh.early_stopping.best_epoch == 1
        assert fresh.early_stopping.best_observed_metric == pytest.approx(0.32)
        assert fresh.early_stopping.epochs_without_improvement == 1

    def test_old_checkpoint_inits_from_pre_resume_validation(self, tmp_path, monkeypatch):
        """Design test 9: legacy checkpoint (no ES key) initializes from the
        current EMA validation and preserves it as the global best."""
        solver = _make_solver(tmp_path, epoches=5, es=_es_yaml())
        solver.last_epoch = 3  # resumed run: pre-resume validation will run
        # pre-resume eval -> 0.25 (init), epoch 4 -> 0.24 (no improvement)
        _run_fit(solver, [0.25, 0.24], monkeypatch)
        meta = _read_meta(tmp_path)
        assert meta["best_epoch"] == 3
        assert meta["best_mAP50_95"] == pytest.approx(0.25)


class TestStageTransition:

    def test_stage_transition_resets_patience_preserves_best(self, tmp_path, monkeypatch):
        """Design test 10: patience resets at the stage boundary; the global
        best checkpoint is preserved."""
        solver = _make_solver(
            tmp_path, epoches=4, stop_epoch=1, es=_es_yaml(min_epochs=0, patience=50),
            ema=True,
        )
        orig_init = solver._init_early_stopping

        def seeded_init():
            orig_init()
            solver.early_stopping.best_observed_metric = 0.30
            solver.early_stopping.best_significant_metric = 0.30
            solver.early_stopping.best_epoch = 0
            solver.early_stopping.epochs_without_improvement = 5

        monkeypatch.setattr(solver, "_init_early_stopping", seeded_init)
        # stage rollback loads best_stg1.pth; neutralize it for this unit test
        monkeypatch.setattr(solver, "load_resume_state", lambda path: None)
        _run_fit(solver, [0.30, 0.29, 0.28, 0.27], monkeypatch)
        last = _load_ckpt(tmp_path, "last.pth")
        # without reset, counter would be 9; reset at epoch 1 leaves 3
        assert last["early_stopping"]["epochs_without_improvement"] == 3
        meta = _read_meta(tmp_path)
        assert meta["best_epoch"] == 0
        assert meta["best_mAP50_95"] == pytest.approx(0.30)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_det_solver_early_stopping.py -v`
Expected: FAIL — `FileNotFoundError` on `final_run_meta.json` (the exit-capture + finalize call in Edit 5.1 is not yet present), so every `TestFinalizeRestore`/`TestResumeCompat`/`TestStageTransition` test fails while all Task-3/4 tests still pass.

- [ ] **Step 3: Write minimal implementation**

Edit `engine/solver/det_solver.py`:

**Edit 5.1 — exit capture + finalize call** (current lines 315–317):

```python
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
```

replace with:

```python
        stop_reason = "max_epochs"
        if getattr(self, "_diagnostic_exit", False):
            stop_reason = "diagnostic"
        elif stop_early:
            stop_reason = "early_stopping"
        stop_epoch = self.last_epoch
        print(f"[Training] exit reason: {stop_reason} at epoch {stop_epoch}")
        self._finalize_training(stop_reason, stop_epoch, box_mode)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
```

**Edit 5.2 — `_finalize_training` method** (insert next to the other new methods):

```python
    def _finalize_training(self, stop_reason, stop_epoch, box_mode):
        """Restore best.pth for normal exits, re-validate, write metadata.

        Capture stop/best metadata BEFORE loading best.pth so the record is
        independent of the restored checkpoint's last_epoch. Never restores on
        the diagnostic path. Never overwrites last.pth.
        """
        if self.early_stopping is None:
            return
        best_epoch = self.early_stopping.best_epoch
        best_metric = self.early_stopping.best_observed_metric
        meta_best = float(best_metric) if math.isfinite(best_metric) else None
        meta = {
            "stop_reason": stop_reason,
            "stop_epoch": stop_epoch,
            "best_epoch": best_epoch,
            "best_mAP50_95": meta_best,
            "restored_mAP50_95": None,
            "epochs_after_best": max(0, stop_epoch - best_epoch),
            "restore_skipped": False,
            "restore_match": None,
        }
        can_restore = (
            self.early_stopping_config.restore_best
            and stop_reason != "diagnostic"
            and self.output_dir is not None
            and (self.output_dir / "best.pth").exists()
        )
        if can_restore:
            if dist_utils.is_dist_available_and_initialized():
                torch.distributed.barrier()
            self.load_resume_state(str(self.output_dir / "best.pth"))
            module = self.ema.module if self.ema else self.model
            restored_stats, _ = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                box_mode=box_mode,
            )
            restored = restored_stats.get("mAP50_95", 0)
            meta["restored_mAP50_95"] = restored
            meta["restore_match"] = (
                abs(restored - best_metric) <= RESTORED_METRIC_TOLERANCE
            )
            if not meta["restore_match"]:
                print(
                    f"[EarlyStopping] WARNING restored mAP50_95={restored:.4f} "
                    f"differs from best {best_metric:.4f} by more than "
                    f"{RESTORED_METRIC_TOLERANCE}"
                )
        else:
            meta["restore_skipped"] = True

        if self.output_dir and dist_utils.is_main_process():
            with (self.output_dir / "final_run_meta.json").open("w") as f:
                json.dump(meta, f, indent=2)
        print(f"[Training] final metadata: {meta}")
```

**Edit 5.3 — add `math` import** (top of file, with the other imports):

```python
import math
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_det_solver_early_stopping.py -v`
Expected: PASS (16 tests: 2 disabled, 2 best-ckpt, 2 patience, 1 ordering, 3 DDP, 3 finalize, 2 resume, 1 stage).

- [ ] **Step 5: Verify diagnostics on changed files**

Run: `python -m py_compile engine/solver/early_stopping.py engine/solver/det_solver.py && python -m pytest test/test_early_stopping.py test/test_det_solver_early_stopping.py -q`
Expected: exit 0, all tests pass.

- [ ] **Step 6: No commit (user constraint)** — proceed.

---

### Task 6: ES-Base FP16/BF16 configs + config inheritance tests

**Files:**
- Create: `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_amp_es.yml`
- Create: `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16_es.yml`
- Test: `test/test_early_stopping_configs.py`

**Interfaces:**
- Consumes: existing `sp_fz_rep0_nloss_amp.yml`, `sp_fz_rep0_nloss_bf16.yml` (untouched).
- Produces: two ES-Base configs resolving the approved `early_stopping` block while preserving base schedule/precision.

- [ ] **Step 1: Write the failing test**

Create `test/test_early_stopping_configs.py` (mirrors `test_obb_loss_experiment_configs.py` patterns):

```python
"""ES-Base config inheritance tests.

Verifies the approved early_stopping block appears only in the two new *_es.yml
configs, that schedule/precision are inherited unchanged, and that the legacy
bases and the abandoned C-group config stay unpolluted.
"""

from pathlib import Path
from typing import Final

import pytest

from engine.core.yaml_config import YAMLConfig


ROOT: Final = Path(__file__).resolve().parents[1]
DLZDT_DIR: Final = ROOT / "configs" / "custom_obb" / "dlzdt"

AMP_BASE: Final = "sp_fz_rep0_nloss_amp.yml"
BF16_BASE: Final = "sp_fz_rep0_nloss_bf16.yml"
ES_AMP: Final = "sp_fz_rep0_nloss_amp_es.yml"
ES_BF16: Final = "sp_fz_rep0_nloss_bf16_es.yml"
C_GROUP: Final = "sp_fz_rep0_nloss_fp16_obb_fp32.yml"

EXPECTED_ES: Final = {
    "enabled": True,
    "metric": "mAP50_95",
    "mode": "max",
    "min_epochs": 100,
    "patience": 12,
    "min_delta": 0.001,
    "restore_best": True,
}


def _dlzdt_config(name: str) -> dict:
    return YAMLConfig(str(DLZDT_DIR / name)).yaml_cfg


def test_legacy_bases_have_no_early_stopping() -> None:
    for name in (AMP_BASE, BF16_BASE, C_GROUP):
        assert "early_stopping" not in _dlzdt_config(name), name


@pytest.mark.parametrize("name", [ES_AMP, ES_BF16])
def test_es_config_carries_approved_block(name: str) -> None:
    assert _dlzdt_config(name)["early_stopping"] == EXPECTED_ES


@pytest.mark.parametrize(
    ("name", "amp_dtype", "scaler_enabled"),
    [
        (ES_AMP, None, True),        # inherits FP16 default from amp base
        (ES_BF16, "bfloat16", False),  # inherits BF16 + disabled scaler
    ],
)
def test_es_config_preserves_precision(name, amp_dtype, scaler_enabled) -> None:
    config = _dlzdt_config(name)
    assert config.get("amp_dtype") == amp_dtype
    # obb_geometry_fp32 is unset (None) in the base chain; ES configs must not
    # introduce it (only the abandoned C-group sets it True).
    assert config.get("obb_geometry_fp32") is not True
    assert config["scaler"]["enabled"] is scaler_enabled


def test_es_configs_preserve_base_training_schedule() -> None:
    base = _dlzdt_config(AMP_BASE)
    for name in (ES_AMP, ES_BF16):
        config = _dlzdt_config(name)
        assert config["epoches"] == base["epoches"] == 150
        assert config["warmup_iter"] == base["warmup_iter"] == 15
        assert config["flat_epoch"] == base["flat_epoch"] == 75
        assert config["no_aug_epoch"] == base["no_aug_epoch"] == 15
        assert config["optimizer"]["lr"] == base["optimizer"]["lr"] == 0.0005
        assert config["DEIMCriterion"]["weight_dict"] == base["DEIMCriterion"]["weight_dict"]
        assert config["DEIMCriterion"]["matcher"] == base["DEIMCriterion"]["matcher"]


def test_es_configs_have_distinct_output_dirs() -> None:
    outputs = {_dlzdt_config(name)["output_dir"] for name in (AMP_BASE, ES_AMP, ES_BF16)}
    assert len(outputs) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_early_stopping_configs.py -v`
Expected: FAIL — `FileNotFoundError` for `sp_fz_rep0_nloss_amp_es.yml`.

- [ ] **Step 3: Write minimal implementation**

Create `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_amp_es.yml`:

```yaml
__include__: ['./sp_fz_rep0_nloss_amp.yml']

#### 项目配置
output_dir: ./outputs/deimv2_obb_dlzdt_sp_fz_rep0_nloss_amp_es

#### ES-Base：EMA mAP50_95 early stopping + 恢复全局最佳 checkpoint
early_stopping:
  enabled: true
  metric: mAP50_95
  mode: max
  min_epochs: 100
  patience: 12
  min_delta: 0.001
  restore_best: true
```

Create `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16_es.yml`:

```yaml
__include__: ['./sp_fz_rep0_nloss_bf16.yml']

#### 项目配置
output_dir: ./outputs/deimv2_obb_dlzdt_sp_fz_rep0_nloss_bf16_es

#### ES-Base：与 FP16 相同的 early-stopping 参数，各自独立选择最佳 epoch
early_stopping:
  enabled: true
  metric: mAP50_95
  mode: max
  min_epochs: 100
  patience: 12
  min_delta: 0.001
  restore_best: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_early_stopping_configs.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: No commit (user constraint)** — proceed.

---

### Task 7: Full regression

**Files:** none (verification only).

- [ ] **Step 1: Run the focused suite**

Run: `python -m pytest test/test_early_stopping.py test/test_det_solver_early_stopping.py test/test_early_stopping_configs.py -q`
Expected: exit 0, all tests pass.

- [ ] **Step 2: Run the neighboring regression suites**

Run: `python -m pytest test/test_det_engine_diagnostics.py test/test_obb_loss_experiment_configs.py -q`
Expected: exit 0. (These cover the mocked-flow style, config-loading style, and the pre-existing diagnostics the solver still depends on.)

- [ ] **Step 3: Sanity-check imports of the two modified modules**

Run: `python -c "import engine.solver.det_solver; import engine.solver.early_stopping; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Report completion**

Summarize: files created/modified, design-test coverage matrix (13/13), and the two runnable experiment commands (from repo root):
- FP16: `python tools/train.py -c configs/custom_obb/dlzdt/sp_fz_rep0_nloss_amp_es.yml` (verify actual entrypoint with `python -m` if the repo uses a module entry)
- BF16: `python tools/train.py -c configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16_es.yml`

---

## Design Test Coverage Matrix

| # | Design requirement | Task / test |
|---|---|---|
| 1 | Missing/disabled config preserves behavior | T1 config tests + T3 `TestDisabledBehavior` |
| 2 | Strict improvement saves complete `best.pth` | T4 `test_strict_improvement_saves_best_pth` |
| 3 | Sub-`min_delta` gain updates observed best, no patience reset | T2 `test_small_improvement_...` + T4 integration |
| 4 | Significant improvement resets patience | T2 `test_significant_improvement_resets_patience` |
| 5 | Patience cannot stop before `min_epochs` | T2 `test_should_stop_respects_min_epochs_floor` + T4 `test_no_stop_before_min_epochs` |
| 6 | Exhausted patience stops all DDP ranks | T4 `TestDDPBroadcast` (3 tests) |
| 7 | 150-epoch limit restores + validates `best.pth` | T5 `test_max_epochs_restores_and_validates_best_pth` |
| 8 | New `last.pth` restores complete ES state | T5 `test_new_last_pth_restores_full_es_state` |
| 9 | Old checkpoint initializes from EMA validation | T5 `test_old_checkpoint_inits_from_pre_resume_validation` |
| 10 | Stage transition resets patience, preserves global best | T5 `TestStageTransition` |
| 11 | Restored eval matches saved best within tolerance | T5 `test_max_epochs_restores...` (`restore_match`) |
| 12 | `last.pth` remains actual final training epoch | T4 `TestCheckpointOrdering` + T5 `last["last_epoch"] == 2` |
| 13 | `max_optimizer_steps` diagnostic exit does not restore | T5 `test_diagnostic_exit_does_not_restore` |

## Non-Goals (unchanged from the design)

- Guaranteeing a flat final-five-epoch curve; this guarantees delivered-checkpoint quality.
- Changing the 150-epoch hard limit, LR boundaries, augmentation, regularization, loss weights, precision, or architecture in ES-Base.
- Redesigning `best_stg1.pth`/`best_stg2.pth` semantics.
- Replacing EMA validation with raw-model validation.
