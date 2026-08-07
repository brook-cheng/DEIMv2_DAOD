# Remove BF16 and FP16/OBB-FP32 Training Modes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the rejected BF16 and FP16-forward/FP32-OBB-geometry training modes so the only remaining mixed-precision path is ordinary CUDA FP16 autocast with the enabled GradScaler.

**Architecture:** `DetSolver.fit()` stops reading `amp_dtype`/`obb_geometry_fp32`; `train_one_epoch` hardcodes `dtype=torch.float16` in model autocast and always passes raw model outputs to the criterion (no recursive geometry cast); `engine/solver/precision.py` loses all callers and is deleted; the three rejected YAML configs and the 2026-08-06 experiment docs are deleted; tests are rewritten to pin ABSENCE (module gone, files gone, keys gone) plus preserved FP16 behavior.

**Tech Stack:** Python 3, PyTorch AMP (`torch.autocast`, `torch.amp.GradScaler`), pytest, RT-DETR-style `YAMLConfig` (`__include__` chains).

## Global Constraints

- **No compatibility shim.** Deleted keys (`amp_dtype`, `obb_geometry_fp32`) and deleted configs are not accepted anywhere. A stale user-supplied override becomes an unused YAML key.
- **No commits.** User constraint from the prior session: do not `git commit` unless explicitly asked.
- **Preserve untouched:** `use_amp` switch, GradScaler construction/checkpointing (`engine/core/_config.py`, `yaml_config.py`), the full AMP branch in `train_one_epoch`, criterion inside the `torch.autocast(..., enabled=False)` block receiving raw model outputs, non-finite diagnostics, gradient clipping, scaler step/update, EMA, scheduler, `sp_fz_rep0_nloss_amp.yml`, `sp_fz_rep0_nloss_amp_es.yml`, and **all DINOv3 backbone BF16/FP8 internals** (`engine/backbone/dinov3/`).
- **Do NOT touch:** `configs/custom_obb/dlzdt/ablation/`, `configs/custom_obb/dlzdt/sp_fz_common.yml` (both untracked user files).
- **Run all pytest from repo root** `/mnt/d/cx/thired/deimv2_daod` with `python -m pytest` (config tests use CWD-relative paths; `python -m pytest` adds CWD to `sys.path`).
- **`yaml_utils.load_config` mutable-default bug:** config tests that load multiple configs must reset `yaml_utils.load_config.__defaults__ = ({},)` before each load (see existing `test_early_stopping_configs.py` comment).
- Historical changelog entries and generic FP16/TensorRT docs remain unchanged.

---

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `engine/solver/precision.py` | `resolve_amp_dtype`, `validate_amp_dtype_support`, `cast_obb_geometry_fp32`, `GEOMETRY_KEYS` | **Delete** |
| `engine/solver/det_engine.py` | Remove precision import + `box_mode`/dtype-resolution lines; hardcode `torch.float16`; criterion gets raw `outputs` | Modify |
| `engine/solver/det_solver.py` | Remove `amp_dtype_name=`/`obb_geometry_fp32=` kwargs in `train_one_epoch` call | Modify |
| `test/test_precision_policy.py` | Rewrite: pin module absence + FP16 wiring preservation | Rewrite |
| `test/test_obb_loss_experiment_configs.py` | Remove `PRECISION_EXPERIMENTS` section; add deleted-config absence test | Modify |
| `test/test_early_stopping_configs.py` | FP16-only; remove BF16/C-group constants; add deleted-config absence test | Modify |
| `test/test_det_solver_early_stopping.py` | Remove `amp_dtype`/`obb_geometry_fp32` from mocked `yaml_cfg` fixture | Modify |
| `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16.yml` | Rejected B-group config | **Delete** |
| `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16_es.yml` | Rejected BF16 ES config | **Delete** |
| `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_fp16_obb_fp32.yml` | Rejected C-group config | **Delete** |
| `docs/superpowers/specs/2026-08-06-amp-bf16-obb-fp32-experiments-design.md` | Rejected experiment design | **Delete** |
| `docs/superpowers/plans/2026-08-06-amp-bf16-obb-fp32-experiments.md` | Rejected experiment plan | **Delete** |
| `docs/superpowers/INDEX.md` | Remove the two 2026-08-06 entries (specs ~line 46, plans ~line 96) | Modify |
| `docs/superpowers/specs/2026-08-07-early-stopping-best-checkpoint-design.md` | Line 23: drop "FP16 and BF16 experiments" wording | Modify |
| `docs/superpowers/plans/2026-08-07-ema-early-stopping-best-checkpoint.md` | Remove BF16/C-group config rows, embedded test examples, launch commands | Modify |
| `docs/superpowers/plans/2026-08-07-remove-bf16-obb-fp32-training-plan.md` | Corrupted single-line plan artifact | **Delete** (replaced by this file) |

---

### Task 1: Rewrite `test_precision_policy.py` to pin absence + FP16 preservation (RED)

**Files:**
- Rewrite: `test/test_precision_policy.py`

**Interfaces:**
- Consumes: `engine.solver.det_engine.train_one_epoch` (current signature with `**kwargs`).
- Produces: the post-removal contract. Later tasks must make `test_precision_module_deleted` pass (delete `precision.py`) while the wiring tests stay green (hardcode `float16`, no geometry cast).

- [ ] **Step 1: Rewrite the test file**

Delete all tests for `resolve_amp_dtype` (3), `validate_amp_dtype_support` (4), `cast_obb_geometry_fp32` (4), and the 5 wiring tests that exercise BF16/OBB-FP32 kwargs. Keep the `_AutocastRecorder`/`_WiringModel`/`_WiringCriterion`/`_wiring_batches` harness but drop the `amp_dtype_name`/`obb_geometry_fp32` parameters. Replace with:

```python
import contextlib
import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.solver import det_engine


# --- precision module absence ------------------------------------------------


def test_precision_module_deleted() -> None:
    # Given: the mode-specific precision helpers were removed.
    # When: the deleted module is imported.
    # Then: an ImportError is raised.
    with pytest.raises(ImportError):
        from engine.solver import precision  # noqa: F401


# --- train_one_epoch FP16 wiring preservation -------------------------------


class _AutocastRecorder:
    """Replaces torch.autocast; records every call's kwargs as a no-op context."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs) -> contextlib.nullcontext:
        self.calls.append(kwargs)
        return contextlib.nullcontext()


class _WiringModel(nn.Module):
    """Tiny module whose output dict has pred_boxes tied to a Parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, targets=None):
        return {"pred_boxes": self.scale * torch.randn(2, 4)}


class _WiringCriterion:
    """Records the outputs object it received; box_mode is configurable."""

    def __init__(self, box_mode: str) -> None:
        self.box_mode = box_mode
        self.last_outputs = None

    def train(self) -> None:
        pass

    def __call__(self, outputs, targets, **metas):
        self.last_outputs = outputs
        return {"loss_bbox": outputs["pred_boxes"].sum()}


def _wiring_batches(n: int = 2):
    return [
        (
            torch.randn(2, 3, 64, 64),
            [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])}],
        )
        for _ in range(n)
    ]


def _run_amp_epoch(monkeypatch, *, box_mode="obb"):
    recorder = _AutocastRecorder()
    monkeypatch.setattr(torch, "autocast", recorder)

    model = _WiringModel()
    criterion = _WiringCriterion(box_mode)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu")

    det_engine.train_one_epoch(
        False, None, model, criterion, _wiring_batches(1), optimizer,
        torch.device("cpu"), 0, max_norm=0.5, print_freq=100, scaler=scaler,
    )
    return recorder, criterion


def test_train_one_epoch_autocast_uses_fixed_float16(monkeypatch) -> None:
    # Given: no precision-mode kwargs exist (post-removal).
    # When: the AMP epoch runs.
    # Then: the model autocast context receives dtype=torch.float16.
    recorder, _ = _run_amp_epoch(monkeypatch)
    model_autocast_kwargs = [c for c in recorder.calls if "dtype" in c]
    assert model_autocast_kwargs, "model autocast call missing dtype kwarg"
    assert model_autocast_kwargs[0]["dtype"] is torch.float16


def test_train_one_epoch_criterion_receives_raw_outputs(monkeypatch) -> None:
    # Given: the OBB criterion with no geometry-cast configuration.
    # When: the AMP epoch runs.
    # Then: the criterion receives the original model outputs unchanged.
    _, criterion = _run_amp_epoch(monkeypatch, box_mode="obb")
    assert criterion.last_outputs is not None
    assert "pred_boxes" in criterion.last_outputs
```

- [ ] **Step 2: Run the test to verify RED**

Run: `python -m pytest test/test_precision_policy.py -v`
Expected: `test_precision_module_deleted` **FAILS** (module still importable → `pytest.raises(ImportError)` did not trigger). The two wiring tests **PASS** (they are preservation anchors and already hold). This mixed state is the correct RED for a deletion task.

---

### Task 2: Delete the precision module and hardcode FP16 (GREEN)

**Files:**
- Modify: `engine/solver/det_engine.py`
- Modify: `engine/solver/det_solver.py`
- Delete: `engine/solver/precision.py`

**Interfaces:**
- Consumes: Task 1's RED contract.
- Produces: `train_one_epoch` with no `amp_dtype_name`/`obb_geometry_fp32` reads and `dtype=torch.float16`; `DetSolver.fit()` without those kwargs.

- [ ] **Step 1: Remove the precision import in `det_engine.py`**

```python
from .precision import (
    cast_obb_geometry_fp32,
    resolve_amp_dtype,
    validate_amp_dtype_support,
)
```
(delete lines 32–36)

- [ ] **Step 2: Remove `box_mode` and precision resolution lines in `train_one_epoch`**

Delete line `box_mode = criterion.box_mode` and the block:

```python
    amp_dtype_name: str | None = kwargs.get("amp_dtype_name", None)
    obb_geometry_fp32: bool = kwargs.get("obb_geometry_fp32", False)
    amp_dtype = resolve_amp_dtype(amp_dtype_name)
    validate_amp_dtype_support(amp_dtype, device)
    use_obb_geometry_fp32 = obb_geometry_fp32 and box_mode == "obb"
```

- [ ] **Step 3: Hardcode `torch.float16` in the model autocast call**

`dtype=amp_dtype` → `dtype=torch.float16` (the scaler branch, inside `with torch.autocast(...)`).

- [ ] **Step 4: Pass raw outputs to the criterion**

Replace:

```python
            with torch.autocast(device_type=str(device), enabled=False):
                criterion_inputs = (
                    cast_obb_geometry_fp32(outputs)
                    if use_obb_geometry_fp32
                    else outputs
                )
                loss_dict = criterion(criterion_inputs, targets, **metas)
```

with:

```python
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
```

- [ ] **Step 5: Remove the kwargs in `det_solver.py`**

Delete from the `train_one_epoch(...)` call (currently lines 197–200):

```python
                amp_dtype_name=args.yaml_cfg.get("amp_dtype"),
                obb_geometry_fp32=bool(
                    args.yaml_cfg.get("obb_geometry_fp32", False)
                ),
```

- [ ] **Step 6: Delete `engine/solver/precision.py`**

```bash
rm engine/solver/precision.py
```

- [ ] **Step 7: Run the precision suite to verify GREEN**

Run: `python -m pytest test/test_precision_policy.py -v`
Expected: all 3 tests **PASS** (`test_precision_module_deleted` now raises ImportError; wiring tests still green).

- [ ] **Step 8: Run adjacent engine suites**

Run: `python -m pytest test/test_det_engine_diagnostics.py test/test_early_stopping.py -v`
Expected: PASS (diagnostics exercise the scaler branch; ES tests are unrelated to precision kwargs).

---

### Task 3: Rewrite config tests to FP16-only and pin deleted-config absence (RED)

**Files:**
- Rewrite: `test/test_early_stopping_configs.py`
- Modify: `test/test_obb_loss_experiment_configs.py`

**Interfaces:**
- Consumes: Task 1's RED style (absence pins).
- Produces: tests asserting the 3 rejected YAML files do not exist and the retained configs carry no `amp_dtype`/`obb_geometry_fp32`.

- [ ] **Step 1: Rewrite `test/test_early_stopping_configs.py`**

Remove `BF16_BASE`, `ES_BF16`, `C_GROUP` constants and every BF16/C-group reference. Keep `_dlzdt_config` with the `yaml_utils` default-reset workaround. New content:

```python
"""ES-Base config inheritance tests (FP16 only).

Verifies the approved early_stopping block appears only in the FP16 *_es.yml
config, that schedule/precision are inherited unchanged, and that the
rejected BF16 / FP16+OBB-FP32 configs are deleted.
"""

from pathlib import Path
from typing import Final

import pytest

from engine.core import yaml_utils
from engine.core.yaml_config import YAMLConfig


ROOT: Final = Path(__file__).resolve().parents[1]
DLZDT_DIR: Final = ROOT / "configs" / "custom_obb" / "dlzdt"

AMP_BASE: Final = "sp_fz_rep0_nloss_amp.yml"
ES_AMP: Final = "sp_fz_rep0_nloss_amp_es.yml"

DELETED_CONFIGS: Final = (
    "sp_fz_rep0_nloss_bf16.yml",
    "sp_fz_rep0_nloss_bf16_es.yml",
    "sp_fz_rep0_nloss_fp16_obb_fp32.yml",
)

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
    # load_config() uses a mutable default dict accumulator that persists
    # across calls (pre-existing RT-DETR yaml_utils.py bug). Reset it so each
    # config load starts from a clean slate.
    yaml_utils.load_config.__defaults__ = ({},)
    return YAMLConfig(str(DLZDT_DIR / name)).yaml_cfg


def test_deleted_precision_configs_absent_from_disk() -> None:
    for name in DELETED_CONFIGS:
        assert not (DLZDT_DIR / name).exists(), name


def test_legacy_base_has_no_early_stopping() -> None:
    assert "early_stopping" not in _dlzdt_config(AMP_BASE)


def test_es_config_carries_approved_block() -> None:
    assert _dlzdt_config(ES_AMP)["early_stopping"] == EXPECTED_ES


def test_es_config_has_no_precision_mode_keys() -> None:
    config = _dlzdt_config(ES_AMP)
    assert "amp_dtype" not in config
    assert "obb_geometry_fp32" not in config
    assert config["scaler"]["enabled"] is True


def test_es_config_preserves_base_training_schedule() -> None:
    base = _dlzdt_config(AMP_BASE)
    config = _dlzdt_config(ES_AMP)
    assert config["epoches"] == base["epoches"] == 150
    assert config["warmup_iter"] == base["warmup_iter"] == 15
    assert config["flat_epoch"] == base["flat_epoch"] == 75
    assert config["no_aug_epoch"] == base["no_aug_epoch"] == 15
    assert config["optimizer"]["lr"] == base["optimizer"]["lr"] == 0.0005
    assert config["DEIMCriterion"]["weight_dict"] == base["DEIMCriterion"]["weight_dict"]
    assert config["DEIMCriterion"]["matcher"] == base["DEIMCriterion"]["matcher"]


def test_es_configs_have_distinct_output_dirs() -> None:
    outputs = {_dlzdt_config(name)["output_dir"] for name in (AMP_BASE, ES_AMP)}
    assert len(outputs) == 2
```

- [ ] **Step 2: Modify `test/test_obb_loss_experiment_configs.py`**

Delete the entire trailing section (currently lines 200–267):

```python
# ── AMP BF16 / OBB FP32 experiment configs ──────────────────────────────────
...
def test_precision_experiments_have_distinct_output_dirs() -> None:
```

This removes `DLZDT_DIR`, `AMP_BASE`, `PRECISION_EXPERIMENTS`, `_dlzdt_config`, and the 3 precision tests. The synthetic-loss config tests above remain untouched. Do **not** delete the file — the synthetic-loss contract tests stay.

- [ ] **Step 3: Run both suites to verify RED**

Run: `python -m pytest test/test_early_stopping_configs.py test/test_obb_loss_experiment_configs.py -v`
Expected: `test_deleted_precision_configs_absent_from_disk` **FAILS** (files still exist). All other tests PASS.

---

### Task 4: Delete the rejected configs and experiment docs (GREEN)

**Files:**
- Delete: 3 rejected YAML configs, 2 experiment docs, corrupted plan artifact.

**Interfaces:**
- Consumes: Task 3's RED absence pins.
- Produces: Task 3 suites GREEN; repository no longer contains the rejected files.

- [ ] **Step 1: Delete the rejected configs**

```bash
rm configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16.yml \
   configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16_es.yml \
   configs/custom_obb/dlzdt/sp_fz_rep0_nloss_fp16_obb_fp32.yml
```

- [ ] **Step 2: Delete the 2026-08-06 experiment docs and corrupted plan**

```bash
rm docs/superpowers/specs/2026-08-06-amp-bf16-obb-fp32-experiments-design.md \
   docs/superpowers/plans/2026-08-06-amp-bf16-obb-fp32-experiments.md \
   docs/superpowers/plans/2026-08-07-remove-bf16-obb-fp32-training-plan.md
```

- [ ] **Step 3: Run both config suites to verify GREEN**

Run: `python -m pytest test/test_early_stopping_configs.py test/test_obb_loss_experiment_configs.py -v`
Expected: all PASS.

- [ ] **Step 4: Real YAMLConfig parse of the retained ES config**

Run:
```bash
python -c "
from engine.core.yaml_config import YAMLConfig
cfg = YAMLConfig('configs/custom_obb/dlzdt/sp_fz_rep0_nloss_amp_es.yml').yaml_cfg
assert 'amp_dtype' not in cfg and 'obb_geometry_fp32' not in cfg
assert cfg['early_stopping']['metric'] == 'mAP50_95'
assert cfg['scaler']['enabled'] is True
print('OK: amp_es.yml resolves FP16-only with scaler enabled')
"
```
Expected: prints the OK line (exit 0).

---

### Task 5: Update `test_det_solver_early_stopping.py` mocked fixture

**Files:**
- Modify: `test/test_det_solver_early_stopping.py`

**Interfaces:**
- Consumes: Task 2 (solver no longer reads these keys).
- Produces: mocked `yaml_cfg` matching the post-removal solver contract.

- [ ] **Step 1: Remove the two stale keys from `_make_solver`**

In `_make_solver`'s `yaml_cfg` dict, delete:

```python
        "amp_dtype": None,
        "obb_geometry_fp32": False,
```

The dict keeps `max_optimizer_steps`, `fail_on_zero_grad`, `nan_max_events`, and the conditional `early_stopping` key.

- [ ] **Step 2: Run the ES integration suite**

Run: `python -m pytest test/test_det_solver_early_stopping.py -v`
Expected: all PASS.

---

### Task 6: Update active documentation

**Files:**
- Modify: `docs/superpowers/INDEX.md`
- Modify: `docs/superpowers/specs/2026-08-07-early-stopping-best-checkpoint-design.md`
- Modify: `docs/superpowers/plans/2026-08-07-ema-early-stopping-best-checkpoint.md`

**Interfaces:**
- Consumes: Task 4 deletions.
- Produces: docs that no longer advertise BF16 / FP16+OBB-FP32 as supported choices.

- [ ] **Step 1: Remove the two 2026-08-06 entries from `INDEX.md`**

Delete the line (specs section):
```
[2026-08-06-amp-bf16-obb-fp32-experiments-design](specs/2026-08-06-amp-bf16-obb-fp32-experiments-design.md) — AMP BF16 与 OBB FP32 实验设计：混合精度（BF16）训练下 OBB 分支 FP32 保持的实验设计。
```
and the line (plans section):
```
[2026-08-06-amp-bf16-obb-fp32-experiments](plans/2026-08-06-amp-bf16-obb-fp32-experiments.md) — AMP BF16 与 OBB FP32 实验实现计划：混合精度训练下 OBB FP32 保持实验的实施计划。
```

- [ ] **Step 2: Update the early-stopping design doc**

Replace line 23–24:
```
- The FP16 and BF16 experiments use identical early-stopping parameters while
  independently selecting their own best epoch.
```
with:
```
- The FP16 experiment uses the approved early-stopping parameters and selects
  its own best epoch.
```

- [ ] **Step 3: Update the early-stopping plan doc**

Surgical reference cleanup (do not restructure the historical plan):
1. Line 19: remove `precision settings (`amp_dtype`, `obb_geometry_fp32`, scaler)` from the "Do not modify" list → keep only "scaler" if desired, or drop the precision parenthetical entirely.
2. Line 20: drop the abandoned C-group sentence (`Do **not** add early_stopping to the abandoned C-group config (sp_fz_rep0_nloss_fp16_obb_fp32.yml) or to the legacy base configs. Only the two new *_es.yml configs carry it.`) → replace with a single sentence: only the FP16 `*_es.yml` config carries the `early_stopping` block.
3. Line 37 (file table): delete the `sp_fz_rep0_nloss_bf16_es.yml` row.
4. Lines 482–483: remove `"amp_dtype": None,` / `"obb_geometry_fp32": False,` from the embedded `_make_solver` example.
5. Task 6 (≈lines 1299–1480): retitle to "ES-Base FP16 configs + config inheritance tests"; remove the `bf16_es.yml` create step, the `BF16_BASE`/`ES_BF16`/`C_GROUP` constants, the `(ES_BF16, "bfloat16", False)` parametrization row, the BF16 config YAML block, and the BF16 launch command (`python tools/train.py -c configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16_es.yml`).
6. Keep the FP16 `amp_es.yml` create step and its embedded test examples intact.

- [ ] **Step 4: Verify no stale references remain in active docs**

Run:
```bash
grep -rn "nloss_bf16\|nloss_fp16_obb_fp32\|amp_dtype\|obb_geometry_fp32" \
  docs/superpowers/INDEX.md \
  docs/superpowers/specs/2026-08-07-early-stopping-best-checkpoint-design.md \
  docs/superpowers/plans/2026-08-07-ema-early-stopping-best-checkpoint.md
```
Expected: no matches (exit 1). The removal design + this plan intentionally mention the deleted symbols and are excluded.

---

### Task 7: Final verification

**Files:**
- Read-only: repository-wide grep, focused + neighboring suites, import/compile checks.

- [ ] **Step 1: Focused suites**

Run: `python -m pytest test/test_precision_policy.py test/test_early_stopping_configs.py test/test_obb_loss_experiment_configs.py test/test_det_solver_early_stopping.py -v`
Expected: all PASS.

- [ ] **Step 2: Neighboring suites**

Run: `python -m pytest test/test_det_engine_diagnostics.py test/test_early_stopping.py -v`
Expected: all PASS.

- [ ] **Step 3: Repository-wide absence sweep**

Run:
```bash
grep -rn "amp_dtype\|obb_geometry_fp32" --include="*.py" --include="*.yml" . \
  | grep -v "^\./docs/superpowers/specs/2026-08-07-remove-bf16-obb-fp32-training-design.md:" \
  | grep -v "^\./docs/superpowers/plans/2026-08-07-remove-bf16-obb-fp32-training.md:"
```
Expected: only the removal design/plan docs (which document the deletion) match. No `.py`, no `.yml`.

Also confirm the rejected config filenames no longer exist:
```bash
ls configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16*.yml configs/custom_obb/dlzdt/sp_fz_rep0_nloss_fp16_obb_fp32.yml 2>&1
```
Expected: `No such file` errors (files gone).

- [ ] **Step 4: Import/compile checks**

Run:
```bash
python -c "from engine.solver.det_engine import train_one_epoch; from engine.solver.det_solver import DetSolver; import engine.solver; print('imports OK')"
python -m compileall -q engine/solver/det_engine.py engine/solver/det_solver.py test/test_precision_policy.py test/test_early_stopping_configs.py test/test_obb_loss_experiment_configs.py test/test_det_solver_early_stopping.py && echo "compile OK"
```
Expected: both print OK (exit 0).

- [ ] **Step 5: Confirm DINOv3 backbone BF16/FP8 untouched**

Run: `git status --short engine/backbone/dinov3/`
Expected: no changes listed.

- [ ] **Step 6: Confirm untracked user files untouched**

Run: `git status --short configs/custom_obb/dlzdt/`
Expected: only `ablation/` and `sp_fz_common.yml` remain as untracked user files; no other config changes.

---

## Self-Review Notes (completed at plan-writing time)

- **Spec coverage:** every "Remove" item in the design doc maps to Tasks 1–6; every "Preserve" item is either asserted (Task 1/3 wiring + config anchors, Task 7 step 5) or explicitly left untouched (constraints). Test strategy items 1–5 map to Tasks 3/1/2/5/7.
- **Placeholder scan:** no TBD/TODO; every step carries real code or exact commands.
- **Type consistency:** `train_one_epoch` retains its `**kwargs` signature (unchanged), so `det_solver.py` edits only drop two kwargs; `_run_amp_epoch` drops the same two kwargs; `_WiringCriterion.box_mode` remains for the preservation test.
- **Known risk (accepted):** `test_precision_module_deleted` uses `from engine.solver import precision` — Python caches modules, but each pytest process imports fresh, so the ImportError assertion is reliable.
