# AMP BF16 and OBB FP32 Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configuration-driven BF16 AMP and FP16-forward/FP32-OBB-geometry experiment modes while preserving the existing FP16 default.

**Architecture:** Resolve the autocast dtype and geometry policy once from YAML, pass both values through `DetSolver` into `train_one_epoch`, and keep the existing optimizer/EMA/scheduler path shared. A focused precision helper recursively casts only OBB geometry tensors before criterion evaluation, preserving nested output structure and autograd.

**Tech Stack:** Python 3, PyTorch 2.5 AMP, YAMLConfig registry, pytest.

## Global Constraints

- Missing `amp_dtype` must preserve CUDA FP16 autocast behavior.
- Supported dtype names are exactly `float16` and `bfloat16`.
- BF16 must fail clearly before the first batch on unsupported CUDA hardware.
- Group C casts all nested `pred_boxes`, `pred_corners`, and `ref_points` tensors to FP32 only for OBB training.
- Classification logits, metadata, optimizer, EMA, scheduler, clipping, and loss formulas remain unchanged.
- Existing YAML files remain valid.

---

### Task 1: Precision Policy Helpers

**Files:**
- Create: `engine/solver/precision.py`
- Create: `test/test_precision_policy.py`

**Interfaces:**
- Produces: `resolve_amp_dtype(name: str | None) -> torch.dtype`
- Produces: `validate_amp_dtype_support(dtype: torch.dtype, device: torch.device) -> None`
- Produces: `cast_obb_geometry_fp32(outputs: object) -> object`

- [ ] **Step 1: Write failing dtype resolution tests**

Test that `None` and `"float16"` resolve to `torch.float16`, `"bfloat16"` resolves to `torch.bfloat16`, and an unknown name raises `ValueError` containing the invalid value.

- [ ] **Step 2: Run dtype tests and verify failure**

Run: `pytest test/test_precision_policy.py -q`

Expected: import failure because `engine.solver.precision` does not exist.

- [ ] **Step 3: Implement dtype resolution and support validation**

Implement exact string mapping. For CUDA BF16, call `torch.cuda.is_bf16_supported()` and raise `RuntimeError` including the CUDA device name when false. Do not reject CPU because this feature does not expand existing CPU AMP behavior.

- [ ] **Step 4: Write failing nested geometry conversion test**

Build nested main/aux/encoder/DN/pre outputs containing FP16 geometry tensors, FP16 logits, dictionaries, lists, and tuples. Assert geometry is FP32, logits remain FP16, source objects are not mutated, and summing converted geometry can backpropagate to the source tensor.

- [ ] **Step 5: Implement recursive geometry conversion**

Recursively rebuild mappings, lists, and tuples. Convert tensors only when their dictionary key is one of `pred_boxes`, `pred_corners`, or `ref_points`. Return scalar metadata and unrelated tensors unchanged.

- [ ] **Step 6: Run helper tests**

Run: `pytest test/test_precision_policy.py -q`

Expected: all tests pass.

### Task 2: Training Loop Precision Wiring

**Files:**
- Modify: `engine/solver/det_engine.py:171-320`
- Modify: `engine/solver/det_solver.py:147-169`
- Test: `test/test_precision_policy.py`

**Interfaces:**
- Consumes: `resolve_amp_dtype`, `validate_amp_dtype_support`, `cast_obb_geometry_fp32`
- Adds kwargs to `train_one_epoch`: `amp_dtype_name: str | None`, `obb_geometry_fp32: bool`

- [ ] **Step 1: Write failing autocast policy tests**

Use monkeypatching to verify the training AMP branch passes the resolved dtype to `torch.autocast`. Add a test that OBB geometry conversion runs before criterion evaluation only when `obb_geometry_fp32=True` and `criterion.box_mode == "obb"`.

- [ ] **Step 2: Run targeted tests and verify failure**

Run: `pytest test/test_precision_policy.py -q`

- [ ] **Step 3: Wire precision policy into `train_one_epoch`**

Resolve and validate dtype once before iteration. Pass `dtype=amp_dtype` to model autocast. In the existing autocast-disabled criterion block, use converted outputs only for criterion input; retain original outputs for model diagnostics. Keep scaler, overflow, EMA, scheduler, and logging control flow unchanged.

- [ ] **Step 4: Pass YAML values from `DetSolver.fit`**

Pass `args.yaml_cfg.get("amp_dtype")` and `bool(args.yaml_cfg.get("obb_geometry_fp32", False))` into `train_one_epoch`.

- [ ] **Step 5: Run AMP and precision tests**

Run: `pytest test/test_precision_policy.py test/test_deim_criterion_obb_loss.py test/test_obb_loss_integration.py -q`

Expected: all tests pass.

### Task 3: Experiment Configurations

**Files:**
- Create: `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16.yml`
- Create: `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_fp16_obb_fp32.yml`
- Modify: `test/test_obb_loss_experiment_configs.py`

**Interfaces:**
- Group B inherits `sp_fz_rep0_nloss_amp.yml`, sets `amp_dtype: bfloat16`, `obb_geometry_fp32: false`, and `scaler.enabled: false`.
- Group C inherits the same base, sets `amp_dtype: float16`, `obb_geometry_fp32: true`, and `scaler.enabled: true`.

- [ ] **Step 1: Write failing config resolution tests**

Load both YAML files with `YAMLConfig`. Assert the precision fields and scaler enabled flag. Assert `epoches`, `flat_epoch`, `no_aug_epoch`, train dataset paths, optimizer LR, and loss weights equal the base experiment.

- [ ] **Step 2: Run config tests and verify failure**

Run: `pytest test/test_obb_loss_experiment_configs.py -q`

Expected: missing configuration files.

- [ ] **Step 3: Add minimal inherited YAML files**

Each file includes only the base experiment plus output directory and precision overrides. Do not duplicate dataset or optimizer configuration.

- [ ] **Step 4: Run config tests**

Run: `pytest test/test_obb_loss_experiment_configs.py -q`

Expected: all tests pass.

### Task 4: Regression Verification

**Files:**
- Verify only; no planned production edits.

**Interfaces:**
- Consumes all prior tasks.

- [ ] **Step 1: Run focused regression suite**

Run: `pytest test/test_precision_policy.py test/test_deim_criterion_obb_loss.py test/test_obb_loss_integration.py test/test_obb_loss_experiment_configs.py test/test_yolo_obb_loss.py -q`

- [ ] **Step 2: Run source diagnostics**

Run LSP diagnostics for `engine/solver/precision.py`, `engine/solver/det_engine.py`, and `engine/solver/det_solver.py`.

- [ ] **Step 3: Run configuration smoke checks**

Instantiate `YAMLConfig` for both experiment files and print resolved `amp_dtype`, `obb_geometry_fp32`, and scaler enabled state. On CUDA hardware, print `torch.cuda.is_bf16_supported()`; do not start a full training run.

- [ ] **Step 4: Review scope**

Confirm default YAML behavior remains FP16, no loss formula was changed, and the diff contains only precision policy, wiring, tests, experiment configs, and documentation.
