# BF16 Forward and FP32 Loss Training - Final Implementation Record

> **Status:** Implemented and verified on 2026-08-08.
>
> **Historical note:** This document originally planned removal of BF16 training
> and convergence on FP16 autocast with GradScaler. That decision was completed
> on 2026-08-07, but subsequent FP16 training produced non-finite OBB losses.
> The active precision design was therefore reversed on 2026-08-08. The old
> FP16-only instructions are superseded and must not be executed.

## Final Goal

Use BF16 autocast for model forward while computing the complete criterion in
FP32. Do not use GradScaler in the epoch training loop.

This design targets the observed `FloatingPointError` failures caused by NaN or
Inf values in numerically sensitive OBB loss paths such as KLD, probabilistic
IoU, and angle geometry calculations.

## Final Precision Contract

When `use_amp=True`:

1. Run model forward under:

   ```python
   torch.autocast(
       device_type=str(device),
       dtype=torch.bfloat16,
       cache_enabled=True,
   )
   ```

2. Convert every floating-point tensor in the model output pytree to
   `torch.float32` with `torch.utils._pytree.tree_map`.
3. Preserve dict/list/tuple structure, integer tensors, booleans, strings, and
   other metadata unchanged.
4. Run the criterion inside `torch.autocast(..., enabled=False)`.
5. Validate individual and total losses before backward.
6. Call `loss.backward()` and `optimizer.step()` directly.
7. Do not call `GradScaler.scale`, `unscale_`, `step`, or `update`.

When `use_amp=False`, preserve the existing FP32/Kendall training behavior.

## Implemented Changes

### `engine/solver/det_engine.py`

- Replaced the epoch-loop `scaler` switch with `use_amp`.
- Changed model autocast from `torch.float16` to `torch.bfloat16`.
- Added a `tree_map` transformation after model forward:

  ```python
  outputs = tree_map(
      lambda t: t.float()
      if isinstance(t, torch.Tensor) and t.is_floating_point()
      else t,
      outputs,
  )
  ```

- Kept the criterion in an autocast-disabled block.
- Replaced scaled backward/step with direct backward/step.
- Removed the AMP gradient-overflow skip branch and its counter.
- Removed `_has_nonfinite_grads` and GradScaler imports from this module.
- Preserved non-finite loss checks, gradient inspection, NaN snapshots,
  gradient clipping, EMA updates, scheduler behavior, optimizer-step caps, and
  metric logging.

### `engine/solver/det_solver.py`

- Replaced:

  ```python
  scaler=self.scaler
  ```

  with:

  ```python
  use_amp=args.yaml_cfg.get("use_amp", False)
  ```

- Kept `self.scaler` construction and checkpoint state compatibility outside
  the epoch loop. Existing configurations may still contain a `scaler` block,
  but it no longer selects or drives the training precision path.

### Tests

`test/test_precision_policy.py` now verifies:

- `use_amp=True` selects `torch.bfloat16` model autocast.
- `use_amp=False` does not enter dtype-selecting autocast.
- A legacy `scaler=` kwarg does not enable AMP.
- Nested floating-point model outputs reach the criterion as FP32.
- Integer tensors and container structure remain unchanged.

`test/test_det_engine_diagnostics.py` now exercises AMP through
`use_amp=True` rather than a GradScaler argument.

## Preserved Historical Cleanup

The following 2026-08-07 cleanup decisions remain valid:

- `amp_dtype` and `obb_geometry_fp32` are not restored as user-selectable
  runtime parameters.
- `engine/solver/precision.py` remains deleted.
- The rejected per-experiment BF16 and FP16/OBB-geometry-FP32 YAML files remain
  deleted.
- DINOv3 backbone BF16/FP8 internals remain untouched; they are independent of
  the detector training precision policy.

The reversal introduces one supported AMP behavior through the existing
`use_amp` switch. It does not restore the deleted multi-mode precision matrix.

## Configuration Behavior

No YAML changes were required.

- `use_amp: true` now means BF16 model forward plus FP32 criterion input.
- `use_amp: false` means the existing FP32 path.
- Existing `scaler:` configuration is retained for checkpoint compatibility.

The real training composition
`configs/custom_obb/dlzdt/sp_fz_common.yml` includes `runtime.yml`, resolves
`use_amp: true`, and can still construct `GradScaler`; the object is not passed
to `train_one_epoch`.

## Verification Evidence

TDD RED failures were observed before implementation:

- BF16 autocast test reported that no dtype-selecting autocast was entered.
- Legacy `scaler=` still selected the old AMP branch.
- Criterion received raw BF16 outputs instead of FP32 outputs.

After implementation:

- `test_precision_policy.py` + `test_det_engine_diagnostics.py`: 44 passed.
- Precision + diagnostics + solver early-stopping suites: 60 passed.
- OBB experiment-config and criterion suites: 28 passed.
- `py_compile` passed for all four changed Python files.
- LSP diagnostics reported no issues in all four changed files.
- Real `sp_fz_common.yml` composition resolved `use_amp: true` and created the
  retained GradScaler object successfully.

## Known Pre-existing Test Failures

The implementation did not cause these failures:

- Five tests in `test/test_early_stopping_configs.py` reference deleted
  `sp_fz_rep0_nloss_amp*.yml` files.
- One test in `test/test_kendall.py` references deleted
  `configs/custom_obb/deimv2_obb_sp.yml`.

The referenced files were deleted by earlier config consolidation commit
`9dd6560`. The failures reproduced identically with the BF16 implementation
stashed.

## Final Success Criteria

- [x] `use_amp=True` uses BF16 model autocast.
- [x] All floating model outputs are FP32 before criterion execution.
- [x] AMP training uses direct backward and optimizer step.
- [x] GradScaler is not used by `train_one_epoch`.
- [x] The FP32/Kendall path remains available when `use_amp=False`.
- [x] Diagnostics, clipping, EMA, scheduler, and step-cap behavior are retained.
- [x] Focused and neighboring tests pass except for documented pre-existing
      missing-config failures.
- [x] No git commit was created.
