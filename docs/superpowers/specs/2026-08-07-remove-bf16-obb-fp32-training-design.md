# Remove BF16 and FP16/OBB-FP32 Training Modes

## Goal

Remove the rejected BF16 and FP16-forward/FP32-OBB-geometry training modes
from DEIMv2 DAOD. The remaining mixed-precision training path is ordinary
CUDA FP16 autocast with the existing enabled GradScaler.

This cleanup removes the runtime parameters, experiment configurations,
mode-specific helpers, tests, and active documentation references. It does
not change FP16 optimizer, scheduler, loss, EMA, checkpoint, diagnostic, or
early-stopping behavior.

## Scope

### Remove

- The `amp_dtype` training parameter and BF16 dtype selection/support checks.
- The `obb_geometry_fp32` training parameter and recursive geometry casting.
- `engine/solver/precision.py` after its mode-specific responsibilities have
  no remaining callers.
- The BF16, BF16 early-stopping, and FP16/OBB-FP32 experiment YAML files.
- Tests whose only contract is support for the removed modes.
- Active index entries and early-stopping documentation that present BF16 or
  FP16/OBB-FP32 as supported experiment choices.

### Preserve

- `sp_fz_rep0_nloss_amp.yml` and `sp_fz_rep0_nloss_amp_es.yml`.
- The existing `use_amp` switch, GradScaler construction/checkpointing, and
  the complete AMP training branch in `train_one_epoch`.
- Model forward under CUDA autocast with `dtype=torch.float16`.
- Criterion evaluation in the existing autocast-disabled block, using the
  model outputs directly.
- Non-finite loss/gradient diagnostics, gradient clipping, scaler step/update,
  EMA updates, and scheduler behavior.
- DINOv3 backbone BF16/FP8 implementation details. These are independent
  pretrained-backbone internals, not the rejected training modes.
- Historical git commits and generic FP16/TensorRT documentation.

## Runtime Design

`DetSolver.fit()` will stop reading `amp_dtype` and `obb_geometry_fp32` from
the resolved YAML configuration. It will call `train_one_epoch` without those
kwargs.

`train_one_epoch` will no longer resolve a configurable AMP dtype or validate
BF16 hardware support. In the scaler-enabled branch it will enter model
autocast with `dtype=torch.float16` directly.

The criterion remains outside model autocast exactly as today. The criterion
will always receive the original model outputs; the optional recursive cast of
OBB geometry tensors to FP32 is removed.

No compatibility shim will accept the deleted keys. Existing rejected-mode
YAML files are deleted, so supported repository configurations cannot request
them. A user-supplied stale override becomes an unused configuration key rather
than a hidden precision-mode switch; active documentation will no longer
advertise such overrides.

## Configuration and Documentation Cleanup

Delete:

- `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16.yml`
- `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16_es.yml`
- `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_fp16_obb_fp32.yml`
- `docs/superpowers/specs/2026-08-06-amp-bf16-obb-fp32-experiments-design.md`
- `docs/superpowers/plans/2026-08-06-amp-bf16-obb-fp32-experiments.md`

Update the Superpowers index and the 2026-08-07 early-stopping design/plan so
their active configuration lists and embedded test examples describe only the
FP16 base and FP16 early-stopping configuration.

Historical changelog entries and generic AMP references remain unchanged.

## Test Strategy

The cleanup is behavior-removing, so tests first define absence and preserved
FP16 behavior.

1. Add or adjust configuration tests so the retained FP16 and FP16-ES configs
   resolve without `amp_dtype` or `obb_geometry_fp32`, preserve enabled scaling,
   and no deleted config filename remains in active test matrices.
2. Retain a focused training-engine regression test proving the scaler branch
   enters autocast with `torch.float16` when no precision-mode kwargs exist.
3. Remove BF16 support tests, BF16 hardware-validation tests, recursive OBB
   geometry-cast tests, and the deleted experiment-config parametrization.
4. Update early-stopping config tests to cover only the FP16 base and FP16-ES
   config while retaining schedule, optimizer, loss, and output-dir assertions.
5. Run the focused precision/early-stopping/config suites, adjacent diagnostic
   and OBB configuration suites, syntax/import checks, and a real YAMLConfig
   parse of `sp_fz_rep0_nloss_amp_es.yml`.

## Success Criteria

- Repository search finds no runtime/config/test use of `amp_dtype`,
  `obb_geometry_fp32`, the deleted config filenames, or BF16 training support.
- `engine/solver/precision.py` has no remaining responsibility and is deleted.
- Ordinary FP16 AMP still uses `torch.float16` autocast and an enabled
  GradScaler.
- The retained FP16 and FP16-ES configs preserve their existing schedule,
  optimizer, loss, EMA, checkpoint, and early-stopping settings.
- Focused and neighboring regression tests pass; unrelated DINOv3 BF16/FP8
  symbols remain untouched.

## Non-Goals

- Removing BF16 or FP8 implementation details from the DINOv3 backbone.
- Disabling AMP globally or changing `use_amp` behavior.
- Reworking GradScaler registration/config construction.
- Changing loss formulas, FP32 criterion execution, EMA behavior, checkpoint
  formats, or early-stopping logic.
- Rewriting git history or deleting historical commit metadata.
