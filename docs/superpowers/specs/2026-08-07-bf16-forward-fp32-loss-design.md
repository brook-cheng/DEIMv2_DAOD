# BF16 Forward and FP32 Loss Training Design

> **Status:** Active design as of 2026-08-08.
>
> This file originally specified removal of BF16 support in favor of FP16 plus
> GradScaler. That design is superseded because the resulting FP16 loss path
> produced non-finite values during OBB training.

## Goal

Provide one stable mixed-precision training path:

- BF16 autocast for model forward.
- FP32 model outputs at the criterion boundary.
- Direct backward and optimizer step without GradScaler.

The design prioritizes numerical stability in OBB geometry losses while
retaining BF16's wider exponent range and reduced model-forward memory cost.

## Scope

### Change

- Interpret `use_amp=True` as BF16 model autocast.
- Convert all floating leaves in nested model outputs to FP32 before criterion
  execution.
- Remove GradScaler operations and overflow-skip logic from
  `train_one_epoch`.
- Pass `use_amp` from `DetSolver.fit()` instead of passing the scaler object.
- Update precision and diagnostics tests to pin the new contract.

### Preserve

- `use_amp=False` FP32/Kendall behavior.
- Non-finite loss validation before backward.
- Gradient inspection, allowed NaN-gradient zeroing, and NaN snapshots.
- Gradient clipping, optimizer-step caps, EMA, schedulers, and logging.
- Existing scaler construction/checkpoint state for compatibility.
- Deleted `amp_dtype` and `obb_geometry_fp32` interfaces remain deleted.
- DINOv3 backbone BF16/FP8 internals remain untouched.

## Runtime Data Flow

For `use_amp=True`:

```text
samples/targets
    -> model forward under BF16 autocast
    -> nested model outputs
    -> tree_map: floating tensors -> FP32
    -> criterion with autocast disabled
    -> per-loss and total-loss finite checks
    -> direct backward
    -> gradient diagnostics and clipping
    -> optimizer step
    -> EMA/scheduler/logging
```

For `use_amp=False`, the existing FP32 path remains unchanged.

## Output Conversion

Use PyTorch's pytree utility instead of a custom recursive helper:

```python
from torch.utils._pytree import tree_map

outputs = tree_map(
    lambda t: t.float()
    if isinstance(t, torch.Tensor) and t.is_floating_point()
    else t,
    outputs,
)
```

This handles dicts, lists, tuples, and other registered pytree containers while
preserving non-floating leaves. Each tensor still requires a dtype conversion;
`tree_map` avoids maintaining a project-specific traversal implementation.

The cast remains connected to autograd, so FP32 loss gradients propagate back
through the BF16 forward graph.

## GradScaler Compatibility Boundary

GradScaler is no longer part of the epoch-loop execution contract. The
configuration and solver checkpoint state are retained so existing checkpoint
formats do not change in the same implementation cycle.

Therefore:

- `train_one_epoch` does not read a `scaler` kwarg.
- A legacy `scaler=` kwarg is ignored by `**kwargs` and does not enable AMP.
- `DetSolver.fit()` passes `use_amp` explicitly.
- Solver setup may still construct `self.scaler` from an existing YAML block.

## Error Handling

- A non-finite individual loss raises `FloatingPointError` before backward.
- A non-finite total loss raises `FloatingPointError` before backward.
- Non-finite gradients continue through the existing `inspect_gradients`
  policy; the deleted GradScaler overflow-skip path is not reproduced.
- Existing NaN model-output and gradient snapshot behavior remains available.

## Test Strategy

1. Record the dtype passed to model autocast and require `torch.bfloat16`.
2. Verify `use_amp=False` does not enter the dtype-selecting autocast path.
3. Verify a legacy scaler argument does not select AMP.
4. Feed nested BF16 outputs and assert that the criterion receives FP32
   floating tensors, unchanged integer tensors, and the same container shape.
5. Run a complete mocked BF16 epoch without GradScaler.
6. Verify NaN loss detection occurs before backward in the BF16 path.
7. Run solver early-stopping and OBB criterion/config regression suites.

## Success Criteria

- `use_amp=True` selects BF16 autocast independently of scaler presence.
- Criterion floating inputs are FP32 for every nested model output leaf.
- Direct backward and optimizer step complete without GradScaler.
- `use_amp=False` behavior does not regress.
- Focused tests, diagnostics tests, solver tests, compile checks, and LSP
  diagnostics pass.

## Non-goals

- Reintroducing selectable `amp_dtype` modes.
- Reintroducing OBB-only geometry casting.
- Removing scaler checkpoint compatibility in this cycle.
- Changing loss formulas or OBB target representations.
- Changing DINOv3 internal precision implementation.
