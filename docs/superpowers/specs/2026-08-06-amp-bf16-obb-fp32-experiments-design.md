# AMP BF16 and OBB FP32 Experiment Design

## Goal

Add two controlled mixed-precision experiment modes without changing the
existing FP16 default:

- Group B: BF16 model forward with loss scaling disabled.
- Group C: FP16 model forward with all OBB geometry paths evaluated in FP32.

The modes must share the existing optimizer, EMA, scheduler, gradient
inspection, and overflow handling paths so their results remain comparable.

## Configuration Contract

The training configuration accepts two new optional keys:

```yaml
amp_dtype: float16
obb_geometry_fp32: false
```

`amp_dtype` accepts only `float16` and `bfloat16`. Missing values preserve the
current CUDA autocast default (`float16`). `obb_geometry_fp32` defaults to
`false`.

Group B uses:

```yaml
use_amp: true
amp_dtype: bfloat16
obb_geometry_fp32: false
scaler:
  type: GradScaler
  enabled: false
```

Group C uses:

```yaml
use_amp: true
amp_dtype: float16
obb_geometry_fp32: true
scaler:
  type: GradScaler
  enabled: true
```

## Runtime Data Flow

`DetSolver.fit()` passes the resolved AMP dtype and OBB geometry flag to
`train_one_epoch()`. The training loop uses the dtype explicitly in the model
autocast context. Existing YAML files that omit the new fields continue to use
FP16.

For BF16, the configured `GradScaler` object remains present but is disabled.
This keeps the existing AMP branch intact: `scale`, `unscale_`, `step`, and
`update` become pass-through operations. Before training, CUDA BF16 support is
validated with `torch.cuda.is_bf16_supported()`; unsupported devices receive a
clear error.

For Group C, model forward remains FP16. Before the criterion is called, a
recursive, non-mutating conversion changes OBB geometry tensors to FP32 while
leaving classification tensors and metadata unchanged. Geometry keys are:

- `pred_boxes`
- `pred_corners`
- `ref_points`

The conversion traverses nested dictionaries, lists, and tuples, covering main,
auxiliary, encoder, denoising, and pre-decoder outputs. Tensor `.float()` casts
remain differentiable, so gradients flow back into the FP16 model forward.

This boundary makes ProbIoU, KLD, periodic angle calculations, MAL quality
targets, FGL geometry targets, and matcher geometry consume FP32 predictions.

## Error Handling

- Unknown `amp_dtype` values raise `ValueError` before the first batch.
- BF16 on unsupported CUDA hardware raises `RuntimeError` with the device name
  and a recommendation to use FP16.
- `obb_geometry_fp32` is ignored for HBB geometry to avoid changing unrelated
  behavior.
- CPU behavior is not expanded by this change; the existing non-CUDA behavior
  remains unchanged.

## Experiment Configurations

Two configurations inherit from the current FP16 experiment file:

- `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_bf16.yml`
- `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_fp16_obb_fp32.yml`

They override only output directory and precision settings. Dataset, optimizer,
loss weights, LR schedule, and augmentations remain identical.

## Tests

Unit tests must verify:

1. Missing `amp_dtype` resolves to FP16.
2. `float16` and `bfloat16` resolve to the correct `torch.dtype`.
3. Invalid precision names fail clearly.
4. Recursive OBB geometry conversion changes every nested geometry tensor to
   FP32 while preserving logits, metadata, structure, and gradients.
5. HBB outputs are not converted by the OBB-only switch.
6. Group B and Group C YAML files resolve the intended flags and preserve the
   base training schedule.
7. Existing AMP flow and OBB loss tests continue to pass.

## Non-Goals

- Changing loss formulas, epsilon values, clipping, optimizer behavior, EMA, or
  the LR schedule.
- Automatically selecting BF16 based on hardware.
- Treating BF16 as a fix for late validation degradation; it is an experiment
  used to isolate FP16 range limitations.
