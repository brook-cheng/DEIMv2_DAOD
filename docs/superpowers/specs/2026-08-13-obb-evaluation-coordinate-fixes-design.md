# OBB Evaluation Coordinate Fixes Design

## Goal

Repair the OBB evaluation pipeline in two isolated phases. Phase 1 fixes the
confirmed height/width ordering error in the standalone offline inference
entry point. Phase 2 will separately unify online and offline evaluation
coordinates and remove the lossy non-uniform OBB reconstruction path.

## Confirmed Baseline

- `imgsz` in `test/tool_deimv2_obb_infer.py` is an `(H, W)` tuple.
- `PostProcessor.forward()` requires `orig_target_sizes` in `[W, H]` order.
- The standalone offline inference tool currently passes `[H, W]`.
- The ordering error is invisible for square input and corrupts rectangular
  input. At `576x1024`, x/width receive a factor of `0.5625` and y/height a
  factor of `1.7778` relative to the correct result.
- Online evaluation and `deim_app` already honor the `[W, H]` PostProcessor
  contract.
- Offline original-space evaluation additionally suffers lossy OBB refitting,
  but that is independent of the ordering bug and is outside Phase 1.

## Phase 1: H/W Ordering Repair

### Scope

Modify only the standalone offline inference boundary in
`test/tool_deimv2_obb_infer.py` and its regression coverage. Do not change
`PostProcessor`, online evaluation, `rescale_obb_to_original()`, or OBB geometry.

### Data Contract

1. Image preprocessing receives `imgsz=(height, width)`.
2. The transformed tensor shape is `(N, C, height, width)`.
3. Immediately before calling the model/PostProcessor wrapper, convert the
   tuple to a tensor containing `[width, height]`.
4. Original-space recovery continues to receive
   `inference_size=(height, width)`.

For `imgsz=(576, 1024)`, the PostProcessor input must therefore be
`[[1024, 576]]`. For `imgsz=(640, 640)`, behavior remains `[[640, 640]]`.

### Implementation Choice

Use a direct boundary conversion at the existing call site. This is preferred
over introducing a new size value object because Phase 1 has one confirmed
faulty boundary and requires the smallest isolated change. A typed geometry
contract may be introduced in Phase 2, where several coordinate interfaces
will change together.

### Test Contract

Tests must precede the production edit and cover these scenarios:

1. **Rectangular happy path:** invoking the offline inference boundary with
   `(576, 1024)` passes `[1024, 576]` to the model/PostProcessor.
2. **Square regression:** invoking it with `(640, 640)` still passes
   `[640, 640]`.
3. **PostProcessor observable:** a normalized OBB under a `1024x576` canvas is
   scaled with `[W, H, W, H, 1]`, not the transposed factors.

The first test must fail against the current code for the expected axis-order
reason before the implementation changes.

### Manual QA

Run the real offline inference surface on a small, isolated image subset with
rectangular `imgsz=(576, 1024)`. Capture the produced coordinates and verify
that the former `0.5625` x/width and `1.7778` y/height distortion signature is
absent. The QA run must use a temporary output directory and remove it after
verification.

### Non-Goals

- Do not attempt to make online and offline mAP equal in Phase 1.
- Do not alter OBB angle, side canonicalization, or affine refitting.
- Do not repair the separate empty-GT evaluator defect.
- Do not change training or validation resize configuration.

## Phase 2: Coordinate-Space and Geometry Unification

Phase 2 starts only after Phase 1 is verified. Its design objective is to make
online and offline evaluation consume predictions and GT represented in the
same coordinate space without asymmetric lossy rectangle refits.

The Phase 2 investigation and design must compare at least these approaches:

1. Evaluate both paths in the resized canvas space and retain original-space
   metrics only as a deployment diagnostic.
2. Preserve quadrilateral or affine geometry through original-space recovery,
   delaying the five-parameter rectangle fit until a single shared boundary.
3. Use aspect-ratio-preserving resize/letterbox so inverse recovery is a
   similarity transform and rotated rectangles remain rotated rectangles.

Phase 2 must define one shared geometry metadata contract, run the same
prediction batch through online and offline metric adapters, and require
numerical metric parity before replacing the existing behavior.

## Success Criteria

Phase 1 is complete when all three automated scenarios pass, rectangular
manual QA shows correct `[W, H]` scaling, square behavior is unchanged, and no
coordinate-space or OBB-refit production code has changed.

Phase 2 is complete only when the chosen geometry design removes the measured
online/offline discrepancy using the same predictions and image set while
retaining explicit canvas-space and original-space semantics.
