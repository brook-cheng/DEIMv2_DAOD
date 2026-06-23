# Proposal: DEIMv2-OBB

## Summary
Add oriented bounding box (OBB) support to DEIMv2, enabling detection of arbitrarily rotated objects in remote sensing imagery. The change uses Angle Distribution Refinement (ADR) — 6-distribution DDF — and is gated behind `box_mode='obb'` with `box_mode='hbb'` as the default.

## Motivation
- DEIMv2 is state-of-the-art for real-time HBB detection
- Remote sensing objects (ships, aircraft, vehicles) appear at arbitrary angles
- HBB bounding boxes include excessive background for rotated objects
- OBB (5-dof: cx,cy,w,h,θ) provides tighter, rotation-aware localization

## Scope
- **In**: DDF decoder, criterion, matcher, denoising, postprocessor, dataset, evaluation, config
- **Out**: Distillation, NMS-free postprocessing (use rotated NMS initially), export/ONNX

## Key Design Decisions

1. **`box_mode` gate over separate classes**: Single codebase, `box_mode='hbb'` default. OBB activated via config `box_mode: obb`.
2. **ADR 6-distribution approach**: 4 distributions for external rectangle edges + 2 for vertex offsets (ε, η), avoiding angular periodicity issues.
3. **Shape-driven internal dispatch**: `if ref.shape[-1] == 5` for rotated cross-attention, not explicit `box_mode` flag.
4. **ProbIoU + KLD + Chamfer**: ProbIoU as primary overlap metric, KLD as loss, Chamfer for matching.
5. **Backward compatibility**: When `box_mode='hbb'`, all tensor shapes, loss values, and model weights are identical to original DEIMv2.

## Non-goals
- This change does NOT modify existing HBB code paths
- This change does NOT add ADR-specific FGL loss initially (Phase 2)

## Multi-Task Loss Weighting

DEIMv2-OBB has 4 concurrent losses (mal, bbox, kld, fgl) with different magnitude scales.
Kendall Uncertainty Weighting (Kendall et al., CVPR 2018) is used for automatic balancing:
learnable σ_i² parameters are optimized by the model's own gradient descent,
eliminating manual `weight_dict` tuning.

GradNorm (Chen et al., ICML 2018) was evaluated and ruled out — it requires a shared
parameter bottleneck that DEIM's architecture does not provide (classification and regression
paths diverge at the decoder). See `design.md` for details.
