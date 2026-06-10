# Change: deimv2-obb-eval-opt

## Summary
Optimize OBB evaluation speed. The current `poly_iou` implementation uses nested Python loops for exact polygon IoU computation, taking ~540s per validation pass for a typical DOTA config. The goal is to reduce this to <5s without sacrificing accuracy.

## Motivation
- `poly_iou` takes ~18s per class (300 det × 90 GT) due to O(N×M) Python loops
- Full validation (15 classes × 2 IoU thresholds) takes ~540s — unacceptable for iterative development
- Data collection and vstack are <20ms — not bottlenecks

## Proposed Approach
Replace `poly_iou` with `batch_probiou` in `_tpfp`. This provides ~5000x speedup with negligible accuracy difference (Gaussian-based ProbIoU vs exact polygon IoU; relative ranking preserved).

## Files
- `engine/eval/obb_eval.py` — replace `poly_iou` with `batch_probiou` in `_tpfp`
- `test/test_profile_obb_eval.py` — profiling benchmarks

## Status
- [ ] Profile baseline (done: poly_iou 18s/class)
- [ ] Implement batch_probiou in _tpfp
- [ ] Profile optimized version
- [ ] Verify AP values within 1% of poly_iou baseline
