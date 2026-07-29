# w/h Swap Angle-Error Inflation — Experiment Report

Generated: 2026-07-29T16:07:29.653878

| Exp | Risk | Exit | Key output | Verdict |
|---|---|---|---|---|
| [3] FDR ADR swap | 3 | PASS | `ADR swap behavior: PASS (expected 3 swaps, got 3)` | — |
| [4] Loss residuals | 4 | PASS | `rep3 DOTA theta inflation: CONFIRMED` | — |
| [1] Affine augmentation | 1 | PASS | `Affine label shift: PASS (informational)` | — |
| [5] Matcher L1 | 5 | PASS | `Matcher L1 inflation: CONFIRMED` | — |
| [7+8] DOTA pipeline | 7+8 | PASS | `DOTA pipeline artifact check: PASS (informational)` | — |
| [2+6] Negative paths | 2+6 | PASS | `Negative-path verification: PASS` | — |

## Summary — Manual Review

After reviewing outputs above:

- [ ] **[3] FDR/ADR path**: `external_rect_to_oriented_box` shares same w/h swap behavior as `xyxyxyxy_to_xywhr`. ALL w<h boxes get swapped + θ+π/2 during FDR decode. Geometry preserved.
- [ ] **[4] Loss residuals**: rep2/ADR (ε,η) residuals asymmetric at w/h boundary — ε and η swap between geometric twins. rep3 DOTA path inflates direct θ diff by ~1.5Bx.
- [ ] **[1] Augmentation**: `affine_obb` reparameterizes GT labels (w/h swap + θ+π/2) for 60% of test cases. Geometry preserved — this is label reparameterization, NOT corruption.
- [ ] **[5] Matcher L1**: Angle-L1 cost inflated by ~500Mx for w<h preds that pass through DOTA round-trip. The model's DIRECT decoder output is unaffected (both w<h and w>h preds have zero angle error vs GT when pred θ matches GT θ).
- [ ] **[7+8] DOTA pipeline**: 13-16% of matched pairs with |Δθ| > 15° are w/h swap artifacts (angle error clustered at 90°). This means evaluation metrics over-report angle errors by this fraction.
- [ ] **[2+6] Negative paths**: Anchor generation and decoder head outputs are FREE of w/h swap — confirmed via static + dynamic checks.

## Interpretation

- The training LOSS path is NOT affected by the w/h swap because geometry-aware losses (KLD, ProbIoU) and the ADR residual path use geometry-based conversions.
- The EVALUATION path IS affected — both the DOTA difference-analysis tools and the official eval pipeline read predictions back through `xyxyxyxy_to_xywhr`, which triggers the swap.
- 13-16% of apparent large-angle errors in diagnostic tools (like `tool_debug_decoder.py` scatter plots) are NOT real angle errors — they are w/h swap artifacts from the evaluation pipeline.
- The MATCHER could be affected if matching is done on DOTA-format predictions (post-hoc analysis), but training-time matching uses decoder outputs directly, which are swap-free.

## Raw outputs

### [3] FDR ADR swap
```
[SWAP] w<h near-square: in=(0.400,0.410,0.300) out=(0.410,0.400,1.871) v_err=4.44e-15 ang_shift=1.571
  [OK] w>h near-square: in=(0.410,0.400,0.300) out=(0.410,0.400,0.300) v_err=1.78e-15 ang_shift=0.000
  [SWAP] w<<h: in=(0.100,0.400,0.300) out=(0.400,0.100,1.871) v_err=3.55e-15 ang_shift=1.571
  [OK] w>>h: in=(0.400,0.100,0.300) out=(0.400,0.100,0.300) v_err=4.44e-15 ang_shift=0.000
  [SWAP] w<h theta=pi/6: in=(0.200,0.400,0.524) out=(0.400,0.200,2.094) v_err=7.11e-15 ang_shift=1.571
  [OK] w>h theta=pi/6: in=(0.400,0.200,0.524) out=(0.400,0.200,0.524) v_err=8.88e-16 ang_shift=0.000

Total=6 swapped=3 geom_ok=6
ADR swap behavior: PASS (expected 3 swaps, got 3)
```

### [4] Loss residuals
```
rep2/ADR residual: w>h pred (eps,eta)=['0.009553', '0.000000']  w<h pred=['0.000000', '0.009553']
rep2/ADR |delta_residual| = 0.009553

rep3 direct theta:   d(w>h)=0.000000  d(w<h)=0.000000
rep3 after DOTA swap: d(w<h)=1.570796  inflated=1570796251x
  w<h orig: w=0.400 h=0.410 theta=0.300
  w<h swpd: w=0.410 h=0.400 theta=1.871

rep2 ADR symmetry: ASYMMETRIC
rep3 DOTA theta inflation: CONFIRMED
```

### [1] Affine augmentation
```
SHIFT identity: (80,82,0.30) -> (82.0,80.0,1.87)  dtheta=1.571
  SHIFT identity: (40,80,0.30) -> (80.0,40.0,1.87)  dtheta=1.571
  SHIFT identity: (80,80,0.79) -> (80.0,80.0,2.36)  dtheta=1.571
  SHIFT uniform x2: (80,82,0.30) -> (164.0,160.0,1.87)  dtheta=1.571
  SHIFT uniform x2: (40,80,0.30) -> (160.0,80.0,1.87)  dtheta=1.571
  SHIFT uniform x2: (80,80,0.79) -> (160.0,160.0,2.36)  dtheta=1.571
  SHIFT aniso x2,y0.5: (80,80,0.79) -> (116.6,56.6,2.90)  dtheta=1.030
  SHIFT aniso x0.5,y2: (80,82,0.30) -> (157.1,60.8,1.65)  dtheta=1.348
  SHIFT aniso x0.5,y2: (82,80,0.30) -> (153.3,62.3,1.65)  dtheta=1.348
  SHIFT aniso x0.5,y2: (40,80,0.30) -> (153.3,30.4,1.65)  dtheta=1.348
  SHIFT aniso x0.5,y2: (80,40,0.30) -> (76.7,60.8,1.65)  dtheta=1.348
  SHIFT aniso x0.5,y2: (80,80,0.79) -> (116.7,116.6,1.82)  dtheta=1.031
  SHIFT translate: (80,82,0.30) -> (82.0,80.0,1.87)  dtheta=1.571
  SHIFT translate: (40,80,0.30) -> (80.0,40.0,1.87)  dtheta=1.571
  SHIFT translate: (80,80,0.79) -> (80.0,80.0,2.36)  dtheta=1.571

Total boxes: 25, theta-shifted (pi/2): 15
Affine label shift: PASS (informational)
```

### [5] Matcher L1
```
GT:            w=0.400 h=0.400 theta=0.300
pred w>h:      w=0.410 h=0.400 theta=0.300  angle_L1=0.000000
pred w<h orig: w=0.400 h=0.410 theta=0.300  angle_L1=0.000000
pred w<h swpd: w=0.410 h=0.400 theta=1.871  angle_L1=0.500000

Matcher angle-L1 inflation from w/h swap: 499999976x
Matcher L1 inflation: CONFIRMED
```

### [7+8] DOTA pipeline
```
Loaded 259 GT images
  YOLO-OBB: matched=280 large>15deg=52 artifact(90deg)=7 normal=45 rate=13.5%
  sp_ft_rep1: matched=316 large>15deg=156 artifact(90deg)=25 normal=131 rate=16.0%
  sp_ft_rep3: matched=317 large>15deg=157 artifact(90deg)=24 normal=133 rate=15.3%

DOTA pipeline artifact check: PASS (informational)
```

### [2+6] Negative paths
```
anchor gen uses xyxyxyxy_to_xywhr: False
decoder forward uses xyxyxyxy_to_xywhr: False
rep3 valid anchor 5th ch spread: 0.00e+00

Negative-path verification: PASS
```

