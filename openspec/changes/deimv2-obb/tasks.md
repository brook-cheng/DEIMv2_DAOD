# Tasks: DEIMv2-OBB

> **Status**: All core phases implemented. Remaining work: bug fixes from code review, Phase 7 smoke tests.

## Phase 1 — Core Geometry ✅

### Step 1.1: `obb_geometry.py`
- [x] Create `engine/deim/obb_geometry.py`
- [x] Implement `oriented_box_to_external_rect(boxes)` — OBB → axis-aligned external rectangle
- [x] Implement `external_rect_to_oriented_box(rect, epsilons, etas)` — external rect + vertex offsets → OBB
- [x] Implement `xywhr_to_xyxyxyxy` / `xyxyxyxy_to_xywhr` — OBB ↔ 4-vertex conversion
- [x] Implement `affine_obb` / `affine_obb_matrix` — pixel-level and general affine transforms
- [x] Unit tests (pytest) for both directions
- **Verify**: All tests pass, HBB config unchanged

## Phase 2 — Cost & Loss Foundation ✅

### Step 2.1: `obb_ops.py`
- [x] Create `engine/deim/obb_ops.py`
- [x] Implement `probiou(boxes1, boxes2)` — Gaussian ProbIoU
- [x] Implement `batch_probiou(boxes1, boxes2)` — pairwise ProbIoU matrix
- [x] Implement `kld_loss(pred, target)` — Kullback-Leibler Divergence loss for OBB
- [x] Implement `rbbox_overlaps_obb(boxes1, boxes2, mode='probiou')`
- [x] Implement `xy_wh_r_2_xy_sigma` — OBB → 2D Gaussian
- [x] Unit tests
- **Verify**: Tests pass, no change to HBB

### Step 2.2: `chamfer_cost.py`
- [x] Create `engine/deim/chamfer_cost.py`
- [x] Implement `chamfer_cost_obb(boxes1, boxes2)` — vertex-set Chamfer distance
- [x] Unit tests
- **Verify**: Tests pass

### Step 2.3: `matcher.py`
- [x] Add `box_mode` parameter to matcher
- [x] Add `cost_chamfer` and `cost_kld` (cost_probiou) as OBB cost types
- [x] Gate OBB costs behind `box_mode == 'obb'`
- [x] Add angle_factor for unified L1 distance scaling
- **Verify**: HBB matching unchanged; OBB matching returns valid indices

## Phase 3 — Decoder Adaptation ✅

### Step 3.1: `deim_decoder.py`
- [x] Add `box_mode='hbb'` parameter to `DEIMTransformer.__init__`
- [x] When `box_mode='obb'`: `_num_box_dof=5`, `num_reg_dist=6`
- [x] When `box_mode='hbb'`: preserve ALL original behavior
- [x] Add `query_pos_head` 5-dof branch for OBB
- [x] Add 6-distribution DDF output path for OBB
- [x] Add `angle_factor` for OBB cross-attention
- [x] Add anchor generation for 5-dof
- [x] Add encoder topk bbox theta scaling
- [x] Add final theta scaling (decoder internal [0,1] → external [0,π])
- **Verify**: HBB forward pass unchanged; OBB forward produces `(bs,nq,5)` output

### Step 3.2: `dfine_utils.py`
- [x] Add `distance2bbox_obb(points, distance, reg_scale)` — OBB ADR decode
- [x] Add `bbox2distance_obb(points, gt_bbox, reg_max, reg_scale, up, eps)` — OBB FGL targets
- [x] Use `oriented_box_to_external_rect` / `external_rect_to_oriented_box` internally
- **Verify**: HBB `bbox2distance` unchanged; OBB version returns 6×N distributions

### Step 3.3: `dfine_decoder.py`
- [x] Add `elif reference_points.shape[-1] == 5:` branch in sampling locations
- [x] Implement rotation matrix: `R(θ) @ (dx·w/2, dy·h/2)` for rotated sampling
- **Verify**: HBB (4-dof) sampling unchanged; OBB (5-dof) sampling rotates sampling points
- **⚠ Known bug**: OBB_CODE_REVIEW.md #1 — rotation math error (elementwise scale instead of rotate-after-scale)

## Phase 4 — Loss Function ✅

### Step 4.1: `deim_criterion.py`
- [x] Add `box_mode='hbb'` parameter
- [x] Gate `loss_boxes` by `box_mode`: HBB uses L1+GIoU, OBB uses L1+KLD
- [x] Gate `loss_local` by `box_mode`: OBB uses `bbox2distance_obb` for FGL targets
- [x] Gate `loss_labels_vfl` and `loss_labels_mal` by `box_mode`: OBB uses `probiou`
- [x] Gate at top-level `loss_boxes()`, `loss_local()` by `box_mode`
- **Verify**: HBB loss values identical; OBB loss returns valid tensors
- **⚠ Known limitation**: `get_loss_meta_info` raises `NotImplementedError` for OBB (boxes_weight_format not ported)

## Phase 5 — Training Infrastructure ✅

### Step 5.1: `denoising.py`
- [x] Add `box_mode` parameter
- [x] Add angle noise injection for OBB (box noise only — no angle noise, per paper)
- [x] OBB: scale theta by 1/π before inverse_sigmoid, concat with spatial noise
- **Verify**: HBB denoising unchanged; OBB denoising generates 5-dof queries

### Step 5.2: `postprocessor.py`
- [x] Add `box_mode='hbb'` parameter
- [x] OBB mode: decode 5-dof boxes (cx,cy,w,h,θ) — preserve format, scale to pixels
- [x] OBB theta already in [0,π] from decoder output
- **Verify**: HBB postprocessing unchanged; OBB outputs 5-dof results

## Phase 6 — Data & Evaluation ✅

### Step 6.1: Dataset + Config
- [x] Create DotaDataset class in `engine/data/dataset/dota_dataset.py`
- [x] Support DOTA annotation format (8-corner polygon)
- [x] Support YOLO-OBB annotation format (normalized 8-corner polygon)
- [x] Handle class names with spaces in classes.txt
- [x] Create config hierarchy:
  - `configs/deimv2_obb/deimv2_obb.yml` — shared OBB override layer
  - `configs/deimv2_obb/deimv2_obb_dinov3_s_dota.yml` — concrete config
  - `configs/custom_obb/deimv2_obb_common.yml` — core OBB model/loss/eval config
  - `configs/custom_obb/deimv2_obb_sp.yml` — main training config with real data paths
  - `configs/custom_obb/deimv2_obb_sp_jyz.yml` — insulator test variant
- **Verify**: Dataset loads correctly, config parses without error
- **⚠**: `configs/deimv2_obb/deimv2_obb_dinov3_s_dota.yml` includes `../base/deimv2_obb.yml` which does not exist (should be `deimv2_obb.yml`). OBB_CODE_REVIEW.md #8.

### Step 6.2: `det_engine.py` — OBBEvaluator
- [x] Add OBB evaluation via `obb_evaluate()` in `evaluate()`
- [x] Gate behind `box_mode`
- **Verify**: HBB evaluation unchanged

### Step 6.3: `det_solver.py`
- [x] Add OBB evaluation dispatch via `box_mode` from postprocessor
- [x] Handle OBB result format difference (flat dict vs list)
- **Verify**: Training loop calls OBB evaluator when `box_mode='obb'`

### Step 6.4: Data Transforms
- [x] `OBBFlip` — horizontal flip (bbox + angle mirror)
- [x] `OBBZoomOut` — random padding (translation only)
- [x] `OBBResize` — scale to target size
- [x] `OBBIoUCrop` — IoU-guided random crop
- [x] `OBBSanitize` — filter tiny/zero boxes
- [x] `OBBConvertBoxes` — normalize to [0,1]
- [x] Mosaic OBB branch — custom affine + placement offsets
- [x] CopyBlend OBB support in collate
- **⚠**: OBBFlip image flip not implemented. OBB_CODE_REVIEW.md #3.

### Step 6.5: `deimv2_det.py` (Inference)
- [x] Add inference and benchmark interfaces (from git history)
- [x] Add prediction-to-COCO annotation conversion
- [x] Handle 5-dof output in prediction
- **Verify**: Inference produces OBB results

## Phase 7 — Integration ⚠ (Partially complete)

### Step 7.1: HBB Compatibility Test
- [ ] Run HBB config end-to-end: `python train.py --config configs/deimv2_hbb_test.yml --test-only`
- [ ] Verify all loss values match original DEIMv2
- **Verify**: Zero regression

### Step 7.2: OBB Training Smoke Test
- [ ] Run OBB config: `python train.py --config configs/deimv2_obb_test.yml --check-forward`
- [ ] Verify forward pass completes without error
- **Verify**: OBB model trains for 1 epoch without NaN

## Phase 8 — Bug Fixes (from OBB_CODE_REVIEW.md) ⚠ (Pending)

Priority-ordered fixes from the code review:

### Severe (should fix first)
- [ ] **#1** `dfine_decoder.py:167-184` — Fix rotation cross-attention math (scale-then-rotate, not elementwise)
- [ ] **#2** `mosaic.py:122-134,158-166` — Fix Mosaic OBB offset: `[dx,dy,0,0,0]` not `[dx,dy,dx,dy,0]`
- [ ] **#3** `obb_transforms.py:11-18` — Implement image flip in OBBFlip
- [ ] **#4** `deimv2_obb_sp.yml` — Unify train/val normalization (both should have or both skip ImageNet Normalize)
- [ ] **#5** `obb_eval.py:157-180` — Remove duplicate AP append (keep only sorted path)

### Major
- [ ] **#6** `obb_eval.py`, `dota_eval.py` — Decide: use exact polygon IoU (`poly_iou`) or ProbIoU (`batch_probiou`) for eval
- [ ] **#7** `obb_transforms.py` — Fix anisotropic resize/crop not updating θ (use vertex transform + refit)
- [ ] **#8** `deimv2_obb_dinov3_s_dota.yml:4` — Fix include path to `deimv2_obb.yml`

### Medium
- [ ] **#9** `obb_ops.py:184,198` — Fix KLD det clamp position (clamp full determinant, not `a01²`)
- [ ] **#10** `matcher.py:173` — Rename `cost_kld` → `cost_probiou` (or compute actual KL divergence)
- [ ] **#11** `dfine_utils.py:212 vs 253-255` — Unify ADR vertex offset forward/backward scaling reference

### Low
- [ ] **#12** `obb_transforms.py:62` — Fix OBBResize H/W index assignment
- [ ] **#13** `deim_criterion.py:702-703` — Implement `get_loss_meta_info` for OBB
- [ ] **#14** `obb_transforms.py:71-85` — Remove hardcoded `img_size=(640,640)` in OBBConvertBoxes

## Phase 9 — Additional Features (from git history) ✅

- [x] `gradnorm.py` — Multi-task learning automatic loss weighting (recent)
- [x] `deimv2_obb_sp_jyz.yml` — Insulator testing config
- [x] YOLO-OBB annotation format support in DotaDataset
- [x] Transforms unit tests
- [x] `origin_size` fix in transforms
- [x] Model comparison test scripts
- [x] Inference and benchmark interfaces
