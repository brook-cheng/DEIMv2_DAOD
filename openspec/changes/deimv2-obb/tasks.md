# Tasks: DEIMv2-OBB

## Phase 1 — Core Geometry

### Step 1.1: `obb_geometry.py`
- [ ] Create `engine/deim/obb_geometry.py`
- [ ] Implement `oriented_box_to_external_rect(boxes)` — OBB → axis-aligned external rectangle
- [ ] Implement `external_rect_to_oriented_box(rect, epsilons, etas)` — external rect + vertex offsets → OBB
- [ ] Unit tests (pytest) for both directions
- **Verify**: All tests pass, HBB config unchanged

## Phase 2 — Cost & Loss Foundation

### Step 2.1: `obb_ops.py`
- [ ] Create `engine/deim/obb_ops.py`
- [ ] Implement `probiou(boxes1, boxes2)` — Gaussian ProbIoU
- [ ] Implement `kld_loss(pred, target)` — Kullback-Leibler Divergence loss for OBB
- [ ] Implement `rbbox_overlaps_obb(boxes1, boxes2, mode='probiou')`
- [ ] Unit tests
- **Verify**: Tests pass, no change to HBB

### Step 2.2: `chamfer_cost.py`
- [ ] Create `engine/deim/chamfer_cost.py`
- [ ] Implement `chamfer_distance(boxes1, boxes2)` — vertex-set Chamfer distance
- [ ] Implement `obb_to_vertices(boxes)` — 5-dof → 4 corner points
- [ ] Unit tests
- **Verify**: Tests pass

### Step 2.3: `matcher.py`
- [ ] Add `box_mode` parameter to matcher
- [ ] Add `cost_chamfer` and `cost_kld` as OBB cost types
- [ ] Gate OBB costs behind `box_mode == 'obb'`
- **Verify**: HBB matching unchanged; OBB matching returns valid indices

## Phase 3 — Decoder Adaptation

### Step 3.1: `deim_decoder.py`
- [ ] Add `box_mode='hbb'` parameter to `DEIMTransformer.__init__`
- [ ] When `box_mode='obb'`: `_num_box_dof=5`, `num_reg_dist=6`
- [ ] When `box_mode='hbb'`: preserve ALL original behavior
- [ ] Add `query_pos_head` 5-dof branch for OBB
- [ ] Add 6-distribution DDF output path for OBB
- [ ] Add `angle_factor` for OBB cross-attention
- **Verify**: HBB forward pass unchanged; OBB forward produces `(bs,nq,5)` output

### Step 3.2: `dfine_utils.py`
- [ ] Add `bbox2distance_obb(points, gt_bbox, reg_max)` — OBB FGL targets
- [ ] Use `oriented_box_to_external_rect` internally
- **Verify**: HBB `bbox2distance` unchanged; OBB version returns 6×N distributions

### Step 3.3: `dfine_decoder.py`
- [ ] Add `elif reference_points.shape[-1] == 5:` branch in sampling locations
- [ ] Implement rotation matrix: `rot @ (w/2, h/2)` for rotated sampling
- **Verify**: HBB (4-dof) sampling unchanged; OBB (5-dof) sampling rotates sampling points

## Phase 4 — Loss Function

### Step 4.1: `deim_criterion.py`
- [ ] Add `box_mode='hbb'` parameter
- [ ] Extract existing HBB logic into `_loss_boxes_hbb()`, `_loss_local_hbb()`
- [ ] Add `_loss_boxes_obb()` — KLD loss
- [ ] Add `_loss_local_obb()` — FGL via `bbox2distance_obb()`
- [ ] Gate at top-level `loss_boxes()`, `loss_local()` by `box_mode`
- **Verify**: HBB loss values identical; OBB loss returns valid tensors

## Phase 5 — Training Infrastructure

### Step 5.1: `denoising.py`
- [ ] Add `noise_mode` parameter with 4 modes: `'only_xyxy'`, `'only_angle'`, `'only_xywh'`, `'all_xyxya'`
- [ ] Add angle noise injection for OBB
- **Verify**: HBB denoising unchanged; OBB denoising generates 5-dof queries

### Step 5.2: `postprocessor.py`
- [ ] Add `box_mode='hbb'` parameter
- [ ] OBB mode: decode 5-dof boxes (cx,cy,w,h,θ)
- [ ] OBB mode: apply rotated NMS if configured
- **Verify**: HBB postprocessing unchanged; OBB outputs 5-dof results

## Phase 6 — Data & Evaluation

### Step 6.1: `obb_dataset.py` + Config
- [ ] Create DOTA/DIOR-R dataset class in `engine/deim/`
- [ ] Create config: `configs/deimv2_obb_dinov3_s_dota.yml`
- **Verify**: Dataset loads correctly, config parses without error

### Step 6.2: `det_engine.py` — OBBEvaluator
- [ ] Add OBB evaluation (AP50 with rotated IoU)
- [ ] Gate behind `box_mode`
- **Verify**: HBB evaluation unchanged

### Step 6.3: `det_solver.py`
- [ ] Add OBB evaluation dispatch
- **Verify**: Training loop calls OBB evaluator when `box_mode='obb'`

### Step 6.4: `deimv2_det.py`
- [ ] Add OBB inference path
- [ ] Handle 5-dof output in prediction
- **Verify**: Inference produces OBB results

## Phase 7 — Integration

### Step 7.1: HBB Compatibility Test
- [ ] Run HBB config end-to-end: `python train.py --config configs/deimv2_hbb_test.yml --test-only`
- [ ] Verify all loss values match original DEIMv2
- **Verify**: Zero regression

### Step 7.2: OBB Training Smoke Test
- [ ] Run OBB config: `python train.py --config configs/deimv2_obb_test.yml --check-forward`
- [ ] Verify forward pass completes without error
- **Verify**: OBB model trains for 1 epoch without NaN
