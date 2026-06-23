# Spec: obb-detection

> **Status**: Implemented
> **Files**: `engine/deim/deim_decoder.py`, `engine/deim/deim_criterion.py`, `engine/deim/matcher.py`, `engine/deim/denoising.py`, `engine/deim/postprocessor.py`, `engine/deim/dfine_decoder.py`, `engine/deim/dfine_utils.py`

Integration of Oriented Bounding Box (OBB) detection into the DEIMv2 detection pipeline. All OBB paths are gated behind `box_mode='obb'` (config-level) or `reference_points.shape[-1] == 5` (internal tensor shape dispatch). HBB paths are preserved unchanged.

---

## Activation & Parameter Cascade

### Entry Point: `DEIMTransformer.__init__` (`deim_decoder.py:345-373`)
```python
self.box_mode = box_mode          # from config
if self.box_mode == "obb":
    self._num_box_dof = 5          # 5-dof OBB vs 4-dof HBB
    self.num_reg_dist = 6          # 6 distributions for ADR vs 4 for HBB
```

### Parameter Propagation
```
box_mode="obb"
  ├── DEIMTransformer._num_box_dof=5, num_reg_dist=6
  ├── TransformerDecoder: num_reg_dist=6, box_mode="obb"
  ├── enc_bbox_head: MLP(..., 5, 3)       # 5-dof output
  ├── query_pos_head: MLP(5, ...)          # 5-dof input
  ├── pre_bbox_head: MLP(..., 5, 3)        # 5-dof output
  ├── Integral: self.reg_max=32, self.num_reg_dist=6
  ├── dec_bbox_head: 6 * (32+1) = 198 ch  # 6-distribution DDF
  ├── PostProcessor(box_mode="obb")
  ├── DEIMCriterion(box_mode="obb")
  └── Matcher(box_mode="obb", angle_factor=π)
```

### Theta Range Convention
| Scope | Range | Where |
|-------|-------|-------|
| Decoder internal | `[0, 1]` | All decoder layers, anchors, cross-attention reference points |
| Decoder output boundary | `[0, π]` | `deim_decoder.py:838-847` — `out_bboxes[..., 4:] * π` |
| Downstream (criterion/matcher/postprocessor) | `[0, π]` | Expected input range for all external consumers |
| Matcher L1 cost | scaled by `1/π` | `matcher.py:173` — `factor = [1,1,1,1, 1/π]` for unified distance |

---

## Component: `deim_decoder.py` — DEIMTransformer

### Anchor Generation (`_generate_anchors`, lines 646-672)
- **HBB**: `(bs, n_anchors, 4)` — `(cx, cy, w, h)`
- **OBB**: `(bs, n_anchors, 5)` — appends `θ = 0.5 * ones(...)` as 5th channel (center of [0,1] range)
- **Validity mask**: Only checks `[..., :4]` bounds (theta excluded from bounds check)
- **OBB anchors** are passed through `inverse_sigmoid` after dividing theta by π

### Encoder Output (`_get_encoder_output`, implicit)
- Encoder produces 5-dof bboxes when `box_mode='obb'`
- Top-k selection for decoder input (`_get_decoder_input:714-718`): selects encoder top bboxes, scales theta from `[0,1]` to `[0,π]` for initial decoder reference points

### Decoder Forward — FDR per Layer (lines 265-282)
Each decoder layer:
1. `distance2bbox_obb(points, distance, reg_scale)` — decode 6-dist DDF → 5-dof OBB
2. Theta scaling: `decoded[..., 4:] *= torch.pi` (internal → external range)
3. Residual accumulation: `new = (prev_bbox + decoded * reg_scale)`
4. Theta normalization: `new[..., 4:] /= torch.pi` (back to internal range for next layer)

### Final Output Postprocessing (lines 838-847)
```python
out_bboxes[..., 4:] *= torch.pi      # [0,1] → [0,π]
```
Applied to: `out_bboxes`, `out_refs`, `pre_bboxes`, `enc_topk_bboxes`. Comment: "为了方便后续处理，criterion/matcher/postprocessor 中均需要theta量纲为[0，π]"

### Refinement vs Non-Refinement Paths
- `enc_bbox_head` and `pre_bbox_head` produce 5-dof outputs via `MLP(..., _num_box_dof, 3)`
- These pass through `inverse_sigmoid` with theta scaled by `1/π`

---

## Component: `dfine_decoder.py` — Rotation-Aware Cross-Attention

### MSDeformableAttention Sampling (lines 167-184)

Dispatch is a 3-way branch on `reference_points.shape[-1]`:

| Dims | Mode | Behavior |
|------|------|----------|
| 2 | HBB (cx,cy) | `ref_xy + offset / feat_map_size` |
| 4 | D-FINE (cx,cy,w,h) | `ref_xy + offset * scale * ref_wh * offset_scale` |
| **5** | **OBB (cx,cy,w,h,θ)** | `ref_xy + R(θ) · (offset * scale * offset_scale * ref_wh/2)` |

**OBB sampling steps**:
1. Compute rotation matrix `R(θ)` from `θ * π` (radians):
   ```
   R = [[cos(θ), -sin(θ)],
        [sin(θ),  cos(θ)]]
   ```
2. Scale learned offsets by half-width/height: `scaled = offset * scale * offset_scale * [w/2, h/2]`
3. Rotate: `rotated = einsum("bqij,bqhpj->bqhpi", R, scaled)`
4. Add to center: `sampling_locations = [cx, cy] + rotated`

This rotates the deformable attention sampling pattern to align with the oriented box axes.

### Additional OBB Parameters in MSDeformableAttention
- `angle_factor` field (stored but not used directly in sampling — θ is already in reference_points)

---

## Component: `deim_criterion.py` — Loss Functions

### Init (`__init__`, lines 58, 86-87)
- `self.box_mode = box_mode`
- `self.num_reg_dist = 4 if box_mode == "hbb" else 6`

### Loss Dispatch Table

| Loss Method | HBB | OBB |
|-------------|-----|-----|
| `loss_labels_vfl` | `box_iou(box_cxcywh_to_xyxy(...))` | `probiou(target, src)` |
| `loss_labels_mal` | `box_iou / generalized_box_iou / diou / ciou / eiou` | `probiou` only |
| `loss_boxes` | L1 + `generalized_box_iou` (loss_giou) | L1 (theta/π scaled) + `kld_loss` (loss_kld) |
| `loss_local` (FGL) | `bbox2distance` on xyxy | `bbox2distance_obb` on 5-dof |
| `loss_local` (IoU weight) | 5 IoU variants supported | `probiou` only |

### OBB Loss Weights (from config)
```yaml
DEIMCriterion:
  losses: ['vfl', 'boxes', 'local']
  weight_dict: {loss_vfl: 1, loss_bbox: 5, loss_kld: 2, loss_fgl: 0.15}
```

### Known Limitation
- `get_loss_meta_info` raises `NotImplementedError()` for OBB (line 702-703) — `boxes_weight_format` not yet ported

---

## Component: `matcher.py` — Hungarian Matching

### Init (`__init__`, lines 48, 63-64, 78-89)
- `self.box_mode = box_mode`
- `self.angle_factor = angle_factor` (default `π`)
- OBB cost weights: `cost_chamfer: 5`, `cost_probiou: 2` (from config weight_dict)
- Assert: allows `cost_probiou != 0` and `cost_chamfer != 0` when `box_mode='obb'`

### Cost Matrix Construction (lines 169-180 for OBB, 181-195 for HBB)

**OBB cost formula**:
```
total_cost = cost_bbox * L1_cost
           + cost_class * class_cost
           + cost_probiou * (1 - Probiou)
           [+ cost_chamfer * chamfer_cost_obb]  # optional, if cost_chamfer != 0
```

**L1 cost details**:
```python
factor = [1, 1, 1, 1, 1/π]     # unify theta scale with cx/cy/w/h
scaled_bbox = out_bbox * factor
L1_cost = torch.cdist(scaled_bbox, tgt_bbox * factor, p=1)
```

**Probiou cost**: Uses `batch_probiou` (Gaussian-based ProbIoU). Cost = `1 - Probiou` (lower = better match).

**change_matcher mechanism**: Uses `probiou` (OBB) or `box_iou` (HBB) to compute IoU between current predictions and GTs for anchor assignment.

### Known Naming Issue
- The `cost_kld` variable name in the matcher code is misleading — it actually computes `-Probiou`, not KL divergence. See `OBB_CODE_REVIEW.md` #10.

---

## Component: `denoising.py` — Contrastive Denoising

### Init
- `box_mode` parameter accepted, sets `_num_box_dof = 5 if obb else 4`

### Noise Injection (lines 91-114)
- **Spatial noise**: Applied only to `[..., :4]` (cx, cy, w, h) — theta is NOT noised
- **HBB**: `input_query_bbox = noise_spatial` (direct replacement)
- **OBB**: 
  1. Original theta = `gt_bbox[..., 4:] / torch.pi` (to [0,1] range)
  2. Apply `inverse_sigmoid` to scaled theta
  3. Concatenate: `[noise_spatial, theta_inv_sigmoid]`
- **Comment**: "仅实现 box-noise（角度不加噪），与论文 'box noise 最优' 一致"

---

## Component: `postprocessor.py` — Result Decoding

### Forward (lines 56-64)

**HBB path**:
```python
boxes = box_convert(boxes, 'cxcywh', 'xyxy')
boxes *= orig_target_sizes[:, None, :]  # [0,1] → pixels
```

**OBB path**:
```python
# Preserve cxcywhθ format — no box_convert
factor = torch.tensor([img_w, img_h, img_w, img_h, 1])  # theta unchanged
boxes *= factor[None, None, :]
```

OBB output remains `(cx, cy, w, h, θ)` in pixel coordinates (theta already in `[0, π]`).

---

## HBB Compatibility Guarantee

When `box_mode='hbb'` (default):
1. All tensor shapes match original DEIMv2: `out_bboxes` = `(bs, nq, 4)`, `num_reg_dist = 4`
2. All loss values identical to pre-OBB DEIMv2
3. `probiou` / `chamfer_cost_obb` / `kld_loss` / `distance2bbox_obb` are never called
4. `dfine_decoder.py` OBB branch (shape==5) never triggered

Verified by: The `if/elif box_mode == "obb":` pattern in every module ensures OBB code is dead when `box_mode='hbb'`.

---

## Known Issues

Refer to `OBB_CODE_REVIEW.md`:
- **#1**: MSDeformableAttention OBB branch has math error — elementwise scaling of rotated half-vectors instead of scale-then-rotate (lines 167-184)
- **#3**: OBBFlip flips bboxes but not image (separate file, but affects detection pipeline)
- **#4**: Train/val normalization inconsistency in `deimv2_obb_sp.yml`
- **#11**: ADR vertex offset forward/backward uses inconsistent w/h reference (`dfine_utils.py:212 vs 253-255`)
- **#13**: `get_loss_meta_info` raises `NotImplementedError` for OBB (criteria, line 702-703)
