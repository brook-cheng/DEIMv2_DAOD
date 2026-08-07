# Spec: obb-geometry

> **Status**: Implemented
> **Files**: `engine/deim/obb_geometry.py`, `engine/deim/obb_ops.py`, `engine/deim/chamfer_cost.py`, `engine/deim/dfine_utils.py` (OBB functions only)

Oriented Bounding Box (OBB) geometry, overlap metrics, and distribution-based regression utilities. All OBBs use the convention `(cx, cy, w, h, θ)` with `θ ∈ [0, π)` radians (le90 convention).

---

## Coordinate Convention

| Field | Range | Meaning |
|-------|-------|---------|
| `cx` | `[0, W]` / `[0, 1]` | Center x (pixel or normalized) |
| `cy` | `[0, H]` / `[0, 1]` | Center y (pixel or normalized) |
| `w` | `> 0` | Width (along θ direction) |
| `h` | `> 0` | Height (perpendicular to θ) |
| `θ` | `[0, π)` | Angle from positive x-axis to width direction (le90) |

Four corner vertices are returned in clockwise order from top-left: `(x1,y1), (x2,y2), (x3,y3), (x4,y4)`.

---

## Module: `obb_geometry.py`

**Purpose**: Bi-directional mapping between OBB representations (5-dof, 4-vertex, ADR external-rect+offsets) and affine transforms.

### Functions

#### `xywhr_to_xyxyxyxy(xywhr) -> Tensor`
Convert OBB 5-dof to four corner vertices.
- **Input**: `xywhr` — `(..., 5)` tensor `(cx, cy, w, h, θ)`
- **Output**: `(..., 4, 2)` tensor of clockwise vertices
- **Math**: `v_i = center + R(θ) · corner_offset_i`

#### `xyxyxyxy_to_xywhr(xyxyxyxy) -> Tensor`
Convert four corner vertices back to OBB 5-dof.
- **Input**: `xyxyxyxy` — `(..., 4, 2)` tensor of vertices
- **Output**: `(..., 5)` tensor `(cx, cy, w, h, θ)`
- **Algorithm**: Mean of opposite edges = center; edge lengths = w, h; direction vector = θ

#### `oriented_box_to_external_rect(obbs) -> Tuple[Tensor, Tensor]`
OBB → axis-aligned external rectangle + vertex offset parameters (ADR decomposition).
- **Input**: `obbs` — `(..., 5)` tensor
- **Output**: 
  - `external_rect` — `(..., 4)` tensor `(x1, y1, x2, y2)` of bounding box
  - `vertex_offsets` — `(..., 2)` tensor `(ε, η)`:
    - `ε`: distance from OBB top vertex to external rect top-right corner
    - `η`: distance from OBB rightmost vertex to external rect bottom-right corner
- **Used by**: `distance2bbox_obb`, `bbox2distance_obb` for ADR 6-distribution DDF

#### `external_rect_to_oriented_box(external_rect, vertex_offsets, eps=1e-9) -> Tensor`
Inverse ADR: external rectangle + vertex offsets → OBB.
- **Input**: `external_rect` `(..., 4)`, `vertex_offsets` `(..., 2)`
- **Output**: `(..., 5)` OBB tensor
- **eps**: Prevents division by zero when w≈0 or h≈0

#### `affine_obb(boxes_xywhr, sx, sy, tx=0.0, ty=0.0) -> Tensor`
Scale + translate OBBs in pixel coordinates.
- **Input**: `boxes` `(N, 5)`, scale factors `sx, sy`, translations `tx, ty`
- **Output**: `(N, 5)` transformed OBBs
- **Algorithm**: Decompose to vertices → scale each vertex → refit to OBB → recompute θ

#### `affine_obb_matrix(boxes_xywhr, mat) -> Tensor`
General forward affine transform on OBBs: `v' = A @ v + b`.
- **Input**: `boxes` `(N, 5)`, `mat` `(2, 3)` affine matrix `[A | b]`
- **Output**: `(N, 5)` transformed OBBs
- **Used by**: Mosaic `_affine_obb` for rotation+scale+translation

---

## Module: `obb_ops.py`

**Purpose**: Gaussian-based overlap metrics (ProbIoU) and KL Divergence loss for OBBs.

### Functions

#### `xy_wh_r_2_xy_sigma(xywhr) -> tuple[Tensor, Tensor]`
Convert OBB to 2D Gaussian distribution.
- **Input**: `xywhr` — `(N, 5)`
- **Output**: `(mean, sigma)` where `mean` is `(N, 2)` and `sigma` is `(N, 2, 2)` covariance matrix
- **Math**: `Σ = R(θ) @ diag(w²/4, h²/4) @ R(θ)ᵀ`

#### `_get_covariance_matrix(boxes) -> tuple[Tensor, Tensor, Tensor]` *(private)*
Extract covariance parameters `(a, b, c)` from OBBs for efficient ProbIoU computation.
- **Output**: `a = Σ[0,0]`, `b = Σ[0,1] = Σ[1,0]`, `c = Σ[1,1]`

#### `probiou(obb1, obb2, CIoU=False, eps=1e-7) -> Tensor`
Probabilistic IoU between two matched sets of OBBs.
- **Input**: `obb1`, `obb2` — both `(N, 5)`
- **Output**: `(N,)` tensor of IoU scores in `[0, 1]`
- **CIoU**: If True, also returns central point distance term
- **Math**: Bhattacharyya distance between two 2D Gaussians → IoU

#### `batch_probiou(obb1, obb2, eps=1e-7) -> Tensor`
Pairwise ProbIoU matrix between two OBB sets.
- **Input**: `obb1` `(N, 5)`, `obb2` `(M, 5)`
- **Output**: `(N, M)` IoU matrix
- **Used by**: `obb_evaluate`, `dota_eval._poly_iou_8coord`, `matcher.forward`

#### `kld_loss(pred, target, fun="log1p", tau=1.0, reduction="mean", eps=1e-7) -> Tensor`
KL Divergence loss between predicted and target OBBs modeled as 2D Gaussians.
- **Input**: `pred` `(N, 5)`, `target` `(N, 5)`
- **Output**: scalar loss (or `(N,)` if `reduction="none"`)
- **fun**: Nonlinearity — `"log1p"` (default) or `"sqrt"`
- **tau**: Temperature for log1p: `log(1 + τ·KLD)`
- **Math**: `KLD = ½[tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁) - 2 - ln(det(Σ₁)/det(Σ₂))]`

#### `rbbox_overlaps_obb(boxes1, boxes2, mode="probiou", eps=1e-7) -> Tensor`
OBB overlap dispatcher.
- **Input**: `boxes1` `(N, 5)`, `boxes2` `(M, 5)`
- **Output**: `(N, M)` overlap scores in `[0, 1]`
- **mode**: Currently only `"probiou"` supported

---

## Module: `chamfer_cost.py`

**Purpose**: Chamfer distance for OBB sets, used in Hungarian matching cost matrix.

### Functions

#### `chamfer_cost_obb(boxes1, boxes2) -> Tensor`
Vertex-set Chamfer distance between two OBB sets.
- **Input**: `boxes1` `(N, 5)`, `boxes2` `(M, 5)`
- **Output**: `(N, M)` cost matrix (lower = better match)
- **Algorithm**:
  1. Decompose each OBB to 4 corner vertices
  2. For each (box1, box2) pair: compute bidirectional min-pointset distance
  3. Cost = mean(min(||v₁_i - v₂_j||²) + min(||v₂_j - v₁_i||²)) / 2
- **Reference**: O2-RTDETR Eq.5 — bidirectional min-mean squared distance on 4 vertices

---

## Module: `dfine_utils.py` (OBB functions only)

**Purpose**: Bridge D-FINE distribution-focused regression to OBB via ADR 6-distribution encoding.

### OBB Functions

#### `distance2bbox_obb(points, distance, reg_scale) -> Tensor`
Decode 6-distribution DDF output → 5-dof OBB (forward/inference direction).
- **Input**:
  - `points` — `(..., 4)` anchor points `(cx, cy, w, h)` (axis-aligned)
  - `distance` — `(..., 6)` predicted distribution offsets `(α, β, γ, δ, ε, η)`
  - `reg_scale` — scalar or `(2,)` scale factor for w/h regression
- **Output**: `(..., 5)` OBB `(cx, cy, w, h, θ)`
- **Algorithm**:
  1. First 4 channels `(α,β,γ,δ)` adjust the axis-aligned external rectangle edges
  2. Last 2 channels `(ε,η)` are vertex offsets relative to external rectangle corners
  3. Calls `external_rect_to_oriented_box(ext_rect, vertex_offsets)` to reconstruct OBB

#### `bbox2distance_obb(points, bbox, reg_max, reg_scale, up, eps=0.1) -> Tuple[Tensor, Tensor, Tensor]`
Encode GT OBB → 6-distribution FGL targets (reverse/inverse of `distance2bbox_obb`).
- **Input**:
  - `points` — `(..., 4)` anchor points `(cx, cy, w, h)`
  - `bbox` — `(..., 5)` GT OBB
  - `reg_max` — max bin index for distribution (typically 32)
  - `reg_scale` — regression scale factor
  - `up` — `(reg_max,)` tensor of non-uniform weighting values from `weighting_function()`
  - `eps` — prevents division by zero
- **Output**: 3 tensors:
  - `dis_left` — `(..., 6, reg_max)` left-side distribution values (α,β,γ,δ,ε,η)
  - `dis_right` — `(..., 6, reg_max)` right-side distribution values
  - `bbox_mask` — `(..., 6)` mask (1.0 where GT exists, 0.0 otherwise)
- **Algorithm**:
  1. Decompose GT OBB → external rectangle + vertex offsets via `oriented_box_to_external_rect`
  2. Each of 6 components encoded as FGL distribution via `weighting_function` + `translate_gt`

---

## Shape Consistency

| Function | Input Shape | Output Shape |
|----------|------------|--------------|
| `xywhr_to_xyxyxyxy` | `(*, 5)` | `(*, 4, 2)` |
| `xyxyxyxy_to_xywhr` | `(*, 4, 2)` | `(*, 5)` |
| `oriented_box_to_external_rect` | `(*, 5)` | `(*, 4)`, `(*, 2)` |
| `external_rect_to_oriented_box` | `(*, 4)`, `(*, 2)` | `(*, 5)` |
| `probiou` | `(N, 5)`, `(N, 5)` | `(N,)` |
| `batch_probiou` | `(N, 5)`, `(M, 5)` | `(N, M)` |
| `chamfer_cost_obb` | `(N, 5)`, `(M, 5)` | `(N, M)` |
| `distance2bbox_obb` | `(*, 4)`, `(*, 6)` | `(*, 5)` |
| `bbox2distance_obb` | `(*, 4)`, `(*, 5)` | `(*, 6, rmax)` each |

---

## Known Issues

Refer to `OBB_CODE_REVIEW.md` for issues found in this layer:
- **#9**: KLD `det` clamp on `a01²` instead of full determinant (line 184, 198)
- **#10**: `cost_kld` in matcher is actually `-ProbiOU`, not KL divergence (semantic)
- **#11**: ADR vertex offset forward/backward scaling uses inconsistent w/h reference
