# Spec: obb-evaluation

> **Status**: Implemented
> **Files**: `engine/data/transforms/obb_transforms.py`, `engine/data/transforms/mosaic.py`, `engine/data/dataset/dota_dataset.py`, `engine/data/dataloader.py`, `engine/eval/obb_eval.py`, `engine/eval/dota_eval.py`, `engine/eval/poly_iou.py`

Data pipeline (transforms, dataset, dataloader) and evaluation (online and offline) for OBB detection. All OBB data flows use the convention `(cx, cy, w, h, θ)` with `θ ∈ [0, π)` radians in pixel coordinates, normalized to `[0, 1]` by `OBBConvertBoxes` before model input.

---

## Data Pipeline Ordering

```
Load Image + Annotations (DotaDataset / YOLO-OBB)
  → OBBFlip (p=0.5)              # horizontal flip + mirror θ
  → OBBZoomOut (random padding)  # center translation only
  → OBBResize ([640, 640])       # scale to target size
  → OBBIoUCrop (random crop)     # IoU-guided crop with center filtering
  → OBBSanitize (min_size=1)     # filter tiny/zero boxes
  → OBBConvertBoxes              # normalize cx/w, cy/h → [0,1], θ unchanged
  → [if Mosaic: combine 4 images]
  → [if CopyBlend/Mixup: mix images with OBB-aware AABB clips]
  → Model forward
```

**Critical invariant**: `OBBFlip` MUST execute before `OBBConvertBoxes` — the flip formula `cx' = W - cx` requires pixel coordinates. The current config order satisfies this.

---

## Transform: `OBBFlip` (`obb_transforms.py:11-18`)

| Property | Value |
|----------|-------|
| Trigger prob | `p=0.5` |
| Image transform | Horizontal flip via `TF.hflip()` |
| Box center | `cx' = W - cx` |
| Box angle | `θ' = (π - θ) % π` (mirror, compress back to [0,π)) |
| w, h | Unchanged |
| **Known issue** | OBB_CODE_REVIEW.md #3: image flip not implemented (source has `# FIXME: 图片没有Flip`) — **implemented in fix code** but may not be committed |

### Forward
```python
def forward(self, sample):
    img, tgt, ds = sample
    if random.random() > self.p: return sample
    img = TF.hflip(img)
    b = tgt["boxes"]  # (N,5) in pixels
    b[:, 0] = w - b[:, 0]
    b[:, 4] = (torch.pi - b[:, 4]) % torch.pi
    return img, tgt, ds
```

---

## Transform: `OBBZoomOut` (`obb_transforms.py:22-47`)

Random padding (pure translation, no scale change).
- Samples padding `(pad_l, pad_r, pad_t, pad_b)` within configured range
- Pad image via `TF.pad()`, fill with mean color
- Call `affine_obb(tx=pad_l, ty=pad_t)` — only shifts centers

---

## Transform: `OBBResize` (`obb_transforms.py:50-67`)

Scale image + boxes to target size `[H, W]`.
- `sx = self.size[1] / w`, `sy = self.size[0] / h`
- Call `affine_obb(sx, sy)` — scales w, h proportionally
- **Known issue**: OBB_CODE_REVIEW.md #12: H/W index may be swapped (line 62 uses `s[0]/w`, should be `s[1]/w`). No effect when size is square (640×640).

---

## Transform: `OBBIoUCrop` (`obb_transforms.py:122-191`)

Random crop guided by box overlap (via HBB axis-aligned approximation).
- Generates candidate crop regions
- For each candidate, computes axis-aligned IoU between cropped boxes and full boxes → picks best
- Applies `affine_obb(tx, ty)` to shift centers
- Filters centers outside crop `[0, w_crop) × [0, h_crop)`
- **Known issue**: OBB_CODE_REVIEW.md #7: uses HBB IoU proxy; anisotropic crops don't update θ

---

## Transform: `OBBSanitize` (`obb_transforms.py`)

Simple filter: `keep = (w > min_size) & (h > min_size)`.
Removes boxes with width or height ≤ threshold after transforms.

---

## Transform: `OBBConvertBoxes` (`obb_transforms.py:71-85`)

Normalize to `[0, 1]` for model input.
- `cx /= W`, `cy /= H`
- `w /= W`, `h /= H`
- **θ is preserved unchanged** (angle is scale-invariant)
- **Hardcoded `img_size=(640, 640)`** — must match `OBBResize` target. See OBB_CODE_REVIEW.md #14.

---

## Transform: Mosaic (`mosaic.py`)

### OBB Branch Detection
```python
is_obb = boxes.shape[-1] == 5
```

### Placement Offsets (lines 131-138, 173-179)
- **OBB**: `offsets = [dx, dy, 0, 0, 0]` — only centers shifted per quadrant
- **HBB**: `offsets = [dx, dy, w, h]` — full bbox translated

### Affine Transform (lines 292-313)
- **HBB**: `RandomAffine(degrees, translate, scale, shear)` via torchvision
- **OBB**: `_affine_obb()` — custom OBB-aware:
  1. Sample `(φ, s, tx, ty)`
  2. Construct forward matrix `mat_fwd = [A | b]` (rotation + scale + translation)
  3. Call `affine_obb_matrix(boxes, mat_fwd)`
  4. Warp image via inverse matrix (`PIL.Image.transform(AFFINE)`)
  5. Filter: centers in bounds, w>1, h>1

### Known Issue
OBB_CODE_REVIEW.md #2: placement offsets `[dx,dy,dx,dy,0]` incorrectly add translation to w,h. Fix: use `[dx,dy,0,0,0]`.

---

## Dataset: `DotaDataset` (`dota_dataset.py`)

Registered via `@register()`. Inherits from `DetDataset`.

### Init Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `img_folder` | str | Path to image directory |
| `ann_folder` | str | Path to annotation directory |
| `classes_file` | str | Path to `classes.txt` (one class per line) |
| `transforms` | Compose | Transform pipeline |
| `format` | str | `"DOTA"` or `"YOLO-OBB"` |

### Annotation Formats

**DOTA format** (per line):
```
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```
→ 8-corner polygon parsed → `xyxyxyxy_to_xywhr` → `(cx,cy,w,h,θ)`

**YOLO-OBB format** (per line):
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```
→ normalized coordinates (÷W,÷H) → denormalize → `xyxyxyxy_to_xywhr` → `(cx,cy,w,h,θ)`

### Key Methods
| Method | Role |
|--------|------|
| `img_ann_dict` (property) | Lazy-builds `{img_name: ann_name}` mapping by extension matching |
| `cat2id` / `id2cat` | Label name ↔ integer ID mappings |
| `load_item(idx)` | Loads image + parses annotations → returns `(PIL.Image, {"boxes":(N,5), "labels":(N,), "orig_size":[w,h]})` |

### Known Fixes
- Handles class names with spaces in `classes.txt` (commit `c4c5408`)
- Supports `difficulty=2` (don't care) annotation skipping

---

## DataLoader OBB Support (`dataloader.py`)

### `BatchImageCollateFunction` (CopyBlend/Mixup)

**OBB detection** (line ~319):
```python
is_obb = box.shape[0] == 5  # 1D box tensor: [cx,cy,w,h,θ]
```

**OBB CopyBlend**:
1. Convert OBB `(cx,cy,w,h,θ)` → 4 vertices via `xywhr_to_xyxyxyxy`
2. Compute axis-aligned bounding box of 4 vertices as crop region
3. Blend source image patch into target at that AABB
4. Preserve `(cx,cy,w,h,θ)` in blended target annotations

### Known Limitation
Multi-scale training resize (line 94) has `# FIXME` — OBB boxes not re-normalized during multi-scale. Currently only square resize works.

---

## Online Evaluation: `obb_evaluate()` (`obb_eval.py`)

### Signature
```python
def obb_evaluate(model, postprocessor, data_loader, device, iou_thrs=(0.5,), num_classes=15):
```
Returns: `{"AP50": float, "mAP": float, "precision": float, "recall": float}`

### Pipeline (3 stages)

**Stage 1 — Collect**:
- Iterates `data_loader`, runs `model + postprocessor`
- Denormalizes GT boxes from `[0,1]` to pixel coordinates
- Groups detections/GT per `(class_id, image_idx)`

**Stage 2 — IoU**:
- Calls `batch_probiou(det_t, gt_t)` — Gaussian ProbIoU (vectorized, machine precision speed)
- **NOT** exact polygon IoU (`poly_iou.py` is dead code)

**Stage 3 — TP/FP + AP**:
- Per class, per IoU threshold, per image:
  - Sort detections descending by score
  - Greedy match: each detection matched to highest-IoU unmatched GT with IoU ≥ threshold
  - Accumulate TP/FP counts, compute precision/recall
- `_voc_ap(rec, prec)`: VOC07 11-point interpolated AP

### Known Issues
- OBB_CODE_REVIEW.md #5: AP appended twice per class (once sorted, once unsorted) → `aps` length = `2 × num_classes`, mAP contaminated
- OBB_CODE_REVIEW.md #6: Uses ProbIoU (Gaussian approximation) instead of exact polygon IoU — differs from DOTA official evaluation

---

## Offline Evaluation: `evaluate_dota()` (`dota_eval.py`)

### Signature
```python
def evaluate_dota(det_dir, gt_dir, image_list, iou_thr=0.5):
```

### Pipeline
1. Read predictions from `Task1_{class}.txt` (8-corner polygon format per DOTA devkit)
2. Read GT from DOTA annotation `.txt` files
3. Convert both to `(cx,cy,w,h,θ)` via `xyxyxyxy_to_xywhr`
4. Compute IoU via `_poly_iou_8coord` → **internally calls `batch_probiou`** (ProbIoU, not polygon)
5. Greedy TP/FP matching, VOC07 AP

### Naming Mismatch
Despite the name `_poly_iou_8coord`, it uses ProbIoU (Gaussian-based), not polygon-based IoU. Same behavior as `obb_evaluate`.

---

## Dead Code: `poly_iou.py`

| Status | DEAD CODE |
|--------|-----------|
| Location | `engine/eval/poly_iou.py` |
| Exported | `engine/eval/__init__.py` (line 1) |
| Used by | `test/test_poly_iou.py`, `test/test_profile_obb_eval.py` (tests only) |
| NOT used by | `obb_evaluate`, `evaluate_dota`, any production path |
| Contains | `poly_iou(obb1, obb2)` — exact polygon IoU via shapely (Python double loop, ~18s/class) |
| Replacement | `batch_probiou` (vectorized, ~5000x faster, Gaussian-based approximation) |

See `2026-07-09-obb-eval-speed-optimization.md` for the optimization rationale.

---

## Known Issues Summary

Refer to `OBB_CODE_REVIEW.md`:
| # | Issue | Severity | File |
|---|-------|----------|------|
| 2 | Mosaic adds [dx,dy,dx,dy] to w,h of 5-dof boxes | Severe | `mosaic.py` |
| 3 | OBBFlip flips boxes but not image | Severe | `obb_transforms.py` |
| 5 | AP appended twice (sorted + unsorted) | Severe | `obb_eval.py` |
| 6 | ProbIoU used instead of exact polygon IoU | Major | `obb_eval.py`, `dota_eval.py` |
| 7 | Anisotropic resize/crop doesn't update θ | Major | `obb_transforms.py` |
| 12 | OBBResize H/W index may be swapped (no effect at 640×640) | Low | `obb_transforms.py` |
| 14 | OBBConvertBoxes hardcoded img_size=(640,640) | Low | `obb_transforms.py` |
