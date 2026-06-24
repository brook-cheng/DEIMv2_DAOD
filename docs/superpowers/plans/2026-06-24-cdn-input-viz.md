# CDN Generation Input Visualization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `test_cdn_generation_visualization()` to `test/test_cdn_inspect.py` — verify CDN **input** data (before decoder) by drawing positive/negative noisy queries vs GT as ellipses.

**Architecture:** Pure unit test, CPU-only (mock data, no model/checkpoint load). Uses `get_contrastive_denoising_training_group()` with `num_denoising=20, num_queries=40`. GT targets drawn as ellipses (natural for OBB Gaussian representation via `oriented_box_to_gaussian`). Positive CDN queries (correct label + box noise → should be near GT) in red, negative CDN queries (wrong label + box noise → should be scattered) in orange. Output: 3-panel PNG per sample.

**Tech Stack:** pytest, torch, matplotlib (Agg backend), engine/deim/denoising.py, engine/deim/obb_ops.py (Gaussian covariance)

## Global Constraints

- Must NOT modify production code (`engine/`, `configs/`)
- Must NOT load checkpoint or GPU
- All tensors on CPU
- Output PNGs to `test/outputs/cdn_inspect/cdn_input_viz_sample{N}.png`
- Use existing `_draw_obb_boxes` helper from test file
- Follow existing test patterns (sys.path insert, no test class)

---

### Task 1: Add ellipse-drawing helper for OBB GT visualization

**Files:**
- Modify: `test/test_cdn_inspect.py` — append `_draw_obb_ellipses()` helper

**Interfaces:**
- Produces: `_draw_obb_ellipses(ax, boxes_px, color, linewidth=1.5)` — draws Gaussian covariance ellipses for OBB boxes on matplotlib axis

- [ ] **Step 1: Append `_draw_obb_ellipses` helper after `_draw_obb_boxes`**

`oriented_box_to_gaussian` from `obb_ops.py` returns `(mu, sigma)` where `mu` is center `(cx,cy)` and `sigma` is 2×2 covariance. The 1σ ellipse is defined by `sigma`'s eigenvectors/eigenvalues. Drawing approach: use `matplotlib.patches.Ellipse` with `width=2*sqrt(eigval1), height=2*sqrt(eigval2), angle=arctan2(eigvec_y, eigvec_x)`.

```python
def _draw_obb_ellipses(ax, boxes_px, color, linewidth=1.5):
    """Draw OBB boxes as Gaussian covariance ellipses (1σ contour).

    boxes_px: (N, 5) — [cx, cy, w, h, theta] in pixel coords, theta in radians.
    """
    import numpy as np
    from matplotlib.patches import Ellipse

    if boxes_px.numel() == 0:
        return
    for i in range(boxes_px.shape[0]):
        cx, cy, w, h, theta = boxes_px[i].tolist()
        # Build 2D Gaussian covariance from (w,h,theta)
        # sigma = R(θ) · diag((w/2)², (h/2)²) · R(θ)ᵀ
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        D = np.diag([(w / 2) ** 2, (h / 2) ** 2])
        sigma = R @ D @ R.T
        # Eigendecomposition for ellipse params
        eigvals, eigvecs = np.linalg.eigh(sigma)
        width = 2 * np.sqrt(eigvals[0])   # 1σ × 2
        height = 2 * np.sqrt(eigvals[1])
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ell = Ellipse(
            (cx, cy), width, height, angle=angle,
            linewidth=linewidth, edgecolor=color, facecolor='none', alpha=0.6,
        )
        ax.add_patch(ell)
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('test/test_cdn_inspect.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add test/test_cdn_inspect.py
git commit -m "feat(test): add _draw_obb_ellipses helper for Gaussian OBB visualization"
```

---

### Task 2: Add `test_cdn_generation_visualization` test function

**Files:**
- Modify: `test/test_cdn_inspect.py` — append `test_cdn_generation_visualization()`

**Interfaces:**
- Consumes: `_draw_obb_boxes`, `_draw_obb_ellipses`, `get_contrastive_denoising_training_group` (already imported)
- Produces: PNG files at `test/outputs/cdn_inspect/cdn_input_viz_sample0.png`, `test/outputs/cdn_inspect/cdn_input_viz_sample1.png`

- [ ] **Step 1: Write the test function**

```python
def test_cdn_generation_visualization():
    """Visualize CDN input queries (before decoder): GT ellipses vs positive/negative noisy queries.

    num_denoising=20, num_queries=40 — small enough to see individual boxes clearly.
    GT targets drawn as Gaussian ellipses. Positive CDN queries (correct label + box noise)
    in red, negative CDN queries (wrong label + box noise) in orange.
    """
    num_classes = 10
    hidden_dim = 64
    num_denoising = 20
    num_queries = 40

    # ── mock class embedding ──
    class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)

    # ── GT targets: 2 images, 2 boxes each (class 3 and 7), boxes in cxcywhθ (normed, θ∈[0,π]) ──
    targets = [
        {
            "labels": torch.tensor([3, 7]),
            "boxes": torch.tensor([
                [0.30, 0.40, 0.20, 0.30, 0.50],
                [0.60, 0.60, 0.15, 0.25, 1.20],
            ]),
        },
        {
            "labels": torch.tensor([3, 7]),
            "boxes": torch.tensor([
                [0.20, 0.30, 0.10, 0.20, 2.00],
                [0.70, 0.50, 0.20, 0.15, 0.80],
            ]),
        },
    ]

    # ── generate CDN inputs ──
    input_query_logits, input_query_bbox_unact, attn_mask, dn_meta = (
        get_contrastive_denoising_training_group(
            targets=targets,
            num_classes=num_classes,
            num_queries=num_queries,
            class_embed=class_embed,
            num_denoising=num_denoising,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            box_mode="obb",
        )
    )
    assert input_query_logits is not None, "CDN generation returned None"

    # ── split positive/negative ──
    # num_denoising=20, max_gt_num=2, num_group=10, actual_denoising=40
    # dn_num_split = [40, num_queries] = [40, 40]
    actual_denoising = input_query_bbox_unact.shape[1]  # 40 = 2 * 2 * 10
    dn_positive_idx = dn_meta["dn_positive_idx"]  # list of 2 tensors

    # ── denormalize: bbox_unact is in inverse_sigmoid space. Apply sigmoid to get [0,1], then rescale θ ──
    cdn_bboxes_norm = input_query_bbox_unact.sigmoid()  # (2, actual_denoising, 5)
    # θ was divided by π in denoising.py:111 → restore to [0,π]
    cdn_bboxes_norm[..., 4] = cdn_bboxes_norm[..., 4] * torch.pi
    # cx,cy,w,h are in [0,1]; θ in [0,π]

    # ── per-image visualization ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = "test/outputs/cdn_inspect"
    os.makedirs(output_dir, exist_ok=True)

    for img_idx in range(2):
        gt_boxes = targets[img_idx]["boxes"]  # (2, 5), normalized
        gt_labels = targets[img_idx]["labels"]

        # Denormalize GT boxes to pixel coords (assume 640×640 image)
        img_w, img_h = 640.0, 640.0
        gt_boxes_px = gt_boxes.clone()
        gt_boxes_px[:, 0] *= img_w
        gt_boxes_px[:, 1] *= img_h
        gt_boxes_px[:, 2] *= img_w
        gt_boxes_px[:, 3] *= img_h

        # CDN positive: correct label + box noise → should be near GT
        pos_idx = dn_positive_idx[img_idx]
        pos_boxes_norm = cdn_bboxes_norm[img_idx][pos_idx]  # (N_pos, 5)
        pos_boxes_px = pos_boxes_norm.clone()
        pos_boxes_px[:, 0] *= img_w
        pos_boxes_px[:, 1] *= img_h
        pos_boxes_px[:, 2] *= img_w
        pos_boxes_px[:, 3] *= img_h

        # CDN negative: wrong label + box noise → scattered
        all_idx = set(range(actual_denoising))
        pos_set = set(pos_idx.tolist())
        neg_idx_list = sorted(all_idx - pos_set)
        neg_idx = torch.tensor(neg_idx_list)
        neg_boxes_norm = cdn_bboxes_norm[img_idx][neg_idx]  # (N_neg, 5)
        neg_boxes_px = neg_boxes_norm.clone()
        neg_boxes_px[:, 0] *= img_w
        neg_boxes_px[:, 1] *= img_h
        neg_boxes_px[:, 2] *= img_w
        neg_boxes_px[:, 3] *= img_h

        # ── 3-panel figure ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: GT as ellipses (green)
        axes[0].set_xlim(0, img_w)
        axes[0].set_ylim(img_h, 0)  # inverted y for image coords
        axes[0].set_aspect("equal")
        axes[0].set_title(f"GT (green ellipses)\nImage {img_idx}")
        _draw_obb_ellipses(axes[0], gt_boxes_px, "green", linewidth=2.0)
        # Also draw GT centers
        axes[0].scatter(gt_boxes_px[:, 0], gt_boxes_px[:, 1],
                         c="green", s=30, marker="x", zorder=5)

        # Panel 2: CDN positive (red) — noisy queries with correct labels
        axes[1].set_xlim(0, img_w)
        axes[1].set_ylim(img_h, 0)
        axes[1].set_aspect("equal")
        axes[1].set_title(f"CDN Positive ({len(pos_idx)} queries)\nCorrect label + box noise")
        _draw_obb_boxes(axes[1], pos_boxes_px, "red", linewidth=1.0)
        axes[1].scatter(pos_boxes_px[:, 0], pos_boxes_px[:, 1],
                         c="red", s=20, marker="o", alpha=0.5, zorder=5)

        # Panel 3: CDN negative (orange) — wrong label queries
        axes[2].set_xlim(0, img_w)
        axes[2].set_ylim(img_h, 0)
        axes[2].set_aspect("equal")
        axes[2].set_title(f"CDN Negative ({len(neg_idx)} queries)\nWrong label + box noise")
        _draw_obb_boxes(axes[2], neg_boxes_px, "orange", linewidth=1.0)
        axes[2].scatter(neg_boxes_px[:, 0], neg_boxes_px[:, 1],
                         c="orange", s=20, marker="o", alpha=0.5, zorder=5)

        for ax in axes:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"cdn_input_viz_sample{img_idx}.png")
        plt.savefig(out_path, dpi=100)
        plt.close()

    # ── assertions ──
    for img_idx in range(2):
        path = os.path.join(output_dir, f"cdn_input_viz_sample{img_idx}.png")
        assert os.path.exists(path), f"Missing {path}"
        size = os.path.getsize(path)
        assert size > 5000, f"{path} too small: {size} bytes"
        with open(path, "rb") as f:
            magic = f.read(4)
        assert magic == b"\x89PNG", f"{path} not a valid PNG: {magic}"
        print(f"  {path}: {size:,} bytes, valid PNG")
```

- [ ] **Step 2: Run the test**

```bash
python -m pytest test/test_cdn_inspect.py::test_cdn_generation_visualization -v
```

Expected: `PASSED` with 4 assertions passing (2 PNGs × 2 assertions = file exists + valid magic)

- [ ] **Step 3: Verify output files**

```bash
ls -lh test/outputs/cdn_inspect/cdn_input_viz_sample*.png
python3 -c "
for i in [0,1]:
    p = f'test/outputs/cdn_inspect/cdn_input_viz_sample{i}.png'
    with open(p,'rb') as f:
        magic = f.read(4)
    import os
    print(f'{p}: {os.path.getsize(p):,}b, magic={magic}, valid={magic==b\"\\x89PNG\"}')"
```

Expected: 2 files, each > 5KB, valid PNG

- [ ] **Step 4: Run full test suite to verify no regression**

```bash
python -m pytest test/test_cdn_inspect.py -v
```

Expected: All 4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add test/test_cdn_inspect.py test/outputs/cdn_inspect/cdn_input_viz_sample0.png test/outputs/cdn_inspect/cdn_input_viz_sample1.png
git commit -m "test(cdn): add CDN input generation visualization with GT ellipses"
```

---

## Self-Review

1. **Spec coverage**: ✅ Single feature — visualize CDN input queries vs GT ellipses
2. **Placeholder scan**: ✅ No TBD/TODO/vague steps. All code shown inline. All commands explicit.
3. **Type consistency**: ✅ `_draw_obb_ellipses` defined in Task 1, consumed in Task 2. Both use same signature. `get_contrastive_denoising_training_group` already imported.

## Execution Handoff

Plan complete. Proceed to Task 1.
