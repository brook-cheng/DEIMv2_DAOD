# Model Output Correctness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comprehensive test suite (`test/test_model_correctness.py`) validating DEIMv2-OBB model output correctness across 7 dimensions: visualization, distribution, matcher, decoder refinement, loss monotonicity, numerical stability, and gradient consistency.

**Architecture:** Single test script with shared `_setup()` / `_get_batch()` helpers (same pattern as `test_model_output.py`). Each test category is a function group, selectable via `--only`. Visual outputs saved to `test/outputs/model_correctness/`.

**Tech Stack:** PyTorch, PIL, numpy; existing `engine.core.YAMLConfig`, `engine.solver.TASKS`, `engine.deim.obb_geometry`

---

### Task 1: Scaffold test file and shared helpers

**Files:**
- Create: `test/test_model_correctness.py`

- [ ] **Step 1: Create test file with imports and helpers**

```python
"""DEIMv2-OBB 模型输出正确性验证：可视化、分布、匹配、decoder、损失、数值、梯度。

用法：
    python test/test_model_correctness.py                      # 全部测试
    python test/test_model_correctness.py --only Visualization  # 单项
"""

import os, sys, argparse, math, copy
import torch
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.deim.obb_geometry import xywhr_to_xyxyxyxy, affine_obb_matrix

OUTPUT_DIR = os.path.join(ROOT, "test", "outputs", "model_correctness")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]


def _setup(use_amp=False):
    from engine.core import YAMLConfig
    from engine.solver import TASKS

    cfg = YAMLConfig(os.path.join(ROOT, "configs/custom_obb/deimv2_obb_sp.yml"))
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = 2
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["checkpoint_freq"] = 100
    cfg.yaml_cfg["epoches"] = 1
    cfg.yaml_cfg["use_amp"] = use_amp

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()
    return solver


def _get_batch(loader):
    samples, targets = next(iter(loader))
    samples = samples.cuda()
    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
    return samples, targets


def _denorm_boxes(boxes, W, H):
    b = boxes.clone()
    b[:, 0] *= W; b[:, 1] *= H; b[:, 2] *= W; b[:, 3] *= H
    return b


def _draw_obb(draw, boxes, labels, color_offset=0):
    if boxes.numel() == 0:
        return
    verts = xywhr_to_xyxyxyxy(boxes)
    for i in range(len(boxes)):
        c = COLORS[(color_offset + i) % len(COLORS)]
        pts = [(float(verts[i, j, 0]), float(verts[i, j, 1])) for j in range(4)]
        draw.polygon(pts, outline=c, width=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()
    # test registration goes here
```

- [ ] **Step 2: Run import check**

Run: `python -c "import py_compile; py_compile.compile('test/test_model_correctness.py', doraise=True)"`
Expected: no output (syntax OK)

- [ ] **Step 3: Commit**

```bash
git add test/test_model_correctness.py
git commit -m "test: scaffold model correctness test file"
```

---

### Task 2: Visualization test (1.1)

**Files:**
- Modify: `test/test_model_correctness.py` — add `test_visualization()`

- [ ] **Step 1: Add visualization test function**

In `test_model_correctness.py`, before `if __name__ == "__main__":`, add:

```python
def test_visualization():
    """对固定 batch 跑推理，将 top-K 预测框叠加到原图上，与 GT 对比。"""
    print("\n=== Visualization ===")
    solver = _setup()
    model = solver.model
    loader = solver.train_dataloader
    samples, targets = _get_batch(loader)

    # 反归一化用
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    model.eval()
    with torch.no_grad():
        outputs = model(samples, targets=targets)

    for i in range(min(len(targets), 2)):
        img_t = samples[i].cpu() * std + mean
        img_t = (img_t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        W, H = img_t.shape[1], img_t.shape[0]

        # GT 框（绿色）
        img_gt = Image.fromarray(img_t.copy())
        gt_boxes_px = _denorm_boxes(targets[i]["boxes"].cpu(), W, H)
        _draw_obb(ImageDraw.Draw(img_gt), gt_boxes_px, targets[i]["labels"].cpu(), color_offset=0)
        img_gt.save(os.path.join(OUTPUT_DIR, f"vis_batch{i}_gt.jpg"))

        # 预测框 — top-10 按置信度
        logits = outputs["pred_logits"][i]
        boxes  = outputs["pred_boxes"][i]
        scores = logits.sigmoid().max(dim=-1).values
        topk = torch.topk(scores, min(10, len(scores)))
        pred_boxes_px = _denorm_boxes(boxes[topk.indices].cpu(), W, H)

        img_pred = Image.fromarray(img_t.copy())
        draw = ImageDraw.Draw(img_pred)
        _draw_obb(draw, pred_boxes_px, torch.zeros(len(topk.indices)), color_offset=3)
        for j, idx in enumerate(topk.indices):
            cx, cy = float(pred_boxes_px[j, 0]), float(pred_boxes_px[j, 1])
            draw.text((cx + 3, cy - 12), f"{scores[idx]:.2f}", fill=COLORS[(3+j)%6])
        img_pred.save(os.path.join(OUTPUT_DIR, f"vis_batch{i}_pred.jpg"))

        # 并排对比
        canvas = Image.new("RGB", (W * 2 + 10, H), (30, 30, 30))
        canvas.paste(img_gt, (0, 0))
        canvas.paste(img_pred, (W + 10, 0))
        ImageDraw.Draw(canvas).text((5, 2), "GT", fill=(0, 255, 0))
        ImageDraw.Draw(canvas).text((W + 15, 2), f"Pred (top-10, max_conf={scores[topk.indices[0]]:.3f})", fill=(255, 200, 0))
        canvas.save(os.path.join(OUTPUT_DIR, f"vis_batch{i}_cmp.jpg"))

        print(f"  img[{i}]: GT={len(gt_boxes_px)} boxes, pred top-10 saved")

    # 统计：pred_boxes 数量和置信度分布
    all_scores = outputs["pred_logits"].sigmoid().max(dim=-1).values.flatten()
    print(f"  score stats: min={all_scores.min():.4f} max={all_scores.max():.4f} "
          f"mean={all_scores.mean():.4f}  top10%_thr={all_scores.kthvalue(int(len(all_scores)*0.9)).values:.4f}")
    print("  ✓ done")
```

- [ ] **Step 2: Run visualization test**

Run: `CUDA_VISIBLE_DEVICES=0 python test/test_model_correctness.py --only Visualization 2>&1`
Expected: output images in `test/outputs/model_correctness/vis_batch*_cmp.jpg`

- [ ] **Step 3: Commit**

```bash
git add test/test_model_correctness.py
git commit -m "test: add visualization correctness test"
```

---

### Task 3: Prediction distribution test (1.2)

**Files:**
- Modify: `test/test_model_correctness.py` — add `test_distribution()`

- [ ] **Step 1: Add distribution test**

```python
def test_distribution():
    """检查 300 个 query 的预测分布：中心散度、θ 分布、w/h 分布、置信度分布。"""
    print("\n=== Prediction Distribution ===")
    solver = _setup()
    model = solver.model
    loader = solver.train_dataloader
    samples, targets = _get_batch(loader)

    model.eval()
    with torch.no_grad():
        outputs = model(samples, targets=targets)

    boxes = outputs["pred_boxes"]   # (B, 300, 5)
    logits = outputs["pred_logits"]  # (B, 300, C)

    B, N, C = logits.shape

    # 1) 中心分布：不应全部坍缩到 (0.5,0.5)
    cx_std = boxes[..., 0].std(dim=1)
    cy_std = boxes[..., 1].std(dim=1)
    print(f"  cx std: {cx_std.tolist()}  (expect > 0.05)")
    print(f"  cy std: {cy_std.tolist()}  (expect > 0.05)")
    assert (cx_std > 0.02).all(), f"cx collapsed: {cx_std}"
    assert (cy_std > 0.02).all(), f"cy collapsed: {cy_std}"

    # 2) θ 分布：不应全部坍缩到 0/π/2
    theta_std = boxes[..., 4].std(dim=1)
    print(f"  θ std: {theta_std.tolist()}  (expect > 0.01)")
    assert (theta_std > 0.005).all(), f"θ collapsed: {theta_std}"

    # 3) w/h 不应坍缩到 anchor 尺寸（anchor grid_size=0.05, w≈h≈0.05）
    w_mean = boxes[..., 2].mean(dim=1)
    h_mean = boxes[..., 3].mean(dim=1)
    print(f"  w mean: {w_mean.tolist()}")
    print(f"  h mean: {h_mean.tolist()}")

    # 4) 置信度：max_prob 的分布
    probs = logits.sigmoid().max(dim=-1).values  # (B, 300)
    for b in range(B):
        top5 = probs[b].topk(5).values
        print(f"  batch[{b}] top-5 scores: {[f'{v:.4f}' for v in top5]}")

    # 5) 分类多样性：是否所有 query 都预测同一类
    pred_classes = logits.sigmoid().argmax(dim=-1)  # (B, 300)
    unique_per_batch = [len(pred_classes[b].unique()) for b in range(B)]
    print(f"  unique classes per batch: {unique_per_batch}  (expect > 1)")

    print("  ✓ done")
```

- [ ] **Step 2: Run distribution test**

Run: `CUDA_VISIBLE_DEVICES=0 python test/test_model_correctness.py --only Distribution 2>&1`
Expected: all assertions pass, reasonable std values

- [ ] **Step 3: Commit**

```bash
git add test/test_model_correctness.py
git commit -m "test: add prediction distribution test"
```

---

### Task 4: Matcher verification test (2.1)

**Files:**
- Modify: `test/test_model_correctness.py` — add `test_matcher()`

- [ ] **Step 1: Add matcher test**

```python
def test_matcher():
    """验证 HungarianMatcher 匹配行为：匹配数、cost 量级、极端 case。"""
    print("\n=== Matcher ===")
    solver = _setup()
    model = solver.model
    criterion = solver.criterion
    loader = solver.train_dataloader
    samples, targets = _get_batch(loader)

    model.train()
    with torch.no_grad():
        outputs = model(samples, targets=targets)

    # 提取无 aux 的输出给 matcher
    outputs_no_aux = {k: v for k, v in outputs.items() if "aux" not in k}
    result = criterion.matcher(outputs_no_aux, targets, epoch=0)
    indices = result["indices"]

    # 1) 匹配数检查
    for i, (src_idx, tgt_idx) in enumerate(indices):
        n_gt = len(targets[i]["boxes"])
        n_match = len(src_idx)
        print(f"  batch[{i}]: {n_gt} GT → {n_match} matched "
              f"(expected ≤ min(300,{n_gt}) but ≥ 1 per GT)")
        assert n_match >= n_gt, f"not all GT matched: {n_match} < {n_gt}"

    # 2) 手动计算 cost 量级（取一个 batch element 验证）
    cost_class = criterion.matcher.cost_class
    cost_bbox  = criterion.matcher.cost_bbox
    cost_chamfer = criterion.matcher.cost_chamfer
    cost_kld  = criterion.matcher.cost_kld

    out_prob = outputs_no_aux["pred_logits"][0].sigmoid()
    out_box  = outputs_no_aux["pred_boxes"][0]
    tgt_box  = targets[0]["boxes"]
    tgt_cls  = targets[0]["labels"]

    # classification cost
    cls_cost = -out_prob[:, tgt_cls]
    print(f"  class cost range: [{cls_cost.min():.3f}, {cls_cost.max():.3f}]")

    # bbox cost (L1)
    factor = tgt_box.new_tensor([1, 1, 1, 1, 1.0 / math.pi])
    bbox_cost = torch.cdist(out_box * factor, tgt_box * factor, p=1)
    print(f"  bbox cost range: [{bbox_cost.min():.3f}, {bbox_cost.max():.3f}]")

    # chamfer cost
    from engine.deim.chamfer_cost import chamfer_cost_obb
    cf_cost = chamfer_cost_obb(out_box, tgt_box)
    print(f"  chamfer cost range: [{cf_cost.min():.3f}, {cf_cost.max():.3f}]")

    # 3) 极端 case：预测=GT → chamfer 应为 0
    gt_cost = chamfer_cost_obb(tgt_box, tgt_box)
    assert gt_cost.max() < 1e-5, f"chamfer(GT,GT) should be 0, got {gt_cost.max():.6f}"
    print(f"  chamfer(GT,GT) max: {gt_cost.max():.2e} ✓")

    print("  ✓ done")
```

- [ ] **Step 2: Run matcher test**

Run: `CUDA_VISIBLE_DEVICES=0 python test/test_model_correctness.py --only Matcher 2>&1`
Expected: all GT matched, cost ranges reasonable, chamfer(GT,GT)=0

- [ ] **Step 3: Commit**

```bash
git add test/test_model_correctness.py
git commit -m "test: add matcher verification test"
```

---

### Task 5: Decoder refinement test (2.2)

**Files:**
- Modify: `test/test_model_correctness.py` — add `test_decoder_refinement()`

- [ ] **Step 1: Add decoder test**

```python
def test_decoder_refinement():
    """验证 6 层 decoder 逐层 refinement：后期层比前期层更接近 GT。"""
    print("\n=== Decoder Refinement ===")
    solver = _setup()
    model = solver.model
    loader = solver.train_dataloader
    samples, targets = _get_batch(loader)

    model.train()
    with torch.no_grad():
        outputs = model(samples, targets=targets)

    aux_list = outputs["aux_outputs"]  # 5 layers
    last = {"pred_boxes": outputs["pred_boxes"], "pred_logits": outputs["pred_logits"]}
    all_layers = aux_list + [last]

    from engine.deim.obb_ops import batch_probiou

    for b in range(len(targets)):
        gt = targets[b]["boxes"].unsqueeze(0)  # (1, N_gt, 5)
        print(f"  batch[{b}] ({len(targets[b]['boxes'])} GT):")
        for li, layer_out in enumerate(all_layers):
            pred = layer_out["pred_boxes"][b:b+1]  # (1, 300, 5)
            ious = batch_probiou(pred.squeeze(0), gt.squeeze(0))  # (300, N_gt)
            best_iou = ious.max(dim=1).values.max().item()
            mean_iou = ious.max(dim=1).values.mean().item()
            max_prob = layer_out["pred_logits"][b].sigmoid().max().item()
            print(f"    layer {li}: best_IoU={best_iou:.4f} mean_IoU={mean_iou:.4f} max_conf={max_prob:.4f}")

        # 验证：最后一层的 best IoU >= 第一层
        ious_l0 = batch_probiou(all_layers[0]["pred_boxes"][b:b+1].squeeze(0), gt.squeeze(0))
        ious_l5 = batch_probiou(all_layers[-1]["pred_boxes"][b:b+1].squeeze(0), gt.squeeze(0))
        best0 = ious_l0.max(dim=1).values.max().item()
        best5 = ious_l5.max(dim=1).values.max().item()
        print(f"    best IoU: layer0={best0:.4f} → layer5={best5:.4f}")

    # 检查 pred_corners 残差趋势（后期层应该变动更小）
    if "pred_corners" in outputs:
        corners_last = outputs["pred_corners"]
        for li, aux in enumerate(aux_list):
            if "pred_corners" in aux:
                diff = (aux["pred_corners"] - corners_last).abs().mean().item()
                print(f"    corners diff layer{li} vs last: {diff:.4f}")

    print("  ✓ done")
```

- [ ] **Step 2: Run decoder test**

Run: `CUDA_VISIBLE_DEVICES=0 python test/test_model_correctness.py --only DecoderRefine 2>&1`
Expected: later layers have higher IoU, corners converge

- [ ] **Step 3: Commit**

```bash
git add test/test_model_correctness.py
git commit -m "test: add decoder refinement test"
```

---

### Task 6: Loss monotonicity test (2.3)

**Files:**
- Modify: `test/test_model_correctness.py` — add `test_loss_monotonicity()`

- [ ] **Step 1: Add loss test**

```python
def test_loss_monotonicity():
    """验证损失单调性：预测=GT → loss≈0；加噪声 → loss 增大。"""
    print("\n=== Loss Monotonicity ===")
    solver = _setup()
    model = solver.model
    criterion = solver.criterion
    loader = solver.train_dataloader
    samples, targets = _get_batch(loader)

    model.train()
    criterion.train()

    # 基准：正常预测的 loss
    with torch.no_grad():
        outputs_base = model(samples, targets=targets)
    losses_base = criterion(outputs_base, targets, epoch=0)
    total_base = sum(v for v in losses_base.values() if isinstance(v, torch.Tensor) and v.numel() == 1)

    # 构造完美预测：令 pred_boxes = GT, pred_logits 不变
    outputs_perfect = copy.deepcopy(outputs_base)
    B = len(targets)
    for b in range(B):
        n_gt = len(targets[b]["boxes"])
        outputs_perfect["pred_boxes"][b, :n_gt] = targets[b]["boxes"]
    losses_perfect = criterion(outputs_perfect, targets, epoch=0)
    total_perfect = sum(v for v in losses_perfect.values() if isinstance(v, torch.Tensor) and v.numel() == 1)

    print(f"  base loss:    {total_base.item():.4f}")
    print(f"  perfect loss: {total_perfect.item():.4f}  (should be lower)")

    # 加噪声：给 θ 加随机扰动
    outputs_noisy = copy.deepcopy(outputs_base)
    noise = torch.randn_like(outputs_noisy["pred_boxes"][..., 4:]) * 0.3
    outputs_noisy["pred_boxes"][..., 4:] = (
        outputs_noisy["pred_boxes"][..., 4:] + noise
    ).clamp(0, math.pi)
    losses_noisy = criterion(outputs_noisy, targets, epoch=0)
    total_noisy = sum(v for v in losses_noisy.values() if isinstance(v, torch.Tensor) and v.numel() == 1)
    print(f"  noisy (θ) loss: {total_noisy.item():.4f}  (should be higher than base)")

    # 验证
    assert total_perfect < total_base * 0.95, \
        f"perfect loss {total_perfect:.2f} not significantly lower than base {total_base:.2f}"
    assert total_noisy > total_base, \
        f"noisy loss {total_noisy:.2f} not higher than base {total_base:.2f}"

    print("  ✓ monotonicity verified")
```

- [ ] **Step 2: Run loss test**

Run: `CUDA_VISIBLE_DEVICES=0 python test/test_model_correctness.py --only LossMonotonicity 2>&1`
Expected: perfect < base < noisy

- [ ] **Step 3: Commit**

```bash
git add test/test_model_correctness.py
git commit -m "test: add loss monotonicity test"
```

---

### Task 7: Numerical stability test (3.2)

**Files:**
- Modify: `test/test_model_correctness.py` — add `test_numerical_stability()`

- [ ] **Step 1: Add numerical stability test**

```python
def test_numerical_stability():
    """压力测试：极小框 KLD、极端值 softmax、Chamfer 边界。"""
    print("\n=== Numerical Stability ===")

    from engine.deim.obb_ops import kld_loss, batch_probiou, probiou
    from engine.deim.chamfer_cost import chamfer_cost_obb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) KLD：极小框 (w→0, h→0)
    tiny = torch.tensor([[0.5, 0.5, 1e-6, 1e-6, 0.5]], device=device)
    normal = torch.tensor([[0.5, 0.5, 0.1, 0.1, 0.5]], device=device)
    loss_tiny = kld_loss(tiny, normal, reduction="none")
    assert torch.isfinite(loss_tiny).all(), f"KLD tiny: {loss_tiny}"
    print(f"  KLD(tiny, normal) = {loss_tiny.item():.6f}  finite ✓")

    # 2) KLD：极小框 vs 极小框
    loss_tiny2 = kld_loss(tiny, tiny, reduction="none")
    assert torch.isfinite(loss_tiny2).all(), f"KLD tiny2: {loss_tiny2}"
    print(f"  KLD(tiny, tiny)   = {loss_tiny2.item():.6f}  finite ✓")

    # 3) KLD：极端 θ (≈0, ≈π)
    extreme_theta = torch.tensor([
        [0.5, 0.5, 0.1, 0.1, 1e-6],
        [0.5, 0.5, 0.1, 0.1, math.pi - 1e-6],
    ], device=device)
    loss_ext = kld_loss(extreme_theta, extreme_theta, reduction="none")
    assert torch.isfinite(loss_ext).all(), f"KLD extreme θ: {loss_ext}"
    print(f"  KLD(extreme θ)   = {loss_ext.tolist()}  finite ✓")

    # 4) Chamfer：完全重叠 → 0
    box_a = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.5]], device=device)
    cf_same = chamfer_cost_obb(box_a, box_a)
    assert cf_same.max() < 1e-5, f"Chamfer(same)={cf_same.max():.2e}"
    print(f"  Chamfer(same) max = {cf_same.max():.2e}  ✓")

    # 5) Chamfer：完全不重叠 → 有合理上界
    box_far = torch.tensor([[0.1, 0.1, 0.05, 0.05, 0.0]], device=device)
    cf_far = chamfer_cost_obb(box_a, box_far)
    assert cf_far.max() < 10.0, f"Chamfer(far)={cf_far.max():.2f} too large"
    print(f"  Chamfer(far) max  = {cf_far.max():.4f}  (< 10) ✓")

    # 6) ProbIoU 尺度不变性
    box_big = torch.tensor([[0.4, 0.4, 0.3, 0.2, 0.8]], device=device)
    box_sml = torch.tensor([[0.2, 0.2, 0.15, 0.1, 0.8]], device=device)  # 一半大小
    iou_ss = probiou(box_sml, box_sml).item()
    iou_bb = probiou(box_big, box_big).item()
    assert abs(iou_ss - iou_bb) < 0.01, f"ProbIoU not scale-invariant: {iou_ss:.4f} vs {iou_bb:.4f}"
    print(f"  ProbIoU scale-inv: {iou_ss:.4f} vs {iou_bb:.4f}  ✓")

    # 7) Softmax 极端值不产生 NaN
    extreme = torch.tensor([[-1e3, 1e3, 0.0, 0.0]], device=device)
    sm = torch.nn.functional.softmax(extreme, dim=-1)
    assert torch.isfinite(sm).all(), f"softmax(extreme) NaN"
    print(f"  softmax(±1000) = {sm.tolist()}  finite ✓")

    print("  ✓ all stability checks passed")
```

- [ ] **Step 2: Run stability test**

Run: `CUDA_VISIBLE_DEVICES=0 python test/test_model_correctness.py --only NumericalStability 2>&1`
Expected: all checks finite, no NaN/Inf

- [ ] **Step 3: Commit**

```bash
git add test/test_model_correctness.py
git commit -m "test: add numerical stability tests"
```

---

### Task 8: Gradient consistency test (3.1)

**Files:**
- Modify: `test/test_model_correctness.py` — add `test_gradient_check()`

- [ ] **Step 1: Add gradient check test**

```python
def test_gradient_check():
    """用 torch.autograd.gradcheck 验证自定义算子的梯度。"""
    print("\n=== Gradient Check ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from engine.deim.obb_geometry import xywhr_to_xyxyxyxy, xyxyxyxy_to_xywhr, affine_obb_matrix
    from engine.deim.obb_ops import probiou

    torch.manual_seed(42)

    # 1) xywhr_to_xyxyxyxy
    x1 = torch.rand(3, 5, device=device, dtype=torch.float64, requires_grad=True)
    # clamp theta to [0, pi)
    x1.data[..., 4].clamp_(1e-6, math.pi - 1e-6)
    ok1 = torch.autograd.gradcheck(xywhr_to_xyxyxyxy, (x1,), eps=1e-4, atol=1e-3)
    print(f"  xywhr_to_xyxyxyxy: {'PASS' if ok1 else 'FAIL'}")

    # 2) probiou
    x2a = torch.rand(2, 5, device=device, dtype=torch.float64, requires_grad=True)
    x2b = torch.rand(2, 5, device=device, dtype=torch.float64, requires_grad=True)
    x2a.data[..., 2:4].abs_().add_(0.01)  # positive w,h
    x2b.data[..., 2:4].abs_().add_(0.01)
    x2a.data[..., 4].clamp_(1e-6, math.pi - 1e-6)
    x2b.data[..., 4].clamp_(1e-6, math.pi - 1e-6)
    def probiou_sum(a, b):
        return probiou(a, b).sum()
    ok2 = torch.autograd.gradcheck(probiou_sum, (x2a, x2b), eps=1e-4, atol=1e-3)
    print(f"  probiou: {'PASS' if ok2 else 'FAIL'}")

    # 3) affine_obb_matrix
    x3 = torch.rand(3, 5, device=device, dtype=torch.float64, requires_grad=True)
    x3.data[..., 2:4].abs_().add_(0.01)
    x3.data[..., 4].clamp_(1e-6, math.pi - 1e-6)
    mat = torch.tensor([[0.9, -0.3, 10.0], [0.2, 0.8, -5.0]], device=device, dtype=torch.float64)
    def aff_fn(boxes):
        return affine_obb_matrix(boxes, mat).sum()
    ok3 = torch.autograd.gradcheck(aff_fn, (x3,), eps=1e-4, atol=1e-3)
    print(f"  affine_obb_matrix: {'PASS' if ok3 else 'FAIL'}")

    assert ok1 and ok2 and ok3, f"gradcheck failed: to_xyxy={ok1} probiou={ok2} affine={ok3}"
    print("  ✓ all gradient checks passed")
```

- [ ] **Step 2: Run gradient test**

Run: `CUDA_VISIBLE_DEVICES=0 python test/test_model_correctness.py --only GradientCheck 2>&1`
Expected: all three gradchecks PASS

- [ ] **Step 3: Commit**

```bash
git add test/test_model_correctness.py
git commit -m "test: add gradient consistency check"
```

---

### Task 9: Wire up test registration and final verification

**Files:**
- Modify: `test/test_model_correctness.py` — add `ALL_TESTS` dict and `main()` logic

- [ ] **Step 1: Add test registry and main**

Replace `if __name__ == "__main__":` block with:

```python
ALL_TESTS = {
    "Visualization": test_visualization,
    "Distribution": test_distribution,
    "Matcher": test_matcher,
    "DecoderRefine": test_decoder_refinement,
    "LossMonotonicity": test_loss_monotonicity,
    "NumericalStability": test_numerical_stability,
    "GradientCheck": test_gradient_check,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    if args.only:
        ALL_TESTS[args.only]()
    else:
        passed = failed = 0
        for name, fn in ALL_TESTS.items():
            try:
                fn()
                passed += 1
            except Exception as e:
                print(f"  FAILED: {e}")
                failed += 1
                import traceback
                traceback.print_exc()
        print(f"\n{'='*50}")
        print(f"{passed} passed, {failed} failed")
        if failed:
            sys.exit(1)
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full test suite**

Run: `CUDA_VISIBLE_DEVICES=0 timeout 600 python test/test_model_correctness.py 2>&1`
Expected: 7/7 passed

- [ ] **Step 3: Run syntax check on all test files**

```bash
python -c "import py_compile; py_compile.compile('test/test_model_correctness.py', doraise=True); print('OK')"
python -c "import py_compile; py_compile.compile('test/test_obb_transforms.py', doraise=True); print('OK')"
python -c "import py_compile; py_compile.compile('test/test_model_output.py', doraise=True); print('OK')"
```

- [ ] **Step 4: Final commit**

```bash
git add test/test_model_correctness.py
git commit -m "test: complete model correctness test suite (7 categories)"
```
