# 匈牙利匹配诊断脚本实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 编写诊断脚本验证 DEIMv2-OBB 匈牙利匹配的正确性，输出 Q1-Q3 统计和可视化

**Architecture:** 单脚本 `test/diagnose_hungarian_matching.py`，加载 density_020 checkpoint，直接调用 matcher 复现匹配过程，收集 assignment/cost/IoU 数据，用 matplotlib 生成可视化图表

**Tech Stack:** Python 3.12, PyTorch, matplotlib, numpy

## Global Constraints

- 只读分析，不修改任何模型参数或训练代码
- 输出目录 `test/outputs/matching_diag/`
- 脚本放在 `test/diagnose_hungarian_matching.py`
- Python 环境：`/home/cx/apps/miniconda3/envs/deimv2/`

---

## File Structure

```
test/
  diagnose_hungarian_matching.py    # 主诊断脚本（唯一新增文件）
  outputs/matching_diag/             # 输出目录（自动创建）
    per_image/                       # Q1 可视化子目录
    cost_distribution.png            # Q2 代价直方图
    cost_heatmap.png                 # Q2 代价热力图
    score_iou_scatter.png            # Q3 散点图
    score_iou_boxplot.png            # Q3 箱线图
    matching_report.txt              # 全局汇总
```

**依存关系**：
- `engine/deim/matcher.py:92` — `HungarianMatcher.forward()` 返回 `{"indices": [(qi, gj), ...]}`
- `engine/deim/obb_ops.py` — `batch_probiou(det_tensor, gt_tensor)` 计算 ProbIoU
- `engine/core/__init__.py` — `YAMLConfig` 加载配置
- `engine/solver/__init__.py` — `TASKS` 创建 solver
- Checkpoint: `outputs/synthetic_exp_020/last.pth`
- Config: `configs/custom_obb/synthetic_exp_020.yml`

---

### Task 1: 脚本骨架 — 模型加载与前向推理

**Files:**
- Create: `test/diagnose_hungarian_matching.py`

**Interfaces:**
- Produces: `load_model_and_data(config_path, ckpt_path)` → `(model, postprocessor, val_loader, matcher)`

- [ ] **Step 1: 写入脚本骨架**

```python
"""DEIMv2-OBB 匈牙利匹配诊断脚本。

加载 density_020 模型，在验证集上复现匈牙利匹配，诊断：
  Q1: 一对一匹配率
  Q2: 代价函数区分度
  Q3: 分类分数与 IoU 相关性

用法: python test/diagnose_hungarian_matching.py
"""

import os, sys, argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.core import YAMLConfig
from engine.solver import TASKS

OUTPUT_DIR = os.path.join(ROOT, "test", "outputs", "matching_diag")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "per_image"), exist_ok=True)


def load_model_and_data(config_path, ckpt_path):
    """加载模型、验证数据加载器和 matcher"""
    cfg = YAMLConfig(config_path)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    solver.model.load_state_dict(ckpt["ema"]["module"])
    model = solver.model.cuda().eval()
    postprocessor = solver.postprocessor
    val_loader = solver.val_dataloader

    matcher = solver.criterion.matcher
    return model, postprocessor, val_loader, matcher, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/synthetic_exp_020/last.pth"))
    parser.add_argument("--config", default=os.path.join(ROOT, "configs/custom_obb/synthetic_exp_020.yml"))
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    print("Loading model...")
    model, postprocessor, val_loader, matcher, cfg = load_model_and_data(args.config, args.ckpt)
    print(f"Model loaded. Val images: {len(val_loader.dataset)}")

    all_data = collect_matching_data(model, postprocessor, val_loader, matcher, args.max_images)

    q1_stats = analyze_q1(all_data, output_dir=OUTPUT_DIR)
    q2_stats = analyze_q2(all_data, output_dir=OUTPUT_DIR)
    q3_stats = analyze_q3(all_data, output_dir=OUTPUT_DIR)

    generate_report(all_data, q1_stats, q2_stats, q3_stats, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证骨架**

```bash
cd /mnt/d/cx/thired/deimv2_daod && python test/diagnose_hungarian_matching.py
```

预期：`NameError: name 'collect_matching_data' is not defined`（函数未实现，加载逻辑验证通过）

---

### Task 2: 匹配数据收集

**Files:**
- Modify: `test/diagnose_hungarian_matching.py`（在 `main()` 前插入）

- [ ] **Step 1: 实现 `collect_matching_data`**

```python
def collect_matching_data(model, postprocessor, val_loader, matcher, max_images):
    """验证集逐图前向 + 匈牙利匹配，收集诊断数据。

    Returns:
        list[dict]: 每张图一个 dict:
            image_idx, scores, labels, pred_boxes, gt_boxes, gt_labels,
            indices, ious, cost_class, cost_bbox, cost_probiou, cost_chamfer, total_cost
    """
    from engine.deim.obb_ops import batch_probiou
    from engine.deim.chamfer_cost import chamfer_cost_obb

    device = next(model.parameters()).device
    all_data = []
    processed = 0

    for samples, targets in val_loader:
        if processed >= max_images:
            break

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(samples)
            outputs_main = {k: v for k, v in outputs.items() if "aux" not in k}
            matcher_result = matcher(outputs_main, targets, epoch=0)
            indices_list = matcher_result["indices"]

        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs_main, orig_sizes)

        for i, (res, tgt, indices) in enumerate(zip(results, targets, indices_list)):
            if processed >= max_images:
                break
            processed += 1

            pred_boxes = res["boxes"].cpu().numpy()
            pred_scores = res["scores"].cpu().numpy()
            pred_labels = res["labels"].cpu().numpy()
            gt_boxes = tgt["boxes"].cpu().numpy()
            gt_labels = tgt["labels"].cpu().numpy()

            ow, oh = orig_sizes[i].cpu().numpy()
            if len(gt_boxes) > 0:
                gt_boxes[:, 0] *= ow; gt_boxes[:, 1] *= oh
                gt_boxes[:, 2] *= ow; gt_boxes[:, 3] *= oh

            qi, gi = indices
            if len(qi) > 0:
                det_t = torch.tensor(pred_boxes[qi.cpu().numpy()], dtype=torch.float32)
                gt_t = torch.tensor(gt_boxes[gi.cpu().numpy()], dtype=torch.float32)
                ious = batch_probiou(det_t, gt_t).numpy()
            else:
                ious = np.array([])

            # --- 代价矩阵计算 ---
            out_prob = torch.sigmoid(outputs_main["pred_logits"][i:i+1].flatten(0, 1))
            out_bbox = outputs_main["pred_boxes"][i:i+1].flatten(0, 1)
            tgt_bbox_t = tgt["boxes"].unsqueeze(0)
            tgt_bbox_flat = tgt_bbox_t.flatten(0, 1)
            tgt_ids = tgt["labels"]

            cost_class = (-out_prob[:, tgt_ids]).cpu().numpy()
            cost_bbox   = torch.cdist(out_bbox[:, :4], tgt_bbox_flat[:, :4], p=1).cpu().numpy()
            cp = -batch_probiou(out_bbox, tgt_bbox_flat, eps=1e-8).unsqueeze(0).squeeze(0)
            cost_probiou = cp.cpu().numpy()
            cost_chamfer = chamfer_cost_obb(out_bbox, tgt_bbox_flat).cpu().numpy()
            total_cost = (2.0 * cost_class + 5.0 * cost_bbox + 5.0 * cost_chamfer + 2.0 * cost_probiou)

            all_data.append({
                "image_idx": processed - 1,
                "scores": pred_scores, "labels": pred_labels, "pred_boxes": pred_boxes,
                "gt_boxes": gt_boxes, "gt_labels": gt_labels, "indices": indices, "ious": ious,
                "cost_class": cost_class, "cost_bbox": cost_bbox,
                "cost_probiou": cost_probiou, "cost_chamfer": cost_chamfer, "total_cost": total_cost,
            })

    print(f"Collected matching data for {len(all_data)} images")
    return all_data
```

- [ ] **Step 2: 验证**

```bash
cd /mnt/d/cx/thired/deimv2_daod && python test/diagnose_hungarian_matching.py
```

预期：收集 100 张图数据后 `NameError: name 'analyze_q1' is not defined`

---

### Task 3: Q1 分析

**Files:**
- Modify: `test/diagnose_hungarian_matching.py`（插入 `analyze_q1` 和 `draw_q1_overlay`）

- [ ] **Step 1: 实现**

```python
def analyze_q1(all_data, output_dir):
    """Q1: per-GT query 数分布 + 叠加可视化"""
    per_img_dir = os.path.join(output_dir, "per_image")

    gt_query_counts = []
    gt_zero, gt_multi, total_gt = 0, 0, 0

    for img_data in all_data:
        indices = img_data["indices"]
        qi, gi = indices
        gi_arr = gi.cpu().numpy()
        n_gt = len(img_data["gt_boxes"])
        per_gt = np.zeros(n_gt, dtype=int)
        for g in gi_arr:
            per_gt[g] += 1
        gt_query_counts.extend(per_gt.tolist())
        total_gt += n_gt
        gt_zero += (per_gt == 0).sum()
        gt_multi += (per_gt > 1).sum()

    # 柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    unique, counts = np.unique(gt_query_counts, return_counts=True)
    colors = ["red" if x == 0 else "orange" if x > 1 else "steelblue" for x in unique]
    bars = ax.bar(unique, counts, color=colors)
    ax.set_xlabel("Queries per GT"); ax.set_ylabel("Number of GTs")
    ax.set_title(f"Q1: Queries per GT ({total_gt} GTs)")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2., b.get_height(), str(c), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, "q1_per_gt_queries.png"), dpi=150); plt.close(fig)

    # 叠加图：选 4 张典型图（按问题严重度排序）
    per_img_stats = []
    for i, img_data in enumerate(all_data):
        qi, gi = img_data["indices"]; gi_arr = gi.cpu().numpy()
        n_gt = len(img_data["gt_boxes"])
        per_gt = np.zeros(n_gt, dtype=int)
        for g in gi_arr: per_gt[g] += 1
        per_img_stats.append((i, (per_gt == 0).sum(), (per_gt > 1).sum()))
    per_img_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
    for sel_idx in [per_img_stats[0][0], per_img_stats[-1][0],
                     per_img_stats[len(per_img_stats)//3][0], per_img_stats[2*len(per_img_stats)//3][0]]:
        draw_q1_overlay(all_data[sel_idx], sel_idx, per_img_dir)

    print(f"\n[Q1] Total GTs: {total_gt}, 0-match: {gt_zero} ({100*gt_zero/total_gt:.1f}%), "
          f"2+match: {gt_multi} ({100*gt_multi/total_gt:.1f}%), avg q/GT: {np.mean(gt_query_counts):.2f}")
    return {"total_gt": total_gt, "zero": gt_zero, "multi": gt_multi, "avg_q": np.mean(gt_query_counts)}


def draw_q1_overlay(img_data, img_idx, output_dir):
    """GT(绿)+匹配(红)+未匹配(黄)叠加图"""
    from PIL import Image, ImageDraw
    from engine.deim.obb_geometry import xywhr_to_xyxyxyxy

    pred_boxes, pred_scores = img_data["pred_boxes"], img_data["scores"]
    gt_boxes, indices = img_data["gt_boxes"], img_data["indices"]
    qi, gi = indices; qi_set = set(qi.cpu().numpy().tolist()); ious = img_data["ious"]

    img_pil = Image.new("RGB", (256, 256), color=(128, 128, 128)); draw = ImageDraw.Draw(img_pil)

    gt_verts = xywhr_to_xyxyxyxy(torch.tensor(gt_boxes))
    for j, verts in enumerate(gt_verts):
        pts = [(float(verts[k, 0]), float(verts[k, 1])) for k in range(4)]
        draw.polygon(pts, outline=(0, 255, 0), width=2)
        n = (gi.cpu().numpy() == j).sum()
        draw.text((float(gt_boxes[j, 0]) - 10, float(gt_boxes[j, 1]) - 20), f"GT{j}:{n}q", fill=(0, 255, 0))

    for k, (q_i, g_i) in enumerate(zip(qi.cpu().numpy(), gi.cpu().numpy())):
        iou_val = ious[k] if k < len(ious) else 0.5
        color = (int(255 * iou_val), 0, 0)
        verts = xywhr_to_xyxyxyxy(torch.tensor(pred_boxes[q_i:q_i+1]))
        pts = [(float(verts[0, j, 0]), float(verts[0, j, 1])) for j in range(4)]
        draw.polygon(pts, outline=color, width=1)

    for q_i in range(300):
        if q_i not in qi_set and pred_scores[q_i] > 0.1:
            verts = xywhr_to_xyxyxyxy(torch.tensor(pred_boxes[q_i:q_i+1]))
            pts = [(float(verts[0, j, 0]), float(verts[0, j, 1])) for j in range(4)]
            draw.polygon(pts, outline=(255, 255, 0), width=1)

    img_pil.save(os.path.join(output_dir, f"img{img_idx:02d}_q1_overlay.png"))
```

- [ ] **Step 2: 验证**

```bash
cd /mnt/d/cx/thired/deimv2_daod && python test/diagnose_hungarian_matching.py
```

预期：生成 `q1_per_gt_queries.png` + 4 张 overlay，终端打印 Q1 统计

---

### Task 4: Q2 分析

**Files:**
- Modify: `test/diagnose_hungarian_matching.py`（插入 `analyze_q2`）

- [ ] **Step 1: 实现**

```python
def analyze_q2(all_data, output_dir):
    """Q2: 代价分量直方图 + 热力图"""
    cost_names = ["class", "bbox", "chamfer", "probiou"]
    matched, unmatched = {k: [] for k in cost_names}, {k: [] for k in cost_names}

    for img_data in all_data:
        qi = img_data["indices"][0]; qi_arr = qi.cpu().numpy()
        gi_arr = img_data["indices"][1].cpu().numpy()
        matched_set = set(qi_arr.tolist())

        for cn in cost_names:
            cmat = img_data[f"cost_{cn}"]
            for q, g in zip(qi_arr, gi_arr):
                matched[cn].append(float(cmat[q, g]))
            for q in range(300):
                if q not in matched_set:
                    unmatched[cn].append(float(np.min(cmat[q])))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, cn in zip(axes.flat, cost_names):
        ax.hist(matched[cn], bins=30, alpha=0.6, color="steelblue", label="Matched", density=True)
        ax.hist(unmatched[cn], bins=30, alpha=0.6, color="salmon", label="Unmatched", density=True)
        ax.set_xlabel(f"{cn} cost"); ax.set_ylabel("Density"); ax.set_title(f"Cost: {cn}"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, "cost_distribution.png"), dpi=150); plt.close(fig)

    # 热力图（第一张图）
    total_cost = all_data[0]["total_cost"]
    top_k = min(50, 300)
    inds = np.argpartition(total_cost, top_k, axis=0)[:top_k, :]
    selected = np.unique(inds.flatten())[:100]
    cost_subset = total_cost[selected, :]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cost_subset, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("GT"); ax.set_ylabel("Query (subset)"); ax.set_title(f"Cost Heatmap ({len(selected)} queries)")
    plt.colorbar(im, ax=ax, label="Total Cost")
    qi_arr, gi_arr = all_data[0]["indices"][0].cpu().numpy(), all_data[0]["indices"][1].cpu().numpy()
    for g in range(cost_subset.shape[1]):
        bx = np.argmin(cost_subset[:, g])
        ax.scatter(g, bx, marker="*", color="white", s=100, edgecolors="black")
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, "cost_heatmap.png"), dpi=150); plt.close(fig)

    sep = {}
    for cn in cost_names:
        m, u = np.mean(matched[cn]), np.mean(unmatched[cn])
        sep[cn] = u / m if m > 0 else 0
        print(f"  {cn:12s}: matched={m:.3f}, unmatched={u:.3f}, ratio={sep[cn]:.1f}x")
    return sep
```

- [ ] **Step 2: 验证**

```bash
cd /mnt/d/cx/thired/deimv2_daod && python test/diagnose_hungarian_matching.py
```

预期：生成 `cost_distribution.png` + `cost_heatmap.png`

---

### Task 5: Q3 分析

**Files:**
- Modify: `test/diagnose_hungarian_matching.py`（插入 `analyze_q3`）

- [ ] **Step 1: 实现**

```python
def analyze_q3(all_data, output_dir):
    """Q3: 分数-IoU 散点图 + 分桶箱线图"""
    from scipy.stats import pearsonr, spearmanr

    all_scores, all_ious = [], []
    for img_data in all_data:
        qi = img_data["indices"][0].cpu().numpy()
        ious = img_data["ious"]; scores = img_data["scores"]
        for k, q in enumerate(qi):
            all_scores.append(float(scores[q]))
            if k < len(ious): all_ious.append(float(ious[k]))
    all_scores, all_ious = np.array(all_scores), np.array(all_ious)

    r, p = pearsonr(all_ious, all_scores)
    rho, pr = spearmanr(all_ious, all_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_ious, all_scores, alpha=0.3, s=10, c="steelblue")
    z = np.polyfit(all_ious, all_scores, 1)
    ax.plot(np.linspace(0, 1, 100), np.poly1d(z)(np.linspace(0, 1, 100)), "r--", label=f"r={r:.3f}")
    ax.set_xlabel("ProbIoU"); ax.set_ylabel("Score")
    ax.set_title(f"Q3: Score vs IoU (r={r:.3f}, ρ={rho:.3f})"); ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, "score_iou_scatter.png"), dpi=150); plt.close(fig)

    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    binned = [all_scores[(all_ious >= lo) & (all_ious < hi)] for lo, hi in bins]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(binned, labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], plt.cm.Blues([0.3, 0.45, 0.6, 0.75, 0.9])):
        patch.set_facecolor(c)
    for i, s in enumerate(binned):
        if len(s) > 0: ax.text(i+1, np.max(s)+0.02, f"n={len(s)}", ha="center", fontsize=8)
    ax.set_xlabel("ProbIoU bin"); ax.set_ylabel("Score"); ax.set_title("Q3: Score by IoU Bin")
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, "score_iou_boxplot.png"), dpi=150); plt.close(fig)

    print(f"\n[Q3] r={r:.4f} (p={p:.3g}), ρ={rho:.4f}, n={len(all_ious)}")
    return {"r": r, "rho": rho, "n": len(all_ious)}
```

- [ ] **Step 2: 验证**

```bash
cd /mnt/d/cx/thired/deimv2_daod && python test/diagnose_hungarian_matching.py
```

预期：生成 `score_iou_scatter.png` + `score_iou_boxplot.png`

---

### Task 6: 全局汇总报告

**Files:**
- Modify: `test/diagnose_hungarian_matching.py`（插入 `generate_report`）

- [ ] **Step 1: 实现**

```python
def generate_report(all_data, q1, q2, q3, output_dir):
    path = os.path.join(output_dir, "matching_report.txt")
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  匈牙利匹配诊断报告 (density_020)\n")
        f.write(f"  图像数: {len(all_data)}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Q1: One-to-Many\n" + "-" * 30 + "\n")
        f.write(f"  Total GTs:           {q1['total_gt']}\n")
        n_one = q1['total_gt'] - q1['zero'] - q1['multi']
        f.write(f"  GT with 0 matches:   {q1['zero']} ({100*q1['zero']/q1['total_gt']:.1f}%)\n")
        f.write(f"  GT with 1 match:     {n_one} ({100*n_one/q1['total_gt']:.1f}%)\n")
        f.write(f"  GT with 2+ matches:  {q1['multi']} ({100*q1['multi']/q1['total_gt']:.1f}%)\n")
        f.write(f"  Avg queries/GT:      {q1['avg_q']:.2f}\n")
        f.write(f"  Verdict: {'PASS' if n_one/q1['total_gt']>0.85 else 'FLAG'}\n\n")

        f.write("Q2: Cost Discriminability\n" + "-" * 30 + "\n")
        for cn, ratio in q2.items():
            f.write(f"  {cn:12s} ratio: {ratio:.1f}x  [{'OK' if ratio>2 else 'FLAG'}]\n")
        f.write("\n")

        f.write("Q3: Score-IoU Correlation\n" + "-" * 30 + "\n")
        f.write(f"  Pearson r:   {q3['r']:.4f}\n  Spearman ρ:  {q3['rho']:.4f}\n  Pairs:       {q3['n']}\n")
        f.write(f"  Verdict: {'PASS' if abs(q3['r'])>0.2 else 'FLAG'}\n\n")

        f.write("=" * 60 + "\nCONCLUSION\n" + "=" * 60 + "\n")
        all_pass = True
        if n_one / q1['total_gt'] <= 0.85: f.write("  - Q1 flagged\n"); all_pass = False
        if min(q2.values()) <= 2: f.write("  - Q2 flagged\n"); all_pass = False
        if abs(q3['r']) <= 0.2: f.write("  - Q3 flagged\n"); all_pass = False
        if all_pass:
            f.write("  ✓ All pass. Matching OK → investigate decoder coupling.\n")
        else:
            f.write("  ✗ Matching issues found → fix matcher before decoder.\n")

    print(f"\nReport: {path}")
    with open(path) as fp: print(fp.read())
```

- [ ] **Step 2: 完整运行**

```bash
cd /mnt/d/cx/thired/deimv2_daod && python test/diagnose_hungarian_matching.py
```

---

### Task 7: 最终验证

- [ ] **Step 1: 运行 + 检查输出文件**

```bash
cd /mnt/d/cx/thired/deimv2_daod && python test/diagnose_hungarian_matching.py && ls -la test/outputs/matching_diag/
```

预期输出：
```
q1_per_gt_queries.png
cost_distribution.png
cost_heatmap.png
score_iou_scatter.png
score_iou_boxplot.png
matching_report.txt
per_image/img*_q1_overlay.png  (4 files)
```

- [ ] **Step 2: 人工审查图表**
  - Q1 柱状图：0-match <5%，multi-match <10%
  - Q2 直方图：蓝（matched）显著左偏于红（unmatched）
  - Q2 热力图：白星孤立，每列唯一最优
  - Q3 散点图：点云有左下→右上趋势
  - Q3 箱线图：中位数随 IoU 单调增
