"""
DEIMv2-OBB 匈牙利匹配诊断脚本
=================================

目的
----
加载 density_020 合成数据集上训练的模型，在验证集上复现训练时的匈牙利匹配过程，
诊断匹配质量是否正常。具体回答三个问题：

  Q1 (一对一匹配): 匈牙利匹配是否做到了每个 GT 恰好分配一个 query？
      理想情况下，300 个 query 中恰好有 N_GT 个被匹配，其余标记为"无目标"。
      如果有 GT 被 0 个 query 匹配（遗漏），或者被 2+ 个 query 匹配（重复），
      说明匹配算法或代价函数设计有问题。

  Q2 (代价函数区分度): matcher 的 4 维代价（class/bbox/chamfer/probiou）
      能否有效地区分"真正的正样本匹配"和"碰巧的负样本"？
      如果代价区分度不足，matcher 可能在噪声中随机匹配，导致训练不稳定。

  Q3 (分数-质量相关性): 被匹配 query 的分类分数是否与其 IoU 质量正相关？
      通俗地说——预测得越准的框，分数是不是越高？如果分数与 IoU 无关，
      说明分类 head 产出的分数不可靠，无法用于后处理筛选。

原理
----
DETR 的训练流程是：
  1. Decoder 输出 300 个 query 的预测结果（OBB + 分类 logits）
  2. 匈牙利匹配器 (HungarianMatcher) 计算 300×N_GT 的代价矩阵，
     用 linear_sum_assignment 找到最优的一对一匹配
  3. 匹配到的 query 作为正样本参与 loss 计算，未匹配的作为负样本

本脚本跳过 loss 计算，只在 inference 阶段调用 matcher 获取匹配结果，
然后对匹配质量做统计分析。

输入
----
  - 模型权重: outputs/synthetic_exp_020/last.pth
  - 配置文件: configs/custom_obb/synthetic_exp_020.yml
  - 验证集: density_020 的 100 张合成椭圆图（每图 20 GT）

输出 (test/outputs/matching_diag/)
----------------------------------
  q1_per_gt_queries.png    - Q1 柱状图：每个 GT 被几个 query 匹配
  per_image/img*_overlay.png - Q1 叠加可视化（绿=GT，红=匹配，黄=未匹配）
  cost_distribution.png    - Q2 代价分布直方图（4 个子图，每个代价分量）
  cost_heatmap.png         - Q2 代价矩阵热力图（首张图，白星=最优匹配）
  score_iou_scatter.png    - Q3 散点图：每个匹配对的 IoU vs 分类分数
  score_iou_boxplot.png    - Q3 箱线图：按 IoU 分桶的分数分布
  matching_report.txt      - 全局汇总报告

用法
----
  python test/diagnose_hungarian_matching.py
  python test/diagnose_hungarian_matching.py --max-images 10  # 快速测试
"""

# ============================================================================
# 导入和全局设置
# ============================================================================

import os, sys, argparse
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无 GUI 后端，适配远程/服务器环境
import matplotlib.pyplot as plt
from collections import defaultdict

# 将项目根目录加入 sys.path，以便导入 engine 下的模块
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.core import YAMLConfig  # YAML 配置加载器
from engine.solver import TASKS  # 任务工厂：根据配置创建 solver

# 所有诊断输出统一放在此目录下
OUTPUT_DIR = os.path.join(ROOT, "test", "outputs", "matching_diag")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "per_image"), exist_ok=True)


# ============================================================================
# 工具函数：加载模型和数据
# ============================================================================


def load_model_and_data(config_path, ckpt_path):
    """
    加载训练好的模型、验证数据加载器、以及匈牙利匹配器。

    参数:
        config_path (str): YAML 配置文件路径
        ckpt_path  (str): 模型权重文件路径 (.pth)

    返回:
        model           (nn.Module):    已加载权重并设为 eval 模式的模型
        postprocessor   (nn.Module):    后处理器（将 decoder 输出转为像素坐标预测）
        val_loader      (DataLoader):   验证集数据加载器
        matcher         (HungarianMatcher): 匈牙利匹配器（从 criterion 中取出）
        cfg             (YAMLConfig):   配置对象
    """
    # Step 1: 加载 YAML 配置，创建 solver（包含模型/dataloader/criterion 等）
    cfg = YAMLConfig(config_path)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()  # train() 会初始化 dataloader 和 criterion

    # Step 2: 加载模型权重（使用 EMA 版本，推理效果更好）
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    solver.model.load_state_dict(ckpt["ema"]["module"])
    model = solver.model.cuda().eval()

    # Step 3: 获取推理所需的组件
    postprocessor = solver.postprocessor
    val_loader = solver.val_dataloader

    # Step 4: 从 criterion 中提取 matcher（训练时用于匈牙利匹配的模块）
    matcher = solver.criterion.matcher

    return model, postprocessor, val_loader, matcher, cfg


# ============================================================================
# 阶段 1: 数据收集 —— 逐图前向推理 + 匈牙利匹配
# ============================================================================


def collect_matching_data(model, postprocessor, val_loader, matcher, max_images):
    """
    对验证集逐图执行前向推理和匈牙利匹配，收集所有诊断所需的原始数据。

    对每张图：
      1. 模型前向 → 得到 300 个 query 的预测 (OBB + 分类 logits)
      2. 后处理 → 将归一化坐标转为像素坐标
      3. 调用 matcher → 获得 300 query 与 N_GT 之间的最优匹配 assignment
      4. 手动计算代价矩阵 → class/bbox/chamfer/probiou 四维代价

    参数:
        model        (nn.Module): 模型
        postprocessor (nn.Module): 后处理器
        val_loader   (DataLoader): 验证数据加载器
        matcher      (HungarianMatcher): 匹配器
        max_images   (int): 最多处理多少张图

    返回:
        list[dict]: 每张图一个字典，包含:
            image_idx   (int):        图片序号
            scores      (ndarray):    (300,) 分类分数 (sigmoid 后)
            labels      (ndarray):    (300,) 预测类别
            pred_boxes  (ndarray):    (300,5) 预测 OBB (像素坐标 cx,cy,w,h,θ)
            gt_boxes    (ndarray):    (N_gt,5) GT OBB (像素坐标)
            gt_labels   (ndarray):    (N_gt,) GT 类别
            indices     (tuple):      (qi, gi) — qi=被选中的 query 索引, gi=对应的 GT 索引
            ious        (ndarray):    (len(qi), len(qi)) 匹配对的 ProbIoU 矩阵
            cost_class  (ndarray):    (300, N_gt) 分类代价矩阵
            cost_bbox   (ndarray):    (300, N_gt) bbox L1 代价矩阵
            cost_probiou(ndarray):    (300, N_gt) probiou 代价矩阵
            cost_chamfer(ndarray):    (300, N_gt) chamfer 代价矩阵
            total_cost  (ndarray):    (300, N_gt) 加权总代价矩阵
    """
    from engine.deim.obb_ops import batch_probiou  # Gaussian ProbIoU 计算
    from engine.deim.chamfer_cost import chamfer_cost_obb  # OBB 倒角距离

    device = next(model.parameters()).device
    all_data = []
    processed = 0

    for samples, targets in val_loader:
        if processed >= max_images:
            break

        # ── 数据转移到 GPU ──
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            # ── 模型前向推理 ──
            outputs = model(samples)
            # 取最后一层 decoder 的输出（排除 aux_outputs 辅助层）
            outputs_main = {k: v for k, v in outputs.items() if "aux" not in k}

            # ── 匈牙利匹配 ──
            # matcher.forward() 返回 {"indices": [...], ...}
            # indices 是一个 list，长度为 batch_size
            # 每个元素是一个元组 (query_idx, gt_idx)
            #   - query_idx: 被选中参与匹配的 query 索引 (shape: (N_matched,))
            #   - gt_idx: 对应的 GT 索引 (shape: (N_matched,))
            # 理论上 N_matched = min(300, N_gt)，实际应该 = N_gt
            matcher_result = matcher(outputs_main, targets, epoch=0)
            indices_list = matcher_result["indices"]

        # ── 后处理：将归一化预测转为像素坐标 ──
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs_main, orig_sizes)

        # ── 逐图收集数据 ──
        for i, (res, tgt, indices) in enumerate(zip(results, targets, indices_list)):
            if processed >= max_images:
                break
            processed += 1

            # --- 提取预测结果（像素坐标） ---
            pred_boxes = res["boxes"].cpu().numpy()  # (300, 5)  (cx,cy,w,h,θ) in pixels
            pred_scores = (
                res["scores"].cpu().numpy()
            )  # (300,)    分类分数（sigmoid 后）
            pred_labels = res["labels"].cpu().numpy()  # (300,)    预测类别 ID

            # --- 提取 GT（归一化坐标 → 像素坐标） ---
            gt_boxes = tgt["boxes"].cpu().numpy()  # (N_gt, 5) 归一化 (cx,cy,w,h,θ)
            gt_labels = tgt["labels"].cpu().numpy()  # (N_gt,)   GT 类别 ID

            ow, oh = orig_sizes[i].cpu().numpy()  # 原始图像宽高
            if len(gt_boxes) > 0:
                # 反归一化：cx,w 乘图像宽度，cy,h 乘图像高度，θ 不变
                gt_boxes[:, 0] *= ow
                gt_boxes[:, 1] *= oh
                gt_boxes[:, 2] *= ow
                gt_boxes[:, 3] *= oh

            # --- 计算每对匹配的 ProbIoU ---
            qi, gi = indices  # qi: matched query indices, gi: corresponding GT indices
            if len(qi) > 0:
                det_t = torch.tensor(pred_boxes[qi.cpu().numpy()], dtype=torch.float32)
                gt_t = torch.tensor(gt_boxes[gi.cpu().numpy()], dtype=torch.float32)
                # batch_probiou 返回 (N, N) 矩阵，对角元 = 每对匹配的 IoU
                ious = batch_probiou(det_t, gt_t).numpy()
            else:
                ious = np.array([])

            # ================================================================
            # ── 手动计算代价矩阵 (复现 matcher 内部逻辑) ──
            #
            # 匈牙利匹配器的代价矩阵由 4 个分量加权求和得到：
            #   total_cost = 2*cost_class + 5*cost_bbox + 5*cost_chamfer + 2*cost_probiou
            #
            # 权重来自 synthetic_exp_020.yml 的 matcher.weight_dict 配置。
            # ================================================================

            # 将当前图像的预测展平为 (300, dim) 格式
            out_prob = torch.sigmoid(
                outputs_main["pred_logits"][i : i + 1].flatten(0, 1)
            )  # (300, 15) 分类概率
            out_bbox = outputs_main["pred_boxes"][i : i + 1].flatten(
                0, 1
            )  # (300, 5) 预测 OBB

            # 展平 GT（注意 unsqueeze(0) 保持 batch 维度以便后续计算）
            tgt_bbox_t = tgt["boxes"].unsqueeze(0)  # (1, N_gt, 5)
            tgt_bbox_flat = tgt_bbox_t.flatten(0, 1)  # (N_gt, 5)
            tgt_ids = tgt["labels"]  # (N_gt,)

            # ① 分类代价：cost_class[i][j] = -P(query_i 预测为 GT_j 的类别)
            #    值越小（越负）= 该 query 对该类别的预测概率越高 → 匹配代价越低
            cost_class = (-out_prob[:, tgt_ids]).cpu().numpy()

            # ② Bbox L1 代价：cost_bbox[i][j] = ||pred_box[i, :4] - gt_box[j, :4]||₁
            #    只比较 (cx,cy,w,h)，不包含 θ（θ 由 probiou 单独衡量）
            cost_bbox = (
                torch.cdist(out_bbox[:, :4], tgt_bbox_flat[:, :4], p=1).cpu().numpy()
            )

            # ③ ProbIoU 代价：Gaussian 近似下的旋转 IoU，范围 [0,1]
            #    取负号使得 IoU 越大 → 代价越小
            cp = (
                -batch_probiou(out_bbox, tgt_bbox_flat, eps=1e-8)
                .unsqueeze(0)
                .squeeze(0)
            )
            cost_probiou = cp.cpu().numpy()

            # ④ Chamfer 代价：OBB 四个顶点之间的双向最近距离
            cost_chamfer = chamfer_cost_obb(out_bbox, tgt_bbox_flat).cpu().numpy()

            # ⑤ 加权总代价（与 matcher 内部权重一致）
            total_cost = (
                2.0 * cost_class
                + 5.0 * cost_bbox
                + 5.0 * cost_chamfer
                + 2.0 * cost_probiou
            )

            # ── 汇总保存 ──
            all_data.append(
                {
                    "image_idx": processed - 1,
                    "scores": pred_scores,
                    "labels": pred_labels,
                    "pred_boxes": pred_boxes,
                    "gt_boxes": gt_boxes,
                    "gt_labels": gt_labels,
                    "indices": indices,
                    "ious": ious,
                    "cost_class": cost_class,
                    "cost_bbox": cost_bbox,
                    "cost_probiou": cost_probiou,
                    "cost_chamfer": cost_chamfer,
                    "total_cost": total_cost,
                }
            )

    print(f"Collected matching data for {len(all_data)} images")
    return all_data


# ============================================================================
# 阶段 2: Q1 分析 —— 一对多匹配检测
# ============================================================================


def analyze_q1(all_data, output_dir):
    """
    Q1: 检查每个 GT 被多少个 query 匹配。

    匈牙利匹配的设计意图是每个 GT 恰好对应 1 个 query（一对一匹配）。
    如果某个 GT 被 0 个 query 匹配（遗漏），或 2+ 个 query 匹配（重复），
    说明匹配过程有问题。

    输出:
      - q1_per_gt_queries.png: 柱状图，X=每个 GT 被匹配的 query 数，Y=GT 数量
      - per_image/img*_q1_overlay.png: 选 4 张典型图的叠加可视化
          绿色粗框  = GT，标注 "GT{j}:{n}q" 表示该 GT 被 n 个 query 匹配
          红色框    = 匹配的预测框，颜色深浅编码 IoU（深红=高 IoU）
          黄色虚线框 = 未匹配的预测框（分数 >0.1 的）

    返回:
        dict: {"total_gt": int, "zero": int, "multi": int, "avg_q": float}
    """
    per_img_dir = os.path.join(output_dir, "per_image")

    # ── 全局统计：所有图的所有 GT ──
    gt_query_counts = []  # 每个 GT 被几个 query 匹配
    gt_zero, gt_multi, total_gt = 0, 0, 0

    for img_data in all_data:
        indices = img_data["indices"]
        qi, gi = indices  # qi=被选中的 query, gi=对应的 GT
        gi_arr = gi.cpu().numpy()
        n_gt = len(img_data["gt_boxes"])

        # 统计每个 GT 被匹配的次数
        per_gt = np.zeros(n_gt, dtype=int)
        for g in gi_arr:
            per_gt[g] += 1  # GT 索引 g 被匹配了一次

        gt_query_counts.extend(per_gt.tolist())
        total_gt += n_gt
        gt_zero += (per_gt == 0).sum()  # 没有被任何 query 匹配的 GT
        gt_multi += (per_gt > 1).sum()  # 被多个 query 匹配的 GT

    # ── 柱状图 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    unique, counts = np.unique(gt_query_counts, return_counts=True)
    # 颜色编码：红色=0 匹配(坏), 橙色=2+匹配(坏), 蓝色=1 匹配(好)
    colors = ["red" if x == 0 else "orange" if x > 1 else "steelblue" for x in unique]
    bars = ax.bar(unique, counts, color=colors)
    ax.set_xlabel("Queries per GT")
    ax.set_ylabel("Number of GTs")
    ax.set_title(f"Q1: Queries per GT ({total_gt} GTs)")
    for b, c in zip(bars, counts):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height(),
            str(c),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "q1_per_gt_queries.png"), dpi=150)
    plt.close(fig)

    # ── 叠加可视化：选 4 张典型图 ──
    # 选择策略：按"问题严重度"排序（先看遗漏最多的，再看重复最多的），
    # 选最差、最好、1/3 分位、2/3 分位各一张。
    per_img_stats = []
    for i, img_data in enumerate(all_data):
        qi, gi = img_data["indices"]
        gi_arr = gi.cpu().numpy()
        n_gt = len(img_data["gt_boxes"])
        per_gt = np.zeros(n_gt, dtype=int)
        for g in gi_arr:
            per_gt[g] += 1
        per_img_stats.append((i, (per_gt == 0).sum(), (per_gt > 1).sum()))

    per_img_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
    for sel_idx in [
        per_img_stats[0][0],
        per_img_stats[-1][0],
        per_img_stats[len(per_img_stats) // 3][0],
        per_img_stats[2 * len(per_img_stats) // 3][0],
    ]:
        draw_q1_overlay(all_data[sel_idx], sel_idx, per_img_dir)

    print(
        f"\n[Q1] Total GTs: {total_gt}, 0-match: {gt_zero} ({100*gt_zero/total_gt:.1f}%), "
        f"2+match: {gt_multi} ({100*gt_multi/total_gt:.1f}%), avg q/GT: {np.mean(gt_query_counts):.2f}"
    )
    return {
        "total_gt": total_gt,
        "zero": gt_zero,
        "multi": gt_multi,
        "avg_q": np.mean(gt_query_counts),
    }


def draw_q1_overlay(img_data, img_idx, output_dir):
    """
    绘制单张图的 Q1 叠加可视化。

    - 灰色背景（256×256 纯色）
    - 绿色粗框 = GT（OBB 外接矩形）
      标注 "GT{j}:{n}q" 表示该 GT 被 n 个 query 匹配
    - 红色框 = 被匹配的预测（颜色深浅 = IoU 高低）
    - 黄色虚线框 = 未匹配但分数 >0.1 的预测（假阳性）
    """
    from PIL import Image, ImageDraw
    from engine.deim.obb_geometry import (
        xywhr_to_xyxyxyxy,
    )  # OBB (cx,cy,w,h,θ) → 4 个顶点

    pred_boxes, pred_scores = img_data["pred_boxes"], img_data["scores"]
    gt_boxes, indices = img_data["gt_boxes"], img_data["indices"]
    qi, gi = indices
    qi_set = set(qi.cpu().numpy().tolist())  # 被匹配的 query 索引集合
    ious = img_data["ious"]
    # batch_probiou 返回 (N,N) 矩阵，对角元 = 每对匹配的 IoU
    ious_vals = np.diag(ious) if ious.ndim >= 2 else ious

    # 创建 256×256 灰色画布
    img_pil = Image.new("RGB", (256, 256), color=(128, 128, 128))
    draw = ImageDraw.Draw(img_pil)

    # ── 绘制 GT 框（绿色） ──
    gt_verts = xywhr_to_xyxyxyxy(torch.tensor(gt_boxes))
    for j, verts in enumerate(gt_verts):
        pts = [(float(verts[k, 0]), float(verts[k, 1])) for k in range(4)]
        draw.polygon(pts, outline=(0, 255, 0), width=2)
        n = (gi.cpu().numpy() == j).sum()  # 该 GT 被几个 query 匹配
        draw.text(
            (float(gt_boxes[j, 0]) - 10, float(gt_boxes[j, 1]) - 20),
            f"GT{j}:{n}q",
            fill=(0, 255, 0),
        )

    # ── 绘制匹配的预测框（红色，深浅 = IoU） ──
    for k, (q_i, g_i) in enumerate(zip(qi.cpu().numpy(), gi.cpu().numpy())):
        iou_val = float(ious_vals[k]) if k < len(ious_vals) else 0.5
        color = (int(255 * iou_val), 0, 0)  # IoU 越高 → 红色越深
        verts = xywhr_to_xyxyxyxy(torch.tensor(pred_boxes[q_i : q_i + 1]))
        pts = [(float(verts[0, j, 0]), float(verts[0, j, 1])) for j in range(4)]
        draw.polygon(pts, outline=color, width=3)

    # ── 绘制未匹配但分数 >0.1 的预测（黄色，假阳性） ──
    for q_i in range(300):
        if q_i not in qi_set and pred_scores[q_i] > 0.1:
            verts = xywhr_to_xyxyxyxy(torch.tensor(pred_boxes[q_i : q_i + 1]))
            pts = [(float(verts[0, j, 0]), float(verts[0, j, 1])) for j in range(4)]
            draw.polygon(pts, outline=(255, 255, 0), width=1)

    img_pil.save(os.path.join(output_dir, f"img{img_idx:02d}_q1_overlay.png"))


# ============================================================================
# 阶段 3: Q2 分析 —— 代价函数区分度
# ============================================================================


def analyze_q2(all_data, output_dir):
    """
    Q2: 评估匹配代价函数能否有效区分「真正的正样本」和「背景噪声」。

    对每张图的每个 GT，matcher 通过代价矩阵找到一个最优 query。
    一个好的代价函数应该满足：
      - matched costs ≪ unmatched costs（被选中的 query 代价远小于其他 query）
      - 两者分布不重叠或重叠很少

    我们分别分析 4 个代价分量：class, bbox, chamfer, probiou。

    输出:
      - cost_distribution.png: 4 子图直方图
          蓝色 = matched query 的代价分布
          红色 = unmatched query 中最小代价的分布
          X 轴 = 代价，Y 轴 = 密度
      - cost_heatmap.png: 首张图的代价矩阵热力图
          X = GT 索引，Y = 选中的 query 子集
          颜色 = 总代价（越深=代价越高=匹配越差）
          白色星号 = 每个 GT 的最优匹配位置

    返回:
        dict: {cost_name: separation_ratio}
    """
    cost_names = ["class", "bbox", "chamfer", "probiou"]
    matched, unmatched = {k: [] for k in cost_names}, {k: [] for k in cost_names}

    for img_data in all_data:
        qi = img_data["indices"][0]
        qi_arr = qi.cpu().numpy()  # 被匹配的 query 索引
        gi_arr = img_data["indices"][1].cpu().numpy()  # 对应的 GT 索引
        matched_set = set(qi_arr.tolist())  # 快速查找：哪些 query 被匹配了

        for cn in cost_names:
            cmat = img_data[f"cost_{cn}"]  # 代价矩阵 (300, N_gt)
            # matched: 提取 assignment 位置的代价值
            for q, g in zip(qi_arr, gi_arr):
                matched[cn].append(float(cmat[q, g]))
            # unmatched: 对每个未匹配 query，取它对所有 GT 的最小代价
            for q in range(300):
                if q not in matched_set:
                    unmatched[cn].append(float(np.min(cmat[q])))

    # ── 代价分量直方图 ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, cn in zip(axes.flat, cost_names):
        ax.hist(
            matched[cn],
            bins=30,
            alpha=0.6,
            color="steelblue",
            label="Matched",
            density=True,
        )
        ax.hist(
            unmatched[cn],
            bins=30,
            alpha=0.6,
            color="salmon",
            label="Unmatched",
            density=True,
        )
        ax.set_xlabel(f"{cn} cost")
        ax.set_ylabel("Density")
        ax.set_title(f"Cost: {cn}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cost_distribution.png"), dpi=150)
    plt.close(fig)

    # ── 代价矩阵热力图 ──
    # 选第一张图，只显示代价较低的 query（每列 top-50 的并集，最多 100 个）
    total_cost = all_data[0]["total_cost"]  # (300, N_gt)
    top_k = min(50, 300)
    inds = np.argpartition(total_cost, top_k, axis=0)[:top_k, :]
    selected = np.unique(inds.flatten())[:100]
    cost_subset = total_cost[selected, :]  # (≤100, N_gt)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cost_subset, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("GT")
    ax.set_ylabel("Query (subset)")
    ax.set_title(f"Cost Heatmap ({len(selected)} queries)")
    plt.colorbar(im, ax=ax, label="Total Cost")

    # 标记每列最优匹配（cost 最低的 query）
    qi_arr = all_data[0]["indices"][0].cpu().numpy()
    gi_arr = all_data[0]["indices"][1].cpu().numpy()
    for g in range(cost_subset.shape[1]):
        bx = np.argmin(cost_subset[:, g])
        ax.scatter(g, bx, marker="*", color="white", s=100, edgecolors="black")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cost_heatmap.png"), dpi=150)
    plt.close(fig)

    # ── 代价分布分离度 ──
    # 注意：matched 和 unmatched 是两组样本，不能用简单的"均值比"来比较。
    # 原因：
    #   1. 不同分量量纲不同（class 是概率负对数 ~[-1,0]，bbox 是 L1 距离 ~[0,∞]）
    #   2. 均值比在负值区间无意义（class 代价恒为负，matched mean=-0.4 时 ratio=0）
    #   3. 两个分布即使均值相同，方差/形状可能完全不同
    #
    # 改用 Wasserstein 距离（Earth Mover's Distance）：
    #   - 衡量将一个分布"搬运"成另一个分布所需的最小代价
    #   - 对正负值均正确处理
    #   - 值越大 = 两个分布越不同 = 区分度越好
    # 同时用 wasserstein / std(unmatched) 做归一化，便于跨分量比较。
    from scipy.stats import wasserstein_distance

    sep = {}
    for cn in cost_names:
        m_arr = np.array(matched[cn])
        u_arr = np.array(unmatched[cn])

        # Wasserstein 距离：衡量两个经验分布之间的最小搬运代价
        w_dist = wasserstein_distance(m_arr, u_arr)

        # 归一化：除以 unmatched 分布的标准差，使不同量纲的分量可比
        # w_dist/u_std ≈ 1.0 表示"搬运代价约等于分布自身的波动量级"
        u_std = np.std(u_arr)
        norm_sep = w_dist / u_std if u_std > 1e-8 else 0.0

        sep[cn] = norm_sep
        print(f"  {cn:12s}: wasserstein={w_dist:.3f}, unmatched_std={u_std:.3f}, "
              f"norm_sep={norm_sep:.2f}")

    return sep


# ============================================================================
# 阶段 4: Q3 分析 —— 分类分数与 IoU 相关性
# ============================================================================


def analyze_q3(all_data, output_dir):
    """
    Q3: 检查分类分数是否与匹配质量（IoU）正相关。

    理想情况：IoU 越高的匹配，其分类分数也应该越高。
    如果分数与 IoU 无关（Pearson r ≈ 0），说明分类 head 无法区分
    "预测得很好"和"预测得很差"的 query。

    输出:
      - score_iou_scatter.png: 散点图，每个点是一个匹配对
          X = ProbIoU，Y = 分类分数
          红色虚线 = 线性回归线
          标注 Pearson r 和 Spearman ρ
      - score_iou_boxplot.png: 箱线图
          X = IoU 分桶 (0-0.2, ..., 0.8-1.0)
          Y = 分类分数
          每个桶显示 mean/median/IQR，标注样本量

    返回:
        dict: {"r": float, "rho": float, "n": int}
    """
    from scipy.stats import pearsonr, spearmanr

    # ── 收集所有匹配对的 (IoU, score) 数据 ──
    all_scores, all_ious = [], []
    for img_data in all_data:
        qi = img_data["indices"][0].cpu().numpy()  # 被匹配的 query 索引
        ious = img_data["ious"]
        scores = img_data["scores"]

        # batch_probiou 返回 (N,N) 矩阵，对角元是每对的 IoU
        ious_vals = np.diag(ious) if len(ious.shape) == 2 else ious

        for k, q in enumerate(qi):
            all_scores.append(float(scores[q]))
            if k < len(ious_vals):
                all_ious.append(float(ious_vals[k]))

    all_scores = np.array(all_scores)
    all_ious = np.array(all_ious)

    # ── 统计检验 ──
    r, p = pearsonr(all_ious, all_scores)  # Pearson 线性相关系数
    rho, pr = spearmanr(all_ious, all_scores)  # Spearman 秩相关系数（对非线性鲁棒）

    # ── 散点图 ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_ious, all_scores, alpha=0.3, s=10, c="steelblue")
    # 线性回归拟合线
    z = np.polyfit(all_ious, all_scores, 1)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, np.poly1d(z)(x_line), "r--", linewidth=2, label=f"r={r:.3f}")
    ax.set_xlabel("ProbIoU")
    ax.set_ylabel("Score")
    ax.set_title(f"Q3: Score vs IoU (r={r:.3f}, rho={rho:.3f})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_iou_scatter.png"), dpi=150)
    plt.close(fig)

    # ── IoU 分桶箱线图 ──
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    binned = [all_scores[(all_ious >= lo) & (all_ious < hi)] for lo, hi in bins]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(binned, labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], plt.cm.Blues([0.3, 0.45, 0.6, 0.75, 0.9])):
        patch.set_facecolor(c)
    for i, s in enumerate(binned):
        if len(s) > 0:
            ax.text(i + 1, np.max(s) + 0.02, f"n={len(s)}", ha="center", fontsize=8)
    ax.set_xlabel("ProbIoU bin")
    ax.set_ylabel("Score")
    ax.set_title("Q3: Score by IoU Bin")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_iou_boxplot.png"), dpi=150)
    plt.close(fig)

    print(f"\n[Q3] r={r:.4f} (p={p:.3g}), rho={rho:.4f}, n={len(all_ious)}")
    return {"r": r, "rho": rho, "n": len(all_ious)}


# ============================================================================
# 阶段 5: 全局汇总报告
# ============================================================================


def generate_report(all_data, q1, q2, q3, output_dir):
    """
    汇总 Q1-Q3 的诊断结果，生成匹配质量报告。

    报告内容:
      - Q1: 一对一匹配统计（遗漏率、重复率）
      - Q2: 各代价分量的区分度
      - Q3: 分数-IoU 相关性
      - 综合判定: 匹配是否合格，下一步应排查 decoder 还是 matcher
    """
    path = os.path.join(output_dir, "matching_report.txt")
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  匈牙利匹配诊断报告 (density_020)\n")
        f.write(f"  图像数: {len(all_data)}\n")
        f.write("=" * 60 + "\n\n")

        # ── Q1 ──
        f.write("Q1: One-to-Many\n" + "-" * 30 + "\n")
        f.write(f"  Total GTs:           {q1['total_gt']}\n")
        n_one = q1["total_gt"] - q1["zero"] - q1["multi"]
        f.write(
            f"  GT with 0 matches:   {q1['zero']} ({100*q1['zero']/q1['total_gt']:.1f}%)\n"
        )
        f.write(f"  GT with 1 match:     {n_one} ({100*n_one/q1['total_gt']:.1f}%)\n")
        f.write(
            f"  GT with 2+ matches:  {q1['multi']} ({100*q1['multi']/q1['total_gt']:.1f}%)\n"
        )
        f.write(f"  Avg queries/GT:      {q1['avg_q']:.2f}\n")
        # 判定标准：一对一匹配率 > 85% 视为合格
        f.write(f"  Verdict: {'PASS' if n_one/q1['total_gt']>0.85 else 'FLAG'}\n\n")

        # ── Q2 ──
        f.write("Q2: Cost Discriminability\n" + "-" * 30 + "\n")
        f.write("  (normalized Wasserstein distance: >1.0 = good separation)\n")
        for cn, norm_sep in q2.items():
            flag = "OK" if norm_sep > 1.0 else "FLAG"
            f.write(
                f"  {cn:12s} norm_sep: {norm_sep:.2f}  [{flag}]\n"
            )
        f.write("\n")

        # ── Q3 ──
        f.write("Q3: Score-IoU Correlation\n" + "-" * 30 + "\n")
        f.write(f"  Pearson r:   {q3['r']:.4f}\n")
        f.write(f"  Spearman rho:{q3['rho']:.4f}\n")
        f.write(f"  Pairs:       {q3['n']}\n")
        # 判定标准：|r| > 0.2 视为有弱相关
        f.write(f"  Verdict: {'PASS' if abs(q3['r'])>0.2 else 'FLAG'}\n\n")

        # ── 综合判定 ──
        f.write("=" * 60 + "\nCONCLUSION\n" + "=" * 60 + "\n")
        all_pass = True
        if n_one / q1["total_gt"] <= 0.85:
            f.write("  - Q1 flagged: matching not one-to-one\n")
            all_pass = False
        if min(q2.values()) <= 1.0:
            f.write("  - Q2 flagged: cost function lacks discriminability\n")
            all_pass = False
        if abs(q3["r"]) <= 0.2:
            f.write("  - Q3 flagged: score unrelated to match quality\n")
            all_pass = False
        if all_pass:
            f.write("  All pass. Matching OK -> investigate decoder coupling.\n")
        else:
            f.write("  Matching issues found -> fix matcher before decoder.\n")

    print(f"\nReport: {path}")
    with open(path) as fp:
        print(fp.read())


# ============================================================================
# 主入口
# ============================================================================


def main():
    """命令行入口：加载模型 → 收集匹配数据 → Q1/Q2/Q3 分析 → 生成报告"""
    parser = argparse.ArgumentParser(description="DEIMv2-OBB 匈牙利匹配诊断")
    parser.add_argument(
        "--ckpt",
        default=os.path.join(ROOT, "outputs/synthetic_exp_020/last.pth"),
        help="模型权重文件路径",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs/custom_obb/synthetic_exp_020.yml"),
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="最多处理多少张验证集图片（默认 100）",
    )
    args = parser.parse_args()

    print("Loading model...")
    model, postprocessor, val_loader, matcher, cfg = load_model_and_data(
        args.config, args.ckpt
    )
    print(f"Model loaded. Val images: {len(val_loader.dataset)}")

    # 阶段 1: 收集匹配数据
    all_data = collect_matching_data(
        model, postprocessor, val_loader, matcher, args.max_images
    )

    # 阶段 2-4: 三组诊断分析
    q1_stats = analyze_q1(all_data, output_dir=OUTPUT_DIR)
    q2_stats = analyze_q2(all_data, output_dir=OUTPUT_DIR)
    q3_stats = analyze_q3(all_data, output_dir=OUTPUT_DIR)

    # 阶段 5: 生成汇总报告
    generate_report(all_data, q1_stats, q2_stats, q3_stats, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
