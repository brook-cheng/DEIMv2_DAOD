#!/usr/bin/env python3
"""
Pred vs GT OBB 几何分布对比工具。

从 DOTA 格式的标注/预测文件中提取 (cx,cy,w,h,θ)，
用直方图对比 pred 和 GT 在 w/h、w、h、angle 上的分布差异。
同时输出格式化的文本统计报告。

用法:
    python test/tool_obb_distribution_compare.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from engine.deim.obb_geometry import xyxyxyxy_to_xywhr
from tools.model_compare.obb_utils import parse_dota_line


def _poly_to_xywhr(poly8):
    poly = torch.tensor(poly8, dtype=torch.float32).reshape(1, 4, 2)
    return xyxyxyxy_to_xywhr(poly).numpy().flatten()


def load_boxes_from_dota_dir(dota_dir, is_gt=True):
    all_boxes = []
    fnames = [f for f in os.listdir(dota_dir) if f.endswith(".txt")]
    if not fnames:
        return np.zeros((0, 5))
    for fname in sorted(fnames):
        fpath = os.path.join(dota_dir, fname)
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if is_gt:
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    try:
                        poly8 = [float(x) for x in parts[:8]]
                    except ValueError:
                        continue
                else:
                    rec = parse_dota_line(line)
                    if rec is None:
                        continue
                    poly8 = rec["poly"]
                cx, cy, w, h, theta = _poly_to_xywhr(poly8)
                all_boxes.append([cx, cy, w, h, theta])
    return np.array(all_boxes, dtype=np.float32) if all_boxes else np.zeros((0, 5))


def compute_stats(boxes):
    if len(boxes) == 0:
        return {}
    w_h = boxes[:, 2] / np.clip(boxes[:, 3], a_min=1e-6, a_max=None)
    ang_deg = boxes[:, 4] * 180.0 / np.pi
    return {
        "n": len(boxes),
        "w_h_mean": float(w_h.mean()),
        "w_h_std": float(w_h.std()),
        "w_h_min": float(w_h.min()),
        "w_h_max": float(w_h.max()),
        "w_h_p25": float(np.percentile(w_h, 25)),
        "w_h_p50": float(np.percentile(w_h, 50)),
        "w_h_p75": float(np.percentile(w_h, 75)),
        "w_mean": float(boxes[:, 2].mean()),
        "w_std": float(boxes[:, 2].std()),
        "w_min": float(boxes[:, 2].min()),
        "w_max": float(boxes[:, 2].max()),
        "h_mean": float(boxes[:, 3].mean()),
        "h_std": float(boxes[:, 3].std()),
        "h_min": float(boxes[:, 3].min()),
        "h_max": float(boxes[:, 3].max()),
        "ang_mean_deg": float(ang_deg.mean()),
        "ang_std_deg": float(ang_deg.std()),
        "ang_min_deg": float(ang_deg.min()),
        "ang_max_deg": float(ang_deg.max()),
    }


def format_stats(label, stats):
    if not stats:
        return f"\n{label}: (empty)\n"
    lines = [
        f"\n{label} (n={stats['n']}):",
        f"  w/h:  mean={stats['w_h_mean']:.2f}  std={stats['w_h_std']:.2f}  "
        f"min={stats['w_h_min']:.1f}  p25={stats['w_h_p25']:.1f}  "
        f"p50={stats['w_h_p50']:.1f}  p75={stats['w_h_p75']:.1f}  max={stats['w_h_max']:.1f}",
        f"  w:    mean={stats['w_mean']:.1f}  std={stats['w_std']:.1f}  "
        f"min={stats['w_min']:.1f}  max={stats['w_max']:.1f}",
        f"  h:    mean={stats['h_mean']:.1f}  std={stats['h_std']:.1f}  "
        f"min={stats['h_min']:.1f}  max={stats['h_max']:.1f}",
        f"  θ:    mean={stats['ang_mean_deg']:.1f}°  std={stats['ang_std_deg']:.1f}°  "
        f"min={stats['ang_min_deg']:.1f}°  max={stats['ang_max_deg']:.1f}°",
    ]
    return "\n".join(lines)


def plot_distribution_comparison(gt_boxes, pred_boxes_list, model_names, output_path, bins=80):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    def _hist(ax, gt_val, pred_vals, labels, colors, xlabel, title, is_angle=False):
        ax.hist(gt_val, bins=bins, density=True, histtype="step",
                linewidth=2, color="black", label=f"GT (n={len(gt_val)})")
        for pred_val, name, c in zip(pred_vals, labels, colors):
            ax.hist(pred_val, bins=bins, density=True, histtype="step",
                    linewidth=1.5, color=c, linestyle="--",
                    label=f"{name} (n={len(pred_val)})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    gt_wh = gt_boxes[:, 2] / np.clip(gt_boxes[:, 3], a_min=1e-6, a_max=None)
    pred_wh_list = [p[:, 2] / np.clip(p[:, 3], a_min=1e-6, a_max=None) for p in pred_boxes_list]
    _hist(axes[0,0], gt_wh, pred_wh_list, model_names, colors, "w / h", "Aspect ratio (w/h)")

    _hist(axes[0,1], gt_boxes[:,2], [p[:,2] for p in pred_boxes_list],
          model_names, colors, "w (pixels)", "Width")
    _hist(axes[1,0], gt_boxes[:,3], [p[:,3] for p in pred_boxes_list],
          model_names, colors, "h (pixels)", "Height")

    gt_ang = gt_boxes[:, 4] * 180.0 / np.pi
    pred_ang_list = [p[:, 4] * 180.0 / np.pi for p in pred_boxes_list]
    _hist(axes[1,1], gt_ang, pred_ang_list, model_names, colors,
          "θ (degrees)", "Angle", is_angle=True)

    fig.suptitle("OBB Geometry Distribution: GT vs Pred", fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    GT_DOTA_DIR = "./test/data/outputs/dlzdt_obb_compare_train/gt_dota"
    DET_DIRS = [
        "./test/data/outputs/dlzdt_sp_rep0_train",
        "./test/data/outputs/dlzdt_sp_rep1_train",
    ]
    MODEL_NAMES = ["DEIMv2-SP-Rep0", "DEIMv2-SP-Rep1"]
    OUTPUT_PNG = "./test/data/outputs/obb_distribution_compare.png"
    OUTPUT_TXT = "./test/data/outputs/obb_distribution_compare.txt"

    report = []
    report.append("=" * 60)
    report.append("OBB Geometry Distribution Report")
    report.append("=" * 60)
    report.append(f"GT dir: {GT_DOTA_DIR}")

    print("Loading GT...")
    gt_boxes = load_boxes_from_dota_dir(GT_DOTA_DIR, is_gt=True)
    gt_stats = compute_stats(gt_boxes)
    s = format_stats("GT", gt_stats)
    print(s.strip())
    report.append(s)

    pred_boxes_list = []
    pred_stats_list = []
    for det_dir, name in zip(DET_DIRS, MODEL_NAMES):
        print(f"\nLoading pred: {name}...")
        pred = load_boxes_from_dota_dir(det_dir, is_gt=False)
        pred_boxes_list.append(pred)
        stats = compute_stats(pred)
        pred_stats_list.append(stats)
        s = format_stats(f"Pred ({name})", stats)
        print(s.strip())
        report.append(s)
        report.append(f"  pred dir: {det_dir}")

    png = plot_distribution_comparison(gt_boxes, pred_boxes_list, MODEL_NAMES, OUTPUT_PNG)
    print(f"\nSaved: {png}")

    with open(OUTPUT_TXT, "w") as f:
        f.write("\n".join(report))
    print(f"Saved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
