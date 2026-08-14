#!/usr/bin/env python3
"""
OBB 几何分布对比工具
=====================

Overview
--------
Loads OBB boxes from multiple DOTA-format directories (GT + predictions),
extracts (cx, cy, w, h, θ) and compares distributions between prediction sets
and ground truth using:

- Overlaid histograms for w/h ratio, w, h, and angle (in degrees)
- Wasserstein distance (EMD) + Jensen-Shannon divergence per metric
- Annotated text statistics on the histogram PNG
- Paired scatter plots with Hungarian matching per image

Outputs a single multi-panel PNG and a text report for quick model ranking.

Entry Points
------------
Programmatic:
    ``plot_distribution_comparison(gt_boxes, pred_boxes_list, model_names, output_png)``
    — generate overlaid histogram comparison.

Script:
    Edit ``main()`` paths and run::

        python test/tool_obb_distribution_compare.py

Configuration (edit in-file)
-----------------------------
GT_DOTA_DIR   : str   — DOTA-format GT directory
DET_DIRS      : list  — list of prediction directories (one per model)
MODEL_NAMES   : list  — display names matching DET_DIRS order
OUTPUT_PNG    : str   — path for histogram PNG (scatter → *_scatter.png)
OUTPUT_TXT    : str   — path for text report

Metrics
-------
Each model gets a report line with:
    w/h W-distance, w/h JS-div, w W-distance, w JS-div,
    h W-distance, h JS-div, angle W-distance, angle JS-div

Lower is better (more similar to GT distribution).

Output Structure
----------------
OUTPUT_PNG                    # histogram comparison
OUTPUT_PNG (→ _scatter.png)  # matched-pair scatter plots per model
OUTPUT_TXT                    # text report with all metrics

Usage
-----
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
from itertools import cycle

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


def _wasserstein_1d(vals1, vals2, n_bins=80):
    """1-D Wasserstein distance (Earth Mover's Distance) between two samples."""
    combined = np.concatenate([vals1, vals2])
    edges = np.linspace(combined.min(), combined.max(), n_bins + 1)
    bin_w = edges[1] - edges[0]
    # probability masses (sum = 1)
    h1, _ = np.histogram(vals1, bins=edges, density=False)
    h2, _ = np.histogram(vals2, bins=edges, density=False)
    cdf1 = np.cumsum(h1) / len(vals1)
    cdf2 = np.cumsum(h2) / len(vals2)
    return np.sum(np.abs(cdf1 - cdf2)) * bin_w


def _js_divergence(vals1, vals2, n_bins=80):
    """Jensen-Shannon divergence between two samples (base-2, ∈ [0, 1])."""
    combined = np.concatenate([vals1, vals2])
    edges = np.linspace(combined.min(), combined.max(), n_bins + 1)
    h1, _ = np.histogram(vals1, bins=edges, density=False)
    h2, _ = np.histogram(vals2, bins=edges, density=False)
    p = h1 / len(vals1) + 1e-10  # probability masses
    q = h2 / len(vals2) + 1e-10
    m = (p + q) / 2
    return 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))


def compute_similarity(gt_boxes, pred_boxes):
    """Compute Wasserstein and JS between GT and pred for w/h, w, h, angle."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return {}
    gt_wh = gt_boxes[:, 2] / np.clip(gt_boxes[:, 3], a_min=1e-6, a_max=None)
    pr_wh = pred_boxes[:, 2] / np.clip(pred_boxes[:, 3], a_min=1e-6, a_max=None)
    return {
        "w_h_wasserstein": float(_wasserstein_1d(gt_wh, pr_wh)),
        "w_h_js": float(_js_divergence(gt_wh, pr_wh)),
        "w_wasserstein": float(_wasserstein_1d(gt_boxes[:, 2], pred_boxes[:, 2])),
        "w_js": float(_js_divergence(gt_boxes[:, 2], pred_boxes[:, 2])),
        "h_wasserstein": float(_wasserstein_1d(gt_boxes[:, 3], pred_boxes[:, 3])),
        "h_js": float(_js_divergence(gt_boxes[:, 3], pred_boxes[:, 3])),
        "ang_wasserstein": float(
            _wasserstein_1d(
                gt_boxes[:, 4] * 180 / np.pi, pred_boxes[:, 4] * 180 / np.pi
            )
        ),
        "ang_js": float(
            _js_divergence(gt_boxes[:, 4] * 180 / np.pi, pred_boxes[:, 4] * 180 / np.pi)
        ),
    }


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
    if "w_h_wasserstein" in stats:
        lines.append(
            f"  ── Distribution similarity vs GT ──\n"
            f"    w/h:   W={stats['w_h_wasserstein']:.2f}  JS={stats['w_h_js']:.3f}\n"
            f"    w:     W={stats['w_wasserstein']:.1f}  JS={stats['w_js']:.3f}\n"
            f"    h:     W={stats['h_wasserstein']:.1f}  JS={stats['h_js']:.3f}\n"
            f"    angle: W={stats['ang_wasserstein']:.1f}° JS={stats['ang_js']:.3f}"
        )
    return "\n".join(lines)


def plot_distribution_comparison(
    gt_boxes, pred_boxes_list, model_names, output_path, bins=80
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors = [
        tuple(c[:3]) for c in plt.cm.tab20(np.linspace(0, 1, max(len(model_names), 20)))
    ]

    def _hist(ax, gt_val, pred_vals, labels, colors, xlabel, title):
        ax.hist(
            gt_val,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color="black",
            label=f"GT (n={len(gt_val)})",
        )
        for pred_val, name, c in zip(pred_vals, labels, cycle(colors)):
            ax.hist(
                pred_val,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=c,
                linestyle="--",
                label=f"{name} (n={len(pred_val)})",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    def _add_similarity_text(ax, gt_val, pred_vals, model_names, colors, unit=""):
        y0 = 0.95
        for i, (pred_val, name, c) in enumerate(
            zip(pred_vals, model_names, cycle(colors))
        ):
            w = _wasserstein_1d(gt_val, pred_val)
            js = _js_divergence(gt_val, pred_val)
            line = f"{name}: W={w:.2f}{unit} JS={js:.3f}"
            ax.text(
                0.97,
                y0 - i * 0.08,
                line,
                transform=ax.transAxes,
                fontsize=7,
                color=c,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

    # ── w/h ──
    gt_wh = gt_boxes[:, 2] / np.clip(gt_boxes[:, 3], a_min=1e-6, a_max=None)
    pred_wh_list = [
        p[:, 2] / np.clip(p[:, 3], a_min=1e-6, a_max=None) for p in pred_boxes_list
    ]
    _hist(
        axes[0, 0],
        gt_wh,
        pred_wh_list,
        model_names,
        colors,
        "w / h",
        "Aspect ratio (w/h)",
    )
    _add_similarity_text(axes[0, 0], gt_wh, pred_wh_list, model_names, colors)

    # ── w ──
    _hist(
        axes[0, 1],
        gt_boxes[:, 2],
        [p[:, 2] for p in pred_boxes_list],
        model_names,
        colors,
        "w (pixels)",
        "Width",
    )
    _add_similarity_text(
        axes[0, 1],
        gt_boxes[:, 2],
        [p[:, 2] for p in pred_boxes_list],
        model_names,
        colors,
        unit=" px",
    )

    # ── h ──
    _hist(
        axes[1, 0],
        gt_boxes[:, 3],
        [p[:, 3] for p in pred_boxes_list],
        model_names,
        colors,
        "h (pixels)",
        "Height",
    )
    _add_similarity_text(
        axes[1, 0],
        gt_boxes[:, 3],
        [p[:, 3] for p in pred_boxes_list],
        model_names,
        colors,
        unit=" px",
    )

    # ── angle ──
    gt_ang = gt_boxes[:, 4] * 180.0 / np.pi
    pred_ang_list = [p[:, 4] * 180.0 / np.pi for p in pred_boxes_list]
    _hist(
        axes[1, 1], gt_ang, pred_ang_list, model_names, colors, "θ (degrees)", "Angle"
    )
    _add_similarity_text(
        axes[1, 1], gt_ang, pred_ang_list, model_names, colors, unit="°"
    )

    fig.suptitle("OBB Geometry Distribution: GT vs Pred", fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_comparison_scatter(
    gt_dota_dir, det_dirs, model_names, output_path, scatter_rate=0.8
):
    """w/h and angle scatter plots: pred vs GT, paired by Hungarian matching."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_diff_mod",
        os.path.join(os.path.dirname(__file__), "tool_obb_difference_analysis.py"),
    )
    diff_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diff_mod)

    gt_per_img = diff_mod.load_boxes_per_image(gt_dota_dir, is_gt=True)
    colors = [
        tuple(c[:3]) for c in plt.cm.tab20(np.linspace(0, 1, max(len(model_names), 20)))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for mi, (det_dir, name) in enumerate(zip(det_dirs, model_names)):
        c = colors[mi % len(colors)]
        pred_per_img = diff_mod.load_boxes_per_image(det_dir, is_gt=False)
        # Collect paired values via Hungarian matching
        gt_wh_vals, pr_wh_vals, gt_ang_vals, pr_ang_vals = [], [], [], []
        common = set(gt_per_img.keys()) & set(pred_per_img.keys())
        for stem in common:
            gt_list = gt_per_img[stem]
            pred_list = pred_per_img[stem]
            M, N = len(gt_list), len(pred_list)
            if M == 0 or N == 0:
                continue
            gt_arr = np.array(
                [[b[0], b[1], b[2], b[3], b[4]] for b in gt_list], dtype=np.float32
            )
            pr_arr = np.array(
                [[b[0], b[1], b[2], b[3], b[4]] for b in pred_list], dtype=np.float32
            )
            gt_t = torch.tensor(gt_arr)
            pr_t = torch.tensor(pr_arr)
            from engine.deim.obb_ops import batch_probiou
            from scipy.optimize import linear_sum_assignment

            iou = batch_probiou(gt_t, pr_t).numpy()
            cost = -iou
            gt_idx, pred_idx = linear_sum_assignment(cost)
            for g, p in zip(gt_idx, pred_idx):
                if iou[g, p] < 0.1:
                    continue
                gt_wh = gt_arr[g, 2] / max(gt_arr[g, 3], 1e-6)
                pr_wh = pr_arr[p, 2] / max(pr_arr[p, 3], 1e-6)
                gt_wh_vals.append(gt_wh)
                pr_wh_vals.append(pr_wh)
                gt_ang_vals.append(gt_arr[g, 4] * 180 / np.pi)
                pr_ang_vals.append(pr_arr[p, 4] * 180 / np.pi)

        if not gt_wh_vals:
            continue
        n = max(1, int(len(gt_wh_vals) * scatter_rate))  # downsample to 10%
        idx = np.random.RandomState(42).choice(len(gt_wh_vals), size=n, replace=False)
        gt_wh = np.array(gt_wh_vals)[idx]
        pr_wh = np.array(pr_wh_vals)[idx]
        gt_ang = np.array(gt_ang_vals)[idx]
        pr_ang = np.array(pr_ang_vals)[idx]

        # w/h scatter
        lim = max(gt_wh.max(), pr_wh.max()) * 1.1
        axes[0].scatter(gt_wh, pr_wh, s=4, alpha=0.4, color=c, label=f"{name} (n={n})")
        axes[0].plot([0, lim], [0, lim], "k--", linewidth=0.8)
        axes[0].set_xlim(0, lim)
        axes[0].set_ylim(0, lim)

        # angle scatter
        axes[1].scatter(
            gt_ang, pr_ang, s=4, alpha=0.4, color=c, label=f"{name} (n={n})"
        )
        axes[1].plot([0, 180], [0, 180], "k--", linewidth=0.8)
        axes[1].set_xlim(0, 180)
        axes[1].set_ylim(0, 180)

    axes[0].set_xlabel("GT w/h")
    axes[0].set_ylabel("Pred w/h")
    axes[0].set_title("w/h: Pred vs GT (Hungarian matched)")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("GT θ (°)")
    axes[1].set_ylabel("Pred θ (°)")
    axes[1].set_title("Angle: Pred vs GT (Hungarian matched)")
    axes[1].legend(fontsize=7)

    fig.suptitle("Per-box scatter: matched pred↔GT pairs", fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    GT_DOTA_DIR = "./test/data/outputs/dlzdt_obb_compare_val/gt_dota"
    DET_DIRS = [
        "./test/data/outputs/dlzdt_obb_compare_val/yolo_dota",
        "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep0",
        "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep0_shifted",
        "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep0_mangle",
        "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep0_offset_post",
        "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep1",
        "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep3",
        "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep3_afp",
        "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep3_fused",
    ]
    MODEL_NAMES = [
        "YOLO-OBB",
        "abl_rep0",
        "abl_rep0_shifted",
        "abl_rep0_mangle",
        "abl_rep0_offset_post",
        "abl_rep1",
        "abl_rep3",
        "abl_rep3_afp",
        "abl_rep3_fused",
    ]
    OUTPUT_PNG = "./test/data/outputs/dlzdt_obb_compare_val/obb_distribution_compare/obb_distribution_compare.png"
    OUTPUT_TXT = "./test/data/outputs/dlzdt_obb_compare_val/obb_distribution_compare/obb_distribution_compare.txt"

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
        print(f"\nLoading pred: {name}, from {det_dir}")
        pred = load_boxes_from_dota_dir(det_dir, is_gt=False)
        pred_boxes_list.append(pred)
        stats = compute_stats(pred)
        sim = compute_similarity(gt_boxes, pred)
        stats.update(sim)  # inject Wasserstein + JS
        pred_stats_list.append(stats)
        s = format_stats(f"Pred ({name})", stats)
        print(s.strip())
        report.append(s)
        report.append(f"  pred dir: {det_dir}")

    png = plot_distribution_comparison(
        gt_boxes, pred_boxes_list, MODEL_NAMES, OUTPUT_PNG
    )
    print(f"\nSaved: {png}")

    scatter_png = OUTPUT_PNG.replace(".png", "_scatter.png")
    plot_comparison_scatter(GT_DOTA_DIR, DET_DIRS, MODEL_NAMES, scatter_png)
    print(f"Saved: {scatter_png}")

    with open(OUTPUT_TXT, "w") as f:
        f.write("\n".join(report))
    print(f"Saved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
