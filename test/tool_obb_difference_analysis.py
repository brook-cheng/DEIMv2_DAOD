#!/usr/bin/env python3
"""
OBB 逐样本配对差异分析工具
============================

Overview
--------
For each image, matches prediction boxes to ground-truth boxes using
Hungarian algorithm on ProbIoU cost, then computes paired differences
for w/h ratio and angle (signed shortest angular difference in degrees).

Unlike the distribution comparison tool (which compares aggregate
distributions), this tool shows *per-pair* errors — revealing whether
the model systematically overestimates/underestimates w/h or has angular
bias toward specific orientations.

Report includes:
- w/h difference distribution (histogram + text: mean, std, % within ±0.2)
- angle difference distribution (histogram + text: mean abs, % within ±5°)
- Matched / unmatched count statistics
- Scatter plots: pred w/h vs GT w/h for matched pairs

Entry Points
------------
Programmatic:
    ``match_and_compute_diffs(gt_per_img, pred_per_img, iou_thr)``
    — returns wh_diffs, angle_diffs, match counts.

Script:
    Edit ``main()`` paths and run::

        python test/tool_obb_difference_analysis.py

Configuration (edit in-file)
-----------------------------
GT_DOTA_DIR   : str   — DOTA-format GT directory
DET_DIRS      : list  — list of prediction directories (one per model)
MODEL_NAMES   : list  — display names matching DET_DIRS order
OUTPUT_DIR    : str   — output directory for PNGs and report
OUTPUT_TXT    : str   — path for text report
IOU_THR       : float — minimum ProbIoU for valid matching (default 0.1)

Metrics
-------
Per model:
    w/h diff mean/std       — bias and spread of aspect ratio error
    angle diff mean abs     — average angular deviation
    angle diff % > 5°       — fraction of matched pairs with >5° error
    matched / unmatched     — matching quality indicator

Output Structure
----------------
OUTPUT_DIR/
├── diff_analysis_<model_name>.png  # histograms + scatter per model
└── difference_report.txt           # text report with all metrics

Usage
-----
    python test/tool_obb_difference_analysis.py
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
from scipy.optimize import linear_sum_assignment

from engine.deim.obb_geometry import xyxyxyxy_to_xywhr
from engine.deim.obb_ops import batch_probiou
from tools.model_compare.obb_utils import parse_dota_line

# ──────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────


def _poly_to_xywhr(poly8):
    poly = torch.tensor(poly8, dtype=torch.float32).reshape(1, 4, 2)
    return xyxyxyxy_to_xywhr(poly).numpy().flatten()


def load_boxes_from_dota_dir(dota_dir, is_gt=True):
    """Return list of (cx,cy,w,h,θ,label) tuples for each box in `dota_dir`."""
    boxes = []
    for fname in sorted(os.listdir(dota_dir)):
        if not fname.endswith(".txt"):
            continue
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
                    poly8 = [float(x) for x in parts[:8]]
                    label = parts[8]
                else:
                    rec = parse_dota_line(line)
                    if rec is None:
                        continue
                    poly8 = rec["poly"]
                    label = rec["label"]
                cx, cy, w, h, theta = _poly_to_xywhr(poly8)
                boxes.append((cx, cy, w, h, theta, label))
    return boxes


def load_boxes_per_image(dota_dir, is_gt=True):
    """Return dict {img_stem: [(cx,cy,w,h,θ,label), ...]}."""
    data = {}
    for fname in sorted(os.listdir(dota_dir)):
        if not fname.endswith(".txt"):
            continue
        stem = os.path.splitext(fname)[0]
        boxes = []
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
                    poly8 = [float(x) for x in parts[:8]]
                    label = parts[8]
                else:
                    rec = parse_dota_line(line)
                    if rec is None:
                        continue
                    poly8 = rec["poly"]
                    label = rec["label"]
                cx, cy, w, h, theta = _poly_to_xywhr(poly8)
                boxes.append((cx, cy, w, h, theta, label))
        if boxes:
            data[stem] = boxes
    return data


# ──────────────────────────────────────────────────────────────────
# 匹配与差异计算
# ──────────────────────────────────────────────────────────────────


def _signed_periodic_angle_diff(pred_theta, gt_theta):
    """Shortest signed angular difference, range [-90°, 90°]."""
    diff = pred_theta - gt_theta
    return ((diff + np.pi / 2) % np.pi - np.pi / 2) * 180.0 / np.pi


def match_and_compute_diffs(gt_boxes_per_img, pred_boxes_per_img, iou_thr=0.1):
    """For each image, Hungarian-match pred↔GT and collect paired differences.

    Returns:
        wh_diffs:   list of (pred_w/h - gt_w/h) for matched pairs
        angle_diffs: list of signed shortest angular difference
        n_matched:   total matched pairs
        n_unmatched_pred: pred boxes unmatched
        n_unmatched_gt:   GT boxes unmatched
        matched_wh:  list of (gt_wh, pred_wh) tuples for matched pairs
        matched_ang: list of (gt_θ°, pred_θ°) tuples for matched pairs
    """
    wh_diffs = []
    angle_diffs = []
    matched_wh = []
    matched_ang = []
    n_matched = 0
    n_unmatched_pred = 0
    n_unmatched_gt = 0

    common = set(gt_boxes_per_img.keys()) & set(pred_boxes_per_img.keys())
    if not common:
        print("  WARNING: no common image stems between GT and pred")
        return wh_diffs, angle_diffs, 0, 0, 0

    for stem in sorted(common):
        gt_list = gt_boxes_per_img[stem]
        pred_list = pred_boxes_per_img[stem]
        M, N = len(gt_list), len(pred_list)
        if M == 0 or N == 0:
            n_unmatched_pred += N
            n_unmatched_gt += M
            continue

        gt_arr = np.array(
            [[b[0], b[1], b[2], b[3], b[4]] for b in gt_list], dtype=np.float32
        )
        pred_arr = np.array(
            [[b[0], b[1], b[2], b[3], b[4]] for b in pred_list], dtype=np.float32
        )

        gt_t = torch.tensor(gt_arr)
        pred_t = torch.tensor(pred_arr)
        iou = batch_probiou(gt_t, pred_t).numpy()  # (M, N)

        # Hungarian: maximise IoU → minimise -IoU
        cost = -iou
        gt_idx, pred_idx = linear_sum_assignment(cost)

        for g, p in zip(gt_idx, pred_idx):
            if iou[g, p] < iou_thr:
                n_unmatched_pred += 1  # this pred didn't really match
                n_unmatched_gt += 1  # this GT wasn't really matched
                continue
            n_matched += 1

            gt_wh = gt_arr[g, 2] / max(gt_arr[g, 3], 1e-6)
            pred_wh = pred_arr[p, 2] / max(pred_arr[p, 3], 1e-6)
            wh_diffs.append(pred_wh - gt_wh)
            angle_diffs.append(
                _signed_periodic_angle_diff(pred_arr[p, 4], gt_arr[g, 4])
            )
            matched_wh.append((gt_wh, pred_wh))
            matched_ang.append(
                (float(gt_arr[g, 4] * 180.0 / np.pi), float(pred_arr[p, 4] * 180.0 / np.pi))
            )

        # unmatched counts
        if N > M:
            n_unmatched_pred += N - M  # extra preds
        elif M > N:
            n_unmatched_gt += M - N  # extra GTs

    return (
        wh_diffs, angle_diffs, n_matched, n_unmatched_pred, n_unmatched_gt,
        matched_wh, matched_ang,
    )


# ──────────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────────


def plot_difference_analysis(
    wh_diffs,
    angle_diffs,
    n_matched,
    n_unmatched_pred,
    n_unmatched_gt,
    matched_wh,
    matched_ang,
    pred_name,
    output_path,
    bins=60,
):
    """4-panel figure: wh-diff hist, angle-diff hist, wh scatter, angle scatter.

    Scatter plots use Hungarian-matched pairs, not arbitrary list-index alignment.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── w/h ratio difference histogram ──
    ax = axes[0, 0]
    ax.hist(
        wh_diffs, bins=bins, density=True, color="tab:blue", alpha=0.7, edgecolor="navy"
    )
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    mean_wh = np.mean(wh_diffs) if wh_diffs else 0
    ax.axvline(
        mean_wh, color="red", linewidth=1.5, linestyle="-", label=f"mean={mean_wh:+.3f}"
    )
    ax.set_xlabel("Δ (pred w/h − gt w/h)")
    ax.set_ylabel("density")
    ax.set_title(f"w/h ratio difference  ({pred_name})")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # ── angle difference histogram ──
    ax = axes[0, 1]
    ax.hist(
        angle_diffs,
        bins=bins,
        density=True,
        color="tab:orange",
        alpha=0.7,
        edgecolor="darkorange",
    )
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    mean_ang = np.mean(np.abs(angle_diffs)) if angle_diffs else 0
    ax.axvline(
        mean_ang, color="red", linewidth=1.5, linestyle="-",
        label=f"|Δθ| mean={mean_ang:.1f}°"
    )
    ax.set_xlabel("Δθ (degrees, signed shortest path)")
    ax.set_ylabel("density")
    ax.set_title(f"Angle difference  ({pred_name})")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # ── w/h scatter: matched pred vs GT ──
    ax = axes[1, 0]
    if matched_wh:
        gt_wh_arr = np.array([p[0] for p in matched_wh])
        pr_wh_arr = np.array([p[1] for p in matched_wh])
        n_pts = min(len(gt_wh_arr), 3000)
        if n_pts > 1:
            idx = np.random.RandomState(42).choice(len(gt_wh_arr), size=n_pts, replace=False)
            lim = max(gt_wh_arr.max(), pr_wh_arr.max()) * 1.1
            ax.scatter(gt_wh_arr[idx], pr_wh_arr[idx], s=5, alpha=0.35, color="tab:blue")
            ax.plot([0, lim], [0, lim], "k--", linewidth=0.8)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
    ax.set_xlabel("GT w/h")
    ax.set_ylabel("Pred w/h")
    ax.set_title("w/h ratio: Pred vs GT (Hungarian-matched pairs)")

    # ── angle scatter: matched pred vs GT ──
    ax = axes[1, 1]
    if matched_ang:
        gt_ang_arr = np.array([p[0] for p in matched_ang])
        pr_ang_arr = np.array([p[1] for p in matched_ang])
        n_pts = min(len(gt_ang_arr), 3000)
        if n_pts > 1:
            idx = np.random.RandomState(42).choice(len(gt_ang_arr), size=n_pts, replace=False)
            ax.scatter(
                gt_ang_arr[idx], pr_ang_arr[idx], s=5, alpha=0.35, color="tab:orange"
            )
            ax.plot([0, 180], [0, 180], "k--", linewidth=0.8)
            ax.set_xlim(0, 180)
            ax.set_ylim(0, 180)
    ax.set_xlabel("GT θ (degrees)")
    ax.set_ylabel("Pred θ (degrees)")
    ax.set_title("Angle: Pred vs GT (Hungarian-matched pairs)")

    # ── summary annotation ──
    summary = (
        f"Matched pairs: {n_matched}\n"
        f"Unmatched pred: {n_unmatched_pred}\n"
        f"Unmatched GT:   {n_unmatched_gt}\n"
        f"IoU threshold:  0.1\n"
        f"Δ w/h mean:     {np.mean(wh_diffs):+.4f}\n"
        f"Δ w/h std:      {np.std(wh_diffs):.4f}\n"
        f"Δ |θ| mean:     {np.mean(np.abs(angle_diffs)):.1f}°"
    )
    fig.text(
        0.02,
        0.02,
        summary,
        fontfamily="monospace",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    fig.suptitle(f"Per-sample OBB Difference: {pred_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def compute_difference_stats(
    wh_diffs, angle_diffs, n_matched, n_unmatched_pred, n_unmatched_gt
):
    wh = np.array(wh_diffs) if wh_diffs else np.array([])
    ang = np.array(angle_diffs) if angle_diffs else np.array([])
    return {
        "n_matched": n_matched,
        "n_unmatched_pred": n_unmatched_pred,
        "n_unmatched_gt": n_unmatched_gt,
        "wh_diff_mean": float(np.mean(wh)) if len(wh) else None,
        "wh_diff_std": float(np.std(wh)) if len(wh) else None,
        "wh_diff_p25": float(np.percentile(wh, 25)) if len(wh) else None,
        "wh_diff_p50": float(np.percentile(wh, 50)) if len(wh) else None,
        "wh_diff_p75": float(np.percentile(wh, 75)) if len(wh) else None,
        "wh_diff_lt_0_pct": float(np.mean(wh < 0) * 100) if len(wh) else None,
        "ang_mean_abs_deg": float(np.mean(np.abs(ang))) if len(ang) else None,
        "ang_std_deg": float(np.std(ang)) if len(ang) else None,
        "ang_p50_abs_deg": float(np.percentile(np.abs(ang), 50)) if len(ang) else None,
        "ang_p90_abs_deg": float(np.percentile(np.abs(ang), 90)) if len(ang) else None,
        "ang_over_15deg_pct": (
            float(np.mean(np.abs(ang) > 15) * 100) if len(ang) else None
        ),
        "ang_over_30deg_pct": (
            float(np.mean(np.abs(ang) > 30) * 100) if len(ang) else None
        ),
    }


def format_diff_stats(name, st):
    if st is None:
        return ""
    lines = [
        f"\n{name}:",
        f"  Matched pairs:   {st['n_matched']}",
        f"  Unmatched pred:  {st['n_unmatched_pred']}",
        f"  Unmatched GT:    {st['n_unmatched_gt']}",
    ]
    if st["wh_diff_mean"] is not None:
        lines += [
            f"  ── w/h ratio diff (pred − gt) ──",
            f"    mean:     {st['wh_diff_mean']:+.4f}",
            f"    std:      {st['wh_diff_std']:.4f}",
            f"    p25/p50/p75: {st['wh_diff_p25']:+.3f} / {st['wh_diff_p50']:+.3f} / {st['wh_diff_p75']:+.3f}",
            f"    % < 0 (pred更方): {st['wh_diff_lt_0_pct']:.1f}%",
        ]
    if st["ang_mean_abs_deg"] is not None:
        lines += [
            f"  ── angle diff (signed shortest) ──",
            f"    |Δθ| mean:  {st['ang_mean_abs_deg']:.1f}°",
            f"    |Δθ| p50:   {st['ang_p50_abs_deg']:.1f}°",
            f"    |Δθ| p90:   {st['ang_p90_abs_deg']:.1f}°",
            f"    % > 15°:    {st['ang_over_15deg_pct']:.1f}%",
            f"    % > 30°:    {st['ang_over_30deg_pct']:.1f}%",
        ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────


def main():
    GT_DOTA_DIR = "./test/data/outputs/dlzdt_obb_compare_val/gt_dota"
    DET_DIRS = [
        "./test/data/outputs/dlzdt_obb_compare_val/yolo_dota",
        "./test/data/outputs/dlzdt_res/sp_ft_rep0_0714_val",
        "./test/data/outputs/dlzdt_res/sp_ft_rep0_0715_val",
        "./test/data/outputs/dlzdt_res/sp_ft_rep1_0714_val",
        "./test/data/outputs/dlzdt_res/sp_ft_rep1_0715_val",
        "./test/data/outputs/dlzdt_res/sp_ft_rep3_0714_val",
    ]
    MODEL_NAMES = [
        "YOLO-OBB",
        "sp_ft_rep0_14",
        "sp_ft_rep0_15",
        "sp_ft_rep1_14",
        "sp_ft_rep1_15",
        "sp_ft_rep3_14",
    ]
    OUTPUT_DIR = "./test/data/outputs/dlzdt_obb_compare_val/obb_diff_analysis"
    OUTPUT_TXT = os.path.join(OUTPUT_DIR, "difference_report.txt")
    IOU_THR = 0.1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = [
        "=" * 60,
        "OBB Per-Sample Difference Report",
        "=" * 60,
        f"GT dir: {GT_DOTA_DIR}",
        f"IoU threshold: {IOU_THR}",
    ]

    # ── Load per-image ──
    print("Loading GT per image...")
    gt_per_img = load_boxes_per_image(GT_DOTA_DIR, is_gt=True)
    print(f"  {len(gt_per_img)} images with GT boxes")
    gt_boxes_raw = [b for boxes in gt_per_img.values() for b in boxes]
    report.append(f"Total GT boxes: {len(gt_boxes_raw)}")
    print(f"  Total GT boxes: {len(gt_boxes_raw)}")

    for det_dir, name in zip(DET_DIRS, MODEL_NAMES):
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        report.append(f"\n{'─'*40}")

        print("  Loading pred per image...")
        pred_per_img = load_boxes_per_image(det_dir, is_gt=False)
        print(f"  {len(pred_per_img)} images with predictions")
        pred_boxes_raw = [b for boxes in pred_per_img.values() for b in boxes]
        print(f"  Total pred boxes: {len(pred_boxes_raw)}")

        print(f"  Matching (Hungarian, IoU threshold={IOU_THR})...")
        (
            wh_diffs, angle_diffs, n_matched, n_unmatched_pred, n_unmatched_gt,
            matched_wh, matched_ang,
        ) = match_and_compute_diffs(gt_per_img, pred_per_img, iou_thr=IOU_THR)
        print(
            f"  Matched: {n_matched},  Unmatched pred: {n_unmatched_pred},  "
            f"Unmatched GT: {n_unmatched_gt}"
        )

        st = compute_difference_stats(
            wh_diffs, angle_diffs, n_matched, n_unmatched_pred, n_unmatched_gt
        )
        s = format_diff_stats(name, st)
        print(s.strip())
        report.append(s)
        report.append(f"  pred dir: {det_dir}")

        plot_difference_analysis(
            wh_diffs,
            angle_diffs,
            n_matched,
            n_unmatched_pred,
            n_unmatched_gt,
            matched_wh,
            matched_ang,
            name,
            os.path.join(OUTPUT_DIR, f"diff_analysis_{name.replace(' ', '_')}.png"),
        )

    with open(OUTPUT_TXT, "w") as f:
        f.write("\n".join(report))
    print(f"\nSaved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
