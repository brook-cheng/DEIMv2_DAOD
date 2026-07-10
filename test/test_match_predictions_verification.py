#!/usr/bin/env python3
"""
match_predictions 四种模式一致性验证。

验证 ``match_mode="no_reorder"`` 和 ``match_mode="reorder"`` 在
torch/numpy 输入下各自输出一致（torch 路径 == numpy 路径），
并展示两种模式之间的差异。

用法:
    python test/test_match_predictions_verification.py
"""

import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.deim.obb_ops import batch_probiou
from engine.eval.obb_eval import match_predictions


# ──────────────────────────────────────────────────────────────────
# 合成数据：模拟 DETR 典型的重复预测场景
# ──────────────────────────────────────────────────────────────────
def make_data():
    """1 GT, 5 preds（都 class=0），IoU 和 confidence 分布不同。"""
    gt_boxes = torch.tensor([[128.0, 128.0, 60.0, 30.0, 0.0]])
    gt_labels = torch.tensor([0])

    pred_boxes = torch.tensor([
        [130.0, 128.0, 58.0, 28.0, 0.0],   # IoU≈0.94
        [126.0, 127.0, 55.0, 26.0, 0.05],  # IoU≈0.89
        [133.0, 130.0, 62.0, 32.0, 0.0],   # IoU≈0.87
        [128.0, 128.0, 50.0, 25.0, 0.0],   # IoU≈0.87
        [135.0, 125.0, 65.0, 35.0, 0.1],   # IoU≈0.79
    ], dtype=torch.float32)

    pred_labels = torch.tensor([0, 0, 0, 0, 0])
    pred_scores = torch.tensor([0.70, 0.95, 0.30, 0.50, 0.40])

    return gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores


# ──────────────────────────────────────────────────────────────────
# 验证
# ──────────────────────────────────────────────────────────────────
def main():
    gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores = make_data()
    iou = batch_probiou(gt_boxes, pred_boxes)
    iouv = (0.5, 0.75)

    print("IoU matrix (gt x pred):")
    for n in range(len(pred_labels)):
        print(f"  pred{n} (conf={pred_scores[n]:.2f}): {iou[0, n]:.3f}")
    print()

    # ── 四种模式 ────────────────────────────────────────────────
    modes = ["no_reorder", "reorder"]
    results = {}

    for mode in modes:
        # Torch input
        correct_t = match_predictions(pred_labels, gt_labels, iou, iouv, match_mode=mode)
        # Numpy input
        correct_n = match_predictions(
            pred_labels.cpu().numpy(),
            gt_labels.cpu().numpy(),
            iou.cpu().numpy(),
            iouv,
            match_mode=mode,
        )
        match_ok = np.array_equal(correct_t, correct_n)
        results[mode] = (correct_t, correct_n, match_ok)

        print(f"[{mode}]  torch==numpy? {'✓' if match_ok else '✗ FAIL'}")
        for j, thr in enumerate(iouv):
            tp_list = [i for i in range(len(pred_labels)) if correct_t[i, j]]
            print(f"  thr={thr:.2f}: TP = pred{tp_list}  (conf={[f'{pred_scores[i]:.2f}' for i in tp_list]})")
        print()

    # ── 跨模式对比 ──────────────────────────────────────────────
    print("=" * 50)
    print("跨模式对比 (torch 路径)")
    for j, thr in enumerate(iouv):
        a = [i for i in range(len(pred_labels)) if results["no_reorder"][0][i, j]]
        b = [i for i in range(len(pred_labels)) if results["reorder"][0][i, j]]
        same = a == b
        print(f"  thr={thr:.2f}: no_reorder TP={a}  reorder TP={b}  same={'✓' if same else '✗ DIFFER'}")

    # ── 最终判定 ─────────────────────────────────────────────────
    all_pass = all(r[2] for r in results.values())
    print(f"\n{'='*50}")
    if all_pass:
        print("ALL PASS: torch == numpy for both modes")
    else:
        print("FAIL: some modes have torch/numpy mismatch")
        for mode in modes:
            if not results[mode][2]:
                print(f"  {mode}: MISMATCH")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
