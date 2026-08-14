#!/usr/bin/env python3
"""
OBB model health diagnostic — box IoU distribution, score calibration,
and per-threshold precision for a given checkpoint.

Usage:
    python test/model_health_diag.py
    python test/model_health_diag.py -c configs/custom_obb/dlzdt/sp_fz_common.yml -r outputs/dlzdt_ablation/abl_rep0.pth -d cuda:0

What it reports:
    Box IoU    — per-GT best IoU with any prediction, distribution and match rate
    Scores     — histogram of all 300×N scores, fraction above key thresholds
    Precision  — TP/FP counts (greedy match at IoU=0.5) at different score cutoffs
    mAP        — full obb_evaluate result (mAP50, mAP50-95, recall, precision, F1)
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch

from engine.core import YAMLConfig
from engine.deim.obb_ops import batch_probiou
from engine.eval.obb_eval import obb_evaluate, match_predictions, DEFAULT_IOUV

DEFAULT_DEVICE = "cuda:0"


def _load_ema_weights(ckpt_path, model):
    state = torch.load(ckpt_path, map_location="cpu")
    if "ema" in state:
        source = state["ema"]["module"]
    elif "model" in state:
        source = state["model"]
    else:
        source = state
    cleaned = {k.replace("module.", ""): v for k, v in source.items()}
    model.load_state_dict(cleaned, strict=False)
    return model


def compute_precision_breakdown(all_raw, all_tp_scores, all_fp_scores):
    """Print TP/FP counts and precision at descending score thresholds."""
    tp_s = np.array(all_tp_scores)
    fp_s = np.array(all_fp_scores)
    print("\n--- Precision breakdown (greedy match at IoU=0.5) ---")
    for thr in [0.50, 0.25, 0.10, 0.05, 0.01]:
        tp_n = int((tp_s > thr).sum())
        fp_n = int((fp_s > thr).sum())
        total = tp_n + fp_n
        if total == 0:
            continue
        print(f"  score >{thr:.2f}: TP={tp_n:>5}  FP={fp_n:>5}  "
              f"precision={100 * tp_n / total:.1f}%  ({total} preds)")


def compute_score_histogram(all_raw_scores):
    """Print score distribution across all queries."""
    raw = np.concatenate(all_raw_scores)
    total = len(raw)
    print("\n--- Score distribution (300 queries × images) ---")
    print(f"  total queries: {total}")
    for thr in [0.50, 0.25, 0.10, 0.05, 0.01]:
        cnt = int((raw > thr).sum())
        print(f"  score >{thr:.2f}: {cnt:>7} ({100 * cnt / total:.1f}%)")
    print(f"  score mean={raw.mean():.4f}  median={np.median(raw):.4f}  "
          f"min={raw.min():.4f}  max={raw.max():.4f}")

    ranges = [(0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.20),
              (0.20, 0.50), (0.50, 1.00)]
    for lo, hi in ranges:
        cnt = int(((raw >= lo) & (raw < hi)).sum())
        print(f"  [{lo:.2f},{hi:.2f}): {cnt:>7} ({100 * cnt / total:.1f}%)")


def compute_box_iou_statistics(all_best_iou, all_matched):
    """Print per-GT best IoU statistics."""
    best = np.array(all_best_iou)
    matched = np.array(all_matched)
    print("\n--- Box IoU (per-GT best IoU with any prediction) ---")
    print(f"  total GTs: {len(best)}")
    print(f"  GTs with IoU >= 0.5: {int(matched.sum())} / {len(matched)} "
          f"= {100 * matched.sum() / len(matched):.1f}%")
    print(f"  IoU: mean={best.mean():.3f}  median={np.median(best):.3f}  "
          f"min={best.min():.3f}  max={best.max():.3f}")

    ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    for lo, hi in ranges:
        cnt = int(((best >= lo) & (best < hi)).sum())
        print(f"  IoU [{lo},{hi}): {cnt:>4} ({100 * cnt / len(best):.1f}%)")


def run_diagnostic(config_path, ckpt_path, device, data_overrides, no_full_eval=False):
    cfg = YAMLConfig(config_path, val_dataloader=data_overrides)
    model = cfg.model
    postprocessor = cfg.postprocessor
    val_loader = cfg.val_dataloader

    model.to(device)
    postprocessor.to(device)
    _load_ema_weights(ckpt_path, model)
    model.eval()
    postprocessor.eval()

    print(f"postprocessor.num_classes={postprocessor.num_classes}  "
          f"num_top_queries={postprocessor.num_top_queries}")

    all_raw_scores = []
    all_best_iou = []
    all_matched_gt = []
    tp_scores = []
    fp_scores = []

    for samples, targets in val_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_sizes)

        for res, tgt, orig_sz in zip(results, targets, orig_sizes):
            pred_boxes = res["boxes"]
            pred_scores = res["scores"]
            pred_labels = res["labels"]
            gt_boxes = tgt["boxes"].clone()
            gt_labels = tgt["labels"].clone()

            all_raw_scores.append(pred_scores.detach().cpu().numpy())

            ow, oh = orig_sz[0].item(), orig_sz[1].item()
            if gt_boxes.shape[0] > 0:
                scale = torch.tensor([ow, oh, ow, oh, 1.0],
                                     device=device, dtype=gt_boxes.dtype)
                gt_boxes_px = gt_boxes.detach() * scale
            else:
                gt_boxes_px = gt_boxes

            conf_mask = pred_scores > 0.01
            pb = pred_boxes[conf_mask]
            ps = pred_scores[conf_mask]
            pl = pred_labels[conf_mask]

            if pb.shape[0] > 0 and gt_boxes.shape[0] > 0:
                iou = batch_probiou(gt_boxes_px, pb[:, :5])
                best_per_gt = iou.max(dim=1).values.detach().cpu().numpy()
                all_best_iou.extend(best_per_gt.tolist())
                all_matched_gt.extend((best_per_gt >= 0.5).tolist())

                correct = match_predictions(pl, gt_labels, iou, [0.5])
                tp_mask = correct[:, 0].astype(bool)
                tp_s = ps.detach().cpu().numpy()
                for j in range(len(ps)):
                    if tp_mask[j]:
                        tp_scores.append(tp_s[j])
                    else:
                        fp_scores.append(tp_s[j])
            elif pb.shape[0] > 0:
                fp_scores.extend(ps.detach().cpu().tolist())

    compute_box_iou_statistics(all_best_iou, all_matched_gt)
    compute_score_histogram(all_raw_scores)
    compute_precision_breakdown(all_raw_scores, tp_scores, fp_scores)

    if no_full_eval:
        print("\n--- Full obb_evaluate skipped (--no-full-eval) ---")
        return {}

    print("\n--- Full obb_evaluate ---")
    stats = obb_evaluate(model, postprocessor, val_loader, device,
                         num_classes=postprocessor.num_classes)
    print(f"  mAP@50     = {stats['mAP50']:.4f}")
    print(f"  mAP@75     = {stats['AP75']:.4f}")
    print(f"  mAP@50:95  = {stats['mAP50_95']:.4f}")
    print(f"  Recall     = {stats['recall']:.4f}")
    print(f"  Precision  = {stats['precision']:.4f}")
    print(f"  F1         = {stats['f1']:.4f}")
    print(f"  Images     = {stats['seen']}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OBB model health diagnostic — IoU, scores, precision, mAP")
    parser.add_argument("-c", "--config",
        default="configs/custom_obb/dlzdt/sp_fz_common.yml")
    parser.add_argument("-r", "--ckpt",
        default="outputs/dlzdt_ablation/abl_rep0.pth")
    parser.add_argument("-d", "--device", default=DEFAULT_DEVICE)
    parser.add_argument("--data-base",
        default="/mnt/d/project_data/model_test/deimv2_obb_train_data/dlzdt_obb_val")
    parser.add_argument("--no-full-eval", action="store_true",
        help="skip obb_evaluate (faster, only run per-prediction stats)")
    args = parser.parse_args()

    data_overrides = {
        "dataset": {
            "img_folder": os.path.join(args.data_base, "images", "val"),
            "ann_folder": os.path.join(args.data_base, "labels", "val"),
            "classes_file": os.path.join(args.data_base, "classes.txt"),
        }
    }

    run_diagnostic(args.config, args.ckpt, args.device, data_overrides,
                   no_full_eval=args.no_full_eval)
