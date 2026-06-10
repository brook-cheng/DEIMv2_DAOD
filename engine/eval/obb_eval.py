"""
Online OBB evaluator for DEIMv2-OBB.

Computes AP50, AP75, mAP using batch_probiou.
Optimized for large validation sets.
"""

import numpy as np
import torch
from ..deim.obb_ops import batch_probiou


def _voc_ap(rec, prec, use_07_metric=True):
    """Compute VOC AP given precision and recall."""
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0.
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
        return ap
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def _tpfp(ious, scores, iou_thr):
    """TP/FP from precomputed IoU matrix.

    Args:
        ious:   (M, N)  IoU matrix (float32)
        scores: (M,)    detection scores
        iou_thr: IoU threshold

    Returns:
        tp: (M,)  fp: (M,)
    """
    M, N = ious.shape
    if M == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    if N == 0:
        return np.zeros(M, dtype=np.float32), np.ones(M, dtype=np.float32)

    sort_idx = np.argsort(-scores)
    ious = ious[sort_idx]

    gt_matched = np.zeros(N, dtype=bool)
    tp = np.zeros(M, dtype=np.float32)
    fp = np.zeros(M, dtype=np.float32)

    for i in range(M):
        iou_row = ious[i]
        best_j = iou_row.argmax()
        if iou_row[best_j] >= iou_thr:
            if not gt_matched[best_j]:
                gt_matched[best_j] = True
                tp[i] = 1.
            else:
                fp[i] = 1.
        else:
            fp[i] = 1.

    tp_reorder = np.zeros(M, dtype=np.float32)
    fp_reorder = np.zeros(M, dtype=np.float32)
    tp_reorder[sort_idx] = tp
    fp_reorder[sort_idx] = fp
    return tp_reorder, fp_reorder


@torch.no_grad()
def obb_evaluate(model, postprocessor, data_loader, device,
                 iou_thrs=(0.5,), num_classes=15):
    """Online OBB evaluation — optimized for large validation sets.

    Data collection and IoU computation are each done once.
    TP/FP assignment is cheap and repeated per IoU threshold.

    Returns:
        dict with AP50, AP75, mAP, precision, recall
    """
    model.eval()
    postprocessor.eval()

    # ── Stage 1: collect all predictions ──
    all_dets = [[] for _ in range(num_classes)]
    all_gts  = [[] for _ in range(num_classes)]

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_sizes)

        for res, tgt in zip(results, targets):
            boxes  = res['boxes'].cpu().numpy()
            scores = res['scores'].cpu().numpy()
            labels = res['labels'].cpu().numpy()
            gt_boxes  = tgt['boxes'].cpu().numpy()
            gt_labels = tgt['labels'].cpu().numpy()

            for cls_id in range(num_classes):
                mask_p = labels == cls_id
                if mask_p.any():
                    det = np.concatenate([boxes[mask_p], scores[mask_p, None]], axis=1)
                    all_dets[cls_id].append(det)
                else:
                    all_dets[cls_id].append(np.zeros((0, 6), dtype=np.float32))

                mask_g = gt_labels == cls_id
                if mask_g.any():
                    all_gts[cls_id].append(gt_boxes[mask_g])
                else:
                    all_gts[cls_id].append(np.zeros((0, 5), dtype=np.float32))

    # ── Stage 2: vstack once ──
    stacked_dets = [np.vstack(all_dets[c]) for c in range(num_classes)]
    stacked_gts  = [np.vstack(all_gts[c]) for c in range(num_classes)]

    # ── Stage 3: precompute IoU per class ──
    iou_cache = []
    for cls_id in range(num_classes):
        det = stacked_dets[cls_id]
        gt  = stacked_gts[cls_id]
        if len(det) > 0 and len(gt) > 0:
            det_t = torch.tensor(det[:, :5], dtype=torch.float32)
            gt_t  = torch.tensor(gt, dtype=torch.float32)
            iou_cache.append(batch_probiou(det_t, gt_t).numpy())
        else:
            iou_cache.append(None)

    # ── Stage 4: AP per IoU threshold (cheap, no IoU recompute) ──
    results_dict = {}
    aps_all = {}

    for iou_thr in iou_thrs:
        aps = []
        for cls_id in range(num_classes):
            det = stacked_dets[cls_id]
            gt  = stacked_gts[cls_id]
            ious = iou_cache[cls_id]

            if len(gt) == 0:
                aps.append(0.0)
                continue

            scores = det[:, 5] if len(det) > 0 else np.zeros(0)
            tp, fp = _tpfp(ious, scores, iou_thr)
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            eps = np.finfo(np.float32).eps
            rec = tp_cum / max(len(gt), 1)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, eps)
            ap = _voc_ap(rec, prec, use_07_metric=True)
            aps.append(ap)

        mean_ap = np.mean(aps)
        key = f'AP{int(iou_thr*100):.0f}'
        aps_all[key] = mean_ap

    results_dict.update(aps_all)
    results_dict['mAP'] = np.mean(list(aps_all.values())) if aps_all else 0.0

    # ── Stage 5: precision / recall at IoU=0.5 ──
    all_tp = all_fp = total_gt = 0
    for cls_id in range(num_classes):
        det = stacked_dets[cls_id]
        gt  = stacked_gts[cls_id]
        ious = iou_cache[cls_id]
        if len(gt) > 0 and ious is not None:
            scores = det[:, 5]
            tp, fp = _tpfp(ious, scores, 0.5)
            all_tp += tp.sum()
            all_fp += fp.sum()
            total_gt += len(gt)
    results_dict['precision'] = all_tp / max(all_tp + all_fp, 1)
    results_dict['recall'] = all_tp / max(total_gt, 1)

    return results_dict
