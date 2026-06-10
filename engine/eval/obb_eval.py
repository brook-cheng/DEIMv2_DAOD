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

    # ── Stage 3: compute AP per IoU threshold (per-image TP/FP) ──
    n_imgs = len(all_dets[0]) if all_dets else 0
    results_dict = {}
    aps_all = {}

    for iou_thr in iou_thrs:
        aps = []
        for cls_id in range(num_classes):
            all_tp, all_fp, all_scores = [], [], []
            total_gt_cls = 0

            for img_idx in range(n_imgs):
                det = all_dets[cls_id][img_idx]
                gt  = all_gts[cls_id][img_idx]
                total_gt_cls += len(gt)

                if len(det) == 0 or len(gt) == 0:
                    if len(det) > 0:
                        all_tp.append(np.zeros(len(det), dtype=np.float32))
                        all_fp.append(np.ones(len(det), dtype=np.float32))
                        all_scores.append(det[:, 5])
                    continue

                det_t = torch.tensor(det[:, :5], dtype=torch.float32)
                gt_t  = torch.tensor(gt, dtype=torch.float32)
                ious = batch_probiou(det_t, gt_t).numpy()
                tp, fp = _tpfp(ious, det[:, 5], iou_thr)
                all_tp.append(tp)
                all_fp.append(fp)
                all_scores.append(det[:, 5])

            if total_gt_cls == 0:
                aps.append(0.0)
                continue

            tp_cat = np.concatenate(all_tp) if all_tp else np.zeros(0)
            fp_cat = np.concatenate(all_fp) if all_fp else np.zeros(0)
            if not all_scores:
                aps.append(0.0)
                continue
            scores_cat = np.concatenate(all_scores)

            sort_idx = np.argsort(-scores_cat)
            tp_cum = np.cumsum(tp_cat[sort_idx])
            fp_cum = np.cumsum(fp_cat[sort_idx])

            eps = np.finfo(np.float32).eps
            rec = tp_cum / max(total_gt_cls, 1)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, eps)
            ap = _voc_ap(rec, prec, use_07_metric=True)
            aps.append(ap)

            tp_cum = np.cumsum(tp_cat)
            fp_cum = np.cumsum(fp_cat)
            eps = np.finfo(np.float32).eps
            rec = tp_cum / max(total_gt_cls, 1)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, eps)
            ap = _voc_ap(rec, prec, use_07_metric=True)
            aps.append(ap)

        mean_ap = np.mean(aps)
        key = f'AP{int(iou_thr*100):.0f}'
        aps_all[key] = mean_ap

    results_dict.update(aps_all)
    results_dict['mAP'] = np.mean(list(aps_all.values())) if aps_all else 0.0

    # ── Stage 5: precision / recall at IoU=0.5 ──
    all_tp_sum = all_fp_sum = total_gt = 0
    for cls_id in range(num_classes):
        for img_idx in range(n_imgs):
            det = all_dets[cls_id][img_idx]
            gt  = all_gts[cls_id][img_idx]
            total_gt += len(gt)
            if len(det) > 0 and len(gt) > 0:
                det_t = torch.tensor(det[:, :5], dtype=torch.float32)
                gt_t  = torch.tensor(gt, dtype=torch.float32)
                ious = batch_probiou(det_t, gt_t).numpy()
                tp, fp = _tpfp(ious, det[:, 5], 0.5)
                all_tp_sum += tp.sum()
                all_fp_sum += fp.sum()
            elif len(det) > 0:
                all_fp_sum += len(det)
    results_dict['precision'] = all_tp_sum / max(all_tp_sum + all_fp_sum, 1)
    results_dict['recall'] = all_tp_sum / max(total_gt, 1)

    # convert numpy scalars to Python native types for JSON serialization
    return {k: float(v) for k, v in results_dict.items()}
