"""
Online OBB evaluator for DEIMv2-OBB.

Computes AP50, AP75, mAP using exact polygon IoU (poly_iou).
Equivalent to DOTA_devkit results.
"""

import numpy as np
import torch
from .poly_iou import poly_iou


def _voc_ap(rec, prec, use_07_metric=True):
    """Compute VOC AP given precision and recall.

    If use_07_metric=True: 11-point interpolation (DOTA standard).
    Otherwise: area under PR curve.
    """
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


def _tpfp(det_boxes, gt_boxes, iou_thr):
    """TP/FP for a single class, single image.

    Args:
        det_boxes: (M, 6)  (cx, cy, w, h, theta, score), pixel coords
        gt_boxes:   (N, 5)  (cx, cy, w, h, theta), pixel coords
        iou_thr:    IoU threshold

    Returns:
        tp: (M,)  fp: (M,)
    """
    M = len(det_boxes)
    if M == 0:
        return np.zeros(0), np.zeros(0)
    if len(gt_boxes) == 0:
        return np.zeros(M), np.ones(M)

    det_t = torch.tensor(det_boxes[:, :5], dtype=torch.float32)
    gt_t  = torch.tensor(gt_boxes, dtype=torch.float32)
    ious  = poly_iou(det_t, gt_t).numpy()              # (M, N)

    # sort by score descending
    sort_idx = np.argsort(-det_boxes[:, 5])
    ious = ious[sort_idx]

    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(M, dtype=np.float32)
    fp = np.zeros(M, dtype=np.float32)

    for i, iou_row in enumerate(ious):
        if iou_row.max() >= iou_thr:
            best = iou_row.argmax()
            if not gt_matched[best]:
                gt_matched[best] = True
                tp[i] = 1.
            else:
                fp[i] = 1.
        else:
            fp[i] = 1.

    # restore original order
    tp_reorder = np.zeros(M, dtype=np.float32)
    fp_reorder = np.zeros(M, dtype=np.float32)
    tp_reorder[sort_idx] = tp
    fp_reorder[sort_idx] = fp
    return tp_reorder, fp_reorder


@torch.no_grad()
def obb_evaluate(model, postprocessor, data_loader, device,
                 iou_thrs=(0.5,), num_classes=15):
    """Online OBB evaluation using exact polygon IoU.

    Returns:
        dict with AP50, AP75, mAP, precision, recall
    """
    model.eval()
    postprocessor.eval()

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

    results_dict = {}
    aps_all = {}

    for iou_thr in iou_thrs:
        aps = []
        for cls_id in range(num_classes):
            cls_dets = np.vstack(all_dets[cls_id])
            cls_gts  = np.vstack(all_gts[cls_id])

            if len(cls_gts) == 0:
                aps.append(0.0)
                continue

            tp, fp = _tpfp(cls_dets, cls_gts, iou_thr)
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            eps = np.finfo(np.float32).eps
            rec = tp_cum / max(len(cls_gts), 1)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, eps)
            ap = _voc_ap(rec, prec, use_07_metric=True)
            aps.append(ap)

        mean_ap = np.mean(aps)
        key = f'AP{int(iou_thr*100):.0f}'
        aps_all[key] = mean_ap

    results_dict.update(aps_all)
    results_dict['mAP'] = np.mean(list(aps_all.values()))

    # precision / recall at IoU=0.5
    all_tp = all_fp = total_gt = 0
    for cls_id in range(num_classes):
        cls_dets = np.vstack(all_dets[cls_id])
        cls_gts  = np.vstack(all_gts[cls_id])
        if len(cls_gts) > 0:
            tp, fp = _tpfp(cls_dets, cls_gts, 0.5)
            all_tp += tp.sum()
            all_fp += fp.sum()
            total_gt += len(cls_gts)
    results_dict['precision'] = all_tp / max(all_tp + all_fp, 1)
    results_dict['recall'] = all_tp / max(total_gt, 1)

    return results_dict
