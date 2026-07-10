"""
Online OBB evaluator for DEIMv2-OBB.

Computes standard mAP@0.5:0.95, mAP@0.5, mAP@0.75 plus precision, recall and
F1 reported at the max-F1 confidence threshold, following the standard
pycocotools / ultralytics approach. Per-class TP/FP are accumulated across
all images and the PR curve is built by sorting detections by confidence — the
reported precision is therefore at the max-F1 point on the PR curve, not at a
fixed confidence threshold. This produces metric values comparable across runs
with different score-head quality.

IoU uses batch_probiou (Gaussian ProbIoU) for oriented boxes.
"""

import numpy as np
import torch
from ..deim.obb_ops import batch_probiou

# Standard COCO/IoU vector: 0.50, 0.55, ..., 0.95
DEFAULT_IOUV = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def compute_ap(recall, precision):
    """Compute AP from recall/precision curves using the 101-point interp method.

    Args:
        recall: array of cumulative recall values (sorted by conf desc).
        precision: array of cumulative precision values.

    Returns:
        ap (float), mpre (precision envelope including sentinels),
        mrec (recall including sentinels).
    """
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    ap = _trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def match_predictions(pred_classes, true_classes, iou, iouv):
    """Match predictions to ground truth for each IoU threshold.

    Greedy one-to-one matching per IoU threshold: pairs sorted by IoU desc,
    then unique detection and unique label are kept.

    Args:
        pred_classes: 1-D tensor of predicted class ids (N,).
        true_classes: 1-D tensor of GT class ids (M,).
        iou: NxM IoU matrix (float).
        iouv: list/tensor of IoU thresholds.

    Returns:
        correct: (N, len(iouv)) boolean array; True if detection i is a TP at threshold j.
    """
    correct = np.zeros((pred_classes.shape[0], len(iouv)), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes
    iou = (
        (iou * correct_class).cpu().numpy()
        if hasattr(iou, "cpu")
        else np.asarray(iou) * correct_class.cpu().numpy()
    )
    for j, thr in enumerate(list(iouv)):
        matches = np.nonzero(iou >= thr)
        matches = np.array(matches).T  # (k, 2): rows=gt_idx, cols=det_idx
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), j] = True
    return correct


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """Compute Average Precision per class from accumulated TP matrix.

    Args:
        tp: (N, n_iou) boolean array of TP per detection per IoU threshold.
        conf: (N,) confidence scores.
        pred_cls: (N,) predicted class ids.
        target_cls: (M,) ground-truth class ids.

    Returns:
        Dictionary with:
            p, r, f1, ap50, ap75, map50_95, ap, tp_count, fp_count,
            unique_classes, p_curve, r_curve, f1_curve.
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    n_iou = tp.shape[1]

    x = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, n_iou))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]
        n_p = i.sum()
        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)

        for j in range(n_iou):
            ap[ci, j], _, _ = compute_ap(recall[:, j], precision[:, j])

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    i = f1_curve.mean(0).argmax()
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
    tp_count = (r * nt).round()
    fp_count = (tp_count / (p + eps) - tp_count).round()

    return {
        "p": p,
        "r": r,
        "f1": f1,
        "ap": ap,
        "ap50": ap[:, 0] if ap.shape[1] > 0 else np.zeros(nc),
        "ap75": ap[:, 5] if ap.shape[1] > 5 else np.zeros(nc),
        "map50_95": ap.mean(1) if ap.size else np.zeros(nc),
        "tp_count": tp_count,
        "fp_count": fp_count,
        "unique_classes": unique_classes.astype(int),
        "p_curve": p_curve,
        "r_curve": r_curve,
        "f1_curve": f1_curve,
        "x": x,
    }


@torch.no_grad()
def obb_evaluate(
    model,
    postprocessor,
    data_loader,
    device,
    iou_thrs=DEFAULT_IOUV,
    num_classes=15,
    conf_thresh=None,
):
    """Online OBB evaluation using the standard mAP@0.5:0.95 approach.

    Per-image detection vs GT IoU uses batch_probiou. Detections are not
    pre-filtered by confidence — the full PR curve is built by sorting by
    score, and precision/recall are reported at the max-F1 point on the curve.

    Args:
        model: nn.Module in eval mode.
        postprocessor: produces topk queries with {boxes, scores, labels}.
        data_loader: validation DataLoader yielding (samples, targets).
        device: torch.device.
        iou_thrs: tuple of IoU thresholds for AP. Default COCO = 0.5..0.95.
        num_classes: number of classes.
        conf_thresh: kept for backward-compat; not used (max-F1 point autoREPORTs P/R).

    Returns:
        dict with AP50, AP75, mAP (mAP50-95), mAP50, precision, recall, f1,
        and per-class breakdown keyed by `class_<id>_*`.
    """
    model.eval()
    postprocessor.eval()
    iou_thrs = tuple(iou_thrs)

    all_tp = []  # list of (Mi, n_iou) bool
    all_conf = []  # list of (Mi,)
    all_pred_cls = []  # list of (Mi,)
    all_target_cls = []  # list of (M_gt,) per image
    seen_imgs = 0

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_sizes)

        for res, tgt, orig_sz in zip(results, targets, orig_sizes):
            seen_imgs += 1
            pred_boxes = res["boxes"].cpu().numpy()
            pred_scores = res["scores"].cpu().numpy()
            pred_labels = res["labels"].cpu().numpy().astype(np.int64)
            gt_boxes = tgt["boxes"].cpu().numpy()
            gt_labels = tgt["labels"].cpu().numpy().astype(np.int64)

            ow, oh = orig_sz.cpu().numpy()
            if len(gt_boxes) > 0:
                gt_boxes[:, 0] *= ow
                gt_boxes[:, 1] *= oh
                gt_boxes[:, 2] *= ow
                gt_boxes[:, 3] *= oh

            all_target_cls.append(gt_labels)

            if pred_labels.shape[0] == 0:
                all_tp.append(np.zeros((0, len(iou_thrs)), dtype=bool))
                all_conf.append(np.zeros(0))
                all_pred_cls.append(np.zeros(0, dtype=np.int64))
                continue

            if conf_thresh is not None and conf_thresh > 0:
                conf_mask = pred_scores > conf_thresh
                pred_boxes = pred_boxes[conf_mask]
                pred_scores = pred_scores[conf_mask]
                pred_labels = pred_labels[conf_mask]

                if pred_boxes.shape[0] == 0:
                    all_tp.append(np.zeros((0, len(iou_thrs)), dtype=bool))
                    all_conf.append(np.zeros(0))
                    all_pred_cls.append(np.zeros(0, dtype=np.int64))
                    continue

            if gt_boxes.shape[0] == 0:
                # all preds are FP at every IoU threshold
                correct = np.zeros((pred_labels.shape[0], len(iou_thrs)), dtype=bool)
                all_tp.append(correct)
                all_conf.append(pred_scores)
                all_pred_cls.append(pred_labels)
                continue

            det_t = torch.tensor(pred_boxes[:, :5], dtype=torch.float32)
            gt_t = torch.tensor(gt_boxes, dtype=torch.float32)
            iou = batch_probiou(gt_t, det_t)  # (M_gt, N_pred) — row=gt, col=pred
            correct = match_predictions(
                torch.tensor(pred_labels),
                torch.tensor(gt_labels),
                iou,
                iou_thrs,
            )
            all_tp.append(correct)
            all_conf.append(pred_scores)
            all_pred_cls.append(pred_labels)

    tp_cat = (
        np.concatenate(all_tp, axis=0)
        if all_tp
        else np.zeros((0, len(iou_thrs)), dtype=bool)
    )
    conf_cat = np.concatenate(all_conf) if all_conf else np.zeros(0)
    pred_cls_cat = (
        np.concatenate(all_pred_cls) if all_pred_cls else np.zeros(0, dtype=np.int64)
    )
    target_cls_cat = (
        np.concatenate(all_target_cls)
        if all_target_cls
        else np.zeros(0, dtype=np.int64)
    )

    stats = ap_per_class(tp_cat, conf_cat, pred_cls_cat, target_cls_cat)

    p, r, f1 = stats["p"], stats["r"], stats["f1"]
    ap50 = stats["ap50"]
    ap75 = stats["ap75"]
    map50_95 = stats["map50_95"]
    unique_classes = stats["unique_classes"]

    results = {
        "AP50": float(np.mean(ap50)) if len(ap50) else 0.0,
        "AP75": float(np.mean(ap75)) if len(ap75) else 0.0,
        "mAP": float(np.mean(map50_95)) if len(map50_95) else 0.0,
        "mAP50": float(np.mean(ap50)) if len(ap50) else 0.0,
        "mAP50_95": float(np.mean(map50_95)) if len(map50_95) else 0.0,
        "precision": float(np.mean(p)) if len(p) else 0.0,
        "recall": float(np.mean(r)) if len(r) else 0.0,
        "f1": float(np.mean(f1)) if len(f1) else 0.0,
        "seen": int(seen_imgs),
        "n_classes": int(len(unique_classes)),
    }

    for i, c in enumerate(unique_classes):
        results[f"class_{c}_precision"] = float(p[i])
        results[f"class_{c}_recall"] = float(r[i])
        results[f"class_{c}_f1"] = float(f1[i])
        results[f"class_{c}_AP50"] = float(ap50[i])
        results[f"class_{c}_AP50_95"] = float(map50_95[i]) if len(map50_95) > i else 0.0

    return results
