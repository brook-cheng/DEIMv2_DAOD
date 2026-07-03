"""
Offline OBB evaluation — DOTA_devkit compatible.

Usage:
    python -m engine.eval.dota_eval \\
        --det_dir runs/exp1/results \\
        --gt_dir /path/to/dota/val/labelTxt \\
        --output dota_eval_results.txt

Prediction file format (one per class, e.g. Task1_plane.txt):
    image_name score x1 y1 x2 y2 x3 y3 x4 y4

Ground truth format (standard DOTA txt):
    x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
"""

import os
import argparse
import numpy as np
import torch
from ..deim.obb_ops import batch_probiou
from ..deim.obb_geometry import xywhr_to_xyxyxyxy


DOTA_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter',
]


def _parse_gt(ann_path):
    """Parse DOTA ground truth txt file."""
    objects = []
    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            obj = {
                'bbox': [float(x) for x in parts[:8]],
                'name': parts[8],
                'difficult': int(parts[9]) if len(parts) >= 10 else 0,
            }
            objects.append(obj)
    return objects


def _voc_ap(rec, prec, use_07_metric=True):
    """VOC AP computation."""
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


def _poly_iou_8coord(a: list, b_list: np.ndarray) -> np.ndarray:
    """IoU between one polygon (8-coord) and a set of polygons."""
    from ..deim.obb_geometry import xyxyxyxy_to_xywhr

    vert_a = np.array(a).reshape(1, 4, 2)
    vert_a_t = torch.tensor(vert_a, dtype=torch.float32)
    # use standard conversion (identical to obb_eval)
    obb_a = xyxyxyxy_to_xywhr(vert_a_t)          # (1, 5)

    verts_b = np.array([np.array(bb).reshape(4, 2) for bb in b_list]) if len(b_list) > 0 \
              else np.zeros((0, 4, 2))
    verts_b_t = torch.tensor(verts_b, dtype=torch.float32)
    obb_b = xyxyxyxy_to_xywhr(verts_b_t) if len(verts_b_t) > 0 \
             else torch.zeros(0, 5)

    if len(obb_b) == 0:
        return np.array([])

    ious = batch_probiou(obb_b, obb_a).squeeze(-1).numpy()
    return ious


def _parse_det_per_image(det_path):
    """Parse per-image prediction txt file.

    Format:  x1 y1 x2 y2 x3 y3 x4 y4 class_name score
    """
    dets = []
    if not os.path.exists(det_path):
        return dets
    with open(det_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            dets.append({
                'bbox': [float(x) for x in parts[:8]],
                'name': parts[8],
                'score': float(parts[9]),
            })
    return dets


def _load_classes(classes):
    """Load class names from file path or return list unchanged.

    Args:
        classes: str (file path, one class per line) or list[str].

    Returns:
        list of class name strings.
    """
    if isinstance(classes, str):
        with open(classes, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return list(classes)


def evaluate_dota(det_dir, gt_dir, classes, image_list=None, iouv=None):
    """Offline DOTA evaluation — aligned with obb_evaluate metrics.

    Uses ProbIoU for oriented-box matching and COCO-style 101-point AP
    interpolation (identical to obb_evaluate / obb_eval.py).

    Args:
        det_dir:   directory containing per-image prediction .txt files.
                   Each line: x1 y1 x2 y2 x3 y3 x4 y4 class_name score
        gt_dir:    directory containing per-image ground truth .txt files.
                   Each line: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
        classes:   str (path to classes.txt) or list[str] of class names.
        image_list: optional list of image names. Auto-discovered from gt_dir.
        iouv:      IoU thresholds (default: 0.5, 0.55, ..., 0.95).

    Returns:
        dict with mAP, AP50, AP75, mAP50_95, precision, recall, f1,
        per_class dict, and seen image count.
    """
    from ..eval.obb_eval import compute_ap, match_predictions, ap_per_class
    from ..deim.obb_geometry import xyxyxyxy_to_xywhr

    class_names = _load_classes(classes)
    name_to_id = {n: i for i, n in enumerate(class_names)}

    if iouv is None:
        iouv = np.linspace(0.5, 0.95, 10)
    iouv = np.asarray(iouv)

    if image_list is None:
        image_list = sorted(
            set(os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.endswith('.txt'))
        )

    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_target_cls = []

    for img_name in image_list:
        # ---- ground truth ----
        ann_path = os.path.join(gt_dir, f'{img_name}.txt')
        gt_objs = _parse_gt(ann_path) if os.path.exists(ann_path) else []
        gt_boxes_8c = [o['bbox'] for o in gt_objs]
        gt_labels = [name_to_id[o['name']] for o in gt_objs if o['name'] in name_to_id]
        # align boxes with valid labels
        valid_gt = [i for i, o in enumerate(gt_objs) if o['name'] in name_to_id]
        gt_boxes_8c = [gt_boxes_8c[i] for i in valid_gt]
        gt_labels = np.array(gt_labels, dtype=np.int64)

        all_target_cls.append(gt_labels)

        # ---- predictions ----
        det_path = os.path.join(det_dir, f'{img_name}.txt')
        dets = _parse_det_per_image(det_path)
        dets = [d for d in dets if d['name'] in name_to_id]

        if len(dets) == 0:
            all_tp.append(np.zeros((0, len(iouv)), dtype=bool))
            all_conf.append(np.zeros(0))
            all_pred_cls.append(np.zeros(0, dtype=np.int64))
            continue

        pred_boxes_8c = np.array([d['bbox'] for d in dets], dtype=np.float64)
        pred_scores = np.array([d['score'] for d in dets], dtype=np.float32)
        pred_labels = np.array([name_to_id[d['name']] for d in dets], dtype=np.int64)

        if len(gt_boxes_8c) == 0:
            all_tp.append(np.zeros((len(dets), len(iouv)), dtype=bool))
            all_conf.append(pred_scores)
            all_pred_cls.append(pred_labels)
            continue

        # 8-coord → xywhr for ProbIoU
        gt_t = xyxyxyxy_to_xywhr(torch.tensor(gt_boxes_8c, dtype=torch.float32).reshape(-1, 4, 2))
        det_t = xyxyxyxy_to_xywhr(torch.tensor(pred_boxes_8c, dtype=torch.float32).reshape(-1, 4, 2))

        iou = batch_probiou(gt_t, det_t)  # (M_gt, N_pred) — row=gt, col=pred

        correct = match_predictions(
            torch.tensor(pred_labels),
            torch.tensor(gt_labels),
            iou,
            iouv,
        )
        all_tp.append(correct)
        all_conf.append(pred_scores)
        all_pred_cls.append(pred_labels)

    tp_cat = np.concatenate(all_tp, axis=0) if all_tp else np.zeros((0, len(iouv)), dtype=bool)
    conf_cat = np.concatenate(all_conf) if all_conf else np.zeros(0)
    pred_cls_cat = np.concatenate(all_pred_cls) if all_pred_cls else np.zeros(0, dtype=np.int64)
    target_cls_cat = np.concatenate(all_target_cls) if all_target_cls else np.zeros(0, dtype=np.int64)

    stats = ap_per_class(tp_cat, conf_cat, pred_cls_cat, target_cls_cat)

    p, r, f1 = stats["p"], stats["r"], stats["f1"]
    ap50 = stats["ap50"]
    ap75 = stats["ap75"]
    map50_95 = stats["map50_95"]
    unique_classes = stats["unique_classes"]

    per_class = {}
    for idx, c in enumerate(unique_classes):
        cname = class_names[c]
        per_class[cname] = {
            "AP50":      float(ap50[idx]),
            "AP75":      float(ap75[idx]),
            "AP50_95":   float(map50_95[idx]),
            "precision": float(p[idx]),
            "recall":    float(r[idx]),
            "f1":        float(f1[idx]),
        }

    results = {
        "mAP":        float(np.mean(map50_95)) if len(map50_95) else 0.0,
        "AP50":       float(np.mean(ap50)) if len(ap50) else 0.0,
        "AP75":       float(np.mean(ap75)) if len(ap75) else 0.0,
        "mAP50_95":   float(np.mean(map50_95)) if len(map50_95) else 0.0,
        "precision":  float(np.mean(p)) if len(p) else 0.0,
        "recall":     float(np.mean(r)) if len(r) else 0.0,
        "f1":         float(np.mean(f1)) if len(f1) else 0.0,
        "per_class":  per_class,
        "seen":       len(image_list),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='DOTA offline OBB evaluation')
    parser.add_argument('--det_dir', required=True,
                        help='directory of per-image prediction .txt files')
    parser.add_argument('--gt_dir', required=True,
                        help='directory of per-image ground truth .txt files')
    parser.add_argument('--classes', required=True,
                        help='path to classes.txt (one class per line)')
    parser.add_argument('--output', default=None,
                        help='output file for evaluation results')
    args = parser.parse_args()

    results = evaluate_dota(args.det_dir, args.gt_dir, args.classes)

    print(f"\nmAP@0.5:0.95 = {results['mAP']:.4f}")
    print(f"mAP@0.5      = {results['AP50']:.4f}")
    print(f"mAP@0.75     = {results['AP75']:.4f}")
    print(f"Precision    = {results['precision']:.4f}")
    print(f"Recall       = {results['recall']:.4f}")
    print(f"F1           = {results['f1']:.4f}")
    print(f"Images evaluated: {results['seen']}")
    print("\nPer-class AP@0.5:0.95:")
    for cname, stats in results['per_class'].items():
        print(f"  {cname}: {stats['AP50_95']:.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"mAP@0.5:0.95 = {results['mAP']:.4f}\n")
            f.write(f"mAP@0.5      = {results['AP50']:.4f}\n")
            f.write(f"mAP@0.75     = {results['AP75']:.4f}\n")
            f.write(f"Precision    = {results['precision']:.4f}\n")
            f.write(f"Recall       = {results['recall']:.4f}\n")
            f.write(f"F1           = {results['f1']:.4f}\n")
            for cname, stats in results['per_class'].items():
                f.write(f"{cname}: AP50={stats['AP50']:.4f} AP75={stats['AP75']:.4f} "
                        f"AP50_95={stats['AP50_95']:.4f}\n")


if __name__ == '__main__':
    main()
