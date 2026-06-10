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


def _parse_det(det_path):
    """Parse prediction txt file (DOTA Task1 format)."""
    dets = []
    if not os.path.exists(det_path):
        return dets
    with open(det_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            dets.append({
                'image_id': parts[0],
                'score': float(parts[1]),
                'bbox': [float(x) for x in parts[2:10]],
            })
    return dets


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


def evaluate_dota(det_dir, gt_dir, image_list=None, iou_thr=0.5):
    """Offline DOTA evaluation.

    Args:
        det_dir:  directory containing Task1_{class}.txt prediction files
        gt_dir:   directory containing ground truth .txt files
        image_list: optional list of image names (otherwise auto-discovered)
        iou_thr:  IoU threshold (default 0.5)

    Returns:
        dict with per-class AP and mAP
    """
    if image_list is None:
        image_list = sorted(
            set(os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.endswith('.txt'))
        )

    class_aps = {}
    mAP = 0.0

    for classname in DOTA_CLASSES:
        det_path = os.path.join(det_dir, f'Task1_{classname}.txt')
        all_dets = _parse_det(det_path)

        # group detections by image
        det_by_image = {}
        for d in all_dets:
            det_by_image.setdefault(d['image_id'], []).append(d)

        # load ground truth
        npos = 0
        class_recs = {}
        for img_name in image_list:
            ann_path = os.path.join(gt_dir, f'{img_name}.txt')
            objects = _parse_gt(ann_path) if os.path.exists(ann_path) else []
            R = [obj for obj in objects if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R]) if R else np.zeros((0, 8))
            difficult = np.array([x['difficult'] == 1 for x in R], dtype=bool)
            class_recs[img_name] = {'bbox': bbox, 'difficult': difficult, 'det': [False] * len(R)}
            npos += sum(~difficult)

        # collect all predictions with scores
        image_ids = []
        confidence = []
        det_bboxes = []
        for img_name, dets in det_by_image.items():
            for d in dets:
                image_ids.append(img_name)
                confidence.append(d['score'])
                det_bboxes.append(d['bbox'])

        confidence = np.array(confidence)
        det_bboxes = np.array(det_bboxes) if det_bboxes else np.zeros((0, 8))

        # sort by confidence descending
        if len(confidence) > 0:
            sorted_ind = np.argsort(-confidence)
            det_bboxes = det_bboxes[sorted_ind]
            image_ids = [image_ids[i] for i in sorted_ind]
        else:
            sorted_ind = []

        # compute TP/FP
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d in range(nd):
            img_name = image_ids[d]
            R = class_recs.get(img_name)
            if R is None:
                fp[d] = 1.
                continue

            bb = det_bboxes[d]
            BBGT = R['bbox']

            if len(BBGT) == 0:
                fp[d] = 1.
                continue

            ious = _poly_iou_8coord(bb, BBGT)
            ovmax = ious.max() if len(ious) > 0 else -np.inf
            jmax = ious.argmax() if len(ious) > 0 else 0

            if ovmax > iou_thr:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = True
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute AP
        fp_cum = np.cumsum(fp)
        tp_cum = np.cumsum(tp)
        rec = tp_cum / float(npos) if npos > 0 else np.zeros_like(tp_cum)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
        ap = _voc_ap(rec, prec, use_07_metric=True)

        class_aps[classname] = ap
        mAP += ap

    mAP /= len(DOTA_CLASSES)
    return {'per_class': class_aps, 'mAP': mAP}


def main():
    parser = argparse.ArgumentParser(description='DOTA offline OBB evaluation')
    parser.add_argument('--det_dir', required=True, help='directory of Task1_{class}.txt files')
    parser.add_argument('--gt_dir', required=True, help='directory of ground truth txt files')
    parser.add_argument('--output', default=None, help='output file for evaluation results')
    args = parser.parse_args()

    results = evaluate_dota(args.det_dir, args.gt_dir)

    print(f"\nmAP: {results['mAP']:.4f}")
    print("\nPer-class AP:")
    for cls, ap in results['per_class'].items():
        print(f"  {cls}: {ap:.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"mAP: {results['mAP']:.4f}\n")
            for cls, ap in results['per_class'].items():
                f.write(f"{cls}: {ap:.4f}\n")


if __name__ == '__main__':
    main()
