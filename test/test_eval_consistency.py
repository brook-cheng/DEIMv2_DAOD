"""
Consistency test: online AP vs offline evaluate_dota on real DOTA data.

Both paths receive identical predictions (GTs perturbed by small noise).
Per-class AP must match within 0.005.

Run:  python -m pytest test/test_eval_consistency.py -v -s
"""

import os, sys, tempfile, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data.dataset.dota_dataset import DotaDataset
from engine.eval.obb_eval import _voc_ap, _tpfp
from engine.eval.dota_eval import evaluate_dota, DOTA_CLASSES
from engine.deim.obb_ops import batch_probiou
from engine.deim.obb_geometry import xywhr_to_xyxyxyxy


DATA = {
    "img_folder": "/mnt/d/project_data/model_test/dota128_ultralystic/images/val",
    "ann_folder": "/mnt/d/project_data/model_test/dota128_ultralystic/labels_dota/val",
    "classes_file": "/mnt/d/project_data/model_test/dota128_ultralystic/classes.txt",
}
NUM_CLASSES = 15
IOU_THR = 0.5


def _perturb_gt(gt_boxes, rng):
    preds = []
    for gt in gt_boxes:
        cx, cy, w, h, t = gt
        for _ in range(2):
            dcx, dcy = rng.uniform(-0.05, 0.05) * w, rng.uniform(-0.05, 0.05) * h
            dw, dh   = rng.uniform(-0.10, 0.10) * w, rng.uniform(-0.10, 0.10) * h
            dt       = rng.uniform(-0.1, 0.1) * np.pi
            preds.append([cx + dcx, cy + dcy, max(w + dw, 4), max(h + dh, 4), (t + dt) % np.pi])
    return np.array(preds, dtype=np.float32) if preds else np.zeros((0, 5), dtype=np.float32)


def _compute_online_aps(all_gt_boxes, all_gt_labels,
                         all_pred_boxes, all_pred_scores, all_pred_labels):
    """Run online AP computation — per-image TP/FP, matching evaluate_dota."""
    n_imgs = len(all_gt_boxes)
    aps = []
    for cls_id in range(NUM_CLASSES):
        all_tp, all_fp, all_scores = [], [], []
        total_gt = 0
        for i in range(n_imgs):
            gb, gl = all_gt_boxes[i], all_gt_labels[i]
            pb, ps, pl = all_pred_boxes[i], all_pred_scores[i], all_pred_labels[i]
            mp, mg = pl == cls_id, gl == cls_id
            det = np.concatenate([pb[mp], ps[mp, None]], axis=1) if mp.any() else np.zeros((0, 6), dtype=np.float32)
            gt  = gb[mg].astype(np.float32) if mg.any() else np.zeros((0, 5), dtype=np.float32)
            total_gt += len(gt)
            if len(det) == 0 or len(gt) == 0:
                if len(det) > 0:
                    all_tp.append(np.zeros(len(det), dtype=np.float32))
                    all_fp.append(np.ones(len(det), dtype=np.float32))
                    all_scores.append(det[:, 5])
                continue
            det_t = torch.tensor(det[:, :5], dtype=torch.float32)
            gt_t  = torch.tensor(gt, dtype=torch.float32)
            ious = batch_probiou(det_t, gt_t).numpy()
            tp, fp = _tpfp(ious, det[:, 5], IOU_THR)
            all_tp.append(tp); all_fp.append(fp); all_scores.append(det[:, 5])
        if total_gt == 0:
            aps.append(0.0); continue
        tp_cat = np.concatenate(all_tp); fp_cat = np.concatenate(all_fp)
        s_cat  = np.concatenate(all_scores)
        idx = np.argsort(-s_cat)
        tp_cum = np.cumsum(tp_cat[idx]); fp_cum = np.cumsum(fp_cat[idx])
        eps = np.finfo(np.float32).eps
        rec = tp_cum / max(total_gt, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, eps)
        aps.append(_voc_ap(rec, prec, use_07_metric=True))
    return aps


def test_online_vs_offline_ap():
    rng = np.random.RandomState(99)

    ds = DotaDataset(**DATA, transforms=None)
    all_gt_boxes, all_gt_labels, img_names = [], [], []
    for idx in range(len(ds)):
        _, tgt = ds[idx]
        img_file = list(ds._img_ann_dict.keys())[idx]
        img_names.append(os.path.splitext(img_file)[0])
        all_gt_boxes.append(tgt["boxes"].numpy().astype(np.float32))
        all_gt_labels.append(tgt["labels"].numpy().astype(np.int64))

    img_pred_boxes, img_pred_scores, img_pred_labels = [], [], []
    for i in range(len(ds)):
        gb, gl = all_gt_boxes[i], all_gt_labels[i]
        n_gt = len(gl)
        if n_gt == 0:
            img_pred_boxes.append(np.zeros((0, 5), dtype=np.float32))
            img_pred_scores.append(np.zeros(0, dtype=np.float32))
            img_pred_labels.append(np.zeros(0, dtype=np.int64))
            continue
        preds = _perturb_gt(gb, rng)
        assert len(preds) == 2 * n_gt, f"Image {i}: expected {2*n_gt} preds, got {len(preds)} (GTs: {n_gt})"
        img_pred_boxes.append(preds)
        img_pred_scores.append(rng.uniform(0.3, 1.0, len(preds)).astype(np.float32))
        img_pred_labels.append(np.repeat(gl, 2).astype(np.int64))

    online_aps = _compute_online_aps(all_gt_boxes, all_gt_labels,
                                      img_pred_boxes, img_pred_scores, img_pred_labels)

    tmpdir = tempfile.mkdtemp()
    try:
        det_dir = os.path.join(tmpdir, "dets")
        gt_dir  = os.path.join(tmpdir, "gts")
        os.makedirs(det_dir); os.makedirs(gt_dir)

        for i, name in enumerate(img_names):
            lines = []
            for j in range(len(all_gt_boxes[i])):
                v = xywhr_to_xyxyxyxy(
                    torch.tensor(all_gt_boxes[i][j]).reshape(1, 5)).numpy().flatten()
                lines.append(" ".join([f"{x:.8f}" for x in v]) +
                             f" {DOTA_CLASSES[int(all_gt_labels[i][j])]} 0\n")
            with open(os.path.join(gt_dir, f"{name}.txt"), "w") as f:
                f.writelines(lines)

        for i, name in enumerate(img_names):
            lines = []
            pb = img_pred_boxes[i]; ps = img_pred_scores[i]; pl = img_pred_labels[i]
            for j in range(len(pb)):
                v = xywhr_to_xyxyxyxy(torch.tensor(pb[j]).reshape(1, 5)).numpy().flatten()
                lines.append(" ".join([f"{x:.8f}" for x in v]) +
                             f" {DOTA_CLASSES[int(pl[j])]} {ps[j]:.8f}\n")
            with open(os.path.join(det_dir, f"{name}.txt"), "w") as f:
                f.writelines(lines)

        result_offline = evaluate_dota(det_dir, gt_dir)
    finally:
        shutil.rmtree(tmpdir)

    offline_aps = [result_offline["per_class"].get(c, 0.0) for c in DOTA_CLASSES]

    print(f"\n{'Class':<20s} {'Online':>8s} {'Offline':>8s} {'Delta':>8s} {'GT':>5s}")
    print("-" * 55)
    max_delta = 0.0
    for cls_id in range(NUM_CLASSES):
        ol, of = online_aps[cls_id], offline_aps[cls_id]
        delta = abs(ol - of)
        max_delta = max(max_delta, delta)
        n_gt = sum((all_gt_labels[i] == cls_id).sum() for i in range(len(all_gt_labels)))
        s = "✓" if delta < 0.005 else "✗"
        print(f"{DOTA_CLASSES[cls_id]:<20s} {ol:8.6f} {of:8.6f} {delta:8.6f} {n_gt:5d}  {s}")

    print(f"\n  Online mAP: {np.mean(online_aps):.6f}")
    print(f"  Offline mAP: {result_offline['mAP']:.6f}")
    print(f"  Max delta:   {max_delta:.6f}")

    assert max_delta < 0.005, f"Max per-class AP delta {max_delta:.6f} > 0.005"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
