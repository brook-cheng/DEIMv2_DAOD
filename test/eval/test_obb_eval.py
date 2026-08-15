"""
Integration test for engine/eval/obb_eval.py

Run:  python -m pytest test/test_obb_eval.py -v
"""

import torch
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from engine.eval.obb_eval import ap_per_class, compute_ap, match_predictions
from engine.deim.obb_ops import batch_probiou


def _make_iou(det_boxes, gt_boxes):
    """Helper: compute ProbIoU matrix from det/gt arrays."""
    if len(det_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(det_boxes), len(gt_boxes)), dtype=np.float32)
    det_t = torch.tensor(det_boxes[:, :5], dtype=torch.float32)
    gt_t  = torch.tensor(gt_boxes, dtype=torch.float32)
    return batch_probiou(det_t, gt_t).numpy()


def test_compute_ap_perfect():
    rec = np.array([0.0, 0.5, 1.0, 1.0, 1.0])
    prec = np.array([1.0, 1.0, 1.0, 0.75, 0.6])
    ap, _, _ = compute_ap(rec, prec)
    assert abs(ap - 0.995) < 1e-6


def test_ap_per_class_no_predictions():
    stats = ap_per_class(
        np.zeros((0, 1), dtype=bool),
        np.zeros(0),
        np.zeros(0, dtype=np.int64),
        np.array([0], dtype=np.int64),
    )
    assert stats["ap"].shape == (1, 1)
    assert stats["ap50"].tolist() == [0.0]


def test_match_predictions_perfect_match():
    """All detections perfectly match GTs."""
    det = np.array([[100., 100., 50., 50., 0., 0.9]])
    gt  = np.array([[100., 100., 50., 50., 0.]])
    ious = _make_iou(det, gt)
    correct = match_predictions(
        np.zeros(len(det), dtype=np.int64),
        np.zeros(len(gt), dtype=np.int64),
        ious.T,
        (0.5,),
    )
    assert correct.sum() == 1


def test_match_predictions_no_detections():
    """No detections, some GTs: all TP=0."""
    det = np.zeros((0, 6))
    gt  = np.array([[100., 100., 50., 50., 0.]])
    ious = _make_iou(det, gt)
    correct = match_predictions(
        np.zeros(0, dtype=np.int64),
        np.zeros(len(gt), dtype=np.int64),
        ious.T,
        (0.5,),
    )
    assert correct.shape == (0, 1)


def test_match_predictions_no_gts():
    """Detections but no GTs: all FP."""
    det = np.array([[100., 100., 50., 50., 0., 0.9]])
    gt  = np.zeros((0, 5))
    ious = _make_iou(det, gt)
    correct = match_predictions(
        np.zeros(len(det), dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        ious.T,
        (0.5,),
    )
    assert correct.shape == (1, 1)
    assert not correct.any()


def test_match_predictions_double_match():
    """Two detections, one GT: higher score wins, second is FP."""
    det = np.array([
        [100., 100., 50., 50., 0., 0.9],
        [108., 108., 50., 50., 0., 0.8],
    ])
    gt  = np.array([[100., 100., 50., 50., 0.]])
    ious = _make_iou(det, gt)
    correct = match_predictions(
        np.zeros(len(det), dtype=np.int64),
        np.zeros(len(gt), dtype=np.int64),
        ious.T,
        (0.5,),
    )
    assert correct.sum() == 1
    assert (~correct[:, 0]).sum() == 1


def test_poly_iou_self_vs_1():
    """N identical boxes should have IoU=1 on diagonal."""
    from engine.deim.obb_ops import batch_probiou
    N = 5
    boxes = torch.rand(N, 5)
    boxes[:, :2] *= 800
    boxes[:, 2:4] *= 200
    boxes[:, 4] *= torch.pi
    iou = batch_probiou(boxes, boxes)
    for i in range(N):
        assert abs(iou[i, i].item() - 1.0) < 1e-3


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
