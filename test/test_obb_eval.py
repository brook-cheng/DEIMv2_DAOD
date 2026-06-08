"""
Integration test for engine/eval/obb_eval.py

Run:  python -m pytest test/test_obb_eval.py -v
"""

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.eval.poly_iou import poly_iou
from engine.eval.obb_eval import _voc_ap, _tpfp
import numpy as np


def test_voc_ap_perfect():
    """Perfect detection: AP should be 1.0."""
    rec = np.array([0.0, 0.5, 1.0, 1.0, 1.0])
    prec = np.array([1.0, 1.0, 1.0, 0.75, 0.6])
    ap = _voc_ap(rec, prec, use_07_metric=True)
    assert abs(ap - 1.0) < 1e-6


def test_voc_ap_zero():
    """No ground truths: AP is 0 (undefined recall)."""
    rec = np.array([])
    prec = np.array([])
    ap = _voc_ap(rec, prec, use_07_metric=True)
    assert ap == 0.0


def test_tpfp_perfect_match():
    """All detections perfectly match GTs."""
    det = np.array([[100., 100., 50., 50., 0., 0.9]])     # 1 detection
    gt  = np.array([[100., 100., 50., 50., 0.]])          # 1 GT
    tp, fp = _tpfp(det, gt, iou_thr=0.5)
    assert tp.sum() == 1
    assert fp.sum() == 0


def test_tpfp_no_detections():
    """No detections, some GTs: all TP=0."""
    det = np.zeros((0, 6))
    gt  = np.array([[100., 100., 50., 50., 0.]])
    tp, fp = _tpfp(det, gt, iou_thr=0.5)
    assert len(tp) == 0


def test_tpfp_no_gts():
    """Detections but no GTs: all FP."""
    det = np.array([[100., 100., 50., 50., 0., 0.9]])
    gt  = np.zeros((0, 5))
    tp, fp = _tpfp(det, gt, iou_thr=0.5)
    assert tp.sum() == 0
    assert fp.sum() == 1


def test_tpfp_double_match():
    """Two detections, one GT: higher score wins, second is FP."""
    det = np.array([
        [100., 100., 50., 50., 0., 0.9],   # matches GT
        [108., 108., 50., 50., 0., 0.8],   # same GT, should be FP
    ])
    gt  = np.array([[100., 100., 50., 50., 0.]])
    tp, fp = _tpfp(det, gt, iou_thr=0.5)
    assert tp.sum() == 1
    assert fp.sum() == 1


def test_poly_iou_self_vs_1():
    """N identical boxes should have IoU=1 on diagonal."""
    N = 5
    boxes = torch.rand(N, 5)
    boxes[:, :2] *= 800
    boxes[:, 2:4] *= 200
    boxes[:, 4] *= torch.pi
    iou = poly_iou(boxes, boxes)
    for i in range(N):
        assert abs(iou[i, i].item() - 1.0) < 1e-4, f"diag[{i}] != 1.0: {iou[i,i]}"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
