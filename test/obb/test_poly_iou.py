"""
Tests for engine/eval/poly_iou.py

Run:  python -m pytest test/test_poly_iou.py -v
"""

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from engine.eval.poly_iou import poly_iou, _polygon_area


def test_polygon_area_square():
    """Area of a unit square."""
    square = torch.tensor([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
    area = _polygon_area(square)
    assert abs(area.item() - 1.0) < 1e-6


def test_polygon_area_rotated():
    """Area of a rotated rectangle."""
    pts = torch.tensor([[[2., 0.], [2., 2.], [0., 2.], [0., 0.]]])
    area = _polygon_area(pts)
    assert abs(area.item() - 4.0) < 1e-6


def test_poly_iou_self():
    """Self IoU should be 1.0."""
    boxes = torch.tensor([[512., 512., 200., 100., 0.785398]])
    iou = poly_iou(boxes, boxes)
    assert abs(iou[0, 0].item() - 1.0) < 1e-6


def test_poly_iou_disjoint():
    """Disjoint boxes should have IoU 0."""
    boxes1 = torch.tensor([[100., 100., 50., 50., 0.]])
    boxes2 = torch.tensor([[500., 500., 50., 50., 0.]])
    iou = poly_iou(boxes1, boxes2)
    assert iou[0, 0].item() == 0.0


def test_poly_iou_overlap():
    """Two overlapping axis-aligned boxes."""
    # box1: (200,200) 100×100, box2: (250,250) 100×100
    boxes1 = torch.tensor([[200., 200., 100., 100., 0.]])
    boxes2 = torch.tensor([[250., 250., 100., 100., 0.]])
    # overlap: (200-300)x(200-300) ∩ (200-350)x(200-350) = (200-300)x(200-300) = 50×50 = 2500
    # area1 = 10000, area2 = 10000, inter = 2500, union = 17500, IoU = 2500/17500 = 0.142857
    iou = poly_iou(boxes1, boxes2)
    expected = 2500.0 / 17500.0
    assert abs(iou[0, 0].item() - expected) < 1e-4


def test_poly_iou_rotated_overlap():
    """Two identical rotated boxes shifted slightly."""
    # Same box, shifted by small amount
    boxes1 = torch.tensor([[512., 512., 200., 100., 0.5]])
    boxes2 = torch.tensor([[520., 520., 200., 100., 0.5]])
    iou = poly_iou(boxes1, boxes2)
    assert 0.0 < iou[0, 0].item() < 1.0


def test_poly_iou_matrix():
    """N×M IoU matrix."""
    boxes1 = torch.tensor([
        [200., 200., 100., 100., 0.],
        [400., 400., 100., 100., 0.],
    ])
    boxes2 = torch.tensor([
        [250., 250., 100., 100., 0.],
        [450., 450., 100., 100., 0.],
    ])
    iou = poly_iou(boxes1, boxes2)
    assert iou.shape == (2, 2)
    assert iou[0, 0].item() > 0
    assert iou[1, 1].item() > 0
    assert iou[0, 1].item() == 0.0  # disjoint


def test_poly_iou_roundtrip():
    """IoU should be symmetric."""
    boxes1 = torch.tensor([[512., 512., 200., 100., 0.785398]])
    boxes2 = torch.tensor([[520., 515., 180., 90., 0.8]])
    iou12 = poly_iou(boxes1, boxes2)
    iou21 = poly_iou(boxes2, boxes1)
    assert abs(iou12.item() - iou21.item()) < 1e-6


def test_poly_iou_batch_random():
    """Batch of random OBBs."""
    torch.manual_seed(42)
    N, M = 5, 3
    boxes1 = torch.rand(N, 5)
    boxes1[:, :2] *= 800
    boxes1[:, 2:4] *= 200
    boxes1[:, 4] *= torch.pi

    boxes2 = torch.rand(M, 5)
    boxes2[:, :2] *= 800
    boxes2[:, 2:4] *= 200
    boxes2[:, 4] *= torch.pi

    iou = poly_iou(boxes1, boxes2)
    assert iou.shape == (N, M)
    assert (iou >= 0).all()
    assert (iou <= 1).all()


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
