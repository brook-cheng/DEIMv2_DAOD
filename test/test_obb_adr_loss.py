"""OBB ADR decomposition loss tests (spec 2026-08-04-obb-adr-loss-design).

Covers the adr_loss flag, weight_dict validation, the ADR loss_boxes
branch (external rect L1/GIoU + offset L1 + optional KLD), and edge
cases (empty matches, keep_kld=False, gradient flow).

Run:
    pytest test/test_obb_adr_loss.py -v
"""

import math

import pytest
import torch

from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.obb_geometry import oriented_box_to_external_rect
from engine.deim.obb_ops import kld_loss
from engine.deim.box_ops import generalized_box_iou, box_xyxy_to_cxcywh


ADR_WEIGHTS = {
    "loss_extrect_l1": 5.0,
    "loss_extrect_giou": 2.0,
    "loss_offset_l1": 1.0,
    "loss_kld": 2.0,
}


def _adr_criterion(*, keep_kld=True, weights=None, **kwargs):
    """Build a DEIMCriterion with adr_loss=True for direct loss_boxes
    testing. matcher is None because loss_boxes does not invoke it.
    """
    w = weights if weights is not None else dict(ADR_WEIGHTS)
    return DEIMCriterion(
        matcher=None,
        weight_dict=w,
        losses=["boxes"],
        num_classes=1,
        box_mode="obb",
        adr_loss=True,
        keep_kld=keep_kld,
        **kwargs,
    )


def _pair(pred, target, *, requires_grad=False):
    """Build matched outputs/targets/indices for a single-box case."""
    pred_boxes = torch.tensor(
        [pred], dtype=torch.float32, requires_grad=requires_grad
    )
    outputs = {"pred_boxes": pred_boxes.unsqueeze(0)}
    targets = [{"boxes": torch.tensor([target]), "labels": torch.tensor([0])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    return pred_boxes, outputs, targets, indices


# ---------------------------------------------------------------------------
# Task 1: adr_loss flag and weight_dict validation
# ---------------------------------------------------------------------------

def test_adr_init_accepts_flag():
    """adr_loss=True with full ADR weights must construct without error."""
    criterion = _adr_criterion()
    assert criterion.adr_loss is True


def test_adr_flag_defaults_false():
    """adr_loss must default to False; legacy construction is unchanged."""
    criterion = DEIMCriterion(
        matcher=None,
        weight_dict={"loss_bbox": 2.0, "loss_kld": 1.0},
        losses=["boxes"],
        num_classes=1,
        box_mode="obb",
    )
    assert criterion.adr_loss is False


@pytest.mark.parametrize(
    "weights",
    [
        # missing loss_extrect_l1
        {"loss_extrect_giou": 2.0, "loss_offset_l1": 1.0, "loss_kld": 2.0},
        # missing loss_extrect_giou
        {"loss_extrect_l1": 5.0, "loss_offset_l1": 1.0, "loss_kld": 2.0},
        # missing loss_offset_l1
        {"loss_extrect_l1": 5.0, "loss_extrect_giou": 2.0, "loss_kld": 2.0},
        # keep_kld=True -> loss_kld required
        {"loss_extrect_l1": 5.0, "loss_extrect_giou": 2.0, "loss_offset_l1": 1.0},
    ],
)
def test_adr_missing_weight_raises(weights):
    """adr_loss=True must raise ValueError naming the missing key."""
    with pytest.raises(ValueError, match="loss_"):
        _adr_criterion(weights=weights)


def test_adr_missing_weight_error_names_key():
    """The ValueError message must contain the missing key name."""
    weights = {"loss_extrect_l1": 5.0, "loss_extrect_giou": 2.0}
    with pytest.raises(ValueError, match="loss_offset_l1"):
        _adr_criterion(weights=weights)


def test_adr_missing_weight_does_not_mutate_dict():
    """Raising must not mutate the caller's weight_dict."""
    weights = {"loss_extrect_l1": 5.0, "loss_extrect_giou": 2.0}
    original = weights.copy()
    with pytest.raises(ValueError):
        _adr_criterion(weights=weights)
    assert weights == original


def test_adr_nokld_does_not_require_loss_kld():
    """keep_kld=False with adr_loss=True must not require loss_kld."""
    weights = {"loss_extrect_l1": 5.0, "loss_extrect_giou": 2.0, "loss_offset_l1": 1.0}
    criterion = _adr_criterion(keep_kld=False, weights=weights)
    assert criterion.keep_kld is False


# ---------------------------------------------------------------------------
# Task 2: ADR loss_boxes branch (main path)
# ---------------------------------------------------------------------------

# Axis-aligned manual anchors:
# pred [0.5,0.5,0.4,0.2,0.0] -> ext (0.3,0.4,0.7,0.6) -> cxcywh (0.5,0.5,0.4,0.2)
# tgt  [0.55,0.45,0.5,0.3,0.0] -> ext (0.3,0.3,0.8,0.6) -> cxcywh (0.55,0.45,0.5,0.3)
# L1 = 0.05+0.05+0.10+0.10 = 0.30
_AA_PRED = [0.5, 0.5, 0.4, 0.2, 0.0]
_AA_TGT = [0.55, 0.45, 0.5, 0.3, 0.0]


def test_adr_keys_and_no_angle_loss():
    """ADR path must return ext-rect/offset/kld keys and never
    loss_angle / loss_probiou / loss_bbox."""
    pred = [0.6, 0.55, 0.35, 0.15, 0.4]
    target = [0.5, 0.5, 0.3, 0.2, 0.5]
    _, outputs, targets, indices = _pair(pred, target)
    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)

    assert set(losses) == {
        "loss_extrect_l1", "loss_extrect_giou", "loss_offset_l1", "loss_kld"
    }
    assert "loss_angle" not in losses
    assert "loss_probiou" not in losses
    assert "loss_bbox" not in losses


def test_adr_extrect_l1_axis_aligned_manual():
    """loss_extrect_l1 must equal manual L1 on (cx,cy,ext_w,ext_h)."""
    _, outputs, targets, indices = _pair(_AA_PRED, _AA_TGT)
    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)

    expected = torch.tensor(0.30)
    assert torch.allclose(losses["loss_extrect_l1"], expected, atol=1e-6), (
        f"loss_extrect_l1={losses['loss_extrect_l1'].item():.6f} != 0.30"
    )


def test_adr_extrect_giou_axis_aligned_manual():
    """loss_extrect_giou must equal 1 - GIoU on xyxy external rects.

    pred ext (0.3,0.4,0.7,0.6) vs tgt ext (0.4,0.4,0.6,0.6):
    IoU=0.5, convex hull area = union area -> GIoU=0.5 -> loss=0.5.
    """
    pred = [0.5, 0.5, 0.4, 0.2, 0.0]
    target = [0.5, 0.5, 0.2, 0.2, 0.0]
    _, outputs, targets, indices = _pair(pred, target)
    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)

    expected = torch.tensor(0.5)
    assert torch.allclose(losses["loss_extrect_giou"], expected, atol=1e-6), (
        f"loss_extrect_giou={losses['loss_extrect_giou'].item():.6f} != 0.5"
    )


def test_adr_rotated_components_match_production_decomposition():
    """For a rotated pair, all three component losses must equal manual
    values computed from the production decomposition."""
    pred = [0.5, 0.5, 0.4, 0.2, math.pi / 4]
    target = [0.5, 0.5, 0.4, 0.2, 0.5]
    _, outputs, targets, indices = _pair(pred, target)

    pred_t = torch.tensor([pred], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)
    ext_p, off_p = oriented_box_to_external_rect(pred_t)
    ext_t, off_t = oriented_box_to_external_rect(target_t)

    exp_l1 = F_l1_on(ext_p, ext_t)
    exp_off = (off_p - off_t).abs().sum(-1).sum()
    exp_giou = 1 - torch.diag(generalized_box_iou(ext_p, ext_t)).sum()

    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)
    assert torch.allclose(losses["loss_extrect_l1"], exp_l1, atol=1e-6)
    assert torch.allclose(losses["loss_extrect_giou"], exp_giou, atol=1e-6)
    assert torch.allclose(losses["loss_offset_l1"], exp_off, atol=1e-6)


def F_l1_on(a, b):
    """Helper: sum of |cxcywh(a) - cxcywh(b)| (matches spec 5.2 formula)."""
    return (box_xyxy_to_cxcywh(a) - box_xyxy_to_cxcywh(b)).abs().sum()


def test_adr_kld_matches_production():
    """loss_kld must equal kld_loss(reduction='none').sum() / num_boxes."""
    pred = [0.6, 0.55, 0.35, 0.15, 0.4]
    target = [0.5, 0.5, 0.3, 0.2, 0.5]
    _, outputs, targets, indices = _pair(pred, target)
    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)

    pred_t = torch.tensor([pred], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)
    expected = kld_loss(pred_t, target_t, reduction="none").sum()
    assert torch.allclose(losses["loss_kld"], expected, atol=1e-6)


def test_adr_empty_matches_return_finite_zeros():
    """Empty matched pairs must return scalar zero for every ADR key."""
    outputs = {"pred_boxes": torch.zeros(1, 0, 5)}
    targets = [{"boxes": torch.zeros(0, 5), "labels": torch.zeros(0, dtype=torch.long)}]
    empty = torch.zeros(0, dtype=torch.long)

    losses = _adr_criterion().loss_boxes(outputs, targets, [(empty, empty)], 1.0)

    assert set(losses) == set(ADR_WEIGHTS)
    assert all(v.ndim == 0 and v.item() == 0.0 for v in losses.values())


# ---------------------------------------------------------------------------
# Task 3: boundary conditions and gradient quality
# ---------------------------------------------------------------------------

def test_adr_nokld_omits_kld_key():
    """keep_kld=False must omit loss_kld and keep the three ADR keys."""
    pred = [0.6, 0.55, 0.35, 0.15, 0.4]
    target = [0.5, 0.5, 0.3, 0.2, 0.5]
    _, outputs, targets, indices = _pair(pred, target)

    weights = {
        "loss_extrect_l1": 5.0,
        "loss_extrect_giou": 2.0,
        "loss_offset_l1": 1.0,
    }
    losses = _adr_criterion(keep_kld=False, weights=weights).loss_boxes(
        outputs, targets, indices, 1.0
    )

    assert set(losses) == {"loss_extrect_l1", "loss_extrect_giou", "loss_offset_l1"}
    assert "loss_kld" not in losses
    assert torch.isfinite(torch.stack(list(losses.values()))).all()


def test_adr_nokld_empty_returns_three_zeros():
    """keep_kld=False + empty matches must return exactly three zero keys."""
    outputs = {"pred_boxes": torch.zeros(1, 0, 5)}
    targets = [{"boxes": torch.zeros(0, 5), "labels": torch.zeros(0, dtype=torch.long)}]
    empty = torch.zeros(0, dtype=torch.long)
    weights = {
        "loss_extrect_l1": 5.0,
        "loss_extrect_giou": 2.0,
        "loss_offset_l1": 1.0,
    }

    losses = _adr_criterion(keep_kld=False, weights=weights).loss_boxes(
        outputs, targets, [(empty, empty)], 1.0
    )

    assert set(losses) == {"loss_extrect_l1", "loss_extrect_giou", "loss_offset_l1"}
    assert all(v.ndim == 0 and v.item() == 0.0 for v in losses.values())


def test_adr_gradient_flows_finite_and_center_directed():
    """Backward through the ADR losses must yield finite gradients with
    non-zero cx/cy components.

    NOTE: vertex offsets are computed via argmin/argmax gather, which
    does not backpropagate through the selected vertex — the offset
    terms contribute gradient only via x_max/y_max. This is a known
    geometric property of oriented_box_to_external_rect; the assertion
    therefore locks finiteness everywhere and non-zero cx/cy only.
    """
    pred, outputs, targets, indices = _pair(
        [0.5, 0.5, 0.4, 0.2, 0.3],
        [0.55, 0.45, 0.3, 0.2, 0.5],
        requires_grad=True,
    )
    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)

    sum(losses.values()).backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all(), f"non-finite grad: {pred.grad}"
    assert pred.grad[0, 0] != 0.0, "gradient on cx must be non-zero"
    assert pred.grad[0, 1] != 0.0, "gradient on cy must be non-zero"


def test_adr_giou_identical_boxes_is_zero():
    """Perfectly matched external rects must give loss_extrect_giou == 0."""
    box = [0.5, 0.5, 0.4, 0.2, 0.0]
    _, outputs, targets, indices = _pair(box, box)
    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)

    assert torch.allclose(losses["loss_extrect_giou"], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(losses["loss_offset_l1"], torch.tensor(0.0), atol=1e-6)
