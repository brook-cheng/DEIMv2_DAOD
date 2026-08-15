import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.deim.matcher import HungarianMatcher


def _matcher(**overrides):
    weight_dict = {
        "cost_class": 0.0,
        "cost_bbox": 1.0,
        "cost_giou": 0.0,
        "cost_chamfer": 0.0,
        "cost_probiou": 0.0,
        "cost_angle": 0.0,
        "late_cost_bbox": 0.0,
    }
    kwargs = {
        "weight_dict": weight_dict,
        "change_matcher": False,
        "iou_order_alpha": 1.0,
        "matcher_change_epoch": 10_000,
        "box_mode": "obb",
        "use_focal_loss": False,
        "angle_order_alpha": 1.0,
    }
    for key, value in overrides.items():
        if key in weight_dict:
            weight_dict[key] = value
        else:
            kwargs[key] = value
    return HungarianMatcher(**kwargs)


def _run(matcher, preds, tgts, *, epoch):
    outputs = {
        "pred_boxes": torch.tensor(preds, dtype=torch.float32).unsqueeze(0),
        "pred_logits": torch.zeros(1, len(preds), 1, dtype=torch.float32),
    }
    tgt_boxes = torch.tensor(tgts, dtype=torch.float32)
    targets = [{"boxes": tgt_boxes, "labels": torch.zeros(len(tgts), dtype=torch.int64)}]
    return matcher(outputs, targets, epoch=epoch)["indices"][0]


def _selected(matcher, preds, tgts, *, epoch):
    src_idx, _ = _run(matcher, preds, tgts, epoch=epoch)
    return src_idx.tolist()


def test_early_phase_preserves_canonical_symmetry_and_angle_control():
    symmetry_preds = [
        [0.5, 0.5, 0.2, 0.4, math.pi / 2],
        [0.5, 0.5, 0.4, 0.2, 0.0],
    ]
    symmetry_tgts = [[0.5, 0.5, 0.4, 0.2, 0.0]]
    assert _selected(_matcher(cost_bbox=1.0), symmetry_preds, symmetry_tgts, epoch=0) == [0]

    angle_preds = [
        [0.5, 0.5, 0.3, 0.2, math.pi / 4],
        [0.5, 0.5, 0.3, 0.2, 0.1],
    ]
    angle_tgts = [[0.5, 0.5, 0.3, 0.2, 0.0]]
    assert _selected(_matcher(cost_bbox=1.0), angle_preds, angle_tgts, epoch=0) == [0]
    assert _selected(
        _matcher(cost_bbox=1.0, cost_angle=3.0),
        angle_preds,
        angle_tgts,
        epoch=0,
    ) == [1]


def test_late_phase_uses_independent_l1_and_zero_weight_removes_it():
    preds = [
        [0.9, 0.5, 0.3, 0.2, 0.0],
        [0.8, 0.5, 0.3, 0.2, 0.0],
    ]
    tgts = [[0.5, 0.5, 0.3, 0.2, 0.0]]
    base = dict(
        cost_bbox=1.0,
        cost_class=1.0,
        cost_probiou=1.0,
        change_matcher=True,
        matcher_change_epoch=0,
        iou_order_alpha=0.0,
    )
    assert _selected(_matcher(**base), preds, tgts, epoch=1) == [0]
    assert _selected(_matcher(**base, late_cost_bbox=0.25), preds, tgts, epoch=1) == [1]

    quality_preds = [
        [0.51, 0.5, 0.3, 0.2, 0.0],
        [0.6, 0.5, 0.3, 0.2, 0.0],
    ]
    assert _selected(_matcher(**base), quality_preds, tgts, epoch=1) == [0]


def test_late_zero_coefficients_restore_legacy_formula_and_accept_angle_order_alpha():
    preds = [
        [0.6, 0.5, 0.3, 0.2, 0.0],
        [0.7, 0.5, 0.3, 0.2, 0.0],
    ]
    tgts = [[0.5, 0.5, 0.3, 0.2, 0.0]]
    assert _selected(
        _matcher(
            cost_bbox=1.0,
            cost_class=1.0,
            cost_probiou=1.0,
            change_matcher=True,
            matcher_change_epoch=0,
        ),
        preds,
        tgts,
        epoch=1,
    ) == [0]

    angle_matcher = _matcher(
        cost_bbox=0.0,
        cost_class=1.0,
        cost_probiou=1.0,
        cost_angle=3.0,
        change_matcher=True,
        matcher_change_epoch=0,
        angle_order_alpha=2.0,
    )
    angle_preds = [
        [0.5, 0.5, 0.3, 0.2, math.pi / 4],
        [0.5, 0.5, 0.3, 0.2, 0.1],
    ]
    assert _selected(angle_matcher, angle_preds, tgts, epoch=1) == [1]


@pytest.mark.parametrize("epoch", [0, 1])
def test_empty_targets_return_empty_indices(epoch):
    src_idx, tgt_idx = _run(
        _matcher(change_matcher=True, matcher_change_epoch=0),
        [[0.5, 0.5, 0.3, 0.2, 0.0]],
        [],
        epoch=epoch,
    )
    assert src_idx.numel() == 0
    assert tgt_idx.numel() == 0


def test_random_valid_obb_costs_are_finite():
    torch.manual_seed(42)
    preds = torch.rand(10, 5, dtype=torch.float32)
    preds[:, 2:4] = preds[:, 2:4] * 0.5 + 0.1
    preds[:, 4] = preds[:, 4] * math.pi
    tgts = torch.rand(3, 5, dtype=torch.float32)
    tgts[:, 2:4] = tgts[:, 2:4] * 0.5 + 0.1
    tgts[:, 4] = tgts[:, 4] * math.pi
    outputs = {
        "pred_boxes": preds.unsqueeze(0),
        "pred_logits": torch.zeros(1, 10, 1, dtype=torch.float32),
    }
    targets = [{"boxes": tgts, "labels": torch.zeros(3, dtype=torch.int64)}]
    result = _matcher(
        cost_bbox=2.0,
        cost_class=2.0,
        cost_probiou=4.0,
        cost_angle=3.0,
        cost_chamfer=5.0,
        late_cost_bbox=0.25,
        change_matcher=True,
        matcher_change_epoch=0,
    )(outputs, targets, epoch=1)
    src_idx, tgt_idx = result["indices"][0]
    assert src_idx.numel() == 3
    assert tgt_idx.numel() == 3
    assert src_idx.max().item() < 10
    assert tgt_idx.max().item() < 3


def test_hbb_matching_is_unchanged():
    outputs = {
        "pred_boxes": torch.tensor(
            [[[0.6, 0.5, 0.3, 0.2], [0.5, 0.5, 0.3, 0.2]]],
            dtype=torch.float32,
        ),
        "pred_logits": torch.zeros(1, 2, 1, dtype=torch.float32),
    }
    targets = [{
        "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.2]], dtype=torch.float32),
        "labels": torch.zeros(1, dtype=torch.int64),
    }]
    result = _matcher(cost_class=1.0, cost_giou=1.0, box_mode="hbb")(outputs, targets, epoch=0)
    assert result["indices"][0][0].tolist() == [1]


def test_matcher_imports_on_disk_repo_module():
    import engine.deim.matcher as mod

    expected = os.path.realpath(os.path.join(ROOT, "engine", "deim", "matcher.py"))
    assert os.path.realpath(mod.__file__) == expected
