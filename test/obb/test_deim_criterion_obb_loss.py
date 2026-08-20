import math
import types

import pytest
import torch

from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.obb_geometry import periodic_angle_distance


LEGACY_WEIGHTS = {"loss_bbox": 2.0, "loss_kld": 1.0}
NEW_WEIGHTS = {
    "loss_bbox": 2.0,
    "loss_probiou": 5.0,
    "loss_angle": 3.0,
    "loss_kld": 1.0,
}


def _criterion(*, weights=LEGACY_WEIGHTS, box_mode="obb", **kwargs):
    return DEIMCriterion(
        matcher=None,
        weight_dict=weights,
        losses=["boxes"],
        num_classes=1,
        box_mode=box_mode,
        **kwargs,
    )


def _pair(pred, target, *, requires_grad=False):
    pred_boxes = torch.tensor([pred], dtype=torch.float32, requires_grad=requires_grad)
    outputs = {"pred_boxes": pred_boxes.unsqueeze(0)}
    targets = [{"boxes": torch.tensor([target]), "labels": torch.tensor([0])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    return pred_boxes, outputs, targets, indices


def _new_criterion(*, keep_kld=True, weights=NEW_WEIGHTS):
    return _criterion(
        weights=weights,
        use_yolo_probiou=True,
        use_yolo_angle=True,
        keep_kld=keep_kld,
        angle_lambda=3.0,
    )


def test_legacy_obb_formula_and_keys_are_unchanged():
    pred = [0.6, 0.55, 0.35, 0.15, 0.4]
    target = [0.5, 0.5, 0.3, 0.2, 0.5]
    _, outputs, targets, indices = _pair(pred, target)

    losses = _criterion().loss_boxes(outputs, targets, indices, 1.0)

    pred_tensor = torch.tensor([pred])
    target_tensor = torch.tensor([target])
    spatial = (pred_tensor[..., :4] - target_tensor[..., :4]).abs().sum()
    angle = periodic_angle_distance(
        pred_tensor[..., 4:], target_tensor[..., 4:]
    ).sum() / torch.pi
    assert set(losses) == {"loss_bbox", "loss_kld"}
    assert torch.allclose(losses["loss_bbox"], spatial + angle)


def test_new_mode_keys_and_canonical_bbox_contract():
    target = [0.5, 0.5, 0.3, 0.2, 0.0]
    _, first, targets, indices = _pair(
        [0.6, 0.55, 0.35, 0.15, 0.0], target
    )
    _, swapped, _, _ = _pair([0.6, 0.55, 0.15, 0.35, math.pi / 3], target)
    criterion = _new_criterion()

    first_losses = criterion.loss_boxes(first, targets, indices, 1.0)
    swapped_losses = criterion.loss_boxes(swapped, targets, indices, 1.0)

    assert set(first_losses) == {
        "loss_bbox", "loss_probiou", "loss_angle", "loss_kld"
    }
    assert torch.allclose(first_losses["loss_bbox"], swapped_losses["loss_bbox"])


def test_new_mode_without_kld_keeps_canonical_bbox():
    weights = {"loss_bbox": 2.0, "loss_probiou": 5.0, "loss_angle": 3.0}
    _, outputs, targets, indices = _pair(
        [0.7, 0.6, 0.35, 0.15, 0.2], [0.5, 0.5, 0.3, 0.2, 0.0]
    )

    losses = _new_criterion(keep_kld=False, weights=weights).loss_boxes(
        outputs, targets, indices, 1.0
    )

    assert "loss_bbox" in losses and "loss_kld" not in losses
    assert torch.isfinite(losses["loss_bbox"])


def test_new_mode_empty_matches_return_finite_scalar_zeros():
    outputs = {"pred_boxes": torch.zeros(1, 0, 5)}
    targets = [{"boxes": torch.zeros(0, 5), "labels": torch.zeros(0, dtype=torch.long)}]
    empty = torch.zeros(0, dtype=torch.long)

    losses = _new_criterion().loss_boxes(outputs, targets, [(empty, empty)], 1.0)

    assert set(losses) == set(NEW_WEIGHTS)
    assert all(value.ndim == 0 and value.item() == 0.0 for value in losses.values())


@pytest.mark.parametrize(
    ("weights", "kwargs", "missing"),
    [
        ({"loss_probiou": 5.0}, {"use_yolo_probiou": True}, "loss_bbox"),
        ({"loss_bbox": 2.0}, {"use_yolo_probiou": True}, "loss_probiou"),
        ({"loss_bbox": 2.0}, {"use_yolo_angle": True}, "loss_angle"),
        (
            {"loss_bbox": 2.0, "loss_probiou": 5.0},
            {"use_yolo_probiou": True, "keep_kld": True},
            "loss_kld",
        ),
    ],
)
def test_new_mode_rejects_missing_weight_without_mutation(weights, kwargs, missing):
    original = weights.copy()
    with pytest.raises(ValueError, match=missing):
        _criterion(weights=weights, **kwargs)
    assert weights == original


def test_hbb_box_values_and_keys_are_unchanged():
    _, outputs, targets, indices = _pair(
        [0.6, 0.5, 0.3, 0.2], [0.5, 0.5, 0.3, 0.2]
    )
    criterion = _criterion(
        weights={"loss_bbox": 2.0, "loss_giou": 2.0}, box_mode="hbb"
    )

    losses = criterion.loss_boxes(outputs, targets, indices, 1.0)

    assert set(losses) == {"loss_bbox", "loss_giou"}
    assert torch.allclose(losses["loss_bbox"], torch.tensor(0.1))


@pytest.mark.parametrize("center", [0.8, 20.0])
def test_new_geometry_has_directed_far_field_center_gradient(center):
    pred, outputs, targets, indices = _pair(
        [center, center, 0.2, 0.1, 0.0],
        [0.2, 0.2, 0.2, 0.1, 0.0],
        requires_grad=True,
    )
    losses = _new_criterion().loss_boxes(outputs, targets, indices, 1.0)

    sum(losses.values()).backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0, 0] > 0 and pred.grad[0, 1] > 0


def test_canonical_bbox_alone_has_far_field_center_gradient():
    weights = {"loss_bbox": 2.0, "loss_probiou": 5.0}
    criterion = _criterion(
        weights=weights, use_yolo_probiou=True, keep_kld=False
    )
    pred, outputs, targets, indices = _pair(
        [20.0, 20.0, 0.2, 0.1, 0.0],
        [0.2, 0.2, 0.2, 0.1, 0.0],
        requires_grad=True,
    )

    criterion.loss_boxes(outputs, targets, indices, 1.0)["loss_bbox"].backward()

    assert pred.grad is not None
    assert pred.grad[0, 0] > 0 and pred.grad[0, 1] > 0


def test_forward_preserves_raw_nonfinite_loss_components():
    # Given: a criterion whose get_loss returns controlled tensors
    # including NaN, +Inf, -Inf, and a finite value.
    weights = {"loss_bbox": 2.0}
    criterion = _criterion(weights=weights)
    criterion.losses = ["boxes"]
    criterion.matcher = lambda *a, **kw: {"indices": []}

    def _controlled_get_loss(
        self,
        loss,  # noqa: ARG001
        outputs,  # noqa: ARG001
        targets,  # noqa: ARG001
        indices,  # noqa: ARG001
        num_boxes,  # noqa: ARG001
        **kwargs,  # noqa: ARG001
    ):
        return {
            "loss_nan": torch.tensor(float("nan")),
            "loss_posinf": torch.tensor(float("inf")),
            "loss_neginf": torch.tensor(float("-inf")),
            "loss_ok": torch.tensor(1.5),
        }

    criterion.get_loss = types.MethodType(_controlled_get_loss, criterion)
    criterion.get_loss_meta_info = lambda *a, **kw: {}
    criterion.weight_dict.update(
        {"loss_nan": 1.0, "loss_posinf": 1.0, "loss_neginf": 1.0, "loss_ok": 1.0}
    )

    outputs = {
        "pred_boxes": torch.zeros(1, 0, 5),
        "pred_logits": torch.zeros(1, 0, 1),
        "aux_outputs": [{"pred_boxes": torch.zeros(1, 0, 5)}],
        "enc_aux_outputs": [{"pred_boxes": torch.zeros(1, 0, 5)}],
        "enc_meta": {"class_agnostic": False},
    }
    targets = [
        {"boxes": torch.zeros(0, 5), "labels": torch.zeros(0, dtype=torch.long)}
    ]

    # When: forward() collects and weights the loss components.
    result = criterion(outputs, targets)

    # Then: finite values survive, non-finite values are NOT silently
    # rewritten to zero or finite extrema.
    assert torch.isfinite(result["loss_ok"])
    assert result["loss_ok"].item() == pytest.approx(1.5)
    assert torch.isnan(result["loss_nan"]), (
        f"NaN was silently converted to {result['loss_nan'].item()}"
    )
    assert result["loss_posinf"].item() == float("inf")
    assert result["loss_neginf"].item() == float("-inf")
    weights = {"loss_bbox": 2.0, "loss_probiou": 5.0}
    criterion = _criterion(
        weights=weights, use_yolo_probiou=True, keep_kld=False
    )
    pred, outputs, targets, indices = _pair(
        [20.0, 20.0, 0.2, 0.1, 0.0],
        [0.2, 0.2, 0.2, 0.1, 0.0],
        requires_grad=True,
    )

    criterion.loss_boxes(outputs, targets, indices, 1.0)["loss_bbox"].backward()

    assert pred.grad is not None
    assert pred.grad[0, 0] > 0 and pred.grad[0, 1] > 0


# ---------------------------------------------------------------------------
# Task 5: 非周期 L1 消融路径 — 等比归一化无越界 + 消融差异记录
# ---------------------------------------------------------------------------


def _loss_angle_term(pred_theta, target_theta, *, periodic_angle_flag):
    """返回 loss_bbox（空间项置零，即纯角度贡献）。"""
    pred = [0.5, 0.5, 0.3, 0.2, pred_theta]
    target = [0.5, 0.5, 0.3, 0.2, target_theta]
    _, outputs, targets, indices = _pair(pred, target)
    criterion = _criterion(periodic_angle_flag=periodic_angle_flag)
    return criterion.loss_boxes(outputs, targets, indices, 1.0)["loss_bbox"]


def test_nonperiodic_no_overflow_beyond_1():
    # pred θ=3π/4: 旧 shifted (θ+π/4)/π = 1.0 (越界), 等比 θ/π = 0.75
    loss = _loss_angle_term(3 * math.pi / 4, 0.0, periodic_angle_flag=False)
    assert abs(loss.item() - 0.75) < 1e-5, f"期望 0.75, got {loss.item():.6f}"


def test_nonperiodic_vs_periodic_at_seam():
    # pred≈0, gt≈π: 非周期 naive L1 ≈ 1 (seam 错配), 周期距离 ≈ 0
    eps = 1e-3
    loss_np = _loss_angle_term(eps, math.pi - eps, periodic_angle_flag=False)
    loss_p = _loss_angle_term(eps, math.pi - eps, periodic_angle_flag=True)
    assert loss_np.item() > 0.9, f"非周期 seam 应 ~1, got {loss_np.item():.4f}"
    assert loss_p.item() < 0.01, f"周期 seam 应 ~0, got {loss_p.item():.4f}"


def test_nonperiodic_equals_periodic_away_from_seam():
    # pred=0.5π, gt=0.3π: 无 wraparound, 两路径一致
    loss_np = _loss_angle_term(0.5 * math.pi, 0.3 * math.pi, periodic_angle_flag=False)
    loss_p = _loss_angle_term(0.5 * math.pi, 0.3 * math.pi, periodic_angle_flag=True)
    assert abs(loss_np.item() - loss_p.item()) < 1e-5, (
        f"远离 seam 应一致: np={loss_np.item():.6f}, p={loss_p.item():.6f}"
    )
