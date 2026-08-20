"""Pure OBB loss helpers for DEIMv2-OBB loss refinement.

Provides canonical center/side L1, ProbIoU loss, periodic angle loss,
and pairwise angle cost matrix. All helpers are stateless and preserve
prediction gradients.

OBB format: [center_x, center_y, width, height, rotation_angle] (xywhr).
The final dimension must be 5; any other shape raises ValueError.

References:
    docs/superpowers/specs/2026-07-16-deimv2-obb-loss-refinement-design.md
"""

import torch

from engine.deim.obb_ops import probiou


def _assert_obb5(t: torch.Tensor, name: str) -> None:
    """Raise ValueError if the final dimension of *t* is not 5."""
    if t.shape[-1] != 5:
        raise ValueError(
            f"{name} must have final dimension 5 (xywhr), "
            f"got shape {tuple(t.shape)}"
        )


def _empty_scalar_like(t: torch.Tensor) -> torch.Tensor:
    """Return a scalar zero with the same dtype and device as *t*."""
    return torch.zeros((), device=t.device, dtype=t.dtype)


def canonical_side_l1_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    normalizer: float | torch.Tensor,
) -> torch.Tensor:
    """Return center plus canonical short/long-side L1 over matched pairs.

    The loss is::

        |pred_cx - tgt_cx| + |pred_cy - tgt_cy|
      + |pred_short - tgt_short| + |pred_long - tgt_long|

    where short/long are obtained by sorting ``[w, h]``. This makes the
    loss invariant to ``w``/``h`` exchange. The sum over all matched
    pairs is divided by *normalizer*.
    """
    _assert_obb5(pred_bboxes, "pred_bboxes")
    _assert_obb5(target_bboxes, "target_bboxes")

    if pred_bboxes.shape[0] == 0:
        return _empty_scalar_like(pred_bboxes)

    pred_center = pred_bboxes[..., :2]
    target_center = target_bboxes[..., :2]
    pred_sides = torch.sort(pred_bboxes[..., 2:4], dim=-1).values
    target_sides = torch.sort(target_bboxes[..., 2:4], dim=-1).values

    center_l1 = (pred_center - target_center).abs().sum(dim=-1)
    sides_l1 = (pred_sides - target_sides).abs().sum(dim=-1)
    total = (center_l1 + sides_l1).sum()
    return total / normalizer


def yolo_probiou_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    normalizer: float | torch.Tensor,
) -> torch.Tensor:
    """Return scalar ``(1 - ProbIoU)`` loss over matched OBB pairs.

    Uses the existing :func:`engine.deim.obb_ops.probiou` without masks,
    quality weights, or detach. The sum over all matched pairs is
    divided by *normalizer*.
    """
    _assert_obb5(pred_bboxes, "pred_bboxes")
    _assert_obb5(target_bboxes, "target_bboxes")

    if pred_bboxes.shape[0] == 0:
        return _empty_scalar_like(pred_bboxes)

    iou = probiou(pred_bboxes, target_bboxes)
    return (1.0 - iou).sum() / normalizer


def yolo_angle_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    normalizer: float | torch.Tensor,
    lambda_val: float = 3.0,
) -> torch.Tensor:
    """Return scalar periodic angle loss over matched OBB pairs.

    The angle delta is wrapped to ``[-pi/2, pi/2]`` via
    ``delta - round(delta / pi) * pi``. The penalty is
    ``sin(2 * delta)^2``, which is zero at ``delta=0`` and
    ``delta=±pi/2`` (matching ``w``/``h`` interchange symmetry).

    The target aspect-ratio weight
    ``exp(-log((w+eps)/(h+eps))^2 / lambda^2)`` is strongest near a
    square and decreases for extreme ratios. The weighted sum over all
    matched pairs is divided by *normalizer*.
    """
    _assert_obb5(pred_bboxes, "pred_bboxes")
    _assert_obb5(target_bboxes, "target_bboxes")

    if pred_bboxes.shape[0] == 0:
        return _empty_scalar_like(pred_bboxes)

    delta = pred_bboxes[..., 4] - target_bboxes[..., 4]
    wrapped = delta - torch.round(delta / torch.pi) * torch.pi
    penalty = torch.sin(2.0 * wrapped).square()

    target_w = target_bboxes[..., 2]
    target_h = target_bboxes[..., 3]
    eps = torch.finfo(pred_bboxes.dtype).eps
    log_ar = torch.log((target_w + eps) / (target_h + eps))
    scale_weight = torch.exp(-log_ar.square() / (lambda_val**2))

    total = (penalty * scale_weight).sum()
    return total / normalizer


def compute_angle_cost_matrix(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    lambda_val: float = 3.0,
) -> torch.Tensor:
    """Return pairwise angle cost with shape ``(num_queries, num_targets)``.

    Uses the same periodic angle penalty and target aspect-ratio weight
    as :func:`yolo_angle_loss`, but computed pairwise for the Hungarian
    matcher. Empty targets return ``(num_queries, 0)``; empty preds
    return ``(0, num_targets)``. No NaN is produced for empty inputs.
    """
    _assert_obb5(pred_bboxes, "pred_bboxes")
    _assert_obb5(target_bboxes, "target_bboxes")

    n_pred = pred_bboxes.shape[0]
    n_tgt = target_bboxes.shape[0]
    if n_tgt == 0:
        return torch.zeros(
            (n_pred, 0), device=pred_bboxes.device, dtype=pred_bboxes.dtype
        )
    if n_pred == 0:
        return torch.zeros(
            (0, n_tgt), device=pred_bboxes.device, dtype=pred_bboxes.dtype
        )

    delta = pred_bboxes[:, None, 4] - target_bboxes[None, :, 4]
    wrapped = delta - torch.round(delta / torch.pi) * torch.pi
    penalty = torch.sin(2.0 * wrapped).square()

    target_w = target_bboxes[None, :, 2]
    target_h = target_bboxes[None, :, 3]
    eps = torch.finfo(pred_bboxes.dtype).eps
    log_ar = torch.log((target_w + eps) / (target_h + eps))
    scale_weight = torch.exp(-log_ar.square() / (lambda_val**2))

    return penalty * scale_weight
