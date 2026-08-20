"""OBB matched-pair error classification.

Distinguishes *genuine* angle prediction errors from *swap artifacts* — pairs
where the predicted box is geometrically near-identical to GT but its ``(w, h,
θ)`` parameterization differs by a ~90° rotation with the width/height axes
swapped. Such pairs show a large raw ``Δθ`` yet retain high ProbIoU, so a naive
"large angle = wrong" rule would falsely flag them.

Public API
----------
    classify_errors(gt, pred, iou_threshold=0.5, angle_threshold_deg=15.0) -> dict
"""

from __future__ import annotations

import math

import torch

from .obb_ops import batch_probiou

__all__ = ["classify_errors"]


def _shortest_angle_diff_rad(pred_theta: torch.Tensor, gt_theta: torch.Tensor) -> torch.Tensor:
    """Shortest signed angular difference in radians, range [-π/2, π/2].

    OBB angles are π-periodic: θ and θ+π describe the same orientation, so the
    shortest difference lives in [-π/2, π/2].
    """
    diff = pred_theta - gt_theta
    return (diff + torch.pi / 2) % torch.pi - torch.pi / 2


def classify_errors(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
    angle_threshold_deg: float = 15.0,
) -> dict[str, int]:
    """Classify index-aligned matched OBB pairs into three categories.

    For each pair ``(gt_boxes[i], pred_boxes[i])``:

    * ``|Δθ| ≤ angle_threshold_deg``                      → ``ok``
    * ``|Δθ| > angle_threshold_deg`` and
      ``ProbIoU ≥ iou_threshold``                         → ``swap_artifact``
      (apparent large angle error from w/h axis swap; geometrically overlapping)
    * ``|Δθ| > angle_threshold_deg`` and
      ``ProbIoU < iou_threshold``                         → ``genuine_angle_error``

    Args:
        gt_boxes:   ``(N, 5)`` tensor of GT OBBs ``(cx, cy, w, h, θ_rad)``;
                    ``θ`` in radians, π-periodic.
        pred_boxes: ``(N, 5)`` tensor of predicted OBBs, index-aligned with
                    ``gt_boxes`` (caller must match first, e.g. Hungarian).
        iou_threshold: ProbIoU at/above which a large-angle pair is deemed a
                    swap artifact rather than a genuine error.
        angle_threshold_deg: pairs whose shortest ``|Δθ|`` is at or below this
                    are classified ``ok``.

    Returns:
        ``{"ok": int, "genuine_angle_error": int, "swap_artifact": int}`` —
        per-category counts. The three counts always sum to ``N``.
    """
    if gt_boxes.shape[0] == 0:
        return {"ok": 0, "genuine_angle_error": 0, "swap_artifact": 0}

    if gt_boxes.shape != pred_boxes.shape:
        raise ValueError(
            f"shape mismatch: gt_boxes {tuple(gt_boxes.shape)} vs "
            f"pred_boxes {tuple(pred_boxes.shape)} (expected index-aligned pairs)"
        )

    ang_diff_rad = _shortest_angle_diff_rad(pred_boxes[:, 4], gt_boxes[:, 4])
    threshold_rad = math.radians(angle_threshold_deg)
    # Tolerance absorbs ULP-scale error from the (a+π/2)%π−π/2 wrap so an input
    # exactly at the threshold is not misclassified as large (~1e-6 rad ≈ 6e-5°).
    large = ang_diff_rad.abs() > threshold_rad + 1e-6

    # Per-pair ProbIoU: batch_probiou returns (N, N); take the diagonal.
    iou_matrix = batch_probiou(gt_boxes, pred_boxes)
    iou_diag = torch.diagonal(iou_matrix)  # (N,)

    swap = large & (iou_diag >= iou_threshold)
    genuine = large & (~swap)

    return {
        "ok": int((~large).sum().item()),
        "genuine_angle_error": int(genuine.sum().item()),
        "swap_artifact": int(swap.sum().item()),
    }
