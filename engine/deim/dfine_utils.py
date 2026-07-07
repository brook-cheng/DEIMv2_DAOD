"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
from .box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from .obb_geometry import oriented_box_to_external_rect, external_rect_to_oriented_box


def weighting_function(reg_max, up, reg_scale, deploy=False):
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        reg_max (int): Max number of the discrete bins.
        up (Tensor): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(reg_max/2)=0
                           and steeper weights at both ends.
        deploy (bool): If True, uses deployment mode settings.

    Returns:
        Tensor: Sequence of Weighting Function.
    """
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor(values, dtype=up.dtype, device=up.device)
    else:
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.cat(values, 0)


def translate_gt(gt, reg_max, reg_scale, up):
    """
    Decodes bounding box ground truth (GT) values into distribution-based GT representations.

    This function maps continuous GT values into discrete distribution bins, which can be used
    for regression tasks in object detection models. It calculates the indices of the closest
    bins to each GT value and assigns interpolation weights to these bins based on their proximity
    to the GT value.

    Args:
        gt (Tensor): Ground truth bounding box values, shape (N, ).
        reg_max (int): Maximum number of discrete bins for the distribution.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (Tensor): Controls the upper bounds of the Weighting Function.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - indices (Tensor): Index of the left bin closest to each GT value, shape (N, ).
            - weight_right (Tensor): Weight assigned to the right bin, shape (N, ).
            - weight_left (Tensor): Weight assigned to the left bin, shape (N, ).
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale)

    # Find the closest left-side indices for each value
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1

    # Calculate the weights for the interpolation
    indices = closest_left_indices.float()

    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()

    # Obtain distances
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]

    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])

    # Valid weights
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    # Invalid weights (out of range)
    invalid_idx_mask_neg = indices < 0
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    invalid_idx_mask_pos = indices >= reg_max
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    return indices, weight_right, weight_left


def distance2bbox(points, distance, reg_scale):
    """
    Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                         where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                           point to the left, top, right, and bottom boundaries.

        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (
        points[..., 2] / reg_scale
    )
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (
        points[..., 3] / reg_scale
    )
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (
        points[..., 2] / reg_scale
    )
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (
        points[..., 3] / reg_scale
    )

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return box_xyxy_to_cxcywh(bboxes)


def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """
    Converts bounding box coordinates to distances from a reference point.

    Args:
        points (Tensor): (n, 4) [x, y, w, h], where (x, y) is the center.
        bbox (Tensor): (n, 4) bounding boxes in "xyxy" format.
        reg_max (float): Maximum bin value.
        reg_scale (float): Controling curvarture of W(n).
        up (Tensor): Controling upper bounds of W(n).
        eps (float): Small value to ensure target < reg_max.

    Returns:
        Tensor: Decoded distances.
    """
    reg_scale = abs(reg_scale)
    left = (points[:, 0] - bbox[:, 0]) / (
        points[..., 2] / reg_scale + 1e-16
    ) - 0.5 * reg_scale
    top = (points[:, 1] - bbox[:, 1]) / (
        points[..., 3] / reg_scale + 1e-16
    ) - 0.5 * reg_scale
    right = (bbox[:, 2] - points[:, 0]) / (
        points[..., 2] / reg_scale + 1e-16
    ) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (
        points[..., 3] / reg_scale + 1e-16
    ) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(
        four_lens, reg_max, reg_scale, up
    )
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()


def distance2bbox_obb(points, distance, reg_scale, offset_scale_source: str = "pre"):
    """
    Decodes 6-distribution DDF output to 5-dof OBB.

    Args:
        points: (B,N,5) or (N,5) — ref obb (cx,cy,w,h,θ),θ belongs to [0,π].
        distance: (B,N,6) — (α,β,γ,δ,ε,η) from Integral.
        reg_scale: curvature of Weighting Function.
        offset_scale_source: which external-rectangle size scales the (ε,η)
            offset residuals. ``"pre"`` uses the pre-adjustment external rect
            (current default, spec 7.1 — keeps residuals in the reference
            coordinate frame, avoids same-layer coupling). ``"post"`` uses the
            post-adjustment external rect (ablation mode, spec 7.1).

    Returns:
        (B,N,5) or (N,5) — (cx,cy,w,h,θ),θ belongs to [0,π].
    """
    if offset_scale_source not in ("pre", "post"):
        raise ValueError(
            f"offset_scale_source must be 'pre' or 'post', "
            f"got {offset_scale_source!r}"
        )

    ext_rect_xyxy, vertex_offsets = oriented_box_to_external_rect(points)
    ext_rect_cxcywh = box_xyxy_to_cxcywh(ext_rect_xyxy)

    # (α,β,γ,δ)
    ext_adj_cxcywh = distance2bbox(ext_rect_cxcywh, distance[..., :4], reg_scale)
    ext_adjust_xyxy = box_cxcywh_to_xyxy(ext_adj_cxcywh)

    # (ε,η): scale offset residuals by external-rect (w,h).
    # 'pre'  -> pre-adjustment ext rect (default, spec 7.1).
    # 'post' -> post-adjustment ext rect (ablation, spec 7.1).
    offset_scale_wh = (
        ext_rect_cxcywh[..., 2:]
        if offset_scale_source == "pre"
        else ext_adj_cxcywh[..., 2:]
    )
    vertex_offsets_adj = (
        vertex_offsets + distance[..., 4:] * offset_scale_wh / reg_scale
    )

    return external_rect_to_oriented_box(ext_adjust_xyxy, vertex_offsets_adj)


def bbox2distance_obb(
    points, bbox, reg_max, reg_scale, up, eps=0.1, offset_scale_source: str = "pre"
):
    """
    Converts GT OBB to 6-distribution FGL targets.

    Args:
        points: (N,5) ref points (cx,cy,w,h,θ),θ belongs to [0,π].
        bbox: (N,5) GT OBB (cx,cy,w,h,θ).
        reg_max (float): Maximum bin value.
        reg_scale (float): Controlling curvature of W(n).
        up (Tensor): Controlling upper bounds of W(n).
        eps (float): Small value to ensure target < reg_max.
        offset_scale_source: which external-rectangle size scales the (ε,η)
            offset residuals. Must match ``distance2bbox_obb`` to keep
            encode/decode inverses. ``"pre"`` (default) uses the
            pre-adjustment pred ext rect; ``"post"`` is an ablation mode
            that uses the GT ext rect (the post-adjustment target).

    Returns:
        six_lens, weight_right, weight_left
    """
    if offset_scale_source not in ("pre", "post"):
        raise ValueError(
            f"offset_scale_source must be 'pre' or 'post', "
            f"got {offset_scale_source!r}"
        )

    ext_rect_xyxy_pred, vertex_offsets_pred = oriented_box_to_external_rect(points)
    ext_rect_xyxy_gt, vertex_offsets_gt = oriented_box_to_external_rect(bbox)
    ext_rect_cxcywh_pred = box_xyxy_to_cxcywh(ext_rect_xyxy_pred)
    ext_rect_cxcywh_gt = box_xyxy_to_cxcywh(ext_rect_xyxy_gt)
    reg_scale = abs(reg_scale)

    pred_ext_rect_w_scaled = ext_rect_cxcywh_pred[..., 2] / reg_scale + 1e-16
    pred_ext_rect_h_sceled = ext_rect_cxcywh_pred[..., 3] / reg_scale + 1e-16
    half_reg_scale = 0.5 * reg_scale

    # 下面的计算与distance2bbox_obb中的互逆
    ext_left = (
        (ext_rect_cxcywh_pred[..., 0] - ext_rect_xyxy_gt[..., 0])
        / pred_ext_rect_w_scaled
        - half_reg_scale
        + 1e-16
    )
    ext_top = (
        (ext_rect_cxcywh_pred[..., 1] - ext_rect_xyxy_gt[..., 1])
        / pred_ext_rect_h_sceled
        - half_reg_scale
        + 1e-16
    )
    ext_right = (
        (ext_rect_xyxy_gt[..., 2] - ext_rect_cxcywh_pred[..., 0])
        / pred_ext_rect_w_scaled
        - half_reg_scale
        + 1e-16
    )
    ext_bottom = (
        (ext_rect_xyxy_gt[..., 3] - ext_rect_cxcywh_pred[..., 1])
        / pred_ext_rect_h_sceled
        - half_reg_scale
        + 1e-16
    )

    # (ε,η): inverse of distance2bbox_obb's offset scaling.
    # 'pre'  -> pre-adjustment pred ext rect (default, matches decode 'pre').
    # 'post' -> GT ext rect = post-adjustment target (matches decode 'post').
    offset_scale_wh = (
        ext_rect_cxcywh_pred[..., 2:]
        if offset_scale_source == "pre"
        else ext_rect_cxcywh_gt[..., 2:]
    )
    two_lens = (vertex_offsets_gt - vertex_offsets_pred) / (
        offset_scale_wh / reg_scale + 1e-16
    )

    six_lens = torch.stack(
        [
            ext_left,
            ext_top,
            ext_right,
            ext_bottom,
            two_lens[..., 0],
            two_lens[..., 1],
        ],
        dim=-1,
    )

    six_lens, weight_right, weight_left = translate_gt(six_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        six_lens = six_lens.clamp(min=0, max=reg_max - eps)

    return six_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()
