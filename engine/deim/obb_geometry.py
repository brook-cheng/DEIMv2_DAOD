"""
Oriented bounding box geometry utilities.

Provides bi-directional mapping between oriented bounding boxes (OBB)
and their external rectangles with vertex offsets, as used in
Angle Distribution Refinement (ADR).

Reference:
    Ding et al., "Real-Time Oriented Object Detection Transformer
    in Remote Sensing Images" (TGRS 2026).
"""

from typing import Tuple
import torch
from torch import Tensor

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .obb_stable_atan2 import stable_atan2 as _stable_atan2


def periodic_angle_distance(pred: Tensor, target: Tensor, with_signal=False) -> Tensor:
    """Shortest distance on the pi-periodic angle domain (radians).

    OBB orientations differing by a multiple of ``pi`` are equivalent,
    so ``0`` and ``pi`` are the same orientation. Returns the shortest
    periodic distance in ``[0, pi/2]``.

    Args:
        pred:   (...,) predicted angles (radians).
        target: (...,) target angles (radians).

    Returns:
        (...,) shortest periodic distance in radians; ``pred`` and
        ``target`` broadcast per PyTorch semantics.
    """
    if not with_signal:
        diff = (pred - target).abs()
        d = torch.remainder(diff, torch.pi)
        return torch.minimum(d, torch.pi - d)
    else:
        diff = target - pred
        return torch.remainder(diff + torch.pi / 2, torch.pi) - torch.pi / 2


def xywhr_to_xyxyxyxy(xywhr: Tensor) -> Tensor:
    """Convert OBB (cx, cy, w, h, theta) to four corner vertices.

    Args:
        xywhr: (..., 5)  —  (cx, cy, w, h, theta)  in normalized coords,θ belongs to [0,π).

    Returns:
        (..., 4, 2)  —  4 corner points in clockwise order.
    """

    w, h, angle = (xywhr[..., i : i + 1] for i in range(2, 5))
    cosa = torch.cos(angle)
    sina = torch.sin(angle)

    ctr = xywhr[..., :2]

    vec1 = [w / 2 * cosa, w / 2 * sina]
    vec2 = [-h / 2 * sina, h / 2 * cosa]
    vec1 = torch.cat(vec1, -1)
    vec2 = torch.cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return torch.stack([pt1, pt2, pt3, pt4], -2)


def xyxyxyxy_to_xywhr(xyxyxyxy: Tensor) -> Tensor:
    """Convert four corner vertices to OBB (cx, cy, w, h, theta).

    Args:
        xyxyxyxy:  (..., 4, 2)  —  4 corner points in clockwise order.

    Returns:
        (..., 5)  —  (cx, cy, w, h, theta)  in normalized coords,θ belongs to [0,π).
    """

    ctr = xyxyxyxy.mean(dim=-2)
    vec0 = xyxyxyxy[..., 0:1, :]
    dists = ((xyxyxyxy - vec0) ** 2).sum(dim=-1)
    sorted_idxs = torch.argsort(dists, dim=-1)
    idx1 = sorted_idxs[..., 1]
    idx2 = sorted_idxs[..., 2]

    _g1 = idx1.unsqueeze(-1).unsqueeze(-1).expand(*idx1.shape, 1, 2)
    _g2 = idx2.unsqueeze(-1).unsqueeze(-1).expand(*idx2.shape, 1, 2)
    edg1 = xyxyxyxy.gather(-2, _g1).squeeze(-2) - vec0.squeeze(-2)
    edg2 = xyxyxyxy.gather(-2, _g2).squeeze(-2) - vec0.squeeze(-2)

    len1 = (edg1**2).sum(dim=-1).sqrt()
    len2 = (edg2**2).sum(dim=-1).sqrt()

    w_mask = len1 >= len2
    w = torch.where(w_mask, len1, len2)
    h = torch.where(w_mask, len2, len1)
    w_dx = torch.where(w_mask.squeeze(-1), edg1[..., 0], edg2[..., 0])
    w_dy = torch.where(w_mask.squeeze(-1), edg1[..., 1], edg2[..., 1])
    theta = torch.atan2(w_dy, w_dx)
    theta = torch.remainder(theta, torch.pi)
    return torch.stack([ctr[..., 0], ctr[..., 1], w, h, theta], dim=-1)


def oriented_box_to_external_xyxy_rect(obbs: Tensor) -> Tuple[Tensor, Tensor]:
    """OBB -> external rectangle + vertex offsets (epsilon, eta).

    Args:
        obbs:  (..., 5)  —  (cx, cy, w, h, theta),θ belongs to [0,pi).

    Returns:
        external_rect:   (..., 4)  —  (x1, y1, x2, y2).
        vertex_offsets:   (..., 2)  —  (epsilon, eta).
    """
    vertices = xywhr_to_xyxyxyxy(obbs)  # (..., 4, 2)

    x_min = vertices[..., 0].amin(dim=-1)
    y_min = vertices[..., 1].amin(dim=-1)
    x_max = vertices[..., 0].amax(dim=-1)
    y_max = vertices[..., 1].amax(dim=-1)

    external_rect = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    # epsilon: x-distance from top-right corner to the top-edge OBB vertex
    top_idx = vertices[..., 1].argmin(dim=-1)  # (...)
    _g = top_idx.unsqueeze(-1).unsqueeze(-1).expand(*top_idx.shape, 1, 2)
    top_vertex = vertices.gather(-2, _g).squeeze(-2)  # (..., 2)
    epsilon = torch.clamp(x_max - top_vertex[..., 0], min=0)

    # eta: y-distance from bottom-right corner to the right-edge OBB vertex
    right_idx = vertices[..., 0].argmax(dim=-1)  # (...)
    _g = right_idx.unsqueeze(-1).unsqueeze(-1).expand(*right_idx.shape, 1, 2)
    right_vertex = vertices.gather(-2, _g).squeeze(-2)  # (..., 2)
    eta = torch.clamp(y_max - right_vertex[..., 1], min=0)

    vertex_offsets = torch.stack([epsilon, eta], dim=-1)

    return external_rect, vertex_offsets


def clamp_vertex_offsets_to_external_rect(
    external_rect: Tensor, vertex_offsets: Tensor
) -> Tensor:
    """Clamp ``(epsilon, eta)`` vertex offsets into the valid range of the
    external rectangle (spec section 9.1):

        0 <= epsilon <= external_width
        0 <= eta     <= external_height

    Non-mutating: returns a new tensor; inputs are not modified in place.
    Degenerate external rectangles (zero or negative width/height) clamp
    the corresponding offset to zero. Intended for decode-time /
    eval-safe paths only; do not apply to the loss-bearing
    ``inter_ref_bbox`` tensor before loss computation (plan Todo 6).

    Args:
        external_rect:  (..., 4)  —  (x1, y1, x2, y2).
        vertex_offsets: (..., 2)  —  (epsilon, eta).

    Returns:
        (..., 2)  —  clamped (epsilon, eta).
    """
    ext_w = (external_rect[..., 2] - external_rect[..., 0]).clamp(min=0)
    ext_h = (external_rect[..., 3] - external_rect[..., 1]).clamp(min=0)
    ep = vertex_offsets[..., 0].clamp(min=0).clamp(max=ext_w)
    et = vertex_offsets[..., 1].clamp(min=0).clamp(max=ext_h)
    return torch.stack([ep, et], dim=-1)


def external_xyxy_rect_to_oriented_box(
    external_rect: Tensor,
    vertex_offsets: Tensor,
    eps=1e-9,
    clamp_offsets: bool = False,
) -> Tensor:
    """External rectangle + vertex offsets -> OBB.

    The four OBB vertices lie on the four edges:
        v_top    = (x2 - epsilon, y1)          — top edge
        v_right  = (x2, y2 - eta)              — right edge
        v_bottom = (x1 + epsilon, y2)          — bottom edge
        v_left   = (x1, y1 + eta)              — left edge

    For non-degenerate orientations these are consecutive vertices
    of the rectangle.  We use the longer-edge-as-w convention.

    Args:
        external_rect:   (..., 4)  —  (x1, y1, x2, y2).
        vertex_offsets:   (..., 2)  —  (epsilon, eta).
        eps: numerical stability for edge-length sqrt.
        clamp_offsets: when True, clamp ``(epsilon, eta)`` into
            ``[0, external_width]`` / ``[0, external_height]`` via
            ``clamp_vertex_offsets_to_external_rect`` before decoding.
            Default ``False`` preserves the unguarded training decode
            path (plan Todo 6: do not destroy gradients on the
            loss-bearing tensor). Use ``True`` only on detached /
            eval-safe decode paths.

    Returns:
        (..., 5)  —  (cx, cy, w, h, theta),θ belongs to [0,pi).
    """

    if clamp_offsets:
        vertex_offsets = clamp_vertex_offsets_to_external_rect(
            external_rect, vertex_offsets
        )

    x1 = external_rect[..., 0]
    y1 = external_rect[..., 1]
    x2 = external_rect[..., 2]
    y2 = external_rect[..., 3]
    ep = vertex_offsets[..., 0]
    et = vertex_offsets[..., 1]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    v0 = torch.stack([x2 - ep, y1], dim=-1)  # top edge
    v1 = torch.stack([x2, y2 - et], dim=-1)  # right edge
    v2 = torch.stack([x1 + ep, y2], dim=-1)  # bottom edge
    v3 = torch.stack([x1, y1 + et], dim=-1)  # left edge

    # Consecutive edges: ab = v1-v0, bc = v2-v1
    edge_ab = v1 - v0
    edge_bc = v2 - v1

    len_ab = torch.sqrt(edge_ab[..., 0] ** 2 + edge_ab[..., 1] ** 2 + eps)
    len_bc = torch.sqrt(edge_bc[..., 0] ** 2 + edge_bc[..., 1] ** 2 + eps)

    # w = longer edge, h = shorter edge
    w_is_ab = len_ab >= len_bc
    w_len = torch.where(w_is_ab, len_ab, len_bc)
    h_len = torch.where(w_is_ab, len_bc, len_ab)

    w_dx = torch.where(w_is_ab, edge_ab[..., 0], edge_bc[..., 0])
    w_dy = torch.where(w_is_ab, edge_ab[..., 1], edge_bc[..., 1])

    theta = _stable_atan2(w_dy, w_dx, eps)
    theta = torch.remainder(theta, torch.pi)
    return torch.stack([cx, cy, w_len, h_len, theta], dim=-1)


def external_xywh_rect_to_oriented_box(
    external_cxcywh: Tensor,
    vertex_offsets: Tensor,
    eps: float = 1e-9,
    clamp_offsets: bool = False,
) -> Tensor:
    external_xyxy = box_cxcywh_to_xyxy(external_cxcywh)
    return external_xyxy_rect_to_oriented_box(
        external_xyxy,
        vertex_offsets,
        eps=eps,
        clamp_offsets=clamp_offsets,
    )


def oriented_box_to_external_xywh_rect(obbs: Tensor) -> Tuple[Tensor, Tensor]:
    external_xyxy, vertex_offsets = oriented_box_to_external_xyxy_rect(obbs)
    return box_xyxy_to_cxcywh(external_xyxy), vertex_offsets


def affine_obb(
    boxes_xywhr: Tensor, sx: float, sy: float, tx: float = 0.0, ty: float = 0.0
) -> Tensor:
    """对 OBB 做像素级仿射变换：v_x' = v_x * sx + tx, v_y' = v_y * sy + ty。
    通过将 4 个顶点变换后重新拟合 (cx,cy,w,h,θ)，在 sx≠sy 时正确更新 w/h/θ。

    Args:
        boxes_xywhr: (N, 5)  —  像素坐标 (cx, cy, w, h, θ), θ∈[0,π)。
        sx, sy:       x、y 方向缩放因子。
        tx, ty:       x、y 方向平移量（像素）。

    Returns:
        (N, 5)  —  变换后的 OBB，像素坐标。N=0 时原样返回。
    """
    if boxes_xywhr.numel() == 0:
        return boxes_xywhr
    vertices = xywhr_to_xyxyxyxy(boxes_xywhr)  # (N, 4, 2)
    vertices = vertices.clone()  # 避免污染输入
    vertices[..., 0] = vertices[..., 0] * sx + tx
    vertices[..., 1] = vertices[..., 1] * sy + ty
    return xyxyxyxy_to_xywhr(vertices)


def affine_obb_matrix(boxes_xywhr: Tensor, mat: Tensor) -> Tensor:
    """对 OBB 施加一般前向仿射：v' = A @ v + b（A=mat[:,:2](2,2)，b=mat[:,2](2,)）。
    将 4 个顶点变换后用 xyxyxyxy_to_xywhr 重拟合 (cx,cy,w,h,θ)。

    Args:
        boxes_xywhr: (N,5) 像素坐标。
        mat:         (2,3) 前向仿射矩阵 [A | b]。
    Returns:
        (N,5)。N=0 时原样返回。保持 dtype/device。
    """
    if boxes_xywhr.numel() == 0:
        return boxes_xywhr
    A = mat[:, :2]  # (2,2)
    b = mat[:, 2]  # (2,)
    v = xywhr_to_xyxyxyxy(boxes_xywhr).clone()  # (N,4,2)
    v = v.to(dtype=A.dtype, device=A.device)
    v_new = v @ A.T + b  # (N,4,2) @ (2,2) + (2,) -> (N,4,2)
    return xyxyxyxy_to_xywhr(v_new)
