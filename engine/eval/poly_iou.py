"""
Exact polygon IoU for oriented bounding boxes.

Computes pairwise IoU equivalent to DOTA_devkit's polyiou,
implemented in pure PyTorch.
"""

import torch
from torch import Tensor
from ..deim.obb_geometry import xywhr_to_xyxyxyxy


def _polygon_area(poly: Tensor) -> Tensor:
    """Shoelace formula area of polygon (..., K, 2)."""
    x = poly[..., 0]
    y = poly[..., 1]
    return 0.5 * torch.abs(
        (x * torch.roll(y, 1, dims=-1)).sum(dim=-1) -
        (y * torch.roll(x, 1, dims=-1)).sum(dim=-1)
    )


def _is_inside_convex(points: Tensor, quad: Tensor) -> Tensor:
    """Check if points are inside a convex quadrilateral.

    Args:
        points: (N, 2) points to check
        quad:   (4, 2) quadrilateral vertices (clockwise or counter-clockwise)

    Returns:
        (N,) boolean mask
    """
    # Cross-product test: a point is inside if it's on the same side of all edges
    insides = []
    K = quad.shape[0]
    for i in range(K):
        p1 = quad[i]
        p2 = quad[(i + 1) % K]
        cross = (p2[0] - p1[0]) * (points[:, 1] - p1[1]) - \
                (p2[1] - p1[1]) * (points[:, 0] - p1[0])
        insides.append(cross <= 0)
    inside = insides[0]
    for i in range(1, K):
        inside = inside & insides[i]
    return inside


def _edge_intersection(a1: Tensor, a2: Tensor, b1: Tensor, b2: Tensor) -> Tensor:
    """Intersection of two line segments a1-a2 and b1-b2.

    Returns (2,) point or (2,) nan if no intersection within segments.
    """
    d = (a1[0] - a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] - b2[0])
    if abs(d) < 1e-10:
        return torch.tensor([float('nan'), float('nan')])

    t = ((a1[0] - b1[0]) * (b1[1] - b2[1]) - (a1[1] - b1[1]) * (b1[0] - b2[0])) / d
    u = -((a1[0] - a2[0]) * (a1[1] - b1[1]) - (a1[1] - a2[1]) * (a1[0] - b1[0])) / d

    if 0 <= t <= 1 and 0 <= u <= 1:
        return torch.stack([a1[0] + t * (a2[0] - a1[0]),
                            a1[1] + t * (a2[1] - a1[1])])
    return torch.tensor([float('nan'), float('nan')])


def _convex_hull_2d(points: Tensor) -> Tensor:
    """Graham scan for 2D convex hull (batch size 1)."""
    if points.shape[0] <= 2:
        return points

    # sort by x, then y
    idx = points[:, 0].argsort()
    pts = points[idx]

    # lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2:
            a, b = lower[-2], lower[-1]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross > 0:
                break
            lower.pop()
        lower.append(p)

    # upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2:
            a, b = upper[-2], upper[-1]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross > 0:
                break
            upper.pop()
        upper.append(p)

    hull = torch.stack(lower[:-1] + upper[:-1])
    return hull


def _convex_intersection(quad1: Tensor, quad2: Tensor) -> Tensor:
    """Intersection of two convex quadrilaterals.

    Args:
        quad1: (4, 2) vertices
        quad2: (4, 2) vertices

    Returns:
        (K, 2) intersection polygon, or (0, 2) if empty
    """
    pts = []

    # edge intersections
    for i in range(4):
        a1, a2 = quad1[i], quad1[(i + 1) % 4]
        for j in range(4):
            b1, b2 = quad2[j], quad2[(j + 1) % 4]
            p = _edge_intersection(a1, a2, b1, b2)
            if not torch.isnan(p[0]):
                pts.append(p)

    # vertices of quad1 inside quad2
    inside_mask = _is_inside_convex(quad1, quad2)
    for i in range(4):
        if inside_mask[i]:
            pts.append(quad1[i])

    # vertices of quad2 inside quad1
    inside_mask = _is_inside_convex(quad2, quad1)
    for i in range(4):
        if inside_mask[i]:
            pts.append(quad2[i])

    if len(pts) < 3:
        return torch.zeros(0, 2, device=quad1.device, dtype=quad1.dtype)

    # deduplicate
    pts_t = torch.stack(pts)
    dists = ((pts_t.unsqueeze(0) - pts_t.unsqueeze(1)) ** 2).sum(dim=-1)
    # keep points that are not too close to any earlier point
    keep = torch.ones(len(pts_t), dtype=torch.bool)
    for i in range(len(pts_t)):
        if keep[i] and (dists[i, :i] < 1e-10).any():
            keep[i] = False
    pts_t = pts_t[keep]

    if pts_t.shape[0] < 3:
        return torch.zeros(0, 2, device=quad1.device, dtype=quad1.dtype)

    # convex hull
    return _convex_hull_2d(pts_t)


def poly_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise polygon IoU between two sets of OBBs.

    Args:
        boxes1: (N, 5) (cx, cy, w, h, theta) in pixel coords
        boxes2: (M, 5)

    Returns:
        (N, M) IoU matrix
    """
    verts1 = xywhr_to_xyxyxyxy(boxes1)     # (N, 4, 2)
    verts2 = xywhr_to_xyxyxyxy(boxes2)     # (M, 4, 2)

    areas1 = _polygon_area(verts1)          # (N,)
    areas2 = _polygon_area(verts2)          # (M,)

    N, M = verts1.shape[0], verts2.shape[0]
    ious = torch.zeros(N, M, device=boxes1.device, dtype=boxes1.dtype)

    for i in range(N):
        x1_min, y1_min = verts1[i, :, 0].min(), verts1[i, :, 1].min()
        x1_max, y1_max = verts1[i, :, 0].max(), verts1[i, :, 1].max()

        for j in range(M):
            x2_min, y2_min = verts2[j, :, 0].min(), verts2[j, :, 1].min()
            x2_max, y2_max = verts2[j, :, 0].max(), verts2[j, :, 1].max()

            if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
                ious[i, j] = 0.0
                continue

            inter_poly = _convex_intersection(verts1[i], verts2[j])
            if inter_poly.shape[0] < 3:
                ious[i, j] = 0.0
            else:
                inter_area = _polygon_area(inter_poly)
                union = areas1[i] + areas2[j] - inter_area
                ious[i, j] = (inter_area / union.clamp(min=1e-10)).clamp(0.0, 1.0)

    return ious


def poly_iou_batch(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Fast batched polygon IoU (alias for poly_iou)."""
    return poly_iou(boxes1, boxes2)
