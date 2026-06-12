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


def xywhr_to_xyxyxyxy(xywhr: Tensor) -> Tensor:
    """Convert OBB (cx, cy, w, h, theta) to four corner vertices.

    Args:
        xywhr: (..., 5)  —  (cx, cy, w, h, theta)  in normalized coords,θ belongs to [0,π].

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
        (..., 5)  —  (cx, cy, w, h, theta)  in normalized coords,θ belongs to [0,π].
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
    theta = torch.atan2(w_dy, w_dx) % torch.pi
    return torch.stack([ctr[..., 0], ctr[..., 1], w, h, theta], dim=-1)


def oriented_box_to_external_rect(obbs: Tensor) -> Tuple[Tensor, Tensor]:
    """OBB -> external rectangle + vertex offsets (epsilon, eta).

    Args:
        obbs:  (..., 5)  —  (cx, cy, w, h, theta),θ belongs to [0,pi].

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


def external_rect_to_oriented_box(
    external_rect: Tensor, vertex_offsets: Tensor
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

    Returns:
        (..., 5)  —  (cx, cy, w, h, theta),θ belongs to [0,pi].
    """

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

    len_ab = torch.sqrt(edge_ab[..., 0] ** 2 + edge_ab[..., 1] ** 2)
    len_bc = torch.sqrt(edge_bc[..., 0] ** 2 + edge_bc[..., 1] ** 2)

    # w = longer edge, h = shorter edge
    w_is_ab = len_ab >= len_bc
    w_len = torch.where(w_is_ab, len_ab, len_bc)
    h_len = torch.where(w_is_ab, len_bc, len_ab)

    w_dx = torch.where(w_is_ab, edge_ab[..., 0], edge_bc[..., 0])
    w_dy = torch.where(w_is_ab, edge_ab[..., 1], edge_bc[..., 1])

    theta = torch.atan2(w_dy, w_dx) % torch.pi

    return torch.stack([cx, cy, w_len, h_len, theta], dim=-1)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cpu")

    passed = 0
    failed = 0

    def _vertex_roundtrip_error(orig_v: Tensor, recon_v: Tensor) -> Tensor:
        """Max bidirectional nearest-neighbour distance between vertex sets."""
        d1 = (
            ((orig_v.unsqueeze(-2) - recon_v.unsqueeze(-3)) ** 2)
            .sum(dim=-1)
            .amin(dim=-1)
        )
        d2 = (
            ((recon_v.unsqueeze(-2) - orig_v.unsqueeze(-3)) ** 2)
            .sum(dim=-1)
            .amin(dim=-1)
        )
        return torch.max(d1.max(dim=-1).values, d2.max(dim=-1).values).max()

    def _check(name, obbs, tol=1e-5):
        global passed, failed
        ext, vo = oriented_box_to_external_rect(obbs)
        recon = external_rect_to_oriented_box(ext, vo)

        # Check geometric equivalence via vertices (tolerates w<->h + angle swap)
        orig_v = xywhr_to_xyxyxyxy(obbs)
        recon_v = xywhr_to_xyxyxyxy(recon)
        # Chamfer-like bidirectional max vertex error
        v_err = _vertex_roundtrip_error(orig_v, recon_v)

        param_err = (recon - obbs).abs().max().item()

        if v_err < tol:
            passed += 1
            print(
                f"  [PASS] {name}:  vertex_err={v_err:.2e}, param_err={param_err:.2e}"
            )
        else:
            failed += 1
            print(
                f"  [FAIL] {name}:  vertex_err={v_err:.2e}, param_err={param_err:.2e}"
            )
            print(f"         original:  {obbs[0].tolist()}")
            print(f"         recon:     {recon[0].tolist()}")

    print("=== Single-box tests ===")
    _check("theta=pi/4", torch.tensor([[0.6, 0.5, 0.4, 0.2, 0.785398]], device=device))
    _check("theta=0", torch.tensor([[0.4, 0.4, 0.3, 0.1, 0.0]], device=device))
    _check("theta=pi/6", torch.tensor([[0.5, 0.5, 0.4, 0.2, 0.523599]], device=device))
    _check("theta=2pi/3", torch.tensor([[0.6, 0.5, 0.4, 0.2, 2.094395]], device=device))
    _check(
        "theta=pi-0.01",
        torch.tensor([[0.5, 0.5, 0.3, 0.15, torch.pi - 0.01]], device=device),
    )
    _check("square pi/4", torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.785398]], device=device))
    _check("narrow 1.2rad", torch.tensor([[0.5, 0.5, 0.05, 0.4, 1.2]], device=device))
    _check("wider box 0.8", torch.tensor([[0.5, 0.5, 0.4, 0.1, 0.8]], device=device))
    _check(
        "theta=pi/2", torch.tensor([[0.5, 0.5, 0.4, 0.2, torch.pi / 2]], device=device)
    )
    _check("theta~0 (1e-6)", torch.tensor([[0.5, 0.5, 0.3, 0.1, 1e-6]], device=device))
    _check(
        "theta~pi/2 (1e-6 off)",
        torch.tensor([[0.5, 0.5, 0.3, 0.1, torch.pi / 2 - 1e-6]], device=device),
    )

    print(f"\n=== Random batch ({2000} boxes) ===")
    N = 2000
    obbs = torch.cat(
        [
            torch.rand(N, 1, device=device),
            torch.rand(N, 1, device=device),
            torch.rand(N, 1, device=device) * 0.5,
            torch.rand(N, 1, device=device) * 0.5,
            torch.rand(N, 1, device=device) * torch.pi,
        ],
        dim=-1,
    )

    ext_batch, vo_batch = oriented_box_to_external_rect(obbs)
    recon_batch = external_rect_to_oriented_box(ext_batch, vo_batch)
    err = (recon_batch - obbs).abs().max().item()
    _check(f"{N} random boxes", obbs, tol=1e-5)

    print(f"\n{'='*40}")
    print(f"Passed: {passed},  Failed: {failed}")
    if failed:
        raise SystemExit(1)
