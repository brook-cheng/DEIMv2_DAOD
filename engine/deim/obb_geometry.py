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


def periodic_angle_distance(pred: Tensor, target: Tensor) -> Tensor:
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
    diff = (pred - target).abs()
    d = torch.remainder(diff, torch.pi)
    return torch.minimum(d, torch.pi - d)


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


def external_rect_to_oriented_box(
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
        (..., 5)  —  (cx, cy, w, h, theta),θ belongs to [0,pi].
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

    theta = torch.atan2(w_dy, w_dx) % torch.pi

    return torch.stack([cx, cy, w_len, h_len, theta], dim=-1)


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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

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

    # ── affine_obb 自测 ──
    print("\n=== affine_obb tests ===")
    tol_strict = 1e-4

    # Test 1: 空框安全
    empty = torch.zeros(0, 5, device=device)
    r = affine_obb(empty, 2.0, 3.0, 10, 20)
    assert r.shape == (0, 5), f"空框应保持 (0,5)，得到 {r.shape}"
    passed += 1
    print("  [PASS] empty boxes safety")

    # Test 2: 纯平移 w/h/θ 不变
    boxes_t = torch.tensor(
        [[100.0, 200.0, 80.0, 40.0, 0.785398], [300.0, 400.0, 60.0, 30.0, 2.094395]],
        device=device,
    )
    r_t = affine_obb(boxes_t, 1.0, 1.0, 50.0, 30.0)
    err_t = (r_t[..., 2:5] - boxes_t[..., 2:5]).abs().max().item()
    cx_ok = (r_t[:, 0] - (boxes_t[:, 0] + 50.0)).abs().max().item()
    cy_ok = (r_t[:, 1] - (boxes_t[:, 1] + 30.0)).abs().max().item()
    if err_t < tol_strict and cx_ok < tol_strict and cy_ok < tol_strict:
        passed += 1
        print(f"  [PASS] pure translation: whθ max_err={err_t:.2e}, cx_err={cx_ok:.2e}")
    else:
        failed += 1
        print(
            f"  [FAIL] pure translation: whθ_err={err_t:.2e}, cx={cx_ok:.2e}, cy={cy_ok:.2e}"
        )

    # Test 3: 等比缩放 (sx=sy) w/h 等比变化，θ 不变
    boxes_u = torch.tensor([[100.0, 200.0, 80.0, 40.0, 1.2]], device=device)
    r_u = affine_obb(boxes_u, 2.0, 2.0, 0.0, 0.0)
    w_ok = abs(r_u[0, 2].item() - 160.0) < 1.0
    h_ok = abs(r_u[0, 3].item() - 80.0) < 1.0
    θ_ok = abs(r_u[0, 4].item() - 1.2) < tol_strict
    if w_ok and h_ok and θ_ok:
        passed += 1
        print(
            f"  [PASS] uniform scale: w={r_u[0,2]:.1f} h={r_u[0,3]:.1f} θ={r_u[0,4]:.4f}"
        )
    else:
        failed += 1
        print(f"  [FAIL] uniform scale: {r_u[0].tolist()}")

    # Test 4: affine_obb 结果与"手动变换顶点再重拟合"一致
    boxes_r = torch.tensor([[50.0, 80.0, 60.0, 20.0, 0.523599]], device=device)
    r_fwd = affine_obb(boxes_r, 0.5, 0.8, 0.0, 0.0)
    # 手动计算 ground truth: 变换顶点后重拟合
    v_orig = xywhr_to_xyxyxyxy(boxes_r).clone()
    v_orig[..., 0] *= 0.5
    v_orig[..., 1] *= 0.8
    gt_refit = xyxyxyxy_to_xywhr(v_orig)
    param_err = (r_fwd - gt_refit).abs().max().item()
    # 验证变换后顶点位置正确
    v_fwd = xywhr_to_xyxyxyxy(r_fwd)
    v_diff = (v_fwd - v_orig).abs().max().item()
    if param_err < 1e-3 and v_diff < 5.0:
        passed += 1
        print(
            f"  [PASS] affine_obb vs manual refit: param_err={param_err:.2e}, v_diff={v_diff:.2e}"
        )
    else:
        failed += 1
        print(
            f"  [FAIL] affine_obb vs manual refit: param_err={param_err:.2e}, v_diff={v_diff:.2e}"
        )

    # Test 5: 各向异性缩放 θ 会改变
    boxes_a = torch.tensor(
        [[200.0, 200.0, 100.0, 40.0, torch.pi / 4]], device=device
    )  # 45°, w≠h
    r_a = affine_obb(boxes_a, 2.0, 0.5, 0.0, 0.0)
    # 手动验证：对 4 顶点各自缩放后重拟合
    v_orig = xywhr_to_xyxyxyxy(boxes_a)
    v_manual = v_orig.clone()
    v_manual[..., 0] *= 2.0
    v_manual[..., 1] *= 0.5
    refit = xyxyxyxy_to_xywhr(v_manual)
    param_err = (r_a - refit).abs().max().item()
    # θ 不应保持 45°
    θ_changed = abs(r_a[0, 4].item() - torch.pi / 4) > 0.01
    if param_err < tol_strict and θ_changed:
        passed += 1
        print(
            f"  [PASS] anisotropic scale: θ {boxes_a[0,4]:.4f}→{r_a[0,4]:.4f}, param_err={param_err:.2e}"
        )
    else:
        failed += 1
        print(
            f"  [FAIL] anisotropic scale: θ_changed={θ_changed}, param_err={param_err:.2e}"
        )
        print(f"         affine_obb: {r_a[0].tolist()}")
        print(f"         manual:     {refit[0].tolist()}")

    # Test 6: 翻转对合性 (flip twice = identity)
    # 模拟水平翻转：sx=-1, tx=W, sy=1, ty=0
    W_img = 640.0
    boxes_f = torch.tensor(
        [[200.0, 300.0, 100.0, 50.0, 0.785398], [500.0, 100.0, 80.0, 30.0, 2.094395]],
        device=device,
    )
    flipped = affine_obb(boxes_f, -1.0, 1.0, W_img, 0.0)
    restored = affine_obb(flipped, -1.0, 1.0, W_img, 0.0)
    v_f_err = _vertex_roundtrip_error(
        xywhr_to_xyxyxyxy(boxes_f), xywhr_to_xyxyxyxy(restored)
    )
    if v_f_err < tol_strict:
        passed += 1
        print(f"  [PASS] flip roundtrip: vertex_err={v_f_err:.2e}")
    else:
        failed += 1
        print(f"  [FAIL] flip roundtrip: vertex_err={v_f_err:.2e}")

    # ── affine_obb_matrix 自测 ──
    print("\n=== affine_obb_matrix tests ===")
    tol_m = 1e-3

    # Test M1: 单位仿射
    mat_I = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
    boxes_m1 = torch.tensor([[200.0, 300.0, 80.0, 40.0, 0.785398]], device=device)
    r_m1 = affine_obb_matrix(boxes_m1, mat_I)
    err_m1 = (r_m1 - boxes_m1).abs().max().item()
    if err_m1 < 1e-4:
        passed += 1
        print(f"  [PASS] identity: err={err_m1:.2e}")
    else:
        failed += 1
        print(f"  [FAIL] identity: err={err_m1:.2e}")

    # Test M2: 纯旋转 (绕原点 φ=30°)
    phi = math.radians(30.0)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    A_rot = torch.tensor([[cos_p, -sin_p], [sin_p, cos_p]], device=device)
    mat_rot = torch.cat([A_rot, torch.zeros(2, 1, device=device)], dim=1)
    # 方框 80×80，中心 (100,200)，θ=0
    boxes_m2 = torch.tensor([[100.0, 200.0, 80.0, 80.0, 0.0]], device=device)
    r_m2 = affine_obb_matrix(boxes_m2, mat_rot)
    # 中心旋转后：A_rot @ [100,200]
    ct_exp = A_rot @ torch.tensor([100.0, 200.0])
    assert abs(r_m2[0, 0].item() - ct_exp[0].item()) < 1.0
    assert abs(r_m2[0, 1].item() - ct_exp[1].item()) < 1.0
    # 方框 w,h 应接近 80（顶点重拟合后方框 w≈h，微小浮动）
    w_ok = abs(r_m2[0, 2].item() - 80.0) < 2.0 and abs(r_m2[0, 3].item() - 80.0) < 2.0
    # θ 应约为 30°（对于正方形，可能是 30° 或 30°+90°=120°，两者几何等价）
    θ_exp = phi % math.pi
    θ_alt = (phi + math.pi / 2) % math.pi
    θ_got = r_m2[0, 4].item() % math.pi
    θ_ok = min(abs(θ_got - θ_exp), abs(θ_got - θ_alt)) < 0.05
    if w_ok and θ_ok:
        passed += 1
        print(
            f"  [PASS] pure rotation: ct=({r_m2[0,0]:.1f},{r_m2[0,1]:.1f}) w={r_m2[0,2]:.1f} h={r_m2[0,3]:.1f} θ={r_m2[0,4]:.4f}"
        )
    else:
        failed += 1
        print(f"  [FAIL] pure rotation: {r_m2[0].tolist()}")

    # Test M3: 与手动顶点变换一致
    phi2 = math.radians(15.0)
    s2 = 1.2
    A_m3 = torch.tensor(
        [
            [s2 * math.cos(phi2), -s2 * math.sin(phi2)],
            [s2 * math.sin(phi2), s2 * math.cos(phi2)],
        ],
        device=device,
    )
    b_m3 = torch.tensor([50.0, -30.0], device=device)
    mat_m3 = torch.cat([A_m3, b_m3[:, None]], dim=1)
    r_m3 = affine_obb_matrix(boxes_m2, mat_m3)
    # 手动: 顶点变换后重拟合
    v_man = xywhr_to_xyxyxyxy(boxes_m2).clone().to(device)
    v_man = v_man @ A_m3.T + b_m3
    gt_m3 = xyxyxyxy_to_xywhr(v_man)
    err_m3 = (r_m3 - gt_m3).abs().max().item()
    if err_m3 < tol_m:
        passed += 1
        print(f"  [PASS] manual consistency: err={err_m3:.2e}")
    else:
        failed += 1
        print(f"  [FAIL] manual consistency: err={err_m3:.2e}")

    # Test M4: 各向异性 + 旋转
    boxes_m4 = torch.tensor(
        [[200.0, 200.0, 100.0, 40.0, math.pi / 4]], device=device
    )  # 45°, w≠h
    A_m4 = torch.tensor([[1.5, -0.5], [0.3, 0.8]], device=device)
    b_m4 = torch.tensor([20.0, 10.0], device=device)
    mat_m4 = torch.cat([A_m4, b_m4[:, None]], dim=1)
    r_m4 = affine_obb_matrix(boxes_m4, mat_m4)
    v_m4 = xywhr_to_xyxyxyxy(boxes_m4).clone().to(device)
    v_m4 = v_m4 @ A_m4.T + b_m4
    gt_m4 = xyxyxyxy_to_xywhr(v_m4)
    err_m4 = (r_m4 - gt_m4).abs().max().item()
    θ_changed = abs(r_m4[0, 4].item() - boxes_m4[0, 4].item()) > 0.01
    if err_m4 < tol_m and θ_changed:
        passed += 1
        print(
            f"  [PASS] aniso+rot: θ {boxes_m4[0,4]:.4f}→{r_m4[0,4]:.4f}, err={err_m4:.2e}"
        )
    else:
        failed += 1
        print(f"  [FAIL] aniso+rot: θ_changed={θ_changed}, err={err_m4:.2e}")

    # Test M5: 空框安全
    r_m5 = affine_obb_matrix(torch.zeros(0, 5, device=device), mat_m3)
    assert r_m5.shape == (0, 5)
    passed += 1
    print("  [PASS] empty boxes")

    print(f"\n{'='*40}")
    print(f"Passed: {passed},  Failed: {failed}")
    if failed:
        raise SystemExit(1)
