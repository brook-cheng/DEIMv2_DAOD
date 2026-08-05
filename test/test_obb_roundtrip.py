"""Tests for OBB geometry round-trip, angle range, w/h swap, and pipeline scaling.

Runnable as: python -m pytest test/test_obb_roundtrip.py -v
"""
import sys, os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import math
import torch
import pytest
from engine.deim.obb_geometry import xywhr_to_xyxyxyxy, xyxyxyxy_to_xywhr
from engine.deim.obb_angle_contract import norm_to_physical_rad, physical_rad_to_norm

TOL = 1e-5


def _vertex_err(orig_v, recon_v):
    d1 = ((orig_v.unsqueeze(-2) - recon_v.unsqueeze(-3)) ** 2).sum(-1).amin(-1)
    d2 = ((recon_v.unsqueeze(-2) - orig_v.unsqueeze(-3)) ** 2).sum(-1).amin(-1)
    return torch.max(d1.max(-1).values, d2.max(-1).values).max()


def _check_vertex(xywhr, tol=TOL):
    v = xywhr_to_xyxyxyxy(xywhr)
    recon = xyxyxyxy_to_xywhr(v)
    v_recon = xywhr_to_xyxyxyxy(recon)
    err = _vertex_err(v, v_recon)
    assert err < tol, f"vertex error {err:.2e} for {xywhr[0].tolist()}"


def _check_param(xywhr, tol=TOL):
    v = xywhr_to_xyxyxyxy(xywhr)
    recon = xyxyxyxy_to_xywhr(v)
    err = (recon - xywhr).abs().max().item()
    assert err < tol, f"param error {err:.2e} for {xywhr[0].tolist()} → {recon[0].tolist()}"


# ── Round-trip: w > h (exact param match) ──


@pytest.mark.parametrize("w,h,theta", [
    (0.4, 0.2, 0.0),
    (0.4, 0.2, 0.523599),
    (0.4, 0.2, 0.785398),
    (0.4, 0.2, 1.570796),
    (0.4, 0.2, 2.094395),
])
def test_canonical_w_gt_h(w, h, theta):
    xywhr = torch.tensor([[0.5, 0.5, w, h, theta]])
    _check_vertex(xywhr)
    _check_param(xywhr)


# ── Round-trip: w < h (vertex match only, params may swap) ──


@pytest.mark.parametrize("w,h,theta", [
    (0.2, 0.4, 0.0),
    (0.2, 0.4, 0.523599),
    (0.4, 0.41, 0.3),
    (0.05, 0.40, 0.3),
])
def test_swapped_w_lt_h(w, h, theta):
    xywhr = torch.tensor([[0.5, 0.5, w, h, theta]])
    _check_vertex(xywhr)


# ── Square: vertex match, θ may differ by π/2 ──


@pytest.mark.parametrize("theta", [0.0, 0.785398])
def test_square(theta):
    xywhr = torch.tensor([[0.5, 0.5, 0.3, 0.3, theta]])
    _check_vertex(xywhr)


# ── Extreme aspect ratios ──


def test_extreme_ratios():
    for w, h in [(0.8, 0.02), (0.02, 0.8), (0.9, 0.001), (0.4, 0.05), (0.05, 0.4)]:
        _check_vertex(torch.tensor([[0.5, 0.5, w, h, 0.3]]))


# ── Angle boundary values ──


def test_angle_boundaries():
    for theta in [1e-6, torch.pi - 1e-6, torch.pi / 2 - 1e-6, torch.pi / 2 + 1e-6]:
        _check_vertex(torch.tensor([[0.5, 0.5, 0.3, 0.1, theta]]))


# ── Near-π boundary: param won't match but vertex will ──


def test_near_pi_boundary():
    _check_vertex(torch.tensor([[0.5, 0.5, 0.4, 0.2, 3.131593]]))


# ── Degenerate coordinates ──


def test_degenerate_coords():
    for cx, cy in [(0.0, 0.0), (1.0, 1.0)]:
        _check_vertex(torch.tensor([[cx, cy, 0.2, 0.1, 0.785398]]))


def test_tiny_box():
    _check_vertex(torch.tensor([[0.5, 0.5, 1e-4, 1e-4, 0.785398]]))


# ── Batch round-trip ──


def test_batch_roundtrip():
    torch.manual_seed(42)
    N = 500
    obbs = torch.cat([
        torch.rand(N, 1), torch.rand(N, 1),
        torch.rand(N, 1) * 0.5, torch.rand(N, 1) * 0.5,
        torch.rand(N, 1) * torch.pi,
    ], dim=-1)
    v = xywhr_to_xyxyxyxy(obbs)
    recon = xyxyxyxy_to_xywhr(v)
    v_recon = xywhr_to_xyxyxyxy(recon)
    err = _vertex_err(v, v_recon)
    assert err < 1e-4, f"batch vertex error {err:.2e}"

    mask_wh = obbs[:, 2] >= obbs[:, 3]
    if mask_wh.sum() > 0:
        spatial_err = (recon[mask_wh, :4] - obbs[mask_wh, :4]).abs().max().item()
        ang_diff = (recon[mask_wh, 4] - obbs[mask_wh, 4]).abs()
        ang_diff = torch.minimum(ang_diff % torch.pi, torch.pi - (ang_diff % torch.pi))
        assert spatial_err < 1e-3, f"spatial error {spatial_err:.2e}"
        assert ang_diff.max().item() < 1e-3, f"angle error {ang_diff.max().item():.2e}"


# ── Angle range: output in [0, π) ──


def test_angle_range():
    torch.manual_seed(42)
    N = 200
    obbs = torch.cat([
        torch.rand(N, 1), torch.rand(N, 1),
        torch.rand(N, 1) * 0.5, torch.rand(N, 1) * 0.5,
        torch.rand(N, 1) * torch.pi,
    ], dim=-1)
    v = xywhr_to_xyxyxyxy(obbs)
    recon = xyxyxyxy_to_xywhr(v)
    thetas = recon[:, 4]
    assert (thetas >= 0).all(), f"min theta={thetas.min():.6f} < 0"
    assert (thetas < math.pi).all(), f"max theta={thetas.max():.6f} >= π"


def test_angle_3pi_over_4_not_folded():
    # 3π/4 与 -π/4 周期等价（同一方向），但等比契约要求输出 [0,π) 内的 3π/4，
    # 旧 shifted 实现会折叠到 -π/4
    obb = torch.tensor([[0.5, 0.5, 0.4, 0.2, 3 * math.pi / 4]])
    v = xywhr_to_xyxyxyxy(obb)
    recon = xyxyxyxy_to_xywhr(v)
    theta = recon[0, 4].item()
    assert theta >= 0, f"等比契约要求 θ ∈ [0,π)，got {theta:.6f}"
    assert abs(theta - 3 * math.pi / 4) < 1e-5, f"期望 3π/4={3*math.pi/4:.6f}, got {theta:.6f}"


# ── Decoder scaling: 等比 [0,1] ↔ [0, π) ──


def test_decoder_output_scaling():
    x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    theta = norm_to_physical_rad(x)
    assert abs(theta[0].item() - 0.0) < 1e-6
    assert abs(theta[1].item() - math.pi / 4) < 1e-6
    assert abs(theta[2].item() - math.pi / 2) < 1e-6
    assert abs(theta[3].item() - 3 * math.pi / 4) < 1e-6
    assert abs(theta[4].item() - math.pi) < 1e-6


def test_decoder_input_scaling():
    theta = torch.tensor([0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi - 1e-6])
    x = physical_rad_to_norm(theta)
    assert abs(x[0].item() - 0.0) < 1e-6
    assert abs(x[1].item() - 0.25) < 1e-6
    assert abs(x[2].item() - 0.5) < 1e-6
    assert abs(x[3].item() - 0.75) < 1e-6
    assert x[4].item() < 1.0


def test_angle_pipeline_roundtrip():
    """End-to-end angle scaling: decoder output → criterion input → back.

    Chain: θ_ext → (θ+π/4)/π → θ_int → (θ_int-0.25)*π → should equal θ_ext.

    Also verifies distance2bbox_obb internal loop consistency:
    θ_int → *π → [0,π] → /π → θ_int (identity within decoder loop).
    """
    torch.manual_seed(42)
    for _ in range(100):
        theta_ext = torch.rand(1).item() * math.pi - math.pi / 4  # [-π/4, 3π/4)

        # decoder output → criterion normalization
        theta_int = (theta_ext + math.pi / 4) / math.pi  # → [0,1]
        assert 0.0 <= theta_int <= 1.0 + 1e-6

        # criterion → back to decoder internal
        theta_back = (theta_int - 0.25) * math.pi  # → [-π/4, 3π/4)
        assert abs(theta_back - theta_ext) < 1e-6, (
            f"round-trip drift: θ_ext={theta_ext:.6f} → θ_back={theta_back:.6f}"
        )

        # decoder internal loop: *π → /π must be identity
        theta_loop = theta_int * math.pi
        theta_loop_back = theta_loop / math.pi
        assert abs(theta_loop_back - theta_int) < 1e-6


def test_denoising_angle_scaling():
    """DN path: θ_ext → (θ+π/4)/π → inverse_sigmoid → sigmoid → should recover θ_int."""
    import torch.nn.functional as F

    for theta_ext in [-0.7, 0.0, 0.5, 1.5, 2.3]:
        theta_int = (theta_ext + math.pi / 4) / math.pi
        theta_int_clamped = max(min(theta_int, 1.0 - 1e-4), 1e-4)
        unact = math.log(theta_int_clamped / (1 - theta_int_clamped))
        recovered = 1 / (1 + math.exp(-unact))
        assert abs(recovered - theta_int_clamped) < 1e-4, (
            f"DN scaling: θ_ext={theta_ext} → recovered={recovered:.6f} ≠ {theta_int_clamped:.6f}"
        )


# ── w/h swap robustness ──


@pytest.mark.parametrize("w,h", [
    (0.4, 0.401), (0.4, 0.41), (0.4, 0.42),
    (0.401, 0.4), (0.41, 0.4), (0.42, 0.4),
    (0.05, 0.40), (0.20, 0.40), (0.40, 0.05), (0.40, 0.20),
])
def test_wh_swap_vertex(w, h):
    xywhr = torch.tensor([[0.5, 0.5, w, h, 0.3]])
    _check_vertex(xywhr)


def test_wh_swap_probiou_invariance():
    try:
        from engine.deim.obb_ops import batch_probiou
    except ImportError:
        pytest.skip("batch_probiou not available")

    ref = torch.tensor([[0.5, 0.5, 0.4, 0.4, 0.3]])
    cases = [
        torch.tensor([[0.5, 0.5, 0.4, 0.41, 0.3]]),
        torch.tensor([[0.5, 0.5, 0.41, 0.4, 0.3]]),
        torch.tensor([[0.5, 0.5, 0.1, 0.80, 0.3]]),
        torch.tensor([[0.5, 0.5, 0.80, 0.1, 0.3]]),
    ]
    for pred in cases:
        iou_before = batch_probiou(ref, pred)[0, 0].item()
        v = xywhr_to_xyxyxyxy(pred)
        pred_rt = xyxyxyxy_to_xywhr(v)
        iou_after = batch_probiou(ref, pred_rt)[0, 0].item()
        assert abs(iou_before - iou_after) < 1e-5, (
            f"ProbIoU changed: {iou_before:.6f} → {iou_after:.6f} for {pred[0].tolist()}"
        )


# ── DEIM vs Ultralytics (skip if not available) ──


def test_deim_vs_ultralytics():
    try:
        sys.path.insert(0, "/home/cx/win_dir/thired/ultralytics_update")
        from ultralytics.utils.ops import xywhr2xyxyxyxy as ult_to_v
        from ultralytics.utils.ops import xyxyxyxy2xywhr as ult_from_v
    except ImportError:
        pytest.skip("Ultralytics not available")

    cases = [
        torch.tensor([[0.5, 0.5, 0.4, 0.2, 0.0]]),
        torch.tensor([[0.5, 0.5, 0.4, 0.2, 0.785398]]),
        torch.tensor([[0.5, 0.5, 0.2, 0.4, 0.0]]),
        torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.785398]]),
    ]
    for obb in cases:
        d_out = xyxyxyxy_to_xywhr(xywhr_to_xyxyxyxy(obb))
        u_out = ult_from_v(ult_to_v(obb).reshape(-1, 8))
        wh_ok = (d_out[0, 2:4] - u_out[0, 2:4]).abs().max().item() < 1e-4
        ang_diff = abs(d_out[0, 4].item() - u_out[0, 4].item())
        ang_diff = min(ang_diff % math.pi, math.pi - (ang_diff % math.pi))
        assert wh_ok, f"w/h mismatch: DEIM={d_out[0, 2:4]} ULT={u_out[0, 2:4]}"
        assert ang_diff < 1e-4, f"angle mismatch: DEIM={d_out[0, 4]:.6f} ULT={u_out[0, 4]:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
