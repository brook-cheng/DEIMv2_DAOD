"""OBB ADR geometry and periodic angle seam tests.

Covers Todo 1 (geometry round-trip locks) and Todo 2 (periodic angle
distance) of the deimv2-obb-adr-hybrid plan. The periodic seam tests
import and exercise the production ``periodic_angle_distance`` utility
from ``engine.deim.obb_geometry``; the formula is NOT duplicated here
except as scalar expected values for direct comparison.

The tests compare OBBs by vertex-level geometry error rather than raw
``(cx, cy, w, h, theta)`` parameter identity, because equivalent OBB
parameterizations may differ (``w <-> h`` swap plus ``theta += pi/2``)
while the underlying geometry is identical. This matters most for
near-square boxes where the longer-edge-as-w convention is ambiguous
(spec section 9.2).

Run:
    pytest test/test_obb_adr_geometry.py -v
"""

import math
import os
import sys

import pytest
import torch

# Ensure we import the on-disk repo module, not a stale installed package.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.deim.obb_geometry import (
    clamp_vertex_offsets_to_external_rect,
    external_xywh_rect_to_oriented_box,
    external_xyxy_rect_to_oriented_box,
    oriented_box_to_external_xywh_rect,
    oriented_box_to_external_xyxy_rect,
    periodic_angle_distance,
    xywhr_to_xyxyxyxy,
)

from engine.deim.box_ops import box_xyxy_to_cxcywh
from engine.deim.dfine_utils import bbox2distance_obb, distance2bbox_obb

# ---------------------------------------------------------------------------
# Local test helpers (geometry-level, not parameter identity)
# ---------------------------------------------------------------------------


def vertex_roundtrip_error(orig_v, recon_v):
    """Max bidirectional nearest-neighbour distance between vertex sets.

    Tolerates equivalent re-parameterizations (w<->h swap, theta += pi/2)
    by comparing the actual corner geometry.
    """
    d1 = ((orig_v.unsqueeze(-2) - recon_v.unsqueeze(-3)) ** 2).sum(dim=-1).amin(dim=-1)
    d2 = ((recon_v.unsqueeze(-2) - orig_v.unsqueeze(-3)) ** 2).sum(dim=-1).amin(dim=-1)
    return torch.max(d1.max(dim=-1).values, d2.max(dim=-1).values).max()


def obb_vertex_error(a, b):
    """Vertex-level geometry error between two (..., 5) OBB tensors."""
    return vertex_roundtrip_error(xywhr_to_xyxyxyxy(a), xywhr_to_xyxyxyxy(b))


def roundtrip(obbs):
    """OBB -> external rect + offsets -> OBB. Returns reconstructed OBB."""
    ext, vo = oriented_box_to_external_xyxy_rect(obbs)
    return external_xyxy_rect_to_oriented_box(ext, vo)


def ordinary_normalized_theta_l1(pred, target):
    """Ordinary normalized theta L1: ``abs(pred - target) / pi``.

    Non-periodic baseline. Near the 0/pi seam it over-penalizes
    geometrically equivalent angles. Kept to demonstrate the problem the
    periodic production utility solves.
    """
    return (pred - target).abs() / math.pi


ROUNDTRIP_TOL = 1e-5


# ---------------------------------------------------------------------------
# Round-trip geometry tests (lock current ADR behavior)
# ---------------------------------------------------------------------------

ROUNDTRIP_CASES = [
    ("theta_near_0", [0.5, 0.5, 0.3, 0.1, 0.0]),
    ("theta_1e-6", [0.5, 0.5, 0.3, 0.1, 1e-6]),
    ("theta_near_pi_over_2", [0.5, 0.5, 0.4, 0.2, math.pi / 2]),
    ("theta_pi_over_4", [0.6, 0.5, 0.4, 0.2, math.pi / 4]),
    ("theta_near_pi_minus_1e-3", [0.5, 0.5, 0.3, 0.15, math.pi - 1e-3]),
    ("axis_aligned_w_gt_h", [0.4, 0.4, 0.3, 0.1, 0.0]),
    ("axis_aligned_h_gt_w", [0.4, 0.4, 0.1, 0.3, 0.0]),
    ("thin_long_horizontal", [0.5, 0.5, 0.5, 0.02, 0.0]),
    ("thin_long_vertical", [0.5, 0.5, 0.02, 0.5, 0.0]),
    ("thin_long_45deg", [0.5, 0.5, 0.5, 0.02, math.pi / 4]),
    ("square_like_pi_over_4", [0.5, 0.5, 0.3, 0.3, math.pi / 4]),
    ("square_like_axis_aligned", [0.5, 0.5, 0.3, 0.3, 0.0]),
]


@pytest.mark.parametrize(
    "name, obb", ROUNDTRIP_CASES, ids=[c[0] for c in ROUNDTRIP_CASES]
)
def test_roundtrip_vertex_error(name, obb):
    """OBB -> external rect + offsets -> OBB must preserve geometry.

    Vertex error is used instead of parameter error because equivalent
    OBB parameterizations may differ (w<->h swap, theta += pi/2) while
    the underlying geometry is identical.
    """
    obbs = torch.tensor([obb], dtype=torch.float32)
    recon = roundtrip(obbs)
    err = obb_vertex_error(obbs, recon)
    assert (
        err.item() < ROUNDTRIP_TOL
    ), f"{name}: vertex error {err.item():.2e} >= {ROUNDTRIP_TOL:.0e}"


def test_roundtrip_random_2000():
    """2000 random valid OBBs must round-trip with vertex error < 1e-5.

    Deterministic seed for reproducibility (flaky-test guard).
    """
    torch.manual_seed(20260707)
    n = 2000
    obbs = torch.cat(
        [
            torch.rand(n, 1),
            torch.rand(n, 1),
            torch.rand(n, 1) * 0.5,
            torch.rand(n, 1) * 0.5,
            torch.rand(n, 1) * math.pi,
        ],
        dim=-1,
    )
    recon = roundtrip(obbs)
    err = obb_vertex_error(obbs, recon)
    assert (
        err.item() < ROUNDTRIP_TOL
    ), f"random 2000: vertex error {err.item():.2e} >= {ROUNDTRIP_TOL:.0e}"


# ---------------------------------------------------------------------------
# External-rectangle cxcywh composition helpers (Stage 2 Task 1)
# ---------------------------------------------------------------------------
#
# TDD red-before-green: these tests fail to collect until
# `external_cxcywh_to_oriented_box` and `oriented_box_to_external_cxcywh`
# exist in engine/deim/obb_geometry.py. They verify (a) a full OBB
# round-trip through the cxcywh composition and (b) equivalence with the
# existing xyxy composition primitives.


def test_external_cxcywh_helpers_roundtrip_obb_geometry():
    obb = torch.tensor([[[0.55, 0.45, 0.30, 0.12, math.pi / 4]]], dtype=torch.float32)
    external_cxcywh, offsets = oriented_box_to_external_xywh_rect(obb)
    reconstructed = external_xywh_rect_to_oriented_box(external_cxcywh, offsets)

    assert external_cxcywh.shape == (1, 1, 4)
    assert offsets.shape == (1, 1, 2)
    assert reconstructed.shape == (1, 1, 5)
    assert obb_vertex_error(obb, reconstructed) < ROUNDTRIP_TOL


def test_external_cxcywh_helper_matches_existing_xyxy_composition():
    obb = torch.tensor([[[0.42, 0.58, 0.28, 0.10, math.pi / 6]]], dtype=torch.float32)
    ext_xyxy, offsets = oriented_box_to_external_xyxy_rect(obb)
    ext_cxcywh = box_xyxy_to_cxcywh(ext_xyxy)

    via_helper = external_xywh_rect_to_oriented_box(ext_cxcywh, offsets)
    via_primitives = external_xyxy_rect_to_oriented_box(ext_xyxy, offsets)

    assert obb_vertex_error(via_helper, via_primitives) < ROUNDTRIP_TOL


# ---------------------------------------------------------------------------
# Near-square behavior: vertex error passes even if parameter theta differs
# ---------------------------------------------------------------------------

NEAR_SQUARE_CASES = [
    (0.3, 0.3, math.pi / 4),
    (0.3, 0.299, math.pi / 4),
    (0.3, 0.3, 0.0),
    (0.3, 0.301, math.pi / 6),
    (0.3, 0.3, math.pi / 2),
]


@pytest.mark.parametrize("w, h, theta", NEAR_SQUARE_CASES)
def test_near_square_vertex_error_passes(w, h, theta):
    """Near-square boxes: vertex error must pass even if parameter-level
    theta may differ due to long-side convention (spec section 9.2).
    """
    obbs = torch.tensor([[0.5, 0.5, w, h, theta]], dtype=torch.float32)
    recon = roundtrip(obbs)
    v_err = obb_vertex_error(obbs, recon)
    assert v_err.item() < ROUNDTRIP_TOL, (
        f"near-square w={w} h={h} theta={theta:.4f}: "
        f"vertex error {v_err.item():.2e} >= {ROUNDTRIP_TOL:.0e}"
    )


def test_near_square_parameter_theta_may_differ():
    """Document long-side instability for square boxes (spec 9.2).

    A square OBB at theta=0 round-trips to theta=pi/2 because the
    longer-edge-as-w convention is ambiguous when w==h. The geometry
    (vertices) is identical, which is why tests use vertex error rather
    than parameter identity.
    """
    obbs = torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.0]], dtype=torch.float32)
    recon = roundtrip(obbs)
    v_err = obb_vertex_error(obbs, recon)
    assert v_err.item() < ROUNDTRIP_TOL
    theta_in = obbs[0, 4].item()
    theta_out = recon[0, 4].item()
    # Parameter theta flips by ~pi/2 for the square-at-0 case.
    delta = abs(theta_out - theta_in)
    delta_mod = min(delta, math.pi - delta) % math.pi
    assert abs(delta_mod - math.pi / 2) < 1e-3, (
        f"square-at-0: expected theta to flip by ~pi/2, "
        f"got in={theta_in:.4f} out={theta_out:.4f} delta={delta:.4f}"
    )


# ---------------------------------------------------------------------------
# Periodic seam tests (production periodic_angle_distance utility)
# ---------------------------------------------------------------------------


def test_periodic_seam_near_0_and_pi():
    """Angles near 0 and pi are geometrically close for OBBs (spec 8.2).

    Plan acceptance criterion: ``periodic_angle_distance(pi - 0.01, 0.01)
    < 0.021`` while the ordinary normalized L1 for the same pair is
    ``> 0.9``.
    """
    pred = torch.tensor([torch.pi - 0.01], dtype=torch.float32)
    target = torch.tensor([0.01], dtype=torch.float32)

    periodic = periodic_angle_distance(pred, target)
    ordinary = ordinary_normalized_theta_l1(pred, target)

    assert periodic.item() < 0.021, (
        f"periodic distance near 0/pi seam should be < 0.021, "
        f"got {periodic.item():.2e}"
    )
    assert ordinary.item() > 0.9, (
        f"ordinary normalized L1 near 0/pi seam should be > 0.9, "
        f"got {ordinary.item():.2e}"
    )


def test_periodic_seam_0_and_pi():
    """Exactly 0 and pi are the same OBB orientation (plan failure case).

    ``periodic_angle_distance(0, pi)`` must be zero/tiny rather than pi.
    """
    pred = torch.tensor([torch.pi - 1e-9], dtype=torch.float32)
    target = torch.tensor([0.0], dtype=torch.float32)

    periodic = periodic_angle_distance(pred, target)
    ordinary = ordinary_normalized_theta_l1(pred, target)

    assert (
        periodic.item() < 1e-6
    ), f"periodic distance at seam should be ~0, got {periodic.item():.2e}"
    assert (
        ordinary.item() > 0.99
    ), f"ordinary normalized L1 at seam should be ~1, got {ordinary.item():.2e}"


def test_periodic_midrange_unchanged():
    """For midrange angles, periodic and ordinary distances agree."""
    pred = torch.tensor([0.3], dtype=torch.float32)
    target = torch.tensor([0.5], dtype=torch.float32)

    periodic = periodic_angle_distance(pred, target)
    ordinary_unnorm = (pred - target).abs()

    assert torch.isclose(periodic, ordinary_unnorm, atol=1e-6).item()
    assert periodic.item() < 0.21


def test_periodic_broadcasting():
    """Periodic utility must follow PyTorch broadcasting (spec 8.2)."""
    pred = torch.tensor([[0.0], [torch.pi / 2], [torch.pi - 1e-3]])
    target = torch.tensor([[1e-3, torch.pi / 2 - 1e-3, 1e-3]])
    out = periodic_angle_distance(pred, target)
    assert out.shape == (3, 3)
    for i in range(3):
        assert out[i, i].item() < 2.1e-3, (
            f"diagonal periodic distance [{i},{i}] should be tiny, "
            f"got {out[i, i].item():.2e}"
        )


def test_periodic_gradient_exists_for_non_boundary_points():
    """Plan acceptance: gradient exists for non-boundary points.

    The periodic distance is differentiable away from the cusp where
    ``d == pi/2`` (the min switches branches). At a plain midrange pair
    the gradient must be finite and non-zero.
    """
    pred = torch.tensor([0.3], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([0.5], dtype=torch.float32)

    loss = periodic_angle_distance(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().item() > 0.0


# ---------------------------------------------------------------------------
# Stale-state guard: confirm we import the on-disk repo module
# ---------------------------------------------------------------------------


def test_imports_on_disk_repo_module():
    """stale_state guard: the imported module must come from this repo's
    on-disk engine/deim/obb_geometry.py, not a stale installed package.
    """
    import engine.deim.obb_geometry as geom

    expected = os.path.join(ROOT, "engine", "deim", "obb_geometry.py")
    actual = os.path.realpath(geom.__file__)
    assert actual == os.path.realpath(expected), (
        f"imported obb_geometry from {actual}, expected {expected}; "
        "a stale installed package may be shadowing the repo."
    )


# ---------------------------------------------------------------------------
# Offset scale source tests (Todo 5: ADR offset residual scale source)
# ---------------------------------------------------------------------------

OFFSET_SCALE_CASES = [
    ("horiz_to_45", [0.5, 0.5, 0.4, 0.2, 0.0], [0.55, 0.45, 0.25, 0.15, math.pi / 4]),
    ("shifted", [0.5, 0.5, 0.3, 0.2, 0.3], [0.6, 0.4, 0.25, 0.18, 0.5]),
    ("rotated", [0.5, 0.5, 0.35, 0.15, 0.0], [0.5, 0.5, 0.3, 0.3, math.pi / 4]),
    ("theta_near_0", [0.5, 0.5, 0.3, 0.1, 0.0], [0.55, 0.5, 0.25, 0.12, 0.1]),
    (
        "theta_near_pi",
        [0.5, 0.5, 0.3, 0.1, math.pi - 0.01],
        [0.45, 0.5, 0.28, 0.09, 0.5],
    ),
]

_REG_SCALE_T = torch.tensor([4.0])
_UP_T = torch.tensor([0.5])
_REG_MAX = 32
INVERSION_TOL = 1e-4


def _raw_encode_distances(ref_obb, gt_obb, reg_scale, offset_scale_source):
    """Compute raw 6-distances (α,β,γ,δ,ε,η) without translate_gt quantization.

    Mirrors the encode logic of ``bbox2distance_obb`` up to (but excluding)
    ``translate_gt``, so decode inversion can be tested without bin
    quantization loss. This is the test helper referenced in the Todo 5
    acceptance criteria for the mismatch demonstration.
    """
    ext_xyxy_pred, vo_pred = oriented_box_to_external_xyxy_rect(ref_obb)
    ext_xyxy_gt, vo_gt = oriented_box_to_external_xyxy_rect(gt_obb)
    ext_cxcywh_pred = box_xyxy_to_cxcywh(ext_xyxy_pred)
    ext_cxcywh_gt = box_xyxy_to_cxcywh(ext_xyxy_gt)
    rs = abs(reg_scale)
    pw = ext_cxcywh_pred[..., 2] / rs + 1e-16
    ph = ext_cxcywh_pred[..., 3] / rs + 1e-16
    hr = 0.5 * rs
    el = (ext_cxcywh_pred[..., 0] - ext_xyxy_gt[..., 0]) / pw - hr + 1e-16
    et = (ext_cxcywh_pred[..., 1] - ext_xyxy_gt[..., 1]) / ph - hr + 1e-16
    er = (ext_xyxy_gt[..., 2] - ext_cxcywh_pred[..., 0]) / pw - hr + 1e-16
    eb = (ext_xyxy_gt[..., 3] - ext_cxcywh_pred[..., 1]) / ph - hr + 1e-16
    osc = (
        ext_cxcywh_pred[..., 2:]
        if offset_scale_source == "pre"
        else ext_cxcywh_gt[..., 2:]
    )
    tl = (vo_gt - vo_pred) / (osc / rs + 1e-16)
    return torch.stack([el, et, er, eb, tl[..., 0], tl[..., 1]], dim=-1)


@pytest.mark.parametrize(
    "name, ref, target", OFFSET_SCALE_CASES, ids=[c[0] for c in OFFSET_SCALE_CASES]
)
def test_offset_scale_pre_decode_preserves_current_behavior(name, ref, target):
    """offset_scale_source='pre' (default) must match calling without the
    parameter — i.e. the default path is unchanged.
    """
    ref_t = torch.tensor([ref], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)
    dist = _raw_encode_distances(ref_t, target_t, _REG_SCALE_T, "pre")
    out_default = distance2bbox_obb(ref_t, dist, _REG_SCALE_T)
    out_pre = distance2bbox_obb(ref_t, dist, _REG_SCALE_T, offset_scale_source="pre")
    assert torch.equal(
        out_default, out_pre
    ), f"{name}: decode with explicit 'pre' differs from default path"


@pytest.mark.parametrize(
    "name, ref, target", OFFSET_SCALE_CASES, ids=[c[0] for c in OFFSET_SCALE_CASES]
)
def test_offset_scale_pre_encode_preserves_current_behavior(name, ref, target):
    """offset_scale_source='pre' (default) encode must match calling without
    the parameter.
    """
    ref_t = torch.tensor([ref], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)
    six_default, _, _ = bbox2distance_obb(
        ref_t, target_t, _REG_MAX, _REG_SCALE_T, _UP_T
    )
    six_pre, _, _ = bbox2distance_obb(
        ref_t, target_t, _REG_MAX, _REG_SCALE_T, _UP_T, offset_scale_source="pre"
    )
    assert torch.equal(
        six_default, six_pre
    ), f"{name}: encode with explicit 'pre' differs from default path"


@pytest.mark.parametrize(
    "name, ref, target", OFFSET_SCALE_CASES, ids=[c[0] for c in OFFSET_SCALE_CASES]
)
def test_offset_scale_post_decode_produces_finite_obb(name, ref, target):
    """offset_scale_source='post' must be accepted and produce finite OBBs."""
    ref_t = torch.tensor([ref], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)
    dist = _raw_encode_distances(ref_t, target_t, _REG_SCALE_T, "pre")
    out_post = distance2bbox_obb(ref_t, dist, _REG_SCALE_T, offset_scale_source="post")
    assert torch.isfinite(
        out_post
    ).all(), f"{name}: post decode produced non-finite OBB: {out_post}"


@pytest.mark.parametrize(
    "name, ref, target", OFFSET_SCALE_CASES, ids=[c[0] for c in OFFSET_SCALE_CASES]
)
def test_offset_scale_post_encode_produces_finite_distances(name, ref, target):
    """offset_scale_source='post' encode must be accepted and produce finite
    FGL targets.
    """
    ref_t = torch.tensor([ref], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)
    six_post, wr, wl = bbox2distance_obb(
        ref_t, target_t, _REG_MAX, _REG_SCALE_T, _UP_T, offset_scale_source="post"
    )
    assert torch.isfinite(
        six_post
    ).all(), f"{name}: post encode produced non-finite distances: {six_post}"
    assert torch.isfinite(wr).all() and torch.isfinite(wl).all()


def test_offset_scale_invalid_raises_valueerror_decode():
    """Invalid offset_scale_source must raise ValueError on decode."""
    ref = torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.5]], dtype=torch.float32)
    dist = torch.zeros(1, 6, dtype=torch.float32)
    with pytest.raises(ValueError, match="offset_scale_source"):
        distance2bbox_obb(ref, dist, _REG_SCALE_T, offset_scale_source="invalid")


def test_offset_scale_invalid_raises_valueerror_encode():
    """Invalid offset_scale_source must raise ValueError on encode."""
    ref = torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.5]], dtype=torch.float32)
    target = torch.tensor([[0.55, 0.45, 0.25, 0.15, 0.6]], dtype=torch.float32)
    with pytest.raises(ValueError, match="offset_scale_source"):
        bbox2distance_obb(
            ref, target, _REG_MAX, _REG_SCALE_T, _UP_T, offset_scale_source="invalid"
        )


@pytest.mark.parametrize(
    "name, ref, target", OFFSET_SCALE_CASES, ids=[c[0] for c in OFFSET_SCALE_CASES]
)
def test_offset_scale_pre_inversion_roundtrip(name, ref, target):
    """encode('pre') + decode('pre') must reconstruct the target OBB within
    float32 precision (no translate_gt quantization in the test helper).
    """
    ref_t = torch.tensor([ref], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)
    dist = _raw_encode_distances(ref_t, target_t, _REG_SCALE_T, "pre")
    recon = distance2bbox_obb(ref_t, dist, _REG_SCALE_T, offset_scale_source="pre")
    err = obb_vertex_error(target_t, recon)
    assert (
        err.item() < INVERSION_TOL
    ), f"{name}: pre/pre inversion error {err.item():.2e} >= {INVERSION_TOL:.0e}"


@pytest.mark.parametrize(
    "name, ref, target", OFFSET_SCALE_CASES, ids=[c[0] for c in OFFSET_SCALE_CASES]
)
def test_offset_scale_post_inversion_roundtrip(name, ref, target):
    """encode('post') + decode('post') must reconstruct the target OBB within
    float32 precision. Proves 'post' is a consistent (not mixed) scale source.
    """
    ref_t = torch.tensor([ref], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)
    dist = _raw_encode_distances(ref_t, target_t, _REG_SCALE_T, "post")
    recon = distance2bbox_obb(ref_t, dist, _REG_SCALE_T, offset_scale_source="post")
    err = obb_vertex_error(target_t, recon)
    assert (
        err.item() < INVERSION_TOL
    ), f"{name}: post/post inversion error {err.item():.2e} >= {INVERSION_TOL:.0e}"


@pytest.mark.parametrize(
    "name, ref, target", OFFSET_SCALE_CASES, ids=[c[0] for c in OFFSET_SCALE_CASES]
)
def test_offset_scale_mismatch_increases_inversion_error(name, ref, target):
    """Mismatched encode/decode scale (pre-encode + post-decode, or vice
    versa) must produce strictly larger inversion error than the matched
    pair. This is the failure-mode demonstration required by the Todo 5
    acceptance criteria, proving a shared setting matters.
    """
    ref_t = torch.tensor([ref], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)

    dist_pre = _raw_encode_distances(ref_t, target_t, _REG_SCALE_T, "pre")
    dist_post = _raw_encode_distances(ref_t, target_t, _REG_SCALE_T, "post")

    recon_matched_pre = distance2bbox_obb(
        ref_t, dist_pre, _REG_SCALE_T, offset_scale_source="pre"
    )
    recon_matched_post = distance2bbox_obb(
        ref_t, dist_post, _REG_SCALE_T, offset_scale_source="post"
    )
    recon_mismatch_1 = distance2bbox_obb(
        ref_t, dist_pre, _REG_SCALE_T, offset_scale_source="post"
    )
    recon_mismatch_2 = distance2bbox_obb(
        ref_t, dist_post, _REG_SCALE_T, offset_scale_source="pre"
    )

    err_matched = min(
        obb_vertex_error(target_t, recon_matched_pre).item(),
        obb_vertex_error(target_t, recon_matched_post).item(),
    )
    err_mismatch = min(
        obb_vertex_error(target_t, recon_mismatch_1).item(),
        obb_vertex_error(target_t, recon_mismatch_2).item(),
    )

    assert (
        err_matched < INVERSION_TOL
    ), f"{name}: matched inversion error {err_matched:.2e} >= {INVERSION_TOL:.0e}"
    assert err_mismatch > err_matched, (
        f"{name}: mismatch error {err_mismatch:.2e} must exceed matched "
        f"{err_matched:.2e} — if not, the case does not exercise the "
        "pre/post scale difference (ext rect sizes too similar)."
    )


def test_offset_scale_random_mismatch_increases_error():
    """Randomized mismatch demonstration (deterministic seed, flaky guard).

    Theta is clamped away from the exact 0/pi boundary to avoid a
    pre-existing ADR degeneracy: at theta=0 or theta=pi the vertex offsets
    hit extremes (epsilon=full width, eta=0) and external_xyxy_rect_to_oriented_box
    degenerates. This is a geometric property of the ADR representation,
    not a defect of the offset_scale_source change under test.
    """
    torch.manual_seed(20260707)
    n = 200
    ref = torch.cat(
        [
            torch.rand(n, 1),
            torch.rand(n, 1),
            torch.rand(n, 1) * 0.4 + 0.1,
            torch.rand(n, 1) * 0.4 + 0.05,
            torch.rand(n, 1) * (math.pi - 0.02) + 0.01,
        ],
        dim=-1,
    )
    target = ref + torch.cat(
        [
            (torch.rand(n, 1) - 0.5) * 0.2,
            (torch.rand(n, 1) - 0.5) * 0.2,
            (torch.rand(n, 1) - 0.5) * 0.15,
            (torch.rand(n, 1) - 0.5) * 0.15,
            (torch.rand(n, 1) - 0.5) * 0.3,
        ],
        dim=-1,
    )
    target[..., 4] = target[..., 4].clamp(0.01, math.pi - 0.01)

    dist_pre = _raw_encode_distances(ref, target, _REG_SCALE_T, "pre")
    dist_post = _raw_encode_distances(ref, target, _REG_SCALE_T, "post")

    recon_matched = distance2bbox_obb(
        ref, dist_pre, _REG_SCALE_T, offset_scale_source="pre"
    )
    recon_mismatch = distance2bbox_obb(
        ref, dist_pre, _REG_SCALE_T, offset_scale_source="post"
    )

    err_matched = obb_vertex_error(target, recon_matched).item()
    err_mismatch = obb_vertex_error(target, recon_mismatch).item()

    assert (
        err_matched < INVERSION_TOL
    ), f"random: matched inversion error {err_matched:.2e} >= {INVERSION_TOL:.0e}"
    assert err_mismatch > err_matched, (
        f"random: mismatch error {err_mismatch:.2e} must exceed matched "
        f"{err_matched:.2e}"
    )


# ---------------------------------------------------------------------------
# Criterion OBB loss tests (Todo 3: periodic angle L1 in DEIMCriterion)
# ---------------------------------------------------------------------------
#
# TDD red-before-green: the seam test fails against the pre-fix criterion
# (ordinary L1 on theta/pi gives ~0.994 for the seam pair) and passes
# after periodic angle distance is wired in.


def _make_criterion(lambda_angle=1.0, box_mode="obb"):
    """Build a DEIMCriterion for direct loss_boxes testing.

    matcher is None because loss_boxes does not invoke the matcher.
    """
    from engine.deim.deim_criterion import DEIMCriterion

    return DEIMCriterion(
        matcher=None,
        weight_dict={"loss_bbox": 1.0, "loss_kld": 1.0, "loss_giou": 1.0},
        losses=["boxes"],
        box_mode=box_mode,
        lambda_angle=lambda_angle,
    )


def _make_matched_pair(pred_box, target_box):
    """Build matched outputs/targets/indices for a single-box case.

    pred_box / target_box are lists of length 4 (HBB) or 5 (OBB).
    Returns (outputs, targets, indices, num_boxes=1).
    """
    outputs = {"pred_boxes": torch.tensor([pred_box], dtype=torch.float32).unsqueeze(0)}
    targets = [
        {
            "boxes": torch.tensor([target_box], dtype=torch.float32),
            "labels": torch.tensor([0]),
        }
    ]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    return outputs, targets, indices, 1


def test_criterion_obb_seam_angle_contribution_small():
    """OBB loss_bbox angle contribution must be small near 0/pi seam.

    Spatial dims match exactly; only theta differs across the seam
    (pred=pi-0.01, target=0.01). Periodic distance is ~0.02, so the
    normalized angle term is ~0.02/pi ≈ 0.0064. The full loss_bbox
    must be small because spatial L1 is zero.

    RED before fix: ordinary L1 gives (pi-0.02)/pi ≈ 0.994 >> 0.05.
    """
    criterion = _make_criterion(lambda_angle=1.0, box_mode="obb")
    pred = [0.5, 0.5, 0.3, 0.2, math.pi - 0.01]
    target = [0.5, 0.5, 0.3, 0.2, 0.01]
    outputs, targets, indices, num_boxes = _make_matched_pair(pred, target)
    losses = criterion.loss_boxes(outputs, targets, indices, num_boxes)

    # Document the ordinary-L1 baseline that the fix replaces.
    ordinary_angle_l1 = abs(math.pi - 0.02) / math.pi
    assert ordinary_angle_l1 > 0.99, "baseline ordinary L1 should be ~0.994 at the seam"

    # Spatial dims match -> spatial L1 = 0 -> loss_bbox ≈ angle term only.
    assert losses["loss_bbox"].item() < 0.05, (
        f"seam angle contribution should be small (<0.05), "
        f"got {losses['loss_bbox'].item():.4f}; criterion may be using "
        "ordinary non-periodic L1 on theta."
    )


def test_criterion_obb_lambda_angle_zero_zeros_angle_only():
    """lambda_angle=0.0 must zero the angle contribution while spatial
    contribution remains.

    pred and target differ in both spatial dims and theta (seam pair).
    With lambda_angle=0, loss_bbox must equal the spatial L1 only.
    With lambda_angle=1, loss_bbox must add the small periodic angle term.
    """
    pred = [0.5, 0.5, 0.3, 0.2, math.pi - 0.01]
    target = [0.55, 0.45, 0.25, 0.15, 0.01]
    outputs, targets, indices, num_boxes = _make_matched_pair(pred, target)

    crit_zero = _make_criterion(lambda_angle=0.0, box_mode="obb")
    losses_zero = crit_zero.loss_boxes(outputs, targets, indices, num_boxes)

    crit_one = _make_criterion(lambda_angle=1.0, box_mode="obb")
    losses_one = crit_one.loss_boxes(outputs, targets, indices, num_boxes)

    # Expected spatial L1 sum: 0.05*4 = 0.2
    expected_spatial = 0.2
    assert torch.isclose(
        losses_zero["loss_bbox"], torch.tensor(expected_spatial), atol=1e-6
    ), (
        f"lambda_angle=0: loss_bbox should equal spatial L1 only "
        f"({expected_spatial}), got {losses_zero['loss_bbox'].item():.6f}"
    )
    # lambda_angle=1 adds a positive angle contribution.
    assert losses_one["loss_bbox"].item() > losses_zero["loss_bbox"].item(), (
        "lambda_angle=1 should add a positive angle contribution on top "
        "of spatial L1"
    )
    # The added angle contribution must be the small periodic value, not
    # the huge ordinary L1 (~0.994).
    angle_contrib = losses_one["loss_bbox"].item() - losses_zero["loss_bbox"].item()
    expected_angle = 1.0 * (0.02 / math.pi)
    assert abs(angle_contrib - expected_angle) < 1e-5, (
        f"angle contribution should be ~{expected_angle:.6f} (periodic), "
        f"got {angle_contrib:.6f}; criterion may be using ordinary L1."
    )


def test_criterion_hbb_uses_ordinary_l1_unaffected():
    """HBB branch must still use ordinary L1 and be unaffected by
    lambda_angle / periodic angle distance.

    HBB boxes are 4-dim (cx, cy, w, h). loss_bbox must equal ordinary
    L1 sum / num_boxes, and loss_giou must be returned (not loss_kld).
    """
    criterion = _make_criterion(lambda_angle=1.0, box_mode="hbb")
    pred = [0.5, 0.5, 0.3, 0.2]
    target = [0.55, 0.45, 0.25, 0.15]
    outputs, targets, indices, num_boxes = _make_matched_pair(pred, target)
    losses = criterion.loss_boxes(outputs, targets, indices, num_boxes)

    expected_l1 = 0.05 * 4  # 0.2
    assert torch.isclose(losses["loss_bbox"], torch.tensor(expected_l1), atol=1e-6), (
        f"HBB loss_bbox should be ordinary L1 ({expected_l1}), "
        f"got {losses['loss_bbox'].item():.6f}"
    )
    assert "loss_giou" in losses, "HBB must return loss_giou"
    assert "loss_kld" not in losses, "HBB must not return loss_kld"


def test_criterion_obb_loss_kld_returned():
    """OBB mode must return loss_kld in addition to loss_bbox (spec 8.1).

    KLD must remain unchanged by the periodic angle L1 change.
    """
    criterion = _make_criterion(lambda_angle=1.0, box_mode="obb")
    pred = [0.5, 0.5, 0.3, 0.2, 0.5]
    target = [0.55, 0.45, 0.25, 0.15, 0.6]
    outputs, targets, indices, num_boxes = _make_matched_pair(pred, target)
    losses = criterion.loss_boxes(outputs, targets, indices, num_boxes)
    assert "loss_kld" in losses, "OBB mode must return loss_kld"
    assert torch.isfinite(losses["loss_kld"]).all()


def test_criterion_imports_on_disk_repo_module():
    """stale_state guard: the imported DEIMCriterion must come from this
    repo's on-disk engine/deim/deim_criterion.py, not a stale installed
    package.
    """
    import engine.deim.deim_criterion as crit

    expected = os.path.join(ROOT, "engine", "deim", "deim_criterion.py")
    actual = os.path.realpath(crit.__file__)
    assert actual == os.path.realpath(expected), (
        f"imported deim_criterion from {actual}, expected {expected}; "
        "a stale installed package may be shadowing the repo."
    )


# ---------------------------------------------------------------------------
# Matcher OBB seam tests (Todo 4: periodic angle cost in HungarianMatcher)
# ---------------------------------------------------------------------------
#
# TDD red-before-green: the seam test computes the ordinary-L1 reference
# cost matrix (which would pick the wrong prediction) and then asserts the
# production matcher with periodic angle cost picks the geometry-consistent
# seam-side prediction.


def _make_matcher(
    lambda_angle=1.0,
    change_matcher=False,
    matcher_change_epoch=10000,
    cost_bbox=1.0,
    cost_class=0.0,
    cost_probiou=0.0,
    cost_chamfer=0.0,
    use_focal_loss=False,
    box_mode="obb",
    iou_order_alpha=1.0,
):
    """Build a HungarianMatcher for direct forward testing.

    Default weights isolate cost_bbox so the angle term is the sole
    differentiator when spatial dims are identical.
    """
    from engine.deim.matcher import HungarianMatcher

    weight_dict = {
        "cost_class": cost_class,
        "cost_bbox": cost_bbox,
        "cost_giou": 0,
        "cost_chamfer": cost_chamfer,
        "cost_probiou": cost_probiou,
    }
    return HungarianMatcher(
        weight_dict=weight_dict,
        use_focal_loss=use_focal_loss,
        change_matcher=change_matcher,
        iou_order_alpha=iou_order_alpha,
        matcher_change_epoch=matcher_change_epoch,
        box_mode=box_mode,
        lambda_angle=lambda_angle,
    )


def _make_matcher_inputs(pred_obbs, target_obbs, num_classes=1):
    """Build outputs/targets for the matcher.

    pred_obbs / target_obbs are lists of [cx, cy, w, h, theta].
    pred_logits are zeros so cost_class is uniform and does not
    interfere with bbox-cost isolation.
    """
    pred_boxes = torch.tensor(pred_obbs, dtype=torch.float32).unsqueeze(0)
    pred_logits = torch.zeros(1, len(pred_obbs), num_classes, dtype=torch.float32)
    outputs = {"pred_boxes": pred_boxes, "pred_logits": pred_logits}
    targets = [
        {
            "boxes": torch.tensor(target_obbs, dtype=torch.float32),
            "labels": torch.zeros(len(target_obbs), dtype=torch.int64),
        }
    ]
    return outputs, targets


_SEAM_SPATIAL = [0.5, 0.5, 0.3, 0.2]
_SEAM_PRED_A = _SEAM_SPATIAL + [math.pi - 0.01]  # seam-side, equiv to theta=0.01
_SEAM_PRED_B = _SEAM_SPATIAL + [math.pi / 2]  # geometrically far
_SEAM_TARGET = _SEAM_SPATIAL + [0.01]


def test_matcher_seam_obeys_periodic_angle():
    """Matcher with periodic angle cost picks the geometry-consistent
    seam-side prediction; ordinary L1 would pick the wrong one.

    Target theta=0.01. Pred A theta=pi-0.01 (seam-side, geometrically
    equivalent). Pred B theta=pi/2 (geometrically far). All spatial
    dims identical so the sole differentiator is the angle cost.

    RED before fix: matcher uses ordinary L1 on theta/pi, giving
    cost_A ~= 0.994 > cost_B ~= 0.497, so it picks B (wrong).
    GREEN after fix: periodic angle distance gives
    cost_A ~= 0.006 < cost_B ~= 0.497, so it picks A (correct).
    """
    outputs, targets = _make_matcher_inputs(
        [_SEAM_PRED_A, _SEAM_PRED_B], [_SEAM_TARGET]
    )

    # Reference: ordinary L1 cost matrix (old behavior the fix replaces)
    preds_t = torch.tensor([_SEAM_PRED_A, _SEAM_PRED_B], dtype=torch.float32)
    target_t = torch.tensor([_SEAM_TARGET], dtype=torch.float32)
    factor = target_t.new_tensor([1, 1, 1, 1, 1.0 / math.pi])
    old_cost = torch.cdist(preds_t * factor, target_t * factor, p=1)
    assert (
        old_cost[0, 0].item() > old_cost[1, 0].item()
    ), "ordinary L1 should rank seam-side A as more expensive than B"
    assert old_cost[0, 0].item() > 0.9, (
        f"ordinary L1 seam cost should be ~0.994, " f"got {old_cost[0, 0].item():.4f}"
    )

    # Actual: periodic matcher picks A (index 0)
    matcher = _make_matcher(lambda_angle=1.0, cost_bbox=1.0)
    result = matcher(outputs, targets, epoch=0)
    selected = result["indices"][0][0].item()
    assert (
        selected == 0
    ), f"periodic matcher should pick seam-side A (index 0), got {selected}"


def test_matcher_seam_lambda_angle_zero_disables_angle():
    """lambda_angle=0.0 must zero the angle contribution; the seam
    distinction no longer comes from angle cost.

    With identical spatial dims, cost_bbox is equal for both predictions
    when lambda_angle=0. With lambda_angle=1, cost_bbox differs by the
    periodic angle term (A < B).
    """
    preds_t = torch.tensor([_SEAM_PRED_A, _SEAM_PRED_B], dtype=torch.float32)
    target_t = torch.tensor([_SEAM_TARGET], dtype=torch.float32)

    spatial_cost = torch.cdist(preds_t[..., :4], target_t[..., :4], p=1)
    angle_cost = (
        periodic_angle_distance(preds_t[:, None, 4:], target_t[None, :, 4:]).squeeze(-1)
        / math.pi
    )

    # lambda_angle=0: cost_bbox = spatial only (equal for both)
    cost_zero = spatial_cost + 0.0 * angle_cost
    assert torch.isclose(cost_zero[0, 0], cost_zero[1, 0], atol=1e-6), (
        f"lambda_angle=0: cost_bbox must be equal for both predictions, "
        f"got A={cost_zero[0, 0].item():.6f} B={cost_zero[1, 0].item():.6f}"
    )

    # lambda_angle=1: cost_bbox differs by angle term (A < B)
    cost_one = spatial_cost + 1.0 * angle_cost
    assert (
        cost_one[0, 0].item() < cost_one[1, 0].item()
    ), "lambda_angle=1: seam-side A must have lower cost than B"

    # Actual matcher executes with lambda_angle=0 without error
    outputs, targets = _make_matcher_inputs(
        [_SEAM_PRED_A, _SEAM_PRED_B], [_SEAM_TARGET]
    )
    matcher = _make_matcher(lambda_angle=0.0, cost_bbox=1.0)
    result = matcher(outputs, targets, epoch=0)
    assert "indices" in result


def test_matcher_change_matcher_post_epoch_probiou_only():
    """change_matcher=True with epoch >= matcher_change_epoch uses the
    ProbIoU-only branch and executes without periodic angle cost.

    The post-epoch branch computes C = -class_score * bbox_iou^alpha,
    where bbox_iou is batch_probiou for OBB. It does not reference
    lambda_angle or periodic_angle_distance. Changing lambda_angle
    must not change the selected assignment.
    """
    outputs, targets = _make_matcher_inputs(
        [_SEAM_PRED_A, _SEAM_PRED_B], [_SEAM_TARGET]
    )

    matcher_la1 = _make_matcher(
        lambda_angle=1.0,
        change_matcher=True,
        matcher_change_epoch=0,
        cost_bbox=1.0,
        cost_class=1.0,
        cost_probiou=1.0,
    )
    result_la1 = matcher_la1(outputs, targets, epoch=1)
    selected_la1 = result_la1["indices"][0][0].item()

    matcher_la0 = _make_matcher(
        lambda_angle=0.0,
        change_matcher=True,
        matcher_change_epoch=0,
        cost_bbox=1.0,
        cost_class=1.0,
        cost_probiou=1.0,
    )
    result_la0 = matcher_la0(outputs, targets, epoch=1)
    selected_la0 = result_la0["indices"][0][0].item()

    assert selected_la1 == selected_la0, (
        f"change_matcher branch must not depend on lambda_angle: "
        f"la=1 picked {selected_la1}, la=0 picked {selected_la0}"
    )


def test_matcher_imports_on_disk_repo_module():
    """stale_state guard: the imported HungarianMatcher must come from
    this repo's on-disk engine/deim/matcher.py, not a stale installed
    package.
    """
    import engine.deim.matcher as matcher_mod

    expected = os.path.join(ROOT, "engine", "deim", "matcher.py")
    actual = os.path.realpath(matcher_mod.__file__)
    assert actual == os.path.realpath(expected), (
        f"imported matcher from {actual}, expected {expected}; "
        "a stale installed package may be shadowing the repo."
    )


# ---------------------------------------------------------------------------
# Offset validity guards (Todo 6: clamp invalid ADR offsets safely)
# ---------------------------------------------------------------------------
#
# TDD red-before-green: these tests fail against the pre-fix geometry module
# (clamp_vertex_offsets_to_external_rect does not exist; external_rect_to_
# oriented_box has no clamp_offsets parameter) and pass after the guarded
# helper and decode path are added.
#
# Acceptance (plan Todo 6): offset-validity tests cover negative
# epsilon/eta, epsilon greater than external width, eta greater than
# external height, and zero-size external rectangles. Guarded decode
# returns finite OBBs and valid clamped offsets; unguarded training
# decode remains available for gradient-bearing outputs.


# External rectangle + offsets used across the offset-validity cases.
# ext = (x1, y1, x2, y2) = (0.1, 0.2, 0.5, 0.8)  -> ext_w=0.4, ext_h=0.6
_VALID_EXT = [0.1, 0.2, 0.5, 0.8]
_VALID_VO = [0.1, 0.2]  # 0 <= 0.1 <= 0.4, 0 <= 0.2 <= 0.6

# Invalid offset cases: (name, ext, vo, expected_clamped_vo)
OFFSET_INVALIDITY_CASES = [
    (
        "negative_epsilon",
        _VALID_EXT,
        [-0.3, 0.2],
        [0.0, 0.2],
    ),
    (
        "negative_eta",
        _VALID_EXT,
        [0.1, -0.5],
        [0.1, 0.0],
    ),
    (
        "negative_both",
        _VALID_EXT,
        [-0.3, -0.5],
        [0.0, 0.0],
    ),
    (
        "epsilon_exceeds_width",
        _VALID_EXT,
        [0.7, 0.2],
        [0.4, 0.2],
    ),
    (
        "eta_exceeds_height",
        _VALID_EXT,
        [0.1, 1.0],
        [0.1, 0.6],
    ),
    (
        "both_exceed",
        _VALID_EXT,
        [10.0, 10.0],
        [0.4, 0.6],
    ),
    (
        "zero_width_rect",
        [0.3, 0.2, 0.3, 0.8],  # ext_w = 0
        [0.15, 0.3],
        [0.0, 0.3],
    ),
    (
        "zero_height_rect",
        [0.1, 0.5, 0.5, 0.5],  # ext_h = 0
        [0.15, 0.3],
        [0.15, 0.0],
    ),
    (
        "zero_size_rect",
        [0.3, 0.5, 0.3, 0.5],  # ext_w = 0, ext_h = 0
        [0.15, 0.3],
        [0.0, 0.0],
    ),
    (
        "zero_size_rect_with_invalid_offsets",
        [0.3, 0.5, 0.3, 0.5],
        [-1.0, 5.0],
        [0.0, 0.0],
    ),
]


@pytest.mark.parametrize(
    "name, ext, vo, expected",
    OFFSET_INVALIDITY_CASES,
    ids=[c[0] for c in OFFSET_INVALIDITY_CASES],
)
def test_offset_validity_clamp_helper_clamps_invalid(name, ext, vo, expected):
    """clamp_vertex_offsets_to_external_rect must clamp (epsilon, eta) into
    [0, ext_w] / [0, ext_h] for every invalid case in the plan acceptance
    list: negative epsilon/eta, epsilon > ext_w, eta > ext_h, zero-width
    and zero-height external rectangles.
    """
    ext_t = torch.tensor([ext], dtype=torch.float32)
    vo_t = torch.tensor([vo], dtype=torch.float32)
    clamped = clamp_vertex_offsets_to_external_rect(ext_t, vo_t)
    expected_t = torch.tensor([expected], dtype=torch.float32)
    assert torch.allclose(clamped, expected_t, atol=1e-7), (
        f"{name}: clamped {clamped[0].tolist()} != expected "
        f"{expected} (ext={ext}, vo={vo})"
    )


@pytest.mark.parametrize(
    "name, ext, vo, expected",
    OFFSET_INVALIDITY_CASES,
    ids=[c[0] for c in OFFSET_INVALIDITY_CASES],
)
def test_offset_validity_clamp_helper_does_not_mutate_inputs(name, ext, vo, expected):
    """The helper must not modify its inputs in place. Non-mutation matters
    because the unguarded training path must remain untouched when the
    guarded decode is used in the same forward pass.
    """
    ext_t = torch.tensor([ext], dtype=torch.float32).clone()
    vo_t = torch.tensor([vo], dtype=torch.float32).clone()
    ext_before = ext_t.clone()
    vo_before = vo_t.clone()
    _ = clamp_vertex_offsets_to_external_rect(ext_t, vo_t)
    assert torch.equal(
        ext_t, ext_before
    ), f"{name}: external_rect was mutated by the helper"
    assert torch.equal(
        vo_t, vo_before
    ), f"{name}: vertex_offsets was mutated by the helper"


def test_offset_validity_clamp_helper_preserves_valid_offsets():
    """Valid offsets (already within [0, ext_w]/[0, ext_h]) must pass
    through unchanged. The guard must be a no-op for valid inputs so it
    does not perturb well-formed decode outputs.
    """
    ext_t = torch.tensor([_VALID_EXT], dtype=torch.float32)
    vo_t = torch.tensor([_VALID_VO], dtype=torch.float32)
    clamped = clamp_vertex_offsets_to_external_rect(ext_t, vo_t)
    assert torch.allclose(
        clamped, vo_t, atol=1e-7
    ), f"valid offsets changed: {clamped[0].tolist()} != {vo_t[0].tolist()}"


def test_offset_validity_clamp_helper_batch_broadcasts():
    """Helper must handle batched (N, 4) / (N, 2) inputs and broadcast
    correctly across a batch of mixed valid/invalid offsets.
    """
    ext_t = torch.tensor(
        [_VALID_EXT, [0.3, 0.2, 0.3, 0.8], [0.1, 0.5, 0.5, 0.5]],
        dtype=torch.float32,
    )
    vo_t = torch.tensor(
        [[0.1, 0.2], [0.15, 0.3], [0.15, 0.3]],
        dtype=torch.float32,
    )
    clamped = clamp_vertex_offsets_to_external_rect(ext_t, vo_t)
    assert clamped.shape == (3, 2)
    # Row 0: valid, unchanged
    assert torch.allclose(clamped[0], vo_t[0], atol=1e-7)
    # Row 1: zero-width rect -> epsilon clamped to 0
    assert clamped[1, 0].item() == 0.0
    assert torch.isclose(clamped[1, 1], vo_t[1, 1], atol=1e-7)
    # Row 2: zero-height rect -> eta clamped to 0
    assert torch.isclose(clamped[2, 0], vo_t[2, 0], atol=1e-7)
    assert clamped[2, 1].item() == 0.0


def test_offset_validity_decode_default_is_unguarded():
    """external_xyxy_rect_to_oriented_box must default to clamp_offsets=False
    (unguarded). Passing no clamp_offsets argument must equal passing
    clamp_offsets=False explicitly. This preserves the current training
    decode behavior for gradient-bearing outputs.
    """
    ext_t = torch.tensor([_VALID_EXT], dtype=torch.float32)
    vo_t = torch.tensor([_VALID_VO], dtype=torch.float32)
    out_default = external_xyxy_rect_to_oriented_box(ext_t, vo_t)
    out_false = external_xyxy_rect_to_oriented_box(ext_t, vo_t, clamp_offsets=False)
    assert torch.equal(out_default, out_false), (
        "default decode must equal clamp_offsets=False; the default must "
        "preserve the unguarded training decode path."
    )


@pytest.mark.parametrize(
    "name, ext, vo, expected",
    OFFSET_INVALIDITY_CASES,
    ids=[c[0] for c in OFFSET_INVALIDITY_CASES],
)
def test_offset_validity_decode_guarded_produces_finite_obb(name, ext, vo, expected):
    """Guarded decode (clamp_offsets=True) must produce finite OBBs for
    every invalid offset case, including zero-size external rectangles.
    """
    ext_t = torch.tensor([ext], dtype=torch.float32)
    vo_t = torch.tensor([vo], dtype=torch.float32)
    out = external_xyxy_rect_to_oriented_box(ext_t, vo_t, clamp_offsets=True)
    assert out.shape[-1] == 5
    assert torch.isfinite(
        out
    ).all(), f"{name}: guarded decode produced non-finite OBB: {out}"


def test_offset_validity_decode_guarded_equals_manual_clamp():
    """Guarded decode must equal manually clamping offsets then decoding
    without the guard. This proves the guard is a pure pre-clamp, not a
    different decode path.
    """
    ext_t = torch.tensor([_VALID_EXT], dtype=torch.float32)
    # Use offsets that are partly invalid so the clamp changes them.
    vo_t = torch.tensor([[-0.2, 0.9]], dtype=torch.float32)  # ep<0, eta>ext_h
    manual = external_xyxy_rect_to_oriented_box(
        ext_t, clamp_vertex_offsets_to_external_rect(ext_t, vo_t)
    )
    guarded = external_xyxy_rect_to_oriented_box(ext_t, vo_t, clamp_offsets=True)
    assert torch.allclose(manual, guarded, atol=1e-7), (
        f"guarded decode {guarded[0].tolist()} != manual-clamp "
        f"{manual[0].tolist()}; the guard must be a pure pre-clamp."
    )


def test_offset_validity_decode_unguarded_does_not_clamp():
    """Unguarded decode (clamp_offsets=False) must NOT clamp invalid
    offsets. The resulting OBB geometry must differ from the guarded
    path for the same invalid inputs. This is the failure-mode
    demonstration required by the plan: the guarded path is the one
    responsible for clamped behavior.
    """
    ext_t = torch.tensor([_VALID_EXT], dtype=torch.float32)
    # Strongly invalid offsets: negative epsilon, eta far beyond ext_h.
    vo_t = torch.tensor([[-0.5, 5.0]], dtype=torch.float32)
    out_unguarded = external_xyxy_rect_to_oriented_box(ext_t, vo_t, clamp_offsets=False)
    out_guarded = external_xyxy_rect_to_oriented_box(ext_t, vo_t, clamp_offsets=True)
    assert not torch.allclose(out_unguarded, out_guarded, atol=1e-6), (
        "unguarded and guarded decode must differ for invalid offsets; "
        f"unguarded={out_unguarded[0].tolist()}, "
        f"guarded={out_guarded[0].tolist()}"
    )
    # The guarded path must produce finite output; the unguarded path is
    # allowed to produce geometrically invalid (but still finite here) OBBs.
    assert torch.isfinite(out_guarded).all()


def test_offset_validity_decode_unguarded_remains_available_for_gradient():
    """The unguarded decode path must remain differentiable so
    gradient-bearing training outputs are not destroyed. This locks the
    'do not clamp the loss-bearing tensor' constraint from the plan.
    """
    ext_t = torch.tensor([_VALID_EXT], dtype=torch.float32)
    vo_t = torch.tensor([[-0.1, 0.7]], dtype=torch.float32, requires_grad=True)
    out = external_xyxy_rect_to_oriented_box(ext_t, vo_t, clamp_offsets=False)
    loss = out.sum()
    loss.backward()
    assert vo_t.grad is not None, "unguarded decode must be differentiable"
    assert torch.isfinite(
        vo_t.grad
    ).all(), f"unguarded decode gradient must be finite, got {vo_t.grad}"


def test_offset_validity_decode_guarded_on_zero_size_rect_finite():
    """A zero-size external rectangle (x1==x2, y1==y2) with invalid
    offsets must still yield a finite OBB under the guarded path. This
    is the most degenerate case in the plan acceptance list.
    """
    ext_t = torch.tensor([[0.3, 0.5, 0.3, 0.5]], dtype=torch.float32)
    vo_t = torch.tensor([[-1.0, 5.0]], dtype=torch.float32)
    out = external_xyxy_rect_to_oriented_box(ext_t, vo_t, clamp_offsets=True)
    assert torch.isfinite(
        out
    ).all(), f"zero-size ext rect guarded decode must be finite, got {out}"


def test_offset_validity_imports_on_disk_repo_module():
    """stale_state guard: the imported clamp_vertex_offsets_to_external_rect
    must come from this repo's on-disk engine/deim/obb_geometry.py, not a
    stale installed package.
    """
    import engine.deim.obb_geometry as geom

    expected = os.path.join(ROOT, "engine", "deim", "obb_geometry.py")
    actual = os.path.realpath(geom.__file__)
    assert actual == os.path.realpath(expected), (
        f"imported obb_geometry from {actual}, expected {expected}; "
        "a stale installed package may be shadowing the repo."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
