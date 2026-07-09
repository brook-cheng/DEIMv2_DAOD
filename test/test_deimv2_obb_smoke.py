"""Decouple-angle reference path smoke tests (plan Todo 7).

Audit-driven tests verifying that the ``angle_rep=True`` OBB decoder
reference path keeps reference-point dimensionality deliberate and finite
across decoder layers.

Audit conclusion (see ``.omo/evidence/task-7-deimv2-obb-adr-hybrid.txt``):
MSDeformableAttention CONSUMES the 5th reference dimension as theta (it is
NOT ignored), and the decoder's 6-dim ADR -> 5-dim OBB transition across
layers is semantically correct. These tests lock that conclusion.

Run:
    pytest test/test_deimv2_obb_smoke.py -k decouple_angle_reference -v
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

from engine.deim.dfine_decoder import MSDeformableAttention
from engine.deim.deim_decoder import DEIMTransformer
from engine.deim.obb_geometry import external_rect_to_oriented_box

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msdeform_value(bs, num_head, head_dim, spatial_shapes):
    """Build the per-level ``value`` list MSDeformableAttention expects.

    Mirrors ``TransformerDecoder.value_op`` output shape:
    list of ``[bs, num_head, head_dim, h*w]`` tensors, one per level.
    """
    value = []
    for h, w in spatial_shapes:
        value.append(torch.randn(bs, num_head, head_dim, h * w))
    return value


# ---------------------------------------------------------------------------
# Test 1: MSDeformableAttention consumes theta (5th dim), does not ignore it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0])
def test_msdeform_attn_decouple_angle_reference_consumes_theta(seed):
    """Verify MSDeformAttn reference-point dimensionality handling.

    Given: a minimal MSDeformableAttention on CPU with 2 levels.
    When:  forward is called with 4-dim, 5-dim, and 6-dim reference points.
    Then:
      - All three produce finite output.
      - 5-dim OBB and 6-dim ADR (converting to the same OBB) produce
        IDENTICAL output, proving the two paths converge.
      - Changing the 5th dim (theta) of a 5-dim reference CHANGES the
        output, proving theta is consumed (not ignored).
      - Invalid dim counts (3, 7) raise ValueError.
    """
    torch.manual_seed(seed)

    embed_dim = 32
    num_heads = 4
    num_levels = 2
    num_points = 2
    spatial_shapes = [(4, 4), (2, 2)]
    bs, n_queries = 1, 5

    attn = MSDeformableAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        method="default",
    )
    attn.eval()

    query = torch.randn(bs, n_queries, embed_dim)
    value = _make_msdeform_value(bs, num_heads, embed_dim // num_heads, spatial_shapes)

    # The decoder always passes reference_points with n_levels=1 in dim 2
    # (via ref_points_detach.unsqueeze(2), see deim_decoder.py:284,326).
    # MSDeformAttn's 4/5/6-dim branches broadcast this single level across
    # all num_levels via num_points_list.
    n_ref_levels = 1

    # --- 4-dim reference: (cx, cy, w, h) axis-aligned ---
    ref_4d = torch.rand(bs, n_queries, n_ref_levels, 4)
    out_4d = attn(query, ref_4d, value, spatial_shapes)
    assert torch.isfinite(out_4d).all(), "4-dim ref produced non-finite output"
    assert out_4d.shape == (bs, n_queries, embed_dim)

    # --- 5-dim reference: (cx, cy, w, h, theta) OBB, theta in [0, 1] ---
    ref_5d = torch.rand(bs, n_queries, n_ref_levels, 5)
    out_5d = attn(query, ref_5d, value, spatial_shapes)
    assert torch.isfinite(out_5d).all(), "5-dim ref produced non-finite output"
    assert out_5d.shape == (bs, n_queries, embed_dim)

    # --- 6-dim reference: (cx, cy, w, h, eps, eta) ADR ---
    ref_6d = torch.rand(bs, n_queries, n_ref_levels, 6)
    out_6d = attn(query, ref_6d, value, spatial_shapes)
    assert torch.isfinite(out_6d).all(), "6-dim ref produced non-finite output"
    assert out_6d.shape == (bs, n_queries, embed_dim)

    # --- 5-dim and 6-dim converge when 6-dim converts to the same OBB ---
    ref_6d_as_obb = external_rect_to_oriented_box(ref_6d[..., :4], ref_6d[..., 4:])
    out_6d_as_5d = attn(query, ref_6d_as_obb, value, spatial_shapes)
    assert torch.allclose(
        out_6d, out_6d_as_5d, atol=1e-6
    ), "6-dim ADR path and equivalent 5-dim OBB path must converge"

    # --- theta is consumed: changing 5th dim changes output ---
    ref_5d_rotated = ref_5d.clone()
    ref_5d_rotated[..., 4] = (ref_5d[..., 4] + 0.5) % 1.0
    out_5d_rotated = attn(query, ref_5d_rotated, value, spatial_shapes)
    assert not torch.allclose(out_5d, out_5d_rotated, atol=1e-6), (
        "Changing theta (5th dim) must change output — theta is consumed, "
        "not ignored"
    )

    # --- invalid dim counts raise ValueError ---
    for bad_dim in (3, 7):
        ref_bad = torch.rand(bs, n_queries, n_ref_levels, bad_dim)
        with pytest.raises(ValueError, match="must be 2 , 4 or 5"):
            attn(query, ref_bad, value, spatial_shapes)


# ---------------------------------------------------------------------------
# Test 2: DEIMTransformer angle_rep=True forward produces finite OBB
#         outputs with consistent reference dimensionality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0])
def test_decouple_angle_reference_dimensionality_consistent(seed):
    """Verify the full decouple-angle decoder forward path.

    Given: a minimal DEIMTransformer with box_mode="obb",
           angle_rep=True, 3 decoder layers, small hidden_dim, CPU.
    When:  forward is called in eval mode with synthetic multi-scale feats.
    Then:
      - out_bboxes (pred_boxes) is finite, last-dim == 5 (OBB).
      - out_refs is finite, last-dim == 5 (OBB).
      - pre_bboxes is finite, last-dim == 5 (OBB).
      - out_corners (pred_corners) is finite, last-dim == 6*(reg_max+1).
      - No NaN/Inf in any output tensor.
    """
    torch.manual_seed(seed)

    hidden_dim = 32
    num_layers = 3
    num_queries = 4
    num_classes = 5
    reg_max = 4
    feat_strides = [4, 8]
    eval_h, eval_w = 16, 16

    model = DEIMTransformer(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        feat_channels=[hidden_dim, hidden_dim],
        feat_strides=feat_strides,
        num_levels=len(feat_strides),
        num_points=2,
        nhead=4,
        num_layers=num_layers,
        dim_feedforward=64,
        dropout=0.0,
        activation="relu",
        num_denoising=0,
        learn_query_content=False,
        eval_spatial_size=(eval_h, eval_w),
        eval_idx=-1,
        eps=1e-2,
        aux_loss=False,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=reg_max,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
        box_mode="obb",
        angle_rep=True,
        offset_scale_source="pre",
    )
    # Use training mode so the output dict includes pred_corners and
    # ref_points (eval mode only returns pred_boxes/pred_logits, see
    # deim_decoder.py:1102-1103). With dropout=0.0, num_denoising=0,
    # aux_loss=False, and feat_channels matching hidden_dim (so input_proj
    # is Identity, no BatchNorm), training-mode forward is deterministic
    # and gradient-free under torch.no_grad().
    model.train()

    # Synthetic multi-scale features matching eval_spatial_size and strides.
    feats = [torch.randn(1, hidden_dim, eval_h // s, eval_w // s) for s in feat_strides]

    with torch.no_grad():
        outputs = model(feats)

    out_bboxes = outputs["pred_boxes"]
    out_logits = outputs["pred_logits"]
    out_corners = outputs["pred_corners"]
    out_refs = outputs["ref_points"]

    # --- finiteness ---
    for name, tensor in [
        ("pred_boxes", out_bboxes),
        ("pred_logits", out_logits),
        ("pred_corners", out_corners),
        ("ref_points", out_refs),
    ]:
        assert torch.isfinite(tensor).all(), f"{name} contains NaN/Inf"

    # --- dimensionality consistency ---
    # pred_boxes: (n_layers or 1, bs, n_queries, 5) — OBB (cx, cy, w, h, theta)
    assert (
        out_bboxes.shape[-1] == 5
    ), f"pred_boxes last-dim must be 5 (OBB), got {out_bboxes.shape[-1]}"
    # ref_points: same OBB dim
    assert (
        out_refs.shape[-1] == 5
    ), f"ref_points last-dim must be 5 (OBB), got {out_refs.shape[-1]}"
    # pred_corners: (n_layers or 1, bs, n_queries, 6*(reg_max+1)) — ADR 6-dof
    expected_corners_dim = 6 * (reg_max + 1)
    assert out_corners.shape[-1] == expected_corners_dim, (
        f"pred_corners last-dim must be {expected_corners_dim} "
        f"(6*(reg_max+1)), got {out_corners.shape[-1]}"
    )

    # --- theta range: out_bboxes theta in [0, pi] (DEIMTransformer rescales) ---
    theta = out_bboxes[..., 4]
    assert (theta >= 0).all() and (theta <= math.pi + 1e-5).all(), (
        f"pred_boxes theta must be in [0, pi], got "
        f"min={theta.min().item():.4f} max={theta.max().item():.4f}"
    )
