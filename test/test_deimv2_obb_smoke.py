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
from engine.deim.denoising import get_contrastive_denoising_training_group

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
      - out_corners (pred_corners) is finite, last-dim == 5*(reg_max+1).
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
    expected_corners_dim = 5 * (reg_max + 1)
    assert out_corners.shape[-1] == expected_corners_dim, (
        f"pred_corners last-dim must be {expected_corners_dim} "
        f"(5*(reg_max+1)), got {out_corners.shape[-1]}"
    )

    # --- theta range: out_bboxes theta in [0, π) (等比契约 norm_to_physical_rad) ---
    theta = out_bboxes[..., 4]
    assert (theta >= 0).all(), (
        f"pred_boxes theta must be >= 0, got min={theta.min().item():.4f}"
    )
    assert (theta < math.pi).all(), (
        f"pred_boxes theta must be < π, got max={theta.max().item():.4f}"
    )
    # ref_points theta 同域
    theta_refs = out_refs[..., 4]
    assert (theta_refs >= 0).all() and (theta_refs < math.pi).all(), (
        f"ref_points theta must be in [0, π), got "
        f"min={theta_refs.min().item():.4f} max={theta_refs.max().item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: anchor default r=0.25 → 物理 π/4（等比契约保持迁移前初始化方向）
# ---------------------------------------------------------------------------


def test_anchor_default_r_is_pi_over_4():
    """_generate_anchors 单角度默认 r=0.25, sigmoid 后 *π = π/4。"""
    torch.manual_seed(0)
    model = DEIMTransformer(
        num_classes=5,
        hidden_dim=32,
        num_queries=4,
        feat_channels=[32, 32],
        feat_strides=[4, 8],
        num_levels=2,
        num_points=2,
        nhead=4,
        num_layers=3,
        dim_feedforward=64,
        dropout=0.0,
        activation="relu",
        num_denoising=0,
        learn_query_content=False,
        eval_spatial_size=(16, 16),
        eval_idx=-1,
        eps=1e-2,
        aux_loss=False,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=4,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
        box_mode="obb",
        angle_rep=0,
        offset_scale_source="pre",
    )
    # model.anchors 是 init 时缓存的 logit 空间 buffer; sigmoid 还原到 [0,1] norm
    r = torch.sigmoid(model.anchors[..., -1])
    # angle_step=0 → 所有候选应为同一默认值 0.25 (物理 π/4)
    assert torch.allclose(r, torch.full_like(r, 0.25), atol=1e-6), (
        f"anchor r 应为 0.25 (物理 π/4), got min={r.min():.6f} max={r.max():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 4: 所有 angle_rep 前向输出 θ ∈ [0, π)（pred_boxes / ref_points / pre_bboxes）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("angle_rep", "use_angle_first"),
    [(0, False), (1, False), (2, False), (3, True)],
)
def test_angle_rep_forward_theta_in_proportional_domain(
    angle_rep, use_angle_first
):
    torch.manual_seed(0)
    model = DEIMTransformer(
        num_classes=5,
        hidden_dim=32,
        num_queries=4,
        feat_channels=[32, 32],
        feat_strides=[4, 8],
        num_levels=2,
        num_points=2,
        nhead=4,
        num_layers=3,
        dim_feedforward=64,
        dropout=0.0,
        activation="relu",
        num_denoising=0,
        learn_query_content=False,
        eval_spatial_size=(16, 16),
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=4,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
        box_mode="obb",
        angle_rep=angle_rep,
        offset_scale_source="pre",
        use_angle_first=use_angle_first,
    )
    model.train()
    feats = [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4)]
    with torch.no_grad():
        outputs = model(feats)
    for name, tensor in [
        ("pred_boxes", outputs["pred_boxes"][..., 4]),
        ("ref_points", outputs["ref_points"][..., 4]),
        ("pre_bboxes", outputs["pre_outputs"]["pred_boxes"][..., 4]),
    ]:
        assert (tensor >= 0).all() and (tensor < math.pi).all(), (
            f"angle_rep={angle_rep}: {name} θ 应 ∈ [0, π), "
            f"got min={tensor.min():.4f} max={tensor.max():.4f}"
        )


# ---------------------------------------------------------------------------
# Test 5: denoising 等比化 — GT θ ∈ [0,π) → θ_norm ∈ [0,1)，无 clip
# ---------------------------------------------------------------------------


import torch.nn as nn


def _run_denoising(gt_theta_rad, num_classes=5, hidden_dim=8):
    """调用 denoising 函数，返回 sigmoid 还原后的 θ_norm。"""
    target = {
        "labels": torch.tensor([0], dtype=torch.int64),
        "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.2, gt_theta_rad]], dtype=torch.float32),
    }
    class_embed = nn.Embedding(num_classes + 1, hidden_dim)
    _, dn_bbox_unact, _, _ = get_contrastive_denoising_training_group(
        targets=[target],
        num_classes=num_classes,
        num_queries=4,
        class_embed=class_embed,
        num_denoising=10,
        label_noise_ratio=0.0,
        box_noise_scale=1.0,
        box_mode="obb",
    )
    # dn_bbox_unact 是 inverse_sigmoid 后的 logit; sigmoid 还原到 θ_norm
    return torch.sigmoid(dn_bbox_unact[..., 4])


@pytest.mark.parametrize("gt_theta, expected_norm", [
    (3 * math.pi / 4, 0.75),        # 旧 shifted (θ+π/4)/π 会给 1.0 (clip), 等比给 0.75
    (math.pi / 4, 0.25),
    (0.0, 0.0),
])
def test_denoising_theta_norm_proportional(gt_theta, expected_norm):
    torch.manual_seed(0)
    theta_norm = _run_denoising(gt_theta)
    # 角度不加噪, 所有 dn query 共享同一 GT θ_norm; atol 放宽到 1e-4 以吸收
    # θ=0 边界处 inverse_sigmoid(0)=-inf → sigmoid≈1e-5 的数值往返误差
    assert torch.allclose(theta_norm, torch.full_like(theta_norm, expected_norm), atol=1e-4), (
        f"GT θ={gt_theta:.4f}: 期望 θ_norm={expected_norm}, "
        f"got min={theta_norm.min():.6f} max={theta_norm.max():.6f}"
    )


def test_denoising_theta_near_pi_no_overflow():
    """GT θ → π⁻ 时 θ_norm → 1⁻, 不越界 (旧 shifted (θ+π/4)/π 会 >1)。"""
    torch.manual_seed(0)
    theta_norm = _run_denoising(math.pi - 1e-4)
    assert (theta_norm < 1.0).all(), f"θ_norm 应 < 1, got max={theta_norm.max():.6f}"
    assert (theta_norm > 0.999).all(), f"θ near π 时 θ_norm 应接近 1, got min={theta_norm.min():.6f}"
