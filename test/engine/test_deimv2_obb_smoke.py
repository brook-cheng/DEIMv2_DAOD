"""Decouple-angle reference path smoke tests (plan Todo 7).

Audit-driven tests verifying that the ``angle_rep=True`` OBB decoder
reference path keeps reference-point dimensionality deliberate and finite
across decoder layers.

Audit conclusion (see internal task audit notes):
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
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.deim.dfine_decoder import MSDeformableAttention
from engine.deim.deim_decoder import DEIMTransformer
from engine.deim.obb_angle_contract import norm_to_physical_rad
from engine.deim.denoising import get_contrastive_denoising_training_group
from engine.deim.utils import inverse_sigmoid

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


def _make_obb_model(
    angle_rep,
    num_denoising=0,
):
    torch.manual_seed(0)
    return DEIMTransformer(
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
        num_denoising=num_denoising,
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
    )


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
# Test 1b: MSDeformableAttention 站点 5 — shifted 编码的注意力旋转
# ---------------------------------------------------------------------------


def test_msdeform_attn_shifted_90deg_rotation_axis():
    """spec §12.3: 90° reference（theta_shift=0.75）在 shifted 模式下改变
    θ 会改变输出（θ 被消费，非忽略），且输出有限。
    """
    torch.manual_seed(0)
    embed_dim, num_heads, num_levels, num_points = 32, 4, 2, 2
    spatial_shapes = [(4, 4), (2, 2)]
    bs, n_queries, n_ref_levels = 1, 3, 1

    attn_shift = MSDeformableAttention(
        embed_dim=embed_dim, num_heads=num_heads, num_levels=num_levels,
        num_points=num_points, method="default",
    )
    attn_shift.eval()

    query = torch.randn(bs, n_queries, embed_dim)
    value = _make_msdeform_value(bs, num_heads, embed_dim // num_heads, spatial_shapes)
    centers = torch.full((bs, n_queries, n_ref_levels, 2), 0.5)
    wh = torch.full((bs, n_queries, n_ref_levels, 2), 0.2)

    ref_90 = torch.cat([centers, wh, torch.full((bs, n_queries, n_ref_levels, 1), 0.75)], dim=-1)
    out_90 = attn_shift(query, ref_90, value, spatial_shapes)
    assert torch.isfinite(out_90).all()

    ref_0 = torch.cat([centers, wh, torch.full((bs, n_queries, n_ref_levels, 1), 0.25)], dim=-1)
    out_0 = attn_shift(query, ref_0, value, spatial_shapes)
    assert not torch.allclose(out_90, out_0, atol=1e-6), (
        "shifted 模式 θ 通道被忽略（90° 与 0° 输出相同）"
    )


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
        angle_rep=3,
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
    assert (
        theta >= 0
    ).all(), f"pred_boxes theta must be >= 0, got min={theta.min().item():.4f}"
    assert (
        theta < math.pi
    ).all(), f"pred_boxes theta must be < π, got max={theta.max().item():.4f}"
    # ref_points theta 同域
    theta_refs = out_refs[..., 4]
    assert (theta_refs >= 0).all() and (theta_refs < math.pi).all(), (
        f"ref_points theta must be in [0, π), got "
        f"min={theta_refs.min().item():.4f} max={theta_refs.max().item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: anchor default r=0.25 → 物理 π/4（等比契约保持迁移前初始化方向）
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 3c: anchor 站点 1/1b + encoder 辅助站点 6（spec §7）
# ---------------------------------------------------------------------------


def test_anchor_default_r_shifted_is_half():
    """shifted 默认 anchor θ=0.5（45° 于 sigmoid 中心）。"""
    model = _make_obb_model(angle_rep=0)
    anchors_unact, _ = model._generate_anchors([[4, 4], [2, 2]], device="cpu")
    anchors = torch.sigmoid(anchors_unact)
    assert torch.allclose(
        anchors[..., 4], torch.full_like(anchors[..., 4], 0.5), atol=1e-6
    ), f"anchor θ 应为 0.5, got {anchors[0, 0, 4].item():.6f}"


def test_encoder_aux_theta_known_answer_shifted():
    """零初始化下 encoder 辅助 θ 必为 π/4（shifted anchor θ_shift=0.5 还原）。"""
    torch.manual_seed(0)
    feats = [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4)]
    model = _make_obb_model(angle_rep=0)
    model.train()
    memory, spatial_shapes = model._get_encoder_input(feats)
    with torch.no_grad():
        _, _, enc_list, _ = model._get_decoder_input(memory, spatial_shapes)
    theta = enc_list[0][..., 4]
    assert torch.isfinite(theta).all(), "encoder 辅助 θ 含 NaN"
    assert torch.allclose(
        theta, torch.full_like(theta, math.pi / 4), atol=1e-4
    ), f"encoder 辅助 θ 应 ≈ π/4, got mean={theta.mean():.4f}"


# ---------------------------------------------------------------------------
# Test 4: 所有 angle_rep 前向输出 θ ∈ [0, π)（pred_boxes / ref_points / pre_bboxes）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle_rep", [0, 3])
def test_angle_rep_forward_theta_in_proportional_domain(angle_rep):
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
# Test 4b: 前向矩阵 4 表示 × 2 配置 + shifted known-answer（spec §12.2）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle_rep", [0, 3])
def test_forward_matrix_public_theta_domain(angle_rep):
    """公开输出 θ ∈ [0, π)、无 NaN。"""
    torch.manual_seed(0)
    model = _make_obb_model(
        angle_rep=angle_rep,
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
        assert torch.isfinite(tensor).all(), (
            f"rep={angle_rep} {name} 含 NaN"
        )
        assert (tensor >= 0).all() and (tensor < math.pi).all(), (
            f"rep={angle_rep} {name} θ 应 ∈ [0, π), "
            f"got min={tensor.min():.4f} max={tensor.max():.4f}"
        )


# ---------------------------------------------------------------------------
# Test 5: denoising — GT θ ∈ [0,π) → θ_shift ∈ [0,1)，无 clip
# ---------------------------------------------------------------------------


import torch.nn as nn


def _run_denoising_shifted(gt_theta_rad, num_classes=5, hidden_dim=8):
    target = {
        "labels": torch.tensor([0], dtype=torch.int64),
        "boxes": torch.tensor(
            [[0.5, 0.5, 0.3, 0.2, gt_theta_rad]], dtype=torch.float32
        ),
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
    return torch.sigmoid(dn_bbox_unact[..., 4])


def test_denoising_box_noise_scale_zero_preserves_original_obb():
    """box_noise_scale=0 returns the original OBB unchanged (angle as shifted norm)."""
    torch.manual_seed(0)
    gt_theta = math.pi / 4
    target = {
        "labels": torch.tensor([0], dtype=torch.int64),
        "boxes": torch.tensor(
            [[0.5, 0.5, 0.3, 0.2, gt_theta]], dtype=torch.float32
        ),
    }
    class_embed = nn.Embedding(6, 8)
    _, dn_bbox_unact, _, _ = get_contrastive_denoising_training_group(
        targets=[target],
        num_classes=5,
        num_queries=4,
        class_embed=class_embed,
        num_denoising=10,
        label_noise_ratio=0.0,
        box_noise_scale=0.0,
        box_mode="obb",
    )
    boxes = torch.sigmoid(dn_bbox_unact)
    # pi/4 -> shifted_norm = remainder(0.25 + 0.25, 1) = 0.5
    expected = torch.tensor([0.5, 0.5, 0.3, 0.2, 0.5])
    assert torch.allclose(
        boxes, expected.reshape(1, 1, 5).expand_as(boxes), atol=1e-4
    ), f"zero-noise must preserve GT OBB, got {boxes[0, 0].tolist()}"


@pytest.mark.parametrize(
    "gt_theta, expected_shift",
    [
        (0.0, 0.25),
        (math.pi / 4, 0.5),
        (math.pi / 2, 0.75),
        (3 * math.pi / 4, 0.0),
    ],
)
def test_denoising_theta_shifted(gt_theta, expected_shift):
    """shifted encoding: GT θ -> θ_shift in [0,1), no clip."""
    torch.manual_seed(0)
    theta_shift = _run_denoising_shifted(gt_theta)
    assert torch.allclose(
        theta_shift, torch.full_like(theta_shift, expected_shift), atol=1e-4
    ), (
        f"GT θ={gt_theta:.4f}: expected θ_shift={expected_shift}, "
        f"got min={theta_shift.min():.6f} max={theta_shift.max():.6f}"
    )
