"""Retained-representation forward contracts (plan Task 1).

Lock the two OBB representations retained after the ablation cleanup
(``angle_rep=0`` and ``angle_rep=3``, both with the ``shifted`` decoder
encoding) so the per-axis deletions in Tasks 3-10 cannot silently break them.

These are characterization tests: they MUST pass on the current (pre-cleanup)
code and stay green throughout the cleanup. They establish the scenario S1/S2
baseline evidence cited in
``docs/superpowers/plans/2026-08-13-obb-ablation-cleanup.md``.

Run:
    pytest test/test_obb_retained_representations.py -v
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

from engine.deim.deim_decoder import DEIMTransformer  # noqa: E402


# ---------------------------------------------------------------------------
# CPU model factory — mirrors test/test_deimv2_obb_smoke.py::_make_obb_model
# kept self-contained so this contract file has no dependency on another test
# module. All kwargs that the cleanup removes are still accepted today and are
# passed explicitly so the file reads correctly after Tasks 3-8 drop them.
# ---------------------------------------------------------------------------

HIDDEN_DIM = 32
NUM_LAYERS = 3
NUM_QUERIES = 4
NUM_CLASSES = 5
REG_MAX = 4
FEAT_STRIDES = [4, 8]
EVAL_H, EVAL_W = 16, 16


def _make_obb_model(angle_rep):
    torch.manual_seed(0)
    return DEIMTransformer(
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_queries=NUM_QUERIES,
        feat_channels=[HIDDEN_DIM, HIDDEN_DIM],
        feat_strides=FEAT_STRIDES,
        num_levels=len(FEAT_STRIDES),
        num_points=2,
        nhead=4,
        num_layers=NUM_LAYERS,
        dim_feedforward=64,
        dropout=0.0,
        activation="relu",
        num_denoising=0,
        learn_query_content=False,
        eval_spatial_size=(EVAL_H, EVAL_W),
        eval_idx=-1,
        eps=1e-2,
        aux_loss=False,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=REG_MAX,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
        box_mode="obb",
        angle_rep=angle_rep,
    )


def _synth_feats():
    """Synthetic multi-scale features matching ``eval_spatial_size``/strides."""
    return [
        torch.randn(1, HIDDEN_DIM, EVAL_H // s, EVAL_W // s) for s in FEAT_STRIDES
    ]


def _assert_public_5d_obb(tensor, name):
    """Public OBB boundary contract: 5D, finite, theta in [0, pi)."""
    assert tensor.shape[-1] == 5, (
        f"{name} last-dim must be 5 (OBB cx,cy,w,h,theta), got {tensor.shape[-1]}"
    )
    assert torch.isfinite(tensor).all(), f"{name} contains NaN/Inf"
    theta = tensor[..., 4]
    assert (theta >= 0).all(), (
        f"{name} theta must be >= 0, got min={theta.min().item():.6f}"
    )
    assert (theta < math.pi).all(), (
        f"{name} theta must be < pi, got max={theta.max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# S1: rep0 + shifted forward contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0])
def test_rep0_shifted_forward_contract(seed):
    """rep0 + shifted: finite 5D public OBBs, theta in [0, pi), 6D ADR corners.

    rep0 is the 5D reference + 6D ADR residual representation
    (num_reg_dist=6). Shifted is the decoder-private angle encoding that
    becomes unconditional after Task 8. This test locks the retained rep0
    behavior on the CURRENT code and must stay green through the cleanup.
    """
    torch.manual_seed(seed)
    model = _make_obb_model(angle_rep=0)
    # train() so the output dict includes pred_corners and ref_points
    # (eval mode returns only pred_boxes/pred_logits). dropout=0,
    # num_denoising=0, aux_loss=False -> deterministic under no_grad.
    model.train()

    with torch.no_grad():
        out = model(_synth_feats())

    _assert_public_5d_obb(out["pred_boxes"], "pred_boxes")
    _assert_public_5d_obb(out["ref_points"], "ref_points")
    assert torch.isfinite(out["pred_logits"]).all(), "pred_logits contains NaN/Inf"
    assert torch.isfinite(out["pred_corners"]).all(), "pred_corners contains NaN/Inf"
    # rep0 ADR residual dim: num_reg_dist=6, corners = 6 * (reg_max + 1)
    assert out["pred_corners"].shape[-1] == 6 * (REG_MAX + 1), (
        f"rep0 pred_corners last-dim must be {6 * (REG_MAX + 1)} (6D ADR), "
        f"got {out['pred_corners'].shape[-1]}"
    )


# ---------------------------------------------------------------------------
# S2: rep3 + shifted forward contract (no angle-first, no gate fusion behavior)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0])
def test_rep3_shifted_forward_contract(seed):
    """rep3 + shifted: finite 5D public OBBs, finite per-layer references,
    5D direct-angle corners.

    rep3 is the 5D direct-angle residual representation (num_reg_dist=5)
    with a decoupled angle stream and no gate fusion. This test locks the
    retained rep3 + shifted forward contract.
    """
    torch.manual_seed(seed)
    model = _make_obb_model(
        angle_rep=3,
    )
    model.train()

    with torch.no_grad():
        out = model(_synth_feats())

    _assert_public_5d_obb(out["pred_boxes"], "pred_boxes")
    _assert_public_5d_obb(out["ref_points"], "ref_points")
    assert torch.isfinite(out["pred_logits"]).all(), "pred_logits contains NaN/Inf"
    assert torch.isfinite(out["pred_corners"]).all(), "pred_corners contains NaN/Inf"
    # rep3 direct-angle residual dim: num_reg_dist=5, corners = 5 * (reg_max + 1)
    assert out["pred_corners"].shape[-1] == 5 * (REG_MAX + 1), (
        f"rep3 pred_corners last-dim must be {5 * (REG_MAX + 1)} (5D direct-angle), "
        f"got {out['pred_corners'].shape[-1]}"
    )
    # ref_points is the last layer's reference (dict stores out_refs[-1]);
    # per-layer finiteness is covered by test_deimv2_obb_smoke.py.
