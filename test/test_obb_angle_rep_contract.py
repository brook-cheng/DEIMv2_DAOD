"""OBB ``angle_rep`` type/value strictness contract (plan Task 2).

Python numeric equality lets ``False == 0``, ``True == 1``, ``0.0 == 0``,
and ``3.0 == 3``.  The old guard ``angle_rep not in (0, 3)`` therefore
accepted booleans and floats that are semantically invalid — a latent
foot-gun where a YAML ``True`` silently means ``1`` and triggers a
``ValueError`` far from its origin, or ``0.0`` silently means ``0`` but
bypasses the intent of an integer-only contract.

These tests lock the strict contract: only ``int`` ``0`` and ``3`` are
accepted by both OBB decoder constructors (``DEIMTransformer`` and
``TransformerDecoder``).  Every other type — ``bool``, ``float``,
out-of-range ``int`` — must raise immediately at construction time.

Run:
    pytest test/test_obb_angle_rep_contract.py -v
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.deim.deim_decoder import (  # noqa: E402
    DEIMTransformer,
    TransformerDecoder,
    TransformerDecoderLayer,
)

# ---------------------------------------------------------------------------
# Minimal CPU model factory — mirrors test_obb_retained_representations.py
# so construction is fast and dataset/checkpoint-free.
# ---------------------------------------------------------------------------

HIDDEN_DIM = 32
NHEAD = 4
NUM_LAYERS = 3
REG_MAX = 4


def _make_decoder_layers():
    """Create minimal valid TransformerDecoderLayer instances."""
    layer = TransformerDecoderLayer(
        d_model=HIDDEN_DIM,
        n_head=NHEAD,
        dim_feedforward=64,
        n_levels=2,
        n_points=2,
    )
    layer_wide = TransformerDecoderLayer(
        d_model=HIDDEN_DIM,
        n_head=NHEAD,
        dim_feedforward=64,
        n_levels=2,
        n_points=2,
        layer_scale=2,
    )
    return layer, layer_wide


def _make_transformer(angle_rep):
    """Construct a minimal CPU DEIMTransformer with ``box_mode='obb'``."""
    torch.manual_seed(0)
    return DEIMTransformer(
        num_classes=5,
        hidden_dim=HIDDEN_DIM,
        num_queries=4,
        feat_channels=[HIDDEN_DIM, HIDDEN_DIM],
        feat_strides=[4, 8],
        num_levels=2,
        num_points=2,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=64,
        num_denoising=0,
        eval_spatial_size=(16, 16),
        reg_max=REG_MAX,
        reg_scale=4.0,
        box_mode="obb",
        angle_rep=angle_rep,
    )


def _make_standalone_decoder(angle_rep):
    """Construct a minimal standalone TransformerDecoder with ``box_mode='obb'``."""
    layer, layer_wide = _make_decoder_layers()
    up = nn_Parameter_scalar()
    reg_scale = nn_Parameter_scalar()
    return TransformerDecoder(
        hidden_dim=HIDDEN_DIM,
        decoder_layer=layer,
        decoder_layer_wide=layer_wide,
        num_layers=NUM_LAYERS,
        num_head=NHEAD,
        num_reg_dist=6,
        reg_max=REG_MAX,
        reg_scale=reg_scale,
        up=up,
        eval_idx=-1,
        layer_scale=1,
        act="relu",
        box_mode="obb",
        angle_rep=angle_rep,
    )


def nn_Parameter_scalar():
    """A scalar nn.Parameter for reg_scale / up."""
    return torch.nn.Parameter(torch.tensor([0.5]), requires_grad=False)


# ---------------------------------------------------------------------------
# DEIMTransformer: only int 0 / 3 accepted for box_mode='obb'
# ---------------------------------------------------------------------------

REJECTED_VALUES = [False, True, 0.0, 3.0, 1, 2]
ACCEPTED_VALUES = [0, 3]


@pytest.mark.parametrize("bad_val", REJECTED_VALUES, ids=repr)
def test_deim_transformer_rejects_invalid_angle_rep(bad_val):
    """DEIMTransformer(box_mode='obb', angle_rep=<bad>) must raise."""
    with pytest.raises((TypeError, ValueError)):
        _make_transformer(bad_val)


@pytest.mark.parametrize("good_val", ACCEPTED_VALUES, ids=repr)
def test_deim_transformer_accepts_int_0_and_3(good_val):
    """DEIMTransformer(box_mode='obb', angle_rep=<0|3>) must construct."""
    model = _make_transformer(good_val)
    assert model.angle_rep == good_val
    assert isinstance(model.angle_rep, int)
    assert not isinstance(model.angle_rep, bool)


# ---------------------------------------------------------------------------
# TransformerDecoder: only int 0 / 3 accepted for box_mode='obb'
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_val", REJECTED_VALUES, ids=repr)
def test_transformer_decoder_rejects_invalid_angle_rep(bad_val):
    """Standalone TransformerDecoder(box_mode='obb', angle_rep=<bad>) must raise."""
    with pytest.raises((TypeError, ValueError)):
        _make_standalone_decoder(bad_val)


@pytest.mark.parametrize("good_val", ACCEPTED_VALUES, ids=repr)
def test_transformer_decoder_accepts_int_0_and_3(good_val):
    """Standalone TransformerDecoder(box_mode='obb', angle_rep=<0|3>) must construct."""
    decoder = _make_standalone_decoder(good_val)
    assert decoder.angle_rep == good_val
    assert isinstance(decoder.angle_rep, int)
    assert not isinstance(decoder.angle_rep, bool)


# ---------------------------------------------------------------------------
# hbb mode: angle_rep is irrelevant, must not be validated
# ---------------------------------------------------------------------------


def test_hbb_mode_does_not_validate_angle_rep():
    """For box_mode='hbb', angle_rep is ignored — no validation fires."""
    # Even a nonsensical angle_rep must not raise for hbb.
    model = DEIMTransformer(
        num_classes=5,
        hidden_dim=HIDDEN_DIM,
        num_queries=4,
        feat_channels=[HIDDEN_DIM, HIDDEN_DIM],
        feat_strides=[4, 8],
        num_levels=2,
        num_points=2,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=64,
        num_denoising=0,
        eval_spatial_size=(16, 16),
        reg_max=REG_MAX,
        reg_scale=4.0,
        box_mode="hbb",
        angle_rep=42,
    )
    assert model.box_mode == "hbb"


# ---------------------------------------------------------------------------
# rep3 preserves internal decouple_angle_layers
# ---------------------------------------------------------------------------


def test_rep3_still_creates_decouple_angle_layers():
    """angle_rep=3 must still build the internal decouple_angle_layers."""
    model = _make_transformer(3)
    assert hasattr(model.decoder, "decouple_angle_layers")
    assert len(model.decoder.decouple_angle_layers) == model.decoder.num_layers


def test_rep0_does_not_create_decouple_angle_layers():
    """angle_rep=0 must NOT build decouple_angle_layers."""
    model = _make_transformer(0)
    assert not hasattr(model.decoder, "decouple_angle_layers")
