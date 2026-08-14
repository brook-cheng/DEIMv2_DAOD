"""OBB checkpoint compatibility contract (plan Task 10).

Old proportional / pre-cleanup rep3 checkpoints are intentionally
incompatible with the shifted-only decoder. They must fail explicitly at
every load path (resume, tuning, inference, export) rather than load
silently via ``strict=False``.

The mechanism: new OBB checkpoints carry
``meta.obb_angle_contract = "shifted_v1"``. ``assert_checkpoint_compat``
enforces it. ``classify_checkpoint_kind`` distinguishes 4D HBB pretraining
(still valid for OBB tuning) from ambiguous 5D/6D OBB weights.

Run:
    pytest test/test_obb_checkpoint_contract.py -v
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.solver._solver import (  # noqa: E402
    OBB_ANGLE_CONTRACT,
    CheckpointIncompatibleError,
    assert_checkpoint_compat,
    classify_checkpoint_kind,
)


def test_marker_must_match_shifted_v1():
    # marked shifted OBB -> accepted
    assert_checkpoint_compat({"meta": {"obb_angle_contract": OBB_ANGLE_CONTRACT}})
    # wrong marker -> rejected
    with pytest.raises(CheckpointIncompatibleError, match="obb_angle_contract"):
        assert_checkpoint_compat({"meta": {"obb_angle_contract": "proportional"}})


def test_missing_marker_rejected():
    # old OBB checkpoint: no marker -> must fail explicitly
    with pytest.raises(CheckpointIncompatibleError, match="obb_angle_contract"):
        assert_checkpoint_compat({"meta": {}})
    with pytest.raises(CheckpointIncompatibleError, match="obb_angle_contract"):
        assert_checkpoint_compat({})


def test_classify_hbb_vs_obb_by_encoder_box_head_dim():
    def state_with(dof):
        return {"model": {"enc_bbox_head.layers.2.bias": _bias(dof)}}

    # dof 4 -> HBB pretraining (accepted for OBB tuning)
    assert classify_checkpoint_kind(state_with(4)) == "hbb"
    # dof 5/6 -> OBB (must carry the marker or be rejected)
    assert classify_checkpoint_kind(state_with(5)) == "obb"
    assert classify_checkpoint_kind(state_with(6)) == "obb"


def test_classify_unknown_when_head_missing():
    assert classify_checkpoint_kind({"model": {}}) == "unknown"


def _bias(dof):
    import torch

    return torch.zeros(dof)
