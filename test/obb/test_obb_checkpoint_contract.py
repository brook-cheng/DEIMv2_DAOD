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
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.solver._solver import (  # noqa: E402
    OBB_ANGLE_CONTRACT,
    BaseSolver,
    CheckpointIncompatibleError,
    assert_checkpoint_compat,
    classify_checkpoint_kind,
    model_is_obb,
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


# ---------------------------------------------------------------------------
# Real DEIM wrapper checkpoints (Task 1 fix).
#
# Real shipped checkpoints store the encoder box-head bias under the
# ``decoder.`` submodule prefix — the encoder box-head lives inside the DEIM
# ``decoder`` submodule — e.g.:
#
#   ckpts/down_stream_task_models/deimv2_dinov3_x_coco.pth
#     model.decoder.enc_bbox_head.layers.2.bias  shape (4,)   # HBB pretraining
#   outputs/dlzdt_ablation/abl_rep0.pth
#     model.decoder.enc_bbox_head.layers.2.bias  shape (5,)   # OBB shifted
#
# ``classify_checkpoint_kind`` must probe that wrapper-prefixed key, otherwise
# real 4D HBB pretraining is misclassified as ``"unknown"`` and wrongly rejected
# for OBB tuning, while real unmarked 5D/6D OBB weights silently slip the
# marker check.
# ---------------------------------------------------------------------------

def _wrapper_state(dof):
    """Mirror the real DEIM wrapper checkpoint key layout exactly."""
    return {"model": {"decoder.enc_bbox_head.layers.2.bias": _bias(dof)}}


def test_classify_real_wrapper_prefixed_hbb_is_hbb():
    # real 4D HBB pretraining (deimv2_dinov3_x_coco.pth layout) -> "hbb"
    assert classify_checkpoint_kind(_wrapper_state(4)) == "hbb"


def test_classify_real_wrapper_prefixed_obb_5d_is_obb():
    # real shifted OBB 5D head (abl_rep0.pth layout) -> "obb"
    assert classify_checkpoint_kind(_wrapper_state(5)) == "obb"


def test_classify_real_wrapper_prefixed_obb_6d_is_obb():
    # legacy 6D (gate-fusion) head under the wrapper prefix -> "obb"
    assert classify_checkpoint_kind(_wrapper_state(6)) == "obb"


def test_classify_wrapper_prefixed_through_ema_module():
    # EMA weights carry the same wrapper-prefixed head key under ema.module.
    ema_state = {
        "ema": {"module": {"decoder.enc_bbox_head.layers.2.bias": _bias(4)}}
    }
    # ema.module is a 4D HBB head -> "hbb" (still valid for OBB tuning).
    assert classify_checkpoint_kind(ema_state) == "hbb"


def test_classify_flat_bare_decoder_key_still_supported():
    # The bare-decoder helper contract (flat ``enc_bbox_head.layers.2.bias``,
    # no ``decoder.`` prefix) remains supported — existing fixtures rely on it.
    assert (
        classify_checkpoint_kind({"model": {"enc_bbox_head.layers.2.bias": _bias(4)}})
        == "hbb"
    )


def test_load_tuning_state_accepts_real_wrapper_prefixed_4d_hbb(tmp_path):
    # Real 4D HBB pretraining under the ``decoder.`` prefix must be accepted
    # for OBB tuning (the verified deimv2_dinov3_x_coco.pth layout).
    ckpt = {"model": {"decoder.enc_bbox_head.layers.2.bias": _bias(4)}}
    path = tmp_path / "wrapper_hbb4d.pth"
    torch.save(ckpt, str(path))

    solver = _solver_with(_WrappedModel("obb"))
    solver.load_tuning_state(str(path))


def test_load_tuning_state_rejects_unmarked_real_wrapper_prefixed_5d_obb(tmp_path):
    # A real unmarked 5D OBB checkpoint under the ``decoder.`` prefix must be
    # rejected explicitly (no marker) rather than silently loaded.
    ckpt = {"model": {"decoder.enc_bbox_head.layers.2.bias": _bias(5)}}
    path = tmp_path / "wrapper_unmarked_obb5d.pth"
    torch.save(ckpt, str(path))

    solver = _solver_with(_WrappedModel("obb"))
    with pytest.raises(CheckpointIncompatibleError, match="obb_angle_contract"):
        solver.load_tuning_state(str(path))


# ---------------------------------------------------------------------------
# Wrapper-boundary integration tests (Task 1).
#
# The real DEIM model wrapper (engine/deim/deim.py DEIM) exposes ``box_mode``
# on its ``decoder`` submodule, NOT at the top level. The lightweight stand-ins
# below reproduce that boundary exactly: a top-level module with a ``decoder``
# carrying ``box_mode`` and no top-level ``box_mode`` attribute. They exercise
# the real ``BaseSolver`` methods (no broad mocks that skip them).
# ---------------------------------------------------------------------------


class _FakeDecoder(nn.Module):
    def __init__(self, box_mode: str) -> None:
        super().__init__()
        self.box_mode = box_mode


class _WrappedModel(nn.Module):
    def __init__(self, box_mode: str) -> None:
        super().__init__()
        self.decoder = _FakeDecoder(box_mode)


def _solver_with(model: nn.Module) -> BaseSolver:
    """Construct a BaseSolver bypassing __init__ (no full config/setup needed)."""
    solver = BaseSolver.__new__(BaseSolver)
    solver.last_epoch = 0
    solver.model = model
    return solver


def test_model_is_obb_detects_through_wrapper_boundary():
    obb_wrapper = _WrappedModel("obb")
    assert not hasattr(obb_wrapper, "box_mode")
    assert model_is_obb(obb_wrapper) is True

    hbb_wrapper = _WrappedModel("hbb")
    assert model_is_obb(hbb_wrapper) is False

    assert model_is_obb(_FakeDecoder("obb")) is True


def test_state_dict_writes_marker_for_wrapped_obb_model():
    solver = _solver_with(_WrappedModel("obb"))
    state = solver.state_dict()
    assert state["meta"]["obb_angle_contract"] == OBB_ANGLE_CONTRACT


def test_state_dict_no_marker_for_wrapped_hbb_model():
    solver = _solver_with(_WrappedModel("hbb"))
    state = solver.state_dict()
    assert "meta" not in state


def test_load_resume_state_rejects_unmarked_obb(tmp_path):
    ckpt = {"model": {}, "last_epoch": 0}
    path = tmp_path / "unmarked_obb.pth"
    torch.save(ckpt, str(path))

    solver = _solver_with(_WrappedModel("obb"))
    with pytest.raises(CheckpointIncompatibleError, match="obb_angle_contract"):
        solver.load_resume_state(str(path))


def test_load_resume_state_accepts_marked_obb(tmp_path):
    ckpt = {"model": {}, "last_epoch": 0, "meta": {"obb_angle_contract": OBB_ANGLE_CONTRACT}}
    path = tmp_path / "marked_obb.pth"
    torch.save(ckpt, str(path))

    solver = _solver_with(_WrappedModel("obb"))
    solver.load_resume_state(str(path))


def test_load_tuning_state_rejects_unmarked_obb(tmp_path):
    ckpt = {"model": {"enc_bbox_head.layers.2.bias": _bias(5)}}
    path = tmp_path / "unmarked_obb5d.pth"
    torch.save(ckpt, str(path))

    solver = _solver_with(_WrappedModel("obb"))
    with pytest.raises(CheckpointIncompatibleError, match="obb_angle_contract"):
        solver.load_tuning_state(str(path))


def test_load_tuning_state_allows_identifiable_4d_hbb(tmp_path):
    ckpt = {"model": {"enc_bbox_head.layers.2.bias": _bias(4)}}
    path = tmp_path / "hbb4d.pth"
    torch.save(ckpt, str(path))

    solver = _solver_with(_WrappedModel("obb"))
    solver.load_tuning_state(str(path))
