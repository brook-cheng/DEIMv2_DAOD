import contextlib
import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.solver.precision import (
    cast_obb_geometry_fp32,
    resolve_amp_dtype,
    validate_amp_dtype_support,
)

GEOMETRY_KEYS = frozenset({"pred_boxes", "pred_corners", "ref_points"})


def _fp16_geometry(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "pred_boxes": torch.rand(2, 3, 5).half(),
        "pred_corners": torch.rand(2, 3, 8).half(),
        "pred_logits": torch.rand(2, 3, 7).half(),
    }


def _nested_obb_outputs() -> dict[str, object]:
    base = _fp16_geometry(0)
    return {
        **base,
        "ref_points": torch.rand(2, 3, 4).half(),
        "aux_outputs": [_fp16_geometry(1), _fp16_geometry(2)],
        "enc_aux_outputs": [_fp16_geometry(3)],
        "pre_outputs": _fp16_geometry(4),
        "dn_outputs": [_fp16_geometry(5)],
        "dn_pre_outputs": _fp16_geometry(6),
        "enc_meta": {"class_agnostic": False, "num_layers": 6, "note": "metadata"},
        "misc": {"extra": torch.rand(2).half()},
    }


# --- resolve_amp_dtype -------------------------------------------------------


def test_resolve_amp_dtype_missing_and_float16_resolve_to_float16() -> None:
    # Given: no name and the explicit FP16 name.
    # When: resolved through resolve_amp_dtype.
    # Then: both map to the existing CUDA FP16 autocast default.
    assert resolve_amp_dtype(None) is torch.float16
    assert resolve_amp_dtype("float16") is torch.float16


def test_resolve_amp_dtype_bfloat16_name_resolves_to_bfloat16() -> None:
    # Given: the explicit BF16 name.
    # When: resolved through resolve_amp_dtype.
    # Then: it maps to torch.bfloat16.
    assert resolve_amp_dtype("bfloat16") is torch.bfloat16


def test_resolve_amp_dtype_unknown_name_raises_value_error_with_value() -> None:
    # Given: a name outside the supported {float16, bfloat16} set.
    # When: resolved through resolve_amp_dtype.
    # Then: a ValueError names the offending value.
    for invalid in ("float32", "fp16", "BF16", "mixed"):
        try:
            resolve_amp_dtype(invalid)
        except ValueError as error:
            assert invalid in str(error)
        else:
            raise AssertionError(f"expected ValueError for {invalid!r}")


# --- validate_amp_dtype_support ----------------------------------------------


def test_validate_amp_dtype_support_float16_never_raises() -> None:
    # Given: the FP16 dtype on any device.
    # When: validated for AMP support.
    # Then: validation passes unconditionally.
    validate_amp_dtype_support(torch.float16, torch.device("cpu"))
    validate_amp_dtype_support(torch.float16, torch.device("cuda:0"))


def test_validate_amp_dtype_support_bfloat16_cpu_passes() -> None:
    # Given: BF16 requested on CPU (existing CPU AMP behavior is untouched).
    # When: validated for AMP support.
    # Then: validation passes without a hardware check.
    validate_amp_dtype_support(torch.bfloat16, torch.device("cpu"))


def test_validate_amp_dtype_support_bfloat16_cuda_supported_passes(
    monkeypatch,
) -> None:
    # Given: CUDA hardware that reports BF16 support.
    # When: BF16 AMP support is validated on that device.
    # Then: validation passes.
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    validate_amp_dtype_support(torch.bfloat16, torch.device("cuda:0"))


def test_validate_amp_dtype_support_bfloat16_cuda_unsupported_raises(
    monkeypatch,
) -> None:
    # Given: CUDA hardware without BF16 support.
    # When: BF16 AMP support is validated on that device.
    # Then: a RuntimeError names the device and recommends FP16.
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Fake GPU")
    try:
        validate_amp_dtype_support(torch.bfloat16, torch.device("cuda:0"))
    except RuntimeError as error:
        message = str(error)
        assert "Fake GPU" in message
        assert "float16" in message
    else:
        raise AssertionError("expected RuntimeError for unsupported BF16 CUDA")


# --- cast_obb_geometry_fp32 --------------------------------------------------


def test_cast_obb_geometry_fp32_converts_all_nested_geometry_only() -> None:
    # Given: nested main/aux/encoder/DN/pre OBB outputs with FP16 geometry
    # and logits, wrapped in dicts, lists, and tuples.
    outputs = _nested_obb_outputs()

    # When: the recursive FP32 conversion runs over the whole tree.
    converted = cast_obb_geometry_fp32(outputs)

    # Then: every geometry tensor is FP32 and every logit stays FP16.
    def assert_geometry_fp32(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in GEOMETRY_KEYS and isinstance(value, torch.Tensor):
                    assert value.dtype is torch.float32
                elif key == "pred_logits":
                    assert value.dtype is torch.float16
                else:
                    assert_geometry_fp32(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                assert_geometry_fp32(item)

    assert_geometry_fp32(converted)
    assert converted["pred_logits"].dtype is torch.float16


def test_cast_obb_geometry_fp32_preserves_logits_metadata_and_container_shapes() -> None:
    # Given: nested OBB outputs carrying logits, metadata, and a tuple slot.
    outputs = _nested_obb_outputs()
    outputs["ref_points"] = (
        torch.rand(2, 3, 4).half(),
        torch.rand(2, 3, 4).half(),
    )

    # When: the recursive FP32 conversion runs over the whole tree.
    converted = cast_obb_geometry_fp32(outputs)

    # Then: logits and metadata keep their exact values, tuple slots remain
    # tuples, and every container keeps its original shape.
    assert converted["pred_logits"].dtype is torch.float16
    assert torch.equal(converted["pred_logits"], outputs["pred_logits"])
    assert converted["enc_meta"] == {"class_agnostic": False, "num_layers": 6, "note": "metadata"}
    assert isinstance(converted["ref_points"], tuple)
    assert len(converted["ref_points"]) == 2
    assert all(t.dtype is torch.float32 for t in converted["ref_points"])
    assert len(converted["aux_outputs"]) == 2
    assert len(converted["dn_outputs"]) == 1
    assert converted["pred_boxes"].shape == outputs["pred_boxes"].shape


def test_cast_obb_geometry_fp32_does_not_mutate_source() -> None:
    # Given: nested OBB outputs with FP16 geometry and logits.
    outputs = _nested_obb_outputs()
    boxes_before = outputs["pred_boxes"]
    aux_before = outputs["aux_outputs"][0]
    logits_before = outputs["pred_logits"]

    # When: the recursive FP32 conversion runs over the whole tree.
    converted = cast_obb_geometry_fp32(outputs)

    # Then: the source tree keeps every original tensor and container intact.
    assert converted is not outputs
    assert boxes_before.dtype is torch.float16
    assert logits_before.dtype is torch.float16
    assert outputs["pred_boxes"] is boxes_before
    assert outputs["aux_outputs"][0] is aux_before  # auxiliary entry unchanged
    assert outputs["aux_outputs"][0]["pred_boxes"].dtype is torch.float16
    assert converted["pred_boxes"].dtype is torch.float32


def test_cast_obb_geometry_fp32_backpropagates_to_source() -> None:
    # Given: FP16 geometry tensors with requires_grad.
    source = {
        "pred_boxes": torch.rand(2, 3, 5, dtype=torch.float16, requires_grad=True),
        "pred_logits": torch.rand(2, 3, 7, dtype=torch.float16, requires_grad=True),
        "ref_points": torch.rand(2, 3, 4, dtype=torch.float16, requires_grad=True),
    }

    # When: converted geometry is summed and backpropagated.
    converted = cast_obb_geometry_fp32(source)
    (converted["pred_boxes"].sum() + converted["ref_points"].sum()).backward()

    # Then: gradients reach the original FP16 source tensors.
    assert source["pred_boxes"].grad is not None
    assert source["pred_boxes"].grad.dtype is torch.float16
    assert source["ref_points"].grad is not None
    # Logits are returned unchanged and were not part of the loss.
    assert source["pred_logits"].grad is None


# --- train_one_epoch precision wiring ----------------------------------------

class _AutocastRecorder:
    """Replaces torch.autocast; records every call's kwargs as a no-op context."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs) -> contextlib.nullcontext:
        self.calls.append(kwargs)
        return contextlib.nullcontext()


class _WiringModel(nn.Module):
    """Tiny module whose output dict has pred_boxes tied to a Parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, targets=None):
        return {"pred_boxes": self.scale * torch.randn(2, 4)}


class _WiringCriterion:
    """Records the outputs object it received; box_mode is configurable."""

    def __init__(self, box_mode: str) -> None:
        self.box_mode = box_mode
        self.last_outputs = None

    def train(self) -> None:
        pass

    def __call__(self, outputs, targets, **metas):
        self.last_outputs = outputs
        return {"loss_bbox": outputs["pred_boxes"].sum()}


def _wiring_batches(n: int = 2):
    return [
        (
            torch.randn(2, 3, 64, 64),
            [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])}],
        )
        for _ in range(n)
    ]


def _run_amp_epoch(monkeypatch, *, amp_dtype_name=None, obb_geometry_fp32=False,
                   box_mode="obb"):
    from engine.solver import det_engine

    recorder = _AutocastRecorder()
    monkeypatch.setattr(torch, "autocast", recorder)

    model = _WiringModel()
    criterion = _WiringCriterion(box_mode)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu")

    det_engine.train_one_epoch(
        False, None, model, criterion, _wiring_batches(1), optimizer,
        torch.device("cpu"), 0, max_norm=0.5, print_freq=100, scaler=scaler,
        amp_dtype_name=amp_dtype_name, obb_geometry_fp32=obb_geometry_fp32,
    )
    return recorder, criterion


def test_train_one_epoch_autocast_uses_resolved_bf16_dtype(monkeypatch) -> None:
    # Given: BF16 requested through the YAML-resolved name.
    # When: the AMP epoch runs.
    # Then: the model autocast context receives dtype=torch.bfloat16.
    recorder, _ = _run_amp_epoch(monkeypatch, amp_dtype_name="bfloat16")
    model_autocast_kwargs = [c for c in recorder.calls if "dtype" in c]
    assert model_autocast_kwargs, "model autocast call missing dtype kwarg"
    assert model_autocast_kwargs[0]["dtype"] is torch.bfloat16


def test_train_one_epoch_autocast_missing_dtype_defaults_to_float16(
    monkeypatch,
) -> None:
    # Given: no amp_dtype configured (the existing FP16 default).
    # When: the AMP epoch runs.
    # Then: the model autocast context receives dtype=torch.float16.
    recorder, _ = _run_amp_epoch(monkeypatch)
    model_autocast_kwargs = [c for c in recorder.calls if "dtype" in c]
    assert model_autocast_kwargs, "model autocast call missing dtype kwarg"
    assert model_autocast_kwargs[0]["dtype"] is torch.float16


def test_train_one_epoch_obb_fp32_converts_before_criterion(monkeypatch) -> None:
    # Given: OBB geometry FP32 requested on an OBB criterion.
    # When: the AMP epoch runs.
    # Then: cast_obb_geometry_fp32 runs and its result reaches the criterion.
    from engine.solver import det_engine

    spy_calls = []

    def spy_cast(outputs):
        spy_calls.append(outputs)
        return cast_obb_geometry_fp32(outputs)

    monkeypatch.setattr(det_engine, "cast_obb_geometry_fp32", spy_cast)
    _, criterion = _run_amp_epoch(
        monkeypatch, obb_geometry_fp32=True, box_mode="obb"
    )
    assert spy_calls, "cast_obb_geometry_fp32 was not called"
    assert criterion.last_outputs is not None
    assert criterion.last_outputs["pred_boxes"].dtype is torch.float32


def test_train_one_epoch_obb_fp32_skipped_for_hbb_criterion(monkeypatch) -> None:
    # Given: OBB geometry FP32 requested but the criterion is HBB.
    # When: the AMP epoch runs.
    # Then: the conversion is ignored entirely (spec HBB guard).
    from engine.solver import det_engine

    spy_calls = []

    def spy_cast(outputs):
        spy_calls.append(outputs)
        return cast_obb_geometry_fp32(outputs)

    monkeypatch.setattr(det_engine, "cast_obb_geometry_fp32", spy_cast)
    _, criterion = _run_amp_epoch(
        monkeypatch, obb_geometry_fp32=True, box_mode="hbb"
    )
    assert spy_calls == [], "cast_obb_geometry_fp32 must be skipped for HBB"
    assert criterion.last_outputs is not None


def test_train_one_epoch_obb_fp32_skipped_when_flag_false(monkeypatch) -> None:
    # Given: the default obb_geometry_fp32=False on an OBB criterion.
    # When: the AMP epoch runs.
    # Then: no geometry conversion happens before the criterion.
    from engine.solver import det_engine

    spy_calls = []

    def spy_cast(outputs):
        spy_calls.append(outputs)
        return cast_obb_geometry_fp32(outputs)

    monkeypatch.setattr(det_engine, "cast_obb_geometry_fp32", spy_cast)
    _, criterion = _run_amp_epoch(monkeypatch, obb_geometry_fp32=False, box_mode="obb")
    assert spy_calls == [], "cast_obb_geometry_fp32 must be skipped when flag is false"
    assert criterion.last_outputs is not None
