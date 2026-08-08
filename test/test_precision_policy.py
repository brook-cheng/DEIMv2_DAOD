import contextlib
import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.solver import det_engine


# --- precision module absence ------------------------------------------------


def test_precision_module_deleted() -> None:
    # Given: the mode-specific precision helpers were removed.
    # When: the deleted module is imported.
    # Then: an ImportError is raised.
    with pytest.raises(ImportError):
        from engine.solver import precision  # noqa: F401


# --- train_one_epoch BF16 + FP32-loss wiring ---------------------------------


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


class _Bf16OutputModel(nn.Module):
    """Simulates a BF16 autocast forward: nested outputs with mixed dtypes."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, targets=None):
        scale = self.scale.to(torch.bfloat16)
        return {
            "pred_boxes": scale * torch.randn(2, 4, dtype=torch.bfloat16),
            "pred_logits": torch.randn(2, 3, dtype=torch.bfloat16),
            "aux_outputs": [{"pred_boxes": scale * torch.randn(2, 4, dtype=torch.bfloat16)}],
            "num_queries": torch.tensor(300, dtype=torch.int64),
        }


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


def _run_amp_epoch(monkeypatch, *, box_mode="obb", model=None):
    recorder = _AutocastRecorder()
    monkeypatch.setattr(torch, "autocast", recorder)

    model = model or _WiringModel()
    criterion = _WiringCriterion(box_mode)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    det_engine.train_one_epoch(
        False, None, model, criterion, _wiring_batches(1), optimizer,
        torch.device("cpu"), 0, max_norm=0.5, print_freq=100, use_amp=True,
    )
    return recorder, criterion


def test_train_one_epoch_autocast_uses_bfloat16(monkeypatch) -> None:
    # Given: use_amp=True requests the BF16 forward path.
    # When: the AMP epoch runs.
    # Then: the model autocast context receives dtype=torch.bfloat16.
    recorder, _ = _run_amp_epoch(monkeypatch)
    model_autocast_kwargs = [c for c in recorder.calls if "dtype" in c]
    assert model_autocast_kwargs, "model autocast call missing dtype kwarg"
    assert model_autocast_kwargs[0]["dtype"] is torch.bfloat16


def test_train_one_epoch_use_amp_false_skips_autocast(monkeypatch) -> None:
    # Given: use_amp=False requests the plain FP32 path.
    # When: the epoch runs.
    # Then: no model autocast call carries a dtype kwarg.
    recorder = _AutocastRecorder()
    monkeypatch.setattr(torch, "autocast", recorder)

    model = _WiringModel()
    criterion = _WiringCriterion("obb")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    det_engine.train_one_epoch(
        False, None, model, criterion, _wiring_batches(1), optimizer,
        torch.device("cpu"), 0, max_norm=0.5, print_freq=100, use_amp=False,
    )
    assert [c for c in recorder.calls if "dtype" in c] == []


def test_train_one_epoch_scaler_kwarg_does_not_enable_amp(monkeypatch) -> None:
    # Given: a legacy call site still passes a GradScaler.
    # When: use_amp is not set.
    # Then: the epoch runs the plain FP32 path (no model autocast with dtype).
    recorder = _AutocastRecorder()
    monkeypatch.setattr(torch, "autocast", recorder)

    model = _WiringModel()
    criterion = _WiringCriterion("obb")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu")
    det_engine.train_one_epoch(
        False, None, model, criterion, _wiring_batches(1), optimizer,
        torch.device("cpu"), 0, max_norm=0.5, print_freq=100, scaler=scaler,
    )
    assert [c for c in recorder.calls if "dtype" in c] == []


def test_train_one_epoch_criterion_receives_float32_outputs(monkeypatch) -> None:
    # Given: a BF16 autocast forward produces BF16 model outputs.
    # When: the AMP epoch runs.
    # Then: the criterion receives nested outputs with all floating tensors cast
    #       to FP32, while container structure and non-float dtypes are preserved.
    _, criterion = _run_amp_epoch(monkeypatch, box_mode="obb", model=_Bf16OutputModel())
    outputs = criterion.last_outputs
    assert outputs is not None
    assert outputs["pred_boxes"].dtype == torch.float32
    assert outputs["pred_logits"].dtype == torch.float32
    assert outputs["aux_outputs"][0]["pred_boxes"].dtype == torch.float32
    assert outputs["num_queries"].dtype == torch.int64
    assert isinstance(outputs["aux_outputs"], list)
    assert set(outputs.keys()) == {
        "pred_boxes", "pred_logits", "aux_outputs", "num_queries",
    }
