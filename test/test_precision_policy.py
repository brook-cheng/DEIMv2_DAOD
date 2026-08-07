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


# --- train_one_epoch FP16 wiring preservation -------------------------------


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


def _run_amp_epoch(monkeypatch, *, box_mode="obb"):
    recorder = _AutocastRecorder()
    monkeypatch.setattr(torch, "autocast", recorder)

    model = _WiringModel()
    criterion = _WiringCriterion(box_mode)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu")

    det_engine.train_one_epoch(
        False, None, model, criterion, _wiring_batches(1), optimizer,
        torch.device("cpu"), 0, max_norm=0.5, print_freq=100, scaler=scaler,
    )
    return recorder, criterion


def test_train_one_epoch_autocast_uses_fixed_float16(monkeypatch) -> None:
    # Given: no precision-mode kwargs exist (post-removal).
    # When: the AMP epoch runs.
    # Then: the model autocast context receives dtype=torch.float16.
    recorder, _ = _run_amp_epoch(monkeypatch)
    model_autocast_kwargs = [c for c in recorder.calls if "dtype" in c]
    assert model_autocast_kwargs, "model autocast call missing dtype kwarg"
    assert model_autocast_kwargs[0]["dtype"] is torch.float16


def test_train_one_epoch_criterion_receives_raw_outputs(monkeypatch) -> None:
    # Given: the OBB criterion with no geometry-cast configuration.
    # When: the AMP epoch runs.
    # Then: the criterion receives the original model outputs unchanged.
    _, criterion = _run_amp_epoch(monkeypatch, box_mode="obb")
    assert criterion.last_outputs is not None
    assert "pred_boxes" in criterion.last_outputs
