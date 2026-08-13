"""OBB offline inference boundary regression: ``orig_target_sizes`` must be ``[W, H]``.

``imgsz=(H, W)`` is the public convention, but ``PostProcessor`` consumes
``orig_target_sizes`` as ``[W, H]``. These tests drive the real
``infer_obb_and_export()`` loop with a recording model wrapper and lock the
boundary argument to the consumer contract. Only the expensive/external seams
(model construction, checkpoint loading, config parsing, DOTA export) are
replaced; transforms, the image loop, and ``dst_sz`` construction stay real.

Run:
    python -m pytest test/test_tool_deimv2_obb_infer_hw_order.py -v
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable, Protocol, cast

import pytest
import torch
from PIL import Image

# test/ is not a package, so the tool module is loaded by file location and
# accessed through a typed surface rather than imported by bare name.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _RecordingModel:
    """Stand-in for ``tool.DEIMv2OBB``: records the ``orig_target_sizes`` the
    entry loop passes in. Emits no detections, so coordinate recovery is never
    reached."""

    def __init__(self) -> None:
        self.seen_orig_target_sizes: list[list[list[int]]] = []

    def eval(self) -> None:
        return None

    def __call__(
        self, _x: torch.Tensor, orig_target_sizes: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        sizes = orig_target_sizes.detach().cpu()
        self.seen_orig_target_sizes.append(
            [
                [int(sizes[r, c].item()) for c in range(sizes.shape[1])]
                for r in range(sizes.shape[0])
            ]
        )
        return [
            {
                "labels": torch.tensor([], dtype=torch.int64),
                "boxes": torch.empty((0, 5)),
                "scores": torch.tensor([], dtype=torch.float32),
            }
        ]


class _InferTool(Protocol):
    """Typed surface of ``test/tool_deimv2_obb_infer.py`` as exercised here."""

    DEIMv2OBB: Callable[[dict[str, object], str], _RecordingModel]
    load_checkpoint: Callable[[_RecordingModel, str], _RecordingModel]
    deimv2_obb_outputs_to_dota: Callable[
        [dict[str, object], str, dict[str, object]], None
    ]

    def infer_obb_and_export(
        self,
        img_dir: str,
        ckpt: str,
        config: str,
        output_dir: str,
        classes_txt: str,
        imgsz: tuple[int, int] = (640, 640),
        max_det: int = 300,
        score_threshold: float = 0.0,
        device: str = "cuda:0",
        vis_dir: str | None = None,
    ) -> None: ...


def _load_tool_module() -> object:
    """Load ``test/tool_deimv2_obb_infer.py`` by file location."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tool_deimv2_obb_infer.py"
    )
    spec = importlib.util.spec_from_file_location("tool_deimv2_obb_infer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = cast(_InferTool, _load_tool_module())


def _run_infer_recording(
    imgsz: tuple[int, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _RecordingModel:
    """Drive the real entry loop against one image; return the recorder."""
    img_dir = tmp_path / "images"
    out_dir = tmp_path / "output"
    img_dir.mkdir()
    out_dir.mkdir()

    Image.new("RGB", (300, 200), color=(90, 60, 30)).save(img_dir / "frame.jpg")
    classes_txt = tmp_path / "classes.txt"
    _ = classes_txt.write_text("class0\n")

    recorder = _RecordingModel()

    def build_model(_config: dict[str, object], _device: str) -> _RecordingModel:
        return recorder

    def noop_load_checkpoint(
        model: _RecordingModel, _ckpt_path: str
    ) -> _RecordingModel:
        return model

    def fake_load_config(_path: str) -> dict[str, object]:
        return {
            "DINOv3STAsResAtten": {"weights_path": str(tmp_path / "unused.pth")},
            "HybridEncoder": {},
            "DEIMTransformer": {},
            "PostProcessor": {},
        }

    def noop_dota_export(
        _outputs_dict: dict[str, object],
        _output_dir: str,
        _labels_map: dict[str, object],
    ) -> None:
        return None

    monkeypatch.setattr(tool, "DEIMv2OBB", build_model)
    monkeypatch.setattr(tool, "load_checkpoint", noop_load_checkpoint)
    monkeypatch.setattr("engine.core.yaml_utils.load_config", fake_load_config)
    monkeypatch.setattr(tool, "deimv2_obb_outputs_to_dota", noop_dota_export)

    tool.infer_obb_and_export(
        img_dir=str(img_dir),
        ckpt=str(tmp_path / "dummy.pth"),
        config=str(tmp_path / "dummy.yml"),
        output_dir=str(out_dir),
        classes_txt=str(classes_txt),
        imgsz=imgsz,
        max_det=300,
        score_threshold=0.0,
        device="cpu",
    )
    return recorder


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def test_infer_obb_passes_width_height_order_for_rectangular_imgsz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given imgsz=(576, 1024), the loop must hand the wrapper [[1024, 576]]."""
    # Given
    imgsz: tuple[int, int] = (576, 1024)

    # When
    recorder = _run_infer_recording(imgsz, tmp_path, monkeypatch)

    # Then
    assert recorder.seen_orig_target_sizes[0] == [[1024, 576]]


def test_infer_obb_preserves_square_target_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a square imgsz=(640, 640), the recorded size stays [[640, 640]]."""
    # Given
    imgsz: tuple[int, int] = (640, 640)

    # When
    recorder = _run_infer_recording(imgsz, tmp_path, monkeypatch)

    # Then
    assert recorder.seen_orig_target_sizes[0] == [[640, 640]]
