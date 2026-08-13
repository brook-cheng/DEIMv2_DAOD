"""Tests for ``deim_app.inference.torch_backend`` and the adapter ``predict``
delegation (Task 6, Step 2 onward).

The stub model + postprocessor defined below avoid constructing real engine
objects. Canned postprocessor templates come straight from the task brief so
the assertions are byte-exact against the spec.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from deim_app.adapters.deim import DeimDetectionAdapter
from deim_app.config.loader import LoadedAppConfig
from deim_app.config.mapping import ResolvedAlgorithmConfig
from deim_app.config.metadata import DatasetMetadata
from deim_app.config.schema import AppConfig, InferenceConfig, RuntimeConfig
from deim_app.errors import InferenceBackendError
from deim_app.inference.inputs import InputImage
from deim_app.inference.preprocessing import Preprocessor
from deim_app.inference.torch_backend import TorchBackend
from deim_app.predictions.collection import PredictionCollection
from deim_app.predictions.types import Detection, HBBDetection, OBBDetection


# ---------------------------------------------------------------------------
# Canned postprocessor templates (from the task brief)
# ---------------------------------------------------------------------------

_LABELS = torch.tensor([0, 1])  # (K,)
_SCORES = torch.tensor([0.9, 0.4])  # (K,)
_HBB_BOXES = torch.tensor(
    [[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 8.0, 9.0]]
)  # (K, 4) xyxy
_OBB_BOXES = torch.tensor(
    [[5.0, 6.0, 7.0, 8.0, 0.5], [1.0, 2.0, 3.0, 4.0, 1.0]]
)  # (K, 5) cxcywhθ


# ---------------------------------------------------------------------------
# Stub engine objects (subclass nn.Module to satisfy TorchBackend's contract)
# ---------------------------------------------------------------------------


class StubModel(nn.Module):
    """Records its forward input and returns a canned outputs dict."""

    def __init__(self, outputs: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.outputs = outputs
        self.last_input: torch.Tensor | None = None
        self.forward_calls: int = 0
        self.forward_inference_mode: list[bool] = []

    def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        self.forward_inference_mode.append(torch.is_inference_mode_enabled())
        self.last_input = tensor
        self.forward_calls += 1
        return self.outputs


class StubPostprocessor(nn.Module):
    """Records ``(outputs, orig_target_sizes)`` and returns a tiled tuple.

    Templates describe a SINGLE image's output; the stub tiles them along a
    fresh batch dimension of size ``orig_target_sizes.shape[0]`` so the deploy
    tuple always matches the input batch size.
    """

    def __init__(
        self,
        labels_template: torch.Tensor,
        boxes_template: torch.Tensor,
        scores_template: torch.Tensor,
    ) -> None:
        super().__init__()
        self.labels_template = labels_template
        self.boxes_template = boxes_template
        self.scores_template = scores_template
        self.calls: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []
        self.forward_calls: int = 0
        self.forward_inference_mode: list[bool] = []

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        orig_target_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.forward_inference_mode.append(torch.is_inference_mode_enabled())
        self.calls.append((outputs, orig_target_sizes))
        self.forward_calls += 1
        n = int(orig_target_sizes.shape[0])
        labels = self.labels_template.unsqueeze(0).repeat(n, 1)
        boxes = self.boxes_template.unsqueeze(0).repeat(n, 1, 1)
        scores = self.scores_template.unsqueeze(0).repeat(n, 1)
        return labels, boxes, scores


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_input_image(image_id: str, width: int, height: int) -> InputImage:
    img = Image.new("RGB", (width, height), (10, 20, 30))
    return InputImage(image_id=image_id, source=f"<test>/{image_id}", image=img)


def _make_metadata(
    class_names_by_label: dict[int, str],
    output_names_by_id: dict[int, str] | None = None,
    box_mode: str = "hbb",
) -> DatasetMetadata:
    return DatasetMetadata(
        box_mode=box_mode,
        num_classes=len(class_names_by_label),
        class_names_by_label=class_names_by_label,
        output_names_by_id=(
            output_names_by_id
            if output_names_by_id is not None
            else dict(class_names_by_label)
        ),
    )


def _outputs() -> dict[str, torch.Tensor]:
    return {
        "pred_logits": torch.zeros(1, 2, 3),
        "pred_boxes": torch.zeros(1, 2, 4),
    }


def _build_backend(
    metadata: DatasetMetadata,
    box_mode: str,
    preprocessor: Preprocessor,
    labels: torch.Tensor = _LABELS,
    boxes: torch.Tensor = _HBB_BOXES,
    scores: torch.Tensor = _SCORES,
    device: str = "cpu",
) -> tuple[TorchBackend, StubModel, StubPostprocessor]:
    model = StubModel(outputs=_outputs())
    post = StubPostprocessor(labels, boxes, scores)
    backend = TorchBackend(
        model=model,
        postprocessor=post,
        preprocessor=preprocessor,
        metadata=metadata,
        box_mode=box_mode,
        device=device,
    )
    return backend, model, post


# ===========================================================================
# Box-mode detection types
# ===========================================================================


def test_hbb_mode_produces_hbb_detections(small_preprocessor: Preprocessor) -> None:
    backend, _, _ = _build_backend(_make_metadata({0: "cat", 1: "dog"}), "hbb", small_preprocessor)

    collection = backend.predict((_make_input_image("img0", 10, 12),), batch_size=1)

    assert collection.box_mode == "hbb"
    dets = collection.predictions[0].detections
    assert len(dets) == 2
    assert all(isinstance(d, HBBDetection) for d in dets)
    assert not any(isinstance(d, OBBDetection) for d in dets)


def test_obb_mode_produces_obb_detections(small_preprocessor: Preprocessor) -> None:
    backend, _, _ = _build_backend(
        _make_metadata({0: "ship", 1: "plane"}, box_mode="obb"),
        "obb",
        small_preprocessor,
        boxes=_OBB_BOXES,
    )

    collection = backend.predict((_make_input_image("img0", 10, 12),), batch_size=1)

    assert collection.box_mode == "obb"
    dets = collection.predictions[0].detections
    assert len(dets) == 2
    assert all(isinstance(d, OBBDetection) for d in dets)


# ===========================================================================
# Class-name lookup (deployed labels → class_names_by_label, always)
# ===========================================================================


def test_class_name_always_uses_class_names_by_label(
    small_preprocessor: Preprocessor,
) -> None:
    """Deployed postprocessor emits raw model labels (0..N-1); the
    ``remap_mscoco_category`` branch does not fire in ``deploy_mode``. So
    ``class_names_by_label`` is always correct, regardless of whether
    ``output_names_by_id`` is populated.

    ``output_names_by_id`` is populated below with a WRONG MS-COCO-style
    mapping to verify it is NOT consulted. If it were, every label would be
    misnamed (label 0 → ``"person"`` instead of ``"cat"``, etc.).
    """
    metadata = DatasetMetadata(
        box_mode="hbb",
        num_classes=3,
        class_names_by_label={0: "cat", 1: "dog", 2: "bird"},
        # output_names_by_id populated with WRONG MS-COCO-style mapping
        # to verify it is NOT consulted.
        output_names_by_id={0: "person", 1: "bicycle", 2: "car"},
    )
    backend, _, _ = _build_backend(
        metadata,
        "hbb",
        small_preprocessor,
        labels=torch.tensor([0, 1, 2]),
        boxes=torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
        ),
        scores=torch.tensor([0.9, 0.8, 0.7]),
    )

    collection = backend.predict((_make_input_image("img0", 8, 8),), batch_size=1)

    dets = collection.predictions[0].detections
    for det, expected_name in zip(dets, ["cat", "dog", "bird"]):
        assert det.class_name == expected_name, (
            f"deployed label {det.class_id} should map via class_names_by_label "
            f"(expected {expected_name!r}, got {det.class_name!r}); "
            "output_names_by_id must NOT be consulted in deploy mode."
        )


# ===========================================================================
# orig_target_sizes ordering: [width, height] (NOT [h, w])
# ===========================================================================


def test_orig_target_sizes_uses_width_height_order(
    small_preprocessor: Preprocessor,
) -> None:
    backend, _, post = _build_backend(
        _make_metadata({0: "x", 1: "y"}), "hbb", small_preprocessor
    )
    # PIL image (width=30, height=20) -> original_size_hw=(20, 30) -> [w, h]=[30, 20].
    backend.predict((_make_input_image("img0", width=30, height=20),), batch_size=1)

    assert len(post.calls) == 1
    captured = post.calls[0][1]
    assert captured.shape == (1, 2)
    assert torch.equal(captured, torch.tensor([[30.0, 20.0]], dtype=torch.float32))


def test_orig_target_sizes_per_image_in_batch(
    small_preprocessor: Preprocessor,
) -> None:
    backend, _, post = _build_backend(
        _make_metadata({0: "x", 1: "y"}), "hbb", small_preprocessor
    )
    inputs = (
        _make_input_image("a", width=11, height=22),
        _make_input_image("b", width=33, height=44),
    )

    backend.predict(inputs, batch_size=2)

    captured = post.calls[0][1]
    assert torch.equal(
        captured,
        torch.tensor([[11.0, 22.0], [33.0, 44.0]], dtype=torch.float32),
    )


# ===========================================================================
# Boxes returned verbatim (no rescale)
# ===========================================================================


def test_hbb_boxes_equal_postprocessor_boxes(
    small_preprocessor: Preprocessor,
) -> None:
    backend, _, _ = _build_backend(
        _make_metadata({0: "a", 1: "b"}), "hbb", small_preprocessor
    )

    collection = backend.predict((_make_input_image("img0", 8, 8),), batch_size=1)

    d0, d1 = collection.predictions[0].detections
    assert isinstance(d0, HBBDetection)
    assert isinstance(d1, HBBDetection)
    assert d0.xyxy == (1.0, 2.0, 10.0, 20.0)
    assert d1.xyxy == (3.0, 4.0, 8.0, 9.0)


def test_obb_boxes_equal_postprocessor_boxes(
    small_preprocessor: Preprocessor,
) -> None:
    backend, _, _ = _build_backend(
        _make_metadata({0: "a", 1: "b"}, box_mode="obb"),
        "obb",
        small_preprocessor,
        boxes=_OBB_BOXES,
    )

    collection = backend.predict((_make_input_image("img0", 8, 8),), batch_size=1)

    d0, d1 = collection.predictions[0].detections
    assert isinstance(d0, OBBDetection)
    assert isinstance(d1, OBBDetection)
    assert d0.xywhr == (5.0, 6.0, 7.0, 8.0, 0.5)
    assert d1.xywhr == (1.0, 2.0, 3.0, 4.0, 1.0)


# ===========================================================================
# Batch splitting (deterministic)
# ===========================================================================


def test_batch_size_splits_deterministically(
    small_preprocessor: Preprocessor,
) -> None:
    backend, _, post = _build_backend(
        _make_metadata({0: "a", 1: "b"}), "hbb", small_preprocessor
    )
    inputs = (
        _make_input_image("i0", 10, 10),
        _make_input_image("i1", 20, 20),
        _make_input_image("i2", 30, 30),
    )

    collection = backend.predict(inputs, batch_size=2)

    # 3 inputs / batch_size=2 -> 2 batches (2 + 1).
    assert len(collection.predictions) == 3
    assert post.forward_calls == 2
    assert post.calls[0][1].shape == (2, 2)
    assert post.calls[1][1].shape == (1, 2)
    assert [p.image_id for p in collection.predictions] == ["i0", "i1", "i2"]


def test_invalid_batch_size_raises(small_preprocessor: Preprocessor) -> None:
    backend, _, _ = _build_backend(
        _make_metadata({0: "a"}), "hbb", small_preprocessor
    )
    with pytest.raises(ValueError):
        backend.predict((_make_input_image("i0", 8, 8),), batch_size=0)


# ===========================================================================
# Autograd safety: model + postprocessor run under torch.inference_mode()
# ===========================================================================


def test_model_forward_runs_under_inference_mode(
    small_preprocessor: Preprocessor,
) -> None:
    """The model forward must execute inside ``torch.inference_mode()`` so a
    deployed inference pass never builds an autograd graph (memory + speed)."""
    backend, model, _ = _build_backend(
        _make_metadata({0: "a", 1: "b"}), "hbb", small_preprocessor
    )

    backend.predict((_make_input_image("img0", 8, 8),), batch_size=1)

    assert model.forward_inference_mode == [True]


def test_postprocessor_runs_under_inference_mode(
    small_preprocessor: Preprocessor,
) -> None:
    """The postprocessor call must also execute inside
    ``torch.inference_mode()`` — its tensor ops would otherwise retain history
    anchored to the model outputs."""
    backend, _, post = _build_backend(
        _make_metadata({0: "a", 1: "b"}), "hbb", small_preprocessor
    )

    backend.predict((_make_input_image("img0", 8, 8),), batch_size=1)

    assert post.forward_inference_mode == [True]


def test_inference_mode_enabled_across_multi_batch(
    small_preprocessor: Preprocessor,
) -> None:
    """Every batch's model + postprocessor forward runs under inference mode."""
    backend, model, post = _build_backend(
        _make_metadata({0: "a", 1: "b"}), "hbb", small_preprocessor
    )
    inputs = (
        _make_input_image("i0", 8, 8),
        _make_input_image("i1", 8, 8),
        _make_input_image("i2", 8, 8),
    )

    backend.predict(inputs, batch_size=2)

    assert model.forward_inference_mode == [True, True]
    assert post.forward_inference_mode == [True, True]


# ===========================================================================
# Full collection retains all results (no truncation)
# ===========================================================================


def test_full_collection_retains_all_results(
    small_preprocessor: Preprocessor,
) -> None:
    backend, _, _ = _build_backend(
        _make_metadata({0: "a", 1: "b"}), "hbb", small_preprocessor
    )
    inputs = tuple(_make_input_image(f"i{i}", 8, 8) for i in range(3))

    collection = backend.predict(inputs, batch_size=1)

    assert len(collection.predictions) == 3
    for pred in collection.predictions:
        assert len(pred.detections) == 2


# ===========================================================================
# Timings + original_image retention
# ===========================================================================


def test_timings_are_non_negative(small_preprocessor: Preprocessor) -> None:
    backend, _, _ = _build_backend(
        _make_metadata({0: "a", 1: "b"}), "hbb", small_preprocessor
    )

    collection = backend.predict((_make_input_image("img0", 8, 8),), batch_size=1)

    timings = collection.predictions[0].timings
    assert timings.preprocess_s >= 0.0
    assert timings.inference_s >= 0.0
    assert timings.postprocess_s >= 0.0


def test_original_image_retained(small_preprocessor: Preprocessor) -> None:
    backend, _, _ = _build_backend(
        _make_metadata({0: "a", 1: "b"}), "hbb", small_preprocessor
    )
    src_image = Image.new("RGB", (13, 17), (1, 2, 3))
    inputs = (InputImage(image_id="keep", source="<test>/keep", image=src_image),)

    collection = backend.predict(inputs, batch_size=1)

    pred = collection.predictions[0]
    assert pred.original_image is src_image
    assert pred.original_size == (17, 13)


# ===========================================================================
# Adapter predict delegation + guard
# ===========================================================================


def _make_loaded_adapter(
    box_mode: str,
    model: StubModel,
    post: StubPostprocessor,
    metadata: DatasetMetadata,
) -> DeimDetectionAdapter:
    app = AppConfig(
        inference=InferenceConfig(device="cpu", batch_size=1),
        runtime=RuntimeConfig(input_size=(16, 16)),
    )
    resolved = ResolvedAlgorithmConfig(
        config_path=Path("/synthetic/app.yml"),
        overrides={},
        metadata=metadata,
        app=app,
    )
    loaded = LoadedAppConfig(
        app=app,
        engine_base={},
        source=Path("/synthetic/app.yml"),
        app_base=Path("/synthetic/base.yml"),
    )
    adapter = DeimDetectionAdapter(resolved=resolved, loaded=loaded)
    adapter._model = model
    adapter._postprocessor = post
    return adapter


def test_adapter_predict_delegates_to_backend(tmp_path: Path) -> None:
    metadata = _make_metadata({0: "cat", 1: "dog"})
    model = StubModel(outputs=_outputs())
    post = StubPostprocessor(_LABELS, _HBB_BOXES, _SCORES)
    adapter = _make_loaded_adapter("hbb", model, post, metadata)

    img_path = tmp_path / "P0001.png"
    Image.new("RGB", (24, 16), (0, 0, 0)).save(img_path)

    collection = adapter.predict(str(img_path))

    assert isinstance(collection, PredictionCollection)
    assert collection.box_mode == "hbb"
    assert len(collection.predictions) == 1
    dets: tuple[Detection, ...] = collection.predictions[0].detections
    assert all(isinstance(d, HBBDetection) for d in dets)
    assert [d.class_name for d in dets] == ["cat", "dog"]


def test_adapter_predict_raises_when_not_loaded() -> None:
    metadata = _make_metadata({0: "x"})
    app = AppConfig(
        inference=InferenceConfig(device="cpu", batch_size=1),
        runtime=RuntimeConfig(input_size=(16, 16)),
    )
    resolved = ResolvedAlgorithmConfig(
        config_path=Path("/synthetic/app.yml"),
        overrides={},
        metadata=metadata,
        app=app,
    )
    loaded = LoadedAppConfig(
        app=app,
        engine_base={},
        source=Path("/synthetic/app.yml"),
        app_base=Path("/synthetic/base.yml"),
    )
    adapter = DeimDetectionAdapter(resolved=resolved, loaded=loaded)

    with pytest.raises(InferenceBackendError):
        adapter.predict("unused-source")


def test_adapter_predict_rejects_checkpoint_argument() -> None:
    metadata = _make_metadata({0: "x"})
    model = StubModel(outputs=_outputs())
    post = StubPostprocessor(_LABELS, _HBB_BOXES, _SCORES)
    adapter = _make_loaded_adapter("hbb", model, post, metadata)

    with pytest.raises(InferenceBackendError):
        adapter.predict("unused", checkpoint="/some/ckpt.pth")
