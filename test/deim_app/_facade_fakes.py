"""Importable test doubles for ``DetectionModel`` facade tests.

Separated into its own module from ``conftest.py`` for two reasons:

1. **Module-name collision**: pytest collects multiple test directories
   without ``__init__.py``, so ``conftest`` resolves ambiguously across
   ``test/deim_app/conftest.py`` and ``test/deim_app/predictions/conftest.py``.
   A uniquely-named module avoids this.

2. **basedpyright ``reportImplicitRelativeImport``**: conftest.py is treated
   as a package member, so ``from <sibling> import ...`` inside it triggers
   the rule. Keeping ``FakeAdapter`` and the fixtures defined locally in this
   module (no sibling import needed here) sidesteps the issue for the spy
   class itself.

Both ``conftest.py`` and ``test_api.py`` / ``test_cli.py`` consume this
module via ``from _facade_fakes import ...``. ``conftest.py`` re-exports the
``fake_adapter`` and ``unloaded_fake_adapter`` fixtures so pytest discovers
them at the conftest scope; it used to register this module via
``pytest_plugins``, but pytest 9 rejects ``pytest_plugins`` in a non-top-level
conftest, so the fixtures are now imported directly (see conftest.py for the
full rationale).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from deim_app.config import LoadedAppConfig
from deim_app.config.metadata import DatasetMetadata
from deim_app.config.schema import AppConfig, InferenceConfig
from deim_app.errors import ExportError
from deim_app.predictions.collection import PredictionCollection
from deim_app.predictions.types import HBBDetection, ImagePrediction, Timings


def make_metadata(num_classes: int = 3) -> DatasetMetadata:
    """Build a 3-class HBB dataset metadata (c0, c1, c2)."""
    names: dict[int, str] = {i: f"c{i}" for i in range(num_classes)}
    return DatasetMetadata(
        box_mode="hbb",
        num_classes=num_classes,
        class_names_by_label=names,
        output_names_by_id=dict(names),
    )


def make_loaded(inference: InferenceConfig) -> LoadedAppConfig:
    """Build a minimal ``LoadedAppConfig`` with the given inference section."""
    return LoadedAppConfig(
        app=AppConfig(inference=inference),
        engine_base={},
        source=Path("/synthetic/app.yml"),
        app_base=Path("/synthetic/base.yml"),
    )


def make_collection() -> PredictionCollection:
    """Build a 3-detection HBB collection with scores [0.9, 0.8, 0.3].

    The spread (two above 0.5, one below) lets tests distinguish
    ``score_threshold=0.5`` (2 survivors) from no-threshold (3 survivors)
    and verify top-k truncation.
    """
    detections = (
        HBBDetection(class_id=0, class_name="c0", score=0.9, xyxy=(0.0, 0.0, 1.0, 1.0)),
        HBBDetection(class_id=1, class_name="c1", score=0.8, xyxy=(1.0, 1.0, 2.0, 2.0)),
        HBBDetection(class_id=2, class_name="c2", score=0.3, xyxy=(2.0, 2.0, 3.0, 3.0)),
    )
    return PredictionCollection(
        box_mode="hbb",
        predictions=(
            ImagePrediction(
                image_id="img0",
                source="/synthetic/img0.png",
                original_image=Image.new("RGB", (8, 8)),
                original_size=(8, 8),
                detections=detections,
                timings=Timings(preprocess_s=0.01, inference_s=0.02, postprocess_s=0.03),
            ),
        ),
    )


class FakeAdapter:
    """Spy implementing the facade-adapter contract with deterministic output.

    Every public method records its call arguments so tests can assert
    delegation. ``predict`` always returns a fresh 3-detection collection
    (immutability of ``PredictionCollection`` means filtering never mutates
    prior returns).
    """

    def __init__(
        self,
        *,
        inference: InferenceConfig | None = None,
        is_loaded: bool = True,
        metadata: DatasetMetadata | None = None,
    ) -> None:
        self._model: object | None = object() if is_loaded else None
        self.box_mode: str = "hbb"
        self.metadata: DatasetMetadata = metadata or make_metadata()
        self.loaded: LoadedAppConfig = make_loaded(inference or InferenceConfig())
        self.predict_calls: list[tuple[object, int | None]] = []
        self.load_calls: list[tuple[str | Path | None, bool]] = []
        self.train_called: bool = False
        self.evaluate_calls: list[str | Path | None] = []
        self.export_calls: list[tuple[str | Path | None, str, str | Path | None]] = []

    def load(
        self,
        checkpoint: str | Path | None = None,
        prefer_ema: bool = True,
    ) -> None:
        self.load_calls.append((checkpoint, prefer_ema))
        if self._model is None:
            self._model = object()

    def predict(
        self,
        source: object,
        *,
        batch_size: int | None = None,
    ) -> PredictionCollection:
        self.predict_calls.append((source, batch_size))
        return make_collection()

    def train(self) -> None:
        self.train_called = True

    def evaluate(self, checkpoint: str | Path | None = None) -> None:
        self.evaluate_calls.append(checkpoint)

    def export(
        self,
        checkpoint: str | Path,
        format: str,
        output: str | Path,
    ) -> Path:
        """Spy mirroring DeimDetectionAdapter.export — always raises ExportError.

        Records the call so CLI tests can assert delegation, then raises the
        same ExportError the real adapter raises in v1.
        """
        self.export_calls.append((checkpoint, format, output))
        raise ExportError(
            "No export format is enabled in the first application-layer version"
        )


# ---------------------------------------------------------------------------
# Fixtures — registered in conftest.py via ``pytest_plugins``.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_adapter() -> FakeAdapter:
    """A loaded FakeAdapter with default inference config and 3-class metadata."""
    return FakeAdapter()


@pytest.fixture
def unloaded_fake_adapter() -> FakeAdapter:
    """A FakeAdapter whose _model is None — predict() must raise InferenceBackendError."""
    return FakeAdapter(is_loaded=False)
