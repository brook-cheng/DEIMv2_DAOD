"""The ``DetectionAdapter`` protocol — the stable contract the application
facade (Task 7) type-hints against.

Only ``deim_app/adapters/`` may import ``engine.*``. By routing facade code
through this Protocol, the facade stays decoupled from the concrete
``DeimDetectionAdapter`` (and any future adapter for other architectures).

``predict``, ``train``, ``evaluate``, and ``export`` are part of the protocol
signature so Task 7 can type-hint them, but the concrete DEIM adapter stubs
them out in Task 5: Task 6 implements ``predict`` and Task 8 implements
``train`` / ``evaluate`` / ``export``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    # These are only needed for Protocol annotations; importing them at module
    # level would create a cycle (deim_app.config.loader imports
    # deim_app.adapters._engine_yaml, which triggers this package's __init__).
    from deim_app.config import LoadedAppConfig, ResolvedAlgorithmConfig
    from deim_app.predictions.collection import PredictionCollection

__all__ = ["DetectionAdapter"]


@runtime_checkable
class DetectionAdapter(Protocol):
    """Structural contract every detection adapter must satisfy.

    Adapters own the lifetime of engine objects (model, postprocessor) and
    normalize engine outputs into the immutable
    :class:`~deim_app.predictions.collection.PredictionCollection`. A facade
    composes one adapter with config loaders and writers; it never touches
    engine internals directly.
    """

    def resolve_config(
        self, loaded: LoadedAppConfig | None = None
    ) -> ResolvedAlgorithmConfig:
        """Re-derive the resolved algorithm config.

        When ``loaded`` is ``None``, re-resolves from the adapter's currently
        loaded ``LoadedAppConfig``. Otherwise resolves the provided config.
        """
        ...

    def load(
        self,
        checkpoint: str | Path | None = None,
        prefer_ema: bool = True,
    ) -> None:
        """Build engine objects and (optionally) load a checkpoint.

        When ``checkpoint`` is ``None`` the model is left at its default
        initialization (useful for ONNX export of an untrained skeleton).
        """
        ...

    def predict(
        self,
        source: Any,
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> PredictionCollection:
        """Run inference over ``source`` and return normalized predictions.

        Implemented in Task 6.
        """
        ...

    def train(self) -> None:
        """Launch a training run. Implemented in Task 8."""
        ...

    def evaluate(self, checkpoint: str | Path | None = None) -> None:
        """Run validation / evaluation. Implemented in Task 8."""
        ...

    def export(
        self,
        checkpoint: str | Path,
        format: str,
        output: str | Path,
    ) -> Path:
        """Export the model to ``format`` (e.g. ONNX). Implemented in Task 8."""
        ...
