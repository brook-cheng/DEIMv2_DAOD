"""Application facade: a thin delegation layer over a detection adapter.

``DetectionModel`` is the public entry point for library users. It selects a
concrete adapter (currently only :class:`~deim_app.adapters.deim.DeimDetectionAdapter`),
wires up config loading, and delegates every operation. The facade performs
NO tensor operations and imports NO ``engine.*`` modules — it is pure
orchestration. The dependency-boundary test
(:mod:`test.deim_app.test_dependency_boundaries`) enforces this at the AST
level.

``predict`` returns the FULL unfiltered collection; ``predict_filtered``
resolves defaults from the loaded config's ``inference`` section and returns
an immutable filtered view (score threshold → class filter → top-k). The
underlying full collection is never mutated because
:class:`~deim_app.predictions.collection.PredictionCollection` filtering is
non-destructive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from deim_app.adapters.deim import DeimDetectionAdapter
from deim_app.errors import AppConfigError, InferenceBackendError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pathlib import Path

    from deim_app.config import LoadedAppConfig
    from deim_app.config.metadata import DatasetMetadata
    from deim_app.inference.inputs import InputSource
    from deim_app.predictions.collection import PredictionCollection

__all__ = ["DetectionModel"]


@runtime_checkable
class _FacadeAdapter(Protocol):
    """Structural shape ``DetectionModel`` requires of its adapter.

    The public :class:`~deim_app.adapters.base.DetectionAdapter` protocol
    declares only methods. The facade additionally reads four adapter
    attributes — ``loaded`` (the typed application config), ``metadata``
    (dataset metadata), ``box_mode``, and the ``_model`` sentinel used to
    gate ``predict``. This protocol captures that full contract so the
    facade stays type-safe without ``Any`` or ``cast``.

    ``DeimDetectionAdapter`` satisfies this protocol structurally; test
    doubles (:class:`~test.deim_app.conftest.FakeAdapter`) do too.
    """

    _model: object | None
    loaded: LoadedAppConfig
    metadata: DatasetMetadata
    box_mode: str

    def load(
        self,
        checkpoint: str | Path | None = ...,
        prefer_ema: bool = ...,
    ) -> None: ...

    def predict(
        self,
        source: InputSource,
        *,
        batch_size: int | None = ...,
    ) -> PredictionCollection: ...

    def train(self) -> None: ...

    def evaluate(self, checkpoint: str | Path | None = ...) -> None: ...


class DetectionModel:
    """Application facade over a detection adapter.

    Pure delegation — no tensor ops, no engine imports, no geometry. The
    facade selects the adapter based on the resolved config and delegates
    every operation to it.

    Typical usage::

        model = DetectionModel.from_config("app.yml").load("checkpoint.pt")
        results = model.predict_filtered("images/")

    For testing or advanced control, pass a pre-built adapter directly::

        model = DetectionModel(my_adapter)
    """

    def __init__(self, adapter: _FacadeAdapter) -> None:
        self._adapter = adapter

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        **cli_overrides: object,
    ) -> DetectionModel:
        """Build a ``DetectionModel`` from an application YAML config.

        Args:
            config_path: Path to the user application YAML.
            cli_overrides: Keyword arguments matching the public whitelist
                (e.g. ``device='cuda:0'``, ``train={'learning_rate': 1e-3}``).
                The loader validates them against the same whitelist as YAML.

        Returns:
            A ``DetectionModel`` wrapping a ``DeimDetectionAdapter``. The
            model is NOT loaded yet — call :meth:`load` next.
        """
        overrides = cli_overrides or None
        adapter = DeimDetectionAdapter.from_config(config_path, cli_overrides=overrides)
        return cls(adapter)

    def load(
        self,
        checkpoint: str | Path | None = None,
        prefer_ema: bool = True,
    ) -> DetectionModel:
        """Load model weights; returns ``self`` for chaining.

        Delegates to the adapter's ``load``. When ``checkpoint`` is ``None``
        the model is left at its default initialization (useful for ONNX
        export of an untrained skeleton).
        """
        self._adapter.load(checkpoint, prefer_ema=prefer_ema)
        return self

    def predict(
        self,
        source: InputSource,
        *,
        batch_size: int | None = None,
    ) -> PredictionCollection:
        """Run inference and return the FULL prediction collection.

        No filtering is applied — use :meth:`predict_filtered` for score
        thresholding, top-k, or class filtering.

        Raises:
            InferenceBackendError: if the model has not been loaded via
                :meth:`load`.
        """
        if not self._is_loaded():
            raise InferenceBackendError(
                "Model is not loaded. Call load(checkpoint) first."
            )
        return self._adapter.predict(source, batch_size=batch_size)

    def predict_filtered(
        self,
        source: InputSource,
        *,
        score_threshold: float | None = None,
        top_k: int | None = None,
        class_filter: Iterable[str] | None = None,
        batch_size: int | None = None,
    ) -> PredictionCollection:
        """Run inference then return an immutable filtered view.

        Defaults come from the loaded config's ``inference`` section when
        arguments are omitted (``score_threshold``, ``top_k``,
        ``class_filter``). The pipeline is: ``filter(score_threshold,
        class_filter)`` → ``top_k(top_k)``.

        Class-filter names are validated against the dataset metadata before
        any inference runs; unknown names raise :class:`AppConfigError`.

        Raises:
            AppConfigError: if ``class_filter`` (explicit or from config)
                contains a name not in the dataset metadata.
            InferenceBackendError: if the model has not been loaded.
        """
        inference = self._adapter.loaded.app.inference
        resolved_score = (
            inference.score_threshold if score_threshold is None else score_threshold
        )
        resolved_top_k = inference.top_k if top_k is None else top_k

        if class_filter is None and inference.class_filter:
            class_filter = inference.class_filter

        resolved_class_filter = self._validate_class_filter(class_filter)

        full = self.predict(source, batch_size=batch_size)
        return full.filter(
            score_threshold=resolved_score,
            class_names=resolved_class_filter,
        ).top_k(resolved_top_k)

    def train(self) -> None:
        """Delegate to the adapter (Task 8 implements the actual training)."""
        self._adapter.train()

    def evaluate(self, checkpoint: str | Path | None = None) -> None:
        """Delegate to the adapter (Task 8 implements the actual evaluation)."""
        self._adapter.evaluate(checkpoint)

    @property
    def box_mode(self) -> str:
        """The box mode (``'hbb'`` or ``'obb'``) of the underlying adapter."""
        return self._adapter.box_mode

    @property
    def metadata(self) -> DatasetMetadata:
        """Dataset metadata (class names, num_classes, box_mode) from the adapter."""
        return self._adapter.metadata

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_loaded(self) -> bool:
        """Check whether the adapter has built model + postprocessor."""
        return getattr(self._adapter, "_model", None) is not None

    def _validate_class_filter(
        self,
        class_filter: Iterable[str] | None,
    ) -> tuple[str, ...] | None:
        """Validate class-filter names against dataset metadata.

        Returns a tuple suitable for
        :meth:`PredictionCollection.filter(class_names=...)`, or ``None``
        when no class filtering is requested (empty or absent filter).

        Raises:
            AppConfigError: if any name is not found among the dataset's
                known class names (``class_names_by_label`` values from the
                resolved metadata).
        """
        if class_filter is None:
            return None
        names = tuple(class_filter)
        if not names:
            return None

        metadata = self._adapter.metadata
        # Per Task 6 contract: deployed inference resolves names via
        # class_names_by_label only. output_names_by_id is for non-deploy
        # evaluation paths and is never produced by the inference backend,
        # so filtering by those names would silently match nothing.
        known = set(metadata.class_names_by_label.values())
        unknown = [n for n in names if n not in known]
        if unknown:
            raise AppConfigError(
                "class_filter contains unknown class name(s): "
                + ", ".join(unknown)
                + f". Known classes (from metadata.class_names_by_label): "
                + f"{sorted(known)}"
            )
        return names
