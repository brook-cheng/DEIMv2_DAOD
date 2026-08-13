"""DEIMv2 application layer: user-facing API over the engine.

Public surface:
    - :class:`DetectionModel` — the application facade (pure delegation).
    - Prediction types: :class:`HBBDetection`, :class:`OBBDetection`,
      :class:`ImagePrediction`, :class:`PredictionCollection`.
    - Stable application error types.

Import order matters for cycle avoidance: ``deim_app.api`` imports
``DeimDetectionAdapter`` from ``deim_app.adapters.deim``, which imports
``engine.core``. This is fine because nothing in this module's top-level
imports triggers a cycle — the adapter's ``deim_app.config`` references are
deferred to function bodies (local imports) or ``TYPE_CHECKING``.
"""

from deim_app.api import DetectionModel
from deim_app.errors import (
    AdapterConfigurationError,
    AppConfigError,
    CheckpointCompatibilityError,
    ExportError,
    InferenceBackendError,
    InputSourceError,
)
from deim_app.predictions import (
    HBBDetection,
    ImagePrediction,
    OBBDetection,
    PredictionCollection,
)

__all__ = [
    "AdapterConfigurationError",
    "AppConfigError",
    "CheckpointCompatibilityError",
    "DetectionModel",
    "ExportError",
    "HBBDetection",
    "ImagePrediction",
    "InferenceBackendError",
    "InputSourceError",
    "OBBDetection",
    "PredictionCollection",
]
