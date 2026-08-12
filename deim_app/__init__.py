"""DEIMv2 application layer: user-facing API over the engine.

Currently exports only the stable application error types. Later tasks add
``DetectionModel``, adapter exports, and prediction types.
"""

from deim_app.errors import (
    AdapterConfigurationError,
    AppConfigError,
    CheckpointCompatibilityError,
    ExportError,
    InferenceBackendError,
    InputSourceError,
)

__all__ = [
    "AdapterConfigurationError",
    "AppConfigError",
    "CheckpointCompatibilityError",
    "ExportError",
    "InferenceBackendError",
    "InputSourceError",
]
