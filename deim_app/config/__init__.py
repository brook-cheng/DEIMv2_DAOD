"""Typed application configuration schema and YAML loader.

Exposes the frozen dataclass schema and the two-stage ``load_app_config``
loader that separates trusted engine content (returned as ``engine_base``)
from the validated public application object (``AppConfig``).
"""

from deim_app.config.loader import LoadedAppConfig, load_app_config
from deim_app.config.schema import (
    AppConfig,
    DataConfig,
    EarlyStoppingConfig,
    EvaluationConfig,
    InferenceConfig,
    ProjectConfig,
    RuntimeConfig,
    TrainConfig,
)

__all__ = [
    "AppConfig",
    "DataConfig",
    "EarlyStoppingConfig",
    "EvaluationConfig",
    "InferenceConfig",
    "LoadedAppConfig",
    "ProjectConfig",
    "RuntimeConfig",
    "TrainConfig",
    "load_app_config",
]
