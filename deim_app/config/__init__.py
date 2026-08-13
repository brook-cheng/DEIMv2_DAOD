"""Typed application configuration schema and YAML loader.

Exposes the frozen dataclass schema and the two-stage ``load_app_config``
loader that separates trusted engine content (returned as ``engine_base``)
from the validated public application object (``AppConfig``).

Task 3 additions: dataset metadata loaders (``DatasetMetadata``,
``load_obb_metadata``, ``load_coco_metadata``) and the application→engine
mapping function (``resolve_algorithm_config``, ``ResolvedAlgorithmConfig``).
"""

from deim_app.config.loader import LoadedAppConfig, load_app_config
from deim_app.config.mapping import ResolvedAlgorithmConfig, resolve_algorithm_config
from deim_app.config.metadata import (
    DatasetMetadata,
    load_coco_metadata,
    load_obb_metadata,
)
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
    "DatasetMetadata",
    "EarlyStoppingConfig",
    "EvaluationConfig",
    "InferenceConfig",
    "LoadedAppConfig",
    "ProjectConfig",
    "ResolvedAlgorithmConfig",
    "RuntimeConfig",
    "TrainConfig",
    "load_app_config",
    "load_coco_metadata",
    "load_obb_metadata",
    "resolve_algorithm_config",
]
