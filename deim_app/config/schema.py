"""Frozen, typed application configuration schema.

These dataclasses model ONLY the six public sections a user may override
(``project``, ``runtime``, ``data``, ``train``, ``evaluation``, ``inference``).
They never carry the trusted algorithm content (``DEIMTransformer``, optimizer,
datloader factory, ...) — that stays in ``LoadedAppConfig.engine_base`` for
downstream mapping.

All dataclasses are ``frozen=True`` (immutable, hashable) and ``slots=True``
(memory-compact). ``AppConfig.from_mapping`` is the single construction entry
point: it validates types, enums, and ranges, raising ``AppConfigError`` on any
violation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from deim_app.errors import AppConfigError

__all__ = [
    "AppConfig",
    "DataConfig",
    "EarlyStoppingConfig",
    "EvaluationConfig",
    "InferenceConfig",
    "ProjectConfig",
    "RuntimeConfig",
    "TrainConfig",
]


# ---------------------------------------------------------------------------
# Enum-like allowed-value sets
# ---------------------------------------------------------------------------

DATA_FORMATS: frozenset[str] = frozenset({"COCO", "DOTA", "YOLO-OBB"})
CACHE_MODES: frozenset[str] = frozenset({"none", "disk", "ram"})


# ---------------------------------------------------------------------------
# Primitive type/range validators
# ---------------------------------------------------------------------------

def _expect_int(value: object, name: str) -> int:
    """Return ``value`` as int, rejecting bools (which are int subclasses)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise AppConfigError(
            f"{name} must be an integer, got {type(value).__name__} ({value!r})"
        )
    return value


def _expect_pos_int(value: object, name: str) -> int:
    n = _expect_int(value, name)
    if n <= 0:
        raise AppConfigError(f"{name} must be a positive integer, got {n}")
    return n


def _expect_nonneg_int(value: object, name: str) -> int:
    n = _expect_int(value, name)
    if n < 0:
        raise AppConfigError(f"{name} must be a non-negative integer, got {n}")
    return n


def _expect_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise AppConfigError(f"{name} must be a number, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    raise AppConfigError(
        f"{name} must be a number, got {type(value).__name__} ({value!r})"
    )


def _expect_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise AppConfigError(
            f"{name} must be a string, got {type(value).__name__} ({value!r})"
        )
    return value


def _expect_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise AppConfigError(
            f"{name} must be a bool, got {type(value).__name__} ({value!r})"
        )
    return value


def _expect_str_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise AppConfigError(f"{name} must be a list of strings, got {type(value).__name__}")
    return tuple(_expect_str(s, f"{name}[i]") for s in value)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ProjectConfig:
    name: str = "deim-app"
    output_dir: str = "outputs"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    input_size: tuple[int, int] = (640, 640)
    seed: int = 42


@dataclass(frozen=True, slots=True)
class DataConfig:
    format: str = "COCO"
    train_images: str = ""
    train_annotations: str = ""
    val_images: str = ""
    val_annotations: str = ""
    classes_file: str | None = None
    num_workers: int = 4
    cache_images: str = "none"


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 10


@dataclass(frozen=True, slots=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1.0e-4
    device: str = "cuda"
    amp: bool = True
    pretrained: str | None = None
    resume: str | None = None
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    batch_size: int = 8
    device: str = "cuda"


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    checkpoint: Path | None = None
    device: str = "cuda"
    batch_size: int = 1
    score_threshold: float = 0.25
    top_k: int = 300
    class_filter: tuple[str, ...] | None = None
    output_formats: tuple[str, ...] = ("json", "visualization")


# ---------------------------------------------------------------------------
# Section builders (validate then construct)
# ---------------------------------------------------------------------------

def _build_project(d: Mapping[str, object]) -> ProjectConfig:
    return ProjectConfig(
        name=_expect_str(d.get("name", "deim-app"), "project.name"),
        output_dir=_expect_str(d.get("output_dir", "outputs"), "project.output_dir"),
    )


def _build_runtime(d: Mapping[str, object]) -> RuntimeConfig:
    raw_size = d.get("input_size", (640, 640))
    if not isinstance(raw_size, (list, tuple)):
        raise AppConfigError(
            f"runtime.input_size must be a list/tuple of exactly two positive "
            f"integers, got {type(raw_size).__name__} ({raw_size!r})"
        )
    if len(raw_size) != 2:
        raise AppConfigError(
            f"runtime.input_size must have exactly two elements, got {len(raw_size)}"
        )
    size = (
        _expect_pos_int(raw_size[0], "runtime.input_size[0]"),
        _expect_pos_int(raw_size[1], "runtime.input_size[1]"),
    )
    return RuntimeConfig(
        input_size=size,
        seed=_expect_int(d.get("seed", 42), "runtime.seed"),
    )


def _build_data(d: Mapping[str, object]) -> DataConfig:
    fmt = d.get("format", "COCO")
    if not isinstance(fmt, str) or fmt not in DATA_FORMATS:
        raise AppConfigError(
            f"data.format must be one of {sorted(DATA_FORMATS)}, got {fmt!r}"
        )
    cache = d.get("cache_images", "none")
    if not isinstance(cache, str) or cache not in CACHE_MODES:
        raise AppConfigError(
            f"data.cache_images must be one of {sorted(CACHE_MODES)}, got {cache!r}"
        )
    if fmt == "COCO" and cache != "none":
        raise AppConfigError(
            f"data.cache_images for HBB (COCO) must be 'none'; "
            f"'{cache}' requires a dataset with a cache contract (DOTA/YOLO-OBB)"
        )
    classes_file = d.get("classes_file")
    return DataConfig(
        format=fmt,
        train_images=_expect_str(d.get("train_images", ""), "data.train_images"),
        train_annotations=_expect_str(d.get("train_annotations", ""), "data.train_annotations"),
        val_images=_expect_str(d.get("val_images", ""), "data.val_images"),
        val_annotations=_expect_str(d.get("val_annotations", ""), "data.val_annotations"),
        classes_file=(
            _expect_str(classes_file, "data.classes_file")
            if classes_file is not None
            else None
        ),
        num_workers=_expect_nonneg_int(d.get("num_workers", 4), "data.num_workers"),
        cache_images=cache,
    )


def _build_early_stopping(d: Mapping[str, object]) -> EarlyStoppingConfig:
    return EarlyStoppingConfig(
        enabled=_expect_bool(d.get("enabled", False), "train.early_stopping.enabled"),
        patience=_expect_pos_int(d.get("patience", 10), "train.early_stopping.patience"),
    )


def _build_train(d: Mapping[str, object]) -> TrainConfig:
    pretrained = d.get("pretrained")
    resume = d.get("resume")
    if pretrained is not None and resume is not None:
        raise AppConfigError(
            "train.pretrained and train.resume are mutually exclusive; "
            "specify at most one"
        )
    es_raw = d.get("early_stopping", {})
    if not isinstance(es_raw, Mapping):
        raise AppConfigError(
            f"train.early_stopping must be a mapping, got {type(es_raw).__name__}"
        )
    return TrainConfig(
        epochs=_expect_pos_int(d.get("epochs", 100), "train.epochs"),
        batch_size=_expect_pos_int(d.get("batch_size", 8), "train.batch_size"),
        learning_rate=_expect_float(d.get("learning_rate", 1.0e-4), "train.learning_rate"),
        device=_expect_str(d.get("device", "cuda"), "train.device"),
        amp=_expect_bool(d.get("amp", True), "train.amp"),
        pretrained=(
            _expect_str(pretrained, "train.pretrained")
            if pretrained is not None
            else None
        ),
        resume=(
            _expect_str(resume, "train.resume")
            if resume is not None
            else None
        ),
        early_stopping=_build_early_stopping(es_raw),
    )


def _build_evaluation(d: Mapping[str, object]) -> EvaluationConfig:
    return EvaluationConfig(
        batch_size=_expect_pos_int(d.get("batch_size", 8), "evaluation.batch_size"),
        device=_expect_str(d.get("device", "cuda"), "evaluation.device"),
    )


def _build_inference(d: Mapping[str, object]) -> InferenceConfig:
    ckpt_raw = d.get("checkpoint")
    if ckpt_raw is None:
        checkpoint: Path | None = None
    elif isinstance(ckpt_raw, str):
        checkpoint = Path(ckpt_raw)
    else:
        raise AppConfigError(
            f"inference.checkpoint must be a string path, "
            f"got {type(ckpt_raw).__name__} ({ckpt_raw!r})"
        )
    score = _expect_float(d.get("score_threshold", 0.25), "inference.score_threshold")
    if not 0.0 <= score <= 1.0:
        raise AppConfigError(
            f"inference.score_threshold must be in [0, 1], got {score}"
        )
    cf_raw = d.get("class_filter")
    class_filter = _expect_str_tuple(cf_raw, "inference.class_filter") if cf_raw is not None else None
    of_raw = d.get("output_formats", ("json", "visualization"))
    output_formats = _expect_str_tuple(of_raw, "inference.output_formats")
    return InferenceConfig(
        checkpoint=checkpoint,
        device=_expect_str(d.get("device", "cuda"), "inference.device"),
        batch_size=_expect_pos_int(d.get("batch_size", 1), "inference.batch_size"),
        score_threshold=score,
        top_k=_expect_pos_int(d.get("top_k", 300), "inference.top_k"),
        class_filter=class_filter,
        output_formats=output_formats,
    )


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AppConfig:
    """Frozen root application config holding the six public sections."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    #: Mapping from public section name to its builder. ClassVar so dataclass
    #: does not treat it as a field.
    _BUILDERS: ClassVar[dict[str, Any]] = {
        "project": _build_project,
        "runtime": _build_runtime,
        "data": _build_data,
        "train": _build_train,
        "evaluation": _build_evaluation,
        "inference": _build_inference,
    }

    @classmethod
    def from_mapping(cls, public_merged: Mapping[str, object]) -> "AppConfig":
        """Construct an ``AppConfig`` from the six public sections.

        ``public_merged`` must contain only the public sections (already
        whitelist-validated by the loader). Values are type-checked, enum-
        checked, and range-checked here; any violation raises
        ``AppConfigError``. Missing sections fall back to dataclass defaults.
        """
        kwargs: dict[str, Any] = {}
        for section, builder in cls._BUILDERS.items():
            section_data = public_merged.get(section, {})
            if section_data is None:
                section_data = {}
            if not isinstance(section_data, Mapping):
                raise AppConfigError(
                    f"public section '{section}' must be a mapping, "
                    f"got {type(section_data).__name__}"
                )
            kwargs[section] = builder(section_data)
        return cls(**kwargs)
