"""Application → engine YAML config mapping.

Transforms a :class:`~deim_app.config.loader.LoadedAppConfig` into a
:class:`ResolvedAlgorithmConfig` containing:

  - ``overrides``: a deep copy of ``loaded.engine_base`` with mutations applied
    according to the public-field → engine-YAML mapping table.  This dict is
    designed to be passed as ``YAMLConfig(str(config_path), **overrides)``.
  - ``metadata``: :class:`~deim_app.config.metadata.DatasetMetadata` derived
    from the on-disk annotation files referenced by ``app.data``.

Boundary rule: this module reads ``engine_base`` as a plain dict — it MUST NOT
import ``engine.*``.  All field names (``epoches``, ``tuning``, etc.) are plain
string keys, resolved by reading the existing engine YAMLs.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deim_app.config.loader import LoadedAppConfig
from deim_app.config.metadata import DatasetMetadata, load_coco_metadata, load_obb_metadata
from deim_app.config.schema import AppConfig
from deim_app.errors import AppConfigError

__all__ = ["ResolvedAlgorithmConfig", "resolve_algorithm_config"]


_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


@dataclass(frozen=True, slots=True)
class ResolvedAlgorithmConfig:
    """Result of :func:`resolve_algorithm_config`.

    Attributes:
        config_path: path to the user application YAML (``loaded.source``).
        overrides: deep-copied and mutated engine dict; pass as
            ``YAMLConfig(str(config_path), **overrides)`` downstream.
        metadata: dataset metadata derived from annotation files.
        app: the frozen :class:`~deim_app.config.schema.AppConfig`.
    """

    config_path: Path
    overrides: dict[str, Any]
    metadata: DatasetMetadata
    app: AppConfig


def resolve_algorithm_config(loaded: LoadedAppConfig) -> ResolvedAlgorithmConfig:
    """Map public application fields onto a deep copy of ``engine_base``.

    The deep copy guarantees ``loaded.engine_base`` (and the process-global
    YAML registry) are never mutated.
    """
    overrides = copy.deepcopy(loaded.engine_base)
    app = loaded.app

    _map_project(overrides, app)
    _map_runtime(overrides, app)
    _map_train(overrides, app)
    _map_evaluation(overrides, app)

    metadata = _load_metadata(overrides, app)
    overrides["num_classes"] = metadata.num_classes

    _map_data(overrides, app, metadata.box_mode)

    return ResolvedAlgorithmConfig(
        config_path=loaded.source,
        overrides=overrides,
        metadata=metadata,
        app=app,
    )


# ---------------------------------------------------------------------------
# Section mappers
# ---------------------------------------------------------------------------


def _map_project(overrides: dict[str, Any], app: AppConfig) -> None:
    overrides["output_dir"] = app.project.output_dir


def _map_runtime(overrides: dict[str, Any], app: AppConfig) -> None:
    overrides["seed"] = app.runtime.seed

    h, w = app.runtime.input_size
    size_list = [h, w]
    overrides["eval_spatial_size"] = list(size_list)

    for dl_key in ("train_dataloader", "val_dataloader"):
        dl = overrides.get(dl_key)
        if not isinstance(dl, dict):
            continue
        dataset = dl.get("dataset")
        if isinstance(dataset, dict):
            transforms = dataset.get("transforms")
            if isinstance(transforms, dict):
                # Engine chains get this key from a dataset-YAML sibling; the
                # resolver is the app layer's sibling (inject contract).
                transforms.setdefault("type", "Compose")
                ops = transforms.get("ops")
                if isinstance(ops, list):
                    for op in ops:
                        if isinstance(op, dict) and "size" in op:
                            op["size"] = list(size_list)

    train_dl = overrides.get("train_dataloader")
    if isinstance(train_dl, dict):
        collate_fn = train_dl.get("collate_fn")
        if isinstance(collate_fn, dict) and "base_size" in collate_fn:
            collate_fn["base_size"] = h


def _map_train(overrides: dict[str, Any], app: AppConfig) -> None:
    overrides["epoches"] = app.train.epochs
    overrides["device"] = app.train.device
    overrides["use_amp"] = app.train.amp

    train_dl = overrides.setdefault("train_dataloader", {})
    train_dl["total_batch_size"] = app.train.batch_size

    optimizer = overrides.get("optimizer")
    if isinstance(optimizer, dict):
        optimizer["lr"] = app.train.learning_rate

    if app.train.pretrained is not None:
        overrides["tuning"] = app.train.pretrained
    if app.train.resume is not None:
        overrides["resume"] = app.train.resume

    _map_early_stopping(overrides, app)


def _map_early_stopping(overrides: dict[str, Any], app: AppConfig) -> None:
    preset_es = overrides.get("early_stopping")
    if not isinstance(preset_es, dict):
        preset_es = {}
    else:
        preset_es = dict(preset_es)

    preset_es["enabled"] = app.train.early_stopping.enabled
    preset_es["patience"] = app.train.early_stopping.patience
    overrides["early_stopping"] = preset_es


def _map_evaluation(overrides: dict[str, Any], app: AppConfig) -> None:
    val_dl = overrides.setdefault("val_dataloader", {})
    val_dl["total_batch_size"] = app.evaluation.batch_size


def _map_data(overrides: dict[str, Any], app: AppConfig, box_mode: str) -> None:
    is_obb = box_mode == "obb"

    train_dl = overrides.setdefault("train_dataloader", {})
    val_dl = overrides.setdefault("val_dataloader", {})
    # Preset include chains carry no type: DataLoader; without it create()
    # dies with KeyError '_pymodule'. The resolver owns the assembly.
    train_dl["type"] = "DataLoader"
    val_dl["type"] = "DataLoader"
    train_ds = train_dl.setdefault("dataset", {})
    val_ds = val_dl.setdefault("dataset", {})

    if is_obb:
        dataset_type = "DotaDataset"
        train_ds["type"] = dataset_type
        val_ds["type"] = dataset_type

        train_ds["img_folder"] = app.data.train_images
        train_ds["ann_folder"] = app.data.train_annotations
        val_ds["img_folder"] = app.data.val_images
        val_ds["ann_folder"] = app.data.val_annotations

        if app.data.classes_file:
            train_ds["classes_file"] = app.data.classes_file
            val_ds["classes_file"] = app.data.classes_file

        train_ds["format"] = app.data.format
        val_ds["format"] = app.data.format

        _apply_obb_cache(train_ds, app.data.cache_images, app.data.train_images)
        _apply_obb_cache(val_ds, app.data.cache_images, app.data.val_images)
    else:
        dataset_type = "CocoDetection"
        train_ds["type"] = dataset_type
        val_ds["type"] = dataset_type

        train_ds["img_folder"] = app.data.train_images
        train_ds["ann_file"] = app.data.train_annotations
        val_ds["img_folder"] = app.data.val_images
        val_ds["ann_file"] = app.data.val_annotations

    train_dl["num_workers"] = app.data.num_workers
    val_dl["num_workers"] = app.data.num_workers


def _apply_obb_cache(dataset: dict[str, Any], mode: str, img_dir: str) -> None:
    if mode == "none":
        dataset["cache_images"] = "none"
        dataset["cache_ram"] = 0
    elif mode == "disk":
        dataset["cache_images"] = "disk"
        dataset["cache_ram"] = 0
    elif mode == "ram":
        dataset["cache_images"] = "none"
        dataset["cache_ram"] = _count_images(img_dir)
    else:
        raise AppConfigError(
            f"Unexpected cache_images mode '{mode}' for OBB dataset"
        )


def _count_images(img_dir: str) -> int:
    path = Path(img_dir)
    if not path.is_dir():
        raise AppConfigError(
            f"data image directory '{img_dir}' does not exist; "
            f"cannot count images for cache_images=ram"
        )
    count = 0
    for entry in os.listdir(path):
        ext = os.path.splitext(entry)[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def _load_metadata(overrides: dict[str, Any], app: AppConfig) -> DatasetMetadata:
    fmt = app.data.format

    if fmt in ("DOTA", "YOLO-OBB"):
        if not app.data.classes_file:
            raise AppConfigError(
                "data.classes_file is required for OBB (DOTA/YOLO-OBB) format"
            )
        return load_obb_metadata(Path(app.data.classes_file))

    if fmt == "COCO":
        ann_path = app.data.train_annotations
        if not ann_path:
            raise AppConfigError(
                "data.train_annotations is required for COCO format metadata loading"
            )
        remap_raw = overrides.get("remap_mscoco_category", None)
        metadata = load_coco_metadata(
            Path(ann_path), remap_mscoco_category=remap_raw
        )
        # Propagate the actual remap decision to overrides so the
        # CocoDetection dataset class (which reads remap_mscoco_category via
        # __share__) gets the correct flag.
        remap_actual = (
            set(metadata.output_names_by_id.keys())
            != set(metadata.class_names_by_label.keys())
        )
        overrides["remap_mscoco_category"] = remap_actual
        return metadata

    raise AppConfigError(f"Unsupported data format '{fmt}' for metadata loading")
