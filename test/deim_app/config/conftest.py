"""Shared fixtures for deim_app.config loader + mapping tests.

These helpers write synthetic application-base YAMLs into ``tmp_path`` so the
loader can be exercised end-to-end without depending on the real
``configs/app/base/{hbb,obb}_app.yml`` files (which Task 3 creates).

Task 3 additions:
  - ``make_engine_base`` / ``make_obb_engine_base`` — synthetic merged engine
    dicts that look like a real preset output (optimizer param groups, resize
    ops, collate_fn, scheduler, etc.) so mapping tests can assert exact paths
    without loading the real preset YAMLs.
  - ``write_classes_file`` / ``write_coco_json`` — metadata fixtures.
  - ``_APPROVED_BASE_NAMES_PATCH`` — autouse fixture that extends the approved
    basename set with the synthetic names used by loader tests
    (``base.yml``, ``base_a.yml``, ``base_b.yml``) so the tightened include
    check does not reject them.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from pathlib import Path


def write_yaml(path: Path, data: dict[str, Any]) -> Path:
    """Write ``data`` as YAML to ``path`` (creating parent dirs) and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Autouse: permit synthetic base basenames in the tightened include check
# ---------------------------------------------------------------------------

#: Synthetic base-file names used by Task 2 loader tests.  After Task 3
#: tightens ``validate_single_application_base`` to require an approved
#: basename, this autouse fixture patches the approved set so these names
#: continue to load from ``tmp_path``.
_SYNTHETIC_BASE_NAMES = ("base.yml", "base_a.yml", "base_b.yml")


@pytest.fixture(autouse=True)
def _patch_approved_base_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from deim_app.config import loader as _loader

    patched = tuple(_loader._APPROVED_BASE_NAMES) + _SYNTHETIC_BASE_NAMES
    monkeypatch.setattr(_loader, "_APPROVED_BASE_NAMES", patched)


def valid_base_dict(device: str = "cuda") -> dict[str, Any]:
    """A trusted HBB (COCO) application base config with all six public sections.

    Mirrors what ``configs/app/base/hbb_app.yml`` will look like once Task 3
    creates it. Algorithm sections (``DEIMTransformer`` etc.) are intentionally
    omitted; ``engine_base`` still round-trips them when present.
    """
    return {
        "project": {"name": "test-app", "output_dir": "outputs/test"},
        "runtime": {"input_size": [640, 640], "seed": 42},
        "data": {
            "format": "COCO",
            "train_images": "data/train/images",
            "train_annotations": "data/train/labels.json",
            "val_images": "data/val/images",
            "val_annotations": "data/val/labels.json",
            "num_workers": 2,
            "cache_images": "none",
        },
        "train": {
            "epochs": 12,
            "batch_size": 4,
            "learning_rate": 1.0e-4,
            "device": device,
            "amp": True,
            "early_stopping": {"enabled": False, "patience": 5},
        },
        "evaluation": {"batch_size": 4, "device": device},
        "inference": {
            "checkpoint": None,
            "device": device,
            "batch_size": 1,
            "score_threshold": 0.25,
            "top_k": 300,
            "output_formats": ["json"],
        },
    }


def valid_obb_base_dict(
    device: str = "cuda",
    fmt: str = "DOTA",
    cache_images: str = "none",
) -> dict[str, Any]:
    """A trusted OBB application base config (DOTA or YOLO-OBB).

    Adds ``classes_file`` (OBB-only) and exercises every cache_images value.
    """
    base = valid_base_dict(device)
    base["data"] = {
        "format": fmt,
        "train_images": "data/train/images",
        "train_annotations": "data/train/labels",
        "val_images": "data/val/images",
        "val_annotations": "data/val/labels",
        "classes_file": "data/classes.txt",
        "num_workers": 2,
        "cache_images": cache_images,
    }
    return base


def valid_user_dict(base_name: str = "base.yml") -> dict[str, Any]:
    """A minimal user YAML that includes the synthetic base and overrides nothing."""
    return {"__include__": [base_name]}


def write_base_and_user(
    tmp_path: Path,
    base: dict[str, Any] | None = None,
    user: dict[str, Any] | None = None,
    base_name: str = "base.yml",
    user_name: str = "user.yml",
) -> Path:
    """Write both files and return the path to the user YAML."""
    write_yaml(tmp_path / base_name, base if base is not None else valid_base_dict())
    return write_yaml(
        tmp_path / user_name,
        user if user is not None else valid_user_dict(base_name),
    )


# ---------------------------------------------------------------------------
# Synthetic engine_base builders for mapping tests
# ---------------------------------------------------------------------------


def _hbb_resize_ops() -> list[dict[str, Any]]:
    return [
        {"type": "RandomPhotometricDistort", "p": 0.5},
        {"type": "RandomZoomOut", "fill": 0},
        {"type": "RandomIoUCrop", "p": 0.8},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "RandomHorizontalFlip"},
        {"type": "Resize", "size": [640, 640]},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
    ]


def _obb_resize_ops() -> list[dict[str, Any]]:
    return [
        {"type": "Mosaic", "output_size": 320},
        {"type": "RandomPhotometricDistort", "p": 0.5},
        {"type": "OBBZoomOut", "fill": 0},
        {"type": "OBBIoUCrop", "p": 0.8},
        {"type": "OBBSanitize", "min_size": 1},
        {"type": "OBBFlip"},
        {"type": "OBBResize", "size": [640, 640]},
        {"type": "OBBSanitize", "min_size": 1},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        {"type": "OBBConvertBoxes", "normalize": True, "img_size": [640, 640]},
        {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    ]


def make_engine_base(box_mode: str = "hbb") -> dict[str, Any]:
    """A synthetic merged engine dict that mirrors a real preset's structure.

    Contains the exact nested keys the mapping function mutates:
    ``optimizer`` (with ``params`` list), ``train_dataloader`` /
    ``val_dataloader`` (each with ``dataset.transforms.ops`` resize entries and
    ``collate_fn.base_size``), ``eval_spatial_size``, ``epoches``, ``use_amp``,
    ``num_classes``, ``early_stopping``, etc.

    ``box_mode`` selects HBB (``Resize`` ops, ``CocoDetection``) vs OBB
    (``OBBResize`` ops, ``DotaDataset``).
    """
    if box_mode == "hbb":
        resize_type = "Resize"
        train_ops = _hbb_resize_ops()
        val_ops: list[dict[str, Any]] = [
            {"type": "Resize", "size": [640, 640]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        ]
    else:
        resize_type = "OBBResize"
        train_ops = _obb_resize_ops()
        val_ops = [
            {"type": "OBBResize", "size": [640, 640]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
            {"type": "OBBConvertBoxes", "normalize": True, "img_size": [640, 640]},
            {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        ]

    return {
        "task": "detection",
        "model": "DEIM",
        "criterion": "DEIMCriterion",
        "postprocessor": "PostProcessor",
        "use_focal_loss": True,
        "eval_spatial_size": [640, 640],
        "num_classes": 80,
        "epoches": 72,
        "use_amp": True,
        "use_ema": True,
        "seed": None,
        "output_dir": "./logs",
        "lrsheduler": "flatcosine",
        "lr_gamma": 0.5,
        "warmup_iter": 2000,
        "flat_epoch": 36,
        "no_aug_epoch": 8,
        "optimizer": {
            "type": "AdamW",
            "params": [
                {
                    "params": "^(?=.*.dinov3)(?!.*(?:norm|bn|bias)).*$",
                    "lr": 0.0000125,
                },
                {
                    "params": "^(?=.*.dinov3)(?=.*(?:norm|bn|bias)).*$",
                    "lr": 0.0000125,
                    "weight_decay": 0.0,
                },
                {
                    "params": "^(?=.*(?:sta|encoder|decoder))(?=.*(?:norm|bn|bias)).*$",
                    "weight_decay": 0.0,
                },
            ],
            "lr": 0.0005,
            "betas": [0.9, 0.999],
            "weight_decay": 0.000125,
        },
        "train_dataloader": {
            "total_batch_size": 4,
            "num_workers": 4,
            "dataset": {
                "type": "CocoDetection" if box_mode == "hbb" else "DotaDataset",
                "transforms": {"type": "Compose", "ops": train_ops},
            },
            "collate_fn": {
                "type": "BatchImageCollateFunction",
                "base_size": 640,
                "base_size_repeat": 3,
                "stop_epoch": 60,
            },
        },
        "val_dataloader": {
            "total_batch_size": 4,
            "num_workers": 4,
            "dataset": {
                "type": "CocoDetection" if box_mode == "hbb" else "DotaDataset",
                "transforms": {"type": "Compose", "ops": val_ops},
            },
        },
        "DEIMTransformer": {"box_mode": box_mode},
        "PostProcessor": {"box_mode": box_mode},
        "DEIMCriterion": {
            "box_mode": box_mode,
            "matcher": {"matcher_change_epoch": 45},
        },
        "early_stopping": {
            "enabled": False,
            "metric": "mAP50_95",
            "mode": "max",
            "min_epochs": 100,
            "patience": 40,
            "min_delta": 0.0001,
            "restore_best": True,
        },
    }


# ---------------------------------------------------------------------------
# Metadata fixtures
# ---------------------------------------------------------------------------


def write_classes_file(path: Path, names: list[str]) -> Path:
    """Write a classes.txt with one class name per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return path


def write_coco_json(
    path: Path,
    categories: list[dict[str, Any]],
    images: list[dict[str, Any]] | None = None,
    annotations: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a minimal COCO-format JSON with the given categories list."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "categories": categories,
        "images": images or [],
        "annotations": annotations or [],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def mscoco_categories() -> list[dict[str, Any]]:
    """Return all 80 standard MS COCO categories with their canonical 1..90 IDs.

    Mirrors the ``categories`` list in COCO ``instances_val2017.json``.
    Used by tests that need auto-detection to recognise the MS COCO standard.
    """
    from deim_app.config.metadata import _MSCOCO_CATEGORY2NAME

    return [
        {"id": cat_id, "name": name}
        for cat_id, name in sorted(_MSCOCO_CATEGORY2NAME.items())
    ]
