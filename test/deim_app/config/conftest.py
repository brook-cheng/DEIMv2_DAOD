"""Shared fixtures for deim_app.config loader tests.

These helpers write synthetic application-base YAMLs into ``tmp_path`` so the
loader can be exercised end-to-end without depending on the real
``configs/app/base/{hbb,obb}_app.yml`` files (which Task 3 creates).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_yaml(path: Path, data: dict[str, Any]) -> Path:
    """Write ``data`` as YAML to ``path`` (creating parent dirs) and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


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
