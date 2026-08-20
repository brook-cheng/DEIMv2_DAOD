"""Tests for Task 3: dataset metadata loaders + application→engine mapping.

Covers (per the task brief):
  - ``load_obb_metadata`` reads classes.txt correctly.
  - ``load_coco_metadata`` rejects non-contiguous IDs (remap=False) and accepts
    MS COCO 1..90 IDs (remap=True).
  - Every field in the mapping table maps to the correct YAML path.
  - ``runtime.input_size`` updates every resize op + collate_fn + eval size.
  - ``train.learning_rate`` changes ONLY ``optimizer.lr``; param-group LRs are
    preserved; ``engine_base`` is not mutated.
  - HBB ``cache_images: none`` is silent; OBB modes map correctly.
  - ``num_classes`` is derived from metadata and propagated to top-level.
  - Loading each example YAML resolves or fails with a precise missing-path
    ``AppConfigError``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from deim_app.config import load_app_config
from deim_app.config.mapping import (
    ResolvedAlgorithmConfig,
    resolve_algorithm_config,
)
from deim_app.config.metadata import (
    DatasetMetadata,
    load_coco_metadata,
    load_obb_metadata,
)
from deim_app.errors import AppConfigError

from conftest import (
    make_engine_base,
    mscoco_categories,
    valid_base_dict,
    valid_obb_base_dict,
    write_classes_file,
    write_coco_json,
    write_yaml,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loaded(
    tmp_path: Path,
    *,
    engine_base: dict | None = None,
    app_overrides: dict | None = None,
    box_mode: str = "hbb",
):
    """Build a ``LoadedAppConfig`` directly without going through the YAML loader.

    Mapping tests need full control over ``engine_base`` so we can assert exact
    mutation paths.  This helper constructs the frozen dataclass by hand and
    writes real metadata fixtures (classes.txt / COCO JSON) into ``tmp_path`` so
    ``resolve_algorithm_config`` can derive metadata without failing on
    placeholder paths.
    """
    from deim_app.config import AppConfig, LoadedAppConfig

    base_dict = valid_base_dict() if box_mode == "hbb" else valid_obb_base_dict(fmt="DOTA")

    # Write real metadata fixtures so resolve_algorithm_config succeeds.
    # Use a dedicated subdirectory to avoid colliding with test-created files.
    fixture_dir = tmp_path / "_fixture"
    fixture_dir.mkdir(exist_ok=True)
    if box_mode == "hbb":
        ann = write_coco_json(
            fixture_dir / "instances.json",
            categories=[{"id": 0, "name": "cls0"}, {"id": 1, "name": "cls1"}],
        )
        base_dict["data"]["train_annotations"] = str(ann)
        base_dict["data"]["val_annotations"] = str(ann)
        train_img = fixture_dir / "train_images"
        val_img = fixture_dir / "val_images"
        train_img.mkdir(exist_ok=True)
        val_img.mkdir(exist_ok=True)
        base_dict["data"]["train_images"] = str(train_img)
        base_dict["data"]["val_images"] = str(val_img)
    else:
        classes = write_classes_file(fixture_dir / "classes.txt", ["cat0", "cat1"])
        base_dict["data"]["classes_file"] = str(classes)
        train_img = fixture_dir / "train_images"
        val_img = fixture_dir / "val_images"
        train_ann = fixture_dir / "train_anns"
        val_ann = fixture_dir / "val_anns"
        train_img.mkdir(exist_ok=True)
        val_img.mkdir(exist_ok=True)
        train_ann.mkdir(exist_ok=True)
        val_ann.mkdir(exist_ok=True)
        base_dict["data"]["train_images"] = str(train_img)
        base_dict["data"]["val_images"] = str(val_img)
        base_dict["data"]["train_annotations"] = str(train_ann)
        base_dict["data"]["val_annotations"] = str(val_ann)

    if app_overrides:
        merged = copy.deepcopy(base_dict)
        for section, vals in app_overrides.items():
            if section in merged and isinstance(merged[section], dict):
                merged[section].update(vals)
            else:
                merged[section] = vals
        app = AppConfig.from_mapping(merged)
    else:
        app = AppConfig.from_mapping(base_dict)

    eb = engine_base if engine_base is not None else make_engine_base(box_mode)

    return LoadedAppConfig(
        app=app,
        engine_base=eb,
        source=tmp_path / "user.yml",
        app_base=tmp_path / "base.yml",
    )


# ===========================================================================
# Metadata tests
# ===========================================================================


class TestObbMetadata:
    def test_reads_classes_file_correctly(self, tmp_path: Path) -> None:
        classes = write_classes_file(tmp_path / "classes.txt", ["cable", "clamp"])
        metadata = load_obb_metadata(classes)
        assert metadata.box_mode == "obb"
        assert metadata.num_classes == 2
        assert metadata.class_names_by_label == {0: "cable", 1: "clamp"}

    def test_strips_blank_lines_and_whitespace(self, tmp_path: Path) -> None:
        classes = tmp_path / "classes.txt"
        classes.write_text("  cable  \n\nclamp\n   \n", encoding="utf-8")
        metadata = load_obb_metadata(classes)
        assert metadata.num_classes == 2
        assert metadata.class_names_by_label == {0: "cable", 1: "clamp"}

    def test_single_class(self, tmp_path: Path) -> None:
        classes = write_classes_file(tmp_path / "classes.txt", ["target"])
        metadata = load_obb_metadata(classes)
        assert metadata.num_classes == 1
        assert metadata.class_names_by_label == {0: "target"}

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AppConfigError, match="classes_file"):
            load_obb_metadata(tmp_path / "nonexistent.txt")

    def test_output_names_match_labels(self, tmp_path: Path) -> None:
        classes = write_classes_file(tmp_path / "classes.txt", ["a", "b", "c"])
        metadata = load_obb_metadata(classes)
        assert metadata.output_names_by_id == metadata.class_names_by_label


class TestCocoMetadataContiguous:
    def test_zero_based_contiguous_accepted(self, tmp_path: Path) -> None:
        ann = write_coco_json(
            tmp_path / "instances.json",
            categories=[
                {"id": 0, "name": "cat"},
                {"id": 1, "name": "dog"},
            ],
        )
        metadata = load_coco_metadata(ann, remap_mscoco_category=False)
        assert metadata.box_mode == "hbb"
        assert metadata.num_classes == 2
        assert metadata.class_names_by_label == {0: "cat", 1: "dog"}

    def test_non_contiguous_rejected(self, tmp_path: Path) -> None:
        ann = write_coco_json(
            tmp_path / "instances.json",
            categories=[
                {"id": 1, "name": "one"},
                {"id": 3, "name": "three"},
            ],
        )
        with pytest.raises(AppConfigError, match="contiguous"):
            load_coco_metadata(ann, remap_mscoco_category=False)

    def test_one_based_contiguous_rejected(self, tmp_path: Path) -> None:
        """IDs 1..N (one-based) are NOT contiguous zero-based — must be rejected."""
        ann = write_coco_json(
            tmp_path / "instances.json",
            categories=[
                {"id": 1, "name": "a"},
                {"id": 2, "name": "b"},
            ],
        )
        with pytest.raises(AppConfigError, match="contiguous"):
            load_coco_metadata(ann, remap_mscoco_category=False)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AppConfigError, match="annotation"):
            load_coco_metadata(tmp_path / "nonexistent.json")


class TestCocoMetadataExplicitRemap:
    """remap_mscoco_category=True — strict: must match MS COCO 80 names exactly."""

    def test_explicit_true_with_full_mscoco_remapped(self, tmp_path: Path) -> None:
        ann = write_coco_json(tmp_path / "instances.json", categories=mscoco_categories())
        metadata = load_coco_metadata(ann, remap_mscoco_category=True)
        assert metadata.box_mode == "hbb"
        assert metadata.num_classes == 80
        assert metadata.class_names_by_label[0] == "person"
        assert metadata.class_names_by_label[1] == "bicycle"
        assert metadata.class_names_by_label[79] == "toothbrush"
        assert metadata.output_names_by_id[1] == "person"
        assert metadata.output_names_by_id[90] == "toothbrush"

    def test_explicit_true_with_custom_names_rejected(self, tmp_path: Path) -> None:
        """Explicit True with non-MS-COCO names must raise, not silently remap."""
        ann = write_coco_json(
            tmp_path / "instances.json",
            categories=[
                {"id": 1, "name": "person"},
                {"id": 2, "name": "bicycle"},
            ],
        )
        with pytest.raises(AppConfigError, match="remap_mscoco_category=True"):
            load_coco_metadata(ann, remap_mscoco_category=True)

    def test_explicit_true_error_suggests_false(self, tmp_path: Path) -> None:
        ann = write_coco_json(
            tmp_path / "instances.json",
            categories=[{"id": 0, "name": "widget"}, {"id": 1, "name": "gadget"}],
        )
        with pytest.raises(AppConfigError, match="remap_mscoco_category=False"):
            load_coco_metadata(ann, remap_mscoco_category=True)


class TestCocoMetadataAutoDetect:
    """remap_mscoco_category=None (default) — auto-detect by name+ID match."""

    def test_auto_with_mscoco_names_remapped(self, tmp_path: Path) -> None:
        ann = write_coco_json(tmp_path / "instances.json", categories=mscoco_categories())
        metadata = load_coco_metadata(ann)
        assert metadata.num_classes == 80
        assert metadata.class_names_by_label[0] == "person"
        assert metadata.output_names_by_id[1] == "person"

    def test_auto_with_custom_contiguous_no_remap(self, tmp_path: Path) -> None:
        """Custom dataset with 0..N-1 IDs → no remap, names used as-is."""
        ann = write_coco_json(
            tmp_path / "instances.json",
            categories=[
                {"id": 0, "name": "widget"},
                {"id": 1, "name": "gadget"},
            ],
        )
        metadata = load_coco_metadata(ann)
        assert metadata.num_classes == 2
        assert metadata.class_names_by_label == {0: "widget", 1: "gadget"}
        assert metadata.output_names_by_id == {0: "widget", 1: "gadget"}

    def test_auto_with_non_contiguous_overlapping_1_90_rejected(self, tmp_path: Path) -> None:
        """Custom non-MS-COCO dataset with IDs in 1..90 range must NOT silently remap."""
        ann = write_coco_json(
            tmp_path / "instances.json",
            categories=[
                {"id": 1, "name": "cat"},
                {"id": 3, "name": "dog"},
            ],
        )
        with pytest.raises(AppConfigError, match="contiguous"):
            load_coco_metadata(ann)

    def test_auto_default_is_none(self, tmp_path: Path) -> None:
        """Verify the function signature default is None (auto-detect)."""
        import inspect

        sig = inspect.signature(load_coco_metadata)
        assert sig.parameters["remap_mscoco_category"].default is None


class TestCocoMetadataExplicitNoRemap:
    """remap_mscoco_category=False — contiguous zero-based IDs required."""

    def test_explicit_false_with_contiguous_accepted(self, tmp_path: Path) -> None:
        ann = write_coco_json(
            tmp_path / "instances.json",
            categories=[{"id": 0, "name": "x"}, {"id": 1, "name": "y"}],
        )
        metadata = load_coco_metadata(ann, remap_mscoco_category=False)
        assert metadata.num_classes == 2

    def test_explicit_false_with_non_contiguous_rejected(self, tmp_path: Path) -> None:
        ann = write_coco_json(
            tmp_path / "instances.json",
            categories=[{"id": 1, "name": "a"}, {"id": 3, "name": "b"}],
        )
        with pytest.raises(AppConfigError, match="contiguous"):
            load_coco_metadata(ann, remap_mscoco_category=False)


# ===========================================================================
# Mapping: field-by-field
# ===========================================================================


class TestProjectOutputDir:
    def test_output_dir_mapped(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"project": {"output_dir": "/out/run1"}})
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["output_dir"] == "/out/run1"


class TestRuntimeSeed:
    def test_seed_mapped(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"runtime": {"seed": 123}})
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["seed"] == 123


class TestRuntimeInputSize:
    def test_updates_eval_spatial_size(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"runtime": {"input_size": [512, 768]}})
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["eval_spatial_size"] == [512, 768]

    def test_updates_every_train_resize_op(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"runtime": {"input_size": [512, 512]}})
        resolved = resolve_algorithm_config(loaded)
        train_ops = resolved.overrides["train_dataloader"]["dataset"]["transforms"]["ops"]
        resize_ops = [op for op in train_ops if op.get("type") == "Resize"]
        assert len(resize_ops) > 0
        for op in resize_ops:
            assert op["size"] == [512, 512]

    def test_updates_every_val_resize_op(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"runtime": {"input_size": [512, 512]}})
        resolved = resolve_algorithm_config(loaded)
        val_ops = resolved.overrides["val_dataloader"]["dataset"]["transforms"]["ops"]
        resize_ops = [op for op in val_ops if op.get("type") == "Resize"]
        assert len(resize_ops) > 0
        for op in resize_ops:
            assert op["size"] == [512, 512]

    def test_updates_collate_fn_base_size(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"runtime": {"input_size": [512, 512]}})
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["train_dataloader"]["collate_fn"]["base_size"] == 512

    def test_updates_obb_resize_ops(self, tmp_path: Path) -> None:
        loaded = _make_loaded(
            tmp_path,
            box_mode="obb",
            app_overrides={"runtime": {"input_size": [512, 512]}},
        )
        resolved = resolve_algorithm_config(loaded)
        train_ops = resolved.overrides["train_dataloader"]["dataset"]["transforms"]["ops"]
        resize_ops = [op for op in train_ops if op.get("type") == "OBBResize"]
        assert len(resize_ops) > 0
        for op in resize_ops:
            assert op["size"] == [512, 512]
        assert resolved.overrides["eval_spatial_size"] == [512, 512]


class TestTrainEpochs:
    def test_epoches_legacy_typo_preserved(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"train": {"epochs": 200}})
        resolved = resolve_algorithm_config(loaded)
        # NOTE: the engine uses the legacy typo "epoches" — must preserve it.
        assert resolved.overrides["epoches"] == 200
        assert "epochs" not in resolved.overrides


class TestTrainBatchSize:
    def test_maps_to_train_dataloader_total_batch_size(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"train": {"batch_size": 16}})
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["train_dataloader"]["total_batch_size"] == 16


class TestEvaluationBatchSize:
    def test_maps_to_val_dataloader_total_batch_size(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"evaluation": {"batch_size": 32}})
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["val_dataloader"]["total_batch_size"] == 32


class TestTrainLearningRate:
    def test_changes_only_optimizer_lr(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"train": {"learning_rate": 1e-3}})
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["optimizer"]["lr"] == 1e-3

    def test_param_group_lrs_preserved(self, tmp_path: Path) -> None:
        """The optimizer-LR invariant from the brief."""
        loaded = _make_loaded(tmp_path, app_overrides={"train": {"learning_rate": 1e-3}})
        before_group_lr = loaded.engine_base["optimizer"]["params"][0]["lr"]
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["optimizer"]["lr"] == 1e-3
        assert resolved.overrides["optimizer"]["params"][0]["lr"] == before_group_lr

    def test_engine_base_not_mutated(self, tmp_path: Path) -> None:
        """resolve_algorithm_config must never mutate loaded.engine_base."""
        loaded = _make_loaded(tmp_path, app_overrides={"train": {"learning_rate": 1e-3}})
        original_base = copy.deepcopy(loaded.engine_base)
        resolve_algorithm_config(loaded)
        assert loaded.engine_base == original_base

    def test_all_param_group_lrs_untouched(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"train": {"learning_rate": 2e-3}})
        original_lrs = [pg.get("lr") for pg in loaded.engine_base["optimizer"]["params"]]
        resolved = resolve_algorithm_config(loaded)
        new_lrs = [pg.get("lr") for pg in resolved.overrides["optimizer"]["params"]]
        assert new_lrs == original_lrs


class TestTrainAmp:
    def test_use_amp_mapped(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"train": {"amp": False}})
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["use_amp"] is False


class TestTrainPretrained:
    def test_tuning_mapped(self, tmp_path: Path) -> None:
        loaded = _make_loaded(
            tmp_path,
            app_overrides={"train": {"pretrained": "/weights/init.pth"}},
        )
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["tuning"] == "/weights/init.pth"


class TestTrainResume:
    def test_resume_mapped(self, tmp_path: Path) -> None:
        loaded = _make_loaded(
            tmp_path,
            app_overrides={"train": {"resume": "/ckpt/last.pth"}},
        )
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["resume"] == "/ckpt/last.pth"


class TestTrainDevice:
    def test_device_mapped(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"train": {"device": "cuda:1"}})
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["device"] == "cuda:1"


class TestEarlyStopping:
    def test_preserves_preset_owned_fields(self, tmp_path: Path) -> None:
        """Public fields override; preset-owned fields (metric, mode, ...) persist."""
        loaded = _make_loaded(
            tmp_path,
            app_overrides={"train": {"early_stopping": {"enabled": True, "patience": 20}}},
        )
        resolved = resolve_algorithm_config(loaded)
        es = resolved.overrides["early_stopping"]
        # public fields overridden
        assert es["enabled"] is True
        assert es["patience"] == 20
        # preset-owned fields preserved
        assert es["metric"] == "mAP50_95"
        assert es["mode"] == "max"
        assert es["min_epochs"] == 100
        assert es["restore_best"] is True


# ===========================================================================
# Data-path mapping
# ===========================================================================


class TestHBBDataPaths:
    def test_img_folder_and_ann_file_set_on_both_splits(self, tmp_path: Path) -> None:
        ann = write_coco_json(
            tmp_path / "custom.json",
            categories=[{"id": 0, "name": "x"}, {"id": 1, "name": "y"}],
        )
        loaded = _make_loaded(
            tmp_path,
            app_overrides={
                "data": {
                    "train_images": str(tmp_path),
                    "train_annotations": str(ann),
                    "val_images": str(tmp_path),
                    "val_annotations": str(ann),
                }
            },
        )
        resolved = resolve_algorithm_config(loaded)
        train_ds = resolved.overrides["train_dataloader"]["dataset"]
        val_ds = resolved.overrides["val_dataloader"]["dataset"]
        assert train_ds["img_folder"] == str(tmp_path)
        assert train_ds["ann_file"] == str(ann)
        assert val_ds["img_folder"] == str(tmp_path)
        assert val_ds["ann_file"] == str(ann)

    def test_dataset_type_set_to_coco(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path)
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["train_dataloader"]["dataset"]["type"] == "CocoDetection"
        assert resolved.overrides["val_dataloader"]["dataset"]["type"] == "CocoDetection"

    def test_num_workers_on_both_dataloaders(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, app_overrides={"data": {"num_workers": 8}})
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["train_dataloader"]["num_workers"] == 8
        assert resolved.overrides["val_dataloader"]["num_workers"] == 8


class TestOBBDataPaths:
    def test_ann_folder_and_classes_file_set(self, tmp_path: Path) -> None:
        classes = write_classes_file(tmp_path / "cls.txt", ["a", "b"])
        loaded = _make_loaded(
            tmp_path,
            box_mode="obb",
            app_overrides={
                "data": {
                    "format": "DOTA",
                    "train_images": str(tmp_path),
                    "train_annotations": str(tmp_path),
                    "val_images": str(tmp_path),
                    "val_annotations": str(tmp_path),
                    "classes_file": str(classes),
                }
            },
        )
        resolved = resolve_algorithm_config(loaded)
        train_ds = resolved.overrides["train_dataloader"]["dataset"]
        val_ds = resolved.overrides["val_dataloader"]["dataset"]
        assert train_ds["img_folder"] == str(tmp_path)
        assert train_ds["ann_folder"] == str(tmp_path)
        assert train_ds["classes_file"] == str(classes)
        assert train_ds["format"] == "DOTA"
        assert val_ds["img_folder"] == str(tmp_path)
        assert val_ds["ann_folder"] == str(tmp_path)
        assert val_ds["classes_file"] == str(classes)
        assert val_ds["format"] == "DOTA"

    def test_dataset_type_set_to_dota(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path, box_mode="obb")
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["train_dataloader"]["dataset"]["type"] == "DotaDataset"

    def test_format_yolo_obb(self, tmp_path: Path) -> None:
        loaded = _make_loaded(
            tmp_path,
            box_mode="obb",
            app_overrides={"data": {"format": "YOLO-OBB"}},
        )
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["train_dataloader"]["dataset"]["format"] == "YOLO-OBB"


# ===========================================================================
# cache_images mapping
# ===========================================================================


class TestCacheImagesHBB:
    def test_none_is_silent(self, tmp_path: Path) -> None:
        """CocoDetection has no cache_images kwarg — 'none' must not set it."""
        loaded = _make_loaded(tmp_path, app_overrides={"data": {"cache_images": "none"}})
        resolved = resolve_algorithm_config(loaded)
        train_ds = resolved.overrides["train_dataloader"]["dataset"]
        assert "cache_images" not in train_ds
        assert "cache_ram" not in train_ds


class TestCacheImagesOBB:
    def test_none_maps_to_cache_none_ram_zero(self, tmp_path: Path) -> None:
        loaded = _make_loaded(
            tmp_path, box_mode="obb", app_overrides={"data": {"cache_images": "none"}}
        )
        resolved = resolve_algorithm_config(loaded)
        ds = resolved.overrides["train_dataloader"]["dataset"]
        assert ds["cache_images"] == "none"
        assert ds["cache_ram"] == 0

    def test_disk_maps_to_cache_disk_ram_zero(self, tmp_path: Path) -> None:
        loaded = _make_loaded(
            tmp_path, box_mode="obb", app_overrides={"data": {"cache_images": "disk"}}
        )
        resolved = resolve_algorithm_config(loaded)
        ds = resolved.overrides["train_dataloader"]["dataset"]
        assert ds["cache_images"] == "disk"
        assert ds["cache_ram"] == 0

    def test_ram_maps_to_cache_none_ram_count(self, tmp_path: Path) -> None:
        """ram mode: cache_images='none', cache_ram=<image count in that split>."""
        custom_dir = tmp_path / "ram_images"
        loaded = _make_loaded(
            tmp_path,
            box_mode="obb",
            app_overrides={
                "data": {
                    "cache_images": "ram",
                    "train_images": str(custom_dir),
                }
            },
        )
        custom_dir.mkdir(exist_ok=True)
        (custom_dir / "a.jpg").write_bytes(b"")
        (custom_dir / "b.png").write_bytes(b"")
        (custom_dir / "c.txt").write_bytes(b"")  # not an image
        resolved = resolve_algorithm_config(loaded)
        ds = resolved.overrides["train_dataloader"]["dataset"]
        assert ds["cache_images"] == "none"
        assert ds["cache_ram"] == 2  # only .jpg + .png counted


# ===========================================================================
# num_classes propagation
# ===========================================================================


class TestNumClassesPropagation:
    def test_num_classes_from_obb_metadata(self, tmp_path: Path) -> None:
        classes = write_classes_file(tmp_path / "three_cls.txt", ["a", "b", "c"])
        loaded = _make_loaded(
            tmp_path,
            box_mode="obb",
            app_overrides={"data": {"classes_file": str(classes)}},
        )
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["num_classes"] == 3
        assert resolved.metadata.num_classes == 3

    def test_num_classes_from_coco_metadata(self, tmp_path: Path) -> None:
        ann = write_coco_json(
            tmp_path / "instances.json",
            categories=[{"id": 0, "name": "x"}, {"id": 1, "name": "y"}],
        )
        loaded = _make_loaded(
            tmp_path,
            app_overrides={"data": {"train_annotations": str(ann)}},
        )
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides["num_classes"] == 2


# ===========================================================================
# ResolvedAlgorithmConfig shape
# ===========================================================================


class TestResolvedShape:
    def test_returns_resolved_algorithm_config(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path)
        resolved = resolve_algorithm_config(loaded)
        assert isinstance(resolved, ResolvedAlgorithmConfig)
        assert resolved.config_path == loaded.source
        assert resolved.app is loaded.app
        assert isinstance(resolved.overrides, dict)
        assert isinstance(resolved.metadata, DatasetMetadata)

    def test_overrides_is_deep_copy_not_same_object(self, tmp_path: Path) -> None:
        loaded = _make_loaded(tmp_path)
        resolved = resolve_algorithm_config(loaded)
        assert resolved.overrides is not loaded.engine_base


# ===========================================================================
# Example YAML end-to-end loading
# ===========================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestExampleYamls:
    """Each example must resolve or fail with a precise missing-path error."""

    @pytest.mark.parametrize(
        "example_name",
        ["hbb_coco", "obb_dota", "obb_yolo"],
    )
    def test_example_loads_or_fails_precisely(self, example_name: str) -> None:
        example_path = REPO_ROOT / "configs" / "app" / "examples" / f"{example_name}.yml"
        if not example_path.exists():
            pytest.skip(f"{example_path} not yet created")
        try:
            loaded = load_app_config(example_path)
            resolve_algorithm_config(loaded)
        except AppConfigError as exc:
            msg = str(exc).lower()
            assert any(
                keyword in msg
                for keyword in (
                    "does not exist",
                    "not found",
                    "no such file",
                    "missing",
                    "classes_file",
                    "annotation",
                    "img_folder",
                    "images",
                )
            ), f"example {example_name} failed with non-path error: {exc}"
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"example {example_name} raised unexpected {type(exc).__name__}: {exc}"
            )
