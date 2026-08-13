"""Tests for the typed application YAML loader (Task 2).

Covers the four explicit scenarios from the task brief plus the full coverage
list: format/cache enums, input_size shape, batch/worker sign, score range,
unknown keys at every depth, CLI whitelist enforcement, include validation,
HBB vs OBB cache rules, and frozen-dataclass immutability.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from deim_app.config import AppConfig, LoadedAppConfig, load_app_config
from deim_app.errors import AppConfigError

from conftest import (
    valid_base_dict,
    valid_obb_base_dict,
    valid_user_dict,
    write_app_base,
    write_base_and_user,
    write_yaml,
)


# ---------------------------------------------------------------------------
# Explicit brief tests
# ---------------------------------------------------------------------------


def test_rejects_algorithm_key_in_user_yaml(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "bad.yml",
        {"__include__": ["base.yml"], "DEIMTransformer": {"angle_rep": 2}},
    )
    write_app_base(tmp_path / "base.yml", valid_base_dict())
    with pytest.raises(AppConfigError, match="DEIMTransformer"):
        load_app_config(path)


def test_rejects_direct_include_of_algorithm_yaml(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "bad.yml",
        {"__include__": ["../../configs/custom_obb/dlzdt/sp_fz_common.yml"]},
    )
    with pytest.raises(AppConfigError, match="application base"):
        load_app_config(path)


def test_cli_device_overrides_user_and_base_yaml(tmp_path: Path) -> None:
    base = write_app_base(tmp_path / "base.yml", valid_base_dict(device="cpu"))
    user = write_yaml(
        tmp_path / "user.yml",
        {"__include__": [base.name], "train": {"device": "cuda:0"}},
    )
    loaded = load_app_config(user, {"train": {"device": "cuda:1"}})
    assert loaded.app.train.device == "cuda:1"
    # evaluation/inference device comes from base (cpu) since user/CLI didn't override
    assert loaded.app.evaluation.device == "cpu"


def test_pretrained_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "bad.yml",
        {
            **valid_user_dict(),
            "train": {"pretrained": "init.pth", "resume": "last.pth"},
        },
    )
    write_yaml(tmp_path / "base.yml", valid_base_dict())
    with pytest.raises(AppConfigError, match="pretrained.*resume"):
        load_app_config(path)


# ---------------------------------------------------------------------------
# data.format enum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_format", ["VOC", "coco", "dota", "", 42, None])
def test_invalid_data_format_rejected(tmp_path: Path, bad_format: object) -> None:
    path = write_base_and_user(
        tmp_path,
        base={**valid_base_dict(), "data": {**valid_base_dict()["data"], "format": bad_format}},
    )
    with pytest.raises(AppConfigError, match="data.format"):
        load_app_config(path)


@pytest.mark.parametrize("good_format", ["COCO", "DOTA", "YOLO-OBB"])
def test_valid_data_formats_accepted(tmp_path: Path, good_format: str) -> None:
    base = valid_base_dict()
    if good_format != "COCO":
        base = valid_obb_base_dict(fmt=good_format)
    path = write_base_and_user(tmp_path, base=base)
    loaded = load_app_config(path)
    assert loaded.app.data.format == good_format


# ---------------------------------------------------------------------------
# data.cache_images enum + HBB/OBB rule
# ---------------------------------------------------------------------------


def test_invalid_cache_mode_rejected(tmp_path: Path) -> None:
    path = write_base_and_user(
        tmp_path,
        base={**valid_base_dict(), "data": {**valid_base_dict()["data"], "cache_images": "ssd"}},
    )
    with pytest.raises(AppConfigError, match="cache_images"):
        load_app_config(path)


@pytest.mark.parametrize("bad_cache", ["disk", "ram"])
def test_hbb_rejects_non_none_cache(tmp_path: Path, bad_cache: str) -> None:
    path = write_base_and_user(
        tmp_path,
        base={**valid_base_dict(), "data": {**valid_base_dict()["data"], "cache_images": bad_cache}},
    )
    with pytest.raises(AppConfigError, match=r"(cache_images|COCO|HBB)"):
        load_app_config(path)


@pytest.mark.parametrize("good_cache", ["none", "disk", "ram"])
@pytest.mark.parametrize("fmt", ["DOTA", "YOLO-OBB"])
def test_obb_accepts_all_cache_modes(
    tmp_path: Path, fmt: str, good_cache: str
) -> None:
    path = write_base_and_user(
        tmp_path, base=valid_obb_base_dict(fmt=fmt, cache_images=good_cache)
    )
    loaded = load_app_config(path)
    assert loaded.app.data.cache_images == good_cache


# ---------------------------------------------------------------------------
# runtime.input_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_size",
    [[640], [640, 640, 640], [-1, 640], [0, 640], [640, 0], "640x640", 640],
)
def test_input_size_must_be_two_positive_ints(
    tmp_path: Path, bad_size: object
) -> None:
    path = write_base_and_user(
        tmp_path,
        base={**valid_base_dict(), "runtime": {"input_size": bad_size, "seed": 1}},
    )
    with pytest.raises(AppConfigError, match="input_size"):
        load_app_config(path)


def test_input_size_accepts_two_positive_ints(tmp_path: Path) -> None:
    path = write_base_and_user(
        tmp_path,
        base={**valid_base_dict(), "runtime": {"input_size": [1024, 1024], "seed": 7}},
    )
    loaded = load_app_config(path)
    assert loaded.app.runtime.input_size == (1024, 1024)
    assert loaded.app.runtime.seed == 7


# ---------------------------------------------------------------------------
# batch sizes + num_workers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("section", ["train", "evaluation", "inference"])
@pytest.mark.parametrize("bad_batch", [0, -1, 1.5, "8", True])
def test_batch_sizes_must_be_positive_int(
    tmp_path: Path, section: str, bad_batch: object
) -> None:
    base = valid_base_dict()
    base[section] = {**base[section], "batch_size": bad_batch}
    path = write_base_and_user(tmp_path, base=base)
    with pytest.raises(AppConfigError, match="batch_size"):
        load_app_config(path)


@pytest.mark.parametrize("bad_nw", [-1, 1.5, "2", True])
def test_num_workers_must_be_nonneg_int(tmp_path: Path, bad_nw: object) -> None:
    path = write_base_and_user(
        tmp_path,
        base={**valid_base_dict(), "data": {**valid_base_dict()["data"], "num_workers": bad_nw}},
    )
    with pytest.raises(AppConfigError, match="num_workers"):
        load_app_config(path)


def test_num_workers_zero_accepted(tmp_path: Path) -> None:
    path = write_base_and_user(
        tmp_path,
        base={**valid_base_dict(), "data": {**valid_base_dict()["data"], "num_workers": 0}},
    )
    loaded = load_app_config(path)
    assert loaded.app.data.num_workers == 0


# ---------------------------------------------------------------------------
# score_threshold + top_k
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_score", [-0.01, 1.01, 2.0, "0.5", True])
def test_score_threshold_must_be_in_unit_range(
    tmp_path: Path, bad_score: object
) -> None:
    path = write_base_and_user(
        tmp_path,
        base={**valid_base_dict(), "inference": {**valid_base_dict()["inference"], "score_threshold": bad_score}},
    )
    with pytest.raises(AppConfigError, match="score_threshold"):
        load_app_config(path)


@pytest.mark.parametrize("bad_topk", [0, -1, 1.5, "300", True])
def test_top_k_must_be_positive_int(tmp_path: Path, bad_topk: object) -> None:
    path = write_base_and_user(
        tmp_path,
        base={**valid_base_dict(), "inference": {**valid_base_dict()["inference"], "top_k": bad_topk}},
    )
    with pytest.raises(AppConfigError, match="top_k"):
        load_app_config(path)


# ---------------------------------------------------------------------------
# Unknown keys at every public section (dotted path in error)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section,subkey",
    [
        ("project", "bad"),
        ("runtime", "bad"),
        ("data", "bad"),
        ("train", "bad"),
        ("evaluation", "bad"),
        ("inference", "bad"),
    ],
)
def test_unknown_section_key_rejected_with_dotted_path(
    tmp_path: Path, section: str, subkey: str
) -> None:
    path = write_base_and_user(
        tmp_path,
        user={**valid_user_dict(), section: {subkey: 1}},
    )
    with pytest.raises(AppConfigError, match=rf"{section}\.{subkey}"):
        load_app_config(path)


def test_unknown_early_stopping_key_rejected(tmp_path: Path) -> None:
    path = write_base_and_user(
        tmp_path,
        user={**valid_user_dict(), "train": {"early_stopping": {"bogus": 1}}},
    )
    with pytest.raises(AppConfigError, match=r"train\.early_stopping\.bogus"):
        load_app_config(path)


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    path = write_base_and_user(
        tmp_path,
        user={**valid_user_dict(), "mystery_section": {"foo": 1}},
    )
    with pytest.raises(AppConfigError, match="mystery_section"):
        load_app_config(path)


# ---------------------------------------------------------------------------
# CLI overrides whitelist
# ---------------------------------------------------------------------------


def test_cli_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    path = write_base_and_user(tmp_path)
    with pytest.raises(AppConfigError, match="mystery_cli"):
        load_app_config(path, {"mystery_cli": {"foo": 1}})


def test_cli_unknown_section_key_rejected(tmp_path: Path) -> None:
    path = write_base_and_user(tmp_path)
    with pytest.raises(AppConfigError, match=r"train\.bad_cli"):
        load_app_config(path, {"train": {"bad_cli": 1}})


def test_cli_override_value_applied(tmp_path: Path) -> None:
    path = write_base_and_user(tmp_path)
    loaded = load_app_config(path, {"train": {"epochs": 99, "batch_size": 16}})
    assert loaded.app.train.epochs == 99
    assert loaded.app.train.batch_size == 16


# ---------------------------------------------------------------------------
# __include__ validation
# ---------------------------------------------------------------------------


def test_missing_include_rejected(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "user.yml", {"train": {"epochs": 5}})
    with pytest.raises(AppConfigError, match="application base"):
        load_app_config(path)


def test_multiple_includes_rejected(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "user.yml",
        {"__include__": ["base.yml", "other.yml"]},
    )
    with pytest.raises(AppConfigError, match="application base"):
        load_app_config(path)


def test_empty_include_list_rejected(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "user.yml", {"__include__": []})
    with pytest.raises(AppConfigError, match="application base"):
        load_app_config(path)


def test_parent_traversal_to_unapproved_basename_rejected(tmp_path: Path) -> None:
    """``..`` traversal itself is allowed (for sibling-directory includes like
    ``../base/hbb_app.yml``), but the target basename must still be approved."""
    path = write_yaml(
        tmp_path / "user.yml",
        {"__include__": ["../secret.yml"]},
    )
    with pytest.raises(AppConfigError, match="application base"):
        load_app_config(path)


def test_non_string_include_entry_rejected(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "user.yml", {"__include__": [42]})
    with pytest.raises(AppConfigError, match="application base"):
        load_app_config(path)


def test_arbitrary_same_name_base_rejected(tmp_path: Path) -> None:
    """An attacker-controlled ``hbb_app.yml`` written outside the repo is refused.

    The basename matches an approved application-base name, but the resolved
    path is not a canonical approved path, so the include must be rejected.
    Trust is anchored to exact resolved paths, never to basenames.
    """
    malicious = write_yaml(tmp_path / "hbb_app.yml", valid_base_dict())
    user = write_yaml(
        tmp_path / "user.yml",
        {"__include__": [malicious.name]},
    )
    with pytest.raises(AppConfigError, match="application base"):
        load_app_config(user)


def test_symlink_to_outside_same_name_rejected(tmp_path: Path) -> None:
    """A symlink named ``hbb_app.yml`` pointing outside the repo is refused.

    ``resolve()`` follows the symlink to its real target; that real target is
    not an approved application-base path, so the include is rejected even
    though the link itself carries an approved basename.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    target = write_yaml(outside / "evil.yml", valid_base_dict())
    link = tmp_path / "hbb_app.yml"
    link.symlink_to(target)
    user = write_yaml(
        tmp_path / "user.yml",
        {"__include__": [link.name]},
    )
    with pytest.raises(AppConfigError, match="application base"):
        load_app_config(user)


# ---------------------------------------------------------------------------
# LoadedAppConfig shape + immutability + engine_base isolation
# ---------------------------------------------------------------------------


def test_full_valid_config_loads_all_fields(tmp_path: Path) -> None:
    path = write_base_and_user(tmp_path)
    loaded = load_app_config(path)
    assert isinstance(loaded, LoadedAppConfig)
    assert isinstance(loaded.app, AppConfig)
    assert loaded.app.project.name == "test-app"
    assert loaded.app.runtime.input_size == (640, 640)
    assert loaded.app.data.format == "COCO"
    assert loaded.app.train.epochs == 12
    assert loaded.app.evaluation.device == "cuda"
    assert loaded.app.inference.score_threshold == 0.25
    assert loaded.source == path.resolve()
    assert loaded.app_base == (tmp_path / "base.yml").resolve()


def test_engine_base_contains_full_merged_dict(tmp_path: Path) -> None:
    base = valid_base_dict()
    base["DEIMTransformer"] = {"num_bins": 16}  # algorithm section the base may carry
    path = write_base_and_user(tmp_path, base=base)
    loaded = load_app_config(path)
    # engine_base carries trusted algorithm keys that AppConfig must NOT expose
    assert loaded.engine_base["DEIMTransformer"] == {"num_bins": 16}
    assert not hasattr(loaded.app, "DEIMTransformer")


def test_engine_base_not_mutated_across_calls(tmp_path: Path) -> None:
    """Two independent loads must not leak state (load_config mutable-default guard)."""
    base_a = write_app_base(tmp_path / "base_a.yml", valid_base_dict())
    user_a = write_yaml(
        tmp_path / "user_a.yml",
        {"__include__": [base_a.name], "train": {"epochs": 5}},
    )
    loaded_a = load_app_config(user_a)
    assert loaded_a.app.train.epochs == 5

    base_b = write_app_base(tmp_path / "base_b.yml", valid_obb_base_dict(fmt="DOTA"))
    user_b = write_yaml(
        tmp_path / "user_b.yml",
        {"__include__": [base_b.name], "train": {"epochs": 50}},
    )
    loaded_b = load_app_config(user_b)
    assert loaded_b.app.train.epochs == 50
    assert loaded_b.app.data.format == "DOTA"
    # A's results unchanged after B loaded
    assert loaded_a.app.train.epochs == 5
    assert loaded_a.app.data.format == "COCO"


def test_app_config_is_frozen(tmp_path: Path) -> None:
    path = write_base_and_user(tmp_path)
    loaded = load_app_config(path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.app.train.epochs = 999


def test_class_filter_and_known_output_formats_coerced_to_tuples(
    tmp_path: Path,
) -> None:
    """List values are coerced to tuples; only known writer formats are used.

    Tuple coercion is exercised independently from format validation: this test
    uses two known formats so it passes once coercion works, even before format
    whitelisting exists. Format rejection is covered by
    ``test_unknown_output_format_rejected`` below.
    """
    base = valid_base_dict()
    base["inference"] = {
        **base["inference"],
        "class_filter": ["car", "truck"],
        "output_formats": ["json", "visualization"],
    }
    path = write_base_and_user(tmp_path, base=base)
    loaded = load_app_config(path)
    assert loaded.app.inference.class_filter == ("car", "truck")
    assert loaded.app.inference.output_formats == ("json", "visualization")
    assert isinstance(loaded.app.inference.class_filter, tuple)
    assert isinstance(loaded.app.inference.output_formats, tuple)


@pytest.mark.parametrize("bad_format", ["csv", "xml", "parquet", "DOTA"])
def test_unknown_output_format_rejected(tmp_path: Path, bad_format: str) -> None:
    """Unknown writer names in ``inference.output_formats`` are rejected at
    ``AppConfig`` construction time (not deferred to CLI/write time)."""
    base = valid_base_dict()
    base["inference"] = {
        **base["inference"],
        "output_formats": ["json", bad_format],
    }
    path = write_base_and_user(tmp_path, base=base)
    with pytest.raises(AppConfigError, match=r"inference\.output_formats"):
        load_app_config(path)


def test_dota_output_rejected_for_hbb_data_format(tmp_path: Path) -> None:
    """``inference.output_formats`` containing ``dota`` must be rejected at
    ``AppConfig`` construction time when ``data.format`` is HBB (COCO).

    DOTA writer requires oriented bounding boxes; HBB-only COCO data is
    incompatible. The rejection must surface at the config boundary, before
    inference or write time.
    """
    base = valid_base_dict()  # data.format == "COCO" (HBB)
    base["inference"] = {
        **base["inference"],
        "output_formats": ["json", "dota"],
    }
    path = write_base_and_user(tmp_path, base=base)
    with pytest.raises(AppConfigError, match=r"(inference\.output_formats|DOTA|HBB)"):
        load_app_config(path)


@pytest.mark.parametrize("fmt", ["DOTA", "YOLO-OBB"])
def test_dota_output_accepted_for_obb_data_format(tmp_path: Path, fmt: str) -> None:
    """``inference.output_formats`` containing ``dota`` is accepted when
    ``data.format`` is OBB (DOTA or YOLO-OBB)."""
    base = valid_obb_base_dict(fmt=fmt)
    base["inference"] = {
        **base["inference"],
        "output_formats": ["json", "dota", "visualization"],
    }
    path = write_base_and_user(tmp_path, base=base)
    loaded = load_app_config(path)
    assert "dota" in loaded.app.inference.output_formats
    assert loaded.app.data.format == fmt


def test_checkpoint_coerced_to_path(tmp_path: Path) -> None:
    base = valid_base_dict()
    base["inference"] = {**base["inference"], "checkpoint": "weights/best.pth"}
    path = write_base_and_user(tmp_path, base=base)
    loaded = load_app_config(path)
    assert isinstance(loaded.app.inference.checkpoint, Path)
    assert loaded.app.inference.checkpoint == Path("weights/best.pth")


def test_missing_public_sections_use_defaults(tmp_path: Path) -> None:
    # Base that only declares train — other sections fall back to dataclass defaults
    write_app_base(tmp_path / "base.yml", {"train": {"epochs": 3, "batch_size": 2}})
    user = write_yaml(tmp_path / "user.yml", {"__include__": ["base.yml"]})
    loaded = load_app_config(user)
    assert loaded.app.train.epochs == 3
    # defaults from DataConfig
    assert loaded.app.data.format == "COCO"
    assert loaded.app.data.num_workers == 4
