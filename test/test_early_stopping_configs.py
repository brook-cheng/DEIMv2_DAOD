"""ES-Base config inheritance tests (FP16 only).

Verifies the approved early_stopping block appears only in the FP16 *_es.yml
config, that schedule/precision are inherited unchanged, and that the
rejected BF16 / FP16+OBB-FP32 configs are deleted.
"""

from pathlib import Path
from typing import Final

import pytest

from engine.core import yaml_utils
from engine.core.yaml_config import YAMLConfig


ROOT: Final = Path(__file__).resolve().parents[1]
DLZDT_DIR: Final = ROOT / "configs" / "custom_obb" / "dlzdt"

AMP_BASE: Final = "sp_fz_rep0_nloss_amp.yml"
ES_AMP: Final = "sp_fz_rep0_nloss_amp_es.yml"

DELETED_CONFIGS: Final = (
    "sp_fz_rep0_nloss_bf16.yml",
    "sp_fz_rep0_nloss_bf16_es.yml",
    "sp_fz_rep0_nloss_fp16_obb_fp32.yml",
)

EXPECTED_ES: Final = {
    "enabled": True,
    "metric": "mAP50_95",
    "mode": "max",
    "min_epochs": 100,
    "patience": 12,
    "min_delta": 0.001,
    "restore_best": True,
}


def _dlzdt_config(name: str) -> dict:
    # load_config() uses a mutable default dict accumulator that persists
    # across calls (pre-existing RT-DETR yaml_utils.py bug). Reset it so each
    # config load starts from a clean slate.
    yaml_utils.load_config.__defaults__ = ({},)
    return YAMLConfig(str(DLZDT_DIR / name)).yaml_cfg


def test_deleted_precision_configs_absent_from_disk() -> None:
    for name in DELETED_CONFIGS:
        assert not (DLZDT_DIR / name).exists(), name


def test_legacy_base_has_no_early_stopping() -> None:
    assert "early_stopping" not in _dlzdt_config(AMP_BASE)


def test_es_config_carries_approved_block() -> None:
    assert _dlzdt_config(ES_AMP)["early_stopping"] == EXPECTED_ES


def test_es_config_has_no_precision_mode_keys() -> None:
    config = _dlzdt_config(ES_AMP)
    assert "amp_dtype" not in config
    assert "obb_geometry_fp32" not in config
    assert config["scaler"]["enabled"] is True


def test_es_config_preserves_base_training_schedule() -> None:
    base = _dlzdt_config(AMP_BASE)
    config = _dlzdt_config(ES_AMP)
    assert config["epoches"] == base["epoches"] == 150
    assert config["warmup_iter"] == base["warmup_iter"] == 15
    assert config["flat_epoch"] == base["flat_epoch"] == 75
    assert config["no_aug_epoch"] == base["no_aug_epoch"] == 15
    assert config["optimizer"]["lr"] == base["optimizer"]["lr"] == 0.0005
    assert config["DEIMCriterion"]["weight_dict"] == base["DEIMCriterion"]["weight_dict"]
    assert config["DEIMCriterion"]["matcher"] == base["DEIMCriterion"]["matcher"]


def test_es_configs_have_distinct_output_dirs() -> None:
    outputs = {_dlzdt_config(name)["output_dir"] for name in (AMP_BASE, ES_AMP)}
    assert len(outputs) == 2
