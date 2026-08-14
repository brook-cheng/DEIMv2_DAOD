"""Retained early_stopping block and deleted precision-config absence tests."""

from pathlib import Path
from typing import Final

from engine.core import yaml_utils
from engine.core.yaml_config import YAMLConfig


ROOT: Final = Path(__file__).resolve().parents[1]
DLZDT_DIR: Final = ROOT / "configs" / "custom_obb" / "dlzdt"

RETAINED_CONFIG: Final = "sp_fz_common.yml"

DELETED_CONFIGS: Final = (
    "sp_fz_rep0_nloss_bf16.yml",
    "sp_fz_rep0_nloss_bf16_es.yml",
    "sp_fz_rep0_nloss_fp16_obb_fp32.yml",
)

EXPECTED_EARLY_STOPPING: Final = {
    "enabled": False,
    "metric": "mAP50_95",
    "mode": "max",
    "min_epochs": 100,
    "patience": 40,
    "min_delta": 0.0001,
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


def test_retained_config_carries_approved_early_stopping_block() -> None:
    assert (
        _dlzdt_config(RETAINED_CONFIG)["early_stopping"] == EXPECTED_EARLY_STOPPING
    )
