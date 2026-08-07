"""ES-Base config inheritance tests.

Verifies the approved early_stopping block appears only in the two new *_es.yml
configs, that schedule/precision are inherited unchanged, and that the legacy
bases and the abandoned C-group config stay unpolluted.
"""

from pathlib import Path
from typing import Final

import pytest

from engine.core import yaml_utils
from engine.core.yaml_config import YAMLConfig


ROOT: Final = Path(__file__).resolve().parents[1]
DLZDT_DIR: Final = ROOT / "configs" / "custom_obb" / "dlzdt"

AMP_BASE: Final = "sp_fz_rep0_nloss_amp.yml"
BF16_BASE: Final = "sp_fz_rep0_nloss_bf16.yml"
ES_AMP: Final = "sp_fz_rep0_nloss_amp_es.yml"
ES_BF16: Final = "sp_fz_rep0_nloss_bf16_es.yml"
C_GROUP: Final = "sp_fz_rep0_nloss_fp16_obb_fp32.yml"

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
    # config load starts from a clean slate and does not inherit stray keys
    # (e.g. bfloat16) from a previously-loaded sibling config.
    yaml_utils.load_config.__defaults__ = ({},)
    return YAMLConfig(str(DLZDT_DIR / name)).yaml_cfg


def test_legacy_bases_have_no_early_stopping() -> None:
    for name in (AMP_BASE, BF16_BASE, C_GROUP):
        assert "early_stopping" not in _dlzdt_config(name), name


@pytest.mark.parametrize("name", [ES_AMP, ES_BF16])
def test_es_config_carries_approved_block(name: str) -> None:
    assert _dlzdt_config(name)["early_stopping"] == EXPECTED_ES


@pytest.mark.parametrize(
    ("name", "amp_dtype", "scaler_enabled"),
    [
        (ES_AMP, None, True),        # inherits FP16 default from amp base
        (ES_BF16, "bfloat16", False),  # inherits BF16 + disabled scaler
    ],
)
def test_es_config_preserves_precision(name, amp_dtype, scaler_enabled) -> None:
    config = _dlzdt_config(name)
    assert config.get("amp_dtype") == amp_dtype
    # obb_geometry_fp32 is unset (None) in the base chain; ES configs must not
    # introduce it (only the abandoned C-group sets it True).
    assert config.get("obb_geometry_fp32") is not True
    assert config["scaler"]["enabled"] is scaler_enabled


def test_es_configs_preserve_base_training_schedule() -> None:
    base = _dlzdt_config(AMP_BASE)
    for name in (ES_AMP, ES_BF16):
        config = _dlzdt_config(name)
        assert config["epoches"] == base["epoches"] == 150
        assert config["warmup_iter"] == base["warmup_iter"] == 15
        assert config["flat_epoch"] == base["flat_epoch"] == 75
        assert config["no_aug_epoch"] == base["no_aug_epoch"] == 15
        assert config["optimizer"]["lr"] == base["optimizer"]["lr"] == 0.0005
        assert config["DEIMCriterion"]["weight_dict"] == base["DEIMCriterion"]["weight_dict"]
        assert config["DEIMCriterion"]["matcher"] == base["DEIMCriterion"]["matcher"]


def test_es_configs_have_distinct_output_dirs() -> None:
    outputs = {_dlzdt_config(name)["output_dir"] for name in (AMP_BASE, ES_AMP, ES_BF16)}
    assert len(outputs) == 3
