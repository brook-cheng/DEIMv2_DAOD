from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
import yaml

from engine.core.yaml_config import YAMLConfig


ROOT: Final = Path(__file__).resolve().parents[1]
CONFIG_DIR: Final = ROOT / "configs" / "custom_obb" / "synthetic_configs"
PROVENANCE_DIR: Final = CONFIG_DIR / "provenance"
ACTIVE_RUNS: Final = {
    "synthetic_exp_020_loss_kld.yml": {
        "switches": (False, False),
        "geometry_weights": {"loss_bbox": 5, "loss_kld": 2},
    },
    "synthetic_exp_020_loss_prob_kld.yml": {
        "switches": (True, False),
        "geometry_weights": {"loss_bbox": 5, "loss_probiou": 5, "loss_kld": 2},
    },
    "synthetic_exp_020_loss_prob_angle_kld.yml": {
        "switches": (True, True),
        "geometry_weights": {
            "loss_bbox": 5,
            "loss_probiou": 5,
            "loss_angle": 3,
            "loss_kld": 2,
        },
    },
}
SNAPSHOT_RUNS: Final = {
    "synthetic_exp_020_loss_kld": {
        "snapshot": "synthetic_exp_020_loss_kld.completed.yml",
        "switches": {"use_yolo_probiou": False, "use_yolo_angle": False},
        "geometry_weights": {"loss_bbox": 5, "loss_kld": 2},
    },
    "synthetic_exp_020_loss_prob_kld": {
        "snapshot": "synthetic_exp_020_loss_prob_kld.completed.yml",
        "switches": {"use_yolo_probiou": True, "use_yolo_angle": True},
        "geometry_weights": {
            "loss_bbox": 5,
            "loss_probiou": 5,
            "loss_angle": 3,
            "loss_kld": 2,
        },
    },
}
GEOMETRY_WEIGHT_KEYS: Final = {"loss_bbox", "loss_probiou", "loss_angle", "loss_kld"}
CRITERION_GEOMETRY_KEYS: Final = {
    "use_yolo_probiou",
    "use_yolo_angle",
    "angle_lambda",
}


def _resolved_configs():
    return {
        name: YAMLConfig(str(CONFIG_DIR / name)).yaml_cfg
        for name in ACTIVE_RUNS
    }


@pytest.mark.parametrize(("name", "expected"), ACTIVE_RUNS.items())
def test_active_loss_config_has_controlled_geometry(name, expected):
    # Given: one member of the controlled three-config loss matrix.
    config = YAMLConfig(str(CONFIG_DIR / name)).yaml_cfg
    criterion = config["DEIMCriterion"]

    # When: its geometry switches and active geometry weights are selected.
    switches = (criterion["use_yolo_probiou"], criterion["use_yolo_angle"])
    weights = {
        key: value
        for key, value in criterion["weight_dict"].items()
        if key in GEOMETRY_WEIGHT_KEYS
    }

    # Then: the config represents exactly its declared matrix member.
    assert config["epoches"] == 80
    assert switches == expected["switches"]
    assert weights == expected["geometry_weights"]


def test_active_loss_configs_have_distinct_outputs():
    # Given: all three resolved active loss configs.
    configs = _resolved_configs()

    # When: their output directories are collected.
    outputs = {config["output_dir"] for config in configs.values()}

    # Then: every controlled run writes to a distinct destination.
    assert len(outputs) == len(ACTIVE_RUNS)


def test_active_loss_configs_share_exact_matcher():
    # Given: all three resolved active loss configs.
    configs = _resolved_configs()

    # When: their criterion matchers are collected.
    matchers = [config["DEIMCriterion"]["matcher"] for config in configs.values()]

    # Then: matching is identical and retains the controlled Chamfer cost.
    assert all(matcher == matchers[0] for matcher in matchers[1:])
    assert matchers[0]["weight_dict"]["cost_chamfer"] == 2


def test_active_loss_configs_differ_only_by_controlled_geometry():
    # Given: independent copies of all resolved active loss configs.
    configs = list(deepcopy(_resolved_configs()).values())

    # When: output and explicitly controlled criterion geometry are removed.
    for config in configs:
        del config["output_dir"]
        criterion = config["DEIMCriterion"]
        for key in CRITERION_GEOMETRY_KEYS:
            criterion.pop(key, None)
        criterion["weight_dict"] = {
            key: value
            for key, value in criterion["weight_dict"].items()
            if key not in GEOMETRY_WEIGHT_KEYS
        }

    # Then: every other resolved setting is identical.
    assert all(config == configs[0] for config in configs[1:])


def test_active_config_set_excludes_completed_run_provenance():
    # Given: the declared active configuration paths.
    active_paths = {CONFIG_DIR / name for name in ACTIVE_RUNS}

    # When: their names and parents are inspected.
    active_names = {path.name for path in active_paths}

    # Then: only the three live configs are active, never provenance artifacts.
    assert active_names == set(ACTIVE_RUNS)
    assert all(PROVENANCE_DIR not in path.parents for path in active_paths)
    assert all(not name.endswith(".completed.yml") for name in active_names)


def test_completed_run_provenance_contains_exact_artifacts():
    # Given: the completed-run provenance directory.
    expected_names = {
        "completed_runs.yml",
        *(run["snapshot"] for run in SNAPSHOT_RUNS.values()),
    }

    # When: its YAML artifacts are listed.
    actual_names = {path.name for path in PROVENANCE_DIR.glob("*.yml")}

    # Then: exactly two snapshots and their manifest are retained.
    assert actual_names == expected_names


@pytest.mark.parametrize(("run_name", "expected"), SNAPSHOT_RUNS.items())
def test_completed_run_manifest_records_truthful_history(run_name, expected):
    # Given: the completed-runs manifest and its historical run declaration.
    manifest = yaml.safe_load((PROVENANCE_DIR / "completed_runs.yml").read_text())
    run = manifest["runs"][run_name]

    # When: immutable run facts and availability flags are inspected.
    snapshot_path = PROVENANCE_DIR / run["snapshot"]

    # Then: known history is truthful and unavailable evidence stays explicit.
    assert run["status"] == "completed"
    assert run["declared_epochs"] == 80
    assert run["snapshot"] == expected["snapshot"]
    assert run["known_switches"] == expected["switches"]
    assert run["known_geometry_weights"] == expected["geometry_weights"]
    assert run["resolved_config_available"] is False
    assert run["output_artifacts_available"] is False
    assert run["snapshot_sha256"] == sha256(snapshot_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(("run_name", "expected"), SNAPSHOT_RUNS.items())
def test_completed_snapshot_matches_manifested_known_behavior(run_name, expected):
    # Given: one immutable launch-config snapshot.
    snapshot_path = PROVENANCE_DIR / expected["snapshot"]
    snapshot = yaml.safe_load(snapshot_path.read_text())

    # When: its known criterion geometry is selected.
    criterion = snapshot["DEIMCriterion"]
    switches = {
        key: criterion[key]
        for key in ("use_yolo_probiou", "use_yolo_angle")
    }
    weights = {
        key: value
        for key, value in criterion["weight_dict"].items()
        if key in GEOMETRY_WEIGHT_KEYS
    }

    # Then: the snapshot itself supports every known provenance claim.
    assert snapshot["epoches"] == 80
    assert switches == expected["switches"]
    assert weights == expected["geometry_weights"]


# ── AMP BF16 / OBB FP32 experiment configs ──────────────────────────────────

DLZDT_DIR: Final = ROOT / "configs" / "custom_obb" / "dlzdt"
AMP_BASE: Final = "sp_fz_rep0_nloss_amp.yml"
PRECISION_EXPERIMENTS: Final = {
    "sp_fz_rep0_nloss_bf16.yml": {
        "amp_dtype": "bfloat16",
        "obb_geometry_fp32": False,
        "scaler_enabled": False,
    },
    "sp_fz_rep0_nloss_fp16_obb_fp32.yml": {
        "amp_dtype": "float16",
        "obb_geometry_fp32": True,
        "scaler_enabled": True,
    },
}


def _dlzdt_config(name: str) -> dict:
    return YAMLConfig(str(DLZDT_DIR / name)).yaml_cfg


def test_precision_experiments_preserve_base_training_schedule() -> None:
    # Given: the base FP16 AMP experiment and both precision experiments.
    base = _dlzdt_config(AMP_BASE)

    # When: schedule and loss settings are resolved for each experiment.
    # Then: dataset, optimizer LR, schedule, and loss weights match the base.
    for name in PRECISION_EXPERIMENTS:
        config = _dlzdt_config(name)
        assert config["epoches"] == base["epoches"] == 150
        assert config["flat_epoch"] == base["flat_epoch"] == 75
        assert config["no_aug_epoch"] == base["no_aug_epoch"] == 15
        assert config["train_dataloader"]["dataset"]["img_folder"] == base[
            "train_dataloader"
        ]["dataset"]["img_folder"]
        assert config["optimizer"]["lr"] == base["optimizer"]["lr"] == 0.0005
        assert config["DEIMCriterion"]["weight_dict"] == base["DEIMCriterion"][
            "weight_dict"
        ]
        assert config["DEIMCriterion"]["matcher"] == base["DEIMCriterion"]["matcher"]


@pytest.mark.parametrize(("name", "expected"), PRECISION_EXPERIMENTS.items())
def test_precision_experiment_resolves_intended_flags(name, expected) -> None:
    # Given: one declared precision experiment.
    config = _dlzdt_config(name)

    # When: its precision fields and scaler enabled state are inspected.
    # Then: the resolved config carries exactly the declared policy.
    assert config["amp_dtype"] == expected["amp_dtype"]
    assert config["obb_geometry_fp32"] == expected["obb_geometry_fp32"]
    assert config["scaler"]["enabled"] == expected["scaler_enabled"]


def test_precision_experiments_have_distinct_output_dirs() -> None:
    # Given: the base experiment and both precision experiments.
    outputs = {
        _dlzdt_config(AMP_BASE)["output_dir"],
        *(
            _dlzdt_config(name)["output_dir"]
            for name in PRECISION_EXPERIMENTS
        ),
    }

    # When: their resolved output directories are collected.
    # Then: every run writes to a distinct destination.
    assert len(outputs) == len(PRECISION_EXPERIMENTS) + 1
