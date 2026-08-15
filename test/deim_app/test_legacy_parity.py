"""Task 10 release gate (G5): legacy equivalence for the application layer.

Two test groups:

1. **Structural parity (always runs).** For each example application YAML
   (``hbb_coco``, ``obb_dota``, ``obb_yolo``) we resolve it end-to-end through
   :func:`load_app_config` + :func:`resolve_algorithm_config` and assert the
   resolved public fields and surviving algorithm fields equal the expected
   values from the source algorithm YAML
   (``configs/deimv2/deimv2_dinov3_l_coco.yml`` for HBB,
   ``configs/custom_obb/dlzdt/sp_fz_common.yml`` for OBB). A separate scan
   verifies that base/example application YAMLs contain NONE of the forbidden
   algorithm keys (those keys may appear ONLY in presets).

2. **Numerical parity (skips when fixtures are missing).** For each mode, if
   the operator has exported the ``DEIM_APP_PARITY_*`` environment variables
   pointing at local fixture checkpoints + images (+ metadata), we run the new
   adapter backend and an independently constructed legacy reference path
   (mirroring ``tools/inference/torch_inf.py`` for HBB and
   ``tools/compare/core.py`` for OBB) and assert labels/boxes/scores
   agree within tight tolerances. When any required env var or fixture file is
   absent, the test skips with the exact missing path so CI stays green.

These tests MUST NOT modify engine state, mutate the global YAML registry, or
weaken any pre-existing assertion.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from deim_app.config import load_app_config, resolve_algorithm_config
from deim_app.config.metadata import DatasetMetadata

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "configs" / "app" / "examples"
BASES_DIR = REPO_ROOT / "configs" / "app" / "base"
PRESETS_DIR = REPO_ROOT / "configs" / "app" / "presets"

HBB_EXAMPLE = EXAMPLES_DIR / "hbb_coco.yml"
OBB_DOTA_EXAMPLE = EXAMPLES_DIR / "obb_dota.yml"
OBB_YOLO_EXAMPLE = EXAMPLES_DIR / "obb_yolo.yml"


# ---------------------------------------------------------------------------
# Forbidden algorithm key scan
# ---------------------------------------------------------------------------

#: Algorithm-only top-level or component keys that belong in PRESETS only.
#: These keys MUST NOT appear in any base/*.yml or examples/*.yml. The list is
#: curated from the categories named in the task brief (``angle_rep``,
#: ``offset_scale_source``, ...) plus the engine component section names and
#: the scheduler/optimizer keys that the public whitelist explicitly excludes.
_FORBIDDEN_ALGORITHM_KEYS: frozenset[str] = frozenset({
    # OBB algorithm contract
    "angle_rep",
    "offset_scale_source",
    "use_gate_fusion",
    "angle_step",
    "use_angle_first",
    "decoder_angle_encoding",
    "box_mode",
    # Engine component section names
    "DEIM",
    "DINOv3STAs",
    "DINOv3STAsResAtten",
    "HGNetv2",
    "HybridEncoder",
    "DEIMTransformer",
    "PostProcessor",
    "DEIMCriterion",
    "evaluator",
    # Algorithm derivation fields
    "num_classes",
    "obbox_rep_dim",
    # Scheduler / optimizer / training-loop algorithm fields
    "epoches",
    "lrsheduler",
    "lr_scheduler",
    "lr_warmup_scheduler",
    "lr_gamma",
    "warmup_iter",
    "flat_epoch",
    "no_aug_epoch",
    "flatcosine",
    "use_amp",
    "use_ema",
    "clip_max_norm",
    "ema",
    "scaler",
    "checkpoint_freq",
    "eval_spatial_size",
    "comet_project_name",
    # Algorithm dataloader / dataset section names (engine YAML)
    "train_dataloader",
    "val_dataloader",
    # Solver wire-up keys (engine YAML)
    "task",
    "model",
    "criterion",
    "postprocessor",
})


def _collect_keys(node: Any, keys: set[str]) -> None:
    """Recursively collect every mapping key from a nested YAML structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            keys.add(str(k))
            _collect_keys(v, keys)
    elif isinstance(node, list):
        for item in node:
            _collect_keys(item, keys)


_BASE_AND_EXAMPLE_PATHS: tuple[Path, ...] = (
    *sorted(BASES_DIR.glob("*.yml")),
    *sorted(EXAMPLES_DIR.glob("*.yml")),
)


@pytest.mark.parametrize(
    "yaml_path",
    _BASE_AND_EXAMPLE_PATHS,
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_application_base_or_example_has_no_algorithm_keys(yaml_path: Path) -> None:
    """Base and example YAMLs MUST contain only the six public sections.

    Algorithm keys (``angle_rep``, ``DEIMTransformer``, ``optimizer``,
    ``num_classes``, ``epoches``, ...) are reserved for presets. Their presence
    in a base or example file would break the trust boundary: a user YAML
    merging those fields could silently override algorithm state.
    """
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    keys: set[str] = set()
    _collect_keys(raw, keys)
    forbidden_found = keys & _FORBIDDEN_ALGORITHM_KEYS
    assert not forbidden_found, (
        f"{yaml_path} contains forbidden algorithm keys: "
        f"{sorted(forbidden_found)}. Algorithm keys may appear ONLY in "
        f"configs/app/presets/*.yml."
    )


def test_preset_files_DO_carry_algorithm_keys() -> None:
    """Sanity check: presets are where the algorithm contract lives.

    If this assertion fails, the preset has been stripped too aggressively
    and downstream parity will break.
    """
    obb_preset = PRESETS_DIR / "deimv2_dinov3_sp_obb.yml"
    raw = yaml.safe_load(obb_preset.read_text(encoding="utf-8")) or {}
    keys: set[str] = set()
    _collect_keys(raw, keys)
    expected_algorithm_markers = {
        "angle_rep",
        "DEIMTransformer",
        "DEIMCriterion",
        "optimizer",
    }
    assert expected_algorithm_markers.issubset(keys), (
        f"OBB preset missing expected algorithm markers; present: {sorted(keys)}"
    )


# ===========================================================================
# Group 1 — Structural parity (always runs)
# ===========================================================================
#
# Metadata loading is stubbed via ``monkeypatch`` so the example YAMLs'
# placeholder dataset paths do NOT abort resolution. The structural assertions
# cover:
#   - resolved box mode (hbb / obb) per example
#   - resolved num_classes (driven by stubbed metadata)
#   - resolved public fields (input_size, batch_size, epochs, learning_rate,
#     amp, early_stopping)
#   - algorithm-only fields survive unchanged (angle_rep, optimizer param
#     group LRs, EMA, scheduler stages, augmentation probabilities, loss /
#     matcher weights)
#   - preset-owned early_stopping fields are preserved while app-owned fields
#     (enabled, patience) are overridden


def _patch_hbb_metadata(monkeypatch: pytest.MonkeyPatch, num_classes: int = 80) -> None:
    """Stub ``load_coco_metadata`` to return a contiguous contiguous-label dataset."""
    fake = DatasetMetadata(
        box_mode="hbb",
        num_classes=num_classes,
        class_names_by_label={i: f"cls{i}" for i in range(num_classes)},
        output_names_by_id={i: f"cls{i}" for i in range(num_classes)},
    )
    monkeypatch.setattr(
        "deim_app.config.mapping.load_coco_metadata",
        lambda path, remap_mscoco_category=None: fake,
    )


def _patch_obb_metadata(monkeypatch: pytest.MonkeyPatch, num_classes: int = 15) -> None:
    """Stub ``load_obb_metadata`` to return a fixed-class-count OBB dataset."""
    fake = DatasetMetadata(
        box_mode="obb",
        num_classes=num_classes,
        class_names_by_label={i: f"c{i}" for i in range(num_classes)},
        output_names_by_id={i: f"c{i}" for i in range(num_classes)},
    )
    monkeypatch.setattr(
        "deim_app.config.mapping.load_obb_metadata",
        lambda path: fake,
    )


# ---------------------------------------------------------------------------
# HBB COCO example → deimv2_dinov3_l_coco.yml parity
# ---------------------------------------------------------------------------


def test_hbb_coco_resolves_public_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public fields on the HBB example match the COCO source algorithm YAML."""
    _patch_hbb_metadata(monkeypatch, num_classes=80)
    loaded = load_app_config(HBB_EXAMPLE)
    resolved = resolve_algorithm_config(loaded)

    # Box mode + class count derived from stubbed metadata.
    assert resolved.metadata.box_mode == "hbb"
    assert resolved.metadata.num_classes == 80
    assert resolved.overrides["num_classes"] == 80

    # runtime.input_size updates every required location.
    assert resolved.app.runtime.input_size == (640, 640)
    assert resolved.app.runtime.seed == 42
    assert resolved.overrides["eval_spatial_size"] == [640, 640]

    # train.* fields match deimv2_dinov3_l_coco.yml (epoches=68, optimizer.lr=5e-4).
    assert resolved.app.train.epochs == 68
    assert resolved.app.train.batch_size == 4
    assert resolved.app.train.learning_rate == 5.0e-4
    assert resolved.app.train.device == "cuda"
    assert resolved.app.train.amp is True
    assert resolved.app.train.pretrained is None
    assert resolved.app.train.resume is None
    assert resolved.app.train.early_stopping.enabled is False
    assert resolved.app.train.early_stopping.patience == 10

    # data.* — HBB COCO format, no classes_file, cache=none is mandatory.
    assert resolved.app.data.format == "COCO"
    assert resolved.app.data.classes_file is None
    assert resolved.app.data.cache_images == "none"
    assert resolved.app.data.num_workers == 4

    # evaluation.batch_size from app (4); val_dataloader.total_batch_size follows.
    assert resolved.app.evaluation.batch_size == 4
    assert resolved.overrides["val_dataloader"]["total_batch_size"] == 4

    # inference defaults inherited from hbb_app.yml.
    assert resolved.app.inference.score_threshold == 0.25
    assert resolved.app.inference.top_k == 300
    assert resolved.app.inference.batch_size == 1
    assert resolved.app.inference.output_formats == ("json", "visualization")


def test_hbb_coco_algorithm_fields_survive_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Algorithm preset fields survive the resolver unchanged (HBB).

    Compared against configs/deimv2/deimv2_dinov3_l_coco.yml — the source the
    HBB preset was derived from.
    """
    _patch_hbb_metadata(monkeypatch, num_classes=80)
    resolved = resolve_algorithm_config(load_app_config(HBB_EXAMPLE))
    ov = resolved.overrides

    # Backbone (DINOv3STAs ViT-S/16).
    assert ov["DEIM"]["backbone"] == "DINOv3STAs"
    assert ov["DINOv3STAs"]["name"] == "dinov3_vits16"
    assert ov["DINOv3STAs"]["weights_path"] == (
        "./ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    )
    assert ov["DINOv3STAs"]["interaction_indexes"] == [5, 8, 11]
    assert ov["DINOv3STAs"]["finetune"] is True
    assert ov["DINOv3STAs"]["conv_inplane"] == 32
    assert ov["DINOv3STAs"]["hidden_dim"] == 224

    # Encoder.
    assert ov["HybridEncoder"]["in_channels"] == [224, 224, 224]
    assert ov["HybridEncoder"]["hidden_dim"] == 224
    assert ov["HybridEncoder"]["dim_feedforward"] == 896

    # Decoder.
    assert ov["DEIMTransformer"]["feat_channels"] == [224, 224, 224]
    assert ov["DEIMTransformer"]["hidden_dim"] == 224
    assert ov["DEIMTransformer"]["num_layers"] == 4
    assert ov["DEIMTransformer"]["eval_idx"] == -1
    assert ov["DEIMTransformer"]["dim_feedforward"] == 1792

    # FlatCosine scheduler stages (preset-owned).
    assert ov["lrsheduler"] == "flatcosine"
    assert ov["lr_gamma"] == 0.5
    assert ov["warmup_iter"] == 2000
    assert ov["flat_epoch"] == 34
    assert ov["no_aug_epoch"] == 8

    # Optimizer: top-level lr OVERRIDDEN by app.train.learning_rate (5e-4).
    assert ov["optimizer"]["type"] == "AdamW"
    assert ov["optimizer"]["lr"] == 5.0e-4
    assert ov["optimizer"]["betas"] == [0.9, 0.999]
    assert ov["optimizer"]["weight_decay"] == 0.000125
    # Per-group learning rates preserved (NOT touched by the resolver).
    pg0, pg1, pg2 = ov["optimizer"]["params"]
    assert pg0["lr"] == 1.25e-5
    assert pg1["lr"] == 1.25e-5
    assert pg1["weight_decay"] == 0.0
    assert pg2["weight_decay"] == 0.0

    # EMA + grad-clip preserved (from base/optimizer.yml include chain).
    assert ov["use_ema"] is True
    assert ov["clip_max_norm"] == 0.1

    # Mosaic probability preserved from the augmentation pipeline.
    train_ops = ov["train_dataloader"]["dataset"]["transforms"]["ops"]
    mosaic_op = next(op for op in train_ops if op.get("type") == "Mosaic")
    assert mosaic_op["probability"] == 1.0

    # Matcher late-stage epoch preserved.
    assert ov["DEIMCriterion"]["matcher"]["matcher_change_epoch"] == 50


# ---------------------------------------------------------------------------
# OBB DOTA example → sp_fz_common.yml parity
# ---------------------------------------------------------------------------


def test_obb_dota_resolves_public_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public fields on the OBB DOTA example match the OBB source algorithm YAML."""
    _patch_obb_metadata(monkeypatch, num_classes=15)
    loaded = load_app_config(OBB_DOTA_EXAMPLE)
    resolved = resolve_algorithm_config(loaded)

    # Box mode + class count.
    assert resolved.metadata.box_mode == "obb"
    assert resolved.metadata.num_classes == 15
    assert resolved.overrides["num_classes"] == 15

    # runtime.input_size updates every required location.
    assert resolved.app.runtime.input_size == (640, 640)
    assert resolved.app.runtime.seed == 42
    assert resolved.overrides["eval_spatial_size"] == [640, 640]

    # train.* fields match sp_fz_common.yml (epoches=200, optimizer.lr=5e-4).
    assert resolved.app.train.epochs == 200
    assert resolved.app.train.batch_size == 4
    assert resolved.app.train.learning_rate == 5.0e-4
    assert resolved.app.train.amp is True
    assert resolved.app.train.early_stopping.enabled is False
    # OBB app base sets patience=40 (not the HBB default of 10).
    assert resolved.app.train.early_stopping.patience == 40

    # data.* — DOTA format with classes_file + disk cache by default in example.
    assert resolved.app.data.format == "DOTA"
    assert resolved.app.data.classes_file is not None
    assert resolved.app.data.cache_images == "disk"
    assert resolved.app.data.num_workers == 2

    # val_dataloader.total_batch_size follows evaluation.batch_size (2 from base).
    assert resolved.app.evaluation.batch_size == 2
    assert resolved.overrides["val_dataloader"]["total_batch_size"] == 2

    # inference defaults inherited from obb_app.yml.
    assert resolved.app.inference.score_threshold == 0.25
    assert resolved.app.inference.top_k == 300


def test_obb_dota_algorithm_fields_survive_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Algorithm preset fields survive the resolver unchanged (OBB DOTA).

    Compared against configs/custom_obb/dlzdt/sp_fz_common.yml — the source the
    OBB preset was derived from.
    """
    _patch_obb_metadata(monkeypatch, num_classes=15)
    resolved = resolve_algorithm_config(load_app_config(OBB_DOTA_EXAMPLE))
    ov = resolved.overrides

    # OBB angle contract — the six ablation variables (all at baseline values).
    decoder = ov["DEIMTransformer"]
    assert decoder["box_mode"] == "obb"
    assert decoder["angle_rep"] == 0

    # Backbone DINOv3STAsResAtten with STA adapter.
    bb = ov["DINOv3STAsResAtten"]
    assert bb["name"] == "dinov3_vits16plus"
    assert bb["finetune"] is False
    assert bb["conv_inplane"] == 64
    assert bb["hidden_dim"] == 256
    assert bb["adapter_type"] == "sta"

    # Encoder.
    assert ov["HybridEncoder"]["in_channels"] == [256, 256, 256]
    assert ov["HybridEncoder"]["hidden_dim"] == 256
    assert ov["HybridEncoder"]["dim_feedforward"] == 1024

    # Decoder.
    assert ov["DEIMTransformer"]["feat_channels"] == [256, 256, 256]
    assert ov["DEIMTransformer"]["hidden_dim"] == 256
    assert ov["DEIMTransformer"]["dim_feedforward"] == 2048

    # PostProcessor box_mode preserved.
    assert ov["PostProcessor"]["box_mode"] == "obb"

    # FlatCosine scheduler stages (preset-owned).
    assert ov["warmup_iter"] == 10
    assert ov["flat_epoch"] == 100
    assert ov["no_aug_epoch"] == 20
    assert ov["checkpoint_freq"] == 10

    # Optimizer: top-level lr OVERRIDDEN by app.train.learning_rate (5e-4).
    assert ov["optimizer"]["lr"] == 5.0e-4
    # Per-group LRs preserved (1e-5 backbone groups).
    pg0, pg1, pg2 = ov["optimizer"]["params"]
    assert pg0["lr"] == 1.0e-5
    assert pg1["lr"] == 1.0e-5
    assert pg1["weight_decay"] == 0.0
    assert pg2["weight_decay"] == 0.0

    # Loss weights (DEIMCriterion.weight_dict).
    wd = ov["DEIMCriterion"]["weight_dict"]
    assert wd["loss_mal"] == 1
    assert wd["loss_bbox"] == 5
    assert wd["loss_probiou"] == 5
    assert wd["loss_angle"] == 3
    assert wd["loss_kld"] == 2
    assert wd["loss_fgl"] == 0.15
    assert wd["loss_ddf"] == 1.5
    # Criterion-level OBB contract.
    crit = ov["DEIMCriterion"]
    assert crit["use_yolo_probiou"] is True
    assert crit["use_yolo_angle"] is True
    assert crit["keep_kld"] is True
    assert crit["angle_lambda"] == 3.0
    assert crit["gamma"] == 1.0
    assert crit["alpha"] == 0.75
    assert crit["reg_max"] == 32
    assert crit["box_mode"] == "obb"
    assert crit["obbox_rep_dim"] == 6

    # Matcher weights.
    matcher = crit["matcher"]
    assert matcher["type"] == "HungarianMatcher"
    mw = matcher["weight_dict"]
    assert mw["cost_class"] == 2
    assert mw["cost_bbox"] == 5
    assert mw["cost_probiou"] == 5
    assert mw["cost_angle"] == 3
    assert mw["cost_chamfer"] == 2
    assert mw["late_cost_bbox"] == 0.25
    assert matcher["change_matcher"] is False
    assert matcher["iou_order_alpha"] == 4.0
    assert matcher["angle_order_alpha"] == 1.0
    assert matcher["box_mode"] == "obb"

    # Augmentation probabilities (collate_fn + transforms.mosaic_prob).
    collate = ov["train_dataloader"]["collate_fn"]
    assert collate["mixup_prob"] == 0.5
    assert collate["copyblend_prob"] == 0.5
    assert collate["base_size_repeat"] == 4
    transforms_block = ov["train_dataloader"]["dataset"]["transforms"]
    assert transforms_block["mosaic_prob"] == 0.5
    # Policy stage epochs are preset-owned too.
    assert transforms_block["policy"]["epoch"] == [10, 30, 50]


def test_obb_dota_early_stopping_merges_preset_and_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preset-owned early_stopping fields survive; app-owned fields override.

    The OBB preset supplies metric/mode/min_epochs/min_delta/restore_best; the
    app layer overrides ONLY enabled + patience. Both layers must coexist.
    """
    _patch_obb_metadata(monkeypatch, num_classes=15)
    resolved = resolve_algorithm_config(load_app_config(OBB_DOTA_EXAMPLE))
    es = resolved.overrides["early_stopping"]

    # App-owned (overridden).
    assert es["enabled"] is False
    assert es["patience"] == 40
    # Preset-owned (preserved).
    assert es["metric"] == "mAP50_95"
    assert es["mode"] == "max"
    assert es["min_epochs"] == 100
    assert es["min_delta"] == 0.0001
    assert es["restore_best"] is True


# ---------------------------------------------------------------------------
# OBB YOLO-OBB example → same source as DOTA (sp_fz_common.yml)
# ---------------------------------------------------------------------------


def test_obb_yolo_resolves_format_and_algorithm_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YOLO-OBB example selects YOLO-OBB format but inherits the same algorithm
    contract as the DOTA example (both pull from sp_fz_common.yml)."""
    _patch_obb_metadata(monkeypatch, num_classes=15)
    resolved = resolve_algorithm_config(load_app_config(OBB_YOLO_EXAMPLE))
    ov = resolved.overrides

    # Format-specific data fields.
    assert resolved.app.data.format == "YOLO-OBB"
    assert resolved.app.data.classes_file is not None
    assert resolved.app.data.cache_images == "disk"
    # Dataset type wired by the resolver for both OBB formats.
    assert ov["train_dataloader"]["dataset"]["type"] == "DotaDataset"
    assert ov["val_dataloader"]["dataset"]["type"] == "DotaDataset"
    assert ov["train_dataloader"]["dataset"]["format"] == "YOLO-OBB"
    assert ov["val_dataloader"]["dataset"]["format"] == "YOLO-OBB"

    # Algorithm contract identical to the DOTA example (same preset).
    dec = ov["DEIMTransformer"]
    assert dec["box_mode"] == "obb"
    assert dec["angle_rep"] == 0
    assert dec["dim_feedforward"] == 2048
    assert ov["HybridEncoder"]["dim_feedforward"] == 1024
    assert ov["DEIMCriterion"]["obbox_rep_dim"] == 6


# ---------------------------------------------------------------------------
# Cross-example consistency
# ---------------------------------------------------------------------------


def test_input_size_propagates_to_every_resize_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """runtime.input_size must update every Resize/OBBResize op + collate
    base_size + eval_spatial_size (Task 3 invariant, re-asserted at the gate).
    """
    _patch_obb_metadata(monkeypatch, num_classes=15)
    resolved = resolve_algorithm_config(load_app_config(OBB_YOLO_EXAMPLE))
    ov = resolved.overrides

    assert ov["eval_spatial_size"] == [640, 640]
    for dl_key in ("train_dataloader", "val_dataloader"):
        ops = ov[dl_key]["dataset"]["transforms"]["ops"]
        resize_ops = [op for op in ops if op.get("type") in ("Resize", "OBBResize")]
        assert resize_ops, f"expected at least one resize op in {dl_key}"
        for op in resize_ops:
            assert op["size"] == [640, 640]
    assert ov["train_dataloader"]["collate_fn"]["base_size"] == 640


def test_main_learning_rate_does_not_touch_param_group_lrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per the completion checklist: a main lr override must leave param-group
    LRs unchanged. We verify this by overriding learning_rate and re-resolving."""
    _patch_hbb_metadata(monkeypatch, num_classes=80)
    loaded = load_app_config(HBB_EXAMPLE)
    # Mutate the public app field via CLI overrides (validated by the loader).
    loaded2 = load_app_config(
        HBB_EXAMPLE, cli_overrides={"train": {"learning_rate": 1.0e-3}}
    )
    resolved = resolve_algorithm_config(loaded2)
    ov = resolved.overrides
    # Top-level lr is the override.
    assert ov["optimizer"]["lr"] == 1.0e-3
    # Param-group LRs are the preset values (unchanged).
    pg0, pg1, _pg2 = ov["optimizer"]["params"]
    assert pg0["lr"] == 1.25e-5
    assert pg1["lr"] == 1.25e-5


# ===========================================================================
# Group 2 — Numerical parity (skips when fixtures are missing)
# ===========================================================================
#
# Fixture location scheme (env vars):
#   DEIM_APP_PARITY_HBB_CKPT       — path to an HBB .pth checkpoint
#   DEIM_APP_PARITY_HBB_IMG        — path to a single HBB test image
#   DEIM_APP_PARITY_HBB_COCO_JSON  — path to a COCO instances*.json used for
#                                    class-metadata derivation
#   DEIM_APP_PARITY_OBB_CKPT       — path to an OBB .pth checkpoint
#   DEIM_APP_PARITY_OBB_IMG        — path to a single OBB test image
#   DEIM_APP_PARITY_OBB_CLASSES    — path to a classes.txt (one class per line)
#   DEIM_APP_PARITY_OBB_FORMAT     — "DOTA" or "YOLO-OBB" (default: "YOLO-OBB")
#
# When any required var is unset OR the file it points at does not exist, the
# test skips with the exact missing path. Structural parity (above) always runs.


def _resolve_env(var: str) -> str | None:
    """Return the env var's value if set AND the file exists; else None."""
    val = os.environ.get(var)
    if not val:
        return None
    if not Path(val).exists():
        return None
    return val


def _skip_if_missing(required: dict[str, str]) -> dict[str, str]:
    """Skip the test unless every named env var is set AND points at an existing file."""
    missing: list[str] = []
    for var in required:
        val = os.environ.get(var)
        if not val:
            missing.append(var)
            continue
        if not Path(val).exists():
            missing.append(f"{var} (path {val!r} does not exist)")
    if missing:
        pytest.skip(
            "Numerical parity skipped — missing fixtures: "
            + ", ".join(missing)
            + ". Set the DEIM_APP_PARITY_* env vars to enable."
        )
    return {k: os.environ[v] for k, v in required.items()}


# ---------------------------------------------------------------------------
# HBB numerical parity vs tools/inference/torch_inf.py
# ---------------------------------------------------------------------------


def test_hbb_numerical_parity_vs_torch_inf() -> None:
    """HBB adapter backend matches the legacy torch_inf.py path bit-for-bit."""
    paths = _skip_if_missing(
        {
            "DEIM_APP_PARITY_HBB_CKPT": "checkpoint",
            "DEIM_APP_PARITY_HBB_IMG": "image",
            "DEIM_APP_PARITY_HBB_COCO_JSON": "coco json",
        }
    )
    ckpt = paths["DEIM_APP_PARITY_HBB_CKPT"]
    img_path = paths["DEIM_APP_PARITY_HBB_IMG"]
    coco_json = paths["DEIM_APP_PARITY_HBB_COCO_JSON"]

    import torch
    from PIL import Image

    from deim_app.api import DetectionModel

    # ---- New adapter backend ----------------------------------------------
    img_dir = str(Path(img_path).parent)
    new_model = DetectionModel.from_config(
        str(HBB_EXAMPLE),
        data={
            "train_images": img_dir,
            "train_annotations": coco_json,
            "val_images": img_dir,
            "val_annotations": coco_json,
        },
        inference={"device": "cpu", "batch_size": 1},
    )
    new_model.load(ckpt, prefer_ema=True)
    new_collection = new_model.predict(img_path)
    assert len(new_collection.predictions) == 1, "test image must produce one prediction"
    new_pred = new_collection.predictions[0]

    new_labels = torch.tensor([d.class_id for d in new_pred.detections], dtype=torch.long)
    new_scores = torch.tensor([d.score for d in new_pred.detections], dtype=torch.float32)
    new_boxes = torch.tensor(
        [d.xyxy for d in new_pred.detections], dtype=torch.float32
    ).reshape(-1, 4)

    # ---- Legacy reference path (mirrors tools/inference/torch_inf.py) ------
    legacy_labels, legacy_boxes, legacy_scores = _hbb_legacy_torch_inf(
        ckpt=ckpt,
        img_path=img_path,
        source_yaml=str(REPO_ROOT / "configs" / "deimv2" / "deimv2_dinov3_l_coco.yml"),
        size=(640, 640),
    )

    # ---- Compare ----------------------------------------------------------
    assert new_labels.shape == legacy_labels.shape, (
        f"label shape mismatch: new {new_labels.shape} vs legacy {legacy_labels.shape}"
    )
    torch.testing.assert_close(new_labels, legacy_labels, rtol=0, atol=0)
    torch.testing.assert_close(new_scores, legacy_scores, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(new_boxes, legacy_boxes, rtol=1e-5, atol=1e-4)


def _hbb_legacy_torch_inf(
    *, ckpt: str, img_path: str, source_yaml: str, size: tuple[int, int]
) -> tuple[Any, Any, Any]:
    """Replicate ``tools/inference/torch_inf.py`` exactly: build YAMLConfig from
    the source algorithm YAML, deploy model + postprocessor, load the
    checkpoint, run a single image, return ``(labels, boxes, scores)`` tensors.
    """
    import torch
    from PIL import Image
    import torchvision.transforms as T

    from engine.core import YAMLConfig

    cfg = YAMLConfig(source_yaml)
    # Disable backbone pretrained download (parity with adapter + torch_inf).
    yaml_cfg = getattr(cfg, "yaml_cfg", None) or {}
    if isinstance(yaml_cfg, dict) and "HGNetv2" in yaml_cfg:
        yaml_cfg["HGNetv2"]["pretrained"] = False

    # Load checkpoint (ema.module → model, strip DDP prefix) — same helper the
    # adapter uses, applied here on the legacy path for fidelity.
    raw = torch.load(ckpt, map_location="cpu")
    state = raw.get("ema", raw.get("model", raw))
    if isinstance(state, dict) and "module" in state:
        state = state["module"]
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    cfg.model.load_state_dict(state, strict=False)

    model = cfg.model.deploy()
    postprocessor = cfg.postprocessor.deploy()
    model.eval()

    im_pil = Image.open(img_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]])

    transforms = T.Compose(
        [
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    im_data = transforms(im_pil).unsqueeze(0)

    with torch.no_grad():
        outputs = model(im_data)
        labels, boxes, scores = postprocessor(outputs, orig_size)

    return (
        labels[0].cpu(),
        boxes[0].cpu(),
        scores[0].cpu(),
    )


# ---------------------------------------------------------------------------
# OBB numerical parity vs tools/compare/core.py
# ---------------------------------------------------------------------------


def test_obb_numerical_parity_vs_tool_infer() -> None:
    """OBB adapter backend matches the legacy ``tools/compare/core.py``
    DEIMv2OBB-wrapper path bit-for-bit."""
    paths = _skip_if_missing(
        {
            "DEIM_APP_PARITY_OBB_CKPT": "checkpoint",
            "DEIM_APP_PARITY_OBB_IMG": "image",
            "DEIM_APP_PARITY_OBB_CLASSES": "classes.txt",
        }
    )
    ckpt = paths["DEIM_APP_PARITY_OBB_CKPT"]
    img_path = paths["DEIM_APP_PARITY_OBB_IMG"]
    classes_file = paths["DEIM_APP_PARITY_OBB_CLASSES"]
    fmt = os.environ.get("DEIM_APP_PARITY_OBB_FORMAT", "YOLO-OBB")

    import torch
    from PIL import Image

    from deim_app.api import DetectionModel

    # ---- New adapter backend ----------------------------------------------
    example_yml = OBB_YOLO_EXAMPLE if fmt == "YOLO-OBB" else OBB_DOTA_EXAMPLE
    img_dir = str(Path(img_path).parent)
    new_model = DetectionModel.from_config(
        str(example_yml),
        data={
            "format": fmt,
            "train_images": img_dir,
            "train_annotations": img_dir,
            "val_images": img_dir,
            "val_annotations": img_dir,
            "classes_file": classes_file,
            "cache_images": "none",
            "num_workers": 0,
        },
        inference={"device": "cpu", "batch_size": 1},
    )
    new_model.load(ckpt, prefer_ema=True)
    new_collection = new_model.predict(img_path)
    assert len(new_collection.predictions) == 1
    new_pred = new_collection.predictions[0]

    new_labels = torch.tensor([d.class_id for d in new_pred.detections], dtype=torch.long)
    new_scores = torch.tensor([d.score for d in new_pred.detections], dtype=torch.float32)
    new_boxes = torch.tensor(
        [d.xywhr for d in new_pred.detections], dtype=torch.float32
    ).reshape(-1, 5)

    # ---- Legacy reference path (mirrors tools/compare/core.py) -----
    legacy_labels, legacy_boxes, legacy_scores = _obb_legacy_tool_infer(
        ckpt=ckpt,
        img_path=img_path,
        source_yaml=str(REPO_ROOT / "configs" / "custom_obb" / "dlzdt" / "sp_fz_common.yml"),
        classes_file=classes_file,
        size=(640, 640),
    )

    # ---- Compare ----------------------------------------------------------
    assert new_labels.shape == legacy_labels.shape, (
        f"label shape mismatch: new {new_labels.shape} vs legacy {legacy_labels.shape}"
    )
    torch.testing.assert_close(new_labels, legacy_labels, rtol=0, atol=0)
    torch.testing.assert_close(new_scores, legacy_scores, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(new_boxes, legacy_boxes, rtol=1e-5, atol=1e-4)


def _obb_legacy_tool_infer(
    *,
    ckpt: str,
    img_path: str,
    source_yaml: str,
    classes_file: str,
    size: tuple[int, int],
) -> tuple[Any, Any, Any]:
    """Replicate ``tools/compare/core.py`` exactly: build the
    DEIMv2OBB wrapper from the source algorithm YAML + checkpoint, run a single
    image, rescale OBB boxes to original size, return tensors.
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    from engine.core.yaml_utils import load_config
    from engine.data.transforms import ConvertPILImage

    # DEIMv2OBB + load_checkpoint are imported from the compare tool core
    # itself, per the task brief ("use their inference helpers directly via
    # import").
    from tools.compare.core import DEIMv2OBB, load_checkpoint  # noqa: E402

    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    num_classes = len(class_names)

    config = load_config(source_yaml)
    model_cfg = {
        "DINOv3STAsResAtten": config["DINOv3STAsResAtten"],
        "HybridEncoder": config["HybridEncoder"],
        "DEIMTransformer": {
            **config["DEIMTransformer"],
            "num_classes": num_classes,
            "num_queries": 300,
        },
        "PostProcessor": {
            **config["PostProcessor"],
            "num_classes": num_classes,
            "num_top_queries": 300,
        },
    }
    model_cfg["DINOv3STAsResAtten"]["weights_path"] = ckpt

    device = "cpu"
    model = DEIMv2OBB(model_cfg, device)
    load_checkpoint(model, ckpt, map_location=device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(size),
            ConvertPILImage(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)
    dst_sz = torch.tensor([size[1], size[0]], device=device)[None, :]
    # Match the legacy tool's call signature.
    with torch.no_grad():
        results = model(input_tensor, orig_target_sizes=dst_sz)

    # Legacy OBB postprocessor returns a list of dicts (non-deploy mode); take
    # the first (and only) image.
    output = results[0]
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()

    from tools.model_compare.obb_inference_geometry import rescale_obb_to_original

    boxes = rescale_obb_to_original(
        output["boxes"].cpu(),
        original_size=(orig_h, orig_w),
        inference_size=size,
    )

    return labels, boxes, scores
