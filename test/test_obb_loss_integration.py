"""OBB criterion forward integration against synthetic_exp_001.yml.

These tests build only the DEIMCriterion (and its injected matcher) from the
synthetic config -- never the full DEIM model -- so they need no DINOv3
checkpoint, dataset, or GPU.

Isolation note: ``engine.core.yaml_utils.load_config`` uses a mutable default
argument (``cfg=dict()``) that accumulates across calls in one process
(upstream RT-DETR bug, not changed here). Any earlier test that loads an OBB
config through it -- notably ``test_obb_config_contract``, which rglobs every
config including ``syn_ablation_adr.yml`` (the sole config setting
``DEIMCriterion.adr_loss=true``) -- leaves ``adr_loss`` in the accumulator, and
it then leaks into this config's resolved DEIMCriterion section. Every load in
this file therefore goes through ``_isolated_synthetic_cfg``, which resets the
accumulator first. See ``test_synthetic_criterion_unpolluted_after_adr_ablation_load``
for the regression guard.
"""

from copy import deepcopy
from pathlib import Path

import torch

from engine.core import yaml_utils
from engine.core.yaml_config import YAMLConfig
from engine.deim.deim_criterion import DEIMCriterion
from engine.solver.kendall import KendallWeighting


CONFIG = Path(__file__).resolve().parents[1] / "configs/custom_obb/synthetic_configs/synthetic_exp_001.yml"
ADR_ABLATION = CONFIG.parent / "ablation" / "syn_ablation_adr.yml"
FAMILIES = ("loss_bbox", "loss_kld")


def _isolated_synthetic_cfg() -> YAMLConfig:
    """Load synthetic_exp_001.yml from a clean load_config accumulator.

    ``load_config(file_path, cfg=dict())`` reuses one dict across calls, so a
    prior load of another OBB config pollutes this one. Resetting
    ``__defaults__`` to a fresh dict (the workaround already used by
    ``test_early_stopping_configs``) restores per-call isolation.
    """
    yaml_utils.load_config.__defaults__ = ({},)
    return YAMLConfig(str(CONFIG))


def _prediction(boxes: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "pred_boxes": boxes.unsqueeze(0),
        "pred_logits": torch.tensor([[[8.0], [7.0]]]),
    }


def test_configured_geometry_forward_and_kendall_suffixes():
    cfg = _isolated_synthetic_cfg()
    criterion = cfg.criterion
    assert isinstance(criterion, DEIMCriterion)
    criterion.losses = ["boxes"]
    boxes = torch.tensor([[0.45, 0.45, 0.20, 0.30, 0.10], [0.75, 0.70, 0.15, 0.25, 0.40]])
    target = {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.40, 0.40, 0.30, 0.20, 0.00]])}
    base = _prediction(boxes)
    outputs = {
        **base,
        "aux_outputs": [deepcopy(base)],
        "enc_aux_outputs": [deepcopy(base)],
        "enc_meta": {"class_agnostic": False},
        "pre_outputs": deepcopy(base),
        "dn_outputs": [deepcopy(base)],
        "dn_meta": {"dn_positive_idx": [torch.tensor([0])], "dn_num_group": 1},
    }

    losses = criterion(outputs, [target], epoch=0)
    suffixes = ("", "_aux_0", "_enc_0", "_pre", "_dn_0")
    expected = {f"{family}{suffix}" for family in FAMILIES for suffix in suffixes}
    assert expected <= losses.keys()
    assert all(torch.isfinite(losses[key]) for key in expected)
    assert criterion.matcher is not None

    kendall_cfg = cfg.yaml_cfg["KendallWeighting"]
    # Must mirror the fallback in engine/solver/det_solver.py (KendallWeighting).
    loss_names = kendall_cfg.get(
        "loss_names", ["loss_mal", "loss_bbox", "loss_kld", "loss_fgl"]
    )
    kendall = KendallWeighting(loss_names, kendall_cfg["init_log_sigma"])
    assert torch.isfinite(kendall.weighted_loss(losses))


def test_explicit_legacy_fixture_keeps_parameter_space_losses():
    configured = _isolated_synthetic_cfg().criterion
    assert isinstance(configured, DEIMCriterion)
    legacy = DEIMCriterion(
        matcher=configured.matcher,
        weight_dict={"loss_bbox": 1, "loss_kld": 1},
        losses=["boxes"],
        num_classes=1,
        box_mode="obb",
    )
    boxes = torch.tensor([[0.65, 0.60, 0.25, 0.15, 0.30], [0.20, 0.20, 0.10, 0.12, 0.00]])
    target = {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.40, 0.40, 0.15, 0.25, 0.10]])}
    base = _prediction(boxes)
    indices = legacy.matcher(base, [target], epoch=0)["indices"]

    losses = legacy.loss_boxes(base, [target], indices, num_boxes=1.0)
    assert losses.keys() == {"loss_bbox", "loss_kld"}
    assert all(torch.isfinite(value) for value in losses.values())


def test_synthetic_criterion_unpolluted_after_adr_ablation_load():
    """Regression: loading syn_ablation_adr.yml (the sole OBB config with
    DEIMCriterion.adr_loss=true) before synthetic_exp_001.yml must not leak
    adr_loss into the synthetic criterion via load_config's process-global
    mutable-default accumulator. Without isolation this returns adr_loss=True
    and the criterion switches to the ADR loss family (loss_extrect_*).
    """
    assert ADR_ABLATION.exists(), f"missing adr ablation config: {ADR_ABLATION}"

    YAMLConfig(str(ADR_ABLATION)).global_cfg

    crit_cfg = _isolated_synthetic_cfg().yaml_cfg["DEIMCriterion"]
    assert not crit_cfg.get("adr_loss"), (
        f"adr_loss leaked into synthetic_exp_001 via load_config accumulator: "
        f"{crit_cfg.get('adr_loss')!r}"
    )
