from copy import deepcopy
from pathlib import Path

import torch

from engine.core.yaml_config import YAMLConfig
from engine.deim.deim_criterion import DEIMCriterion
from engine.solver.kendall import KendallWeighting


CONFIG = Path(__file__).resolve().parents[1] / "configs/custom_obb/deimv2_obb_sp.yml"
FAMILIES = ("loss_bbox", "loss_probiou", "loss_angle", "loss_kld")


def _prediction(boxes: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "pred_boxes": boxes.unsqueeze(0),
        "pred_logits": torch.tensor([[[8.0], [7.0]]]),
    }


def test_configured_geometry_forward_and_kendall_suffixes():
    cfg = YAMLConfig(str(CONFIG))
    model = cfg.model
    criterion = cfg.criterion
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
    assert model is not None and criterion.matcher is not None

    kendall_cfg = cfg.yaml_cfg["KendallWeighting"]
    kendall = KendallWeighting(kendall_cfg["loss_names"], kendall_cfg["init_log_sigma"])
    assert torch.isfinite(kendall.weighted_loss(losses))


def test_explicit_legacy_fixture_keeps_parameter_space_losses():
    configured = YAMLConfig(str(CONFIG)).criterion
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
