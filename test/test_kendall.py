"""Kendall Uncertainty Weighting 单元测试。

验证：初始权重、训练后权重分化、聚合 aux/dn、正则项、完整训练流程。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import yaml

from engine.solver.kendall import KendallWeighting


OBB_FAMILIES = [
    "loss_mal",
    "loss_bbox",
    "loss_probiou",
    "loss_angle",
    "loss_kld",
    "loss_fgl",
]


def test_init_weights():
    kw = KendallWeighting(["loss_a", "loss_b"])
    w = kw.get_weights()
    assert abs(w["loss_a"] - 0.5) < 0.01
    assert abs(w["loss_b"] - 0.5) < 0.01
    assert torch.allclose(kw.log_sigma, torch.zeros(2))
    print("[PASS] test_init_weights: both 0.5")


def test_weights_diverge_after_training():
    """大 loss 应获得更小的权重（σ² 增大 → weight 减小）。"""
    kw = KendallWeighting(["loss_a", "loss_b"])
    opt = torch.optim.Adam([kw.log_sigma], lr=0.01)

    for step in range(200):
        loss_dict = {"loss_a": torch.tensor(100.0), "loss_b": torch.tensor(1.0)}
        loss = kw.weighted_loss(loss_dict)
        opt.zero_grad()
        loss.backward()
        opt.step()

    w = kw.get_weights()
    assert w["loss_a"] < w["loss_b"], f"loss_a={w['loss_a']:.4f} should be < loss_b={w['loss_b']:.4f}"
    print(f"[PASS] test_weights_diverge_after_training: "
          f"w_a={w['loss_a']:.4f} < w_b={w['loss_b']:.4f}")


def test_aggregate_loss():
    kw = KendallWeighting(["loss_a", "loss_b"])
    ld = {
        "loss_a": torch.tensor(1.0), "loss_a_aux_0": torch.tensor(2.0),
        "loss_a_aux_1": torch.tensor(3.0),
        "loss_b": torch.tensor(4.0), "loss_b_dn_0": torch.tensor(5.0),
        "loss_b_enc_0": torch.tensor(6.0), "loss_b_pre": torch.tensor(7.0),
    }
    agg_a = kw._aggregate_loss(ld, "loss_a")
    agg_b = kw._aggregate_loss(ld, "loss_b")
    assert abs(agg_a.item() - 6.0) < 0.01, f"Expected 6.0, got {agg_a.item()}"
    assert abs(agg_b.item() - 22.0) < 0.01, f"Expected 22.0, got {agg_b.item()}"
    print("[PASS] test_aggregate_loss")


def test_obb_families_aggregate_all_supported_suffixes():
    kw = KendallWeighting(OBB_FAMILIES)
    suffixes = ["", "_aux_0", "_dn_0", "_enc_0", "_pre"]
    loss_dict = {
        f"{family}{suffix}": torch.tensor(1.0)
        for family in OBB_FAMILIES
        for suffix in suffixes
    }

    for family in OBB_FAMILIES:
        assert kw._aggregate_loss(loss_dict, family).item() == len(suffixes)


def test_obb_configured_families_have_produced_criterion_keys():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "custom_obb",
        "synthetic_configs", "synthetic_exp_001.yml",
    )
    with open(config_path, encoding="utf-8") as stream:
        kw_cfg = yaml.safe_load(stream)["KendallWeighting"]

    # Must mirror the fallback in engine/solver/det_solver.py (KendallWeighting).
    configured = kw_cfg.get(
        "loss_names", ["loss_mal", "loss_bbox", "loss_kld", "loss_fgl"]
    )
    produced = {
        f"{family}{suffix}": torch.tensor(1.0)
        for family in OBB_FAMILIES
        for suffix in ("", "_aux_0", "_dn_0", "_enc_0", "_pre")
    }

    assert kw_cfg["enabled"] is True
    assert set(configured) <= set(OBB_FAMILIES)
    assert all(family in produced for family in configured)


def test_regularizer_present():
    """log_sigma 正则项应使 weighted_loss 对 log_sigma 有非零梯度。
    注意：s=0 时若 L=1 梯度恰为 0（均衡点），用大 loss 值触发。"""
    kw = KendallWeighting(["loss_a"], init_log_sigma=0.0)
    ld = {"loss_a": torch.tensor(100.0)}
    loss = kw.weighted_loss(ld)
    loss.backward()
    g = kw.log_sigma.grad
    assert g is not None and g.abs().sum() > 0, "log_sigma should have nonzero grad for large loss"
    print("[PASS] test_regularizer_present")


def test_full_train_flow():
    """完整训练流程：Kendall 加权损失递减。"""
    model = nn.ModuleDict({
        "shared": nn.Linear(10, 20, bias=False),
        "head_a": nn.Linear(20, 5, bias=False),
        "head_b": nn.Linear(20, 3, bias=False),
    })
    kw = KendallWeighting(["loss_a", "loss_b"])
    optimizer = torch.optim.SGD(list(model.parameters()) + [kw.log_sigma], lr=0.001)
    x = torch.randn(4, 10)

    initial_loss = None
    for step in range(60):
        h = model["shared"](x)
        out_a = model["head_a"](h)
        out_b = model["head_b"](h)
        loss_dict = {"loss_a": out_a.sum().pow(2), "loss_b": out_b.sum().pow(2)}
        loss = kw.weighted_loss(loss_dict)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == 59:
            final_loss = loss.item()

    assert final_loss < initial_loss, \
        f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
    print(f"[PASS] test_full_train_flow: loss {initial_loss:.4f} → {final_loss:.4f}")


def test_prior_multiplies_initial_weight():
    """prior=2 的 loss 初始权重应为 prior=1 的 2 倍。"""
    kw = KendallWeighting(["loss_a", "loss_b"], prior=[2.0, 1.0])
    w = kw.get_weights()
    assert abs(w["loss_a"] / w["loss_b"] - 2.0) < 0.01, \
        f"Expected w_a/w_b ≈ 2.0, got {w['loss_a']:.4f}/{w['loss_b']:.4f}"
    print(f"[PASS] test_prior_multiplies_initial_weight: w_a={w['loss_a']:.4f} w_b={w['loss_b']:.4f}")


def test_prior_in_regularizer():
    """prior 不同 → 正则梯度不同（s_i 的梯度含 p_i 因子）。"""
    kw1 = KendallWeighting(["loss_a"], prior=[1.0])
    kw2 = KendallWeighting(["loss_a"], prior=[2.0])
    ld = {"loss_a": torch.tensor(100.0)}
    loss1 = kw1.weighted_loss(ld)
    loss2 = kw2.weighted_loss(ld)
    loss1.backward()
    loss2.backward()
    g1 = kw1.log_sigma.grad.item()
    g2 = kw2.log_sigma.grad.item()
    assert abs(g2 / g1 - 2.0) < 0.2, \
        f"Expected grad ratio ≈ 2.0, got {g2/g1:.3f}"
    print(f"[PASS] test_prior_in_regularizer: grad1={g1:.4f} grad2={g2:.4f}")


if __name__ == "__main__":
    test_init_weights()
    test_weights_diverge_after_training()
    test_aggregate_loss()
    test_regularizer_present()
    test_full_train_flow()
    test_prior_multiplies_initial_weight()
    test_prior_in_regularizer()
    print("\n✅ All Kendall tests passed")
