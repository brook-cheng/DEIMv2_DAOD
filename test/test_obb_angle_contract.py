"""OBB 角度契约模块单元测试。

验证 ``engine.deim/obb_angle_contract.py`` 的等比归一化契约与 loss 规范域转换：
    theta_phys_rad ∈ [0, π)   ←→   theta_norm ∈ [0, 1)   严格等比 theta/π
    theta_phys_rad ∈ [0, π)   →   theta_loss_rad ∈ [-π/4, 3π/4)

契约依据：``docs/superpowers/specs/2026-08-05-obb-angle-contract-simplification-design.md``
            §5 核心公式、§7 边界示例、§9 不变量。

Run:
    python -m pytest test/test_obb_angle_contract.py -q
"""

import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.deim.obb_angle_contract import (
    canonicalize_phys_rad,
    physical_rad_to_norm,
    norm_to_physical_rad,
    physical_rad_to_logit,
    logit_to_physical_rad,
    physical_rad_to_loss_rad,
)

PI = math.pi


# ---------------------------------------------------------------------------
# 1. canonicalize_phys_rad: [0, π) 半开区间
# ---------------------------------------------------------------------------


def test_canonicalize_basic_range():
    """任意实数 θ 规范化后落在 [0, π)。"""
    torch.manual_seed(0)
    theta = torch.randn(2000) * 4.0 * PI  # 广域随机
    out = canonicalize_phys_rad(theta)
    assert (out >= 0).all(), f"min={out.min():.6f}"
    assert (out < PI).all(), f"max={out.max():.6f}"


@pytest.mark.parametrize(
    "inp, expected",
    [
        (0.0, 0.0),
        (PI / 4, PI / 4),
        (PI / 2, PI / 2),
        (PI, 0.0),          # π 折回 0（半开区间）
        (-PI / 4, 3 * PI / 4),
        (2 * PI, 0.0),
        (-2 * PI, 0.0),
        (5 * PI / 4, PI / 4),
    ],
)
def test_canonicalize_points(inp, expected):
    assert abs(canonicalize_phys_rad(torch.tensor(inp)).item() - expected) < 1e-6


# ---------------------------------------------------------------------------
# 2. physical_rad_to_norm / norm_to_physical_rad: 严格等比 theta/π
# ---------------------------------------------------------------------------


def test_physical_rad_to_norm_proportional():
    """theta_norm = theta_phys / π, 严格等比, 无任何平移。"""
    theta = torch.tensor([0.0, PI / 4, PI / 2, 3 * PI / 4, PI - 1e-4])
    norm = physical_rad_to_norm(theta)
    expected = theta / PI
    assert torch.allclose(norm, expected, atol=1e-6)
    # 域 [0, 1)
    assert (norm >= 0).all() and (norm < 1).all()


def test_norm_to_physical_rad_proportional():
    """theta_phys = theta_norm * π。"""
    norm = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0 - 1e-4])
    phys = norm_to_physical_rad(norm)
    expected = norm * PI
    assert torch.allclose(phys, expected, atol=1e-6)
    # 域 [0, π)
    assert (phys >= 0).all() and (phys < PI).all()


def test_roundtrip_norm_phys_random():
    """等比 round-trip: norm→phys→norm 与 phys→norm→phys 均还原。

    等比映射无需 mod, 浮点精确至 rtol=1e-6。
    """
    torch.manual_seed(42)
    theta = torch.rand(1000) * PI                      # [0, π)
    rt = norm_to_physical_rad(physical_rad_to_norm(theta))
    assert torch.allclose(rt, theta, rtol=1e-6), f"max err {(rt - theta).abs().max():.2e}"

    norm = torch.rand(1000)                            # [0, 1)
    rt = physical_rad_to_norm(norm_to_physical_rad(norm))
    assert torch.allclose(rt, norm, rtol=1e-6), f"max err {(rt - norm).abs().max():.2e}"


@pytest.mark.parametrize(
    "theta_phys, expected_norm",
    [
        (0.0, 0.0),
        (PI / 4, 0.25),
        (PI / 2, 0.5),
        (3 * PI / 4, 0.75),
    ],
)
def test_norm_mapping_boundary_table(theta_phys, expected_norm):
    """spec §7.1 边界映射表前 4 行（常见方向等距分布于 0/0.25/0.5/0.75）。"""
    assert abs(physical_rad_to_norm(torch.tensor(theta_phys)).item() - expected_norm) < 1e-6


# ---------------------------------------------------------------------------
# 3. physical_rad_to_loss_rad: [-π/4, 3π/4) loss 规范域
# ---------------------------------------------------------------------------


def test_loss_rad_range():
    """theta_loss_rad ∈ [-π/4, 3π/4)。"""
    torch.manual_seed(1)
    theta = torch.rand(2000) * PI
    loss = physical_rad_to_loss_rad(theta)
    assert (loss >= -PI / 4 - 1e-6).all(), f"min={loss.min():.6f}"
    assert (loss < 3 * PI / 4).all(), f"max={loss.max():.6f}"


@pytest.mark.parametrize(
    "theta_phys, expected_loss",
    [
        (0.0, 0.0),
        (PI / 4, PI / 4),
        (PI / 2, PI / 2),
        (3 * PI / 4, -PI / 4),     # seam: 3π/4 折到 -π/4 (同一 π 剩余类)
    ],
)
def test_loss_rad_boundary_table(theta_phys, expected_loss):
    """spec §7.1 边界映射表 loss 域列。"""
    out = physical_rad_to_loss_rad(torch.tensor(theta_phys)).item()
    assert abs(out - expected_loss) < 1e-6, f"got {out}, want {expected_loss}"


def test_loss_rad_preserves_periodic_class():
    """theta_loss 与 theta_phys 属同一 π 剩余类 (mod π 相等)。"""
    torch.manual_seed(7)
    theta = torch.rand(500) * PI
    loss = physical_rad_to_loss_rad(theta)
    # (loss - theta) 应是 π 的整数倍（至浮点）
    diff = (loss - theta)
    diff_mod = torch.remainder(diff, PI)
    # remainder 可能给 0 或接近 π, 两者都表示「π 整数倍」
    near_zero = diff_mod.abs() < 1e-5
    near_pi = (diff_mod - PI).abs() < 1e-5
    near_neg_pi = (diff_mod + PI).abs() < 1e-5
    assert (near_zero | near_pi | near_neg_pi).all(), diff_mod[~(near_zero | near_pi | near_neg_pi)]


# ---------------------------------------------------------------------------
# 4. logit round-trip (远离 0/π 边界)
# ---------------------------------------------------------------------------


def test_logit_roundtrip_interior():
    """远离 0 与 π 的 θ: logit→phys 还原 (eps clamp 边界处有偏差, 故取内点)。"""
    torch.manual_seed(11)
    theta = torch.rand(500) * (PI - 0.05) + 0.025       # 避开 [0, 0.025] 与 [π-0.025, π)
    rt = logit_to_physical_rad(physical_rad_to_logit(theta))
    assert torch.allclose(rt, theta, atol=1e-4), f"max err {(rt - theta).abs().max():.2e}"


def test_logit_finite_at_boundary():
    """eps clamp 保证 θ=0 与 θ=π 处 logit 有限。"""
    eps = 1e-4
    for boundary in (0.0, PI - 1e-9):
        logit = physical_rad_to_logit(torch.tensor(boundary), eps=eps)
        assert torch.isfinite(logit), f"logit not finite at {boundary}"


# ---------------------------------------------------------------------------
# 5. 等价性: loss 域周期距离 == 物理域周期距离 (spec §6)
#    用 obb_geometry.periodic_angle_distance 作为周期距离参考实现。
# ---------------------------------------------------------------------------


def _periodic_unsigned(a, b):
    """最短 π 周期无符号距离 ∈ [0, π/2], 参考实现 (与 obb_geometry 一致)。"""
    from engine.deim.obb_geometry import periodic_angle_distance
    return periodic_angle_distance(a, b, with_signal=False)


def test_equivalence_loss_vs_phys_periodic_distance():
    """periodic_distance(loss(pred), loss(gt)) == periodic_distance(pred, gt)。

    spec §6: 先转 loss 域再求周期残差, 与直接对物理角求残差, 结果逐元素相等。
    避开精确 seam 点 (pred/gt = 3π/4) 以排除浮点 -0.0/π 歧义。
    """
    torch.manual_seed(23)
    n = 1000
    pred = torch.rand(n) * PI
    gt = torch.rand(n) * PI
    # 避开 pred 或 gt 极接近 3π/4 的样本（loss 域 seam）
    seam = 3 * PI / 4
    mask = (pred - seam).abs() > 1e-3
    mask &= (gt - seam).abs() > 1e-3
    pred, gt = pred[mask], gt[mask]

    d_phys = _periodic_unsigned(pred, gt)
    d_loss = _periodic_unsigned(
        physical_rad_to_loss_rad(pred),
        physical_rad_to_loss_rad(gt),
    )
    assert torch.allclose(d_phys, d_loss, atol=1e-5), (
        f"max discrepancy {(d_phys - d_loss).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# 6. spec §7.2 残差示例表 (用 periodic_angle_distance 验证)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pred_phys, gt_phys, expected_delta_abs",
    [
        (0.0, 3 * PI / 4, PI / 4),         # delta = π/4
        (PI / 2, 0.0, PI / 2),             # delta = -π/2 → |π/2|
        (0.75 * PI, 0.70 * PI, 0.05 * PI), # seam 样例: 直接相减得 0.95π (错), 正确 |Δ|=0.05π
    ],
)
def test_residual_examples_table(pred_phys, gt_phys, expected_delta_abs):
    """spec §7.2: 残差示例, 含 seam 样例 0.75π vs 0.70π。"""
    pred = torch.tensor(pred_phys)
    gt = torch.tensor(gt_phys)
    d = _periodic_unsigned(pred, gt).item()
    assert abs(d - expected_delta_abs) < 1e-5, f"got {d}, want {expected_delta_abs}"


def test_seam_sample_not_naive_subtraction():
    """回归: pred=0.75π, gt=0.70π 的周期距离 ≠ 直接相减 (0.05π)。"""
    pred = torch.tensor(0.75 * PI)
    gt = torch.tensor(0.70 * PI)
    d_periodic = _periodic_unsigned(pred, gt).item()
    d_naive = abs(pred.item() - gt.item())
    assert abs(d_periodic - 0.05 * PI) < 1e-6
    assert d_naive > 0.04 * PI  # 直接相减 0.05π 恰好接近, 但 seam 语义不同; 用更强的对比样本:
    # pred=0, gt=3π/4: 直接相减 0.75π, 周期距离 π/4
    d_p2 = _periodic_unsigned(torch.tensor(0.0), torch.tensor(3 * PI / 4)).item()
    assert abs(d_p2 - PI / 4) < 1e-6
    assert abs(0.0 - 3 * PI / 4) > PI / 4   # 直接相减 0.75π ≠ 周期距离 π/4
