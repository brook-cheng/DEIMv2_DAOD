import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore")


def mal_loss(p, q, y, gamma=1.5, mal_alpha=None):
    """
    MAL Loss 计算

    Args:
        p: pred_score = sigmoid(logits), 预测概率 [N, ], ∈ [0, 1]
        q: target_score, 目标分数 (IoU-aware) [N, ], ∈ [0, 1]
        y: target, 真实标签 (one-hot 后的正样本位置) [N, ], ∈ {0, 1}
        gamma: 聚焦参数，控制难例权重
        mal_alpha: alpha 参数，None 表示不使用

    Returns:
        loss: MAL loss 值
    """
    # w = p^γ * (1-y) + y
    if mal_alpha != None:
        w = mal_alpha * p.pow(gamma) * (1 - y) + y
    else:
        w = p.pow(gamma) * (1 - y) + y

    # loss(y,p,q) = -w * (q^γ * log(p) + (1-q^γ) * log(1-p))
    # 注意：这里 p 已经是概率，所以直接用 binary_cross_entropy
    # 但原代码用的是 binary_cross_entropy_with_logits，所以需要转换
    # 为了可视化，我们直接计算数学公式

    # 避免 log(0)，添加小的 epsilon
    eps = 1e-7
    p = torch.clamp(p, eps, 1 - eps)

    # 标准 BCE 形式
    loss = -(q.pow(gamma) * torch.log(p) + (1 - q.pow(gamma)) * torch.log(1 - p))
    loss = loss * w

    return loss


def mal_loss_gradients(p, q, y, gamma=1.5, mal_alpha=None):
    """
    MAL Loss 的梯度计算

    Args:
        p: pred_score = sigmoid(logits), 预测概率 [N, ], ∈ [0, 1]
        q: target_score, 目标分数 (IoU-aware) [N, ], ∈ [0, 1]
        y: target, 真实标签 (one-hot 后的正样本位置) [N, ], ∈ {0, 1}
        gamma: 聚焦参数，控制难例权重
        mal_alpha: alpha 参数，None 表示不使用

    Returns:
        loss: MAL loss 值
    """
    if y == 1:
        grad = p - q.pow(gamma)
    elif y == 0:
        grad = p.pow(gamma)(p - gamma * (1 - p) * torch.log(1 - p))
    else:
        raise ValueError("y must be 0 or 1")

    return grad


def visual_mal_loss(p, q, y, gamma=1.5, mal_alpha=None):
    """
    可视化 MAL Loss 在不同参数下的行为

    Args:
        p: 预测概率范围
        q: 目标分数 (IoU)
        y: 真实标签 (0 或 1)
        gamma: 聚焦参数
        mal_alpha: alpha 参数
    """

    # 创建网格
    p_range = torch.linspace(0.01, 0.99, 100)  # 预测概率
    q_range = torch.linspace(0.01, 0.99, 100)  # 目标 IoU
    P, Q = torch.meshgrid(p_range, q_range, indexing="ij")

    # 计算 loss
    Y = torch.ones_like(P) * y  # 固定 y
    Loss = mal_loss(P, Q, Y, gamma, mal_alpha)

    # 创建图形
    fig = plt.figure(figsize=(16, 6))

    # === 图 1: 3D 表面图 ===
    ax1 = fig.add_subplot(131, projection="3d")
    surf = ax1.plot_surface(
        P.numpy(), Q.numpy(), Loss.numpy(), cmap="viridis", alpha=0.8, edgecolor="none"
    )
    ax1.set_xlabel("Predicted Probability (p)", fontsize=10)
    ax1.set_ylabel("Target IoU (q)", fontsize=10)
    ax1.set_zlabel("MAL Loss", fontsize=10)
    ax1.set_title(
        f"MAL Loss 3D Surface\n(y={y}, γ={gamma}, α={mal_alpha})",
        fontsize=12,
        fontweight="bold",
    )
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

    # === 图 2: 等高线图 ===
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(
        P.numpy(), Q.numpy(), Loss.numpy(), levels=50, cmap="viridis"
    )
    ax2.set_xlabel("Predicted Probability (p)", fontsize=10)
    ax2.set_ylabel("Target IoU (q)", fontsize=10)
    ax2.set_title(
        f"MAL Loss Contour\n(y={y}, γ={gamma}, α={mal_alpha})",
        fontsize=12,
        fontweight="bold",
    )
    fig.colorbar(contour, ax=ax2)

    # === 图 3: 不同 q 值下的 loss 曲线 ===
    ax3 = fig.add_subplot(133)
    q_values = [0.3, 0.5, 0.7, 0.9]  # 不同的 IoU 值
    colors = ["red", "orange", "green", "blue"]

    for q_val, color in zip(q_values, colors):
        Q_fixed = torch.ones_like(p_range) * q_val
        Y_fixed = torch.ones_like(p_range) * y
        loss_curve = mal_loss(p_range, Q_fixed, Y_fixed, gamma, mal_alpha)
        ax3.plot(
            p_range.numpy(),
            loss_curve.numpy(),
            color=color,
            linewidth=2,
            label=f"q={q_val}",
        )

    ax3.set_xlabel("Predicted Probability (p)", fontsize=10)
    ax3.set_ylabel("MAL Loss", fontsize=10)
    ax3.set_title(
        f"Loss vs Prediction (y={y})\nDifferent Target IoU",
        fontsize=12,
        fontweight="bold",
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"mal_loss_y{y}_gamma{gamma}_alpha{mal_alpha}.png", dpi=300, bbox_inches="tight"
    )
    # plt.show()

    print(f"✓ 已保存可视化：mal_loss_y{y}_gamma{gamma}_alpha{mal_alpha}.png")


def visual_mal_loss_comparison():
    """
    对比不同参数设置下的 MAL Loss
    """

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    p_range = torch.linspace(0.01, 0.99, 100)
    q_values = [0.5, 0.7, 0.9]  # 低、中、高 IoU

    # === 第一行：正样本 (y=1) ===
    y = 1
    configs = [
        {"gamma": 0.0, "mal_alpha": None, "title": "γ=0 (无聚焦)"},
        {"gamma": 1.5, "mal_alpha": None, "title": "γ=1.5 (默认)"},
        {"gamma": 2.0, "mal_alpha": None, "title": "γ=2.0 (强聚焦)"},
    ]

    for idx, config in enumerate(configs):
        ax = axes[0, idx]
        for q_val, color in zip(q_values, ["red", "green", "blue"]):
            Q_fixed = torch.ones_like(p_range) * q_val
            Y_fixed = torch.ones_like(p_range) * y
            loss_curve = mal_loss(
                p_range, Q_fixed, Y_fixed, config["gamma"], config["mal_alpha"]
            )
            ax.plot(
                p_range.numpy(),
                loss_curve.numpy(),
                color=color,
                linewidth=2,
                label=f"q={q_val}",
            )

        ax.set_xlabel("Predicted Probability (p)", fontsize=10)
        ax.set_ylabel("MAL Loss", fontsize=10)
        ax.set_title(
            f'Positive Sample (y=1)\n{config["title"]}', fontsize=12, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    # === 第二行：负样本 (y=0) ===
    y = 0
    configs_neg = [
        {"gamma": 0.0, "mal_alpha": None, "title": "γ=0, α=None"},
        {"gamma": 1.5, "mal_alpha": None, "title": "γ=1.5, α=None"},
        {"gamma": 1.5, "mal_alpha": 0.25, "title": "γ=1.5, α=0.25"},
    ]

    for idx, config in enumerate(configs_neg):
        ax = axes[1, idx]
        for q_val, color in zip(q_values, ["red", "green", "blue"]):
            Q_fixed = torch.ones_like(p_range) * q_val
            Y_fixed = torch.ones_like(p_range) * y
            loss_curve = mal_loss(
                p_range, Q_fixed, Y_fixed, config["gamma"], config["mal_alpha"]
            )
            ax.plot(
                p_range.numpy(),
                loss_curve.numpy(),
                color=color,
                linewidth=2,
                label=f"q={q_val}",
            )

        ax.set_xlabel("Predicted Probability (p)", fontsize=10)
        ax.set_ylabel("MAL Loss", fontsize=10)
        ax.set_title(
            f'Negative Sample (y=0)\n{config["title"]}', fontsize=12, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("MAL Loss Behavior Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("mal_loss_comparison.png", dpi=300, bbox_inches="tight")
    # plt.show()

    print("✓ 已保存对比图：mal_loss_comparison.png")


def visual_weight_distribution():
    """
    可视化权重 w 的分布
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    p_range = torch.linspace(0.01, 0.99, 100)

    # === 图 1: 不同 gamma 下的权重 (正样本) ===
    ax = axes[0]
    y = 1
    gamma_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    for gamma in gamma_values:
        w = p_range.pow(gamma) * (1 - y) + y
        ax.plot(p_range.numpy(), w.numpy(), linewidth=2, label=f"γ={gamma}")

    ax.set_xlabel("Predicted Probability (p)", fontsize=10)
    ax.set_ylabel("Weight (w)", fontsize=10)
    ax.set_title("Weight for Positive Samples (y=1)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # === 图 2: 不同 gamma 下的权重 (负样本) ===
    ax = axes[1]
    y = 0
    for gamma in gamma_values:
        w = p_range.pow(gamma) * (1 - y) + y
        ax.plot(p_range.numpy(), w.numpy(), linewidth=2, label=f"γ={gamma}")

    ax.set_xlabel("Predicted Probability (p)", fontsize=10)
    ax.set_ylabel("Weight (w)", fontsize=10)
    ax.set_title("Weight for Negative Samples (y=0)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # === 图 3: alpha 的影响 (负样本) ===
    ax = axes[2]
    y = 0
    gamma = 1.5
    alpha_values = [None, 0.1, 0.25, 0.5]
    for alpha in alpha_values:
        if alpha is None:
            w = p_range.pow(gamma) * (1 - y) + y
            label = "α=None"
        else:
            w = alpha * p_range.pow(gamma) * (1 - y) + y
            label = f"α={alpha}"
        ax.plot(p_range.numpy(), w.numpy(), linewidth=2, label=label)

    ax.set_xlabel("Predicted Probability (p)", fontsize=10)
    ax.set_ylabel("Weight (w)", fontsize=10)
    ax.set_title(f"Effect of Alpha (y=0, γ={gamma})", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Weight Distribution Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("mal_loss_weights.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ 已保存权重图：mal_loss_weights.png")


def visualize_mal_loss(save_dir="./mal_loss_vis", show=True):
    """
    主函数，执行 MAL Loss 的完整可视化分析

    Args:
        save_dir: 保存可视化结果的目录
        show: 是否立即显示图像

    Returns:
        dict: 生成的文件路径列表
    """
    import os
    from pathlib import Path

    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" " * 20 + "MAL Loss 可视化分析工具")
    print("=" * 70)
    print(f"\n📁 保存目录：{save_path.absolute()}")
    print(f"🎯 参数设置：gamma=1.5, alpha=None (默认)")
    print("-" * 70)

    generated_files = []

    # =========================================================================
    # 第一部分：单个样本类型的详细分析
    # =========================================================================
    print("\n【Part 1】单个样本类型的损失曲面分析")
    print("-" * 70)

    for y_val in [1, 0]:
        sample_type = "Positive" if y_val == 1 else "Negative"
        print(f"\n▶ 生成 {sample_type} Sample (y={y_val}) 的可视化...")

        # 调用详细可视化函数
        filename = f"mal_loss_detail_y{y_val}.png"
        _plot_single_setting(y=y_val, save_path=save_path / filename, show=show)
        generated_files.append(str(save_path / filename))
        print(f"  ✓ 已保存：{filename}")

    # =========================================================================
    # 第二部分：Gamma 参数影响分析
    # =========================================================================
    print("\n【Part 2】Gamma 参数对损失的影响")
    print("-" * 70)

    filename = "mal_loss_gamma_comparison.png"
    _plot_gamma_comparison(save_path=save_path / filename, show=show)
    generated_files.append(str(save_path / filename))
    print(f"  ✓ 已保存：{filename}")

    # =========================================================================
    # 第三部分：Alpha 参数影响分析
    # =========================================================================
    print("\n【Part 3】Alpha 参数对权重的影响")
    print("-" * 70)

    filename = "mal_loss_alpha_effect.png"
    _plot_alpha_effect(save_path=save_path / filename, show=show)
    generated_files.append(str(save_path / filename))
    print(f"  ✓ 已保存：{filename}")

    # =========================================================================
    # 第四部分：损失曲线对比（不同 IoU）
    # =========================================================================
    print("\n【Part 4】不同目标 IoU 下的损失曲线")
    print("-" * 70)

    filename = "mal_loss_iou_curves.png"
    _plot_iou_curves(save_path=save_path / filename, show=show)
    generated_files.append(str(save_path / filename))
    print(f"  ✓ 已保存：{filename}")

    # =========================================================================
    # 第五部分：权重分布热力图
    # =========================================================================
    print("\n【Part 5】权重分布热力图")
    print("-" * 70)

    filename = "mal_loss_weight_heatmap.png"
    _plot_weight_heatmap(save_path=save_path / filename, show=show)
    generated_files.append(str(save_path / filename))
    print(f"  ✓ 已保存：{filename}")

    # =========================================================================
    # 第六部分：梯度分析
    # =========================================================================
    print("\n【Part 6】损失函数的梯度分析")
    print("-" * 70)

    filename = "mal_loss_gradient.png"
    _plot_gradient_analysis(save_path=save_path / filename, show=show)
    generated_files.append(str(save_path / filename))
    print(f"  ✓ 已保存：{filename}")

    # =========================================================================
    # 总结
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ 所有可视化已完成！")
    print("=" * 70)
    print(f"\n📊 共生成 {len(generated_files)} 个可视化文件:")
    for i, filepath in enumerate(generated_files, 1):
        print(f"   {i}. {Path(filepath).name}")

    print(f"\n💾 保存位置：{save_path.absolute()}")
    print("=" * 70)

    return {"files": generated_files, "save_dir": str(save_path)}


def _plot_single_setting(y, save_path, show=False):
    """绘制单个参数设置的详细图"""
    fig = plt.figure(figsize=(18, 5))

    # 创建网格
    p_range = torch.linspace(0.01, 0.99, 150)
    q_range = torch.linspace(0.01, 0.99, 150)
    P, Q = torch.meshgrid(p_range, q_range, indexing="ij")
    Y = torch.ones_like(P) * y
    gamma = 1.5

    Loss = mal_loss(P, Q, Y, gamma, mal_alpha=None)

    # 图 1: 3D 表面
    ax1 = fig.add_subplot(131, projection="3d")
    surf = ax1.plot_surface(
        P.numpy(), Q.numpy(), Loss.numpy(), cmap="viridis", alpha=0.9, edgecolor="none"
    )
    ax1.set_xlabel("Predicted Probability (p)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Target IoU (q)", fontsize=11, fontweight="bold")
    ax1.set_zlabel("MAL Loss", fontsize=11, fontweight="bold")
    title = f"Positive" if y == 1 else "Negative"
    ax1.set_title(
        f"{title} Sample (y={y})\n3D Loss Surface", fontsize=12, fontweight="bold"
    )
    fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=12, label="Loss Value")

    # 图 2: 等高线
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(
        P.numpy(), Q.numpy(), Loss.numpy(), levels=30, cmap="viridis"
    )
    ax2.set_xlabel("Predicted Probability (p)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Target IoU (q)", fontsize=11, fontweight="bold")
    ax2.set_title(f"Contour Map", fontsize=12, fontweight="bold")
    cbar = fig.colorbar(contour, ax=ax2)
    cbar.set_label("Loss Value", fontsize=10)

    # 图 3: 曲线
    ax3 = fig.add_subplot(133)
    q_values = [0.3, 0.5, 0.7, 0.9]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    for q_val, color in zip(q_values, colors):
        Q_fixed = torch.ones_like(p_range) * q_val
        Y_fixed = torch.ones_like(p_range) * y
        loss_curve = mal_loss(p_range, Q_fixed, Y_fixed, gamma, mal_alpha=None)
        ax3.plot(
            p_range.numpy(),
            loss_curve.numpy(),
            color=color,
            linewidth=2.5,
            label=f"Target IoU q={q_val}",
        )

    ax3.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax3.set_xlabel("Predicted Probability (p)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("MAL Loss", fontsize=11, fontweight="bold")
    ax3.set_title(
        f"Loss vs Prediction\n(Different Target IoU)", fontsize=12, fontweight="bold"
    )
    ax3.legend(loc="best", fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _plot_gamma_comparison(save_path, show=False):
    """绘制 Gamma 参数对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    p_range = torch.linspace(0.01, 0.99, 150)
    gamma_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    q_val = 0.7
    colors = plt.cm.viridis(torch.linspace(0, 1, len(gamma_values)))

    # 正样本
    ax = axes[0]
    for gamma, color in zip(gamma_values, colors):
        Q_fixed = torch.ones_like(p_range) * q_val
        Y_fixed = torch.ones_like(p_range) * 1
        loss_curve = mal_loss(p_range, Q_fixed, Y_fixed, gamma, mal_alpha=None)
        ax.plot(
            p_range.numpy(),
            loss_curve.numpy(),
            color=color,
            linewidth=2.5,
            label=f"γ={gamma}",
        )

    ax.set_xlabel("Predicted Probability (p)", fontsize=11, fontweight="bold")
    ax.set_ylabel("MAL Loss", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Positive Sample (y=1)\nEffect of Gamma (q={q_val})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # 负样本
    ax = axes[1]
    for gamma, color in zip(gamma_values, colors):
        Q_fixed = torch.ones_like(p_range) * q_val
        Y_fixed = torch.ones_like(p_range) * 0
        loss_curve = mal_loss(p_range, Q_fixed, Y_fixed, gamma, mal_alpha=None)
        ax.plot(
            p_range.numpy(),
            loss_curve.numpy(),
            color=color,
            linewidth=2.5,
            label=f"γ={gamma}",
        )

    ax.set_xlabel("Predicted Probability (p)", fontsize=11, fontweight="bold")
    ax.set_ylabel("MAL Loss", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Negative Sample (y=0)\nEffect of Gamma (q={q_val})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("Gamma Parameter Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _plot_alpha_effect(save_path, show=False):
    """绘制 Alpha 参数影响图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    p_range = torch.linspace(0.01, 0.99, 150)
    gamma = 1.5
    y = 0
    alpha_values = [None, 0.1, 0.25, 0.5, 0.75]
    colors = plt.cm.plasma(torch.linspace(0, 1, len(alpha_values)))

    for alpha, color in zip(alpha_values, colors):
        if alpha is None:
            w = p_range.pow(gamma) * (1 - y) + y
            label = "α=None (default)"
        else:
            w = alpha * p_range.pow(gamma) * (1 - y) + y
            label = f"α={alpha}"
        ax.plot(p_range.numpy(), w.numpy(), color=color, linewidth=2.5, label=label)

    ax.set_xlabel("Predicted Probability (p)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Weight (w)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Effect of Alpha on Weight Distribution\n(Negative Sample, γ={gamma})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _plot_iou_curves(save_path, show=False):
    """绘制不同 IoU 下的损失曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    p_range = torch.linspace(0.01, 0.99, 150)
    q_values = [0.2, 0.4, 0.6, 0.8, 0.95]
    gamma = 1.5
    colors = plt.cm.rainbow(torch.linspace(0, 1, len(q_values)))

    # 正样本
    ax = axes[0]
    for q_val, color in zip(q_values, colors):
        Q_fixed = torch.ones_like(p_range) * q_val
        Y_fixed = torch.ones_like(p_range) * 1
        loss_curve = mal_loss(p_range, Q_fixed, Y_fixed, gamma, mal_alpha=None)
        ax.plot(
            p_range.numpy(),
            loss_curve.numpy(),
            color=color,
            linewidth=2.5,
            label=f"q={q_val}",
        )
        # 标记最优点
        ax.axvline(x=q_val, color=color, linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("Predicted Probability (p)", fontsize=11, fontweight="bold")
    ax.set_ylabel("MAL Loss", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Positive Sample (y=1)\nLoss Curves for Different Target IoU",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # 负样本
    ax = axes[1]
    for q_val, color in zip(q_values, colors):
        Q_fixed = torch.ones_like(p_range) * q_val
        Y_fixed = torch.ones_like(p_range) * 0
        loss_curve = mal_loss(p_range)
        ax.plot(
            p_range.numpy,
            Q_fixed,
            Y_fixed,
            gamma(),
            loss_curve.numpy,
            mal_alpha=None(),
            color=color,
            linewidth=2.5,
            label=f"q={q_val}",
        )

    ax.set_xlabel("Predicted Probability (p)", fontsize=11, fontweight="bold")
    ax.set_ylabel("MAL Loss", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Negative Sample (y=0)\nLoss Curves for Different Target IoU",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("Target IoU Impact Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _plot_weight_heatmap(save_path, show=False):
    """绘制权重分布热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    p_range = torch.linspace(0.01, 0.99, 150)
    gamma_values = [0.5, 1.5, 2.0]

    for idx, gamma in enumerate(gamma_values):
        # 正样本权重
        ax = axes[0] if idx == 0 else None
        if idx == 0:
            w_pos = torch.ones_like(p_range)  # y=1 时 w 恒为 1
            ax.plot(
                p_range.numpy(),
                w_pos.numpy(),
                "g-",
                linewidth=3,
                label=f"Positive (y=1), γ={gamma}",
            )
            ax.fill_between(p_range.numpy(), 0, w_pos.numpy(), alpha=0.3, color="green")

        # 负样本权重
        ax = axes[1] if idx == 0 else None
        if idx == 0:
            w_neg = p_range.pow(gamma)  # y=0 时 w = p^γ
            ax.plot(
                p_range.numpy(),
                w_neg.numpy(),
                "r-",
                linewidth=3,
                label=f"Negative (y=0), γ={gamma}",
            )
            ax.fill_between(p_range.numpy(), 0, w_neg.numpy(), alpha=0.3, color="red")

    # 正样本图
    axes[0].set_xlabel("Predicted Probability (p)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Weight (w)", fontsize=11, fontweight="bold")
    axes[0].set_title(
        "Weight for Positive Samples (y=1)", fontsize=12, fontweight="bold"
    )
    axes[0].legend(loc="best", fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].set_ylim([0, 1.2])

    # 负样本图
    axes[1].set_xlabel("Predicted Probability (p)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Weight (w)", fontsize=11, fontweight="bold")
    axes[1].set_title(
        "Weight for Negative Samples (y=0)", fontsize=12, fontweight="bold"
    )
    axes[1].legend(loc="best", fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].set_ylim([0, 1.2])

    plt.suptitle("Weight Distribution Heatmap", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _plot_gradient_analysis(save_path, show=False):
    """绘制梯度分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    p_range = torch.linspace(0.01, 0.99, 150)
    gamma = 1.5
    q_val = 0.7
    eps = 1e-7

    # 正样本梯度
    ax = axes[0]
    Q_fixed = torch.ones_like(p_range) * q_val
    Y_fixed = torch.ones_like(p_range) * 1
    p_clamped = torch.clamp(p_range, eps, 1 - eps)

    # 计算梯度 d(loss)/dp
    q_gamma = Q_fixed.pow(gamma)
    w = p_range.pow(gamma) * (1 - Y_fixed) + Y_fixed
    gradient_pos = w * (-q_gamma / p_clamped + (1 - q_gamma) / (1 - p_clamped))

    ax.plot(
        p_range.numpy(),
        gradient_pos.numpy(),
        "b-",
        linewidth=2.5,
        label=f"Gradient dL/dp",
    )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(
        x=q_val, color="red", linestyle=":", linewidth=2, label=f"Target q={q_val}"
    )

    ax.set_xlabel("Predicted Probability (p)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gradient (dL/dp)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Gradient Analysis - Positive Sample (y=1)\nq={q_val}, γ={gamma}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # 负样本梯度
    ax = axes[1]
    Y_fixed = torch.ones_like(p_range) * 0
    w = p_range.pow(gamma) * (1 - Y_fixed) + Y_fixed
    gradient_neg = w * (-q_gamma / p_clamped + (1 - q_gamma) / (1 - p_clamped))

    ax.plot(
        p_range.numpy(),
        gradient_neg.numpy(),
        "r-",
        linewidth=2.5,
        label=f"Gradient dL/dp",
    )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=0.0, color="green", linestyle=":", linewidth=2, label=f"Target (p→0)")

    ax.set_xlabel("Predicted Probability (p)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gradient (dL/dp)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Gradient Analysis - Negative Sample (y=0)\nq={q_val}, γ={gamma}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("Gradient Magnitude Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MAL Loss Visualization Tool")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./mal_loss_vis",
        help="Directory to save visualization results",
    )
    parser.add_argument("--show", action="store_true", help="Show plots immediately")
    parser.add_argument(
        "--single", action="store_true", help="Run single analysis only"
    )

    args = parser.parse_args()

    if args.single:
        # 单次分析模式
        print("运行单次分析模式...")
        _plot_gradient_analysis(save_dir=args.save_dir, show=args.show)
    else:
        # 完整分析模式
        print("运行完整分析模式...")
        visualize_mal_loss(save_dir=args.save_dir, show=args.show)
