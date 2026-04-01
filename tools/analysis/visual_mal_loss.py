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


def mal_loss_new(p, q, y, gamma=1.5, mal_alpha=None, beta=0.5):
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
    ones_tensor = torch.ones_like(p)
    loss = -(
        q.pow(gamma) * torch.log(p) + (1 - q.pow(gamma)) * torch.log(1 - p)
    ) + beta * (ones_tensor - p) * (ones_tensor - q)
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


def visual_mal_loss(p, q, y, gamma=1.5, mal_alpha=None, beta=0.5, use_new_loss=False):
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
    q_range = torch.linspace(0.01, 1.99, 200)  # 目标 IoU
    P, Q = torch.meshgrid(p_range, q_range, indexing="ij")

    # 计算 loss
    Y = torch.ones_like(P) * y  # 固定 y
    if use_new_loss:
        Loss = mal_loss_new(P, Q, Y, gamma, mal_alpha, beta)
    else:
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
        f"MAL Loss 3D Surface\n(y={y}, γ={gamma}, α={mal_alpha} beta={beta})",
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
        f"MAL Loss Contour\n(y={y}, γ={gamma}, α={mal_alpha} beta={beta})",
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
        f"mal_loss_y{y}_gamma{gamma}_alpha{mal_alpha}_beta{beta}_new{use_new_loss}.png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()

    print(
        f"✓ 已保存可视化：mal_loss_y{y}_gamma{gamma}_alpha{mal_alpha}_beta{beta}_new{use_new_loss}.png"
    )


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


if __name__ == "__main__":
    print("=" * 60)
    print("MAL Loss Visualization Tool")
    print("=" * 60)

    # === 示例 1: 单个参数设置可视化 ===
    print("\n1. 可视化单个参数设置...")
    visual_mal_loss(
        p=None, q=None, y=1, gamma=0.5, mal_alpha=None, beta=0.5, use_new_loss=False
    )
    visual_mal_loss(
        p=None, q=None, y=1, gamma=0.5, mal_alpha=None, beta=0.5, use_new_loss=True
    )
    visual_mal_loss(
        p=None, q=None, y=1, gamma=0.5, mal_alpha=None, beta=1, use_new_loss=False
    )
    visual_mal_loss(
        p=None, q=None, y=1, gamma=0.5, mal_alpha=None, beta=1, use_new_loss=True
    )
    visual_mal_loss(
        p=None, q=None, y=1, gamma=0.5, mal_alpha=None, beta=1.5, use_new_loss=False
    )
    visual_mal_loss(
        p=None, q=None, y=1, gamma=0.5, mal_alpha=None, beta=1.5, use_new_loss=True
    )

    # # === 示例 2: 参数对比 ===
    # print("\n2. 对比不同参数设置...")
    # visual_mal_loss_comparison()

    # # === 示例 3: 权重分布 ===
    # print("\n3. 可视化权重分布...")
    # visual_weight_distribution()

    print("\n" + "=" * 60)
    print("所有可视化已完成！请检查生成的 PNG 文件。")
    print("=" * 60)
