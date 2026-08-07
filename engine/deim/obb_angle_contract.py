"""统一管理 OBB 角度变换。

角度变量速查表（下列所有接口注释均遵循这些量纲和数值域）：

  =======================  ==============================================
  变量名                   量纲 / 数值域 / 约束条件
  =======================  ==============================================
  ``theta_phys_rad``       物理角，单位为弧度，范围 ``[0, pi)``。公开使用：dataset、transforms、geometry、criterion、matcher、postprocessor、eval、export。
  ``theta_norm``           归一化角，无量纲，范围 ``[0, 1)``。decoder内部使用。
  ``theta_logit``          无界 logit，即 ``logit(theta_norm)``。decoder 各 head 的原始输出与 `ref_points_unact` 角度通道。
  ``theta_loss_rad``       loss计算中的角度，范围``[-pi/4, 3pi/4)``。仅 criterion 内部。
  =======================  ==============================================
"""

import torch
from torch import Tensor


def canonicalize_phys_rad(theta_rad: Tensor) -> Tensor:
    """将任意弧度角规范化到标准物理角范围。

    Args:
        theta_rad: 原始角度，单位为**弧度**，可取任意实数，例如负数或
            ``>= pi`` 的值。

    Returns:
        ``theta_phys_rad``：标准物理角，单位为**弧度**，范围为半开区间
        ``[0, pi)``；其值等于 ``theta_rad`` 对 ``pi`` 取模后的结果。
    """
    return torch.remainder(theta_rad, torch.pi)


def physical_rad_to_norm(theta_phys_rad: Tensor) -> Tensor:
    """将物理弧度角转换为网络内部使用的归一化角。

    Args:
        theta_phys_rad: 输入物理角，单位为**弧度**，范围 ``[0, pi)``。

    Returns:
        ``theta_norm``：输出归一化角，**无量纲**，范围 ``[0, 1)``。
    """
    return theta_phys_rad / torch.pi


def norm_to_physical_rad(theta_norm: Tensor) -> Tensor:
    """将网络内部的归一化角还原为物理弧度角。

    这是 :func:`physical_rad_to_norm` 的逆转换。

    Args:
        theta_norm: 输入归一化角，**无量纲**，范围 ``[0, 1)``

    Returns:
        ``theta_phys_rad``：输出标准物理角，单位为**弧度**，范围
        ``[0, pi)``。
    """
    return theta_norm * torch.pi


def physical_rad_to_logit(theta_phys_rad: Tensor, eps: float = 1e-4) -> Tensor:
    """将物理弧度角编码为网络预测头使用的无界 logit。

    Args:
        theta_phys_rad: 输入物理角，单位为**弧度**，范围 ``[0, pi)``。
        eps: 执行 logit 变换前，对归一化角使用的截断边距；用于保证输出
            为有限值。该参数无量纲。

    Returns:
        ``theta_logit``：输出的**无界 logit**，即对截断后的无量纲的 theta_norm 执行 logit 变换所得的值
    """
    theta_norm = physical_rad_to_norm(theta_phys_rad).clamp(min=eps, max=1.0 - eps)
    return torch.logit(theta_norm)


def logit_to_physical_rad(theta_logit: Tensor) -> Tensor:
    """将网络输出的无界 logit 解码为物理弧度角。

    这是 :func:`physical_rad_to_logit` 的逆向解码路径。

    Args:
        theta_logit: 输入的**无界 logit**，可取任意实数，通常来自网络
            预测头的原始输出。

    Returns:
        ``theta_phys_rad``：输出标准物理角，单位为**弧度**，范围
        ``[0, pi)``。
    """
    return norm_to_physical_rad(torch.sigmoid(theta_logit))


def physical_rad_to_loss_rad(theta_rad: Tensor) -> Tensor:
    """保持几何表征不变，将物理弧度角转换为loss计算中的角度范围。

    Args:
        theta_rad: 输入物理弧度角，范围``[0, pi)``。

    Returns:
        ``theta_loss_rad``：输出物理弧度角，范围``[-pi/4, 3pi/4)``。
    """
    return torch.remainder(theta_rad + torch.pi / 4, torch.pi) - torch.pi / 4


def physical_rad_to_shifted_norm(theta_phys_rad: Tensor) -> Tensor:
    """将物理弧度角编码为 decoder 私有 shifted 归一化角。

    Args:
        theta_phys_rad: 输入物理角，单位为**弧度**，范围 ``[0, pi)``。

    Returns:
        ``theta_shift``：输出 shifted 归一化角，**无量纲**，范围 ``[0, 1)``，
        即 ``remainder(theta_phys_rad / pi + 0.25, 1)``。0→0.25, π/2→0.75,
        3π/4→0（seam 移至 135°）。
    """
    return torch.remainder(theta_phys_rad / torch.pi + 0.25, 1.0)


def shifted_norm_to_physical_rad(theta_shift: Tensor) -> Tensor:
    """将 decoder 私有 shifted 归一化角还原为物理弧度角。

    这是 :func:`physical_rad_to_shifted_norm` 的逆转换。

    Args:
        theta_shift: 输入 shifted 归一化角，**无量纲**，范围 ``[0, 1)``。

    Returns:
        ``theta_phys_rad``：输出标准物理角，单位为**弧度**，范围
        ``[0, pi)``。
    """
    return torch.remainder(theta_shift - 0.25, 1.0) * torch.pi
