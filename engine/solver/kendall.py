"""Kendall Uncertainty Weighting for Multi-Task Loss Balancing.

基于 Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses
for Scene Geometry and Semantics", CVPR 2018.

总 loss = Σ_i [ 1/(2σ_i²) * L_i + log σ_i ]
令 s_i = log σ_i（可学习），则：
  weight_i = 0.5 * exp(-2 * s_i)
  regularizer_i = s_i

不需要共享参数，不需要二阶梯度，天然兼容 grid_sample 等操作为了。
每个 loss_name 有一个独立的学习 s_i，自动根据 loss 量纲调整权重。
"""

import re
import torch
import torch.nn as nn


class KendallWeighting(nn.Module):
    """可学习 σ² 权重（Kendall uncertainty weighting）。

    用法：
        kw = KendallWeighting(['loss_mal','loss_bbox','loss_kld','loss_fgl'])
        optimizer.add_param_group({'params': [kw.log_sigma], 'lr': 0.001,
                                   'weight_decay': 0.0})

        for batch in dataloader:
            loss_dict = criterion(...)
            loss = kw.weighted_loss(loss_dict)    # weight*L + log σ 正则
            loss.backward()
            optimizer.step()

    配置（YAML）：
        KendallWeighting:
          enabled: true
          sigma_lr: 0.001
    """

    def __init__(self, loss_names: list, init_log_sigma: float = 0.0, prior: list = None):
        super().__init__()
        self.loss_names = loss_names
        self.T = len(loss_names)
        self.log_sigma = nn.Parameter(torch.full((self.T,), init_log_sigma))

        if prior is None:
            prior = [1.0] * self.T
        self.register_buffer("prior", torch.tensor(prior, dtype=torch.float32))

        # 用于匹配 aux/dn/enc/pre 后缀的 regex
        pat_parts = "|".join(["aux", "dn", "enc", "pre"])
        self._suffix_re = re.compile(
            rf"^(({pat_parts})_\d+|({pat_parts}))$"
        )

    def get_weights(self):
        """返回各 loss 当前权重 dict（用于日志）。包含 prior 乘子。"""
        with torch.no_grad():
            w = 0.5 * torch.exp(-2.0 * self.log_sigma) * self.prior
        return {n: w[i].item() for i, n in enumerate(self.loss_names)}

    def _aggregate_loss(self, loss_dict: dict, name: str) -> torch.Tensor:
        """聚合同名 loss 的所有贡献（main + aux/dn/enc/pre 后缀）。"""
        total = None
        prefix = name + "_"
        for k, v in loss_dict.items():
            if k == name or (k.startswith(prefix) and self._suffix_re.match(k[len(prefix):])):
                total = v if total is None else total + v
        if total is None:
            return loss_dict.get(name, torch.zeros((), device=next(iter(loss_dict.values())).device))
        return total

    def weighted_loss(self, loss_dict: dict) -> torch.Tensor:
        """计算加权总 loss = Σ p_i·w_i·(聚合 L_i) + Σ p_i·s_i。

        w_i = 0.5 * exp(-2 * s_i)，s_i = log σ_i 是正则项，
        p_i = weight_dict_i / mean(weight_dict) 是固定先验乘子。
        aux/dn/enc/pre 子项自动与对应主 loss_name 共享同一 w_i 和 p_i。
        """
        loss = None
        for i, name in enumerate(self.loss_names):
            agg = self._aggregate_loss(loss_dict, name)
            w = 0.5 * torch.exp(-2.0 * self.log_sigma[i]) * self.prior[i]
            contrib = w * agg
            loss = contrib if loss is None else loss + contrib

        # 加上不在 management 中的 loss（等权 1.0）
        for k, v in loss_dict.items():
            is_managed = False
            for n in self.loss_names:
                if k == n or (k.startswith(n + "_") and self._suffix_re.match(k[len(n + "_"):])):
                    is_managed = True
                    break
            if not is_managed:
                loss = v if loss is None else loss + v

        # Kendall 正则项：Σ p_i·log σ_i = Σ p_i·s_i
        loss = loss + (self.log_sigma * self.prior).sum()
        return loss