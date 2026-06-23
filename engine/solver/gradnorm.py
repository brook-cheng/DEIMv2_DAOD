"""GradNorm: Gradient Normalization for Adaptive Loss Balancing.

基于 dav-ell/gradnorm 的简洁 API，结合 lucidrains/gradnorm-pytorch 的工程特性。
在 loss.backward() 之前调用 step() 计算自适应权重，然后用 w_i.detach() 加权 loss
再 backward()，使模型梯度受自适应权重影响。

参考：Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss
Balancing in Deep Multitask Networks", ICML 2018.
"""

import re
import torch
import torch.nn as nn
from ..core import register


@register()
class GradNorm:
    """自适应多任务 loss 权重。

    用法：
        gradnorm = GradNorm(model, ['loss_mal','loss_bbox','loss_kld','loss_fgl'],
                            shared_param_pattern='decoder.*layers\\.0\\.')

        # 在训练循环中（loss.backward() 之前）：
        current_weights = gradnorm.step(loss_dict)   # 计算 w_i（保留 forward graph）
        loss = sum(loss_dict[n] * gradnorm.w_i[i].detach() for i, n in enumerate(...))
        loss.backward()                              # 模型梯度受 w_i 影响
        optimizer.step()

    配置（YAML）：
        GradNorm:
          enabled: true
          alpha: 0.12
          lr: 5e-5
          shared_param_pattern: 'decoder.*layers\\.0\\.'
    """

    def __init__(
        self,
        model: nn.Module,
        loss_names: list,
        shared_param_pattern: str = r"decoder.*layers\.0\.",
        alpha: float = 0.12,
        lr: float = 5e-5,
    ):
        self.loss_names = loss_names
        self.T = len(loss_names)
        self.alpha = alpha
        self.lr = lr

        # 权重初始为 1.0（从 config weight_dict 独立出来）
        self.w_i = nn.Parameter(torch.ones(self.T), requires_grad=True)
        self._device = None
        self.L0 = None

        # 找到共享参数（分类和回归都经过的瓶颈层）
        self.shared_params = self._find_shared_params(model, shared_param_pattern)
        if not self.shared_params:
            print(
                f"[GradNorm] WARNING: no params matched pattern '{shared_param_pattern}' — falling back to decoder first layer"
            )
            self.shared_params = self._find_shared_params(
                model, r"decoder\.layers\.0\."
            )

        print(f"[GradNorm] {self.T} losses: {loss_names}")
        print(
            f"[GradNorm] shared params: {len(self.shared_params)} tensors, "
            f"total {sum(p.numel() for p in self.shared_params):,} params"
        )
        print(f"[GradNorm] alpha={alpha}, lr={lr}")

    def _find_shared_params(self, model, pattern):
        """正则匹配模型参数名。"""
        params = []
        seen = set()
        for name, p in model.named_parameters():
            if re.search(pattern, name) and p.requires_grad:
                # 只取 weight（第一个匹配的），避免 bias 等重复
                base = re.sub(r"(\.weight|\.bias|\.scale)$", "", name)
                if base not in seen and "weight" in name:
                    seen.add(base)
                    params.append(p)
        return params

    def step(self, loss_dict):
        """计算并更新自适应权重。返回当前权重 dict（用于日志）。

        调用时 forward graph 必须仍在（retain_graph 由调用方保证 backward 在 step 之后）。
        使用解析梯度替代 create_graph=True，避免 grid_sampler_2d 二阶导未实现的问题。
        """
        losses = torch.stack([loss_dict[n].detach() for n in self.loss_names])

        # 首次调用时将 w_i 移到正确 device
        if self._device is None:
            self._device = losses.device
            self.w_i = self.w_i.to(self._device)

        if self.L0 is None:
            self.L0 = losses.clone()

        # 计算各 RAW loss 在共享参数上的梯度 L2 范数（不用 create_graph）
        # G_i = w_i * ||∇L_i|| → dG_i/dw_i = ||∇L_i|| （解析可导，无需二阶 autograd）
        G = []
        grad_norms = []
        for i in range(self.T):
            grads = torch.autograd.grad(
                loss_dict[self.loss_names[i]],
                self.shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            g_norm = torch.zeros((), device=losses.device)
            for g in grads:
                if g is not None:
                    g_norm = g_norm + g.pow(2).sum()
            g_norm = g_norm.sqrt()
            grad_norms.append(g_norm)
            G.append(self.w_i[i] * g_norm)

        G = torch.stack(G)
        G_avg = G.mean()

        # 计算相对训练速率 r_i = L_i(t) / L_i(0)，归一化
        r = losses / (self.L0 + 1e-8)
        r_norm = r / (r.mean() + 1e-8)

        # 目标梯度范数：G_avg × r_i^α
        target = (G_avg * r_norm.pow(self.alpha)).detach()

        # 解析梯度：dL_grad/dw_i = sign(G_i - target_i) * ||∇L_i||
        w_grad = torch.stack([
            (G[i] - target[i]).sign() * grad_norms[i]
            for i in range(self.T)
        ]).detach()

        with torch.no_grad():
            self.w_i -= self.lr * w_grad
            # 重归一化：保持均值 = T（即均值 ≈ 1）
            self.w_i.data = self.w_i / (self.w_i.sum() + 1e-8) * self.T
            self.w_i.clamp_(min=1e-8)

        return {n: self.w_i[i].item() for i, n in enumerate(self.loss_names)}
