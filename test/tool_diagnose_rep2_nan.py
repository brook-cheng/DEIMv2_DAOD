#!/usr/bin/env python3
"""DEIMv2-OBB rep2 NaN 远程诊断 runner。

设计依据: docs/superpowers/specs/2026-08-10-rep2-nan-diagnostic-runner-design.md
"""

import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main(argv=None) -> int:
    raise NotImplementedError("main 在 Task 7 实现")


if __name__ == "__main__":
    sys.exit(main())


def tensor_stats(t: torch.Tensor) -> dict[str, int | float | None]:
    """有限性统计：区分 finite/NaN/+Inf/-Inf，极值只对有限元素统计。"""
    flat = t.detach().float().flatten()
    finite_mask = torch.isfinite(flat)
    finite_vals = flat[finite_mask]
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "finite": int(finite_mask.sum().item()),
        "nan": int(torch.isnan(flat).sum().item()),
        "pos_inf": int(torch.isposinf(flat).sum().item()),
        "neg_inf": int(torch.isneginf(flat).sum().item()),
        "min": float(finite_vals.min().item()) if finite_vals.numel() else None,
        "max": float(finite_vals.max().item()) if finite_vals.numel() else None,
        "absmax": float(finite_vals.abs().max().item()) if finite_vals.numel() else None,
    }


def scan_gradients(
    model: torch.nn.Module,
) -> tuple[float, list[dict]]:
    """扫描全部参数梯度，报告所有非有限参数（不在首个参数停止）。

    返回 (aggregate_norm, anomalies)；无梯度参数跳过。
    """
    aggregate_sq = 0.0
    anomalies: list[dict] = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad
        if grad.is_sparse:
            grad = grad.coalesce()
            values = grad.values()
        else:
            values = grad

        if not torch.isfinite(values).all():
            flat = values.detach().float().flatten()
            finite_vals = flat[torch.isfinite(flat)]
            anomalies.append(
                {
                    "name": name,
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "nan": int(torch.isnan(flat).sum().item()),
                    "pos_inf": int(torch.isposinf(flat).sum().item()),
                    "neg_inf": int(torch.isneginf(flat).sum().item()),
                    "finite_min": (
                        float(finite_vals.min().item()) if finite_vals.numel() else None
                    ),
                    "finite_max": (
                        float(finite_vals.max().item()) if finite_vals.numel() else None
                    ),
                    "finite_norm": float(finite_vals.norm(2).item()),
                }
            )
        else:
            aggregate_sq += values.float().pow(2).sum().item()

    return float(aggregate_sq**0.5), anomalies
