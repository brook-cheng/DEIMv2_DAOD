#!/usr/bin/env python3
"""DEIMv2-OBB rep2 NaN 远程诊断 runner。

设计依据: docs/superpowers/specs/2026-08-10-rep2-nan-diagnostic-runner-design.md
"""

import os
import shutil
import sys
from pathlib import Path

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


def checkpoint_inspect(state: dict) -> dict:
    """分类 checkpoint 顶层结构，返回 full / weights_only / invalid。"""
    top_keys = sorted(state.keys())
    has_model = isinstance(state.get("model"), dict)
    has_ema = isinstance(state.get("ema"), dict)
    has_optimizer = isinstance(state.get("optimizer"), dict)
    last_epoch = state.get("last_epoch")

    bare_params = bool(
        not has_model
        and not has_ema
        and top_keys
        and all(isinstance(v, torch.Tensor) for v in state.values())
    )

    if isinstance(last_epoch, int) and (has_model or has_ema) and has_optimizer:
        kind = "full"
    elif (has_model or has_ema or bare_params) and not has_optimizer:
        kind = "weights_only"
    else:
        kind = "invalid"

    return {
        "kind": kind,
        "top_keys": top_keys,
        "last_epoch": last_epoch,
        "has_model": has_model,
        "has_ema": has_ema,
        "has_optimizer": has_optimizer,
        "notes": [],
    }


def ensure_output_dir(output_dir: Path | str, overwrite: bool) -> Path:
    """校验/创建输出目录。非空目录默认拒绝；overwrite=True 清空重建。"""
    d = Path(output_dir)
    if d.exists() and any(d.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output-dir 已存在且非空: {d}（使用 --overwrite 显式允许覆盖）"
            )
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d
