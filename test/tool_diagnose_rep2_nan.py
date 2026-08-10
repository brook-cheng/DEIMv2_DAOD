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


def atan2_zero_probe(device: str = "cpu") -> dict:
    """验证 torch.atan2(0,0) 前向有限、反向梯度非有限（复现 rep2 候选奇点）。"""
    x = torch.zeros(1, device=device, requires_grad=True)
    y = torch.zeros(1, device=device, requires_grad=True)
    z = torch.atan2(y, x)
    with torch.no_grad():
        forward_finite = bool(torch.isfinite(z).all().item())
        forward_value = float(z.item())
    z.backward()
    grads = [g for g in (x.grad, y.grad) if g is not None]
    backward_finite = bool(grads) and all(bool(torch.isfinite(g).all().item()) for g in grads)
    return {
        "device": device,
        "forward_value": forward_value,
        "forward_finite": forward_finite,
        "backward_grads_finite": backward_finite,
        "grad_x": float(x.grad.item()) if x.grad is not None else None,
        "grad_y": float(y.grad.item()) if y.grad is not None else None,
    }


def compute_edge_stats(
    external_rect: torch.Tensor,
    vertex_offsets: torch.Tensor,
    eps: float = 1e-9,
) -> dict:
    """复刻 external_xyxy_rect_to_oriented_box 的边构造与 atan2 输入统计（只读、detach）。

    依据 obb_geometry.py 第 209-238 行：
        v0=(x2-ep, y1), v1=(x2, y2-et), v2=(x1+ep, y2)
        edge_ab=v1-v0, edge_bc=v2-v1
        w_dx/w_dy = 较长边（len_ab>=len_bc 时取 edge_ab）的分量
    """
    rect = external_rect.detach().float()
    offs = vertex_offsets.detach().float()

    x1 = rect[..., 0]
    y1 = rect[..., 1]
    x2 = rect[..., 2]
    y2 = rect[..., 3]
    ep = offs[..., 0]
    et = offs[..., 1]

    v0 = torch.stack([x2 - ep, y1], dim=-1)
    v1 = torch.stack([x2, y2 - et], dim=-1)
    v2 = torch.stack([x1 + ep, y2], dim=-1)

    edge_ab = v1 - v0
    edge_bc = v2 - v1

    len_ab = torch.sqrt(edge_ab[..., 0] ** 2 + edge_ab[..., 1] ** 2 + eps)
    len_bc = torch.sqrt(edge_bc[..., 0] ** 2 + edge_bc[..., 1] ** 2 + eps)

    w_is_ab = len_ab >= len_bc
    w_len = torch.where(w_is_ab, len_ab, len_bc)
    h_len = torch.where(w_is_ab, len_bc, len_ab)

    w_dx = torch.where(w_is_ab, edge_ab[..., 0], edge_bc[..., 0])
    w_dy = torch.where(w_is_ab, edge_ab[..., 1], edge_bc[..., 1])

    flat_ab = edge_ab.reshape(-1, 2)
    flat_bc = edge_bc.reshape(-1, 2)

    return {
        "n": int(flat_ab.shape[0]),
        "w_min": float(w_len.min().item()),
        "h_min": float(h_len.min().item()),
        "edge_ab_zero": int((flat_ab == 0).all(dim=-1).sum().item()),
        "edge_bc_zero": int((flat_bc == 0).all(dim=-1).sum().item()),
        "near_zero_edges": int(
            ((flat_ab.norm(dim=-1) < 1e-6) | (flat_bc.norm(dim=-1) < 1e-6)).sum().item()
        ),
        "atan2_zero_inputs": int(((w_dx == 0) & (w_dy == 0)).sum().item()),
        "w_dx_absmin": float(w_dx.abs().min().item()),
        "w_dy_absmin": float(w_dy.abs().min().item()),
        "eps_min": float(ep.min().item()),
        "eta_min": float(et.min().item()),
        "eps_max": float(ep.max().item()),
        "eta_max": float(et.max().item()),
    }


class GeometryProbe:
    """包装 rep2 两个 decode 调用点，只读累计退化边/atan2 输入统计。

    不改动生产张量：包装器调用原函数得到原输出，统计全部 detach 后执行。
    包装点是消费模块命名空间内已绑定的名字（dfine_utils / deim_decoder
    均 `from .obb_geometry import ...` 按名导入），因此必须就地替换
    各消费模块的模块级属性，不能只替换 obb_geometry 模块本身。
    """

    def __init__(self):
        self._originals: dict[str, tuple] = {}
        self._snapshots: list[dict] = []
        self._calls = 0

    def install(self) -> None:
        import engine.deim.dfine_utils as dfine_utils
        import engine.deim.deim_decoder as deim_decoder

        targets = {
            "dfine_utils.external_xyxy_rect_to_oriented_box": (
                dfine_utils,
                "external_xyxy_rect_to_oriented_box",
            ),
            "deim_decoder.external_xywh_rect_to_oriented_box": (
                deim_decoder,
                "external_xywh_rect_to_oriented_box",
            ),
        }
        for key, (mod, name) in targets.items():
            orig = getattr(mod, name)
            self._originals[key] = (mod, name, orig)
            setattr(mod, name, self._make_wrapper(orig))

    def uninstall(self) -> None:
        for mod, name, orig in self._originals.values():
            setattr(mod, name, orig)
        self._originals.clear()

    def reset(self) -> None:
        self._snapshots.clear()
        self._calls = 0

    def _make_wrapper(self, orig):
        def wrapper(*args, **kwargs):
            out = orig(*args, **kwargs)
            self._calls += 1
            try:
                if len(args) >= 2 and isinstance(args[0], torch.Tensor):
                    self._snapshots.append(compute_edge_stats(args[0], args[1]))
            except Exception:
                pass  # 探针统计失败不干扰训练
            return out

        return wrapper

    def snapshot(self) -> dict:
        if not self._snapshots:
            return {"calls": self._calls}
        agg: dict = {"calls": self._calls, "n": 0}
        for k in ("edge_ab_zero", "edge_bc_zero", "near_zero_edges", "atan2_zero_inputs", "n"):
            agg[k] = sum(s[k] for s in self._snapshots)
        for k in ("w_min", "h_min", "w_dx_absmin", "w_dy_absmin", "eps_min", "eta_min"):
            agg[k] = min(s[k] for s in self._snapshots)
        for k in ("eps_max", "eta_max"):
            agg[k] = max(s[k] for s in self._snapshots)
        return agg
