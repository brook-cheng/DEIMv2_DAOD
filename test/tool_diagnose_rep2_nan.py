#!/usr/bin/env python3
"""DEIMv2-OBB rep2 NaN 远程诊断 runner。

设计依据: docs/superpowers/specs/2026-08-10-rep2-nan-diagnostic-runner-design.md
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback as _tb
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils._pytree import tree_map

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


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


def write_run_manifest(path: Path | str, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def append_event(path: Path | str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_progress(path: Path | str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _to_cpu(d):
    """递归把 dict/list 中的 tensor detach 并移到 CPU；其余原样。"""
    if isinstance(d, torch.Tensor):
        return d.detach().cpu()
    if isinstance(d, dict):
        return {k: _to_cpu(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_to_cpu(v) for v in d]
    return d


def save_failure(
    output_dir: Path | str,
    *,
    traceback_text: str,
    failure_summary: dict,
    trigger_batch: dict,
    outputs: dict,
    losses: dict,
    geometry_snapshot: dict,
    gradients_summary: dict,
    model_state: dict,
    optimizer_state: dict,
) -> dict:
    """持久化失败现场到 output-dir/failure/。tensor 一律 detach+CPU。

    任一产物保存失败：追加到 failure_summary["secondary_errors"]，不覆盖 traceback。
    返回 {键: 绝对路径}。
    """
    fail_dir = Path(output_dir) / "failure"
    fail_dir.mkdir(parents=True, exist_ok=True)
    secondary_errors = failure_summary.setdefault("secondary_errors", [])

    artifacts = {
        "traceback": fail_dir / "traceback.txt",
        "failure_summary": fail_dir / "failure_summary.json",
        "trigger_batch": fail_dir / "trigger_batch.pt",
        "outputs": fail_dir / "outputs.pt",
        "losses": fail_dir / "losses.pt",
        "geometry_snapshot": fail_dir / "geometry_snapshot.pt",
        "gradients_summary": fail_dir / "gradients_summary.json",
        "model_state": fail_dir / "model_state.pt",
        "optimizer_state": fail_dir / "optimizer_state.pt",
    }

    def _safe(kind: str, path: Path, payload, *, text: bool = False):
        try:
            if text:
                if isinstance(payload, dict):
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                else:
                    path.write_text(payload)
            else:
                torch.save(_to_cpu(payload), path)
            return str(path)
        except Exception as e:  # noqa: BLE001 - 次级保存错误需记录不抛出
            secondary_errors.append({"artifact": kind, "error": f"{type(e).__name__}: {e}"})
            return None

    _safe("traceback", artifacts["traceback"], traceback_text, text=True)
    _safe("trigger_batch", artifacts["trigger_batch"], trigger_batch)
    _safe("outputs", artifacts["outputs"], outputs)
    _safe("losses", artifacts["losses"], losses)
    _safe("geometry_snapshot", artifacts["geometry_snapshot"], geometry_snapshot)
    _safe("gradients_summary", artifacts["gradients_summary"], gradients_summary, text=True)
    _safe("model_state", artifacts["model_state"], model_state)
    _safe("optimizer_state", artifacts["optimizer_state"], optimizer_state)
    # failure_summary 最后写，确保早先产物保存失败也记入 secondary_errors
    _safe("failure_summary", artifacts["failure_summary"], failure_summary, text=True)

    return {k: str(v) for k, v in artifacts.items()}


def restore_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    state: dict,
    start_epoch_override: int | None = None,
) -> dict:
    """把 checkpoint state 恢复到 model/optimizer，返回恢复报告。

    fidelity: full（model+optimizer 全加载）/ partial（某状态失败）/
              weights_only（checkpoint 无 optimizer）。
    """
    last_epoch = state.get("last_epoch", -1)
    start_epoch = (
        start_epoch_override
        if start_epoch_override is not None
        else (last_epoch + 1 if isinstance(last_epoch, int) else 0)
    )
    loaded: dict[str, str] = {}
    missing: list[str] = []
    unexpected: list[str] = []
    notes: list[str] = []
    fidelity = "full"

    model_sd = None
    if isinstance(state.get("model"), dict):
        model_sd = state["model"]
    elif isinstance(state.get("ema"), dict) and isinstance(state["ema"].get("module"), dict):
        model_sd = state["ema"]["module"]
        notes.append("checkpoint 无 model 键，已从 ema.module 恢复训练权重")
    else:
        raise ValueError("checkpoint 缺少可恢复的 model/ema.module 权重")

    try:
        res = model.load_state_dict(model_sd, strict=False)
        missing = list(res.missing_keys)
        unexpected = list(res.unexpected_keys)
        loaded["model"] = "ok"
    except Exception as e:  # noqa: BLE001
        loaded["model"] = f"error: {type(e).__name__}: {e}"
        notes.append(f"model 加载失败: {loaded['model']}")
        fidelity = "partial"

    if "optimizer" in state and isinstance(state["optimizer"], dict):
        if optimizer is None:
            notes.append("checkpoint 含 optimizer 但运行器未构建 optimizer")
            fidelity = "partial"
        else:
            try:
                optimizer.load_state_dict(state["optimizer"])
                loaded["optimizer"] = "ok"
            except Exception as e:  # noqa: BLE001
                loaded["optimizer"] = f"error: {type(e).__name__}: {e}"
                notes.append(f"optimizer 加载失败: {loaded['optimizer']}")
                fidelity = "partial"
    else:
        loaded["optimizer"] = "skipped (not in checkpoint)"
        if fidelity == "full":
            fidelity = "weights_only"

    if missing:
        notes.append(f"model missing_keys={missing}")
    if unexpected:
        notes.append(f"model unexpected_keys={unexpected}")

    return {
        "start_epoch": start_epoch,
        "fidelity": fidelity,
        "loaded": loaded,
        "missing": missing,
        "unexpected": unexpected,
        "notes": notes,
    }


@dataclass
class Pipeline:
    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer | None
    lr_scheduler: object | None
    train_dataloader: object
    device: torch.device
    probe: object | None = None


def _metas(epoch: int, step: int, global_step: int, epoch_step: int) -> dict:
    return dict(epoch=epoch, step=step, global_step=global_step, epoch_step=epoch_step)


def _to_device(samples, targets, device):
    samples = samples.to(device)
    targets = [
        {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
        for t in targets
    ]
    return samples, targets


def run_diagnostic(pipeline: Pipeline, args) -> int:
    """最小单 GPU 诊断训练循环。返回退出码 0/2/4。"""
    from engine.solver.training_diagnostics import (
        raise_for_nonfinite_losses,
        raise_for_nonfinite_total,
    )

    model, criterion, optimizer = pipeline.model, pipeline.criterion, pipeline.optimizer
    loader = pipeline.train_dataloader
    device = pipeline.device
    probe = pipeline.probe
    output_dir = Path(args.output_dir)

    model.train()
    criterion.train()
    if getattr(args, "detect_anomaly", True):
        torch.autograd.set_detect_anomaly(True)

    # 循环体/失败路径共享的占位（forward_output 失败点早于 criterion 赋值）
    samples = targets = outputs = None
    loss_dict: dict = {}
    grad_norm = None
    anomalies: list[dict] = []

    epoch_step = len(loader) if hasattr(loader, "__len__") else 0

    def _write_event(record: dict) -> None:
        append_event(output_dir / "events.jsonl", record)

    def _fail(kind: str, exit_code: int, extra: dict) -> int:
        summary = {
            "exit_code": exit_code,
            "kind": kind,
            **extra,
        }
        save_failure(
            output_dir,
            traceback_text=_tb.format_exc() if _tb.sys.exc_info()[0] else f"kind={kind}",
            failure_summary=summary,
            trigger_batch={"samples": samples, "targets": targets},
            outputs=outputs,
            losses=loss_dict,
            geometry_snapshot=probe.snapshot() if probe else {},
            gradients_summary={"aggregate_norm": grad_norm, "anomalies": anomalies},
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict() if optimizer else {},
        )
        return exit_code

    try:
        for epoch in range(args.start_epoch, args.start_epoch + args.max_epochs):
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch)
            cur_iters = epoch * epoch_step

            for i, (samples, targets) in enumerate(loader):
                if args.max_steps_per_epoch is not None and i >= args.max_steps_per_epoch:
                    break

                global_step = epoch * epoch_step + i
                t0 = time.time()
                samples, targets = _to_device(samples, targets, device)

                if getattr(args, "use_amp", False):
                    with torch.autocast(
                        device_type=str(device).split(":")[0],
                        dtype=torch.bfloat16,
                        cache_enabled=True,
                    ):
                        outputs = model(samples, targets=targets)
                    outputs = tree_map(
                        lambda t: t.float()
                        if isinstance(t, torch.Tensor) and t.is_floating_point()
                        else t,
                        outputs,
                    )
                else:
                    outputs = model(samples, targets=targets)

                bad_keys = [
                    k
                    for k in ("pred_logits", "pred_boxes", "pred_corners", "ref_points")
                    if k in outputs
                    and isinstance(outputs[k], torch.Tensor)
                    and not torch.isfinite(outputs[k]).all()
                ]
                if bad_keys:
                    return _fail(
                        "forward_output", 2,
                        {"epoch": epoch, "step": i, "global_step": global_step, "bad_keys": bad_keys},
                    )

                loss_dict = criterion(outputs, targets, **_metas(epoch, i, global_step, epoch_step))

                try:
                    raise_for_nonfinite_losses(
                        loss_dict, epoch=epoch, step=i, global_step=global_step
                    )
                    loss = sum(loss_dict.values())
                    raise_for_nonfinite_total(
                        loss, epoch=epoch, step=i, global_step=global_step
                    )
                except FloatingPointError:
                    return _fail(
                        "loss", 2,
                        {
                            "epoch": epoch, "step": i, "global_step": global_step,
                            "loss_dict": {
                                k: (float(v.detach().cpu()) if isinstance(v, torch.Tensor) else None)
                                for k, v in loss_dict.items()
                            },
                        },
                    )

                grad_norm = None
                anomalies = []
                try:
                    loss.backward()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        return _fail(
                            "cuda_oom", 4,
                            {"epoch": epoch, "step": i, "global_step": global_step, "error": str(e)},
                        )
                    return _fail(
                        "backward_anomaly", 2,
                        {"epoch": epoch, "step": i, "global_step": global_step, "error": str(e)},
                    )

                grad_norm, anomalies = scan_gradients(model)
                if anomalies:
                    return _fail(
                        "gradient", 2,
                        {
                            "epoch": epoch, "step": i, "global_step": global_step,
                            "aggregate_norm": grad_norm,
                            "anomaly_params": [a["name"] for a in anomalies],
                        },
                    )

                # 梯度全部有限 → 才允许 step
                if args.clip_max_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_max_norm
                    )
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad()

                if pipeline.lr_scheduler is not None:
                    optimizer = pipeline.lr_scheduler.step(cur_iters + i, optimizer)

                cur_lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
                _write_event(
                    {
                        "epoch": epoch,
                        "step": i,
                        "global_step": global_step,
                        "lr": float(cur_lr),
                        "loss_total": float(loss.detach().cpu()),
                        "loss_dict": {
                            k: float(v.detach().cpu())
                            for k, v in loss_dict.items()
                            if isinstance(v, torch.Tensor) and v.dim() == 0
                        },
                        "grad_norm": float(grad_norm),
                        "step_duration_s": round(time.time() - t0, 4),
                        "vram_mb": int(
                            torch.cuda.max_memory_allocated(device) // (1024 * 1024)
                        )
                        if str(device).startswith("cuda")
                        else 0,
                    }
                )

                if global_step % args.save_every_steps == 0:
                    write_progress(
                        output_dir / "progress.json",
                        {"epoch": epoch, "global_step": global_step},
                    )

    except Exception as e:  # noqa: BLE001 - 兜底运行时异常（exit 4）
        try:
            save_failure(
                output_dir,
                traceback_text=_tb.format_exc(),
                failure_summary={
                    "exit_code": 4,
                    "kind": "runtime",
                    "error": f"{type(e).__name__}: {e}",
                },
                trigger_batch={},
                outputs={},
                losses={},
                geometry_snapshot=probe.snapshot() if probe else {},
                gradients_summary={},
                model_state={},
                optimizer_state={},
            )
        except Exception:
            pass
        return 4

    write_progress(
        output_dir / "progress.json",
        {"done": True, "end_epoch": args.start_epoch + args.max_epochs},
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEIMv2-OBB rep2 NaN 远程诊断 runner"
    )
    parser.add_argument("--config", type=str, required=True, help="训练 YAML 配置")
    parser.add_argument("--checkpoint", type=str, required=True, help="完整或 weights-only checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="诊断输出目录（非空默认拒绝覆盖）")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--start-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--detect-anomaly", dest="detect_anomaly", action="store_true", default=True)
    parser.add_argument("--no-detect-anomaly", dest="detect_anomaly", action="store_false")
    parser.add_argument("--save-every-steps", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args(argv)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_pipeline(
    cfg_path: str,
    checkpoint_path: str,
    device: str,
    start_epoch: int | None,
    seed: int | None,
) -> tuple[Pipeline, dict]:
    """用 YAMLConfig 构建诊断 pipeline，恢复 checkpoint，返回 (pipeline, manifest_meta)。

    不调用 solver.fit()；不构建 ema/evaluator/scaler/writer。
    """
    from engine.core import YAMLConfig
    from engine.optim.lr_scheduler import FlatCosineLRScheduler

    cfg = YAMLConfig(cfg_path)

    # seed 未指定时沿用配置 seed（spec §3）；配置也无 seed 时回退 0（与 train.py 默认对齐）。
    # 在构建 model/optimizer/dataloader 之前应用，保证随机初始化与 shuffle 可复现。
    effective_seed = seed if seed is not None else getattr(cfg, "seed", None)
    effective_seed = effective_seed if effective_seed is not None else 0
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)

    dev = torch.device(device)
    model = cfg.model.to(dev)
    criterion = cfg.criterion.to(dev)
    optimizer = cfg.optimizer

    loader = cfg.train_dataloader
    train_ds = loader.dataset
    if getattr(train_ds, "cache_images", "none") == "disk":
        train_ds.precache_images(num_workers=8)

    lr_scheduler = None
    if getattr(cfg, "lrsheduler", None) is not None:
        lr_scheduler = FlatCosineLRScheduler(
            optimizer,
            cfg.lr_gamma,
            iter_per_epoch=len(loader),
            total_epochs=cfg.epoches,
            warmup_iter=cfg.warmup_iter,
            flat_epochs=cfg.flat_epoch,
            no_aug_epochs=cfg.no_aug_epoch,
        )

    recovery = {"fidelity": "weights_only", "start_epoch": start_epoch if start_epoch is not None else 0, "notes": []}
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        recovery = restore_checkpoint(
            model, optimizer, state, start_epoch_override=start_epoch
        )

    pipeline = Pipeline(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=loader,
        device=dev,
        probe=None,  # main() 中安装 GeometryProbe
    )

    git_meta = {}
    try:
        import subprocess

        git_meta["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        git_meta["dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        )
    except Exception:
        pass

    meta = {
        "config_path": cfg_path,
        "config_sha256": _sha256(cfg_path),
        "checkpoint_path": checkpoint_path or None,
        "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
        "device": device,
        "seed": seed,
        "effective_seed": effective_seed,
        "use_amp": bool(getattr(cfg, "use_amp", False)),
        "clip_max_norm": float(getattr(cfg, "clip_max_norm", 0.0)),
        "recovery": recovery,
        "git": git_meta,
        "amp_dtype": "bfloat16",
        "detect_anomaly": True,
        "dataset": str(getattr(train_ds, "img_folder", "")),
        "dataloader_len": len(loader),
    }
    return pipeline, meta


def _collect_env() -> dict:
    import datetime

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 装配：构建 → manifest → probe → 运行。返回退出码 0/2/3/4。"""
    args = parse_args(argv)

    try:
        output_dir = ensure_output_dir(args.output_dir, args.overwrite)
    except FileExistsError as e:
        print(f"[ERROR] {e}")
        return 3

    probe = GeometryProbe()
    probe.install()
    try:
        pipeline, meta = build_pipeline(
            args.config,
            args.checkpoint,
            args.device,
            args.start_epoch,
            args.seed,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] pipeline 构建失败: {type(e).__name__}: {e}")
        try:
            write_run_manifest(
                output_dir / "run_manifest.json",
                {
                    **_collect_env(),
                    "config_path": args.config,
                    "checkpoint_path": args.checkpoint,
                    "error": f"{type(e).__name__}: {e}",
                    "exit_code": 3,
                },
            )
        except Exception:
            pass
        return 3

    meta["env"] = _collect_env()
    meta["probe_atan2_zero"] = atan2_zero_probe(str(args.device).split(":")[0])
    meta["cli"] = vars(args)
    # 必须在 write_run_manifest 之前覆盖：build_pipeline 硬编码 True，
    # --no-detect-anomaly 时不得把实际关闭误记为 True（re-review 发现放置顺序问题）
    meta["detect_anomaly"] = args.detect_anomaly
    write_run_manifest(output_dir / "run_manifest.json", meta)
    (output_dir / "command.txt").write_text("python " + " ".join(sys.argv) + "\n")

    # run_diagnostic 需要的运行参数：use_amp/clip_max_norm/start_epoch 从配置/恢复结果注入
    #（parse_args 无这些参数或默认 None）。clip_max_norm/start_epoch 缺失会导致
    # run_diagnostic 首个有限 step 抛 AttributeError/TypeError → 伪 exit 4
    #（最终 review 发现并修正：配置含 clip_max_norm=0.1；start_epoch 由恢复结果解析）。
    args.use_amp = meta["use_amp"]
    args.effective_seed = meta["effective_seed"]
    args.clip_max_norm = meta["clip_max_norm"]
    args.start_epoch = meta["recovery"]["start_epoch"]

    pipeline.probe = probe
    code = run_diagnostic(pipeline, args)
    probe.uninstall()
    return code


if __name__ == "__main__":
    sys.exit(main())
