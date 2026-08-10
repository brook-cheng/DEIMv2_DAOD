#!/usr/bin/env python3
"""DEIMv2-OBB rep2 保存失败回放工具。

对远程诊断 runner 捕获的失败现场（trigger_batch.pt / model_state.pt /
optimizer_state.pt / failure_summary.json）做单步重放：

- 用 YAMLConfig 仅构建 model / criterion / optimizer（不访问 dataloader）；
- 恢复保存的权重与优化器状态；
- 镜像 run_diagnostic 的单步顺序执行一次 forward → loss → backward；
- 按种类分类结果：正常 0 / 数值异常 2 / 配置错误 3 / 运行时错误 4。

设计依据: docs/superpowers/plans/2026-08-10-rep2-stable-atan2.md Task 4。
"""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils._pytree import tree_map

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tool_diagnose_rep2_nan import scan_gradients  # noqa: E402

EXIT_OK = 0
EXIT_NUMERIC = 2
EXIT_CONFIG = 3
EXIT_RUNTIME = 4

MANDATORY_ARTIFACTS = (
    "trigger_batch.pt",
    "model_state.pt",
    "optimizer_state.pt",
)

PUBLIC_OUTPUT_KEYS = ("pred_logits", "pred_boxes", "pred_corners", "ref_points")


def load_failure_artifacts(failure_dir: Path | str) -> dict:
    """加载失败现场必需产物，返回 ``dict``。

    缺失任一强制产物抛 ``FileNotFoundError`` 并点名缺失文件；
    ``failure_summary.json`` 缺失时元数据归零。
    """
    fail_dir = Path(failure_dir)
    if not fail_dir.is_dir():
        raise FileNotFoundError(f"failure dir not found: {fail_dir}")
    missing = [n for n in MANDATORY_ARTIFACTS if not (fail_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing mandatory artifacts in {fail_dir}: {', '.join(missing)}"
        )
    trigger_batch = torch.load(
        fail_dir / "trigger_batch.pt", map_location="cpu", weights_only=False
    )
    model_state = torch.load(
        fail_dir / "model_state.pt", map_location="cpu", weights_only=False
    )
    optimizer_state = torch.load(
        fail_dir / "optimizer_state.pt", map_location="cpu", weights_only=False
    )
    summary_path = fail_dir / "failure_summary.json"
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {"epoch": 0, "step": 0, "global_step": 0}
    return {
        "trigger_batch": trigger_batch,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "failure_summary": summary,
    }


def restore_states(model, optimizer, artifacts: dict) -> None:
    """恢复 model 权重（strict=False，缺键/多余键视为配置错误）与优化器状态。"""
    res = model.load_state_dict(artifacts["model_state"], strict=False)
    missing = list(res.missing_keys)
    unexpected = list(res.unexpected_keys)
    if missing or unexpected:
        raise ValueError(
            f"model state mismatch: missing={missing} unexpected={unexpected}"
        )
    opt_state = artifacts.get("optimizer_state")
    if opt_state and optimizer is not None:
        optimizer.load_state_dict(opt_state)


def _to_device(samples, targets, device: torch.device):
    samples = samples.to(device)
    targets = [
        {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
        for t in targets
    ]
    return samples, targets


def _scan_params_and_state(model, optimizer) -> list[str]:
    """返回所有非有限参数名 + optimizer 非有限张量状态键；空列表 = 全部有限。"""
    bad: list[str] = []
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            bad.append(f"param:{name}")
    for idx, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                bad.append(f"optimizer[{idx}].{key}")
    return bad


def replay_step(
    model,
    criterion,
    optimizer,
    samples,
    targets,
    *,
    device,
    use_amp,
    step_optimizer,
    clip_max_norm,
    detect_anomaly,
    metas,
) -> dict:
    """单步重放，返回至少含 ``exit_code`` / ``kind`` 的 dict。

    镜像 ``run_diagnostic`` 的单步顺序：forward（含 autocast 与 FP32 提升）→
    输出有限性 → criterion → loss 有限性 → backward（异常/OOM 分类）→
    梯度扫描 → 仅 ``step_optimizer`` 时 clip + step + 参数/优化器状态扫描。

    forward 阶段的异常不在此吞掉，交由 ``main`` 兜底为 exit 4。
    """
    from engine.solver.training_diagnostics import (
        raise_for_nonfinite_losses,
        raise_for_nonfinite_total,
    )

    model.train()
    criterion.train()
    anomaly_cm = torch.autograd.detect_anomaly() if detect_anomaly else nullcontext()

    dev = torch.device(device)
    samples, targets = _to_device(samples, targets, dev)

    with anomaly_cm:
        if use_amp:
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
            for k in PUBLIC_OUTPUT_KEYS
            if k in outputs
            and isinstance(outputs[k], torch.Tensor)
            and not torch.isfinite(outputs[k]).all()
        ]
        if bad_keys:
            return {"exit_code": EXIT_NUMERIC, "kind": "forward_output", "bad_keys": bad_keys}

        loss_dict = criterion(outputs, targets, **metas)

        try:
            raise_for_nonfinite_losses(
                loss_dict,
                epoch=metas["epoch"],
                step=metas["step"],
                global_step=metas["global_step"],
            )
            loss = sum(loss_dict.values())
            raise_for_nonfinite_total(
                loss,
                epoch=metas["epoch"],
                step=metas["step"],
                global_step=metas["global_step"],
            )
        except FloatingPointError:
            return {
                "exit_code": EXIT_NUMERIC,
                "kind": "loss",
                "loss_dict": {
                    k: (float(v.detach().cpu()) if isinstance(v, torch.Tensor) else None)
                    for k, v in loss_dict.items()
                },
            }

        grad_norm = None
        try:
            loss.backward()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return {"exit_code": EXIT_RUNTIME, "kind": "cuda_oom", "error": str(e)}
            return {"exit_code": EXIT_NUMERIC, "kind": "backward_anomaly", "error": str(e)}

    grad_norm, anomalies = scan_gradients(model)
    if anomalies:
        return {
            "exit_code": EXIT_NUMERIC,
            "kind": "gradient",
            "aggregate_norm": grad_norm,
            "anomaly_params": [a["name"] for a in anomalies],
        }

    if step_optimizer:
        if optimizer is None:
            raise ValueError("--step-optimizer requires an optimizer")
        if clip_max_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_max_norm
            )
        optimizer.step()
        optimizer.zero_grad()
        bad = _scan_params_and_state(model, optimizer)
        if bad:
            return {
                "exit_code": EXIT_NUMERIC,
                "kind": "post_step_nonfinite",
                "names": bad,
            }

    return {"exit_code": EXIT_OK, "kind": "ok", "grad_norm": grad_norm}


def _build_components(config_path: str, device: str):
    """用 YAMLConfig 仅构建 model / criterion / optimizer，返回 4 元组含 use_amp。"""
    from engine.core import YAMLConfig

    cfg = YAMLConfig(config_path)
    dev = torch.device(device)
    model = cfg.model.to(dev)
    criterion = cfg.criterion.to(dev)
    optimizer = cfg.optimizer
    use_amp = bool(getattr(cfg, "use_amp", False))
    return model, criterion, optimizer, use_amp


def _derive_metas(artifacts: dict) -> dict:
    """从 failure_summary 提取 criterion 需要的元数据；缺键归零。"""
    s = artifacts["failure_summary"]
    return {
        "epoch": int(s.get("epoch", 0)),
        "step": int(s.get("step", 0)),
        "global_step": int(s.get("global_step", 0)),
        "epoch_step": int(s.get("epoch_step", 0)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEIMv2-OBB rep2 保存失败单步回放工具"
    )
    parser.add_argument("--config", type=str, required=True, help="训练 YAML 配置")
    parser.add_argument("--failure-dir", type=str, required=True, help="失败现场目录")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--step-optimizer", action="store_true", default=False)
    parser.add_argument("--clip-max-norm", type=float, default=0.0)
    parser.add_argument(
        "--detect-anomaly", dest="detect_anomaly", action="store_true", default=True
    )
    parser.add_argument("--no-detect-anomaly", dest="detect_anomaly", action="store_false")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 装配：加载现场 → 构建 → 恢复 → 重放。返回 0/2/3/4。"""
    args = parse_args(argv)

    try:
        artifacts = load_failure_artifacts(args.failure_dir)
        model, criterion, optimizer, use_amp = _build_components(
            args.config, args.device
        )
        restore_states(model, optimizer, artifacts)
    except Exception as e:  # noqa: BLE001 - 构建/加载/恢复失败统一 exit 3
        print(f"[ERROR] {type(e).__name__}: {e}")
        return EXIT_CONFIG

    metas = _derive_metas(artifacts)
    samples = artifacts["trigger_batch"]["samples"]
    targets = artifacts["trigger_batch"]["targets"]

    try:
        result = replay_step(
            model,
            criterion,
            optimizer,
            samples,
            targets,
            device=args.device,
            use_amp=use_amp,
            step_optimizer=args.step_optimizer,
            clip_max_norm=args.clip_max_norm,
            detect_anomaly=args.detect_anomaly,
            metas=metas,
        )
    except Exception as e:  # noqa: BLE001 - 未分类重放失败统一 exit 4
        print(f"[ERROR] replay failed: {type(e).__name__}: {e}")
        return EXIT_RUNTIME

    for key, value in result.items():
        if isinstance(value, list):
            value = ",".join(str(v) for v in value) if value else ""
        print(f"{key}={value}")
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
