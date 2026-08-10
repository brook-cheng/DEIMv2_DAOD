# rep2 NaN 远程诊断 runner 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增独立单 GPU 诊断脚本 `test/tool_diagnose_rep2_nan.py` 与纯 CPU 测试 `test/test_rep2_nan_diagnostic.py`，在远程训练服务器复现 rep2 `angle_rep=2` 训练的 NaN 梯度，自动定位首个异常算子并完整持久化失败现场。

**Architecture:** 脚本复用 `YAMLConfig` 构建 model/criterion/optimizer/train_dataloader，但不调用 `solver.fit()`；自带最小训练循环（BF16 autocast forward → `tree_map` 转 FP32 → FP32 criterion → anomaly backward → 全参数梯度扫描 → 仅有限时 step）。checkpoint 恢复自动分类（full/partial/weights_only）并记录 `recovery_fidelity`。几何探针通过包装 `dfine_utils.external_xyxy_rect_to_oriented_box` 与 `deim_decoder.external_xywh_rect_to_oriented_box` 两个已绑定调用点，只读统计 atan2 退化输入，不改动生产张量。

**Tech Stack:** Python 3.11+、PyTorch ≥ 2.5（`torch.autocast` BF16、`torch.autograd.set_detect_anomaly`）、`torch.utils._pytree.tree_map`、pytest、标准库 `argparse/json/pathlib/hashlib/sys/platform`。

## Global Constraints

- 只新增两个文件：`test/tool_diagnose_rep2_nan.py` 与 `test/test_rep2_nan_diagnostic.py`；**不修改** `engine/` 下任何生产代码。
- 单 GPU、单进程；不执行验证、Comet 上报、EMA 指标比较、DDP。
- 不尝试修复 rep2 数值问题；脚本只收集证据。
- 精度路径与正式训练一致：BF16 autocast forward → 浮点输出递归 `tree_map` 转 FP32 → FP32 criterion。`use_amp` 从配置合并值为 `True`（`sp_fz_common.yml` 链，已用 `YAMLConfig('configs/custom_obb/dlzdt/ablation/abl_rep2.yml')` 实测确认）。
- checkpoint 语义完全对齐 `BaseSolver.state_dict()/load_state_dict()`（`engine/solver/_solver.py`）：顶层键含 `date`、`last_epoch` 及各组件的 `state_dict()`；EMA 结构为 `{"module": ..., "updates": ...}`；加载用 `torch.load(path, map_location="cpu", weights_only=False)`。
- 退出码契约：`0`=运行上限内无异常；`2`=成功捕获非有限 forward/loss/backward/gradient 并保存现场；`3`=配置/checkpoint/数据集/恢复阶段失败；`4`=CUDA OOM 或其他运行时异常（仍保存 traceback）。
- 产物目录结构、字段名严格按设计 spec §7；异常 step 绝不执行 `optimizer.step()`。
- 非空 output-dir 默认拒绝覆盖，`--overwrite` 显式放行。
- 工作区有 4 个用户既有改动（`test/tool_deimv2_obb_infer.py`、`test/tool_dlzdt_obb_compare.py`、`tools/model_compare/obb_utils.py`、`test/test_obb_utils.py`），**不得**混入本计划提交。
- 测试运行方式：`python -m pytest test/test_rep2_nan_diagnostic.py -v`（仓库无 pytest.ini；测试文件自行 `sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))`，与 `test/test_det_engine_diagnostics.py` 一致；同目录模块用 `import tool_diagnose_rep2_nan` 导入）。

---

### Task 1: 模块骨架 + `tensor_stats` + `scan_gradients`

**Files:**
- Create: `test/tool_diagnose_rep2_nan.py`
- Create: `test/test_rep2_nan_diagnostic.py`

**Interfaces:**
- Produces:
  - `tensor_stats(t: torch.Tensor) -> dict[str, int | float | None]` — 有限性统计：`{shape, dtype, device, finite, nan, pos_inf, neg_inf, min, max, absmax}`；`min/max/absmax` 只对有限元素统计（无有限元素时为 `None`）。
  - `scan_gradients(model: torch.nn.Module) -> tuple[float, list[dict]]` — 返回 `(aggregate_norm, anomalies)`；`anomalies` 为**全部**非有限梯度参数列表（不在首个参数停止），每项 `{name, shape, dtype, nan, pos_inf, neg_inf, finite_min, finite_max, finite_norm}`。

- [ ] **Step 1: 创建模块骨架（仅 sys.path + docstring + `__main__` 占位）**

`test/tool_diagnose_rep2_nan.py`：

```python
#!/usr/bin/env python3
"""DEIMv2-OBB rep2 NaN 远程诊断 runner。

设计依据: docs/superpowers/specs/2026-08-10-rep2-nan-diagnostic-runner-design.md
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main(argv=None) -> int:
    raise NotImplementedError("main 在 Task 7 实现")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 编写失败测试**

`test/test_rep2_nan_diagnostic.py`：

```python
"""rep2 NaN 诊断 runner 单元测试（纯 CPU）。"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402 - 模块级导入（Task 7 skipif 装饰器依赖）
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def _make_param(name, shape=(4,), *, grad=None):
    # grad 提供时以 grad 形状为准（test_skips/test_finite 用长度 2 的 grad）
    if grad is not None:
        shape = (
            grad.shape
            if isinstance(grad, torch.Tensor)
            else torch.tensor(grad).shape
        )
    p = nn.Parameter(torch.randn(shape))
    if grad is not None:
        p.grad = (
            grad.clone().detach()
            if isinstance(grad, torch.Tensor)
            else torch.tensor(grad, dtype=torch.float32)
        )
    return p


class TestTensorStats:

    def test_finite_tensor(self):
        from tool_diagnose_rep2_nan import tensor_stats

        t = torch.tensor([1.0, -2.0, 3.5])
        s = tensor_stats(t)
        assert s["finite"] == 3
        assert s["nan"] == 0
        assert s["pos_inf"] == 0
        assert s["neg_inf"] == 0
        assert s["min"] == -2.0
        assert s["max"] == 3.5
        assert s["absmax"] == 3.5

    def test_mixed_nonfinite(self):
        from tool_diagnose_rep2_nan import tensor_stats

        t = torch.tensor([float("nan"), float("inf"), float("-inf"), 0.0, 5.0])
        s = tensor_stats(t)
        assert s["finite"] == 2
        assert s["nan"] == 1
        assert s["pos_inf"] == 1
        assert s["neg_inf"] == 1
        assert s["min"] == 0.0
        assert s["max"] == 5.0
        assert s["absmax"] == 5.0

    def test_all_nan_has_none_extremes(self):
        from tool_diagnose_rep2_nan import tensor_stats

        t = torch.tensor([float("nan"), float("nan")])
        s = tensor_stats(t)
        assert s["finite"] == 0
        assert s["nan"] == 2
        assert s["min"] is None
        assert s["max"] is None
        assert s["absmax"] is None

    def test_empty_tensor(self):
        from tool_diagnose_rep2_nan import tensor_stats

        s = tensor_stats(torch.empty(0))
        assert s["finite"] == 0
        assert s["min"] is None


class TestScanGradients:

    def test_reports_all_nonfinite_params_not_first_only(self):
        from tool_diagnose_rep2_nan import scan_gradients

        model = nn.Module()
        model.register_parameter("p0", _make_param("p0", grad=[1.0, 2.0, 3.0, 4.0]))
        model.register_parameter("p1", _make_param("p1", grad=[float("nan"), 1.0, 2.0, 3.0]))
        model.register_parameter("p2", _make_param("p2", grad=[float("inf"), 1.0, 2.0, 3.0]))
        norm, anomalies = scan_gradients(model)
        assert len(anomalies) == 2
        names = {a["name"] for a in anomalies}
        assert names == {"p1", "p2"}
        assert all(a["nan"] + a["pos_inf"] + a["neg_inf"] > 0 for a in anomalies)
        assert norm > 0.0

    def test_skips_params_without_grad(self):
        from tool_diagnose_rep2_nan import scan_gradients

        model = nn.Module()
        model.register_parameter("a", _make_param("a", grad=[1.0, 2.0]))
        model.register_parameter("b", _make_param("b", grad=None))
        norm, anomalies = scan_gradients(model)
        assert anomalies == []
        assert norm == float(1.0**2 + 2.0**2) ** 0.5

    def test_empty_grads_returns_zero(self):
        from tool_diagnose_rep2_nan import scan_gradients

        model = nn.Module()
        model.register_parameter("a", _make_param("a", grad=None))
        norm, anomalies = scan_gradients(model)
        assert anomalies == []
        assert norm == 0.0

    def test_finite_grad_reports_zero_anomalies(self):
        from tool_diagnose_rep2_nan import scan_gradients

        model = nn.Module()
        model.register_parameter("a", _make_param("a", grad=[1.0, 2.0]))
        norm, anomalies = scan_gradients(model)
        assert anomalies == []
        assert norm > 0.0
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tool_diagnose_rep2_nan'`

- [ ] **Step 4: 最小实现**

追加到 `test/tool_diagnose_rep2_nan.py`（文件顶部加 `import torch`）：

```python
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
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: PASS（TestTensorStats 4 个 + TestScanGradients 4 个 = 8 个）

- [ ] **Step 6: Commit**

```bash
git add test/tool_diagnose_rep2_nan.py test/test_rep2_nan_diagnostic.py
git commit -m "test: rep2 诊断 runner 骨架与有限性统计/梯度全扫描（TDD Task 1）"
```

---

### Task 2: checkpoint 分类 `checkpoint_inspect` + 输出目录保护 `ensure_output_dir`

**Files:**
- Modify: `test/tool_diagnose_rep2_nan.py`
- Modify: `test/test_rep2_nan_diagnostic.py`

**Interfaces:**
- Consumes: Task 1 骨架。
- Produces:
  - `checkpoint_inspect(state: dict) -> dict` — 返回 `{kind, top_keys, last_epoch, has_model, has_ema, has_optimizer, notes}`；`kind ∈ {"full", "weights_only", "invalid"}`。判定：`last_epoch`（int）且存在 model/ema 且存在 optimizer → `full`；存在 model/ema（或裸参数键）且无 optimizer → `weights_only`；否则 `invalid`。
  - `ensure_output_dir(output_dir: Path | str, overwrite: bool) -> Path` — 非空目录且非 overwrite 时抛 `FileExistsError`；`overwrite=True` 时 `shutil.rmtree` 清空重建；返回 `Path`。

- [ ] **Step 1: 编写失败测试**

```python
class TestCheckpointInspect:

    def test_full_checkpoint(self):
        from tool_diagnose_rep2_nan import checkpoint_inspect

        state = {
            "date": "2026-08-10T00:00:00",
            "last_epoch": 87,
            "model": {"backbone.sta.stem.0.weight": torch.zeros(2, 2)},
            "ema": {"module": {}, "updates": 100},
            "optimizer": {"state": {}, "param_groups": []},
        }
        r = checkpoint_inspect(state)
        assert r["kind"] == "full"
        assert r["last_epoch"] == 87
        assert r["has_model"] is True
        assert r["has_ema"] is True
        assert r["has_optimizer"] is True

    def test_weights_only_with_model_key(self):
        from tool_diagnose_rep2_nan import checkpoint_inspect

        state = {"model": {"a": torch.zeros(1)}}
        r = checkpoint_inspect(state)
        assert r["kind"] == "weights_only"
        assert r["has_optimizer"] is False
        assert r["last_epoch"] is None

    def test_weights_only_ema_module(self):
        from tool_diagnose_rep2_nan import checkpoint_inspect

        state = {"ema": {"module": {"a": torch.zeros(1)}, "updates": 5}}
        r = checkpoint_inspect(state)
        assert r["kind"] == "weights_only"
        assert r["has_model"] is False
        assert r["has_ema"] is True

    def test_weights_only_bare_state_dict(self):
        from tool_diagnose_rep2_nan import checkpoint_inspect

        state = {"backbone.sta.stem.0.weight": torch.zeros(2, 2)}
        r = checkpoint_inspect(state)
        assert r["kind"] == "weights_only"

    def test_invalid_empty_dict(self):
        from tool_diagnose_rep2_nan import checkpoint_inspect

        r = checkpoint_inspect({})
        assert r["kind"] == "invalid"

    def test_invalid_non_dict_value(self):
        from tool_diagnose_rep2_nan import checkpoint_inspect

        r = checkpoint_inspect({"model": "not-a-dict"})
        assert r["kind"] == "invalid"


class TestEnsureOutputDir:

    def test_new_dir_created(self, tmp_path):
        from tool_diagnose_rep2_nan import ensure_output_dir

        d = tmp_path / "out"
        out = ensure_output_dir(d, overwrite=False)
        assert out == d and d.is_dir()

    def test_empty_existing_dir_allowed(self, tmp_path):
        from tool_diagnose_rep2_nan import ensure_output_dir

        d = tmp_path / "out"
        d.mkdir()
        ensure_output_dir(d, overwrite=False)  # 不抛异常

    def test_nonempty_refused_without_overwrite(self, tmp_path):
        import pytest
        from tool_diagnose_rep2_nan import ensure_output_dir

        d = tmp_path / "out"
        d.mkdir()
        (d / "events.jsonl").write_text("{}")
        with pytest.raises(FileExistsError):
            ensure_output_dir(d, overwrite=False)

    def test_overwrite_clears_dir(self, tmp_path):
        from tool_diagnose_rep2_nan import ensure_output_dir

        d = tmp_path / "out"
        d.mkdir()
        (d / "events.jsonl").write_text("{}")
        ensure_output_dir(d, overwrite=True)
        assert list(d.iterdir()) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: FAIL — `ImportError: cannot import name 'checkpoint_inspect'`

- [ ] **Step 3: 最小实现**

追加到 `test/tool_diagnose_rep2_nan.py`（文件顶部加 `import shutil`、`from pathlib import Path`）：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: PASS（新增 10 个）

- [ ] **Step 5: Commit**

```bash
git add test/tool_diagnose_rep2_nan.py test/test_rep2_nan_diagnostic.py
git commit -m "test: checkpoint 分类与输出目录保护（TDD Task 2）"
```

---

### Task 3: `atan2_zero_probe` + `compute_edge_stats` + `GeometryProbe`

**Files:**
- Modify: `test/tool_diagnose_rep2_nan.py`
- Modify: `test/test_rep2_nan_diagnostic.py`

**Interfaces:**
- Consumes: Task 1 骨架。
- Produces:
  - `atan2_zero_probe(device: str = "cpu") -> dict` — 运行 `torch.atan2(0,0)` forward + backward，返回 `{device, forward_value, forward_finite, backward_grads_finite, grad_x, grad_y}`。
  - `compute_edge_stats(external_rect: torch.Tensor, vertex_offsets: torch.Tensor, eps: float = 1e-9) -> dict` — 纯函数，detach 后复刻 `obb_geometry.external_xyxy_rect_to_oriented_box` 第 209–238 行（v0/v1/v2 顶点、edge_ab/edge_bc、len、w 选择与 atan2 输入 w_dx/w_dy）：返回 `{n, w_min, h_min, edge_ab_zero, edge_bc_zero, near_zero_edges, atan2_zero_inputs, w_dx_absmin, w_dy_absmin, eps_min, eta_min, eps_max, eta_max}`。
  - `class GeometryProbe` — `install() -> None`（包装 `engine.deim.dfine_utils.external_xyxy_rect_to_oriented_box` 与 `engine.deim.deim_decoder.external_xywh_rect_to_oriented_box` 两个模块命名空间内已绑定名字）、`uninstall() -> None`、`reset() -> None`、`snapshot() -> dict`。

- [ ] **Step 1: 编写失败测试**

```python
class TestAtan2ZeroProbe:

    def test_forward_finite_backward_nonfinite(self):
        from tool_diagnose_rep2_nan import atan2_zero_probe

        r = atan2_zero_probe("cpu")
        assert r["forward_finite"] is True
        assert r["forward_value"] == 0.0
        assert r["backward_grads_finite"] is False


class TestComputeEdgeStats:

    def test_degenerate_edges_detected(self):
        from tool_diagnose_rep2_nan import compute_edge_stats

        # ext_rect=(0,0,0,0), offsets=(0,0) → v0=v1=v2 → 两退化边 + atan2(0,0)
        rect = torch.zeros(1, 1, 4)
        offs = torch.zeros(1, 1, 2)
        s = compute_edge_stats(rect, offs)
        assert s["n"] == 1
        assert s["edge_ab_zero"] == 1
        assert s["edge_bc_zero"] == 1
        assert s["atan2_zero_inputs"] == 1
        # eps=1e-9 稳定化使退化边长 ≈ sqrt(1e-9)=3.16e-5，非 0
        assert s["w_min"] < 1e-4
        assert s["h_min"] < 1e-4

    def test_normal_rect_no_atan2_zero(self):
        from tool_diagnose_rep2_nan import compute_edge_stats

        # ext_rect=(0,0,2,1), offsets=(0.5,0.25)
        rect = torch.tensor([[[0.0, 0.0, 2.0, 1.0]]])
        offs = torch.tensor([[[0.5, 0.25]]])
        s = compute_edge_stats(rect, offs)
        assert s["edge_ab_zero"] == 0
        assert s["edge_bc_zero"] == 0
        assert s["atan2_zero_inputs"] == 0
        assert s["w_min"] > 1.0
        assert s["h_min"] > 0.5

    def test_returns_scalar_types(self):
        from tool_diagnose_rep2_nan import compute_edge_stats

        rect = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
        offs = torch.tensor([[[0.0, 0.0]]])
        s = compute_edge_stats(rect, offs)
        for k in ("n", "edge_ab_zero", "edge_bc_zero", "near_zero_edges", "atan2_zero_inputs"):
            assert isinstance(s[k], int)
        for k in ("w_min", "h_min", "w_dx_absmin", "w_dy_absmin", "eps_min", "eta_min", "eps_max", "eta_max"):
            assert isinstance(s[k], float)


class TestGeometryProbe:

    def test_install_snapshot_uninstall(self):
        from tool_diagnose_rep2_nan import GeometryProbe

        probe = GeometryProbe()
        probe.install()
        try:
            import torch
            # 探针包装的是消费模块绑定（dfine_utils 按名导入），必须经此调用才能命中包装器
            from engine.deim.dfine_utils import external_xyxy_rect_to_oriented_box

            rect = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
            offs = torch.tensor([[[0.0, 0.0]]])
            out = external_xyxy_rect_to_oriented_box(rect, offs)
            snap = probe.snapshot()
            assert snap["calls"] >= 1
            assert snap.get("edge_ab_zero", 0) >= 1
            assert out.shape == (1, 1, 5)  # 输出不受影响
        finally:
            probe.uninstall()

    def test_probe_is_noop_after_uninstall(self):
        from tool_diagnose_rep2_nan import GeometryProbe

        probe = GeometryProbe()
        probe.install()
        probe.uninstall()
        import torch
        from engine.deim.dfine_utils import external_xyxy_rect_to_oriented_box

        rect = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
        offs = torch.tensor([[[0.25, 0.25]]])
        external_xyxy_rect_to_oriented_box(rect, offs)
        assert probe.snapshot()["calls"] == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: FAIL — ImportError（`atan2_zero_probe` 等未定义）

- [ ] **Step 3: 最小实现**

追加到 `test/tool_diagnose_rep2_nan.py`：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: PASS（新增 6 个；TestGeometryProbe 依赖真实 `engine.deim.obb_geometry`，CPU 可运行）

- [ ] **Step 5: Commit**

```bash
git add test/tool_diagnose_rep2_nan.py test/test_rep2_nan_diagnostic.py
git commit -m "test: atan2 奇点探针与 rep2 几何退化边统计（TDD Task 3）"
```

---

### Task 4: 产物写入器（manifest / events / progress / failure）

**Files:**
- Modify: `test/tool_diagnose_rep2_nan.py`
- Modify: `test/test_rep2_nan_diagnostic.py`

**Interfaces:**
- Consumes: Task 1 骨架。
- Produces:
  - `write_run_manifest(path: Path | str, meta: dict) -> None` — `json.dump(meta, indent=2, ensure_ascii=False)`。
  - `append_event(path: Path | str, record: dict) -> None` — JSONL 追加一行。
  - `write_progress(path: Path | str, state: dict) -> None` — `json.dump`。
  - `save_failure(output_dir: Path | str, *, traceback_text: str, failure_summary: dict, trigger_batch: dict, outputs: dict, losses: dict, geometry_snapshot: dict, gradients_summary: dict, model_state: dict, optimizer_state: dict) -> dict` — 按 spec §7.3 写入 `failure/` 目录；所有 tensor detach+CPU；任一产物保存失败追加到 `failure_summary["secondary_errors"]`，不覆盖原 traceback；返回产物路径字典。

- [ ] **Step 1: 编写失败测试**

```python
class TestArtifactWriters:

    def test_write_run_manifest(self, tmp_path):
        import json
        from tool_diagnose_rep2_nan import write_run_manifest

        p = tmp_path / "run_manifest.json"
        write_run_manifest(p, {"a": 1, "b": {"c": [1, 2]}})
        assert json.loads(p.read_text()) == {"a": 1, "b": {"c": [1, 2]}}

    def test_append_event_jsonl(self, tmp_path):
        import json
        from tool_diagnose_rep2_nan import append_event

        p = tmp_path / "events.jsonl"
        append_event(p, {"epoch": 0, "step": 0})
        append_event(p, {"epoch": 0, "step": 1})
        lines = p.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["step"] == 1

    def test_write_progress(self, tmp_path):
        import json
        from tool_diagnose_rep2_nan import write_progress

        p = tmp_path / "progress.json"
        write_progress(p, {"epoch": 3, "global_step": 42})
        assert json.loads(p.read_text())["global_step"] == 42

    def test_save_failure_moves_tensors_to_cpu(self, tmp_path):
        import json
        from tool_diagnose_rep2_nan import save_failure

        out = tmp_path / "diag"
        paths = save_failure(
            out,
            traceback_text="Traceback (most recent call last):\n  boom",
            failure_summary={"exit_code": 2, "kind": "gradient"},
            trigger_batch={
                "samples": torch.randn(2, 3, 4, 4),
                "targets": [{"boxes": torch.randn(1, 5)}],
            },
            outputs={"pred_boxes": torch.randn(2, 10, 5)},
            losses={"loss_total": torch.tensor(float("nan"))},
            geometry_snapshot={"atan2_zero_inputs": 3},
            gradients_summary={"aggregate_norm": 1.0, "anomalies": []},
            model_state={"m": torch.randn(2)},
            optimizer_state={"state": {}},
        )
        fail_dir = out / "failure"
        assert (fail_dir / "traceback.txt").exists()
        assert (fail_dir / "trigger_batch.pt").exists()
        assert (fail_dir / "outputs.pt").exists()
        assert (fail_dir / "losses.pt").exists()
        assert (fail_dir / "model_state.pt").exists()
        assert (fail_dir / "optimizer_state.pt").exists()
        fs = json.loads((fail_dir / "failure_summary.json").read_text())
        assert fs["kind"] == "gradient"
        tb = torch.load(fail_dir / "trigger_batch.pt", map_location="cpu", weights_only=False)
        assert tb["samples"].device.type == "cpu"
        assert tb["samples"].requires_grad is False
        assert paths["traceback"].endswith("traceback.txt")

    def test_save_failure_records_secondary_errors(self, tmp_path, monkeypatch):
        import json
        from tool_diagnose_rep2_nan import save_failure

        out = tmp_path / "diag"
        save_failure(
            out,
            traceback_text="t",
            failure_summary={},
            trigger_batch={"samples": torch.randn(2)},
            outputs={},
            losses={},
            geometry_snapshot={},
            gradients_summary={},
            model_state={},
            optimizer_state={},
        )
        fs = json.loads((out / "failure" / "failure_summary.json").read_text())
        assert "secondary_errors" in fs
        assert (out / "failure" / "traceback.txt").read_text() == "t"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: 最小实现**

追加到 `test/tool_diagnose_rep2_nan.py`（文件顶部加 `import json`）：

```python
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
    _safe("failure_summary", artifacts["failure_summary"], failure_summary, text=True)
    _safe("trigger_batch", artifacts["trigger_batch"], trigger_batch)
    _safe("outputs", artifacts["outputs"], outputs)
    _safe("losses", artifacts["losses"], losses)
    _safe("geometry_snapshot", artifacts["geometry_snapshot"], geometry_snapshot)
    _safe("gradients_summary", artifacts["gradients_summary"], gradients_summary, text=True)
    _safe("model_state", artifacts["model_state"], model_state)
    _safe("optimizer_state", artifacts["optimizer_state"], optimizer_state)

    return {k: str(v) for k, v in artifacts.items()}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: PASS（新增 5 个）

- [ ] **Step 5: Commit**

```bash
git add test/tool_diagnose_rep2_nan.py test/test_rep2_nan_diagnostic.py
git commit -m "test: 诊断产物写入器 manifest/events/progress/failure（TDD Task 4）"
```

---

### Task 5: checkpoint 状态恢复 `restore_checkpoint`（含 fidelity 报告）

**Files:**
- Modify: `test/tool_diagnose_rep2_nan.py`
- Modify: `test/test_rep2_nan_diagnostic.py`

**Interfaces:**
- Consumes: Task 2 `checkpoint_inspect`（信息复用，不强制调用）。
- Produces:
  - `restore_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, state: dict, start_epoch_override: int | None = None) -> dict` — 返回 `{start_epoch, fidelity, loaded, missing, unexpected, notes}`。`fidelity ∈ {"full", "partial", "weights_only"}`。规则：
    - `last_epoch = state.get("last_epoch", -1)`；`start_epoch = override if override is not None else (last_epoch + 1 if isinstance(last_epoch, int) else 0)`。
    - 优先 `state["model"]`（`load_state_dict(strict=False)`，记录 missing/unexpected）；无 model 时用 `state["ema"]["module"]`（记 note）。
    - `state["optimizer"]` 存在且 optimizer 非 None 时加载；失败 → `fidelity="partial"` + error note。
    - 完全无 model/ema.module → 抛 `ValueError`（由 `main` 映射为退出码 3）。
    - 无 optimizer 键 → `fidelity="weights_only"`。

- [ ] **Step 1: 编写失败测试**

```python
class TestRestoreCheckpoint:

    def _model(self):
        return torch.nn.Linear(2, 2)

    def test_full_restore(self):
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        state = {
            "last_epoch": 87,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        r = restore_checkpoint(model, optimizer, state)
        assert r["start_epoch"] == 88
        assert r["fidelity"] == "full"
        assert r["loaded"]["model"] == "ok"
        assert r["loaded"]["optimizer"] == "ok"
        assert r["missing"] == []
        assert r["unexpected"] == []

    def test_start_epoch_override(self):
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        state = {
            "last_epoch": 87,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        r = restore_checkpoint(model, optimizer, state, start_epoch_override=10)
        assert r["start_epoch"] == 10

    def test_weights_only(self):
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        state = {"model": model.state_dict()}
        r = restore_checkpoint(model, None, state)
        assert r["fidelity"] == "weights_only"
        assert r["start_epoch"] == 0  # last_epoch 缺失 → -1 + 1

    def test_weights_only_ema_module(self):
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        state = {"ema": {"module": model.state_dict(), "updates": 5}}
        r = restore_checkpoint(model, None, state)
        assert r["fidelity"] == "weights_only"
        assert any("ema.module" in n for n in r["notes"])

    def test_partial_on_optimizer_failure(self):
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        bad_opt = {
            "state": {"999999": {}},
            "param_groups": [
                {
                    "lr": 1e-4,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                    "params": [999999],
                }
            ],
        }
        state = {"last_epoch": 3, "model": model.state_dict(), "optimizer": bad_opt}
        r = restore_checkpoint(model, optimizer, state)
        assert r["fidelity"] == "partial"
        assert r["loaded"]["optimizer"] != "ok"

    def test_missing_unexpected_recorded(self):
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        sd = dict(model.state_dict())
        sd["extra.key"] = torch.zeros(1)  # unexpected
        r = restore_checkpoint(model, None, {"model": sd})
        assert "extra.key" in r["unexpected"]
        assert r["loaded"]["model"] == "ok"

    def test_invalid_raises(self):
        import pytest
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        with pytest.raises(ValueError):
            restore_checkpoint(model, None, {"optimizer": {}})
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: 最小实现**

追加到 `test/tool_diagnose_rep2_nan.py`：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: PASS（新增 7 个）

- [ ] **Step 5: Commit**

```bash
git add test/tool_diagnose_rep2_nan.py test/test_rep2_nan_diagnostic.py
git commit -m "test: checkpoint 状态恢复与 recovery_fidelity 报告（TDD Task 5）"
```

---

### Task 6: 主诊断循环 `run_diagnostic`（Pipeline dataclass + 完整 step 流程）

**Files:**
- Modify: `test/tool_diagnose_rep2_nan.py`
- Modify: `test/test_rep2_nan_diagnostic.py`

**Interfaces:**
- Consumes: Task 1 `scan_gradients`、Task 3 `GeometryProbe`、Task 4 `append_event`/`write_progress`/`save_failure`；`engine.solver.training_diagnostics` 的 `raise_for_nonfinite_losses`/`raise_for_nonfinite_total`（复用，不重写）。
- Produces:
  - `@dataclass class Pipeline` — `model, criterion, optimizer, lr_scheduler, train_dataloader, device, probe=None`。
  - `run_diagnostic(pipeline: Pipeline, args) -> int` — 完整训练循环，返回退出码 0/2/4。`args` 为 `argparse.Namespace`，字段：`start_epoch, max_epochs, max_steps_per_epoch, clip_max_norm, use_amp, detect_anomaly, output_dir, save_every_steps`。
  - 循环语义（与 spec §5 对齐）：`loader.set_epoch(epoch)` → BF16 autocast forward（`use_amp=True` 时）→ `tree_map` 转 FP32 → 检查 forward 输出有限 → FP32 criterion → `raise_for_nonfinite_losses`/`raise_for_nonfinite_total` → anomaly backward → `scan_gradients`（报告全部异常参数）→ 仅全部有限时 clip/step/zero_grad → `lr_scheduler.step(cur_iters + i, optimizer)`（flatcosine 按 iter 步进，与 `det_solver.fit()` 一致）→ 每 step 写 `events.jsonl` → 每 `save_every_steps` 写 `progress.json`。
  - 异常路径绝不执行 optimizer.step；失败时保存完整现场并返回 2；CUDA OOM 返回 4。

- [ ] **Step 1: 编写失败测试（toy pipeline 有限路径 + NaN 路径）**

```python
class ToyModel(torch.nn.Module):
    def __init__(self, nan_at_step=None):
        super().__init__()
        self.nan_at_step = nan_at_step
        self.fc = torch.nn.Linear(4, 5)

    def forward(self, samples, targets=None):
        b = samples.shape[0]
        out = {
            "pred_logits": torch.randn(b, 3, 2),
            "pred_boxes": torch.randn(b, 3, 5),
            "pred_corners": torch.randn(b, 3, 6),
            "ref_points": torch.randn(b, 3, 5),
        }
        return out


class ToyCriterion(torch.nn.Module):
    def forward(self, outputs, targets, **metas):
        return {
            "loss_a": torch.nn.functional.mse_loss(
                outputs["pred_boxes"], torch.zeros_like(outputs["pred_boxes"])
            )
        }


class ToyDataloader:
    def __init__(self, n=4):
        self._n = n

    def __len__(self):
        return self._n

    def __iter__(self):
        for _ in range(self._n):
            samples = torch.randn(2, 4)
            targets = [{"boxes": torch.randn(2, 5), "labels": torch.tensor([0, 1])}]
            yield samples, targets


def _pipeline(nan_at_step=None, probe=None):
    from tool_diagnose_rep2_nan import Pipeline

    model = ToyModel(nan_at_step=nan_at_step)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    return Pipeline(
        model=model,
        criterion=ToyCriterion(),
        optimizer=opt,
        lr_scheduler=None,
        train_dataloader=ToyDataloader(),
        device=torch.device("cpu"),
        probe=probe,
    )


def _args(tmp_path, **kw):
    import argparse

    ns = argparse.Namespace(
        start_epoch=0,
        max_epochs=1,
        max_steps_per_epoch=None,
        clip_max_norm=0.0,
        use_amp=False,
        detect_anomaly=True,
        output_dir=str(tmp_path),
        save_every_steps=50,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class TestRunDiagnostic:

    def test_finite_path_exit_0(self, tmp_path):
        from tool_diagnose_rep2_nan import run_diagnostic

        code = run_diagnostic(_pipeline(), _args(tmp_path))
        assert code == 0
        assert (tmp_path / "events.jsonl").exists()
        lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
        assert len(lines) == 4  # ToyDataloader n=4

    def test_nan_loss_exit_2_saves_failure(self, tmp_path):
        from tool_diagnose_rep2_nan import run_diagnostic

        class NanCriterion(ToyCriterion):
            def forward(self, outputs, targets, **metas):
                return {"loss_a": torch.tensor(float("nan"))}

        pipe = _pipeline()
        pipe.criterion = NanCriterion()
        code = run_diagnostic(pipe, _args(tmp_path))
        assert code == 2
        assert (tmp_path / "failure" / "traceback.txt").exists()
        assert (tmp_path / "failure" / "failure_summary.json").exists()
        assert (tmp_path / "failure" / "trigger_batch.pt").exists()

    def test_nan_backward_exit_2(self, tmp_path):
        from tool_diagnose_rep2_nan import run_diagnostic

        class NanBackwardModel(ToyModel):
            def forward(self, samples, targets=None):
                out = super().forward(samples, targets)
                # 损失依赖 atan2(0,0) → backward 产生 NaN 梯度
                z = torch.atan2(
                    torch.zeros(1, requires_grad=True),
                    torch.zeros(1, requires_grad=True),
                )
                out["pred_boxes"] = out["pred_boxes"] + z
                return out

        pipe = _pipeline()
        pipe.model = NanBackwardModel()
        code = run_diagnostic(pipe, _args(tmp_path))
        assert code == 2

    def test_no_optimizer_step_on_failure(self, tmp_path):
        from tool_diagnose_rep2_nan import run_diagnostic

        pipe = _pipeline()
        before = {n: p.clone() for n, p in pipe.model.named_parameters()}

        class NanBackwardModel(ToyModel):
            def forward(self, samples, targets=None):
                out = super().forward(samples, targets)
                z = torch.atan2(
                    torch.zeros(1, requires_grad=True),
                    torch.zeros(1, requires_grad=True),
                )
                out["pred_boxes"] = out["pred_boxes"] + z
                return out

        pipe.model = NanBackwardModel()
        run_diagnostic(pipe, _args(tmp_path))
        after = {n: p.clone() for n, p in pipe.model.named_parameters()}
        assert all(torch.equal(before[n], after[n]) for n in before)

    def test_max_steps_per_epoch_cap(self, tmp_path):
        from tool_diagnose_rep2_nan import run_diagnostic

        code = run_diagnostic(_pipeline(), _args(tmp_path, max_steps_per_epoch=2))
        assert code == 0
        lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: FAIL — ImportError（`run_diagnostic`/`Pipeline` 未定义）

- [ ] **Step 3: 最小实现**

追加到 `test/tool_diagnose_rep2_nan.py`（文件顶部加 `from dataclasses import dataclass`、`from torch.utils._pytree import tree_map`、`import time`、`import traceback as _tb`）：

```python
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
```

注意：`_fail` 内引用的 `samples`/`targets`/`outputs`/`loss_dict`/`grad_norm`/`anomalies` 已在循环外预初始化为占位（`None`/`{}`/`[]`），所有失败调用点都在赋值之后，无未定义变量问题。`traceback_text` 用 `_tb.sys.exc_info()[0]` 判断当前是否处于异常处理上下文：在异常分支（FloatingPointError/RuntimeError）中返回 `_tb.format_exc()`，在主动检查分支（forward_output）中回退为 `f"kind={kind}"`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: PASS（新增 5 个；TestRunDiagnostic 全部走 CPU toy pipeline）

- [ ] **Step 5: Commit**

```bash
git add test/tool_diagnose_rep2_nan.py test/test_rep2_nan_diagnostic.py
git commit -m "test: rep2 诊断主循环 run_diagnostic（有限/NaN/OOM 路径，TDD Task 6）"
```

---

### Task 7: CLI `parse_args` + `build_pipeline` + `main` 装配

**Files:**
- Modify: `test/tool_diagnose_rep2_nan.py`
- Modify: `test/test_rep2_nan_diagnostic.py`

**Interfaces:**
- Consumes: Task 2 `ensure_output_dir`、Task 3 `atan2_zero_probe`/`GeometryProbe`、Task 4 写入器、Task 5 `restore_checkpoint`、Task 6 `Pipeline`/`run_diagnostic`；`engine.core.YAMLConfig`、`engine.optim.lr_scheduler.FlatCosineLRScheduler`。
- Produces:
  - `parse_args(argv: list[str] | None = None) -> argparse.Namespace` — CLI 契约见 spec §3：必填 `--config/--checkpoint/--output-dir`；可选 `--device`（默认 `cuda:0`）、`--max-epochs`（默认 10）、`--max-steps-per-epoch`（默认 None）、`--start-epoch`（默认 None）、`--seed`（默认 None）、`--detect-anomaly/--no-detect-anomaly`（默认启用）、`--save-every-steps`（默认 50）、`--overwrite`（默认 False）。
  - `build_pipeline(cfg_path: str, checkpoint_path: str, device: str, start_epoch: int | None, seed: int | None) -> tuple[Pipeline, dict]` — 构建 YAMLConfig → model/criterion/optimizer/train_dataloader（**不调用 `solver.fit()`**）；`cfg.lrsheduler` 非 None 时构建 `FlatCosineLRScheduler(optimizer, cfg.lr_gamma, iter_per_epoch=len(loader), total_epochs=cfg.epoches, warmup_iter=cfg.warmup_iter, flat_epochs=cfg.flat_epoch, no_aug_epochs=cfg.no_aug_epoch)`；dataset 若 `cache_images == "disk"` 则 `precache_images(num_workers=8)`（镜像 `fit()`）；`restore_checkpoint` 恢复；返回 `(pipeline, manifest_meta)`。
  - `main(argv: list[str] | None = None) -> int` — `ensure_output_dir` → `build_pipeline` → `atan2_zero_probe` 写入 manifest → 写 `run_manifest.json`/`command.txt` → `run_diagnostic` → 退出码。构建/恢复失败 → 3；`run_diagnostic` 返回 0/2/4 原样透传。

- [ ] **Step 1: 编写失败测试（parse_args + build_pipeline 集成）**

```python
class TestParseArgs:

    def test_required_args_and_defaults(self):
        from tool_diagnose_rep2_nan import parse_args

        args = parse_args(
            [
                "--config", "cfg.yml",
                "--checkpoint", "ckpt.pth",
                "--output-dir", "out",
            ]
        )
        assert args.config == "cfg.yml"
        assert args.checkpoint == "ckpt.pth"
        assert args.output_dir == "out"
        assert args.device == "cuda:0"
        assert args.max_epochs == 10
        assert args.max_steps_per_epoch is None
        assert args.start_epoch is None
        assert args.seed is None
        assert args.detect_anomaly is True
        assert args.save_every_steps == 50
        assert args.overwrite is False

    def test_missing_required_raises(self):
        import pytest
        from tool_diagnose_rep2_nan import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--config", "cfg.yml"])

    def test_no_detect_anomaly_flag(self):
        from tool_diagnose_rep2_nan import parse_args

        args = parse_args(
            [
                "--config", "cfg.yml",
                "--checkpoint", "ckpt.pth",
                "--output-dir", "out",
                "--no-detect-anomaly",
            ]
        )
        assert args.detect_anomaly is False

    def test_optional_override(self):
        from tool_diagnose_rep2_nan import parse_args

        args = parse_args(
            [
                "--config", "cfg.yml",
                "--checkpoint", "ckpt.pth",
                "--output-dir", "out",
                "--device", "cpu",
                "--max-epochs", "3",
                "--max-steps-per-epoch", "100",
                "--start-epoch", "5",
                "--seed", "42",
                "--save-every-steps", "10",
                "--overwrite",
            ]
        )
        assert args.device == "cpu"
        assert args.max_epochs == 3
        assert args.max_steps_per_epoch == 100
        assert args.start_epoch == 5
        assert args.seed == 42
        assert args.save_every_steps == 10
        assert args.overwrite is True


class TestBuildPipeline:

    @pytest.mark.skipif(
        not os.path.exists("/data/lyj_dir"),
        reason="remote data path unavailable",
    )
    def test_builds_pipeline_and_manifest_meta(self):
        from tool_diagnose_rep2_nan import build_pipeline

        pipeline, meta = build_pipeline(
            cfg_path="configs/custom_obb/dlzdt/ablation/abl_rep2.yml",
            checkpoint_path="",
            device="cpu",
            start_epoch=None,
            seed=None,
        )
        assert pipeline is not None
        assert "recovery" in meta
        assert meta["config_path"].endswith("abl_rep2.yml")
        # checkpoint_path 为空 → 视为 weights_only 空状态（不抛错）
        assert meta["recovery"]["fidelity"] in ("full", "partial", "weights_only")
```

注意：`test_builds_pipeline_and_manifest_meta` 依赖真实 DOTA 数据路径（`/data/lyj_dir/...`），已用 `@pytest.mark.skipif(not os.path.exists("/data/lyj_dir"), ...)` 标记，本地跳过、远程服务器上运行。因此测试文件顶部模块级必须 `import pytest`（Task 2/5/7 的测试方法内 `import pytest` 也可保留，但 skipif 装饰器求值发生在模块导入时，必须模块级可用）。其余 parse_args 测试本地全跑。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: FAIL — ImportError（`parse_args`/`build_pipeline` 未定义；本地跳过的 skipif 测试不计数失败）

- [ ] **Step 3: 最小实现**

追加到 `test/tool_diagnose_rep2_nan.py`（文件顶部加 `import argparse`、`import hashlib`、`import platform`、`import subprocess`、`import socket`、`import datetime`）：

```python
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
    write_run_manifest(output_dir / "run_manifest.json", meta)
    (output_dir / "command.txt").write_text("python " + " ".join(sys.argv) + "\n")

    # run_diagnostic 需要的运行参数：use_amp 从配置合并值注入（parse_args 无此参数）；
    # seed 已在 build_pipeline 内解析（--seed → 配置 seed → 0，spec §3）
    args.use_amp = meta["use_amp"]
    args.effective_seed = meta["effective_seed"]

    pipeline.probe = probe
    code = run_diagnostic(pipeline, args)
    probe.uninstall()
    return code
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/test_rep2_nan_diagnostic.py -v`
Expected: PASS（parse_args 4 个本地跑；build_pipeline 集成测试本地 skipif、远程跑）

- [ ] **Step 5: Commit**

```bash
git add test/tool_diagnose_rep2_nan.py test/test_rep2_nan_diagnostic.py
git commit -m "test: 诊断 runner CLI/build_pipeline/main 装配（TDD Task 7）"
```

---

## 自检记录（写完后执行）

1. **Spec 覆盖对照**：§3 CLI ✓（Task 7）、§4 checkpoint 分类/恢复 ✓（Task 2/5）、§5 训练循环 ✓（Task 6）、§6.1 自检 probe ✓（Task 3 + main）、§6.2 几何统计 ✓（Task 3）、§7 产物目录 ✓（Task 4 + main）、§8 退出码 ✓（Task 6/7）、§9 七类测试 ✓（Task 1–7）。
2. **占位符扫描**：全文无 TBD/TODO/"待实现细节"；`main` 在 Task 7 前为 `NotImplementedError` 占位属于 TDD 预期（Task 7 Step 3 替换）。
3. **类型一致性**：`Pipeline` 字段、`run_diagnostic` args 字段、`save_failure` 关键字参数、`compute_edge_stats` 返回键在 Task 3→4→6→7 间一致。
4. **原子提交边界**：每个 Task 一个 commit；只提交 `test/tool_diagnose_rep2_nan.py` 与 `test/test_rep2_nan_diagnostic.py`；不含 4 个用户既有改动文件。
5. **验证命令**：每个 Task 后运行 `python -m pytest test/test_rep2_nan_diagnostic.py -v`；Task 7 完成后运行 `git diff --check` 确认无空白错误。
