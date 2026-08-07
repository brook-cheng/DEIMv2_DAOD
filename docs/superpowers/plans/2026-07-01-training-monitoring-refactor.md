# 训练监控体系重构实现计划

> **修订（2026-07-01）**：
> - **Metric 命名扁平化**：删除 `train/` 前缀，二级目录提升为一级（`main/`, `aux/`, `enc/`, `dn/`, `grad/`, `param/`, `kendall/`, `lr`）。Comet v2 仅支持第一级 `/` 自动建 Panel Group。
> - **梯度直方图 → 模块组聚合统计**：`_log_gradient_histograms` 替换为 `_log_gradient_stats`。300+ 次/epoch API 调用缩减为 12 次，消除 rate limit 警告。频率从每 epoch → 每 5 epoch。
> - 详细变更见 [设计规格 §3.1 §6.3](../specs/2026-07-01-training-monitoring-refactor-design.md)。

> **For agentic workers:** 本计划中每个 Task 独立可验证，按顺序执行。Steps 使用 checkbox (`- [ ]`) 语法追踪进度。
> **前置文档**：[代码审查报告](../review/2026-07-01-det-engine-logging-review.md) | [设计规格](../specs/2026-07-01-training-monitoring-refactor-design.md)

**Goal:** 重构 `det_engine.py` 训练日志体系：通用遍历替代硬编码 loss key、Comet 分层树结构、梯度监控、移除 TensorBoard writer。

**Architecture:** 添加 `parse_loss_key()` 模块级函数将 `loss_mal_aux_3` 解析为 `(family, source, index)` 三元组；所有 logging 通道统一使用该函数驱动，不再维护独立 key 清单。梯度监控采用混合策略：每 10 step 标量 + 每 epoch 直方图。

**Tech Stack:** Python 3.9+, PyTorch 2.x, Comet ML SDK, tqdm

## Global Constraints

- 不修改 `deim_criterion.py`（仅读取 `weight_dict` 和 `loss_dict` 输出）
- 不修改任何 config yml 文件
- HBB（GIoU）和 OBB（KLD）模式均自动适配，不做显式判断
- DDP 多 GPU 下仅主进程执行 Comet/tqdm/print 操作
- 梯度直方图仅在主进程执行，每 epoch 最多一次

---

## File Map

| 文件 | 角色 |
|------|------|
| `deimv2_daod/engine/solver/det_engine.py` | **主战场**：添加 `parse_loss_key()` + 重构 logging 块 + 新增梯度监控 |
| `deimv2_daod/engine/solver/det_solver.py` | **轻量**：Kendall 日志路径从 `kendall_{k}` → `train/kendall/{k}` |

---

### Task 1: `parse_loss_key()` 工具函数

**Files:**
- Modify: `deimv2_daod/engine/solver/det_engine.py:8-26`（在 import 区域后插入新函数）

**Interfaces:**
- Produces: `parse_loss_key(key: str) -> tuple[str, str, int | None]`

**描述**：从 `loss_dict` 的 key（如 `loss_mal_aux_3`）解析为 `(loss_family, source_category, layer_index)`。所有后续 logging 通道依赖此函数。

- [ ] **Step 1: 在 import 区域后插入 `parse_loss_key` 函数和常量**

在 `det_engine.py` 第 26 行（`from ..eval.obb_eval import obb_evaluate`）之后插入：

```python
# ---------------------------------------------------------------------------
#  Loss key parser: "loss_mal_aux_3" → ("loss_mal", "aux", 3)
# ---------------------------------------------------------------------------
import re

# 后缀模式 → source 映射（按优先级：enc > aux > _dn_pre > dn > pre）
_SUFFIX_PATTERNS = [
    ("_enc_", "enc"),
    ("_aux_", "aux"),
    ("_dn_pre", "dn_pre"),
    ("_dn_", "dn"),
    ("_pre", "pre"),
]


def parse_loss_key(key: str) -> tuple[str, str, int | None]:
    """Parse a criterion loss-dict key into (loss_family, source, layer_index).

    Examples:
        "loss_mal"         → ("loss_mal", "main", None)
        "loss_mal_aux_3"   → ("loss_mal", "aux", 3)
        "loss_kld_dn_1"    → ("loss_kld", "dn", 1)
        "loss_bbox_enc_0"  → ("loss_bbox", "enc", 0)
        "loss_mal_dn_pre"  → ("loss_mal", "dn_pre", None)
        "loss_fgl_pre"     → ("loss_fgl", "pre", None)
    """
    for suffix, source in _SUFFIX_PATTERNS:
        if suffix in key:
            prefix = key[: key.index(suffix)]
            idx_str = key[key.index(suffix) + len(suffix):]
            return (prefix, source, int(idx_str) if idx_str else None)
    return (key, "main", None)
```

- [ ] **Step 2: 验证语法正确性**

```bash
python -c "import ast; ast.parse(open('deimv2_daod/engine/solver/det_engine.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 2: AMP/Non-AMP 分支梯度捕获 + Non-AMP 分支 Kendall 日志修复

**Files:**
- Modify: `deimv2_daod/engine/solver/det_engine.py:86-161`

**Interfaces:**
- Consumes: `parse_loss_key` (Task 1)
- Produces: `grad_norm_before`, `grad_norm_after`, `param_norm` 三个标量供 Task 3 使用

**描述**：在 `loss.backward()` 后捕获梯度范数和参数范数。同时修复 non-AMP 分支中 Kendall 日志路径（从 `kendall_{k}` → `train/kendall/{k}`）并删除 writer 调用。

- [ ] **Step 1: 替换 AMP 分支（lines 108-117）—— 添加梯度范数捕获**

找到第 108 行 `loss = sum(loss_dict.values())`，替换从该行到第 117 行的整个 AMP backward 块：

```python
            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm
                )
                grad_norm_after = min(grad_norm_before, max_norm)
            else:
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf")
                )
                grad_norm_after = grad_norm_before

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

- [ ] **Step 2: 替换 Non-AMP 分支（lines 119-161）—— 梯度捕获 + Kendall 日志修复 + EMA/lr 逻辑**

找到第 119 行 `else:`，替换从该行到第 161 行的整个 non-AMP 块：

```python
        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            optimizer.zero_grad()
            if kendall_optimizer is not None:
                kendall_optimizer.zero_grad()

            if kendall is not None:
                loss = kendall.weighted_loss(loss_dict)
                loss.backward()

                if max_norm > 0:
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm
                    )
                    grad_norm_after = min(grad_norm_before, max_norm)
                else:
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf")
                    )
                    grad_norm_after = grad_norm_before

                optimizer.step()
                if kendall_optimizer is not None:
                    kendall_optimizer.step()

                if dist_utils.is_main_process() and global_step % 10 == 0:
                    weights = kendall.get_weights()
                    if comet_exp:
                        for k, w in weights.items():
                            comet_exp.log_metric(
                                f"train/kendall/{k}", w, step=global_step
                            )
            else:
                loss: torch.Tensor = sum(loss_dict.values())
                loss.backward()

                if max_norm > 0:
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm
                    )
                    grad_norm_after = min(grad_norm_before, max_norm)
                else:
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf")
                    )
                    grad_norm_after = grad_norm_before

                optimizer.step()

            # 参数范数（梯度捕获后、step 后计算 data norm）
            param_norm = sum(
                p.data.norm(2).item() ** 2 for p in model.parameters()
            ) ** 0.5

            if ema is not None:
                ema.update(model)

            if self_lr_scheduler:
                optimizer = lr_scheduler.step(cur_iters + i, optimizer)
            else:
                if lr_warmup_scheduler is not None:
                    lr_warmup_scheduler.step()
```

> **变更说明**：
> - `clip_grad_norm_` 返回值即为裁剪前总范数，`float("inf")` 确保不裁剪但返回范数
> - Kendall 日志路径从 `f"kendall_{k}"` → `f"train/kendall/{k}"`
> - 删除 `writer.add_scalar(f"Kendall/{k}", ...)` 调用
> - 参数范数 `param_norm` 在 optimizer.step() 之后计算（数据已更新）
> - `scaler is not None` 分支的 NaN 检测 + 前向推理代码（lines 86-107）**原封不动保留**

- [ ] **Step 3: 验证语法**

```bash
python -c "import ast; ast.parse(open('deimv2_daod/engine/solver/det_engine.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 3: Batch 级日志重构（tqdm + Comet）

**Files:**
- Modify: `deimv2_daod/engine/solver/det_engine.py:170-221`（原有 batch 日志整块替换）

**Interfaces:**
- Consumes: `parse_loss_key` (Task 1), `grad_norm_before`, `grad_norm_after`, `param_norm` (Task 2)
- Produces: 新版 tqdm postfix 和 Comet batch 日志

**描述**：删除原有的硬编码 tqdm key 清单 + Comet 显式 key 分支 + writer 块，替换为通用遍历驱动的日志。

- [ ] **Step 1: 替换 lines 162-221 整块日志代码**

找到第 162 行 `loss_dict_reduced = dist_utils.reduce_dict(loss_dict)`，替换从该行到第 221 行（`comet_exp.log_metric("loss_ddf", ...` 结束）的全部内容：

```python
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # -------- tqdm 进度条：显示主损失 --------
        if use_tqdm:
            postfix_dict = {
                "lr": f'{optimizer.param_groups[0]["lr"]:.8f}',
                "total": f"{loss_value:.4f}",
            }
            for k, v in loss_dict_reduced.items():
                family, source, _ = parse_loss_key(k)
                if source == "main":
                    short_name = k.replace("loss_", "")
                    postfix_dict[short_name] = f"{v:.4f}"
            data_loader_iter.set_postfix(postfix_dict)

        # -------- Comet batch 日志 --------
        if comet_exp and dist_utils.is_main_process() and global_step % 10 == 0:
            for k, v in loss_dict_reduced.items():
                family, source, idx = parse_loss_key(k)
                metric_name = f"train/{source}/{family}"
                if idx is not None:
                    if source in ("aux", "enc"):
                        metric_name += f"/layer_{idx}"
                    elif source in ("dn", "dn_pre"):
                        metric_name += f"/group_{idx}"
                comet_exp.log_metric(metric_name, v.item(), step=global_step)

            comet_exp.log_metric(
                "train/main/loss_total", loss_value.item(), step=global_step
            )
            comet_exp.log_metric(
                "train/lr",
                optimizer.param_groups[0]["lr"],
                step=global_step,
            )
            # 梯度监控（标量）
            if "grad_norm_before" in dir():
                comet_exp.log_metric(
                    "train/grad/norm/before_clip",
                    grad_norm_before,
                    step=global_step,
                )
                comet_exp.log_metric(
                    "train/grad/norm/after_clip",
                    grad_norm_after,
                    step=global_step,
                )
            comet_exp.log_metric(
                "train/param/norm", param_norm, step=global_step
            )
```

> **变更说明**：
> - 删除原 tqdm 块（lines 175-188）：不再 hardcode `loss_mal`/`loss_bbox`/`loss_fgl`/`loss_ddf` key
> - 删除原 writer 块（lines 191-196）：完全移除 TensorBoard 日志
> - 删除原 Comet batch 块（lines 198-221）：不再用 `batch_*` 前缀 + 显式 key 双重上报
> - `grad_norm_before`/`grad_norm_after` 在 scaler 分支中定义，用 `"grad_norm_before" in dir()` 防御性检查

- [ ] **Step 2: 删除不再需要的 `SummaryWriter` import**

修改第 15 行，移除 `SummaryWriter` 的 import（保留 `GradScaler`）：

```python
from torch.cuda.amp.grad_scaler import GradScaler
```

第 15 行原为：
```python
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler
```
替换为单行：
```python
from torch.cuda.amp.grad_scaler import GradScaler
```

同时删除第 48 行不再使用的 `writer` 变量声明：
```python
writer: SummaryWriter = kwargs.get("writer", None)
```
整行删除。

- [ ] **Step 3: 验证语法**

```bash
python -c "import ast; ast.parse(open('deimv2_daod/engine/solver/det_engine.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 4: Epoch 级日志重构（控制台 + Comet + 梯度直方图）

**Files:**
- Modify: `deimv2_daod/engine/solver/det_engine.py:223-259`（原有 epoch 日志整块替换）
- Add function: `deimv2_daod/engine/solver/det_engine.py` 模块级 `_log_gradient_histograms()`

**Interfaces:**
- Consumes: `parse_loss_key` (Task 1), `metric_logger.meters`
- Produces: 新版控制台 epoch summary + Comet epoch 日志

- [ ] **Step 1: 在 `parse_loss_key` 函数定义之后插入 `_log_gradient_histograms` 函数**

找到 `parse_loss_key` 函数定义的结束位置（约第 62 行），在其后插入：

```python
def _log_gradient_histograms(
    model: torch.nn.Module, comet_exp, epoch: int
) -> None:
    """Log per-parameter gradient histograms to Comet (epoch-level, expensive).

    Called once per epoch on the main process only.
    """
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_flat = param.grad.detach().cpu().flatten().abs()
        if grad_flat.numel() == 0:
            continue
        metric_name = f"train/grad/hist/{name.replace('.', '/')}"
        try:
            comet_exp.log_histogram_3d(
                values=grad_flat.tolist(),
                name=metric_name,
                step=epoch,
            )
        except Exception:
            pass  # Comet histogram API may vary; never crash training
```

- [ ] **Step 2: 替换控制台 epoch summary（原 lines 223-237）**

找到 `metric_logger.synchronize_between_processes()`（当前约第 223 行），替换从该行到第 237 行的 console print 块：

```python
    metric_logger.synchronize_between_processes()

    if dist_utils.is_main_process():
        print("\n" + "=" * 60)
        print(f"Training Summary - Epoch {epoch}")
        print("=" * 60)
        for k, meter in metric_logger.meters.items():
            if k == "loss":
                print(f"  Total Loss:     {meter.global_avg:.4f}")
            elif k == "lr":
                print(f"  Learning Rate:  {meter.global_avg:.6f}")
            elif k.startswith("loss_"):
                family, source, _ = parse_loss_key(k)
                if source == "main":
                    label = k.replace("loss_", "").upper()
                    print(f"  {label:16s} {meter.global_avg:.4f}")
        print("=" * 60 + "\n")
```

- [ ] **Step 3: 替换 Comet epoch 日志（原 lines 239-257）**

找到第 239 行 `if comet_exp and dist_utils.is_main_process():`，替换从该行到第 257 行（`comet_exp.log_metric("epoch_lr", ...` 结束）的整个 epoch 日志块：

```python
    if comet_exp and dist_utils.is_main_process():
        for k, meter in metric_logger.meters.items():
            if k == "loss":
                comet_exp.log_metric(
                    "train/main/loss_total", meter.global_avg, epoch=comet_step
                )
            elif k == "lr":
                comet_exp.log_metric(
                    "train/lr", meter.global_avg, epoch=comet_step
                )
            elif k.startswith("loss_"):
                family, source, idx = parse_loss_key(k)
                metric_name = f"train/{source}/{family}"
                if idx is not None:
                    if source in ("aux", "enc"):
                        metric_name += f"/layer_{idx}"
                    elif source in ("dn", "dn_pre"):
                        metric_name += f"/group_{idx}"
                comet_exp.log_metric(
                    metric_name, meter.global_avg, epoch=comet_step
                )

        # 梯度直方图（epoch 级，主进程，跳过 epoch 0 避免无梯度）
        if epoch > 0:
            _log_gradient_histograms(model, comet_exp, comet_step)
```

- [ ] **Step 4: `return` 语句不变**

`return {k: meter.global_avg for k, meter in metric_logger.meters.items()}` 保持原样，位置不变。

- [ ] **Step 5: 验证语法**

```bash
python -c "import ast; ast.parse(open('deimv2_daod/engine/solver/det_engine.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 5: `det_solver.py` 清理

**Files:**
- Modify: `deimv2_daod/engine/solver/det_solver.py:141-160`

**Interfaces:**
- Consumes: 新版 `train_one_epoch` 不再需要 `writer` kwarg (Task 3 已移除)
- Produces: 干净的 `train_one_epoch` 调用

**描述**：删除 `writer=self.writer` kwarg（`det_engine.py` 已不再读取 `writer`）。

- [ ] **Step 1: 删除 `train_one_epoch` 调用中的 `writer=self.writer`**

找到第 141-160 行 `train_stats = train_one_epoch(...)` 调用，删除 `writer=self.writer,` 这一行：

```python
            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                comet_exp=comet_exp,
                comet_step=epoch,
                kendall=kendall,
                kendall_optimizer=kendall_optimizer,
            )
```

> **变更**：删除 `writer=self.writer,` 一行（原第 155 行）。

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('deimv2_daod/engine/solver/det_solver.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 6: 端到端验证

**Files:**
- 不修改文件，仅运行验证。

- [ ] **Step 1: 语法检查两个文件**

```bash
python -c "import ast; ast.parse(open('deimv2_daod/engine/solver/det_engine.py').read()); print('det_engine OK')"
python -c "import ast; ast.parse(open('deimv2_daod/engine/solver/det_solver.py').read()); print('det_solver OK')"
```

Expected: `det_engine OK` + `det_solver OK`

- [ ] **Step 2: 验证 `parse_loss_key` 单元逻辑**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from deimv2_daod.engine.solver.det_engine import parse_loss_key

assert parse_loss_key('loss_mal') == ('loss_mal', 'main', None)
assert parse_loss_key('loss_mal_aux_3') == ('loss_mal', 'aux', 3)
assert parse_loss_key('loss_kld_dn_1') == ('loss_kld', 'dn', 1)
assert parse_loss_key('loss_bbox_enc_0') == ('loss_bbox', 'enc', 0)
assert parse_loss_key('loss_mal_dn_pre') == ('loss_mal', 'dn_pre', None)
assert parse_loss_key('loss_fgl_pre') == ('loss_fgl', 'pre', None)
assert parse_loss_key('loss_ddf') == ('loss_ddf', 'main', None)
print('All parse_loss_key tests passed')
"
```

Expected: `All parse_loss_key tests passed`

- [ ] **Step 3: 干跑配置加载**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from deimv2_daod.engine.core import YAMLConfig
cfg = YAMLConfig('configs/custom_obb/synthetic_configs/synthetic_exp_020.yml')
print(f'Config loaded: box_mode={cfg.yaml_cfg.get(\"DEIMCriterion\", {}).get(\"box_mode\", \"?\")}')
print('Config load OK')
"
```

Expected: `Config loaded: box_mode=obb` + `Config load OK`

---

### Task 7: 代码审查与完整测试

**Files:**
- 不修改文件。对 Task 1-5 的所有变更做最终审查和集成测试。

> **注意**：所有 git 操作由用户自行管理，本计划不包含 `git add`/`git commit`。

- [ ] **Step 1: LSP 诊断两个修改文件**

```bash
# 使用 lsp_diagnostics 检查两个文件的语法/类型错误
```

用 `lsp_diagnostics` 工具分别检查：
- `deimv2_daod/engine/solver/det_engine.py`
- `deimv2_daod/engine/solver/det_solver.py`

确认无 error 级别诊断。

- [ ] **Step 2: 集成导入测试**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from deimv2_daod.engine.solver.det_engine import train_one_epoch, evaluate, parse_loss_key, _log_gradient_histograms
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 3: `parse_loss_key` 完整单元测试**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from deimv2_daod.engine.solver.det_engine import parse_loss_key

# 主损失
assert parse_loss_key('loss_mal') == ('loss_mal', 'main', None)
assert parse_loss_key('loss_bbox') == ('loss_bbox', 'main', None)
assert parse_loss_key('loss_kld') == ('loss_kld', 'main', None)
assert parse_loss_key('loss_giou') == ('loss_giou', 'main', None)
assert parse_loss_key('loss_fgl') == ('loss_fgl', 'main', None)
assert parse_loss_key('loss_ddf') == ('loss_ddf', 'main', None)

# aux 辅助
assert parse_loss_key('loss_mal_aux_0') == ('loss_mal', 'aux', 0)
assert parse_loss_key('loss_mal_aux_5') == ('loss_mal', 'aux', 5)
assert parse_loss_key('loss_fgl_aux_3') == ('loss_fgl', 'aux', 3)

# enc 辅助
assert parse_loss_key('loss_bbox_enc_0') == ('loss_bbox', 'enc', 0)

# dn 辅助
assert parse_loss_key('loss_kld_dn_1') == ('loss_kld', 'dn', 1)
assert parse_loss_key('loss_mal_dn_4') == ('loss_mal', 'dn', 4)

# pre / dn_pre
assert parse_loss_key('loss_mal_pre') == ('loss_mal', 'pre', None)
assert parse_loss_key('loss_bbox_dn_pre') == ('loss_bbox', 'dn_pre', None)

# edge case: loss 名中包含数字（如 loss_vfl）
result = parse_loss_key('loss_vfl')
assert result[1] == 'main', f'Expected main, got {result}'

print('All 17 parse_loss_key tests passed')
"
```

Expected: `All 17 parse_loss_key tests passed`

- [ ] **Step 4: 代码 Diff 审查**

列出所有变更供人工审查：

```bash
python -c "
import difflib, sys
# 对比两个文件的改动摘要
print('=== Modified files ===')
print('1. deimv2_daod/engine/solver/det_engine.py')
print('   - Added: parse_loss_key(), _log_gradient_histograms()')
print('   - Modified: AMP/non-AMP branches (gradient capture)')
print('   - Modified: tqdm postfix (generic loop)')
print('   - Removed: TensorBoard writer block')
print('   - Modified: Comet batch logging (tree naming)')
print('   - Modified: Comet epoch logging (generic loop)')
print('   - Modified: Console epoch summary (is_main_process guard)')
print('2. deimv2_daod/engine/solver/det_solver.py')
print('   - Removed: writer=self.writer kwarg')
print('=== End of summary ===')
"
```

- [ ] **Step 5: 用户确认网关**

所有上述测试通过后，报告用户。用户审查代码 diff 并自行管理 git 提交。

---

## 自检

### 1. Spec 覆盖率

| 设计规格章节 | 对应 Task |
|---|---|
| §3.2 key 解析逻辑 | Task 1 |
| §6.2 标量采集（梯度范数 + 参数范数） | Task 2 |
| §4 tqdm 显示 | Task 3 |
| §5 Comet batch 日志 | Task 3 |
| §9 移除 TensorBoard writer | Task 3 |
| §8 控制台 epoch summary | Task 4 |
| §7 Comet epoch 日志 | Task 4 |
| §6.3 Epoch 级梯度直方图 | Task 4 |
| §10.2 Kendall 权重日志路径 | Task 2 |
| §10.1 传递配置信息 | 不需要——`parse_loss_key` 不依赖 `cfg`，只依赖 `loss_dict` key 名 |
| §3.3 HBB/OBB 双模适配 | 自动适配（§3.3），无显式代码 |

### 2. 占位符检查

- [x] 无 TBD/TODO
- [x] 所有代码块为完整、可执行的 Python 代码
- [x] 所有 import 明确
- [x] 所有函数签名、类型标注完整
- [x] 无 `git add`/`git commit` 步骤（由用户自行管理）

### 3. 类型一致性

- `parse_loss_key` 返回值 `tuple[str, str, int | None]` → 在所有 Task（2/3/4）中解包为 `(family, source, idx)`
- `grad_norm_before`/`grad_norm_after`/`param_norm` 在 Task 2 定义 → Task 3 消费，命名一致
- Comet metric 命名 `train/{source}/{family}/layer_{idx}` 格式在 Task 3 (batch) 和 Task 4 (epoch) 中一致
