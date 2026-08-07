# 训练监控体系重构设计

> **创建日期**：2026-07-01
> **前置审查**：[2026-07-01-det-engine-logging-review.md](../review/2026-07-01-det-engine-logging-review.md)
> **状态**：设计阶段，待评审

---

## 1. 背景

当前 `det_engine.py` 中训练日志/监控代码存在以下结构性问题（详见前置审查报告）：

| 问题 | 严重度 |
|------|--------|
| Epoch 级 Comet 日志缺失 `loss_fgl`/`loss_ddf` | 重要 |
| Comet batch 日志双重上报（通用 loop + 显式 key 重复） | 重要 |
| Hardcoded loss key 分散在 4 处代码，新增 loss 需改 4 处 | 主要 |
| TensorBoard writer 冗余（且不完整） | 次要 |
| 无梯度监控指标 | 缺失 |

同时用户提出需求：
1. Aux 系列 loss 按损失类型（而非 decoder 层序）分组显示
2. 主损失和 eval 结果为关键信息，置顶突出显示
3. 增加梯度范数、各 loss 梯度贡献、epoch 级梯度直方图等监控指标

---

## 2. 方案概述

### 2.1 核心原则

1. **通用遍历 + 命名驱动**：所有 Comet/tqdm 日志从 `loss_dict` 和 `metric_logger.meters` 通用遍历生成，不再 hardcode 具体 loss key。通过 metric 名中的 `/` 层级自动组织 Comet 树结构。
2. **命名即结构**：metric 命名采用 `train/{group}/{loss_type}/{layer}` 格式，利用 Comet 的 `/` 分隔符自动建树，无需手动创建 Panel。
3. **单一数据源**：所有 logging 通道从 `loss_dict_reduced`（batch 级）和 `metric_logger`（epoch 级）读取，不再维护独立 key 清单。
4. **增量监控**：梯度监控采用混合策略（每 10 step 记录标量 + 每 epoch 记录全量直方图）。

### 2.2 变更范围

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `deimv2_daod/engine/solver/det_engine.py` | **主要修改** | 重构整个日志块（~90 行 → ~60 行），新增梯度监控 |
| `deimv2_daod/engine/solver/det_solver.py` | 轻量修改 | 传递 `cfg` 用于 loss key 解析配置 |
| `deimv2_daod/engine/deim/deim_criterion.py` | 不修改 | 仅读取 `weight_dict` 用于 key 名到 loss 类型的映射 |
| `configs/custom_obb/synthetic_configs/synthetic_exp_020.yml` | 不修改 | Config 中 losses 顺序保持 |

---

## 3. Metric 命名规范

### 3.1 树结构

> **设计决策（2026-07-01 修订）**：Comet v2 不支持 `/` 分隔符自动建子 Panel — 仅第一级 `/` 会创建 Panel Group。因此将原 `train/` 前缀剥离，每个 source 提升为一级目录，利用 Comet 自动分组。

```
main/                    ← 主损失（第一组，字母序 m < v，天然在前）
 ├── loss_total
 ├── loss_mal
 ├── loss_bbox
 ├── loss_kld          # OBB 模式
 ├── loss_giou          # HBB 模式
 ├── loss_fgl
 └── loss_ddf

aux/                     ← decoder 辅助损失（第二组）
 ├── loss_mal/layer_0, layer_1, ..., layer_5
 ├── loss_bbox/layer_0, ..., layer_5
 ├── loss_kld/layer_0, ..., layer_5
 ├── loss_fgl/layer_0, ..., layer_5
 └── loss_ddf/layer_0, ..., layer_5

enc/                     ← encoder 辅助损失
 ├── loss_mal/layer_0, layer_1, ...
 └── loss_bbox/layer_0, ...

pre/                     ← pre_outputs（单层）
 ├── loss_mal
 ├── loss_bbox
 └── ...

dn/                      ← CDN 去噪组损失
 ├── loss_mal/group_0, group_1, ...
 ├── loss_bbox/group_0, ...
 └── ...

dn_pre/                  ← CDN pre 损失
 └── ...

kendall/                 ← Kendall 权重
 ├── loss_mal, loss_bbox, loss_kld, loss_fgl, loss_ddf

grad/                    ← 梯度监控（新增，见 §6.3 修订）
 ├── norm/before_clip
 ├── norm/after_clip
 ├── backbone/norm        ← 各模块组梯度范数
 ├── backbone/max
 ├── encoder/norm
 ├── encoder/max
 ├── decoder/norm
 ├── decoder/max
 ├── head/norm
 └── head/max

param/                   ← 参数范数（新增）
 └── norm

lr                       ← 学习率（单标量）

val/                     ← evaluate 接口结果（始终在后，v > m）
 ├── mAP
 ├── AP50
 ├── AP75
 ├── precision
 ├── recall
 └── f1
```
 ├── AP50
 ├── AP75
 ├── precision
 ├── recall
 └── f1
```

**排序原理**：Comet 按 metric 创建顺序 + 字母序排列。`train/` 在前（字母 `t` < `v`），`main` 在 `aux` 前（字母 `m` < `a`），`val/` 独立分组自然靠后。

### 3.2 key 解析逻辑

从 `criterion()` 返回的 `loss_dict` key（如 `loss_mal_aux_3`）自动解析为 `(family, source, index)` 三元组：

```python
# 后缀模式 → source 映射（按优先级排序）
_SUFFIX_PATTERNS = [
    ("_enc_",         "enc"),
    ("_aux_",         "aux"),
    ("_dn_pre",       "dn_pre"),
    ("_dn_",          "dn"),
    ("_pre",          "pre"),
]

def parse_loss_key(key: str) -> tuple[str, str, int | None]:
    """解析 loss key 为 (loss_family, source, layer_index).
    
    Examples:
        "loss_mal"        → ("loss_mal", "main", None)
        "loss_mal_aux_3"  → ("loss_mal", "aux", 3)
        "loss_kld_dn_1"   → ("loss_kld", "dn", 1)
        "loss_bbox_enc_0" → ("loss_bbox", "enc", 0)
    """
    for suffix, source in _SUFFIX_PATTERNS:
        if suffix in key:
            prefix = key[: key.index(suffix)]
            idx_str = key[key.index(suffix) + len(suffix):]
            return (prefix, source, int(idx_str))
    return (key, "main", None)
```

### 3.3 HBB/OBB 双模适配

不做显式判断。metric 名直接使用 `loss_dict` 中的原始 key family：
- OBB 模式：criterion 生成 `loss_kld`，metric 名为 `train/main/loss_kld`
- HBB 模式：criterion 生成 `loss_giou`，metric 名为 `train/main/loss_giou`

两种模式的 metric 树结构完全一致，仅叶子节点名不同，Comet 自动区分。

---

## 4. tqdm 进度条显示

### 4.1 设计

移除硬编码 key，改为从 `loss_dict_reduced` 动态构建。显示策略：只展示主损失 key（无后缀）的前 N 个，固定顺序确保可读性。

```python
# 仅显示主损失（无后缀的 key），避免进度条过长
_TQDM_DISPLAY_KEYS = None   # None = 显示所有主损失（无 _aux_、_enc_ 等后缀）

if use_tqdm:
    postfix_dict = {
        "lr": f'{optimizer.param_groups[0]["lr"]:.8f}',
        "total": f"{loss_value:.4f}",
    }
    for k, v in loss_dict_reduced.items():
        family, source, _ = parse_loss_key(k)
        if source == "main":
            short_name = k.replace("loss_", "")  # "loss_mal" → "mal"
            postfix_dict[short_name] = f"{v:.4f}"
    data_loader_iter.set_postfix(postfix_dict)
```

**效果**（OBB 模式）：
```
Epoch 5: 100%|████| 42/42 [02:15<00:00, 3.21it/s, lr=0.00010000, total=5.4231, mal=1.8523, bbox=0.0341, kld=0.0189, fgl=3.1142, ddf=0.4036]
```

---

## 5. Comet Batch 日志

### 5.1 设计

统一使用通用遍历，metric 命名按 §3.1 树结构生成，不再保留显式 key 分支。

```python
if comet_exp and dist_utils.is_main_process() and global_step % LOG_FREQ == 0:
    # 通用遍历：自动覆盖所有 loss key（包括 aux/dn/enc/pre）
    for k, v in loss_dict_reduced.items():
        family, source, idx = parse_loss_key(k)
        metric_name = f"{source}/{family}"
        if idx is not None:
            if source in ("aux", "enc"):
                metric_name += f"/layer_{idx}"
            elif source in ("dn", "dn_pre"):
                metric_name += f"/group_{idx}"
        comet_exp.log_metric(metric_name, v.item(), step=global_step)
    
    # 梯度监控（标量）
    comet_exp.log_metric("grad/norm/before_clip", grad_norm_before, step=global_step)
    comet_exp.log_metric("grad/norm/after_clip", min(grad_norm_before, max_norm), step=global_step)
    comet_exp.log_metric("param/norm", param_norm, step=global_step)
    comet_exp.log_metric("lr", optimizer.param_groups[0]["lr"], step=global_step)
```

**关键变更**：
- 删除原有 `batch_*` 前缀的所有显式 key 上报（解决 C1 双重上报）
- 删除原有 lines 205-220 的显式 key 分支
- `LOG_FREQ` 为可配置常量（默认 10）

### 5.2 Loss 类型过滤

为避免 Comet 中 metric 爆炸（aux 有 5 type × 6 layer = 30 条 batch 级曲线），仅上报 `loss_dict` 中实际存在的 key（已由 `criterion()` 保证）。若有特殊过滤需求，可配置 `LOG_LOSS_FILTER` 正则：

```python
_LOG_FILTER_PATTERN = None   # None = log all
# _LOG_FILTER_PATTERN = re.compile(r"loss_(mal|bbox|kld|giou)")  # 仅 log 部分 loss type
```

默认不过滤，保持完整性。

---

## 6. 梯度监控（新增）

### 6.1 采集策略

采用**混合策略（方案 C）**：
- **每 10 step**（与 loss 日志同频）：记录梯度总 norm、梯度最大值（标量）
- **每 epoch 结束**：做一次全模型参数梯度直方图（histogram）

### 6.2 标量采集

在 `loss.backward()` 后、`optimizer.step()` 前，利用 `torch.nn.utils.clip_grad_norm_` 返回值获取裁剪前/后 norm：

```python
# 在 backwards 之后，step 之前
if max_norm > 0:
    grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    grad_norm_after = min(grad_norm_before, max_norm)  # 裁剪后 ≤ max_norm
else:
    grad_norm_before = compute_grad_norm(model.parameters())
    grad_norm_after = grad_norm_before
```

参数范数直接计算：
```python
param_norm = sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
```

### 6.3 Epoch 级梯度统计（替代直方图）

> **修订（2026-07-01）**：原 `log_histogram_3d` 每参数一次调用导致 300-400 次上传/epoch，触发 Comet rate limit 产生大量 "throttled" 警告。改为按模块组聚合的标量统计。

每 epoch 结束时（仅在主进程，每 5 epoch 一次），将模型参数按名称前缀分组到 4 个模块组，每组计算梯度的 norm/min/max/mean 并作为标量上报：

| 模块组 | 匹配前缀 | 上报指标 |
|--------|---------|---------|
| `backbone` | `backbone.*` | `grad/backbone/norm`, `grad/backbone/max`, `grad/backbone/mean` |
| `encoder` | `encoder.*` | `grad/encoder/norm`, `grad/encoder/max`, `grad/encoder/mean` |
| `decoder` | `decoder.*` | `grad/decoder/norm`, `grad/decoder/max`, `grad/decoder/mean` |
| `head` | `class_embed.*`, `bbox_embed.*`, `enc_score_head.*`, `enc_bbox_head.*` | `grad/head/norm`, `grad/head/max`, `grad/head/mean` |

```python
_GRAD_GROUPS = {
    "backbone": "backbone",
    "encoder": "encoder",
    "decoder": "decoder",
    "head": ("class_embed", "bbox_embed", "enc_score_head", "enc_bbox_head"),
}
_GRAD_HIST_EVERY_N_EPOCH = 5  # 每 5 epoch 上报一次


def _log_gradient_stats(
    model: nn.Module, comet_exp, epoch: int
) -> None:
    """按模块组聚合梯度统计量，替代逐参数直方图。"""
    import numpy as np
    from collections import defaultdict

    group_grads = defaultdict(list)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_abs = param.grad.detach().cpu().float().abs()
        for group_key, patterns in _GRAD_GROUPS.items():
            if isinstance(patterns, str):
                patterns = (patterns,)
            if any(name.startswith(p) for p in patterns):
                group_grads[group_key].append(grad_abs)
                break

    for group_key, grads in group_grads.items():
        if not grads:
            continue
        all_grads = torch.cat([g.flatten() for g in grads])
        comet_exp.log_metric(
            f"grad/{group_key}/norm", all_grads.norm(2).item(), epoch=epoch
        )
        comet_exp.log_metric(
            f"grad/{group_key}/max", all_grads.max().item(), epoch=epoch
        )
        comet_exp.log_metric(
            f"grad/{group_key}/mean", all_grads.mean().item(), epoch=epoch
        )
```

**调用频率**：仅在 `epoch % _GRAD_HIST_EVERY_N_EPOCH == 0` 时触发，仅在主进程执行。每次上报 4 组 × 3 标量 = 12 次 API 调用，远低于原来的 300-400 次。

**性能考量**：
- 4 个模块组的 `.cpu()` 拷贝总量与原方案相近
- API 调用数从 ~300 → 12，完全消除 rate limit 警告
- 每 5 epoch 一次，额外开销可忽略

---

## 7. Comet Epoch 日志

### 7.1 设计

改为通用遍历 `metric_logger.meters`，一行代码替代现有的 20 行 if/elif 分支：

```python
if comet_exp and dist_utils.is_main_process():
    for k, meter in metric_logger.meters.items():
        if k == "loss":
            comet_exp.log_metric("main/loss_total", meter.global_avg, epoch=comet_step)
        elif k == "lr":
            comet_exp.log_metric("lr", meter.global_avg, epoch=comet_step)
        elif k.startswith("loss_"):
            family, source, idx = parse_loss_key(k)
            metric_name = f"{source}/{family}"
            if idx is not None:
                if source in ("aux", "enc"):
                    metric_name += f"/layer_{idx}"
                elif source in ("dn", "dn_pre"):
                    metric_name += f"/group_{idx}"
            comet_exp.log_metric(metric_name, meter.global_avg, epoch=comet_step)

    if epoch > 0 and epoch % _GRAD_HIST_EVERY_N_EPOCH == 0:
        _log_gradient_stats(model, comet_exp, comet_step)
```

**关键变更**：
- 自动覆盖 FGL/DDF（解决 C2 遗漏）
- 删除所有显式 `if "loss_kld" / elif "loss_giou"` 分支
- metric 命名与 batch 级一致（如 `train/main/loss_fgl`）
- 同时 log 梯度直方图（§6.3）

---

## 8. 控制台 Epoch Summary

### 8.1 设计

增加 `is_main_process()` 守卫，显示改为动态遍历：

```python
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

**效果**（OBB 模式）：
```
============================================================
Training Summary - Epoch 5
============================================================
  Total Loss:       5.4231
  MAL              1.8523
  BBOX             0.0341
  KLD              0.0189
  FGL              3.1142
  DDF              0.4036
  Learning Rate:    0.000100
============================================================
```

---

## 9. 移除 TensorBoard Writer

删除 `det_engine.py:191-196` 整个 writer 块。`SummaryWriter` 相关 import（line 15）保留不删（可能被其他模块使用），仅删除调用处。

---

## 10. detect_solver.py 配套修改

### 10.1 传递配置信息

在 `det_solver.py:141-159` 的 `train_one_epoch()` 调用中额外传递 `cfg` 或 `num_layers`，使 `det_engine.py` 能获取 decoder 层数（用于日志格式化判断，如确定 layer 索引范围）。

```python
train_stats = train_one_epoch(
    ...
    cfg=args,                      # 新增：传递整个配置
    kendall=kendall,
    kendall_optimizer=kendall_optimizer,
)
```

### 10.2 Kendall 权重日志

现有 Kendall 日志（lines 138-144）从独立路径 `Kendall/{k}` 改为 `kendall/{k}`：

```python
comet_exp.log_metric(f"kendall/{k}", w, step=global_step)
```

---

## 11. 文件变更清单

| 文件 | 变更 | 行数估计 |
|------|------|---------|
| `deimv2_daod/engine/solver/det_engine.py` | 重构日志 + 新增梯度监控 + 添加 `parse_loss_key()` | ~90 行 → ~110 行（+20 净增，旧代码删除 + 新逻辑更紧凑） |
| `deimv2_daod/engine/solver/det_solver.py` | 传递 `cfg` 参数 + Kendall 日志路径调整 | ~3 行修改 |
| `deimv2_daod/engine/solver/__init__.py` | 不修改 | — |
| `deimv2_daod/engine/deim/deim_criterion.py` | 不修改 | — |
| 任意 config yml | 不修改 | — |

---

## 12. 风险与约束

| 风险 | 缓解 |
|------|------|
| Comet `log_histogram_3d` 每 epoch 调用量大（~200-400 次） | 可通过 `LOG_GRAD_HIST_EVERY_N_EPOCH` 降频，默认 batch 级不 histogram |
| `parse_loss_key()` 解析后缀可能与未来扩展 key 冲突 | 后缀匹配按优先级顺序（`enc` > `aux` > `dn_pre` > `dn` > `pre`），用下划线边界确保不会把 `loss_enc` 主干 key 误匹配 |
| DDP 多 GPU 下 gradient histogram 重复采集 | 仅在 `is_main_process()` 下执行，其他 rank 不采集不传输 |
| 移除 writer 后缺少备用监控通道 | Comet 为主监控平台，tqdm 为实时反馈，功能完整 |

---

## 13. 自检

- [x] 无 TBD/TODO 占位符
- [x] §3.1 树结构与 §5 §7 的实现一致（`train/{source}/{family}/{layer|group}`）
- [x] HBB/OBB 双模已在 §3.3 覆盖，无需额外分支
- [x] 所有 C/R/O 问题（C1/C2/C3/R1/R2/R3/O1-O9）均有对应解决方案
- [x] 变更范围限定在 2 个 `.py` 文件，无 config 修改需求
- [x] 与现有 Kendall 机制兼容（Kendall 在 `loss_dict` 之外修改 `loss`，不影响 `loss_dict_reduced` 的 key 组成）
