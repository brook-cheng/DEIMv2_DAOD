# det_engine.py 日志/监控代码审查报告

> **审查日期**：2026-07-01
> **审查范围**：`deimv2_daod/engine/solver/det_engine.py:162-259`（loss reduce → tqdm → writer → comet batch → comet epoch → console summary）
> **审查方法**：逐行人工审查，对照 `deim_criterion.py` 中 `forward()` 的 loss 生成逻辑，验证 logging 管道的端到端正确性。
> **上下文**：OBB 合成椭圆训练 (`box_mode="obb"`)，loss 列表为 `[mal, boxes, local]`，对应 `loss_mal`、`loss_bbox`、`loss_kld`、`loss_fgl`、`loss_ddf` 五个 key。

---

## 0. 结论速览

| # | 位置（det_engine.py） | 问题 | 严重度 | 类别 | 状态 |
|---|----------------------|------|--------|------|------|
| C1 | 199-200 vs 205-220 | Comet batch 日志双重上报（通用 loop + 显式 key 重复） | 重要 | 正确性 | ❌ 未修复 |
| C2 | 239-257 | Epoch 级 Comet 日志缺失 `loss_fgl` 和 `loss_ddf` | 重要 | 正确性 | ❌ 未修复 |
| C3 | 199-257 | Comet metric 命名不一致（`batch_*` / 无前缀 / `epoch_*` 混用） | 次要 | 正确性 | ❌ 未修复 |
| R1 | 175-188, 198-221, 227-237, 239-257 | 4 处 hardcoded loss key 清单，新增 loss 类型需改 4 处 | 主要 | 合理性 | ❌ 未修复 |
| R2 | 206 vs 215 | `.get()` 安全访问与直接 `[]` 访问混用，不一致 | 次要 | 合理性 | ❌ 未修复 |
| R3 | 223-237 | Epoch summary 打印未做 `is_main_process()` 守卫 | 次要 | 合理性 | ❌ 未修复 |

---

## C 组：正确性问题

### C1. Comet batch 日志双重上报（重要）

**位置**：`det_engine.py:199-200` vs `205-220`

**现状**：

```python
# 第 1 层：通用循环，所有 key 记录为 batch_{k}
for k, v in loss_dict_reduced.items():                          # line 199
    comet_exp.log_metric(f"batch_{k}", v.item(), step=global_step)  # line 200

# 第 2 层：显式 key，用无前缀名再次记录
comet_exp.log_metric("loss_mal", loss_dict_reduced.get("loss_mal", 0), step=global_step)  # line 206
comet_exp.log_metric("loss_fgl", loss_dict_reduced["loss_fgl"], step=global_step)          # line 215
```

**后果**：
- `loss_mal` 等 key 在每个 step 被上报两次：一次作为 `batch_loss_mal`，一次作为 `loss_mal`
- 浪费 Comet API 配额（当前 5 个显式 key × step 数 × epoch 数 = 上千次冗余调用）
- Comet UI 中出现两个同名含义的 metric，造成混淆

**修复方向**：删除 lines 205-220 的显式 key 分支，通用循环已覆盖所有 key。或将通用循环改为白名单过滤。

**根因**：显式 key 是早期只有 3 个 loss 时写的，后来新增 FGL/DDF 时仅追加了显式 key 而忘记同步清理通用循环。

---

### C2. Epoch 级 Comet 日志缺失 `loss_fgl` 和 `loss_ddf`（重要）

**位置**：`det_engine.py:239-257`

**现状**：Epoch 级 Comet summary 仅记录 `total`、`mal`、`bbox`、`kld|giou`、`lr`：

```python
comet_exp.log_metric("epoch_loss_total", metric_logger.loss.global_avg, epoch=comet_step)
comet_exp.log_metric("epoch_loss_mal", metric_logger.loss_mal.global_avg, epoch=comet_step)
comet_exp.log_metric("epoch_loss_bbox", metric_logger.loss_bbox.global_avg, epoch=comet_step)
if "loss_kld" in metric_logger.meters: ...
elif "loss_giou" in metric_logger.meters: ...
comet_exp.log_metric("epoch_lr", ...)
# ← 缺失: epoch_loss_fgl, epoch_loss_ddf
```

**对照**：batch 级 Comet（lines 214-221）**有条件地**记录了 `loss_fgl` 和 `loss_ddf`，控制台 epoch summary（lines 227-237）也**完全没有**显示 FGL/DDF。

**后果**：FGL 和 DDF 是 DEIM/DFINE 论文的核心创新（Fine-Grained Localization + Decoupled Distillation），但在 epoch 级监控中完全不可见。训练趋势调试时只能从 batch 级粒度（每 10 step）查看，噪音大且不直观。

**根因**：`loss_fgl` 和 `loss_ddf` 是项目定制新增的 loss 类型，新增时只改了 batch 级日志和 tqdm 显示，遗漏了 epoch 级 summary 和 console print。

---

### C3. Comet metric 命名不一致（次要）

**位置**：`det_engine.py:199-257`

**现状**：同一语义的 metric 在不同上下文使用不同前缀：

| 位置 | 示例 key | 前缀 |
|------|----------|------|
| Batch 通用 loop（line 200） | `batch_loss_mal` | `batch_` |
| Batch 显式（line 206） | `loss_mal` | 无前缀 |
| Epoch（line 244） | `epoch_loss_mal` | `epoch_` |

**后果**：Comet 面板中 `loss_mal`、`batch_loss_mal`、`epoch_loss_mal` 出现在不同 metric 组，无法直接关联对比。

---

## R 组：合理性问题

### R1. Hardcoded loss key 清单分散在 4 处（主要）

**问题描述**：新增一种 loss 类型时，需要修改 4 个独立代码块：

| 位置 | 行号 | 含义 | 修改难度 |
|------|------|------|---------|
| tqdm postfix | 175-188 | 进度条实时显示 | 需手动加 `if "new_loss" in loss_dict_reduced` |
| Comet batch 显式 | 205-220 | batch 级上报 | 需手动加 `if "new_loss"` 分支 |
| 控制台 epoch summary | 227-237 | 训练结束打印 | 需手动加 print 行 |
| Comet epoch summary | 239-257 | epoch 级上报 | 需手动加 `if "new_loss" in metric_logger.meters` |

**唯一不 hardcode 的通道**：writer 块（lines 195-196）用 `for k, v in loss_dict_reduced.items()` 遍历所有 key，无论新增什么 loss 都自动覆盖。但该通道即将被移除（用户选择仅保留 tqdm + comet）。

**证据**：FGL/DDF 在 C2 中的遗漏就是 hardcode 模式的直接后果 —— 新增 loss 时 batch 级改了一半、epoch 级完全没改。

---

### R2. 安全访问与直接访问混用（次要）

**位置**：`det_engine.py:206` vs `215`

```python
# line 206: 安全 —— 缺失时返回 0
loss_dict_reduced.get("loss_mal", 0)

# line 215: 不安全 —— 缺失时抛出 KeyError
loss_dict_reduced["loss_fgl"]
```

**风险**：如果某天配置从 `losses` 列表中移除 `local`（从而移除 fgl/ddf），Comet batch 日志会在 line 215 crash，而之前的 tqdm/loss_total 都正常。

**建议**：统一使用 `.get(k, 0)` 或统一断言 key 存在后直接用 `[]`。

---

### R3. Epoch summary 打印无多进程守卫（次要）

**位置**：`det_engine.py:223-237`

```python
metric_logger.synchronize_between_processes()   # line 223
print("\n" + "=" * 60)                           # line 224 ← 每个进程都执行
```

**对照**：batch 级日志（lines 191, 198）正确使用了 `dist_utils.is_main_process()` 守卫。

**后果**：DDP 训练时每个 GPU 进程输出一份相同摘要，控制台噪声增加 N 倍。值本身由 `synchronize_between_processes()` 保证一致，不产生数据错误，仅影响可读性。

---

## O 组：优化建议

以下建议已经过用户确认采纳，将在设计文档中展开：

| # | 建议 | 来源 |
|---|------|------|
| O1 | 将所有 logging 改为通用遍历，仅用一个 `LOGGED_LOSS_KEYS` 白名单控制过滤 | R1 |
| O2 | Comet metric 统一层级命名：`train/main/*`、`train/aux/*`、`train/dn/*`、`train/grad/*`、`val/*` | C3 |
| O3 | Aux 损失按 type（`loss_mal`, `loss_bbox`...）而非 layer 分组，利用 Comet `/` 分隔符自动建树 | 用户需求 |
| O4 | 主损失和 eval 结果置顶、突出显示 | 用户需求 |
| O5 | 新增梯度监控：batch 级 norm/max，epoch 级直方图（策略 C） | 用户需求 |
| O6 | 移除 TensorBoard writer | 用户需求 |
| O7 | tqdm 显示改为动态构建（遍历 DISPLAY_KEYS 列表） | O3 |
| O8 | 删除 Comet batch 显式 key 双重上报 | C1 |
| O9 | Epoch 级 Comet 日志切换为通用遍历 | C2 |

---

## 审查上下文

### 日志管道路径

```
criterion(outputs, targets)                                     # deim_criterion.py:494
  → loss_dict: {loss_mal: ..., loss_bbox: ..., ..., loss_kld_aux_2: ..., ...}
  → 每个值已乘以 weight_dict[k]（deim_criterion.py:554-558）

loss_dict_reduced = dist_utils.reduce_dict(loss_dict)           # det_engine.py:162
loss_value = sum(loss_dict_reduced.values())                    # det_engine.py:163

├── metric_logger.update(loss=loss_value, **loss_dict_reduced)  # 170: 做滑动平均
├── tqdm.set_postfix(postfix_dict)                              # 175-188: 实时显示
├── writer.add_scalar(...)                                      # 191-196: TensorBoard（将被移除）
├── comet_exp.log_metric("batch_*", ...)                        # 199-200: batch 通用
├── comet_exp.log_metric("loss_*", ...)                         # 205-220: batch 显式（冗余）
└── comet_exp.log_metric("epoch_*", ...)                        # 239-257: epoch summary
```

### 当前 loss_dict 示例（synthetic_exp_020, OBB）

```
loss_mal           = raw_mal × 4.0
loss_bbox          = raw_bbox × 1.0
loss_kld           = raw_kld × 1.0
loss_fgl           = raw_fgl × 0.5
loss_ddf           = raw_ddf × 1.5

loss_mal_aux_0..5   = (各 decoder 层辅助)
loss_bbox_aux_0..5
loss_kld_aux_0..5
loss_fgl_aux_0..5

loss_mal_enc_0..1   = (encoder 辅助)
loss_bbox_enc_0..1

loss_mal_dn_0..4    = (CDN 去噪组)
loss_bbox_dn_0..4
loss_kld_dn_0..4
loss_fgl_dn_0..4

loss_mal_pre / loss_bbox_pre / loss_kld_pre / loss_fgl_pre
loss_mal_dn_pre / loss_bbox_dn_pre / loss_kld_dn_pre / loss_fgl_dn_pre
```
