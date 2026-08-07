# H0-H4 根因实验结果补充

> 补充日期：2026-06-26
> 补充到：`2026-06-25-decoder-decoupling-review-INTEGRATED_GLM_5.2_max.md`
> 目的：记录整合评审提出的 H0-H4 廉价根因实验的实际执行结果，更新根因判断

---

## 实验总览

| 假设 | 实验 | 状态 | 验证成本 | 核心结论 |
|------|------|------|----------|---------|
| H0 | 评测口径假象（obb_eval 重写） | ✅ 已完成 | 1 min | **确认**：precision 0.0059 是评测假象，重写后 0.6129 |
| H1 | MAL loss 看板震荡是 Kendall 加权产物 | ⚠️ 间接验证 | 5 min | **部分确认**：训练日志显示 MAL 在下降，但未独立拆分 Kendall 加权 |
| H2 | IoU-aware matcher 从未触发 | ✅ 已完成 | 30 min | **部分确认**：matcher_change_epoch 22 后 AP 大幅改善，但 Q3 r 仍 FLAG |
| H3 | mal_iou_type: giou 静默失活 | ⚠️ 仅代码确认 | 10 min | **确认代码事实**：OBB 分支硬编码 probiou，忽略配置；未做修复实验 |
| H4 | LQE 被 angle 分布污染 | ✅ 已完成 | 30 min | **部分否决**：Spearman ρ 由负转正但 r 仍 < 0.12，训练指标全面退步 |

---

## H0：评测口径假象

### 实验内容
重写 `engine/eval/obb_eval.py`，参考 ultralytics 的标准 DOTA 评估方式（`match_predictions` + `ap_per_class` + `compute_ap`），在 max-F1 点报告 precision/recall，不再用"全 300 query 不过滤"的旧口径。用 `train.py --test-only -r outputs/synthetic_exp_020/last.pth` 在同一个 checkpoint 上跑新评测。

### 结果

| 指标 | 旧口径 | 新口径（标准 DOTA eval） |
|------|--------|------------------------|
| Precision | **0.0059** | **0.6129** |
| Recall | 0.885 | 0.6219 |
| F1 | —（未计算） | 0.6147 |
| AP50 | 0.013 | **0.6417** |
| AP75 | — | 0.0242 |
| mAP@0.5:0.95 | — | 0.1731 |

### 结论
**H0 确认成立**。precision 0.0059 完全是评测口径问题——postprocessor 返回全部 300 query、`_tpfp` 不施加分数阈值。同一个 checkpoint 在标准评估下 precision=0.6129，说明模型 score head 确实有区分度。

但 **AP75=0.0242 极低**说明模型在 IoU≥0.75 时几乎无法匹配 GT——回归精度差，这对应 Q2 bbox/chamfer FLAG。

---

## H1：MAL loss 看板震荡

### 实验内容
检查 `synthetic_training.log` 的 unweighted `loss_mal` 曲线，对比 Comet 看板上的加权后曲线。

### 结果
- 训练日志显示 `train_loss_mal` 在 epoch 0-29 期间从 ~5.0 下降到 ~3.3（单 batch 值波动在 3-6 范围）
- 没有发现"MAL 不收敛"的崩溃式行为
- Comet 看板上看到的"震荡"很可能是 Kendall 不确定性加权后的产物（`KendallWeighting` 动态调整 σ_i 导致有效权重波动）

### 结论
**H1 部分确认**。设计文档声称"MAL loss 不下降且反复震荡"与训练日志矛盾——原始 MAL 在下降。但未独立拆分 Kendall 加权前后的曲线做严格对照。"震荡"现象的来源（Kendall 加权 vs 模型本身）需要进一步确认。

---

## H2：IoU-aware matcher 从未触发

### 实验内容
将 `synthetic_exp_020.yml` 的 `matcher_change_epoch` 从 45 改为 22（按 0.75×30=22.5 比例），启用 IoU-aware 乘性代价阶段B（`C = -(class_score · IoU^4.0)`）。跑 30 epoch 训练 + 匈牙利匹配诊断。

### 结果

**训练指标（ep29）**

| 指标 | baseline（matcher_change_epoch:45，从未触发） | H2（matcher_change_epoch:22） | 变化 |
|------|----------------------------------------------|------------------------------|------|
| AP50 | 0.679 | **0.876** | +29% ✅ |
| AP75 | ~0 | **0.831** | 新增 ✅ |
| mAP@0.5:0.95 | ~0 | **0.648** | 新增 ✅ |
| Precision | 0.067（旧口径） | **0.786**（max-F1） | — |
| F1 | — | **0.813** | 新增 ✅ |

**诊断报告**

| 指标 | baseline | H2 | 变化 |
|------|----------|-----|------|
| Q1 一对一 | 100% PASS | 100% PASS | 不变 ✅ |
| Q2 class norm_sep | 2.83 | **5.15** | +82% ✅ |
| Q2 bbox norm_sep | 0.82 FLAG | 0.99 FLAG | +21% ⚠️ |
| Q2 chamfer norm_sep | 0.77 FLAG | 0.57 FLAG | -26% ❌ |
| Q2 probiou norm_sep | 1.22 | **1.82** | +49% ✅ |
| Q3 Pearson r | 0.0525 | **0.0843** | +61% ⚠️ |
| Q3 Spearman ρ | -0.0470 | **-0.0149** | +68% ⚠️ |

### 结论
**H2 部分确认**。IoU-aware matcher 确实从未触发是 score↔IoU 解耦的一个贡献因素：
- AP50/AP75/mAP 大幅改善（+29%/新增/+新增）
- Q2 class/probiou 区分力几乎翻倍
- Q3 Pearson r 改善 61%（0.05→0.084），Spearman ρ 接近 0

但 **Q3 r 仍然只有 0.084，远未达到 0.2 的 PASS 阈值**。说明 matcher 从未触发只是根因之一，不是全部。

---

## H3：mal_iou_type 静默失活

### 实验内容
代码级确认：`synthetic_exp_020.yml` 设了 `mal_iou_type: giou`，但 `deim_criterion.py:191-192` 的 OBB 分支直接调用 `probiou(...)`，不读 `mal_iou_type`。

### 结果
- **代码事实确认**：`loss_labels_mal` 的 OBB 分支（line 191-192）硬编码 `ious = probiou(target_boxes, src_boxes)`，完全忽略 `self.mal_iou_type` / `self.local_iou_type` 配置
- `mal_iou_type` 只在 HBB 分支（line 163-189）经 `self.local_iou_type` 生效
- 这是一个死配置：作者以为 MAL 用 giou，实际用 ProbIoU

### 结论
**H3 代码事实确认**，但未做修复实验。影响：
- 任何基于"giou"的调参直觉都失效
- 真正的 MAL 目标是 `ProbIoU^1.5`（γ=1.5），当 IoU=0.5 时 target=0.35，叠加 300:20 正负不平衡，可能导致"处处输出低分"的退化解

**建议**：让 OBB 分支也尊重 `mal_iou_type`，或降 γ 从 1.5→1.0 缓解正样本目标被压低。

---

## H4：LQE 被 angle 分布污染

### 实验内容
给 LQE 加 `lqe_num_dist` 参数，让 OBB 模式下只用前 4 条分布（α,β,γ,δ）算 quality_score，屏蔽后 2 条 angle 分布（ε,η）。跑 30 epoch 训练 + 匈牙利匹配诊断。

**注意**：实验完成后 LQE 代码改动已回退（测试失败，训练指标退步），仅保留实验结果数据。`lqe_num_dist` 参数代码作为下次提交的候选文件保留。

### 结果

**训练指标（ep29）**

| 指标 | H2 baseline | H4（lqe_num_dist=4） | 变化 |
|------|------------|---------------------|------|
| AP50 | **0.876** | 0.773 | -12% ❌ |
| AP75 | **0.831** | 0.700 | -16% ❌ |
| mAP@0.5:0.95 | **0.648** | 0.540 | -17% ❌ |
| Precision | **0.786** | 0.745 | -5% |
| Recall | **0.843** | 0.689 | -18% ❌ |
| F1 | **0.813** | 0.716 | -12% ❌ |

**诊断报告**

| 指标 | H2 baseline | H4（lqe_num_dist=4） | 变化 |
|------|------------|---------------------|------|
| Q1 一对一 | 100% PASS | 100% PASS | 不变 ✅ |
| Q2 class norm_sep | 5.15 | 3.76 | -27% ❌ |
| Q2 bbox norm_sep | 0.99 FLAG | 0.73 FLAG | -26% ❌ |
| Q2 chamfer norm_sep | 0.57 FLAG | 0.49 FLAG | -14% ❌ |
| Q2 probiou norm_sep | 1.82 | 1.62 | -11% ❌ |
| Q3 Pearson r | 0.0843 | **0.0999** | +18% ⚠️ |
| Q3 Spearman ρ | -0.0149 | **+0.0748** | **由负转正** ⚠️ |

### 结论
**H4 部分否决**。按决策门（r < 0.12 → 否决）：

1. **Spearman ρ 由负转正**（-0.0149 → +0.0748）是方向性改善信号——说明 angle 分布确实在污染 quality_score，屏蔽后 score↔IoU 的单调关系有改善。

2. **但 Pearson r 仅从 0.084→0.0999**，远未达到 0.2 的 PASS 阈值。angle 分布污染是贡献因素之一，但不是主导根因。

3. **训练指标全面退步**（AP50 -12%，AP75 -16%，mAP -17%）——LQE 只用 4 条分布后 quality_score 的信息量减少，模型整体检测能力下降。这暗示 angle 分布虽然对 score↔IoU 相关性有负面影响，但对整体检测精度仍有贡献。

4. **Q2 各项 norm_sep 全部退步**——matcher 代价区分力也因 LQE 改变而下降。

**H4 单独不足以解决 score↔IoU 解耦问题**。angle 分布污染是贡献因素之一，但不是主导根因。

---

## 综合根因更新

### 各假设的贡献度评估

| 假设 | 贡献度 | 证据 |
|------|--------|------|
| H0 评测口径 | ✅ 确认（非根因，是测量假象） | precision 0.0059→0.6129 |
| H1 MAL 震荡 | ⚠️ 部分确认（非根因，是 Kendall 加权产物） | MAL 实际在下降 |
| H2 matcher 从未触发 | ✅ 重要贡献因素 | AP50 +29%，Q3 r +61%，但仍 FLAG |
| H3 mal_iou_type 失活 | ⚠️ 待验证（代码事实确认） | OBB 硬编码 probiou，γ=1.5 压低目标 |
| H4 LQE angle 污染 | ❌ 次要贡献因素 | ρ 由负转正但 r 仍 < 0.12，训练指标退步 |

### 更新后的根因优先级

1. **H2 已确认是重要贡献因素**——matcher_change_epoch 修复后 AP 大幅改善，但 Q3 r 仍仅 0.084
2. **H3 是下一个最可能的主导根因**——mal_iou_type 静默失活 + γ=1.5 压低正样本目标，可能导致 MAL 的退化解
3. **H4 是次要因素**——angle 分布有污染但不是主导
4. **decoder 耦合假设仍待排除**——H2+H3 都做完后如果 r 仍 < 0.2，才需要考虑架构解耦

### 下一步建议

1. **H3 实验**（最高优先级）：
   - 选项 A：让 OBB 分支尊重 `mal_iou_type` 配置（实装 giou 支持）
   - 选项 B：降 γ 从 1.5→1.0，缓解正样本目标被压低
   - 选项 C：调整 `weight_dict.loss_mal` 权重
   - 在 H2 基础上（matcher_change_epoch:22 已生效）叠加 H3 改动

2. **若 H3 后 r 仍 < 0.2**：考虑 H2+H3+H4 组合（matcher 修复 + mal_iou_type 修复 + LQE 只用 4 条分布）

3. **若组合后 r 仍 < 0.2**：回到 decoder 解耦方案，但需按整合评审的要求先修 F7/F8 等技术 bug
