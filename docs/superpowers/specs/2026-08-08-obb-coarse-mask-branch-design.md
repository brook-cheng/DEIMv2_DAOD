# DEIMv2 OBB 粗 mask 分支设计（双头并行）

日期：2026-08-08

状态：设计已确认（双头并行 / 对齐 GT OBB / 边界带 BCE / 仅匹配 query）。

关联文档：

- 设计动机来源：`test/tool_debug_decoder.py` Phase 5 特征 cosine-sim 可视化
  （`tools/analysis/feature_similarity.py` 的 `cosine_similarity_map_imgsz`）
- 角度契约：`docs/superpowers/specs/2026-08-05-obb-angle-contract-simplification-design.md`
  （公开物理域 `[0, π)` / decoder 私有归一域）
- decoder 解耦：`docs/superpowers/specs/2026-06-25-decoder-decoupling-design.md`
- 消融基线：`configs/custom_obb/dlzdt/sp_fz_common.yml` 及 `ablation/` 9 配置

## 1. 背景与动机

在 `test/tool_debug_decoder.py` 的 Phase 5 中，对 backbone c2/c3/c4 与 encoder memory
计算 GT 中心点到全图的 cosine 相似度热力图，观察到：

- **目标区域内部一致性高**：同一物体覆盖范围内的特征与 GT 中心特征高度相似；
- **显著突出于背景**：目标与背景之间存在清晰的可分性。

这提示 encoder memory（stride-8 per-pixel embedding）已经编码了物体形状信息。
本设计将该性质显式化：在保留现有参数化 OBB 检测头（保证检测精度）的同时，
新增一个 **coarse mask 分支**，把网络当作一个"分割精度不高的弱监督实例分割器"使用：

- 训练时：mask 的（可微近似）外接旋转矩形对齐 GT OBB 计算 loss；
- 推理时：额外输出 mask，作为低精度实例分割产物；
- 检测任务（DOTA/OBB 指标）仍由原 OBB 头负责，指标零回归。

## 2. 目标与非目标

### 2.1 目标

1. 新增 `MaskHead` 模块与 `use_mask_head` 配置开关，**默认 `False`**，
   现有训练、checkpoint、配置、指标完全不受影响。
2. mask 训练信号三项（仅 matched query）：
   - PCA-外接矩形对齐 GT OBB（GIoU + L1）；
   - GT 旋转矩形边界带 BCE（锚定矩形边界、内部形状自由）；
   - 匹配仍用现有 box+class Hungarian，mask 不参与匹配。
3. mask logits 在 criterion 侧懒计算（路径 a），训练前向只对 matched query
   （通常 < 50）算 mask，成本可忽略。
4. 推理时对所有 query 输出 mask（sigmoid 阈值化 + 上采样到原图分辨率）。
5. mask 头为空间操作，对 angle_rep 0/1/2/3 天然兼容（PCA 矩形监督产物即 5-DOF OBB）。

### 2.2 非目标

1. 不替换、不修改现有 OBB 回归头（`dec_bbox_head`、`distance2bbox_obb` 链路零改动）。
2. 不追求高精度实例分割；mask 为粗粒度副产品（stride-8 分辨率）。
3. 不做真 `minAreaRect`（旋转卡壳）——其不可微、非 batch、无 GPU 实现；
   统一使用矩/PCA 可微近似，见 §5。
4. 不使用整矩形填充伪 mask + BCE（会把 mask 训成实心矩形，违背物体形状目标）。
5. 不修改公开角度契约 `[0, π)`；PCA 矩形角度经规范化后与 GT 同约定。
6. 当前仅设计 rep0 基线验证，不承诺立即推广到 rep1/2/3 配置族（结构兼容，
   推广为后续工作）。

## 3. 术语表与域

| 变量 | 形状 / 量纲 | 用途 |
| --- | --- | --- |
| `memory` | `[B, H*W, 256]`，stride-8 | encoder 输出的 per-pixel embedding（已有，`DEIMTransformer.forward` 内可见） |
| `mask_embeds` | `[B, N_q, 256]` | decoder 最后一层 query embedding 经 `MaskHead` 投影后的 mask query 向量 |
| `mask_logits` | `[B, N_q, H*W]` | `mask_embeds · memoryᵀ`，stride-8 分辨率 |
| `soft_mask` | `[B, N_q, H*W]`，`(0,1)` | `sigmoid(mask_logits)`，训练时用于矩计算 |
| `pca_rect` | `(cx, cy, w, h, θ)`，θ∈`[0,π)` | soft_mask 的矩/PCA 可微外接矩形近似 |
| GT OBB | `(cx, cy, w, h, θ)`，θ∈`[0,π)` | 现有 dataset/criterion 公开物理角契约 |
| `band_mask` | `[B, N_q, H*W]`，`{0,1}` | GT 旋转矩形边界带（带内 1、带外 0、忽略区不参与 loss） |

## 4. 架构

### 4.1 数据流

```
encoder memory (stride-8, [B, 80×80, 256]) ──► per-pixel embedding（已有）
                        │
decoder 最后一层 output ─┼─► dec_bbox_head → OBB（不变，检测指标零回归）
                        └─► MaskHead（新）→ mask_embeds [B, 300, 256]
                                             │
                                             ▼
训练: mask_logits[matched] = mask_embeds[matched] · memoryᵀ  （criterion 侧，仅 matched）
推理: mask_logits[all]     = mask_embeds[all] · memoryᵀ      （transformer 侧，无 backward）
```

### 4.2 挂载点

- `DEIMTransformer` 新增构造参数 `use_mask_head: bool = False`。
- 新增模块 `MaskHead`（`engine/deim/mask_head.py`）：
  `Linear(hidden, hidden) → RMSNorm(hidden) → Linear(hidden, hidden)`，输出 mask query 向量。
- `TransformerDecoder.forward` 在最后一层（eval_idx）的 query embedding（即喂给
  `dec_bbox_head[eval_idx]` 的 `output` 张量）处**分支**：原路不动，另存
  `last_query_embeds` 返回给 `DEIMTransformer`。
- `DEIMTransformer.forward`：
  - 训练时：把 `mask_embeds = mask_head(last_query_embeds)` 与 `memory` 一并放入
    preds `out` dict（键 `mask_embeds`、`mask_memory`），由 criterion 消费。
    **注意**：`mask_embeds` 须与 `out_logits`/`out_bboxes` 同步应用 denoising
    `dn_meta["dn_num_split"]` 拆分，仅保留常规 query 部分（denoising query
    不参与 matched mask loss）；
  - 推理时（`use_mask_head=True`）：直接计算 `mask_logits` 并入输出。
- `mask_head` 仅在 `use_mask_head=True` 时实例化，否则为零开销占位（`None`）。

### 4.3 与 angle_rep 的关系

- mask 头输入为 decoder 最后一层的 `output`（隐藏层，与 box 头同源），
  其维度不随 `angle_rep` 变化（均为 `hidden_dim`）。
- PCA 矩形监督天然输出 5-DOF `(cx, cy, w, h, θ)`，与 GT OBB 契约一致；
  对 rep0/1/2/3 的 box 头无任何耦合。

## 5. PCA-外接矩形（可微近似）

训练链 `soft_mask → pca_rect → loss(GT OBB)` 必须全程可微，故采用矩/PCA 近似：

### 5.1 公式

对 soft mask `M ∈ [0,1]^{H×W}`（展平后与坐标网格 `(X, Y)` 对应）：

1. **质量（面积）**：`A = Σ M`（加 `eps=1e-4` 防除零）。
2. **中心（一阶矩）**：`cx = Σ(M·X)/A`，`cy = Σ(M·Y)/A`。
3. **协方差矩阵（二阶中心矩）**：
   `S = [Σ M·(X-cx)² , Σ M·(X-cx)(Y-cy) ; ·, Σ M·(Y-cy)²] / A`。
4. **角度**：`θ_pca = 0.5 · atan2(2·S_xy, S_xx − S_yy)`，主特征向量方向。
   - `atan2` 输出域 `[−π/2, π/2)`，需规范化到 `[0, π)`：
     `θ = θ_pca % π`（Python 取模语义保证结果落在 `[0, π)`）。
5. **w / h（主轴投影范围）**：投影坐标
   `u = (X−cx)·cos θ + (Y−cy)·sin θ`，
   `v = −(X−cx)·sin θ + (Y−cy)·cos θ`；
   `w = max(u·M)/A* · 2`，`h = max(v·M)/A* · 2`（其中 `A* = max(A, eps)`，
   或使用 soft 加权分位数以保证对离群像素鲁棒——实现时以分位数版本优先）。
   - 说明：投影范围版本比 `2·sqrt(eigenvalue)` 更能贴合物体实际外接范围，
     且对"边界带锚定的 blob 状 mask"偏差有界。

### 5.2 与真 minAreaRect 的差异

- 矩/PCA 矩形是**最小外接矩形的凸性近似**：对 L 形、弯曲、细长物体的 mask，
  PCA 矩形可能略大于或小于真最小外接矩形。
- 缓解：边界带 BCE 将 mask 锚定为紧贴 GT 矩形的 blob；且监督目标是 GT OBB
  （非预测 box），无累积误差。设计接受该近似，作为"粗 mask"语义的一部分。

## 6. 训练信号

### 6.1 匹配（不变）

- Hungarian 匹配完全沿用现有 box+class 成本，mask 不参与匹配。
- 仅在匹配完成后，对 matched query 计算 mask loss。

### 6.2 Loss 项（仅 matched query）

1. **`mask_giou`**：`pca_rect` 与 GT OBB 的旋转 GIoU。
   复用现有 OBB GIoU 实现（`yolo_obb_loss` / OBB GIoU 函数），角度域 `[0, π)`。
2. **`mask_l1`**：`pca_rect` 与 GT OBB 的参数 L1（cx/cy/w/h 归一域、θ 物理域，
   与现有 box L1 约定一致）。
3. **`mask_bce`（边界带 BCE）**：由 GT 旋转矩形生成距离场，取边界附近
   `δ` 像素宽（起点 `δ=4`，相对 stride-8 特征图）的窄带：
   - 带内像素目标 `1`，带外目标 `0`；
   - 远离边界（距离 > δ）的像素**不参与 loss**（内部形状自由）。
   - 实现：对 GT 矩形内/外符号距离场取 `|d| ≤ δ` 掩码，再对该带做 BCE。

### 6.3 权重起点

`mask_giou=1.0`，`mask_l1=1.0`，`mask_bce=1.0`（可调，需标定）。
在 criterion 中以独立 key 输出，训练日志可单独监控。

### 6.4 计算位置（路径 a：criterion 侧懒计算）

- `DEIMTransformer` 训练时**不计算** mask logits，只透传 `mask_embeds` 与 `memory`。
- `DEIMCriterion` 在 Hungarian 匹配后：
  1. 取 matched query 索引 `idx`；
  2. `matched_embeds = mask_embeds[idx]`（`[B, n_match, 256]`）；
  3. `matched_logits = matched_embeds · memoryᵀ`（`[B, n_match, H*W]`）；
  4. 计算三项 loss。
- 成本：`n_match × H×W × 256`，通常 `n_match < 50`，相对主 loss 可忽略。

## 7. 推理输出

- `use_mask_head=True` 时，eval 路径对所有 query 计算 `mask_logits`（无 backward）。
- `soft_mask = sigmoid(mask_logits)`，阈值 `τ`（起点 0.3–0.5）→ 二值 mask。
- 双线性上采样到原图分辨率，作为每 query 的粗实例 mask。
- 检测/DOTA 评估仍走 OBB 头，输出不变（兼容 `PostProcessor` 既有契约）。
- mask 导出：per-query PNG 或 RLE（`PostProcessor`/导出工具扩展，见 §9 测试）。

## 8. 配置与实验范围

### 8.1 新增配置键

- `DEIMTransformer.use_mask_head: bool`（默认 `False`）。
- `DEIMCriterion` 新增 loss 权重键：
  `mask_giou`、`mask_l1`、`mask_bce`（默认 `0.0`，即不开时不产生 loss）。

### 8.2 实验编排

- 独立配置族：`sp_fz_common.yml` + `use_mask_head: True`（如
  `sp_fz_maskhead.yml`），与当前 6 变量消融（`ablation/`）分离。
- **时序**：在消融实验完成后启动；先以 rep0 基线验证可行性，
  再决定是否推广到 rep1/2/3。

## 9. 测试计划

### 9.1 单元测试

1. **PCA 矩形可微与正确性**：给定已知旋转矩形二值 mask，
   `pca_rect` 应与 GT 参数一致（含角度规范化 `[0, π)`）；
   梯度可回传到 soft_mask。
2. **边界带 BCE**：窄带掩码生成正确；带外/远离像素不贡献 loss；
   GT 矩形 `θ=0` 与 `θ=90°` 边界情况。
3. **MaskHead 前向**：`use_mask_head=False` 时零开销；
   `True` 时输出形状 `[B, 300, 256]`。
4. **criterion 集成**：matched-only 懒计算路径产出三项 loss；
   无 matched query（空图）不崩。
5. **推理输出**：eval 时 mask logits 形状、阈值化、上采样正确。

### 9.2 回归

- 现有 OBB 全量测试（308 项）必须全部通过（`use_mask_head=False` 零影响）。
- 开启 mask 头训练一轮后：OBB 检测指标与基线一致（双头并行零回归）；
  mask 可视化确认物体形状（非实心矩形、非空心退化）。

## 10. 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| PCA 矩形 ≠ 真 minAreaRect（L形/弯曲偏差） | 边界带锚定 blob；监督目标是 GT，无累积误差；语义上接受"粗 mask" |
| mask 内部欠约束 → 空心/哑铃/边缘退化 | 边界带 BCE + cosine-sim 先验（`MLP(memory)` 平滑）；权重标定；测试 9.1.2 |
| soft-mask 矩梯度不稳定 | `eps` 防护、soft 分位数求 w/h、权重标定 |
| 检测指标回退 | 双头并行，box 头零改动；回归测试 9.2 |
| 训练成本 | 路径 a 懒计算，`n_match < 50`，可忽略 |
| 角度规范化歧义 | `θ_pca mod π` 统一到 `[0, π)`，测试 9.1.1 覆盖 |

## 11. 开放项

1. `w/h` 投影范围实现选型：加权极值 vs soft 分位数（实现时以分位数优先，
   测试验证对离群像素鲁棒性）。
2. 边界带宽度 `δ` 与三项 loss 权重的标定值（需一轮训练实验确定）。
3. 推理阈值 `τ` 与 mask 导出格式（PNG/RLE）的最终选择。
