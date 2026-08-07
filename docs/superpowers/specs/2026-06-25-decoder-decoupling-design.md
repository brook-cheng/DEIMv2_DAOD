# DEIMv2-OBB Decoder 解耦设计

> 创建日期：2026-06-25
> 修订日期：2026-06-27（根据用户反馈修正模型结构、LQE、角度锚点）
> 目的：解决分类分数与 IoU 质量无关（MAL loss 不收敛）的问题，对 decoder 进行空间-语义-角度三层解耦
> 状态：评审后修订版，待实施
> 原始设计来源：`/mnt/d/cx/apps_data/obsidian_notes/brook_pkm/知识/文献研究/目标检测/DETR系列/DEIM/DEIMv2-OBB调整与优化.md`

## 背景

经过匈牙利匹配诊断（Q1=PASS, Q2=部分PASS, Q3=FLAG r=0.05）和合成数据集密度消融实验，确认：

1. 匹配 assignment 正确（100% 一对一），但分类分数与 IoU 几乎零相关
2. MAL loss 设计意图（分数与 IoU 正相关）无法实现，Comet 看板上 MAL loss 不下降且反复震荡
3. DEIMv2 的 HBB 模式已验证可行——(cx,cy,w,h,label) 共用 decoder 是正确的

**根因假设**：θ（角度/r）与 (cx,cy,w,h,label) 共用同一个 decoder layer 导致空间-语义-角度信息耦合。θ 同时具有几何和语义属性，需要一个独立的处理路径。参考 STD 的空间侧分支 + 渐进式预测，以及 RIO-DETR 的 Content-Driven θ 预测和正交旋转注意力。

### H0-H4 实验结论（2026-06-26 补充）

| 假设 | 结论 | 对 score↔IoU 解耦的贡献 |
|------|------|------------------------|
| H0 评测口径 | ✅ 确认（非根因，是测量假象） | 无（precision 0.0059 是假象） |
| H1 MAL 震荡 | ⚠️ 部分确认（Kendall 加权产物） | 无（MAL 实际在下降） |
| H2 matcher 从未触发 | ✅ 重要贡献因素（AP +29%） | 部分（Q3 r 从 0.05→0.084，但仍 FLAG） |
| H3 mal_iou_type 失活 | ⚠️ 代码事实确认，不影响训练 | 无 |
| H4 LQE angle 污染 | ❌ 次要因素（ρ 由负转正但 r<0.12，训练退步） | 次要 |

**结论**：H2+H4 都做了，但 Q3 r 仍只有 0.0999。matcher 修复（H2）改善了 AP 但没解决 score↔IoU 解耦。decoder 耦合假设仍待排除。

## 设计目的

解决 baseline 设计中分类分数与 IoU 质量无关（MAL loss 不收敛）的问题，对 decoder 进行空间-语义-角度三层解耦。

解耦方案大致如下：
- **角度拆分**：拆分 `(α,β,γ,δ,ε,η,label)` 的 decoder-layer 为两个，分别 `decoder_layer_box`（解码 `α,β,γ,δ,label`，与原来的 DEIMv2-HBB 处理相同）和 `decoder_layer_angle`（解码 `ε,η`）
- **特征融合**：`r` 不仅依赖于空间特征，也依赖语义特征，采用特征融合的方式，将原始 enc_memory 和每层 decoder_layer_xywh 的输出融合后，输入到每层 decoder_layer_angle 中。融合采用简单 Gate Fusion 方式，后续根据情况来决定是否使用较为复杂的 VMoE 或 SMoE 的方式
- **调整旋转注意力**：参考 RIO-DETR 文献设计，调整可变注意力为 Rotation-Rectified Orthogonal Attention，具体来说就是将可变注意力模块的多头中 `head_idx >= len_head` 的旋转矩阵的角度加上 π/2
- **anchor-box 添加多尺度角度**：在生成 anchor-box 时，按照 `angle_step` 生成不同角度的 anchor-box，为避免生成的 anchor-box 较多，`angle_step` 默认取 10
- **标签预测拆分**：如果单独的角度拆分无法很好解决置信度低的问题，再考虑这种方案。进一步将角度拆分中的 decoder_layer_box，拆分为 decoder_layer_xywh（解码 cx,cy,w,h）和 decoder_layer_label（解码 label），参考 STD 文献，Label Decoder 接收共享 decoder 的输出特征与 encoder memory 作为 cross-attn KV，使用 `Gate` 模块（`deim_utils.py:70`）门控混合空间特征与原始图形特征
- **refinement**：保持原有 OBB refinement 的方式基本不变，唯一变化的是 `(α,β,γ,δ)` 和 `(ε,η)` 分别来自两个不同前馈网络。`decoder_layer_box` 提供 `(α,β,γ,δ,label)` 特征，`decoder_layer_angle` 提供 `(ε,η)` 特征。`pre_bbox_head` 前馈输出初始的旋转矩形的 `(cx,cy,w,h)`，`pre_scores` 前馈输出初始类别 logit，`pre_angle_head` 前馈输出初始角度 `r`，`bbox_head` 前馈输出后续 refinement 需要的 `(α,β,γ,δ)`，`angle_head` 前馈输出 refinement 需要的 `(ε,η)`

## 总体架构

```mermaid
flowchart TD
    subgraph Input["输入准备"]
        ENC["encoder feats"]
        DN["get_contrastive_denoising_training_group<br/>dn_logits, dn_bbox, dn_mask"]
        ANCHOR["_get_decoder_input<br/>grid_w × grid_h × angle_step×N<br/>select_topk → ref_points_initial"]
    end

    subgraph Split["参考点拆分"]
        XYWH["ref_points[:,:,:4]<br/>(cx,cy,w,h) ∈ [0,1]"]
        R_REF["ref_points[:,:,4:6]<br/>(ε,η) ∈ [0,1]"]
    end

    subgraph XYWH_path["XYWH+Label Decoder (6层)"]
        L0["layer_xywh_0: pre_bbox_head + pre_score_head<br/>→ pre_bboxes(cx,cy,w,h), pre_scores"]
        L1["layer_xywh_1: bbox_head → pred_corners(αβγδ)<br/>distance2bbox_obb_xywh → inter_ref_bbox_xywh<br/>LQE(scores, pred_corners_αβγδ)"]
        L2["layer_xywh_2: 重复"]
        L3["layer_xywh_3: 重复"]
        L4["layer_xywh_4: 重复"]
        L5["layer_xywh_5: 重复 → 最终 xywh + scores"]
        L0 -->|"inter_ref_bbox(cx,cy,w,h) 作参考点"| L1
        L1 -->|"inter_ref_bbox(cx,cy,w,h) 作参考点"| L2
        L2 -->|"inter_ref_bbox(cx,cy,w,h) 作参考点"| L3
        L3 -->|"inter_ref_bbox(cx,cy,w,h) 作参考点"| L4
        L4 -->|"inter_ref_bbox(cx,cy,w,h) 作参考点"| L5
    end

    subgraph R_path["R Decoder (6层) + Gate Fusion @每层"]
        R0["layer_r_0: pre_angle_head → pre_angle(ε,η)<br/>(内部 cross-attn 处理 encoder memory)"]
        R1["layer_r_1: angle_head → pred_corners(εη)<br/>distance2bbox_obb_angle → inter_ref_angle"]
        GF1["Gate Fusion<br/>融合 L1_xywh_feat + r_feat"]
        R2["layer_r_2: 重复"]
        GF2["Gate Fusion"]
        R3["layer_r_3: 重复"]
        GF3["Gate Fusion"]
        R4["layer_r_4: 重复"]
        GF4["Gate Fusion"]
        R5["layer_r_5: 重复 → 最终角度"]

        R0 --> R1 --> GF1 --> R2 --> GF2 --> R3 --> GF3 --> R4 --> GF4 --> R5
    end

    ENC --> L0
    ENC --> R0
    DN --> L0
    ANCHOR --> XYWH
    ANCHOR --> R_REF

    L1 -->|"L1 xywh 特征"| GF1
    L2 -->|"L2 xywh 特征"| GF2
    L3 -->|"L3 xywh 特征"| GF3
    L4 -->|"L4 xywh 特征"| GF4

    L0 -->|"inter_ref_bbox(cx,cy,w,h,ε,η) 作参考点"| R1
    L1 -->|"inter_ref_bbox(cx,cy,w,h,ε,η) 作参考点"| R2
    L2 -->|"inter_ref_bbox(cx,cy,w,h,ε,η) 作参考点"| R3
    L3 -->|"inter_ref_bbox(cx,cy,w,h,ε,η) 作参考点"| R4
    L4 -->|"inter_ref_bbox(cx,cy,w,h,ε,η) 作参考点"| R5

    subgraph Output["输出（参考 deim_decoder.py:285-308）"]
        XYWH_OUT["XYWH 路径每层输出:<br/>scores (经 LQE, 用 αβγδ)<br/>inter_ref_bbox_xywh (cx,cy,w,h)<br/>pred_corners_xywh (αβγδ)<br/>ref_points_initial_xywh (cx,cy,w,h)"]
        R_OUT["R 路径每层输出:<br/>inter_ref_angle (ε,η)<br/>pred_corners_angle (εη)<br/>ref_angle_initial (ε,η)"]
        MERGE["拼装为兼容输出:<br/>dec_out_bboxes = cat(xywh, εη) → 6-dof → 转为 5-dof (cx,cy,w,h,θ) ∈ [0,π]<br/>dec_out_logits = scores (来自 XYWH)<br/>dec_out_pred_corners = cat(αβγδ, εη) → 6 条分布<br/>dec_out_refs = cat(xywh_ref, εη_ref) → 6-dof<br/>pre_bboxes = cat(pre_xywh, pre_εη) → 6-dof → 转为 5-dof (仅 i=0)<br/>pre_scores = pre_scores (来自 XYWH)"]
        XYWH_OUT --> MERGE
        R_OUT --> MERGE
    end

    L5 --> XYWH_OUT
    R5 --> R_OUT
```

### 架构关键决策

| 决策 | 说明 |
|------|------|
| **Gate Fusion 频率** | R1~R4 之间各一次（共 4 次），**逐层对应**（L_i xywh 特征 → GF_i） |
| **Gate Fusion 源** | L_i xywh 特征 + r 特征（**不含 enc_memory**，因 layer_r 自带 cross-attn） |
| **无 GF0** | layer_r_0 内部已有 cross-attn 处理 encoder memory，不需要额外融合 |
| **无 GF5** | R5 是最后一层，之后无 R 层消费融合结果 |
| **XYWH 参考点链式传递** | L(n) 输出 `inter_ref_bbox(cx,cy,w,h)` → L(n+1) 的参考点（4-dof），参考 `deim_decoder.py:298` |
| **R 参考点链式传递** | L(n) 输出 `inter_ref_bbox(cx,cy,w,h)` + R(n) 输出 `inter_ref_angle(ε,η)` → 拼接为 6-dof `inter_ref_bbox(cx,cy,w,h,ε,η)` → R(n+1) 的参考点 |
| **R 参考点进入旋转注意力** | 6-dof 参考点 `(cx,cy,w,h,ε,η)` 先经 `external_rect_to_oriented_box` 转换为 5-dof `(cx,cy,w,h,θ)`，再送入 MSDeformableAttention 的 `elif reference_points.shape[-1] == 5:` 分支执行 Rotation-Rectified Orthogonal Attention |
| **pre_angle 表示** | `pre_angle_head` 输出 `(ε,η)`，与 `angle_head` 的 `(ε,η)` 一致，不输出 `r` |
| **distance2bbox_obb** | XYWH 路径只用 `(α,β,γ,δ)` 解码外接矩形；角度部分由 R 路径用 `(ε,η)` 单独解码 |
| **refinement** | 保持原有 OBB refinement 不变，仅 `(α,β,γ,δ)` 和 `(ε,η)` 分别来自不同前馈网络 |
| **角度表示** | 保持 DEIMv2 ADR 方式：`(ε,η)` 作为顶点偏移（DFL 分布 + Integral 解码），初始预测和 refinement 统一用 `(ε,η)` |
| **角度量纲** | decoder 内部 `(cx,cy,w,h,ε,η)` ∈ [0,1]（6-dof，sigmoid 空间）；外部输出 `(cx,cy,w,h,θ)` ∈ [0,π]（5-dof，经 `external_rect_to_oriented_box` 转换） |
| **LQE 输入** | `pred_corners` 只含 `(α,β,γ,δ)`，**不含 `(ε,η)`** |
| **Mode B** | 暂不实现，先做 Mode A（label 留在 xywh 路径） |
| **参数量** | decoder 层数翻倍（6→12），在预期之内 |

## 角度量纲设计

由于在 decoder_layer 中 box_head(MLP) 处理后会使用 sigmoid 归一化输出到 [0,1]，这样满足在 decoder_layer 间传递，而在后续的计算（criterion/matcher/postprocess）中大部分需要 θ ∈ [0,π]，故这里对角度的量纲做一些说明。

| 阶段 | 表示 | 量纲 | 说明 |
|------|------|------|------|
| **dataloader / 数据增强** | `(cx,cy,w,h,θ)` | [0,π] | DOTA 标注 → cxcywhr 格式 |
| **OBBConvertBoxes** | `(cx,cy,w,h,θ)` | [0,π] | 归一化空间坐标，保持 θ ∈ [0,π] |
| **anchor 生成** | `(cx,cy,w,h,ε,η)` | [0,1] | 从 `(cx,cy,w,h,θ)` 经 `oriented_box_to_external_rect` 转换，归一化到 [0,1] |
| **decoder 初始预测** | `(cx,cy,w,h,ε,η)` | [0,1] | `pre_bboxes = sigmoid(pre_bbox_head + inverse_sigmoid(ref))` → `(ε,η)` ∈ [0,1] |
| **decoder FDR 逐层调整** | `(cx,cy,w,h,ε,η)` | [0,1] | `distance2bbox_obb` 输入/输出含 (ε,η)，内部计算时转换为 θ 做 `external_rect_to_oriented_box` |
| **decoder 输出** | `(cx,cy,w,h,θ)` | [0,π] | 经 `external_rect_to_oriented_box` 将 `(ε,η)` 转为 θ，`out_bboxes[..., 4:] *= π` |
| **criterion L1 损失** | `(cx,cy,w,h,θ)` | [0,1] | 统一量纲：`src_boxes[..., 4:] /= π` |
| **criterion MAL/KLD** | `(cx,cy,w,h,θ)` | [0,π] | 直接使用 |
| **matcher** | `(cx,cy,w,h,θ)` | [0,π] | 直接使用 |
| **postprocessor** | `(cx,cy,w,h,θ)` | [0,π] | 直接使用 |

**关键规则**：decoder 内部传递用 `(cx,cy,w,h,ε,η)` 6-dof [0,1]，decoder 输出时经 `external_rect_to_oriented_box` 转为 `(cx,cy,w,h,θ)` 5-dof [0,π]，criterion/matcher/postprocessor 全部用 5-dof [0,π]。

## 设计要点

### 1. 保留 DEIMv2 Refinement

保持原有 OBB refinement 的方式基本不变，唯一变化的是 `(α,β,γ,δ)` 和 `(ε,η)` 分别来自两个不同前馈网络：

- `decoder_layer_box` 提供 `(α,β,γ,δ,label)` 特征
- `decoder_layer_angle` 提供 `(ε,η)` 特征
- `pre_bbox_head` 前馈输出初始的旋转矩形的 `(cx,cy,w,h)`
- `pre_scores` 前馈输出初始类别 logit
- `pre_angle_head` 前馈输出初始角度 `(ε,η)`（与 `angle_head` 一致，不是 `r`）
- `bbox_head` 前馈输出后续 refinement 需要的 `(α,β,γ,δ)`
- `angle_head` 前馈输出 refinement 需要的 `(ε,η)`

**输出流程兼容**：参考 `deim_decoder.py:285-308`，每层 decoder 的输出需保持 `(dec_out_bboxes, dec_out_logits, dec_out_pred_corners, dec_out_refs, pre_bboxes, pre_scores)` 结构。解耦后两条路径分别输出，再拼装：

**XYWH 路径每层输出**：
- `scores`：`score_head[i](output)` → LQE（用 `pred_corners_xywh` 的 αβγδ）→ 分类 logit
- `inter_ref_bbox_xywh`：`distance2bbox_obb_xywh` 解码 — 4-dof `(cx,cy,w,h)`
- `pred_corners_xywh`：`bbox_head[i]` 输出 — 4 条 DFL 分布 `(α,β,γ,δ)`
- `ref_points_initial_xywh`：初始参考点 — 4-dof

**R 路径每层输出**：
- `inter_ref_angle`：`distance2bbox_obb_angle` 解码 — 2-dof `(ε,η)`
- `pred_corners_angle`：`angle_head[i]` 输出 — 2 条 DFL 分布 `(ε,η)`
- `ref_angle_initial`：初始角度参考点 — 2-dof `(ε,η)`

**拼装为兼容输出**（`deim_decoder.py:301-308` 的 return 结构）：
- `dec_out_bboxes` = `torch.cat([inter_ref_bbox_xywh, inter_ref_angle], dim=-1)` — 6-dof `(cx,cy,w,h,ε,η)`，输出时经 `external_rect_to_oriented_box` 转为 5-dof `(cx,cy,w,h,θ)` 并 `*= π`
- `dec_out_logits` = `scores`（来自 XYWH 路径，已含 LQE）
- `dec_out_pred_corners` = `torch.cat([pred_corners_xywh, pred_corners_angle], dim=-1)` — 6 条分布
- `dec_out_refs` = `torch.cat([ref_points_initial_xywh, ref_angle_initial], dim=-1)` — 6-dof
- `pre_bboxes` = `torch.cat([pre_bboxes_xywh, pre_angle], dim=-1)` — 6-dof `(cx,cy,w,h,ε,η)`，输出时转为 5-dof（仅 i=0）
- `pre_scores` = `pre_scores`（来自 XYWH 路径，仅 i=0）

### 2. Label 解耦：双模式（共用 / 独立）

通过配置参数 `decouple_label: true/false` 控制：

| 模式 | label 处理 | 优点 | 缺点 |
|------|-----------|------|------|
| **Mode A**（默认） | 和 (cx,cy,w,h) 共用共享 decoder | HBB baseline，改动最小 | θ 解耦后 label 仍与空间耦合 |
| **Mode B** | 独立 Label Decoder | label 不受空间干扰 | 增加参数量 |

**Mode B 细节**（仅当角度拆分无法解决置信度低的问题时考虑）：
- 进一步将 decoder_layer_box 拆分为 decoder_layer_xywh（解码 cx,cy,w,h）和 decoder_layer_label（解码 label）
- Label Decoder 接收共享 decoder 的输出特征 + encoder memory 作为 cross-attn KV
- 使用 `Gate` 模块（`deim_utils.py:70`）门控混合空间特征与原始图形特征

### 3. 多源特征融合：Gate Fusion

`r` 不仅依赖于空间特征，也依赖语义特征。采用特征融合的方式，将每层 decoder_layer_xywh 的输出融合后，输入到每层 decoder_layer_angle 中。

**逐层对应**：第 i 层 Gate Fusion 融合的是第 i 层 XYWH decoder 的输出特征（L_i → GF_i）。

**GF1~GF4 二源融合**（无 encoder memory，因 layer_r 自带 cross-attn）：

| Gate Fusion | 源 | 位置 |
|------------|-----|------|
| **GF1** | L1 xywh 特征 + r 特征 | R1 与 R2 之间 |
| **GF2** | L2 xywh 特征 + r 特征 | R2 与 R3 之间 |
| **GF3** | L3 xywh 特征 + r 特征 | R3 与 R4 之间 |
| **GF4** | L4 xywh 特征 + r 特征 | R4 与 R5 之间 |

**无 GF0**：layer_r_0 内部已有 cross-attn 处理 encoder memory，不需要额外融合。

**无 GF5**：R5 是最后一层，之后无 R 层消费融合结果。

融合方式：
```
# GF1~GF4 二源融合（所有源都是 [B, 300, d]）
cat = concat([L_i_xywh_feat, r_feat])               # [B, 300, 2d]
w = softmax(MLP(cat))                                # [B, 300, 2]
fused = w_A·L_i_xywh_feat + w_B·r_feat
```

后续根据情况来决定是否使用较为复杂的 VMoE 或 SMoE 的方式。

### 4. 正交旋转注意力（Rotation-Rectified Orthogonal Attention）

参考 RIO-DETR 文献设计，调整可变注意力为 Rotation-Rectified Orthogonal Attention。具体来说就是将可变注意力模块的多头中 `head_idx >= len_head` 的旋转矩阵的角度加上 π/2：

```
H 个 attention head 分成两组：
  Heads 1 ~ H/2: 使用 θ 做旋转矩阵 R(θ)，采样轴向特征
  Heads H/2+1 ~ H: 使用 R(θ+π/2)，采样正交方向特征
```

**R 路径的参考点**：layer_r 的 MSDeformableAttention 接收 5-dof 参考点 `(cx,cy,w,h,r)`，进入 `elif reference_points.shape[-1] == 5:` 分支。其中 `r` 来自上一层的 `inter_ref_angle`，经旋转注意力采样轴向 + 正交方向特征。

**前提说明**：OBB_CODE_REVIEW.md #1 已修复旋转注意力的数学错误，当前实现是"先按半边尺寸缩放、再用 R(θ) 整体旋转"的正确实现。正交注意力在此基础上作为增强添加。

### 5. Multi-Angle Anchoring（angle_step=10°）

在生成 anchor-box 时，按照 `angle_step` 生成不同角度的 anchor-box。为避免生成的 anchor-box 较多，`angle_step` 默认取 10。

**anchor 格式为 `(cx,cy,w,h,ε,η)` 6-dof**（与 decoder 内部表示一致），生成流程：
1. 对每个角度 θ ∈ {0°, 10°, 20°, ..., 170°}
2. 构造 OBB `(cx,cy,w,h,θ)`
3. 经 `oriented_box_to_external_rect` 转换为 `(external_rect, vertex_offsets(ε,η))`
4. 拼接为 `(cx,cy,w,h,ε,η)` 并归一化到 [0,1]（sigmoid 空间）

```
angle_step = 10°  →  18 个角度 →  18 组 (ε,η)  →  18 个锚点/位置
total_anchors = Σ(H_i × W_i) × 18
queries = select_topk(anchors, k=300)  # 仍然只选 300
```

**避免边界 -inf**：`(ε,η)` 经 sigmoid 归一化到 [0,1]，`inverse_sigmoid` 在 0 和 1 边界无界。使用 `max(k * angle_step / 180.0, eps)` 确保 θ > 0（从而 (ε,η) 不退化），其中 eps 取 1e-6。

匈牙利匹配会自然选出每个 GT 对应的最佳角度锚点。若硬件不足，增大 angle_step（如 30°）。

### 6. LQE 输入

**[2026-06-27 新增]** LQE 在解耦后只用 XYWH 路径的 4 条分布 `(α,β,γ,δ)`，不含 angle 分布 `(ε,η)`。

参考 `deim_decoder.py:288`：
```python
scores = self.lqe_layers[i](scores, pred_corners)
```

解耦后改为：
```python
# pred_corners_xywh 只含 (α,β,γ,δ)，4*(reg_max+1) 维
scores = self.lqe_layers[i](scores, pred_corners_xywh)
```

R 路径不需要 LQE（angle 是连续值回归，不是分类）。

### 7. 空间→语义注意力门控

参考 DEIMv2 的 `Gate` 模式（`deim_utils.py:70-83`）：

```
f_spatial = shared_decoder_output     # 空间分支输出
f_semantic = label_or_angle_query     # 语义/角度分支的 query
gate_weights = σ(Linear(2d→2d)([f_spatial; f_semantic]))
modulated = gate1 * attn_out + gate2 * f_spatial
```

添加到 Label Decoder（Mode B）和 Angle Decoder 的自注意力或交叉注意力中。

## 配置参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `decouple_label` | `False` | 是否独立 label decoder（Mode B） |
| `decouple_angle` | `True` | 是否独立 angle decoder |
| `angle_step` | `10` | 角度锚点步长（度） |
| `fusion_method` | `"gated_softmax"` | 多源融合方式（TODO: `"moe"`） |
| `use_orthogonal_attn` | `True` | 是否使用正交旋转注意力 |
| `gate_spatial_semantic` | `True` | 是否使用 Gate 做空间→语义门控 |

## RIO-DETR 论文核心设计

- 解耦角度与位置查询（Content-Driven Angle Estimation）
- 旋转注意力分为轴向和正交两个方向采样（Rotation-Rectified Orthogonal Attention）
- 周期性优化（Decoupled Periodic Refinement + Shortest-Path Periodic Loss）

## MoE TODO

- 基于 Google V-MoE（`/home/cx/win_dir/thired/MoE/vmoe/vmoe/moe.py`）
- 在真实 DOTA 数据上对比 Gated Softmax Fusion vs MoE Fusion
- 评估指标：角度预测精度、MAL loss 收敛性

## 自检

- [x] 所有设计要点对应明确的代码改动位置
- [x] angle_step 有硬件降级方案（增大 step）
- [x] 双模式 label 解耦有明确的配置控制
- [x] MoE 作为独立 TODO，不影响 baseline 搭建
- [x] **[2026-06-26]** F9 修复：encoder memory 先经 cross-attn 汇聚到 300 query 位置
- [x] **[2026-06-26]** 角度表示回归原始设计：保持 ADR 方式 `(ε,η)` 作为 DFL 分布
- [x] **[2026-06-26]** refinement 回归原始设计：保持原有 OBB refinement 不变
- [x] **[2026-06-26]** 角度量纲设计完整说明（decoder 内 [0,1]，外部 [0,π]）
- [x] **[2026-06-26]** Gate Fusion 逐层对应（L_i → GF_i，不是只用最终层）
- [x] **[2026-06-27]** encoder memory 不额外注入 Gate Fusion（layer_r 自带 cross-attn）；删除 GF0 和 GF5
- [x] **[2026-06-27]** 角度表示统一：pre_angle_head 输出 (ε,η)，与 angle_head 一致；anchor 格式 (cx,cy,w,h,ε,η) 6-dof
- [x] **[2026-06-27]** R 参考点 6-dof (cx,cy,w,h,ε,η)，经 external_rect_to_oriented_box 转为 5-dof 后进 MSDeformableAttention
- [x] **[2026-06-27]** XYWH 参考点逐层链式传递（4-dof），R 参考点逐层链式传递（6-dof，含 ε,η）
- [x] **[2026-06-27]** 输出流程参考 deim_decoder.py:285-308，XYWH 和 R 路径分别输出后拼装
- [x] **[2026-06-27]** LQE 只用 (α,β,γ,δ)，不含 (ε,η)
- [x] **[2026-06-27]** 角度锚点用 `max(k*angle_step/180, eps)` 避免 -inf，不用 `(k+0.5)*...`
