# DEIMv2-OBB Decoder 解耦设计

> 创建日期：2026-06-25
> 目的：解决分类分数与 IoU 质量无关（MAL loss 不收敛）的问题，对 decoder 进行空间-语义-角度三层解耦
> 状态：设计阶段，待评审

## 背景

经过匈牙利匹配诊断（Q1=PASS, Q2=部分PASS, Q3=FLAG r=0.05）和合成数据集密度消融实验，确认：

1. 匹配 assignment 正确（100% 一对一），但分类分数与 IoU 几乎零相关
2. MAL loss 设计意图（分数与 IoU 正相关）无法实现，Comet 看板上 MAL loss 不下降且反复震荡
3. DEIMv2 的 HBB 模式已验证可行——(cx,cy,w,h,label) 共用 decoder 是正确的

**根因假设**：θ（角度/r）与 (cx,cy,w,h,label) 共用同一个 decoder layer 导致空间-语义-角度信息耦合。θ 同时具有几何和语义属性，需要一个独立的处理路径。参考 STD 的空间侧分支 + 渐进式预测，以及 RIO-DETR 的 Content-Driven θ 预测和正交旋转注意力。

## 总体架构

```mermaid
flowchart TD
    subgraph Input["输入准备"]
        ENC["encoder feats"]
        DN["get_contrastive_denoising_training_group<br/>dn_logits, dn_bbox, dn_mask"]
        ANCHOR["_get_decoder_input<br/>grid_w × grid_h × angle_step×N<br/>select_topk → ref_points_initial"]
    end

    subgraph Split["参考点拆分"]
        XYWH["ref_points[:,:,:4]<br/>(cx,cy,w,h)"]
        R_REF["ref_points[:,:,4]<br/>(r) × π"]
        SCALED["ref_points_initial_scaled<br/>concat(cx,cy,w,h,r)"]
        XYWH --> SCALED
        R_REF --> SCALED
    end

    subgraph XYWH_path["XYWH Decoder (6层)"]
        L0["layer_xywh_0: pre_bbox_head + score_head_0<br/>→ pre_bboxes, pre_scores"]
        L1["layer_xywh_1: box_head → pred_corners(αβγδ)<br/>distance2bbox_obb → inter_ref_bbox"]
        L2["layer_xywh_2: 重复"]
        L3["layer_xywh_3: 重复"]
        L4["layer_xywh_4: 重复"]
        L5["layer_xywh_5: 重复 → 最终 xywh + scores"]
        L0 --> L1 --> L2 --> L3 --> L4 --> L5
    end

    subgraph R_path["R Decoder (6层) + Gated Softmax Fusion @每层"]
        R0["layer_r_0: pre_angle_head → pre_angles"]
        GF0["Gated Softmax Fusion<br/>w_xywh·F_xywh + w_r·F_r + w_enc·F_enc"]
        R1["layer_r_1: angle_head → pred_angle(εη)"]
        GF1["Gated Softmax Fusion"]
        R2["layer_r_2: 重复"]
        GF2["Gated Softmax Fusion"]
        R3["layer_r_3: 重复"]
        GF3["Gated Softmax Fusion"]
        R4["layer_r_4: 重复"]
        GF4["Gated Softmax Fusion"]
        R5["layer_r_5: 重复 → 最终角度偏移"]
        GF5["Gated Softmax Fusion"]

        R0 --> GF0 --> R1 --> GF1 --> R2 --> GF2 --> R3 --> GF3 --> R4 --> GF4 --> R5 --> GF5
    end

    ENC --> L0
    ENC --> R0
    DN --> L0
    ANCHOR --> XYWH
    ANCHOR --> R_REF

    L5 -->|"xywh 特征"| GF0
    L5 -->|"xywh 特征"| GF1
    L5 -->|"xywh 特征"| GF2
    L5 -->|"xywh 特征"| GF3
    L5 -->|"xywh 特征"| GF4
    L5 -->|"xywh 特征"| GF5

    ENC -->|"encoder memory"| GF0
    ENC -->|"encoder memory"| GF1

    L1 -->|"inter_ref_bbox 作参考点"| L2
    L1 -->|"inter_ref_bbox 作参考点"| R1

    subgraph Output["输出"]
        SCORES["scores (分类)"]
        BBOX["pred_boxes (cx,cy,w,h) + 角度偏移"]
    end

    L5 --> SCORES
    L5 --> BBOX
    R5 --> BBOX
```

### 架构关键决策（已确认）

| 决策 | 说明 |
|------|------|
| **Gate Fusion 频率** | 每个 R decoder layer 都发生（共 6 次） |
| **distance2bbox_obb** | 仅在 xywh 路径计算，`inter_ref_bbox` 传给 xywh 下一层和 R 路径 |
| **Encoder memory** | 同时注入 xywh decoder（cross-attn）和 Gated Softmax Fusion（第三路源） |
| **Mode B** | 暂不实现，先做 Mode A（label 留在 xywh 路径） |
| **参数量** | decoder 层数翻倍（6→12），在预期之内 |

## 设计要点

### 1. 保留 DEIMv2 Refinement

- 共享 decoder 的 (cx,cy,w,h) refinement（D-FINE ADR + DEIM gateway）不改动
- 这个 pipeline 在 HBB 已验证，不需要重新设计

### 2. Label 解耦：双模式（共用 / 独立）

通过配置参数 `decouple_label: true/false` 控制：

| 模式 | label 处理 | 优点 | 缺点 |
|------|-----------|------|------|
| **Mode A**（默认） | 和 (cx,cy,w,h) 共用共享 decoder | HBB baseline，改动最小 | θ 解耦后 label 仍与空间耦合 |
| **Mode B** | 独立 Label Decoder | label 不受空间干扰 | 增加参数量 |

**Mode B 细节**：
- Label Decoder 接收共享 decoder 的输出特征 + encoder memory 作为 cross-attn KV
- 使用 `Gate` 模块（`deim_utils.py:70`）门控：`Gate(f_spatial, f_label)` 融合共享 decoder 输出和 label decoder 内部特征
- Label Decoder 不需要 refinement（label 是离散的），只需要足够的 self-attn + cross-attn 层完成语义推理

### 3. 多源特征融合：Gated Softmax Fusion

三路径：
- **src_A**: Encoder memory（原始视觉特征，纹理/边缘/形状）
- **src_B**: 共享 decoder 的 (cx,cy,w,h) 输出特征（位置和尺寸信息）
- **src_C**: label 输出特征（类别语义，Mode B 时有效；Mode A 时只用 src_A + src_B）

融合方式：
```
cat = concat(src_A, src_B, src_C)          # [B, N, 3d]
w = softmax(Linear(d→3d)→ReLU→Linear(3d→3))  # 三个 token 级权重
fused = w_A·A + w_B·B + w_C·C
```

TODO：后续在真实 DOTA 数据上对比 MoE 融合（Google V-MoE，`/home/cx/win_dir/thired/MoE/vmoe/vmoe/moe.py`）。

### 4. Angle Decoder：Rotation-Rectified Orthogonal Attention

融入 RIO-DETR 的正交旋转注意力：

```
H 个 attention head 分成两组：
  Heads 1 ~ H/2: 使用 θ 做旋转矩阵 R(θ)，采样轴向特征
  Heads H/2+1 ~ H: 使用 R(θ+π/2)，采样正交方向特征
```

当前 DEIMv2-OBB 的旋转交叉注意力只沿长轴方向采样 → 容易特征坍塌（所有采样点在一条线上）。正交注意力的第二组 head 补充了垂直于长轴的信息。

### 5. Multi-Angle Anchoring（angle_step=10°）

当前每个网格位置只有 1 个角度锚点（θ=π/2）。改为每个位置生成 18 个角度锚点：

```
angle_step = 10°  →  θ ∈ {0°, 10°, 20°, ..., 170°}  →  18 个锚点/位置
total_anchors = Σ(H_i × W_i) × 18
queries = select_topk(anchors, k=300)  # 仍然只选 300
```

匈牙利匹配会自然选出每个 GT 对应的最佳角度锚点。若硬件不足，增大 angle_step（如 30°）。

### 6. 空间→语义注意力门控

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
| `decouple_label` | `False` | 是否独立 label decoder |
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

- [ ] 所有设计要点对应明确的代码改动位置
- [ ] angle_step 有硬件降级方案（增大 step）
- [ ] 双模式 label 解耦有明确的配置控制
- [ ] MoE 作为独立 TODO，不影响 baseline 搭建
