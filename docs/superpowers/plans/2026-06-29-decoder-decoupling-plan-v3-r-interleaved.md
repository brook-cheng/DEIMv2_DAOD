# DEIMv2-OBB Decoder 解耦实施计划 v3 — R Decoder 交错修正

> **来源**: 基于 v2 计划修订，修正 R Decoder 不应作为独立 `forward_r_decoder` 方法，而应嵌入 `TransformerDecoder.forward` 主循环中与 XYWH 逐层交错执行
> **设计文档**: `docs/superpowers/specs/2026-06-25-decoder-decoupling-design.md`
> **创建日期**: 2026-06-29

**Goal:** 将 R Decoder 嵌入 `TransformerDecoder.forward` 主循环，与 XYWH 路径逐层交错执行，符合设计文档 mermaid 图的 L0→R0→L1→GF1→R1→L2→GF2→R2→...→L5→R5 流程。

**Architecture:** 在 `TransformerDecoder.forward` 的 `for i, layer in enumerate(self.layers)` 循环内，每执行完一层 XYWH decoder layer 后，紧接着执行对应层的 R decoder layer。R decoder 的 cross-attn 使用当前层 XYWH 输出的 (cx,cy,w,h) + R 的 (ε,η) 组合的 5-dof 参考点。Gate Fusion 在 R decoder layer 后、下一层之前的间隙执行。

**Tech Stack:** PyTorch, DEIMv2 engine

## Global Constraints

- 不破坏 HBB 模式（`box_mode="hbb"` 时行为不变）
- `decouple_angle` 默认 `False`，仅在 `box_mode="obb" and decouple_angle=True` 时激活
- R decoder 组件存放在 `TransformerDecoder` 实例上（不是 `DEIMTransformer`），因为主循环在 `TransformerDecoder.forward` 中

---

## 核心改动概述

### 当前问题
- `forward_r_decoder` 是独立方法，在 XYWH decoder 全部 6 层跑完后才执行
- R decoder 使用 `out_bboxes`（XYWH 最终输出）构造参考点，不是逐层的实时输出
- 不符合设计的逐层交错 + Gate Fusion 模式

### 修正方案
- **删除** `DEIMTransformer.forward_r_decoder` 方法
- **删除** `DEIMTransformer._convert_6dof_to_5dof` 方法
- **删除** `DEIMTransformer.forward` 中的 R decoder 调用和输出转换块
- **移动** R decoder 组件（r_layers, dec_angle_head, integral_angle, gate_fusions, pre_angle_head）从 `DEIMTransformer.__init__` 到 `TransformerDecoder.__init__`
- **扩展** `TransformerDecoder.forward` 签名，接收 R decoder 所需的额外参数
- **在** `TransformerDecoder.forward` 主循环内添加 R decoder 交错逻辑
- **在** `TransformerDecoder.forward` 返回时直接输出 5-dof（转换在循环内完成）

---

## Task A: 移动 R decoder 组件到 TransformerDecoder

**Files:** Modify `engine/deim/deim_decoder.py`

**当前状态:** R decoder 组件在 `DEIMTransformer.__init__`（约 line 520-560）中创建
**目标状态:** R decoder 组件在 `TransformerDecoder.__init__`（约 line 132-175）中创建

### 需要移动的组件

| 组件 | 当前位置 | 目标位置 |
|------|---------|---------|
| `self.integral_xywh` | DEIMTransformer.__init__ | TransformerDecoder.__init__ |
| `self.integral_angle` | DEIMTransformer.__init__ | TransformerDecoder.__init__ |
| `self.pre_angle_head` | DEIMTransformer.__init__ | TransformerDecoder.__init__ |
| `self.dec_angle_head` | DEIMTransformer.__init__ | TransformerDecoder.__init__ |
| `self.r_layers` | DEIMTransformer.__init__ | TransformerDecoder.__init__ |
| `self.gate_fusions` | DEIMTransformer.__init__ | TransformerDecoder.__init__ |

### TransformerDecoder.__init__ 签名扩展

```python
def __init__(
    self,
    hidden_dim,
    decoder_layer,
    decoder_layer_wide,
    num_layers,
    num_head,
    reg_max,
    reg_scale,
    up,
    eval_idx=-1,
    layer_scale=2,
    act="relu",
    num_reg_dist=4,
    box_mode="hbb",
    decouple_angle=False,
    num_r_layers=6,        # 新增
    angle_step=10,          # 新增（仅传递，不在 TransformerDecoder 使用）
):
```

### 在 TransformerDecoder.__init__ 中创建 R decoder 组件

当 `decouple_angle=True` 时：

```python
if self.decouple_angle:
    from .gated_fusion import GatedSoftmaxFusion

    self.num_reg_dist_xywh = 4
    self.num_reg_dist_angle = 2

    self.integral_xywh = Integral(self.reg_max, self.num_reg_dist_xywh)
    self.integral_angle = Integral(self.reg_max, self.num_reg_dist_angle)

    self.pre_angle_head = MLP(hidden_dim, hidden_dim, 2, 3, act=act)

    self.dec_angle_head = nn.ModuleList([
        MLP(hidden_dim, hidden_dim,
            self.num_reg_dist_angle * (self.reg_max + 1), 3, act=act)
        for _ in range(self.num_r_layers)
    ])

    r_layer_template = self.layers[-1]
    self.r_layers = nn.ModuleList([
        copy.deepcopy(r_layer_template) for _ in range(self.num_r_layers)
    ])

    self.gate_fusions = nn.ModuleList([
        GatedSoftmaxFusion(d_model=hidden_dim, n_sources=2, hidden_dim=128)
        for _ in range(self.num_r_layers - 1)
    ])
```

### 从 DEIMTransformer.__init__ 中删除

删除约 line 520-560 的整个 `if self.box_mode == "obb" and self.decouple_angle:` 块（创建 R decoder 组件的部分）。

### DEIMTransformer.__init__ 的 decouple 分支保留

保留创建 `self.dec_bbox_head`（4-dist）、`self.pre_bbox_head`（4-dof）的分支，因为这些是 XYWH 路径的 heads，属于 DEIMTransformer 级别（通过参数传递给 TransformerDecoder.forward）。

---

## Task B: 扩展 TransformerDecoder.forward 签名

**Files:** Modify `engine/deim/deim_decoder.py`

### 当前签名

```python
def forward(
    self,
    target,
    ref_points_unact,
    memory,
    spatial_shapes,
    bbox_head,        # dec_bbox_head
    score_head,       # dec_score_head
    query_pos_head,
    pre_bbox_head,
    integral,         # integral_xywh 或 integral
    up,
    reg_scale,
    attn_mask=None,
    memory_mask=None,
    dn_meta=None,
):
```

### 新增参数（仅 decouple_angle=True 时使用）

```python
def forward(
    self,
    target,
    ref_points_unact,
    memory,
    spatial_shapes,
    bbox_head,
    score_head,
    query_pos_head,
    pre_bbox_head,
    integral,
    up,
    reg_scale,
    attn_mask=None,
    memory_mask=None,
    dn_meta=None,
    pre_angle_head=None,  # 新增：R 路径初始角度 head
):
```

注意：`dec_angle_head`、`integral_angle`、`r_layers`、`gate_fusions` 已经是 `TransformerDecoder` 的属性（Task A 创建），不需要通过参数传递。只有 `pre_angle_head` 需要从 `DEIMTransformer` 传递，因为它属于 DEIMTransformer 级别（和 `pre_bbox_head` 平行）。

但实际上 `pre_angle_head` 也可以放在 `TransformerDecoder` 上。考虑到 `pre_bbox_head` 是通过参数传递的（保持原始设计），`pre_angle_head` 也应通过参数传递以保持一致。

---

## Task C: 在 TransformerDecoder.forward 主循环中添加 R decoder 交错逻辑

**Files:** Modify `engine/deim/deim_decoder.py`

这是最核心的改动。当前主循环结构：

```
for i, layer in enumerate(self.layers):
    1. ref_points_input = ref_points_detach.unsqueeze(2)
    2. output = layer(output, ref_points_input, value, ...)     ← XYWH decoder layer
    3. if i == 0: pre_bboxes, pre_scores, ref_points_initial
    4. pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach  ← XYWH FDR
    5. inter_ref_bbox = distance2bbox(...)                      ← XYWH 解码
    6. if training: scores, LQE, collect outputs
    7. pred_corners_undetach = pred_corners
    8. ref_points_detach = inter_ref_bbox.detach()
    9. output_detach = output.detach()
```

### 修正后的主循环结构（decouple_angle=True 时）

```
初始化 R 路径状态:
  r_output = target  (R 的初始 query content，与 XYWH 相同)
  r_output_detach = 0
  pred_corners_undetach_angle = 0
  ref_angle = 0.5 * ones  (初始 ε,η)
  ref_angle_initial = None

for i, layer in enumerate(self.layers):
    === XYWH 路径（与当前相同）===
    1. ref_points_input = ref_points_detach.unsqueeze(2)
    2. output = layer(output, ref_points_input, value, ...)     ← XYWH decoder layer
    3. if i == 0: pre_bboxes, pre_scores, ref_points_initial
    4. pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
    5. inter_ref_bbox = distance2bbox(ref_points_initial[:,:,:4], integral_xywh(pred_corners), reg_scale)
       inter_ref_bbox = cat([inter_ref_bbox, ref_points_detach[:,:,4:]])  ← 保持角度不变
    6. if training: scores, LQE(只用 αβγδ), collect XYWH outputs

    === R 路径（新增，交错执行）===
    7. 构造 R 路径的 5-dof 参考点:
       xywh_current = inter_ref_bbox[:,:,:4]  ← 当前层 XYWH 输出的 (cx,cy,w,h)
       ext_rect = cat([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
       vertex_offsets = cat([ref_angle[:,:,0:1], ref_angle[:,:,1:2]])
       obb_5dof = external_rect_to_oriented_box(ext_rect, vertex_offsets)  ← θ ∈ [0,π]
       ref_5dof = cat([obb_5dof[:,:,:4], obb_5dof[:,:,4:5] / π])  ← θ ∈ [0,1]
    8. r_ref_points_input = ref_5dof.unsqueeze(2)
       r_query_pos_embed = query_pos_head(ref_5dof).clamp(min=-10, max=10)
    9. r_output = self.r_layers[i](r_output, r_ref_points_input, value, spatial_shapes, attn_mask, r_query_pos_embed)
       ← R decoder layer (内部 cross-attn 处理 encoder memory)
    10. if i == 0:
        pre_angle = sigmoid(pre_angle_head(r_output) + inverse_sigmoid(ref_angle))
        ref_angle_initial = pre_angle.detach()
        ref_angle = ref_angle_initial
    11. pred_corners_angle = self.dec_angle_head[i](r_output + r_output_detach) + pred_corners_undetach_angle
    12. angle_delta = self.integral_angle(pred_corners_angle, project)
    13. ref_angle = ref_angle + angle_delta * reg_scale
        ref_angle = ref_angle.clamp(1e-6, 1-1e-6)
    14. if training: collect R outputs (ref_angle, pred_corners_angle, ref_angle_initial)

    === Gate Fusion（R1~R4 之间，不含 R0 前和 R5 后）===
    15. if i < len(self.gate_fusions):
        r_output = self.gate_fusions[i]([output, r_output], query=r_output)
        ← 融合当前层 XYWH 特征 (output) + R 特征 (r_output)

    === 更新两路径状态 ===
    16. pred_corners_undetach = pred_corners
        pred_corners_undetach_angle = pred_corners_angle
    17. ref_points_detach = inter_ref_bbox.detach()     ← XYWH 参考点更新
        ref_angle = ref_angle.detach()                  ← R 参考点更新
    18. output_detach = output.detach()
        r_output_detach = r_output.detach()

    === 输出转换（在循环内完成，返回 5-dof）===
    19. inter_ref_bbox_combined = cat([inter_ref_bbox[:,:,:4], ref_angle])  ← 6-dof (cx,cy,w,h,ε,η)
        → 转换为 5-dof (cx,cy,w,h,θ) 通过 external_rect_to_oriented_box
        → θ ∈ [0,π] / π → [0,1]
        dec_out_bboxes.append(inter_ref_bbox_5dof)
    20. dec_out_pred_corners.append(cat([pred_corners, pred_corners_angle]))  ← 6 条分布
    21. dec_out_refs.append(cat([ref_points_initial[:,:,:4], ref_angle_initial_or_current]))  ← 6-dof → 5-dof
```

### 关键设计点

1. **R decoder 的 query content (r_output)**：初始化为 `target`（与 XYWH 相同），后续通过 Gate Fusion 更新
2. **R decoder 的参考点**：由当前层 XYWH 输出的 (cx,cy,w,h) + 当前 (ε,η) 组合，经 `external_rect_to_oriented_box` 转为 5-dof (cx,cy,w,h,θ)
3. **Gate Fusion 的输入**：XYWH 的 `output`（当前层特征） + R 的 `r_output`（当前层特征）
4. **Gate Fusion 的输出**：更新 `r_output`，作为下一层 R decoder 的输入
5. **输出转换在循环内完成**：不再需要 `DEIMTransformer.forward` 中的后处理转换
6. **`value` 共享**：XYWH 和 R decoder 共享同一个 `value`（encoder memory 处理后）

### HBB 兼容性

当 `decouple_angle=False` 时，R 路径的所有代码不执行，主循环与当前完全相同。

---

## Task D: 修改 DEIMTransformer.forward

**Files:** Modify `engine/deim/deim_decoder.py`

### 删除

1. `forward_r_decoder` 方法（约 60 行）
2. `_convert_6dof_to_5dof` 方法（约 15 行）
3. `forward` 中 R decoder 调用 + 输出转换块（约 15 行，在 `self.decoder()` 返回后）
4. `forward` 中 `out_corners` padding 块（约 5 行）

### 修改 self.decoder() 调用

```python
out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = (
    self.decoder(
        init_ref_contents,
        init_ref_points_unact,
        memory,
        spatial_shapes,
        self.dec_bbox_head,
        self.dec_score_head,
        self.query_pos_head,
        self.pre_bbox_head,
        self.integral_xywh if (self.box_mode == "obb" and self.decouple_angle) else self.integral,
        self.up,
        self.reg_scale,
        attn_mask=attn_mask,
        memory_mask=memory_mask if hasattr(self, '_memory_mask') else None,
        dn_meta=dn_meta,
        pre_angle_head=self.pre_angle_head if (self.box_mode == "obb" and self.decouple_angle) else None,
    )
)
```

注意：`pre_angle_head` 从 `DEIMTransformer` 传递给 `TransformerDecoder.forward`，因为它和 `pre_bbox_head` 平行，都是 DEIMTransformer 级别的组件。其余 R decoder 组件（`r_layers`, `dec_angle_head`, `integral_angle`, `gate_fusions`）已经是 `TransformerDecoder` 的属性。

### θ 量纲转换

当前在 `DEIMTransformer.forward` 中有一段 `out_bboxes[..., 4:] * torch.pi` 的转换。由于修正后 `TransformerDecoder.forward` 已经返回 5-dof (cx,cy,w,h,θ) 且 θ ∈ [0,1]，这段转换保持不变。

---

## Task E: 更新 LQE 创建逻辑

**Files:** Modify `engine/deim/deim_decoder.py`

当前 LQE 在 `TransformerDecoder.__init__` 中创建，使用 `num_reg_dist` 参数。

当 `decouple_angle=True` 时：
- `num_reg_dist` 传入为 4（从 `DEIMTransformer.__init__` 传入）
- LQE 使用 4-dist（只有 αβγδ），符合设计

这已经在当前实现中正确处理，不需要改动。

---

## 测试验证

### Smoke Test

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && timeout 180 python3 -c "
import sys, torch; sys.path.insert(0, '.')
from engine.core import YAMLConfig
from engine.solver import TASKS
cfg = YAMLConfig('configs/custom_obb/deimv2_obb_decouple.yml')
solver = TASKS[cfg.yaml_cfg['task']](cfg)
solver.train()
samples, targets = next(iter(solver.train_dataloader))
device = torch.device('cuda')
samples = samples.to(device)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
outputs = solver.model(samples, targets=targets)
print(f'Forward OK: pred_logits {outputs[\"pred_logits\"].shape}, pred_boxes {outputs[\"pred_boxes\"].shape}')
loss_dict = solver.criterion(outputs, targets, epoch=0)
loss = sum(loss_dict.values())
loss.backward()
print(f'Backward OK: total_loss = {loss.item():.4f}')
print('Smoke test PASS')
"
```

### HBB 兼容性

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from engine.core import YAMLConfig
cfg = YAMLConfig('configs/custom_obb/synthetic_configs/synthetic_exp_020.yml')
print('HBB config loads OK')
"
```

---

## 实施顺序

```
Task A: 移动 R decoder 组件到 TransformerDecoder.__init__
  ↓
Task B: 扩展 TransformerDecoder.forward 签名（加 pre_angle_head 参数）
  ↓
Task C: 在 TransformerDecoder.forward 主循环中添加 R decoder 交错逻辑
  ↓
Task D: 修改 DEIMTransformer.forward（删除 forward_r_decoder，修改 self.decoder() 调用）
  ↓
Task E: 确认 LQE 逻辑（可能不需要改动）
  ↓
Smoke Test + HBB 兼容性验证
```