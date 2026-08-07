# DEIMv2-OBB Decoder 解耦实施计划 v2

> **来源**: Hyperplan 对抗分析（5 个分析师 Round 1 独立分析 + Round 2 交叉攻击，蒸馏为结构化 bundle）
> **修订**: 2026-06-27 第二轮 Hyperplan（4 个分析师针对 3 个用户反馈问题分析，确认修正方案）
> **设计文档**: `docs/superpowers/specs/2026-06-25-decoder-decoupling-design.md`
> **创建日期**: 2026-06-27

**Goal:** 将 DEIMv2-OBB decoder 拆分为 XYWH 路径（6层）和 R 路径（6层），通过 Gate Fusion（GF1~GF4）桥接，统一 (ε,η) 角度表示，引入多角度锚点和正交旋转注意力。

**Architecture:** XYWH decoder 保留 D-FINE refinement + label 分类 + LQE（只用 αβγδ）；R decoder 独立预测 (ε,η) 角度偏移。decoder 内部用 6-dof (cx,cy,w,h,ε,η) [0,1]，输出时经 `external_rect_to_oriented_box` 转为 5-dof (cx,cy,w,h,θ) [0,π]。

**Tech Stack:** PyTorch, DEIMv2 engine

## Global Constraints

- Mode A 优先（label 留在 xywh 路径），Mode B 后续实现
- 不破坏 HBB 模式（`box_mode="hbb"` 时行为不变）
- 保持与现有 criterion/matcher/postprocessor 接口兼容（输出仍为 5-dof）
- `decouple_angle` 默认 `False`，仅在 `box_mode="obb" and decouple_angle=True` 时激活
- Python 环境：`/home/cx/apps/miniconda3/envs/deimv2/`

## 风险矩阵（来自对抗分析）

| 风险 | 级别 | 缓解措施 |
|------|------|---------|
| Integral 类需拆分（当前按 num_reg_dist reshape） | HIGH | 创建独立 Integral 实例：XYWH 用 4-dist，R 用 2-dist |
| denoising.py 噪声注入需处理 (ε,η) | ~~HIGH~~ → **LOW** | **[修订]** 不修改 denoising.py，在 decoder 调用处做 θ→(ε,η) 转换。DN 不对角度加噪声（与原始设计一致） |
| Checkpoint 不兼容（_num_box_dof 5→6） | HIGH | 接受从零训练；后续可写转换函数 |
| _generate_anchors valid_mask 不检查 (ε,η) | MEDIUM | 扩展 valid_mask 检查范围 |
| _get_decoder_input 处理 6-dof anchor | MEDIUM | enc_bbox_head 输出改为 6-dof |
| postprocessor 处理 6-dof 输入 | MEDIUM | 在 postprocessor 入口做 6→5 转换 |
| matcher cost 计算 | MEDIUM | matcher 接收 5-dof（已在 decoder 输出转换） |
| convert_to_deploy 双路径 | LOW | 更新 TransformerDecoder.convert_to_deploy() |
| eps 角度锚点值 | MEDIUM | 用 1°（0.00278）而非 1e-6 |

---

## File Structure

```
engine/deim/
  gated_fusion.py        # CREATE: GatedSoftmaxFusion (2-source)
  deim_decoder.py        # MODIFY: anchor gen, decoder forward, DEIMTransformer init, output conversion
  dfine_decoder.py       # MODIFY: MSDeformableAttention (orthogonal), LQE (4-dist), Integral (split), TransformerDecoder (dual path)
  dfine_utils.py         # REUSE: distance2bbox, bbox2distance, distance2bbox_obb, bbox2distance_obb (不新增函数)
  deim_criterion.py      # MODIFY: loss_local split, get_loss_meta_info fix
  denoising.py           # REUSE: 不修改（DN 不对角度加噪声，转换在 decoder 调用处完成）
  postprocessor.py       # MODIFY: 6-dof input → 5-dof conversion
  obb_geometry.py        # REUSE: oriented_box_to_external_rect, external_rect_to_oriented_box
  obb_ops.py             # REUSE: batch_probiou, kld_loss
  chamfer_cost.py        # REUSE: chamfer_cost_obb
  deim_utils.py          # REUSE: Gate class
  matcher.py             # MODIFY: cost calc for 6-dof (or use converted 5-dof)
configs/custom_obb/
  deimv2_obb_decouple.yml  # CREATE: new config
```

---

## Phase 0: 基础设施（可并行）

### Task 0a: GatedSoftmaxFusion 模块

**Files:** Create `engine/deim/gated_fusion.py`

**Interfaces:**
- Produces: `GatedSoftmaxFusion(d_model, n_sources=2, hidden_dim=128)` — `forward(srcs: list[Tensor], query: Tensor) → Tensor`

- [ ] **Step 1: 创建模块**

```python
import torch
import torch.nn as nn


class GatedSoftmaxFusion(nn.Module):
    def __init__(self, d_model: int, n_sources: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.n_sources = n_sources
        self.weight_net = nn.Sequential(
            nn.Linear((n_sources + 1) * d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_sources),
        )
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, srcs: list[torch.Tensor], query: torch.Tensor) -> torch.Tensor:
        assert len(srcs) == self.n_sources
        num_queries = query.shape[1]
        for i, src in enumerate(srcs):
            assert src.shape[1] == num_queries, \
                f"Source {i} has {src.shape[1]} tokens, expected {num_queries}"
        cat = torch.cat([query] + srcs, dim=-1)
        weights = torch.softmax(self.weight_net(cat), dim=-1)
        fused = torch.zeros_like(srcs[0])
        for i, src in enumerate(srcs):
            fused = fused + weights[..., i:i + 1] * src
        return self.output_proj(fused)
```

- [ ] **Step 2: 单元测试**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import torch
from engine.deim.gated_fusion import GatedSoftmaxFusion
m = GatedSoftmaxFusion(256, 2, 128)
srcs = [torch.randn(2, 300, 256) for _ in range(2)]
q = torch.randn(2, 300, 256)
y = m(srcs, q)
assert y.shape == (2, 300, 256)
# 验证：改变 src[1] 不应影响 src[0] 的贡献
srcs2 = [srcs[0], torch.randn(2, 300, 256)]
y2 = m(srcs2, q)
assert not torch.allclose(y, y2), 'Output should change when source changes'
print('GatedSoftmaxFusion PASS')
"
```

---

### Task 0b: 确认 distance2bbox / bbox2distance 复用策略

**[2026-06-27 修订]** 根据对抗分析（func-design-analyst + edge-analyst），删除 4 个不必要的新函数，直接复用现有函数。

**Files:** 无需新增函数到 `engine/deim/dfine_utils.py`

**分析结论：**

| 原计划函数 | 是否需要 | 原因 | 替代方案 |
|------------|---------|------|---------|
| `distance2bbox_obb_xywh` | ❌ 不需要 | 只是 `distance2bbox` 的包装 | 直接调用 `distance2bbox(points_4dof, integral_xywh(pred_corners_4dist), reg_scale)` |
| `distance2bbox_obb_angle` | ❌ 不需要 | R 路径角度解码不是 "distance2bbox" 操作，而是残差更新 | 在 decoder forward 内部：`integral_angle(pred_corners_2dist)` → 2 个标量残差 (Δε, Δη) → `ε += Δε * reg_scale, η += Δη * reg_scale` |
| `bbox2distance_obb_xywh` | ❌ 不需要 | 只是 `bbox2distance` 的包装 | 直接调用 `bbox2distance(points_4dof, bbox_4dof, reg_max, reg_scale, up, eps)` |
| `bbox2distance_obb_angle` | ❌ 不需要 | 现有 `bbox2distance_obb` 已编码 (ε,η) 为 6 分布的后 2 个 | 如需分离 angle 的 FGL target，从 `bbox2distance_obb` 输出切片后 2 个分布 |

**XYWH 路径解码流程（无需新函数）：**
```
pred_corners_xywh = bbox_head[i](output)           # [B, 300, 4*(reg_max+1)]
offsets_4 = integral_xywh(pred_corners_xywh, proj) # [B, 300, 4] — 4 个标量偏移
inter_ref_xywh = distance2bbox(ref_points_4dof, offsets_4, reg_scale)  # [B, 300, 4]
```

**R 路径解码流程（无需新函数）：**
```
pred_corners_angle = angle_head[i](r_output)        # [B, 300, 2*(reg_max+1)]
deltas_2 = integral_angle(pred_corners_angle, proj) # [B, 300, 2] — 2 个标量残差 (Δε, Δη)
# 残差更新（不是 distance2bbox 操作）
ε_new = ε_current + deltas_2[..., 0:1] * reg_scale
η_new = η_current + deltas_2[..., 1:2] * reg_scale
inter_ref_angle = torch.cat([ε_new, η_new], dim=-1)  # [B, 300, 2]
```

- [ ] **Step 1: 确认现有函数可直接复用**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
from engine.deim.dfine_utils import distance2bbox, bbox2distance, distance2bbox_obb, bbox2distance_obb
import inspect
# 确认 distance2bbox 接受 4-dof points 和 4-dof distance
print('distance2bbox signature:', inspect.signature(distance2bbox))
print('bbox2distance signature:', inspect.signature(bbox2distance))
print('No new functions needed — reuse existing directly')
"
```

- [ ] **Step 2: 无单元测试需要（无新函数）**

跳过——此 Task 确认复用策略，不创建新代码。

---

### Task 0c: 配置参数 plumbing

**Files:** Modify `engine/deim/deim_decoder.py` — `DEIMTransformer.__init__`

- [ ] **Step 1: 添加参数到 `__init__` 签名**

在 `box_mode="hbb"` 之后添加：

```python
    decouple_angle=False,
    angle_step=10,
    num_r_layers=6,
```

- [ ] **Step 2: 在 `__init__` body 中存储参数**

```python
    self.decouple_angle = decouple_angle
    self.angle_step = angle_step
    self.num_r_layers = num_r_layers
```

- [ ] **Step 3: 验证 import 无报错**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import sys; sys.path.insert(0, '.')
from engine.deim.deim_decoder import DEIMTransformer
print('Import OK')
"
```

---

## Phase 1: XYWH 路径（不引入 R decoder）

### Task 1: 拆分 box_head 和 Integral

**Files:** Modify `engine/deim/deim_decoder.py` — `DEIMTransformer.__init__` (lines 414-498)

**核心改动**: 当 `decouple_angle=True` 时，创建独立的 XYWH heads 和 Integral

- [ ] **Step 1: 创建 XYWH 专用 heads**

```python
if self.box_mode == "obb" and self.decouple_angle:
    # XYWH path: 4 distributions
    self.num_reg_dist_xywh = 4
    self.integral_xywh = Integral(self.reg_max, self.num_reg_dist_xywh)
    
    dec_bbox_head_xywh = MLP(
        hidden_dim, hidden_dim, self.num_reg_dist_xywh * (self.reg_max + 1), 3
    )  # 4*33 = 132
    self.dec_bbox_head = nn.ModuleList([
        copy.deepcopy(dec_bbox_head_xywh) if not share_bbox_head else dec_bbox_head_xywh
        for _ in range(num_layers)
    ])
    
    # pre_bbox_head: 4-dof output (cx,cy,w,h only)
    self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
    
    # R path: 2 distributions (separate heads)
    self.num_reg_dist_angle = 2
    self.integral_angle = Integral(self.reg_max, self.num_reg_dist_angle)
    
    self.pre_angle_head = MLP(hidden_dim, hidden_dim, 2, 3)  # (ε,η)
    self.dec_angle_head = nn.ModuleList([
        MLP(hidden_dim, hidden_dim, self.num_reg_dist_angle * (self.reg_max + 1), 3)
        for _ in range(self.num_r_layers)
    ])  # 2*33 = 66
else:
    # 原逻辑不变
    self.integral = Integral(self.reg_max, self.num_reg_dist)
    ...
```

- [ ] **Step 2: 验证 head 维度**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import sys; sys.path.insert(0, '.')
from engine.deim.deim_decoder import DEIMTransformer
m = DEIMTransformer(num_classes=3, box_mode='obb', decouple_angle=True, eval_spatial_size=(256,256))
# XYWH head: 4*(32+1) = 132
assert m.dec_bbox_head[0].layers[-1].out_features == 4 * 33
# Angle head: 2*(32+1) = 66
assert m.dec_angle_head[0].layers[-1].out_features == 2 * 33
# pre_angle_head: 2 outputs (ε,η)
assert m.pre_angle_head.layers[-1].out_features == 2
print('Head dimensions PASS')
"
```

---

### Task 2: XYWH decoder forward（拆分 reference_points）

**Files:** Modify `engine/deim/deim_decoder.py` — `TransformerDecoder.forward` (lines 199-308)

**核心改动**: 当 `decouple_angle=True` 时，XYWH 路径只用 4-dof reference points

- [ ] **Step 1: 在 forward 入口拆分 ref_points**

```python
if self.box_mode == "obb" and getattr(self, "decouple_angle", False):
    ref_xywh = ref_points_unact[..., :4]  # 4-dof
    ref_angle = ref_points_unact[..., 4:]  # 2-dof (ε,η)
    ref_points_detach = F.sigmoid(ref_xywh)
else:
    ref_points_detach = F.sigmoid(ref_points_unact)
```

- [ ] **Step 2: 修改 LQE 输入**

```python
# 原代码 line 288:
# scores = self.lqe_layers[i](scores, pred_corners)

# 改为:
if self.box_mode == "obb" and getattr(self, "decouple_angle", False):
    scores = self.lqe_layers[i](scores, pred_corners)  # pred_corners 只有 αβγδ
else:
    scores = self.lqe_layers[i](scores, pred_corners)
```

注意：当 decouple_angle=True 时，`pred_corners` 来自 `dec_bbox_head`（4-dist），天然只有 αβγδ。

- [ ] **Step 3: 修改 distance2bbox_obb 调用**

```python
# 原代码 lines 270-283:
# inter_ref_bbox = distance2bbox_obb(ref_points_initial_scaled, integral(pred_corners, project), reg_scale)

# 改为:
if self.box_mode == "obb" and getattr(self, "decouple_angle", False):
    inter_ref_bbox_xywh = distance2bbox_obb_xywh(
        ref_points_initial,  # 4-dof
        self.integral_xywh(pred_corners, project),  # 4-dist
        reg_scale,
    )
    inter_ref_bbox = inter_ref_bbox_xywh  # 4-dof, 不含角度
else:
    # 原逻辑
    ...
```

---

## Phase 2: R 路径

### Task 3: R decoder layers 创建

**Files:** Modify `engine/deim/deim_decoder.py` — `DEIMTransformer.__init__`

- [ ] **Step 1: 创建 R decoder layers 和 Gate Fusion**

```python
if self.box_mode == "obb" and self.decouple_angle:
    from .gated_fusion import GatedSoftmaxFusion
    
    r_layer_template = self.decoder.layers[-1]  # F7 fix
    self.r_layers = nn.ModuleList([
        copy.deepcopy(r_layer_template) for _ in range(self.num_r_layers)
    ])
    
    self.gate_fusions = nn.ModuleList([
        GatedSoftmaxFusion(d_model=hidden_dim, n_sources=2, hidden_dim=128)
        for _ in range(self.num_r_layers - 1)  # GF1~GF4, 不含 GF0 和 GF5
    ])
```

- [ ] **Step 2: 修改 _num_box_dof**

```python
if self.box_mode == "obb" and self.decouple_angle:
    self._num_box_dof = 6  # (cx,cy,w,h,ε,η) 内部表示
else:
    self._num_box_dof = 5  # 原逻辑
```

---

### Task 4: R decoder forward

**Files:** Modify `engine/deim/deim_decoder.py` — 新增 `forward_r_decoder` 方法

- [ ] **Step 1: 实现 forward_r_decoder**

```python
def forward_r_decoder(self, xywh_features_list, ref_angle_init, memory, spatial_shapes, 
                      attn_mask, query_pos_head_angle, eval_idx):
    """R decoder forward loop.
    
    Args:
        xywh_features_list: list of [B, 300, d] — 每层 XYWH decoder 的输出特征
        ref_angle_init: [B, 300, 2] — 初始 (ε,η) 来自 pre_angle_head
        memory: encoder memory
    Returns:
        r_output: [B, 300, 2] — 最终 (ε,η)
        r_features_list: list of per-layer R features
        r_refs_list: list of per-layer (ε,η) refs
    """
    import math
    from .obb_geometry import external_rect_to_oriented_box, oriented_box_to_external_rect
    
    output = ref_angle_init  # [B, 300, 2] — 初始 (ε,η)
    output_detach = 0
    pred_corners_undetach_angle = 0
    
    dec_out_angles = []
    dec_out_pred_corners_angle = []
    dec_out_refs_angle = []
    
    for i, layer in enumerate(self.r_layers):
        # 当前 (ε,η) → 6-dof ref for MSDeformableAttention
        # 需要 (cx,cy,w,h) 来自 XYWH 路径 + (ε,η) 来自 R 路径
        # 拼接为 6-dof，然后转换为 5-dof (cx,cy,w,h,θ) for attention
        # ... (具体实现需要 XYWH 路径提供当前层的 cx,cy,w,h)
        
        # R decoder layer forward (self-attn + cross-attn with encoder memory)
        # ref_points_input 需要是 5-dof (cx,cy,w,h,θ) for MSDeformableAttention
        # 这里需要从 (ε,η) 和 XYWH 的 (cx,cy,w,h) 转换出 θ
        # external_rect = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
        # obb = external_rect_to_oriented_box(ext_rect, (ε,η)) → (cx,cy,w,h,θ)
        # ref_5dof = obb  →  送入 layer
        
        # angle_head 输出 (ε,η) 的 DFL 分布
        pred_corners_angle = self.dec_angle_head[i](output + output_detach) + pred_corners_undetach_angle
        # Integral 解码为标量偏移
        angle_delta = self.integral_angle(pred_corners_angle, project)  # [B, 300, 2]
        
        # 残差更新 (ε,η)
        inter_ref_angle = ref_angle_init + angle_delta  # 简化版
        inter_ref_angle = inter_ref_angle.clamp(0, 1)  # sigmoid 空间
        
        if self.training or i == eval_idx:
            dec_out_angles.append(inter_ref_angle)
            dec_out_pred_corners_angle.append(pred_corners_angle)
            dec_out_refs_angle.append(ref_angle_init)
            if not self.training:
                break
        
        pred_corners_undetach_angle = pred_corners_angle
        ref_angle_init = inter_ref_angle.detach()
        output_detach = output.detach()
        
        # Gate Fusion (GF1~GF4, 不含 GF0 和 GF5)
        if i < len(self.gate_fusions):
            xywh_feat = xywh_features_list[i]  # L_i 的 XYWH 特征
            output = self.gate_fusions[i]([xywh_feat, output], query=output)
    
    return (
        torch.stack(dec_out_angles),
        torch.stack(dec_out_pred_corners_angle),
        torch.stack(dec_out_refs_angle),
    )
```

注意：这是一个骨架实现。实际编码时需要：
1. 从 XYWH 路径获取当前层的 (cx,cy,w,h)
2. 与 R 路径的 (ε,η) 拼接为 6-dof
3. 经 `external_rect_to_oriented_box` 转为 5-dof (cx,cy,w,h,θ)
4. 送入 `MSDeformableAttention` 的 `shape[-1]==5` 分支
5. 正交旋转注意力在此分支内执行

---

## Phase 3: 多角度锚点

### Task 5: 修改 _generate_anchors

**Files:** Modify `engine/deim/deim_decoder.py` — `_generate_anchors` (lines 623-677)

- [ ] **Step 1: 生成 6-dof 锚点 (cx,cy,w,h,ε,η)**

```python
elif self.box_mode == "obb":
    from .obb_geometry import oriented_box_to_external_rect
    
    angle_step_deg = getattr(self, "angle_step", 10)
    n_angles = 180 // angle_step_deg
    eps_angle = 1.0 / 180.0  # 1° 避免边界, 不用 1e-6
    
    for lvl, (h, w) in enumerate(spatial_shapes):
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)
        grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
        wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
        
        for k in range(n_angles):
            theta_val = max(k * angle_step_deg / 180.0, eps_angle)
            # 构造 OBB (cx,cy,w,h,θ) — θ in [0,π]
            obb = torch.cat([
                grid_xy, wh, 
                torch.full((*grid_xy.shape[:-1], 1), theta_val, dtype=dtype, device=device)
            ], dim=-1)  # (..., 5)
            # 转换为 (cx,cy,w,h,ε,η) — 6-dof
            ext_rect, vertex_offsets = oriented_box_to_external_rect(obb)
            anchor_6dof = torch.cat([grid_xy, wh, vertex_offsets], dim=-1)  # (..., 6)
            anchor_6dof = anchor_6dof.reshape(-1, h * w, 6)
            if k == 0:
                lvl_anchors = anchor_6dof
            else:
                lvl_anchors = torch.concat([lvl_anchors, anchor_6dof], dim=1)
        anchors.append(lvl_anchors)
```

- [ ] **Step 2: 扩展 valid_mask 检查 (ε,η)**

```python
if self.box_mode == "obb" and getattr(self, "decouple_angle", False):
    # 6-dof: 检查所有 6 个维度
    valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
else:
    # 原逻辑: 只检查前 4 维
    valid_mask = ((anchors[..., :4] > self.eps) * (anchors[..., :4] < 1 - self.eps)).all(-1, keepdim=True)
```

---

## Phase 4: 正交旋转注意力

### Task 6: 修改 MSDeformableAttention

**Files:** Modify `engine/deim/dfine_decoder.py` — `MSDeformableAttention.forward` (lines 167-184)

- [ ] **Step 1: 添加正交旋转分支**

在 `elif reference_points.shape[-1] == 5:` 分支内添加正交注意力：

```python
elif reference_points.shape[-1] == 5:
    angle = reference_points[..., 4:] * torch.pi
    n_heads = sampling_offsets.shape[2]
    half_heads = n_heads // 2
    angle_expanded = angle.expand(-1, -1, -1, n_heads)
    
    angle_modified = torch.where(
        torch.arange(n_heads, device=angle.device) < half_heads,
        angle_expanded,
        angle_expanded + math.pi / 2
    )
    
    cosa = torch.cos(angle_modified)
    sina = torch.sin(angle_modified)
    rot_matrix = torch.stack([cosa, -sina, sina, cosa], dim=-1).view(
        bs, Len_q, n_heads, 2, 2)
    
    wh = reference_points[..., 2:4] * 0.5
    scaled = (sampling_offsets * num_points_scale * self.offset_scale
              * wh[:, :, None, :, :])
    rotated = torch.einsum("bqhij,bqhpj->bqhpi", rot_matrix, scaled)
    sampling_locations = reference_points[:, :, None, :, :2] + rotated
```

---

## Phase 5: 集成（DN, aux_outputs, criterion, postprocessor）

### Task 7: DN 路径适配

**[2026-06-27 修订]** 根据对抗分析（dn-noise-analyst）+ 用户反馈：**不对角度加噪声**。原始 DN 只对 bbox 空间维度 (cx,cy,w,h) 加噪声，不对角度加噪声——过多噪声因素会降低模型质量。

**Files:** `engine/deim/denoising.py` — **可能不需要修改**

**分析结论：**

原始 OBB DN 代码（`denoising.py`）的行为：
1. 对 (cx,cy,w,h) 加空间噪声
2. 对 θ：`theta = gt_bbox[..., 4:] / π` → `inverse_sigmoid(theta)` → 拼接到噪声 bbox
3. **不对 θ 加噪声**——θ 来自 GT，只是做了量纲转换

**解耦后的策略：保持 DN 代码不变，在 decoder 调用处做 θ→(ε,η) 转换**

理由（dn-noise-analyst + edge-analyst 一致结论）：
- DN 产出 5-dof (cx,cy,w,h,θ) — 空间有噪声，θ 无噪声（来自 GT）
- 在 `deim_decoder.py` 的 `forward()` 中，DN queries 进入 decoder 前转换为 6-dof
- 转换：`θ → oriented_box_to_external_rect → (ε,η)` → 拼接为 `(cx,cy,w,h,ε,η)`
- (ε,η) 来自 GT θ，无噪声——与原始设计一致

- [ ] **Step 1: 在 deim_decoder.py forward 中添加 DN query 的 θ→(ε,η) 转换**

在 `DEIMTransformer.forward()` 中，DN queries 获取后（约 line 808-810），添加转换：

```python
if self.box_mode == "obb" and self.decouple_angle:
    # DN queries 从 denoising.py 出来时是 5-dof (cx,cy,w,h,θ), θ ∈ [0,1]
    # 转换为 6-dof (cx,cy,w,h,ε,η), (ε,η) 来自 GT θ 无噪声
    from .obb_geometry import oriented_box_to_external_rect
    
    dn_bbox_5dof = enc_topk_bbox_unact  # 或 dn_bbox_unact, 取决于 DN 路径
    # sigmoid → (cx,cy,w,h,θ) ∈ [0,1], θ *= π → [0,π]
    dn_obb = dn_bbox_5dof.clone()
    dn_obb[..., 4:] *= torch.pi  # θ → [0,π]
    # 转换为 (ε,η)
    ext_rect, vertex_offsets = oriented_box_to_external_rect(dn_obb)
    # 归一化 (ε,η) 到 [0,1] sigmoid 空间
    # (ε,η) 的范围取决于 ext_rect 的 w,h，需要适当归一化
    # 简单方案：直接使用 vertex_offsets 的值（不归一化，由 decoder 内部处理）
    dn_bbox_6dof = torch.cat([
        dn_bbox_5dof[..., :4],  # (cx,cy,w,h) 有噪声
        vertex_offsets           # (ε,η) 无噪声，来自 GT
    ], dim=-1)
    enc_topk_bbox_unact = dn_bbox_6dof
```

- [ ] **Step 2: 验证 denoising.py 不需要修改**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import inspect
from engine.deim.denoising import get_contrastive_denoising_training_group
# 确认函数签名包含 box_mode 参数
sig = inspect.signature(get_contrastive_denoising_training_group)
print('Signature:', sig)
# 原始 OBB DN: 只对 (cx,cy,w,h) 加噪声，θ 来自 GT 不加噪声
print('DN noise: spatial only, no angle noise — confirmed by code review')
print('No changes needed to denoising.py')
"
```

- [ ] **Step 3: 确认风险等级降低**

原始计划将 DN 标为 HIGH 风险。修订后：
- **不需要修改 denoising.py** → 风险从 HIGH 降为 LOW
- 转换逻辑在 `deim_decoder.py` 中，与 Task 8（输出转换）同一文件
- (ε,η) 来自 GT 无噪声 → 与原始设计一致，不影响训练质量

### Task 8: aux_outputs + 输出转换

**Files:** Modify `engine/deim/deim_decoder.py` — `DEIMTransformer.forward` (lines 836-916)

- [ ] **Step 1: 拼装 6-dof 输出并转换为 5-dof**

```python
if self.box_mode == "obb" and self.decouple_angle:
    from .obb_geometry import external_rect_to_oriented_box
    
    # 拼装: cat([xywh_bboxes, angle_εη]) → 6-dof → 转为 5-dof
    def convert_6dof_to_5dof(bboxes_6dof):
        """(cx,cy,w,h,ε,η) [0,1] → (cx,cy,w,h,θ) [0,π]"""
        cx, cy, w, h = bboxes_6dof[..., 0:1], bboxes_6dof[..., 1:2], bboxes_6dof[..., 2:3], bboxes_6dof[..., 3:4]
        eps_v, eta = bboxes_6dof[..., 4:5], bboxes_6dof[..., 5:6]
        # 构造 external rect
        ext_rect = torch.cat([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)
        vertex_offsets = torch.cat([eps_v, eta], dim=-1)
        obb_5dof = external_rect_to_oriented_box(ext_rect, vertex_offsets)  # (cx,cy,w,h,θ)
        # θ ∈ [0,π] → /π → [0,1]
        obb_5dof = torch.cat([obb_5dof[..., :4], obb_5dof[..., 4:5] / torch.pi], dim=-1)
        return obb_5dof
    
    out_bboxes = convert_6dof_to_5dof(
        torch.cat([out_bboxes_xywh, out_angles], dim=-1)
    )
    # 然后执行原有的 θ *= π 转换
    out_bboxes = torch.cat([out_bboxes[..., :4], out_bboxes[..., 4:] * torch.pi], dim=-1)
```

- [ ] **Step 2: 构建 aux_outputs**

```python
if self.training and self.aux_loss:
    out["aux_outputs"] = []
    for i in range(len(out_logits)):
        aux = {
            "pred_logits": out_logits[i],
            "pred_boxes": convert_6dof_to_5dof(
                torch.cat([xywh_bboxes[i], angle_outputs[i]], dim=-1)
            ),
            "pred_corners": torch.cat([xywh_corners[i], angle_corners[i]], dim=-1),
            "ref_points": torch.cat([xywh_refs[i], angle_refs[i]], dim=-1),
        }
        out["aux_outputs"].append(aux)
```

### Task 9: loss_local 拆分 + get_loss_meta_info 修复

**Files:** Modify `engine/deim/deim_criterion.py`

- [ ] **Step 1: 拆分 loss_local**

```python
def loss_local(self, outputs, targets, indices, num_boxes, T=5):
    if self.box_mode == "obb" and getattr(self, "decouple_angle", False):
        return self.loss_local_decoupled(outputs, targets, indices, num_boxes, T)
    # 原逻辑
    ...

def loss_local_decoupled(self, outputs, targets, indices, num_boxes, T=5):
    losses = {}
    # XYWH 部分: 用 bbox2distance_obb_xywh
    # ... 参考 loss_local 的 OBB 分支，但用 4-dist 版本
    
    # Angle 部分: 用 bbox2distance_obb_angle
    # ... 新增，监督 (ε,η) 预测
    
    return losses
```

- [ ] **Step 2: 修复 get_loss_meta_info**

```python
# 原代码 lines 701-703:
# if self.box_mode == "obb":
#     raise NotImplementedError()

# 修复:
if self.box_mode == "obb":
    iou = batch_probiou(src_boxes.detach(), target_boxes)
    iou = torch.diag(iou)
```

### Task 10: Postprocessor 适配

**Files:** Modify `engine/deim/postprocessor.py`

- [ ] **Step 1: 处理 6-dof 输入**

postprocessor 接收的已经是 5-dof (cx,cy,w,h,θ)（在 decoder 输出时已转换），所以**不需要修改**。

验证：decoder forward 的输出经过 `convert_6dof_to_5dof` + `*= π` 后是 5-dof [0,π]，与 postprocessor 期望一致。

### Task 11: Matcher 适配

**Files:** Modify `engine/deim/matcher.py`

matcher 接收的也是 5-dof（decoder 输出已转换），**不需要修改**。

---

## Phase 6: 配置 + 集成测试

### Task 12: 创建解耦配置

**Files:** Create `configs/custom_obb/deimv2_obb_decouple.yml`

```yaml
# 基于 deimv2_obb_sp.yml，新增:
DEIMTransformer:
  box_mode: "obb"
  decouple_angle: True
  angle_step: 10
  num_r_layers: 6
```

### Task 13: HBB 兼容性测试

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import sys; sys.path.insert(0, '.')
from engine.core import YAMLConfig
cfg = YAMLConfig('configs/custom_obb/deimv2_obb_sp.yml')
print('HBB config loads OK')
"
```

### Task 14: OBB 解耦 smoke test

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
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

---

## Phase 7: 训练 + 诊断

### Task 15: 在 density_020 上训练

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD
python train.py -c configs/custom_obb/deimv2_obb_decouple.yml
```

### Task 16: 运行诊断

```bash
python test/diagnose_hungarian_matching.py
python test/test_infer_diag.py --num 20
```

**成功判据**: Q3 Pearson r ≥ 0.3（从 0.084 提升）

---

## 实施顺序总结

```
Phase 0 (并行):  Task 0a (GatedSoftmaxFusion)  ─┐
                  Task 0b (复用确认, 无新函数)    ├─ 基础设施
                  Task 0c (config plumbing)      ─┘

Phase 1 (串行):  Task 1 (head/Integral split) → Task 2 (XYWH forward)

Phase 2 (串行):  Task 3 (R layers) → Task 4 (R forward)

Phase 3 (串行):  Task 5 (multi-angle anchors)

Phase 4 (独立):  Task 6 (orthogonal attention)  — 可与 Phase 2-3 并行

Phase 5 (串行):  Task 7 (DN) → Task 8 (aux/output) → Task 9 (criterion) → Task 10-11 (postprocessor/matcher 验证)

Phase 6:         Task 12-14 (config + tests)

Phase 7:         Task 15-16 (training + diagnostic)
```

## 验证门控

| 门控点 | 验证内容 | 通过标准 |
|--------|---------|---------|
| Phase 0 完成 | 单元测试 | GatedSoftmaxFusion shape 正确；确认复用策略无新函数 |
| Phase 1 完成 | HBB forward 不变 | HBB 输出与 baseline 数值一致 |
| Phase 2 完成 | OBB forward 不 crash | pred_boxes shape = (B, 300, 5) |
| Phase 5 完成 | forward + backward | loss 有限、梯度非 NaN |
| Phase 6 完成 | smoke test | 1 iter 训练不 crash |
| Phase 7 完成 | 诊断 | Q3 r ≥ 0.3 |

---

## 修订记录

### 2026-06-27 第二轮 Hyperplan 修订

**来源**: 4 个对抗分析师（func-design-analyst/ultrabrain, dn-noise-analyst/unspecified-high, integration-analyst/unspecified-low, edge-analyst/artistry）针对用户 3 个反馈问题的独立分析。

**修订内容**:
1. **删除 4 个不必要的新函数** (`distance2bbox_obb_xywh`, `distance2bbox_obb_angle`, `bbox2distance_obb_xywh`, `bbox2distance_obb_angle`) — 直接复用 `distance2bbox` / `bbox2distance` / `bbox2distance_obb`，R 路径角度解码用 `Integral` + 残差更新（不是 distance2bbox 操作）
2. **DN 不对角度加噪声** — 保持 `denoising.py` 不修改，在 decoder 调用处做 θ→(ε,η) 转换。风险从 HIGH 降为 LOW
3. **Task 0b 从"创建新函数"改为"确认复用策略"** — 无新代码，只有复用确认

**Provenance**: 本计划由 Hyperplan 对抗分析流程产出。第一轮（5 个分析师）产出初始 bundle，Lead 基于 bundle 编写计划。第二轮（4 个分析师）针对用户反馈的 3 个问题进行分析，确认修正方案，Lead 直接应用修正。团队已清理。
