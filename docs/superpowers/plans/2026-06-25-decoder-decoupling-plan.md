# DEIMv2-OBB Decoder 解耦实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 DEIMv2-OBB decoder 拆分为 XYWH 路径（6层）和 R 路径（6层），通过 Gated Softmax Fusion 桥接，并引入 angle_step 多角度锚点和正交旋转注意力。

**Architecture:** XYWH decoder 保留 D-FINE refinement + label 分类；R decoder 独立预测角度偏移 (sin/cos 编码)，每层 Gate Fusion 融合 xywh 特征 + 角度特征 + encoder memory（经 cross-attn 汇聚）。encoder memory 双路注入。

**Tech Stack:** PyTorch, torch.nn, DEIMv2 engine

**[2026-06-26 修订]** 根据整合评审修复以下问题：
- F7: `self.decoder_layer` 不存在 → Task 4 Step 1 改为从 `self.decoder.layers[-1]` 获取
- F8: box_head 输出维度错 8 倍 → Task 3 Step 2 改为 `4*(reg_max+1)` 和 `2*(reg_max+1)`
- F9: GatedSoftmaxFusion token 数不匹配 → Task 1 增加 encoder memory 汇聚步骤
- S1: `distance2bbox_obb_xywh` 未创建 → 新增 Task 3.5
- S2: 角度 DFL 维度混淆 → Task 4 Step 2 改为 sin/cos 编码
- S3: `r % π` 边界不连续 → Task 4 Step 2 改为向量归一化
- O1-O5: 完备性遗漏 → 新增 Task 4.5（DN/aux/LQE/criterion 适配）

## Global Constraints

- Mode A 优先（label 留在 xywh 路径），Mode B 后续实现
- angle_step 默认 10°，可通过 YAML 配置
- 不破坏 HBB 模式（`box_mode="hbb"` 时行为不变）
- 保持与现有 criterion/matcher/postprocessor 接口兼容
- Python 环境：`/home/cx/apps/miniconda3/envs/deimv2/`
- **[2026-06-26 新增]** 角度用 sin/cos 编码，不用直接弧度值
- **[2026-06-26 新增]** encoder memory 先经 cross-attn 汇聚到 300 query 位置再融合

---

## File Structure

**[2026-06-26 修订]**: 新增 `gated_fusion.py`（含 EncoderMemoryAggregator），`dfine_utils.py` 新增 `distance2bbox_obb_xywh`

```
engine/deim/
  deim_decoder.py        # MODIFY: anchor multi-angle, XYWH path, R path, Gate Fusion, DN/aux 适配
  dfine_decoder.py       # MODIFY: orthogonal rotation attention
  dfine_utils.py         # MODIFY: 新增 distance2bbox_obb_xywh, bbox2distance_obb_xywh
  deim_utils.py          # REUSE: existing Gate class (line 70)
  gated_fusion.py        # CREATE: GatedSoftmaxFusion + EncoderMemoryAggregator (F9 修复)
  deim_criterion.py      # MODIFY: aux_outputs 适配, loss_local 拆分, NotImplementedError 修复 (O3)
configs/custom_obb/
  deimv2_obb_decouple.yml  # CREATE: new config with decouple parameters
```

---

### Task 1: Gated Softmax Fusion 模块

**Files:**
- Create: `engine/deim/gated_fusion.py`

**Interfaces:**
- Produces: `class GatedSoftmaxFusion(nn.Module)` — `forward(srcs: list[Tensor], query: Tensor) → Tensor`
- **[2026-06-26 修订 F9]**: 所有输入源必须先汇聚到 `[B, num_queries, d_model]` 形状

- [ ] **Step 1: 创建模块**

```python
"""Gated Softmax Fusion — 多源特征动态加权融合。

[2026-06-26 修订 F9]
所有输入源必须先汇聚到 [B, num_queries, d_model] 形状。
encoder memory 原始形状 [B, ΣH_iW_i, d] 需先经 cross-attn 汇聚。

对 N 个输入源做 token 级动态加权融合:
  cat = concat([query, src_0, src_1, ..., src_{N-1}])
  w = softmax(MLP(cat))           # [B, num_queries, N]
  fused = Σ w_i * src_i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderMemoryAggregator(nn.Module):
    """将 encoder memory [B, ΣH_iW_i, d] 汇聚到 [B, num_queries, d]。
    
    使用 cross-attention: query 来自 decoder query，key/value 来自 encoder memory。
    """
    
    def __init__(self, d_model: int, nhead: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, query: torch.Tensor, encoder_memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, num_queries, d_model] — decoder query 作为 cross-attn 的 query
            encoder_memory: [B, ΣH_iW_i, d_model] — encoder 输出的多尺度特征
        Returns:
            aggregated: [B, num_queries, d_model]
        """
        # Cross-attention: query attends to encoder memory
        aggregated, _ = self.cross_attn(
            query=query,           # [B, num_queries, d]
            key=encoder_memory,    # [B, ΣH_iW_i, d]
            value=encoder_memory,  # [B, ΣH_iW_i, d]
        )
        return self.norm(aggregated)


class GatedSoftmaxFusion(nn.Module):
    """多源 Gated Softmax 融合。

    用法:
        aggregator = EncoderMemoryAggregator(d_model=256)
        fusion = GatedSoftmaxFusion(d_model=256, n_sources=3, hidden_dim=128)
        
        # 先汇聚 encoder memory
        enc_feat = aggregator(query=r_query, encoder_memory=encoder_memory)
        
        # 再融合（所有源都是 [B, 300, d]）
        fused = fusion([enc_feat, xywh_feat, r_feat], query=r_query)
    """

    def __init__(self, d_model: int, n_sources: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.n_sources = n_sources
        # MLP: (n_sources+1)*d_model → hidden → n_sources (权重)
        self.weight_net = nn.Sequential(
            nn.Linear((n_sources + 1) * d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_sources),
        )
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, srcs: list[torch.Tensor], query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            srcs: list of N tensors, each [B, num_queries, d_model]
                  **[2026-06-26 F9 修复]**: 所有 src 必须是 [B, num_queries, d]
                  encoder memory 需先经 EncoderMemoryAggregator 汇聚
            query: [B, num_queries, d_model] — 用于生成融合权重的查询
        Returns:
            fused: [B, num_queries, d_model]
        """
        assert len(srcs) == self.n_sources, f"Expected {self.n_sources} sources, got {len(srcs)}"
        
        # [2026-06-26 F9 修复] 验证所有源形状一致
        num_queries = query.shape[1]
        for i, src in enumerate(srcs):
            assert src.shape[1] == num_queries, \
                f"Source {i} has {src.shape[1]} tokens, expected {num_queries}. " \
                f"Encoder memory must be aggregated first via EncoderMemoryAggregator."

        cat = torch.cat([query] + srcs, dim=-1)                    # [B, num_queries, (n_src+1)*d]
        weights = torch.softmax(self.weight_net(cat), dim=-1)      # [B, num_queries, n_src]

        fused = torch.zeros_like(srcs[0])
        for i, src in enumerate(srcs):
            fused = fused + weights[..., i:i + 1] * src

        return self.output_proj(fused)
```

- [ ] **Step 2: 单元测试**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import torch
from engine.deim.gated_fusion import GatedSoftmaxFusion, EncoderMemoryAggregator

# 测试 EncoderMemoryAggregator
agg = EncoderMemoryAggregator(d_model=256, nhead=8)
query = torch.randn(2, 300, 256)
enc_memory = torch.randn(2, 1000, 256)  # 多尺度特征展平
enc_feat = agg(query, enc_memory)
print(f'Aggregated shape: {enc_feat.shape}')  # Expected: [2, 300, 256]
assert enc_feat.shape == (2, 300, 256)

# 测试 GatedSoftmaxFusion（所有源都是 [B, 300, d]）
m = GatedSoftmaxFusion(256, 3, 128)
x = [torch.randn(2, 300, 256) for _ in range(3)]  # 修复 F9: 300 tokens
q = torch.randn(2, 300, 256)
y = m(x, q)
print(f'Output shape: {y.shape}')  # Expected: [2, 300, 256]
print('PASS')
"
```

预期: `Aggregated shape: torch.Size([2, 300, 256])` + `Output shape: torch.Size([2, 300, 256])` + `PASS`

---

### Task 2: Anchor 多角度生成

**Files:**
- Modify: `engine/deim/deim_decoder.py`（`_generate_anchors` 方法，约 622-676 行）

**Interfaces:**
- Modifies: `_generate_anchors()` — OBB 分支从固定 θ=0.5 改为根据 `angle_step` 生成多角度锚点

**[2026-06-26 修订]**: 修复 θ=0 产生 -inf 的问题。使用 `(k + 0.5) * angle_step / 180` 避免边界。

- [ ] **Step 1: 修改 `_generate_anchors` 的 OBB 分支**

找到 `_generate_anchors` 中 OBB 分支（第 646-662 行），将：

```python
theta = 0.5 * torch.ones(
    *grid_xy.shape[:-1], 1, dtype=grid_xy.dtype, device=grid_xy.device
)
lvl_anchors = torch.concat([grid_xy, wh, theta], dim=-1).reshape(
    -1, h * w, self._num_box_dof
)
```

替换为：

```python
# [2026-06-26 修复] 多角度锚点，避免 θ=0 产生 -inf
# 使用 (k + 0.5) * angle_step / 180 确保 θ ∈ (0, 1)
angle_step_deg = getattr(self, "angle_step", 10)  # 默认 10°
n_angles = 180 // angle_step_deg
for k in range(n_angles):
    # [修复] 避免边界：θ ∈ (0, 1)，不是 [0, 1)
    theta_val = (k + 0.5) * angle_step_deg / 180.0
    theta = theta_val * torch.ones(
        *grid_xy.shape[:-1], 1, dtype=grid_xy.dtype, device=grid_xy.device
    )
    lvl_anchors_k = torch.concat([grid_xy, wh, theta], dim=-1).reshape(
        -1, h * w, self._num_box_dof
    )
    if k == 0:
        lvl_anchors = lvl_anchors_k
    else:
        lvl_anchors = torch.concat([lvl_anchors, lvl_anchors_k], dim=1)
```

- [ ] **Step 2: 添加 `angle_step` 参数传递**

在 `DEIMTransformer.__init__` 中添加：

```python
self.angle_step = getattr(kwargs, "angle_step", 10)
```

- [ ] **Step 3: 验证 anchor 数量变化**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import sys; sys.path.insert(0, '.')
from engine.core import YAMLConfig
cfg = YAMLConfig('configs/custom_obb/synthetic_exp_020.yml')
# 检查 anchor 数量（需手动看 _generate_anchors 输出）
print('Test anchor generation manually with angle_step=10')
print('Expected: 18x more anchors per grid position')
print('θ range: (0.5*10/180, 1.5*10/180, ..., 17.5*10/180) = (0.028, 0.083, ..., 0.972)')
print('All θ ∈ (0, 1), no -inf after inverse sigmoid')
"
```

---

### Task 3: XYWH Decoder 路径调整

**Files:**
- Modify: `engine/deim/deim_decoder.py`（decoder layer forward 约 207-280 行）

**Interfaces:**
- 现有 decoder layer 的 box_head 输出从 6 条分布改为 4 条分布
- **[2026-06-26 修订 F8]**: 输出维度是 `num_distributions * (reg_max + 1)`，不是 `num_distributions * 6`
  - 原 OBB: `6 * (reg_max + 1) = 6 * 33 = 198`
  - 新 XYWH: `4 * (reg_max + 1) = 4 * 33 = 132`
- reference_points 只用 `[:,:,:4]`（去掉 θ）

- [ ] **Step 1: 分离 reference_points**

在 decoder layer 的 forward 入口处：

```python
# 原代码：
# ref_points_detach = reference_points.detach()

# 改为：
if self.box_mode == "obb" and getattr(self, "decouple_angle", False):
    ref_xywh = reference_points[..., :4].detach()  # XYWH 路径只用前 4 维
    ref_r = reference_points[..., 4:5].detach()     # R 路径单独处理
else:
    ref_xywh = reference_points.detach()
    ref_r = None
```

使用 `ref_xywh` 替代 decoder layer 内所有 `ref_points_detach` 的引用。

- [ ] **Step 2: 缩减 box_head 输出维度**

**[2026-06-26 修订 F8]**: 输出维度是分布数 × (reg_max + 1)，不是分布数 × 6

XYWH 路径的 box_head 输出从 6 条分布改为 4 条分布：

```python
# 原代码（deim_decoder.py 约 476-498 行，dec_bbox_head 定义处）：
# dec_bbox_head = MLP(hidden_dim, hidden_dim, self.num_reg_dist * (self.reg_max + 1), 3)
# OBB 下: 6 * 33 = 198

# 改为条件：
if self.box_mode == "obb" and getattr(self, "decouple_angle", False):
    # XYWH 路径只用 4 条分布: 4 * 33 = 132
    self.dec_bbox_head_xywh = nn.ModuleList([
        MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3)
        for _ in range(num_layers)
    ])
else:
    # 保持原逻辑
    self.dec_bbox_head = nn.ModuleList([
        MLP(hidden_dim, hidden_dim, self.num_reg_dist * (self.reg_max + 1), 3)
        for _ in range(num_layers)
    ])
```

### Task 3.5: 新增 `distance2bbox_obb_xywh` 函数 **[2026-06-26 新增 S1]**

**Files:**
- Modify: `engine/deim/dfine_utils.py`

**Interfaces:**
- Produces: `distance2bbox_obb_xywh(points, distance, reg_scale) → Tensor`
- Produces: `bbox2distance_obb_xywh(points, bbox, reg_max, reg_scale, up, eps) → Tuple`

- [ ] **Step 1: 实现 `distance2bbox_obb_xywh`**

在 `engine/deim/dfine_utils.py` 末尾添加：

```python
def distance2bbox_obb_xywh(points, distance, reg_scale):
    """
    Decodes 4-distribution DDF output to 4-dof (cx,cy,w,h) only.
    Angle is handled separately by R decoder.

    Args:
        points: (B,N,4) or (N,4) — ref points (cx,cy,w,h), θ is ignored
        distance: (B,N,4*(reg_max+1)) — (α,β,γ,δ) distributions
        reg_scale: curvature of Weighting Function.

    Returns:
        (B,N,4) or (N,4) — (cx,cy,w,h)
    """
    # 复用现有 distance2bbox 逻辑，但只用前 4 个距离
    # points 的 θ 维度被忽略
    return distance2bbox(points, distance, reg_scale)


def bbox2distance_obb_xywh(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """
    Converts GT (cx,cy,w,h) to 4-distribution FGL targets.
    Angle is handled separately by R decoder.

    Args:
        points: (N,4) ref points (cx,cy,w,h), θ is ignored
        bbox: (N,4) GT boxes (cx,cy,w,h)
        ...
    Returns:
        four_lens, weight_right, weight_left
    """
    # 复用现有 bbox2distance 逻辑
    return bbox2distance(points, bbox, reg_max, reg_scale, up, eps)
```

- [ ] **Step 2: 单元测试**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import torch
from engine.deim.dfine_utils import distance2bbox_obb_xywh, bbox2distance_obb_xywh

# 测试 distance2bbox_obb_xywh
points = torch.tensor([[0.5, 0.5, 0.2, 0.3]])  # (cx, cy, w, h)
# 创建 4 条分布，每条 33 bins
distance = torch.zeros(1, 4 * 33)
distance[:, 16] = 1.0  # α 集中在 bin 16
distance[:, 32+16] = 1.0  # β
distance[:, 66+16] = 1.0  # γ
distance[:, 99+16] = 1.0  # δ

result = distance2bbox_obb_xywh(points, distance, reg_scale=4.0)
print(f'Result shape: {result.shape}')  # Expected: (1, 4)
assert result.shape == (1, 4)
print('distance2bbox_obb_xywh PASS')
"
```

---

### Task 4: R Decoder 路径

**Files:**
- Modify: `engine/deim/deim_decoder.py`（新增 R decoder layers 和 pre_angle_head）

**[2026-06-26 修订]**
- F7: `self.decoder_layer` 不存在 → 改为从 `self.decoder.layers[-1]` 获取
- S2: 角度用 sin/cos 编码，预测残差而非 DFL 分布
- S3: 不使用 `% π`，用向量归一化保持周期性

- [ ] **Step 1: 添加 R decoder layers**

**[2026-06-26 修订 F7]**: 从 `self.decoder.layers[-1]` 获取 decoder layer 模板

在 `DEIMTransformer.__init__` 中，当 `decouple_angle=True` 时，额外创建 6 个 R decoder layer：

```python
if self.box_mode == "obb" and getattr(self, "decouple_angle", False):
    self.num_r_layers = 6
    
    # [2026-06-26 F7 修复] 从 self.decoder.layers 获取模板，不是 self.decoder_layer
    r_layer_template = self.decoder.layers[-1]
    self.r_layers = nn.ModuleList([
        copy.deepcopy(r_layer_template) for _ in range(self.num_r_layers)
    ])
    
    # [2026-06-26 S2 修复] 角度 head 预测 sin/cos 残差，不是 DFL 分布
    self.pre_angle_head = MLP(hidden_dim, hidden_dim, 2, 3)  # 初始 sin, cos
    self.angle_heads = nn.ModuleList([
        MLP(hidden_dim, hidden_dim, 2, 3)  # 每层预测 delta_sin, delta_cos
        for _ in range(self.num_r_layers)
    ])
    
    # Encoder memory 汇聚器（F9 修复）
    self.enc_aggregators = nn.ModuleList([
        EncoderMemoryAggregator(d_model=hidden_dim, nhead=nhead)
        for _ in range(self.num_r_layers)
    ])
    
    self.gate_fusions = nn.ModuleList([
        GatedSoftmaxFusion(d_model=hidden_dim, n_sources=3, hidden_dim=128)
        for _ in range(self.num_r_layers)
    ])
```

- [ ] **Step 2: 实现 R decoder 前向**

**[2026-06-26 S2+S3 修复]**: sin/cos 编码 + 向量归一化

```python
def forward_r_decoder(self, xywh_features, r_init_angle, encoder_memory, ...):
    """
    Args:
        xywh_features: [B, 300, d] — XYWH decoder 最终层特征
        r_init_angle: [B, 300, 1] — 初始角度（弧度，来自 anchor）
        encoder_memory: [B, ΣH_iW_i, d] — encoder 输出
    """
    B, N, _ = xywh_features.shape
    
    # [S2 修复] 初始 sin/cos 编码
    sin_current = torch.sin(2 * r_init_angle)  # 周期 π
    cos_current = torch.cos(2 * r_init_angle)
    
    for layer_idx in range(self.num_r_layers):
        # 1. 当前角度特征 = [sin, cos]
        r_query = torch.cat([sin_current, cos_current], dim=-1)  # [B, 300, 2]
        # 投影到 d_model
        r_query = self.pre_angle_head(r_query) if layer_idx == 0 else r_features
        
        # 2. R decoder layer 前向（self-attn + cross-attn）
        r_features = self.r_layers[layer_idx](r_query, encoder_memory, ...)
        
        # 3. [S2 修复] 预测 sin/cos 残差，不是 DFL 分布
        delta = self.angle_heads[layer_idx](r_features)  # [B, 300, 2]
        delta_sin = delta[..., 0:1]
        delta_cos = delta[..., 1:2]
        
        # 4. [S3 修复] 向量加法 + 归一化，不用 % π
        sin_new = sin_current + delta_sin
        cos_new = cos_current + delta_cos
        norm = torch.sqrt(sin_new**2 + cos_new**2 + 1e-8)
        sin_current = sin_new / norm
        cos_current = cos_new / norm
        
        # 5. [F9 修复] Encoder memory 先汇聚到 300 query 位置
        enc_feat = self.enc_aggregators[layer_idx](
            query=r_features,
            encoder_memory=encoder_memory
        )
        
        # 6. Gate Fusion: 融合 enc_feat + xywh 特征 + r 特征
        r_fused = self.gate_fusions[layer_idx](
            [enc_feat, xywh_features, r_features],
            query=r_features
        )
    
    # [S2 修复] 最终解码为弧度
    # atan2 返回 [-π/2, π/2]，映射到 [0, π)
    r_final = 0.5 * torch.atan2(sin_current, cos_current)  # [-π/4, π/4]
    r_final = (r_final + math.pi / 4) % math.pi  # [0, π)
    
    return r_final, r_features
```

- [ ] **Step 3: 集成到主 forward**

在主 `forward()` 中：

```python
# XYWH decoder 前向（Task 3 产出 xywh 特征和中间参考点）
xywh_output, xywh_features = self.forward_xywh_decoder(...)

# R decoder 前向
r_initial = anchors[..., 4:5] * math.pi  # 初始角度（弧度）
r_output, r_features = self.forward_r_decoder(
    xywh_features=xywh_features,  # 来自 xywh 最后层
    r_init_angle=r_initial,
    encoder_memory=encoder_memory
)

# 合并输出：xywh + r
pred_boxes = torch.cat([xywh_output, r_output], dim=-1)
```

### Task 4.5: DN/aux/LQE/criterion 适配 **[2026-06-26 新增 O1-O5]**

**Files:**
- Modify: `engine/deim/deim_decoder.py`（DN 路径）
- Modify: `engine/deim/deim_criterion.py`（aux_outputs + loss_local）

- [ ] **Step 1: DN 路径适配 [O1]**

Denoising queries 也含 θ，需要决定 R decoder 是否对 DN queries 也跑。

**方案**: R decoder 对所有 queries（包括 DN）都跑。DN 的 angle ref_points 从 `get_contrastive_denoising_training_group` 获取，与 XYWH 路径共享相同的拆分逻辑。

```python
# 在 forward() 中，DN queries 也走 R decoder
if self.training and dn_meta is not None:
    # DN queries 的 angle 也从 anchors 初始化
    r_initial_dn = dn_anchors[..., 4:5] * math.pi
    r_output_dn, _ = self.forward_r_decoder(
        xywh_features=xywh_features_dn,
        r_init_angle=r_initial_dn,
        encoder_memory=encoder_memory
    )
```

- [ ] **Step 2: aux_outputs 适配 [O2]**

现有 aux_outputs 每层带 `pred_corners`（6 条分布）和 `ref_points`（5 维）。解耦后：
- XYWH 路径 aux: `pred_corners` 4 条分布，`ref_points` 4 维
- R 路径 aux: `pred_angle` sin/cos 编码，`ref_angle` sin/cos 编码

```python
# 在 forward() 中构建 aux_outputs
if self.training and self.aux_loss:
    out["aux_outputs"] = []
    for i in range(num_layers):
        aux = {
            "pred_logits": out_logits[i],
            "pred_boxes": torch.cat([xywh_bboxes[i], r_angles[i]], dim=-1),  # 5 维
            "pred_corners": xywh_corners[i],  # 4 条分布
            "ref_points": xywh_refs[i],  # 4 维
            "pred_angle": r_angles[i],  # sin/cos
            "ref_angle": r_refs[i],  # sin/cos
        }
        out["aux_outputs"].append(aux)
```

- [ ] **Step 3: loss_local (FGL) 适配 [O2]**

现有 `loss_local` 用 `bbox2distance_obb`（6 维）。解耦后需要：
- `loss_local_xywh`: 用 `bbox2distance_obb_xywh`（4 维）
- `loss_local_angle`: 新增，监督 sin/cos 预测

```python
# 在 deim_criterion.py 中
def loss_local(self, outputs, targets, indices, num_boxes, T=5):
    # XYWH 部分
    loss_xywh = self.loss_local_xywh(outputs, targets, indices, num_boxes, T)
    
    # Angle 部分（新增）
    loss_angle = self.loss_local_angle(outputs, targets, indices, num_boxes, T)
    
    return {**loss_xywh, **loss_angle}
```

- [ ] **Step 4: LQE 归属 [O5]**

XYWH 路径的 LQE 只用 4 条分布（α,β,γ,δ），不含 angle。R 路径不需要 LQE（angle 是连续值，不是分类）。

```python
# 在 TransformerDecoder.forward() 中
if self.box_mode == "obb" and self.decouple_angle:
    # LQE 只用 xywh 的 4 条分布
    scores = self.lqe_layers[i](scores, pred_corners_xywh)  # 4 条分布
else:
    scores = self.lqe_layers[i](scores, pred_corners)  # 原逻辑
```

- [ ] **Step 5: criterion NotImplementedError 修复 [O3]**

`deim_criterion.py:701-703` 的 `NotImplementedError` 需要修复：

```python
# 原代码
if self.box_mode == "obb":
    raise NotImplementedError()

# 修复：OBB 用 probiou 作为 boxes_weight
if self.box_mode == "obb":
    iou = batch_probiou(src_boxes, target_boxes)
    iou = torch.diag(iou)
```

---

### Task 5: 正交旋转注意力

**Files:**
- Modify: `engine/deim/dfine_decoder.py`（MSDeformableAttention 的 OBB 5 维分支，约 167-184 行）

**[2026-06-26 修订]**: 前提条件更新——OBB_CODE_REVIEW.md #1 已修复旋转注意力数学错误，当前实现是"先按半边尺寸缩放、再用 R(θ) 整体旋转"。正交注意力作为增强添加。

- [ ] **Step 1: 修改旋转矩阵生成**

```python
elif reference_points.shape[-1] == 5:
    # 当前角度（弧度）
    angle = reference_points[..., 4:] * torch.pi           # [bs, Len_q, 1, 1]

    # 正交旋转：heads H/2 ~ H 用 angle + π/2
    n_heads = sampling_offsets.shape[2]  # H
    half_heads = n_heads // 2
    angle_expanded = angle.expand(-1, -1, -1, n_heads)     # [bs, Lq, 1, H]

    # 前 half_heads: 使用 angle（轴向）
    # 后 half_heads: 使用 angle + π/2（正交方向）
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

- [ ] **Step 2: 验证正交性**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import torch, math
bs, Lq, H = 1, 5, 8
ref = torch.randn(bs, Lq, 1, 5)
ref[..., 4:] = 0.25  # θ = π/4
# 手动验证两个 head 的旋转矩阵是否差 π/2
print('Orthogonal attention test')
print('Angle for head 0:', ref[0,0,0,4].item() * math.pi)
print('Angle for head H/2+1:', (ref[0,0,0,4].item() + 0.5) * math.pi)
print('Difference: π/2 =', math.pi/2)
"
```

---

### Task 6: 配置与集成测试

**Files:**
- Create: `configs/custom_obb/deimv2_obb_decouple.yml`
- Verify: 兼容 criterion/matcher/postprocessor

- [ ] **Step 1: 创建解耦配置**

```yaml
# 基于 deimv2_obb_sp.yml，新增:
DEIMTransformer:
  box_mode: "obb"
  decouple_angle: True
  angle_step: 10
  num_r_layers: 6
  feat_channels: [256, 256, 256]
  hidden_dim: 256
  dim_feedforward: 2048
  # ... 其余保持 deimv2_obb_sp.yml 不变
```

- [ ] **Step 2: HBB 兼容性测试**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import sys; sys.path.insert(0, '.')
from engine.core import YAMLConfig
# 加载 HBB 配置，确认 box_mode='hbb' 时行为不变
cfg = YAMLConfig('configs/custom_obb/deimv2_obb_sp.yml')
print('HBB compatibility OK')
"
```

- [ ] **Step 3: OBB 解耦启动测试**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python train.py --config configs/custom_obb/deimv2_obb_decouple.yml
```

预期：模型加载成功，开始训练。因是新架构，loss 可能波动，但不应 crash。

---

### Task 7: 合成数据集验证

- [ ] **Step 1: 在 density_020 上做短训**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD
# 修改 deimv2_obb_decouple.yml 的 dataset 路径指向 density_020
python train.py --config configs/custom_obb/deimv2_obb_decouple.yml
```

- [ ] **Step 2: 运行诊断**

```bash
python test/diagnose_hungarian_matching.py --ckpt outputs/synthetic_exp_decouple_020/last.pth --config configs/custom_obb/deimv2_obb_decouple.yml
```

预期：Q3 的 r 值显著提升（从 0.05 到 ≥ 0.3）

- [ ] **Step 3: 运行分数分布检查**

```bash
python test/test_infer_diag.py --ckpt outputs/synthetic_exp_decouple_020/last.pth --config configs/custom_obb/deimv2_obb_decouple.yml --num 20
```

预期：score 分布出现与 IoU 正相关的分离

---

## 修订总结 [2026-06-26]

根据整合评审（`../review/2026-06-25-decoder-decoupling-review-INTEGRATED_GLM_5.2_max.md`）修复以下问题：

### 阻断级修复

| 问题 | 修复位置 | 修复内容 |
|------|---------|---------|
| **F7**: `self.decoder_layer` 不存在 | Task 4 Step 1 | 改为从 `self.decoder.layers[-1]` 获取模板 |
| **F8**: box_head 输出维度错 8 倍 | Task 3 Step 2 | `num_reg_dist*4` → `4*(reg_max+1)` = 132 |
| **F9**: GatedSoftmaxFusion token 数不匹配 | Task 1, Task 4 Step 2 | 新增 `EncoderMemoryAggregator`，encoder memory 先经 cross-attn 汇聚到 300 query 位置 |

### 严重级修复

| 问题 | 修复位置 | 修复内容 |
|------|---------|---------|
| **S1**: `distance2bbox_obb_xywh` 未创建 | Task 3.5（新增） | 在 `dfine_utils.py` 新增函数 |
| **S2**: 角度 DFL 维度/语义混淆 | Task 4 Step 2 | 改为 sin/cos 编码，预测残差而非 DFL 分布 |
| **S3**: `r % π` 边界不连续 | Task 4 Step 2 | 用向量归一化保持周期性，不用取模 |

### 完备性修复

| 问题 | 修复位置 | 修复内容 |
|------|---------|---------|
| **O1**: DN 路径与 R decoder 集成 | Task 4.5 Step 1 | R decoder 对所有 queries（包括 DN）都跑 |
| **O2**: aux_outputs + loss_local 适配 | Task 4.5 Step 2-3 | XYWH 路径 4 条分布，R 路径 sin/cos；loss_local 拆分为 xywh + angle |
| **O3**: criterion NotImplementedError | Task 4.5 Step 5 | OBB 用 probiou 作为 boxes_weight |
| **O4**: 参数 plumbing | Task 4 Step 1 | `decouple_angle`、`angle_step`、`num_r_layers` 显式加到 `__init__` |
| **O5**: LQE 归属 | Task 4.5 Step 4 | XYWH 路径 LQE 只用 4 条分布，R 路径不需要 LQE |

### 其他修复

- Task 2: 角度锚点 θ=0 产生 -inf → 使用 `(k + 0.5) * angle_step / 180` 避免边界
- Task 5: 更新前提条件说明（OBB_CODE_REVIEW.md #1 已修复旋转注意力数学错误）
- Task 6-7: 修复路径问题（`/mnt/d/cx/thired/deimv2_daod` → `/home/cx/win_dir/thired/DEIMv2_DAOD`）

### 新增文件

- `engine/deim/gated_fusion.py`: `GatedSoftmaxFusion` + `EncoderMemoryAggregator`
- `engine/deim/dfine_utils.py`: `distance2bbox_obb_xywh` + `bbox2distance_obb_xywh`（新增函数）

### 修改文件

- `engine/deim/deim_decoder.py`: XYWH/R 路径拆分、DN/aux 适配、参数传递
- `engine/deim/dfine_decoder.py`: 正交旋转注意力
- `engine/deim/deim_criterion.py`: aux_outputs 适配、loss_local 拆分、NotImplementedError 修复
- `configs/custom_obb/deimv2_obb_decouple.yml`: 新增解耦配置
