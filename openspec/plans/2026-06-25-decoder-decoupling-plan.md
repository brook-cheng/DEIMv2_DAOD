# DEIMv2-OBB Decoder 解耦实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 DEIMv2-OBB decoder 拆分为 XYWH 路径（6层）和 R 路径（6层），通过 Gated Softmax Fusion 桥接，并引入 angle_step 多角度锚点和正交旋转注意力。

**Architecture:** XYWH decoder 保留 D-FINE refinement + label 分类；R decoder 独立预测角度偏移 (ε,η)，每层 Gate Fusion 融合 xywh 特征 + 角度特征 + encoder memory。encoder memory 双路注入。

**Tech Stack:** PyTorch, torch.nn, DEIMv2 engine

## Global Constraints

- Mode A 优先（label 留在 xywh 路径），Mode B 后续实现
- angle_step 默认 10°，可通过 YAML 配置
- 不破坏 HBB 模式（`box_mode="hbb"` 时行为不变）
- 保持与现有 criterion/matcher/postprocessor 接口兼容
- Python 环境：`/home/cx/apps/miniconda3/envs/deimv2/`

---

## File Structure

```
engine/deim/
  deim_decoder.py        # MODIFY: anchor multi-angle, XYWH path, R path, Gate Fusion
  dfine_decoder.py       # MODIFY: orthogonal rotation attention
  dfine_utils.py         # MODIFY: xywh-only bbox ops (remove angle from xywh path)
  deim_utils.py          # REUSE: existing Gate class (line 70)
  gated_fusion.py        # CREATE: GatedSoftmaxFusion module
configs/custom_obb/
  deimv2_obb_decouple.yml  # CREATE: new config with decouple parameters
```

---

### Task 1: Gated Softmax Fusion 模块

**Files:**
- Create: `engine/deim/gated_fusion.py`

**Interfaces:**
- Produces: `class GatedSoftmaxFusion(nn.Module)` — `forward(srcs: list[Tensor], query: Tensor) → Tensor`

- [ ] **Step 1: 创建模块**

```python
"""Gated Softmax Fusion — 多源特征动态加权融合。

对 N 个输入源做 token 级动态加权融合:
  cat = concat([query, src_0, src_1, ..., src_{N-1}])
  w = softmax(MLP(cat))           # [B, num_tokens, N]
  fused = Σ w_i * src_i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedSoftmaxFusion(nn.Module):
    """多源 Gated Softmax 融合。

    用法:
        fusion = GatedSoftmaxFusion(d_model=256, n_sources=3, hidden_dim=128)
        fused = fusion([enc_memory, xywh_feat, r_feat], query=r_query)
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
            srcs: list of N tensors, each [B, num_tokens, d_model]
            query: [B, num_tokens, d_model] — 用于生成融合权重的查询
        Returns:
            fused: [B, num_tokens, d_model]
        """
        assert len(srcs) == self.n_sources, f"Expected {self.n_sources} sources, got {len(srcs)}"

        cat = torch.cat([query] + srcs, dim=-1)                    # [B, N_tok, (n_src+1)*d]
        weights = torch.softmax(self.weight_net(cat), dim=-1)      # [B, N_tok, n_src]

        fused = torch.zeros_like(srcs[0])
        for i, src in enumerate(srcs):
            fused = fused + weights[..., i:i + 1] * src

        return self.output_proj(fused)
```

- [ ] **Step 2: 单元测试**

```bash
cd /mnt/d/cx/thired/deimv2_daod && python3 -c "
import torch
from engine.deim.gated_fusion import GatedSoftmaxFusion
m = GatedSoftmaxFusion(256, 3, 128)
x = [torch.randn(2, 100, 256) for _ in range(3)]
q = torch.randn(2, 100, 256)
y = m(x, q)
print(f'Output shape: {y.shape}')  # Expected: [2, 100, 256]
print('PASS')
"
```

预期: `Output shape: torch.Size([2, 100, 256])` + `PASS`

---

### Task 2: Anchor 多角度生成

**Files:**
- Modify: `engine/deim/deim_decoder.py`（`_generate_anchors` 方法，约 622-676 行）

**Interfaces:**
- Modifies: `_generate_anchors()` — OBB 分支从固定 θ=0.5 改为根据 `angle_step` 生成多角度锚点

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
# 多角度锚点：每个网格位置生成 N_theta = 180 / angle_step 个角度变体
angle_step_deg = getattr(self, "angle_step", 10)  # 默认 10°
n_angles = 180 // angle_step_deg
for k in range(n_angles):
    theta_val = (k * angle_step_deg) / 180.0  # 归一化到 [0, 1)
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
cd /mnt/d/cx/thired/deimv2_daod && python3 -c "
import sys; sys.path.insert(0, '.')
from engine.core import YAMLConfig
cfg = YAMLConfig('configs/custom_obb/synthetic_exp_020.yml')
# 检查 anchor 数量（需手动看 _generate_anchors 输出）
print('Test anchor generation manually with angle_step=10')
"
```

---

### Task 3: XYWH Decoder 路径调整

**Files:**
- Modify: `engine/deim/deim_decoder.py`（decoder layer forward 约 207-280 行）

**Interfaces:**
- 现有 decoder layer 的 box_head 输出从 6 维 `(α,β,γ,δ,ε,η)` 改为 4 维 `(α,β,γ,δ)`
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

XYWH 路径的 box_head 输出从 6 改为 4：

```python
# 原代码（deim_decoder.py 约 840 行，box_head 定义处）：
# self.bbox_head = MLP(hidden_dim, hidden_dim, num_reg_dist * 6, 3)  # 6 用于 OBB

# 改为条件：
if self.box_mode == "obb" and getattr(self, "decouple_angle", False):
    self.bbox_head_xywh = MLP(hidden_dim, hidden_dim, num_reg_dist * 4, 3)  # 仅 xywh
else:
    self.bbox_head = MLP(hidden_dim, hidden_dim, num_reg_dist * 6, 3)  # 兼容原逻辑
```

- [ ] **Step 3: 调整 distance2bbox_obb 调用**

`distance2bbox_obb` 现在只接收 xywh 的偏移（4 维），角度部分由 R 路径补充：

```python
# 原：inter_ref_bbox = distance2bbox_obb(ref_points_scaled, integral(pred_corners, project))
# 改为：
inter_xywh = distance2bbox_obb_xywh(ref_xywh_scaled, integral(pred_corners_xywh[:4], project))
# 角度从 reference_points 保持不变：
inter_ref_bbox = torch.cat([inter_xywh, ref_r], dim=-1)  # 拼接回 5 维
```

---

### Task 4: R Decoder 路径

**Files:**
- Modify: `engine/deim/deim_decoder.py`（新增 R decoder layers 和 pre_angle_head）

- [ ] **Step 1: 添加 R decoder layers**

在 `__init__` 中，当 `decouple_angle=True` 时，额外创建 6 个 R decoder layer：

```python
if self.box_mode == "obb" and getattr(self, "decouple_angle", False):
    self.num_r_layers = 6
    self.r_layers = nn.ModuleList([
        copy.deepcopy(self.decoder_layer) for _ in range(self.num_r_layers)
    ])
    self.pre_angle_head = MLP(hidden_dim, hidden_dim, 1, 3)
    self.angle_heads = nn.ModuleList([
        MLP(hidden_dim, hidden_dim, num_reg_dist * 2, 3)  # (ε,η) per layer
        for _ in range(self.num_r_layers)
    ])
    self.gate_fusions = nn.ModuleList([
        GatedSoftmaxFusion(d_model=hidden_dim, n_sources=3, hidden_dim=128)
        for _ in range(self.num_r_layers)
    ])
```

- [ ] **Step 2: 实现 R decoder 前向**

R path 的 forward 逻辑：

```python
def forward_r_decoder(self, xywh_features, r_init, encoder_memory, ...):
    r_current = r_init  # 初始角度参考点
    for layer_idx in range(self.num_r_layers):
        # 1. R decoder layer 前向（self-attn + cross-attn）
        r_features = self.r_layers[layer_idx](r_current, encoder_memory, ...)

        # 2. 角度偏移预测
        angle_delta = self.angle_heads[layer_idx](r_features)  # (ε,η)

        # 3. Gate Fusion: 融合 xywh 特征 + r 特征 + encoder memory
        r_fused = self.gate_fusions[layer_idx](
            [xywh_features, r_features, encoder_memory],
            query=r_features
        )

        # 4. 更新角度参考点（类似 FDR 的 residual update）
        r_current = r_current + angle_delta
        # 保持 θ 在 [0, π) 范围
        r_current = r_current % math.pi

    return r_current, r_features
```

- [ ] **Step 3: 集成到主 forward**

在主 `forward()` 中：

```python
# XYWH decoder 前向（Task 3 产出 xywh 特征和中间参考点）
xywh_output, xywh_features = self.forward_xywh_decoder(...)

# R decoder 前向
r_initial = anchors[..., 4:5] * math.pi  # 初始角度
r_output, r_features = self.forward_r_decoder(
    xywh_features=xywh_features,  # 来自 xywh 最后层
    r_init=r_initial,
    encoder_memory=encoder_memory
)

# 合并输出：xywh + r
pred_boxes = torch.cat([xywh_output, r_output], dim=-1)
```

---

### Task 5: 正交旋转注意力

**Files:**
- Modify: `engine/deim/dfine_decoder.py`（MSDeformableAttention 的 OBB 5 维分支，约 167-184 行）

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
cd /mnt/d/cx/thired/deimv2_daod && python3 -c "
import torch, math
bs, Lq, H = 1, 5, 8
ref = torch.randn(bs, Lq, 1, 5)
ref[..., 4:] = 0.25  # θ = π/4
# 手动验证两个 head 的旋转矩阵是否差 π/2
print('Orthogonal attention test')
print('Angle for head 0:', ref[0,0,0,4].item() * math.pi)
print('Angle for head H/2+1:', (ref[0,0,0,4].item() + 0.5) * math.pi)
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
cd /mnt/d/cx/thired/deimv2_daod && python3 -c "
import sys; sys.path.insert(0, '.')
from engine.core import YAMLConfig
# 加载 HBB 配置，确认 box_mode='hbb' 时行为不变
cfg = YAMLConfig('configs/custom_obb/deimv2_obb_sp.yml')
print('HBB compatibility OK')
"
```

- [ ] **Step 3: OBB 解耦启动测试**

```bash
cd /mnt/d/cx/thired/deimv2_daod && python train.py --config configs/custom_obb/deimv2_obb_decouple.yml
```

预期：模型加载成功，开始训练。因是新架构，loss 可能波动，但不应 crash。

---

### Task 7: 合成数据集验证

- [ ] **Step 1: 在 density_020 上做短训**

```bash
cd /mnt/d/cx/thired/deimv2_daod
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
