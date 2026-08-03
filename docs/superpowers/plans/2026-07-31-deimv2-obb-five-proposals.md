# DEIMv2-OBB 五项改进实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过五项改进提升 DEIMv2-OBB 的角度预测精度：启用特征融合、ADR 去角度损失、角度范围调整、多角度锚点、先角度后位置。

**Architecture:** 每项改进独立可消融，以 `configs/custom_obb/dlzdt/sp_fz_rep0_nloss.yml`（angle_rep=0）为基础对照组。改进按风险递增实施，每项完成后做训练验证。

**Tech Stack:** PyTorch, engine.deim.*, configs/custom_obb/dlzdt/

## Global Constraints

- 基线配置：`configs/custom_obb/dlzdt/sp_fz_rep0_nloss.yml`（angle_rep=0, box_mode=obb）
- 消融配置命名：`sp_fz_rep{N}_nloss_{改进标识}.yml`，每个提案一个配置文件
- 所有改动必须同时支持 rep0/rep1/rep2/rep3 四种 angle_rep
- HBB 路径不受影响——所有改动在 `box_mode == "obb"` 分支内
- `periodic_angle_distance` 的周期是 π，不因角度范围变化而改变——它是数学属性
- 角度量纲审计表见本计划末尾附录

---

## 提案实施顺序

| 顺序 | 提案 | 改动量 | 风险 | 配置标识 |
|---|---|---|---|---|
| Task 1 | 提案 5：启用 GatedSoftmaxFusion | 最小 | 低 | `_fused` |
| Task 2 | 提案 3：ADR 去角度损失 | 小 | 低 | `_noangle` |
| Task 3 | 提案 1：角度范围 [-π/4, 3π/4) | 中 | 中 | `_arange` |
| Task 4 | 提案 2：多角度锚点 | 中 | 中高 | `_mangle` |
| Task 5 | 提案 4：先角度后位置 | 大 | 高 | `_afp` |

---

### Task 1: 启用 GatedSoftmaxFusion（提案 5）

**Files:**
- Modify: `engine/deim/deim_decoder.py:124-197`（TransformerDecoder.__init__）
- Modify: `engine/deim/deim_decoder.py:328-374`（TransformerDecoder.forward 中被注释的 fusion 代码）
- Create: `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_fused.yml`

**Interfaces:**
- Consumes: `engine.deim.gated_fusion.GatedSoftmaxFusion`（已定义，line 1-27）
- Produces: box/angle 分支特征融合，影响 rep2/rep3 的 `decouple_angle_layers` 路径

**背景：** `GatedSoftmaxFusion` 已在 `deim_decoder.py:190-197` 定义但未启用。fusion 代码在 `forward` 中被注释（line 364-367）。需要取消注释并验证 `output`（box 分支）和 `dec_angle_output`（angle 分支）的形状匹配。

- [ ] **Step 1: 阅读当前被注释的 fusion 代码**

`deim_decoder.py:364-367`（当前注释状态）:
```python
# if i < len(self.gate_fusions):
#     offset_output = self.gate_fusions[i](
#         [output, offset_output], query=offset_output
#     )
```

`gate_fusions` 在 `__init__` line 190-196 已创建：
```python
self.gate_fusions = nn.ModuleList([
    GatedSoftmaxFusion(d_model=hidden_dim, n_sources=2, hidden_dim=hidden_dim)
    for _ in range(self.num_decouple_layers - 1)
])
```

- [ ] **Step 2: 取消注释并修正变量名**

当前 `forward` 中 angle 分支输出变量名是 `dec_angle_output`，不是 `offset_output`。修改 line 364-367 为：

```python
if i < len(self.gate_fusions):
    dec_angle_output = self.gate_fusions[i](
        [output, dec_angle_output], query=dec_angle_output
    )
```

注意：fusion 融合的是 `output`（box 分支当前层输出）和 `dec_angle_output`（angle 分支当前层输出），用 `dec_angle_output` 作为 query。

- [ ] **Step 3: 添加配置开关**

在 `DEIMTransformer.__init__` 添加参数 `use_gate_fusion: bool = False`。在 `TransformerDecoder.__init__` 传递该参数。在 `forward` 中用 `if self.use_gate_fusion and i < len(self.gate_fusions):` 守护。

- [ ] **Step 4: 创建消融配置**

基于 `sp_fz_rep0_nloss.yml` 创建 `sp_fz_rep0_nloss_fused.yml`，在 `DEIMTransformer` 下添加：
```yaml
DEIMTransformer:
  ...
  angle_rep: 2  # 或 3，需要 decouple_angle_layers 才有 fusion
  use_gate_fusion: true
```

注意：rep0/rep1 没有 `decouple_angle_layers`，fusion 无效。此配置仅对 rep2/rep3 有意义。

- [ ] **Step 5: 验证**

运行 `python -c "import py_compile; py_compile.compile('engine/deim/deim_decoder.py', doraise=True); print('OK')"` 确认语法正确。

- [ ] **Step 6: 提交**

```bash
git add engine/deim/deim_decoder.py configs/custom_obb/dlzdt/sp_fz_rep2_nloss_fused.yml
git commit -m "feat(decoder): enable GatedSoftmaxFusion for box/angle branch interaction"
```

---

### Task 2: ADR 去角度损失（提案 3）

**Files:**
- Modify: `configs/custom_obb/dlzdt/sp_fz_rep2_nloss.yml`（或新建 `_noangle` 变体）
- 不需要修改 `deim_criterion.py` 或 `matcher.py` 代码——纯配置调整

**Interfaces:**
- Consumes: 现有 `deim_criterion.py` 的 `use_yolo_probiou`/`use_yolo_angle`/`keep_kld` 开关
- Produces: rep2 配置中移除角度相关 loss/cost

**背景：** rep2（angle_rep=2）使用 ADR 表示 `(cx,cy,w,h,ε,η)`。角度通过顶点偏移间接编码。当前的 `use_yolo_angle: true` 添加了一个显式角度 loss，但 ADR 已经通过 (ε,η) 编码了朝向。移除显式角度 loss 让训练完全依赖 ADR 残差 + KLD/ProbIoU。

- [ ] **Step 1: 创建无角度 loss 的 rep2 配置**

基于 `sp_fz_rep2_nloss.yml` 创建 `sp_fz_rep2_nloss_noangle.yml`：
```yaml
DEIMCriterion:
  weight_dict: {loss_mal: 1, loss_bbox: 5, loss_probiou: 5, loss_kld: 2, loss_fgl: 0.15, loss_ddf: 1.5}
  losses: ['mal', 'boxes', 'local']
  use_yolo_probiou: true
  use_yolo_angle: false      # ← 关闭显式角度 loss
  keep_kld: true
  angle_lambda: 0.0           # ← 角度 lambda 归零
  reg_max: 32
  box_mode: obb
  obbox_rep_dim: 6
  matcher:
    type: HungarianMatcher
    weight_dict: {cost_class: 2, cost_bbox: 5, cost_probiou: 5, cost_chamfer: 2, late_cost_bbox: 0.25}
    # cost_angle 移除（不列出即默认 0）
    angle_order_alpha: 1.0
    alpha: 0.25
    gamma: 2.0
    change_matcher: False
    iou_order_alpha: 4.0
    matcher_change_epoch: 112
    box_mode: obb
```

关键差异：
- `weight_dict` 中移除 `loss_angle: 3`
- `use_yolo_angle: false`
- `angle_lambda: 0.0`
- matcher `weight_dict` 中移除 `cost_angle: 3`

- [ ] **Step 2: 验证配置可被正确解析**

运行：
```bash
python -c "
import sys; sys.path.insert(0, '.')
from engine.core.yaml_config import YAMLConfig
cfg = YAMLConfig('configs/custom_obb/dlzdt/sp_fz_rep2_nloss_noangle.yml')
print('Config loaded successfully')
print('use_yolo_angle:', cfg.yaml_cfg.get('DEIMCriterion', {}).get('use_yolo_angle'))
"
```

- [ ] **Step 3: 提交**

```bash
git add configs/custom_obb/dlzdt/sp_fz_rep2_nloss_noangle.yml
git commit -m "feat(config): add rep2 ablation without explicit angle loss"
```

---

### Task 3: 角度范围 [-π/4, 3π/4)（提案 1）

**Files:**
- Modify: `engine/deim/obb_geometry.py`（xyxyxyxy_to_xywhr, external_rect_to_oriented_box, periodic_angle_distance）
- Modify: `engine/deim/deim_decoder.py`（所有 `* torch.pi` 和 `/ torch.pi` 的角度缩放）
- Modify: `engine/deim/deim_criterion.py`（L1 角度归一化）
- Modify: `engine/deim/matcher.py`（angle_factor）
- Modify: `engine/deim/dfine_utils.py`（distance2bbox_obb 角度细化）
- Modify: `engine/deim/dfine_decoder.py`（MSDeformableAttention 角度使用）
- Modify: `engine/deim/denoising.py`（DN 角度归一化）
- Modify: `engine/deim/postprocessor.py`（输出角度注释）
- Create: `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_arange.yml`

**核心设计决策：内部表示方案 A**

decoder 内部保持 `[0, 1]`（sigmoid 空间）。**输出边界**从 `θ * π` 改为 `(θ - 0.25) * π`。这样 raw=0 时 θ=π/4=45°（新范围中点），raw=1 时 θ=3π/4=135°，raw=0.25 时 θ=0°（水平）。

**外部范围 `[-π/4, 3π/4)` 的属性：**
- 宽度 = π（与 `[0, π)` 相同）
- 中点 = π/4 = 45°
- 回绕点在 ±45°/135°（w≈h 的几何歧义区域）
- 水平（0°）和垂直（90°）都在范围内部，远离边界

- [ ] **Step 1: 修改 obb_geometry.py 的角度归一化**

`xyxyxyxy_to_xywhr`（line 99）当前：
```python
theta = torch.atan2(w_dy, w_dx) % torch.pi
```
改为：
```python
theta = torch.atan2(w_dy, w_dx)
# 折叠到 [-pi/4, 3*pi/4)
theta = torch.remainder(theta + torch.pi / 4, torch.pi) - torch.pi / 4
```

`external_rect_to_oriented_box`（line 236）同样修改。

- [ ] **Step 2: 确认 periodic_angle_distance 不需要修改**

`periodic_angle_distance`（line 18-39）使用 `% torch.pi` 和 `torch.minimum(d, torch.pi - d)`。这是纯数学周期运算，与角度范围的偏移无关——周期始终是 π。**不需要修改。**

验证：
```python
# 在 [-pi/4, 3pi/4) 范围下，两个等价角度的周期距离仍然为 0
a = -torch.pi / 4       # -45°
b = 3 * torch.pi / 4    # 135° (= -45° + 180°)
# periodic_angle_distance(a, b) 应该 ≈ 0
```

- [ ] **Step 3: 修改 deim_decoder.py 的角度缩放**

所有 `[..., 4:] * torch.pi` 改为 `(...[..., 4:] - 0.25) * torch.pi`。
所有 `[..., 4:] / torch.pi` 改为 `(...[..., 4:] / torch.pi) + 0.25`。

具体行号：
- line 383: `theta_scale[..., 4] *= torch.pi` → 保留（这是在 decoder 内部做 [0,1]→[0,π] 给 distance2bbox_obb，distance2bbox_obb 仍然接收 [0,π]）
- line 395: `inter_ref_bbox[..., 4:] / torch.pi` → 改为 `(inter_ref_bbox[..., 4:] - (-torch.pi/4)) / torch.pi`（从 [-π/4,3π/4) 映射回 [0,1]）

**重要：** decoder 内部传递用 `[0, 1]` sigmoid 空间。输出到 criterion/matcher/postprocessor 时用 `[-π/4, 3π/4)`。转换公式：
- 内部→外部：`θ_ext = (θ_int - 0.25) * π`
- 外部→内部：`θ_int = θ_ext / π + 0.25`

修改 line 1002（encoder aux 输出）:
```python
# 原: enc_topk_bboxes[..., 4:] * torch.pi
# 新: (enc_topk_bboxes[..., 4:] - 0.25) * torch.pi
enc_topk_bboxes = torch.cat(
    [enc_topk_bboxes[..., :4], (enc_topk_bboxes[..., 4:] - 0.25) * torch.pi],
    dim=-1,
)
```

修改 line 1139-1145（decoder 输出）:
```python
# 原: out_bboxes[..., 4:] * torch.pi
# 新: (out_bboxes[..., 4:] - 0.25) * torch.pi
out_bboxes = torch.cat(
    [out_bboxes[..., :4], (out_bboxes[..., 4:] - 0.25) * torch.pi], dim=-1
)
out_refs = torch.cat(
    [out_refs[..., :4], (out_refs[..., 4:] - 0.25) * torch.pi], dim=-1
)
pre_bboxes = torch.cat(
    [pre_bboxes[..., :4], (pre_bboxes[..., 4:] - 0.25) * torch.pi], dim=-1
)
```

修改 line 383/395（decoder 内部 [0,1]↔[-π/4,3π/4)）:
```python
# line 383: theta_scale 用于将 ref_points_initial 的 [0,1] 角度转到 [0,π] 给 distance2bbox_obb
# distance2bbox_obb 接收 θ ∈ [0,π]，所以这里保持 * π 不变
# 但 ref_points_initial 本身来自 decoder 内部，是 [0,1] 空间
# distance2bbox_obb 输出的 θ 是 [0,π]，需要转回 [-π/4,3π/4)
# line 395: 原 / torch.pi 转回 [0,1]
# 现在需要从 [0,π] 先转到 [-π/4,3π/4) 再除 π + 0.25 转回 [0,1]
# 但 distance2bbox_obb 内部的 % torch.pi 已经保证 [0,π)
# 所以: [0,π) → [-π/4,3π/4) 的映射: θ_new = θ - π/4 if θ >= 3π/4 else θ
# 实际上 [0,π) 和 [-π/4,3π/4) 差了一个 π/4 的偏移
# 更简单: distance2bbox_obb 内部也改为输出 [-π/4,3π/4)
```

- [ ] **Step 4: 修改 dfine_utils.py 的角度处理**

`distance2bbox_obb` line 245（rep3 路径）：
```python
# 原: n_obboxes_angle = (points[..., 4:5] + distance[..., 4:5] / reg_scale) % torch.pi
# 新: 折叠到 [-π/4, 3π/4)
raw = points[..., 4:5] + distance[..., 4:5] / reg_scale
n_obboxes_angle = torch.remainder(raw + torch.pi / 4, torch.pi) - torch.pi / 4
```

- [ ] **Step 5: 修改 deim_criterion.py**

`use_yolo_angle`/`use_yolo_probiou` 路径（line 295-309）：这些使用 `yolo_obb_loss.py` 中的 `sin(2Δθ)²`，周期是 π/2，**与角度范围无关，不需要修改**。

`periodic_angle_flag` 路径（line 314-323）：
```python
# 原: angle_term = self.lambda_angle * periodic_angle_distance(...) / torch.pi
# periodic_angle_distance 输出 [0, π/2]，除以 π 得到 [0, 0.5]
# 这个归一化与角度范围无关——不需要修改
```

非周期路径（line 328-340）：
```python
# 原: src_boxes[..., 4:] / torch.pi → [0,1]
# 新: (src_boxes[..., 4:] + torch.pi/4) / torch.pi → [0,1]
# 把 [-π/4, 3π/4) 映射到 [0, 1)
src_boxes_l1 = torch.cat(
    [src_boxes[..., :4], (src_boxes[..., 4:] + torch.pi / 4) / torch.pi], dim=-1
)
target_boxes_l1 = torch.cat(
    [target_boxes[..., :4], (target_boxes[..., 4:] + torch.pi / 4) / torch.pi], dim=-1
)
```

- [ ] **Step 6: 修改 matcher.py**

`angle_factor=math.pi`（line 41）：这是用于 L1 成本的归一化因子。
```python
# 原: angle_factor=math.pi
# 新: 保持 math.pi — 周期不变，宽度仍是 π
```
**不需要修改**——angle_factor 是角度范围的宽度（π），不是偏移。

- [ ] **Step 7: 修改 denoising.py**

line 111:
```python
# 原: input_query_bbox[..., 4] = input_query_bbox[..., 4] / torch.pi
# 新: 映射 [-π/4, 3π/4) → [0, 1]
input_query_bbox[..., 4] = (input_query_bbox[..., 4] + torch.pi / 4) / torch.pi
```

- [ ] **Step 8: 修改 postprocessor.py**

line 74 注释更新：
```python
# θ 不变(归一化到[-π/4, 3π/4))
```
代码不需要改——postprocessor 只做像素缩放，θ 不变。

- [ ] **Step 9: 修改 anchor 初始值**

`_generate_anchors` line 901:
```python
# 原: r = 0.5 * torch.ones(...)  → sigmoid(0.5) → θ = 0.5*π = π/2 = 90°
# 新: r = 0.5 * torch.ones(...)  → sigmoid(0.5) → θ = (0.5-0.25)*π = π/4 = 45°（中点）
# 不需要改！0.5 在新映射下恰好是 45°（新范围中点）
```

- [ ] **Step 10: 创建消融配置并测试**

创建 `sp_fz_rep0_nloss_arange.yml`（与基线相同，只改注释说明角度范围）。

运行测试：
```bash
python test/test_obb_roundtrip.py  # 验证几何转换仍正确
python test/test_obb_error_classify.py  # 验证分类器仍工作
```

- [ ] **Step 11: 提交**

```bash
git add engine/deim/obb_geometry.py engine/deim/deim_decoder.py engine/deim/deim_criterion.py engine/deim/dfine_utils.py engine/deim/denoising.py engine/deim/postprocessor.py
git commit -m "feat(obb): change angle range from [0,pi) to [-pi/4, 3pi/4)"
```

---

### Task 4: 多角度锚点（提案 2）

**Files:**
- Modify: `engine/deim/deim_decoder.py:_generate_anchors`（line 847-923）
- Create: `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_mangle.yml`

**背景：** 为每个空间位置生成多个角度候选锚点，增加初始角度多样性。步长 `angle_step` 控制角度候选数量。

- [ ] **Step 1: 添加 angle_step 参数**

`DEIMTransformer.__init__` 添加：
```python
angle_step: float = 0.0,  # 0 = 禁用（单角度），>0 = 步长（如 0.1 = 10 个候选）
```

- [ ] **Step 2: 修改 _generate_anchors**

在 OBB 非 rep2 路径（line 890-914），当 `angle_step > 0` 时：
```python
if self.angle_step > 0:
    n_angles = int(1.0 / self.angle_step)
    angle_candidates = torch.arange(n_angles, dtype=dtype, device=device) * self.angle_step
    # angle_candidates: [0, step, 2*step, ..., (n-1)*step]
    
    # 对每个空间位置扩展 n_angles 个角度候选
    grid_xy_exp = grid_xy.unsqueeze(-2).expand(1, h, w, n_angles, 2)
    wh_exp = wh.unsqueeze(-2).expand(1, h, w, n_angles, 2)
    r_exp = angle_candidates.reshape(1, 1, 1, n_angles, 1).expand(1, h, w, n_angles, 1)
    
    lvl_anchors = torch.concat([grid_xy_exp, wh_exp, r_exp], dim=-1)
    lvl_anchors = lvl_anchors.reshape(1, h * w * n_angles, self._num_box_dof)
else:
    # 原有逻辑（单角度 r=0.5）
    r = 0.5 * torch.ones(...)
    ...
```

- [ ] **Step 3: 评估内存影响**

640×640 输入：8400 anchors × 10 = 84000 anchors。encoder score head 对 84000 个位置计算 logits——在 256 hidden_dim 下约 84000×256 = 21.5M floats = 86MB（fp32）。可接受。

- [ ] **Step 4: 创建消融配置**

```yaml
DEIMTransformer:
  ...
  angle_step: 0.1  # 10 个角度候选
```

- [ ] **Step 5: 提交**

```bash
git add engine/deim/deim_decoder.py configs/custom_obb/dlzdt/sp_fz_rep0_nloss_mangle.yml
git commit -m "feat(decoder): add multi-angle anchor generation with configurable step"
```

---

### Task 5: 先角度后位置（提案 4）

**Files:**
- Modify: `engine/deim/deim_decoder.py:TransformerDecoder.forward`（line 258-414）
- Modify: `engine/deim/deim_decoder.py:DEIMTransformer.__init__`（query_pos_head 维度）
- Create: `configs/custom_obb/dlzdt/sp_fz_rep3_nloss_afp.yml`

**背景：** 当前 decoder 先跑 box 分支（self-attn + cross-attn → bbox_head），再跑 angle 分支（decouple_angle_layers）。提案 4 反转顺序：先跑 angle 分支获得预测角度，再把角度融入 box 分支的 query_pos_embed 和 ref_points_input。

**关键改动：**
1. `query_pos_head` 从 `MLP(4, ...)` 扩展为 `MLP(5, ...)`（加入角度通道）
2. `ref_points_input` 从 4 通道扩展到 5 通道
3. decoder forward 循环中先执行 angle 分支，再执行 box 分支

- [ ] **Step 1: 扩展 query_pos_head 输入维度**

`DEIMTransformer.__init__` line 617-625:
```python
# rep2/rep3 当前: num_query_pos_in = 4
# 改为: 当 use_angle_first=True 时，num_query_pos_in = 5
if use_angle_first:
    num_query_pos_in = 5  # (cx, cy, w, h, θ_predicted)
```

- [ ] **Step 2: 修改 TransformerDecoder.forward 循环顺序**

当前顺序（line 284-414）：
```
for each layer:
    1. ref_points_input = ref_points_detach[:, :, :4]  # 4 通道
    2. query_pos_embed = query_pos_head(ref_points_detach[:, :, :4])
    3. output = layer(output, ref_points_input, ...)  # box self+cross attn
    4. pre_bbox_head → pred_corners → inter_ref_bbox
    5. decouple_angle_layers → dec_angle_output → dec_angle_initial
```

新顺序：
```
for each layer:
    1. decouple_angle_layers → dec_angle_output → dec_angle_initial  ← 先跑角度
    2. ref_points_with_angle = cat([ref_points_detach[:,:,:4], dec_angle_initial], dim=-1)  # 5 通道
    3. ref_points_input = ref_points_with_angle.unsqueeze(2)
    4. query_pos_embed = query_pos_head(ref_points_with_angle)  # 5→hidden_dim
    5. output = layer(output, ref_points_input, ...)  # box self+cross attn（含角度参考点）
    6. pre_bbox_head → pred_corners → inter_ref_bbox
```

- [ ] **Step 3: 处理 MSDeformableAttention 的 5 通道参考点**

`dfine_decoder.py:150-199`：当 `reference_points.shape[-1] == 5` 时，执行旋转注意力。box 分支的 ref_points_input 变成 5 通道后会进入旋转采样路径。需要确认 angle 通道的值在 `[0, 1]` 空间（sigmoid 后）。

- [ ] **Step 4: 创建消融配置**

```yaml
DEIMTransformer:
  ...
  angle_rep: 3
  use_angle_first: true
  use_gate_fusion: true  # 可选：与提案 5 叠加
```

- [ ] **Step 5: 提交**

```bash
git add engine/deim/deim_decoder.py configs/custom_obb/dlzdt/sp_fz_rep3_nloss_afp.yml
git commit -m "feat(decoder): angle-first then position prediction order"
```

---

## 附录：角度量纲全链路审计表

| 文件 | 行号 | 当前代码 | 当前量纲 | 提案 1 后 | 是否需改 |
|---|---|---|---|---|---|
| `obb_geometry.py:99` | `theta = atan2(...) % π` | `[0, π)` | `[-π/4, 3π/4)` | ✅ |
| `obb_geometry.py:236` | `theta = atan2(...) % π` | `[0, π)` | `[-π/4, 3π/4)` | ✅ |
| `obb_geometry.py:35-36` | `periodic_angle_distance` | 周期 π | 周期 π（不变） | ❌ |
| `deim_decoder.py:383` | `theta_scale[..., 4] *= π` | [0,1]→[0,π] | 保持（distance2bbox_obb 内部用 [0,π]） | ❌ |
| `deim_decoder.py:395` | `[..., 4:] / π` | [0,π]→[0,1] | `(θ+π/4)/π` → [0,1] | ✅ |
| `deim_decoder.py:1002` | `[..., 4:] * π` | [0,1]→[0,π] | `(θ-0.25)*π` → [-π/4,3π/4) | ✅ |
| `deim_decoder.py:1139-1145` | `* π` | [0,1]→[0,π] | `(θ-0.25)*π` | ✅ |
| `deim_decoder.py:901` | `r = 0.5` | sigmoid(0.5)=0.5→90° | sigmoid(0.5)=0.5→(0.5-0.25)π=45° | ❌（恰好是新中点） |
| `dfine_utils.py:245` | `% π` | [0,π) 折叠 | `remainder(θ+π/4, π) - π/4` | ✅ |
| `dfine_utils.py:311-314` | `periodic_angle_distance` | 周期 π | 周期 π（不变） | ❌ |
| `dfine_decoder.py:177` | `angle = ref[..., 4:5] * π` | [0,1]→[0,π] | 保持（decoder 内部仍是 [0,1]→[0,π]） | ❌ |
| `deim_criterion.py:323` | `/ π` | [0,π]→[0,1] | periodic 路径不变 | ❌ |
| `deim_criterion.py:329,334` | `/ π` | [0,π]→[0,1] | `(θ+π/4)/π` | ✅ |
| `matcher.py:41` | `angle_factor=π` | π | π（宽度不变） | ❌ |
| `denoising.py:111` | `/ π` | [0,π]→[0,1] | `(θ+π/4)/π` | ✅ |
| `postprocessor.py:74` | θ 不变 | [0,π] | [-π/4,3π/4) | ❌（只缩放像素） |
| `yolo_obb_loss.py:112,152` | `round(δ/π)*π` | 周期 π | 周期 π（不变） | ❌ |

## Final Verification Wave

每个 Task 完成后：
- [ ] 语法检查：`python -c "import py_compile; py_compile.compile('<file>', doraise=True)"`
- [ ] 几何测试：`python test/test_obb_roundtrip.py`
- [ ] 配置解析：加载新配置确认无报错
- [ ] 1-step smoke：用新配置实例化模型，跑一个 forward pass

## Commit Strategy

每个 Task 一个 commit，不自动提交。用户管理 git。

## Success Criteria

- 5 个消融配置文件创建完成
- 所有几何测试通过
- 每个改动可通过配置开关启用/禁用
- HBB 路径不受影响
- `periodic_angle_distance` 不被修改
