# DEIMv2 OBB decoder 私有 shifted 角度编码设计

日期：2026-08-05

状态：设计已确认；配置面仅保留 `proportional` 与 `shifted` 两种模式。

关联文档：

- 既有契约设计：`docs/superpowers/specs/2026-08-05-obb-angle-contract-simplification-design.md`
  （公开物理域 `[0, π)` / decoder 私有等比归一域 `[0,1)` / loss 内部规范域，边界转换显式）
- 审计记录：`docs/superpowers/review/2026-08-05-obb-angle-units-final-audit.md`
- 上一轮修复：A1/A2/A3（rep1 角度残差不再乘 π；rep2 decoder 参考角物理弧度；四表示输出域统一）

## 1. 背景与动机

当前 decoder 内部的绝对角度 reference 使用**等比归一** `theta_norm = theta_phys_rad / π ∈ [0,1)`，
通过 `sigmoid(logit)` 参数化：

- 物理角 `0°` 与 `180°`（等价方向）落在归一域边界 `0/1`，即 sigmoid 的饱和区，
  梯度趋近于零、预测分布呈双峰分裂；
- 常见方向 `0°`/`90°` 贴近边界，条件不佳。

`periodic_angle_distance` 周期损失解决了 **loss 度量层面**的 `0° ≈ 180°` 等价问题，
但无法改善 **decoder 内部编码分布**（sigmoid 边界饱和、双峰预测）。

本设计将 decoder 私有绝对角 reference 的编码改为 **shifted 归一**
`theta_shift = (theta_norm + 0.25) mod 1 ∈ [0,1)`，使：

- `0° → 0.25`，`90° → 0.75`，sigmoid 中心 `0.5 → 45°`；
- seam 移到 `135°`，远离常见方向；
- 公开契约 `theta_phys_rad ∈ [0, π)` 保持不变（decoder 输出边界转换后返回）。

## 2. 目标与非目标

### 2.1 目标

1. 仅改变 decoder 私有的**绝对角度 reference 编码**；公开物理角契约、criterion、
   matcher、postprocessor、geometry、eval、export 全部不动。
2. 通过一个配置旋钮 `decoder_angle_encoding` 在两种模式间切换：
   - `proportional`（默认，当前行为，数值逐位不变，checkpoint 兼容）；
   - `shifted`（rep0/1/3 的显式 θ reference 全部 shifted，rep2 不变）。
3. 新增转换 API 集中于 `obb_angle_contract.py`，转换点清单显式、可枚举、可测试。
4. 保持四种 `angle_rep` 的组件边界、配置键名与默认值语义不变。

### 2.2 非目标

1. 不改变 OBB 的 π 周期几何定义、公开 `[0, π)` 契约。
2. 不更换激活函数（`sigmoid`/`inverse_sigmoid` 机制保持不变，只改编码值域语义）。
3. 不把 `physical_rad_to_loss_rad` 接入周期 loss 或 `yolo_angle_loss`（既有结论：
   对周期损失数学等价，无额外收益；其价值仅在非周期 L1 消融实验，见 §9）。
4. 不修改 rep0 的 `(α,β,γ,δ,ε,η)` refinement、rep1/3 的 Δθ 残差、rep2 的
   `(xywh,ε,η)` 私有表示。
5. 不修复 rep2 `use_angle_first` 注意力路径的既有 double-scale 现象（见 §10，记录为
   既有问题，不在本设计范围）。
6. 不做新实验对比；anchor 角度初始化值仅记录为可调旋钮。

## 3. 术语表与域

| 变量 | 量纲 / 数值域 | 用途 |
| --- | --- | --- |
| `theta_phys_rad` | 弧度，`[0, π)` | 公开物理角：dataset、transforms、geometry、criterion、matcher、postprocessor、eval、export |
| `theta_norm` | 无量纲，`[0,1)`，`= θ_phys/π` | 等比归一，decoder 内部当前编码 |
| `theta_shift` | 无量纲，`[0,1)`，`= (θ_phys/π + 0.25) mod 1` | shifted 归一，本设计新增的 decoder 私有编码 |
| `theta_logit` | 无界 logit | `logit(theta_norm)` 或 `logit(theta_shift)`，`ref_points_unact` 与各 head 原始输出 |
| `theta_loss_rad` | 弧度，`[−π/4, 3π/4)` | loss 内部规范域（`physical_rad_to_loss_rad`），仅 criterion 内部；本设计不接入 |

## 4. 张量级角度数据流矩阵（现状核实结果）

以下为逐代码核实（`deim_decoder.py`、`dfine_decoder.py`、`denoising.py`、`obb_angle_contract.py`、
`dfine_utils.py`、`obb_geometry.py`）后的完整矩阵。`→ logit` 表示经 `logit()`/`inverse_sigmoid`
进入无界 logit 域；`→ θ_norm` 表示经 `sigmoid` 进入 `[0,1)` 归一域。

### 4.1 入口（生产 decoder reference 的 θ）

| 生产者 | rep0/1/3 θ 通道 | rep2 | 域 |
| --- | --- | --- | --- |
| Anchors `_generate_anchors:996` | 默认 `r=0.25`（45°）；`angle_step>0` 时 `arange(n)*step`，随后 `logit()` | 6D `(xy,wh,ε,η)`，无 θ | **logit** |
| Encoder top-k `_get_decoder_input:1091` | `enc_bbox_head(memory) + anchors` | 同上，6D | **logit** |
| Denoising `denoising.py:112,117` | `physical_rad_to_norm(θ)` 后 `inverse_sigmoid` | `oriented_box_to_external_rect` 拆分 | **logit** |
| Decoder 初始化 `forward:268-280` | `sigmoid(ref_points_unact)` 全 5D → `query_pos_head` | ε/η 流，无 θ | **θ_norm [0,1]** |

### 4.2 内部（decoder 归一域 [0,1] 全链路）

| 位置 | rep0/1 | rep2/3 |
| --- | --- | --- |
| 初始 reference | `ref_points_detach = sigmoid(ref_points_unact)`（5D 含 θ_norm） | rep2/3 拆分位置流 4D + 角度流；rep2 为 6D ε/η |
| Layer-0 初始框 | `pre_bboxes = sigmoid(pre_bbox_head + inverse_sigmoid(ref_points_detach))` — θ_norm 残差代数 | rep2/3 由 `pre_angle_head + inverse_sigmoid(ref_dec_angle_detach)[...,4:]` 得 `dec_angle_initial` |
| rep3 角度流 | — | `dec_angle_initial ∈ [0,1]` = θ_norm，concat 进 5D reference |
| rep2 角度流 | — | `dec_angle_initial ∈ [0,1]` = **ε/η**，经 `external_rect_to_oriented_box` 生成物理 OBB |

### 4.3 两个「norm → physical」转换点（均硬编码 `×π`）

1. **Geometry decode** `forward:450-461`：`theta_scale[...,4] *= π` 后进
   `distance2bbox_obb`，输出再 `/π`。rep2 因 `angle_rep == 2` 分支不乘 π（其 ref 已是
   物理 OBB）。
2. **Deformable attention** `dfine_decoder.py:177`：
   `angle = reference_points[..., 4:5] * torch.pi` —— **关键约束**。注意力要求通道 4
   是等比 θ_norm [0,1] 并内部转为物理弧度构造旋转矩阵；它无法区分等比与 shifted 编码，
   改动编码必须同步修改此行。

### 4.4 公开边界

- `deim_decoder.py:1240-1264`（wrapper）：`out_bboxes`/`out_refs`/`pre_bboxes` 的
  `[...,4:]` 经 `norm_to_physical_rad` 转物理 `[0, π)`；rep2 的 `out_refs` 为 6D 直通、
  `pre_bboxes` 已是物理角直通。
- `deim_criterion.py:378/385` 非周期 L1 消融：对**公开物理输出**再 `physical_rad_to_norm`
  —— 与 decoder 内部编码完全隔离，不受影响。

## 5. 转换 API（新增，`obb_angle_contract.py`）

```python
def physical_rad_to_shifted_norm(theta_phys_rad: Tensor) -> Tensor:
    """物理弧度 → decoder 私有 shifted 归一 [0,1)。θ_phys=0→0.25, π/2→0.75, 3π/4→0。"""
    return torch.remainder(theta_phys_rad / torch.pi + 0.25, 1.0)

def shifted_norm_to_physical_rad(theta_shift: Tensor) -> Tensor:
    """decoder 私有 shifted 归一 [0,1) → 物理弧度 [0, π)。上述的逆。"""
    return torch.remainder(theta_shift - 0.25, 1.0) * torch.pi
```

与既有 API 的关系：

- `physical_rad_to_norm` / `norm_to_physical_rad`（等比）保持不变，`proportional`
  模式继续使用；
- `physical_rad_to_loss_rad`（`[−π/4, 3π/4)`）**不复用**：seam 位置相同但用途不同
  （loss 域 vs decoder 编码域），各自独立实现，避免语义耦合；
- 代数关系：`theta_shift = physical_rad_to_loss_rad(θ)/π + 0.25`（验证用，不构成 API）。

## 6. 配置面：两种编码模式

```yaml
decoder_angle_encoding: proportional    # 默认；当前行为，checkpoint 逐位兼容
decoder_angle_encoding: shifted         # rep0/1/3 shifted，rep2 不变
```

rep2 硬性规则（所有模式）：**始终 `proportional`** —— 其原生 reference
`(cx,cy,w,h,ε,η)` 不含显式 θ；ε/η 流、层间 θ_norm reference、注意力 6D 路径全部保持现状。

配置传递路径（与既有 `angle_rep`/`use_angle_first` 一致）：

- `TransformerDecoder.__init__` 新增构造参数 `decoder_angle_encoding: str = "proportional"`；
- 构建 `MSDeformableAttention`（`cross_attn`）时把解析后的编码传给其
  `angle_encoding` 参数（见 §7 站点 5）；
- 模型 config / yaml 顶层键 `decoder_angle_encoding`，经现有配置→构造参数管道传入
  `TransformerDecoder`。

decoder 内解析辅助：

```python
_VALID_DECODER_ANGLE_ENCODINGS = ("proportional", "shifted")

# TransformerDecoder.__init__
if decoder_angle_encoding not in _VALID_DECODER_ANGLE_ENCODINGS:
    raise ValueError(
        "decoder_angle_encoding must be 'proportional' or 'shifted', "
        f"got {decoder_angle_encoding!r}"
    )
self.decoder_angle_encoding = decoder_angle_encoding

def _resolved_angle_encoding(self) -> str:
    if self.angle_rep == 2:
        return "proportional"
    return self.decoder_angle_encoding
```

`shifted` 的作用集合固定为 rep0/1/3；rep2 因私有 reference 不含显式 θ，始终解析为
`proportional`。非法配置值必须在构造阶段显式拒绝，不允许静默回退。

## 7. 转换点完整清单

| # | 位置 | 现状 | shifted 分支 |
| --- | --- | --- | --- |
| 1 | `_generate_anchors:996` 默认 `r` | `0.25`（45°） | `0.5`（45°，sigmoid 中心） |
| 1b | `_generate_anchors` `angle_step` 候选 | `arange(n)*step` | `remainder(r + 0.25, 1)` |
| 2 | `denoising.py:112` | `physical_rad_to_norm(θ)` | `physical_rad_to_shifted_norm(θ)` |
| 3 | `forward:450-452` geometry decode 前 | `theta_scale[...,4] *= π` | `shifted_norm_to_physical_rad(ref[...,4:5])` |
| 4 | `forward:461` geometry decode 后 | `inter_ref_bbox[...,4] / π` | `physical_rad_to_shifted_norm(...)` |
| 5 | `dfine_decoder.py:177` 注意力旋转 | `ref[...,4:5] * π` | `shifted_norm_to_physical_rad(...)`（需构造参数） |
| 6 | `deim_decoder.py:1104` encoder 辅助输出 | `norm_to_physical_rad` | shifted 逆变换 |
| 7 | `deim_decoder.py:1241-1264` 公开输出 | `norm_to_physical_rad` | shifted 逆变换 |

实现要点：

- **全部站点（1/1b/2/3/4/5/6/7）均由 `_resolved_angle_encoding() == "shifted"` 守卫**：
  站点 1/1b/2 是 shifted 值的**生产者**（anchor、denoising 入口），站点 3/4/5/6/7
  是物理域的**还原者**；任一站点缺守卫都会破坏 `proportional` 逐位一致。`proportional`
  配置与 rep2 在所有站点走与现状完全相同的代码路径（逐位一致，checkpoint 安全）。
- **注意力构造参数**：`MSDeformableAttention` 增加 `angle_encoding: str = "proportional"`
  构造参数（默认值使 `dfine_decoder.py` / `rtdetrv2_decoder.py` / `rtv4` 现有实例全部
  不受影响），由 decoder layer 构建 `cross_attn` 时传入解析结果。
- 站点 6 仅影响训练期 encoder 辅助输出（`enc_topk_bboxes_list`）；推理无此路径。

## 8. 不需要改动的部分（已逐项核实）

| 组件 | 原因 |
| --- | --- |
| `query_pos_head` / `query_angle_head` | 可学习 MLP，直接消费编码值；无代码改动。其语义随编码改变而改变——这正是实验目的；也是 shifted 运行 checkpoint 不兼容的来源 |
| `sigmoid(inverse_sigmoid(ref) + head)` 残差代数 | 编码无关：θ 仍在 [0,1]，仅物理含义不同 |
| rep0 6D refinement `(α,β,γ,δ,ε,η)` | 不直接回归角度，几何间接路径，保持物理 |
| rep1/3 Δθ 残差 | 保持有符号物理弧度；`periodic_angle_distance` 不变 |
| rep2 全部路径 | 强制 `proportional` |
| `distance2bbox_obb` / `bbox2distance_obb` / `obb_geometry` 全部几何函数 | 永远消费物理 `[0, π)`（几何边界处已转换） |
| criterion（周期距离；非周期 L1 消融基于公开物理输出） | 与 decoder 内部编码隔离 |
| matcher / postprocessor / eval / export / `physical_rad_to_loss_rad` | 公开契约不变 |

## 9. 实验顺序（二阶段）

1. **编码实验**：`shifted` vs `proportional` 基线，同 seed。
   这是本设计的唯一直接目标：改善 decoder 内部绝对角 reference 的 sigmoid 参数化。
2. **独立轴**（编码实验定稿后再议）：`physical_rad_to_loss_rad` 接入显式非周期 L1
   消融，比较 seam 位置——与本设计正交。

## 10. 已知既有问题（记录，不修复）

rep2 且 `use_angle_first=True` 的注意力路径：`dfine_decoder.py:172-177` 将 6D
reference 经 `external_rect_to_oriented_box` 转为 5D **物理** OBB 后，仍执行
`* torch.pi` —— 对物理角二次放大。这是当前行为（A1/A2/A3 验证的 forward 矩阵未覆盖
注意力内部旋转），与 shifted 设计无关（rep2 被排除），仅记录待后续单独审计。

## 11. 兼容性与回滚

- 默认 `proportional` 与现状逐位一致；所有既有 checkpoint 原样加载。
- `shifted` 仅用于新训练；不支持用 shifted 配置加载 proportional
  checkpoint（anchor logit 与 query embedding 语义不同）。回滚 = 配置改回 + 重新加载
  checkpoint。
- rep2 在所有配置下可证明不受影响（强制 `proportional`）。

## 12. 测试与验证

1. **纯函数测试**（`test_obb_angle_contract.py` 扩展）：
   - 随机角 roundtrip：`phys → shifted → phys` 恒等；
   - 边界映射：`0→0.25`、`π/2→0.75`、`3π/4→0`；
   - 与 `physical_rad_to_loss_rad` 的代数等价（§5）。
2. **前向矩阵测试**（`test_deimv2_obb_smoke.py` 扩展）：4 表示 × 2 配置，断言：
   - 公开输出 θ ∈ `[0, π)`；
   - shifted 模式 ref θ ∈ `[0,1)` 且无 NaN（`remainder` 数值稳定）；
   - rep2 在两种配置下输出**逐位一致**（golden 值）。
3. **注意力旋转 sanity**：`MSDeformableAttention` 5D 分支单测，90° reference 在
   shifted 模式下采样沿旋转轴正确。
4. **proportional golden 回归**：既有 406 测试套件必须全绿。

## 13. 待实现文件清单（供后续 writing-plans 引用）

- `engine/deim/obb_angle_contract.py`：新增 `physical_rad_to_shifted_norm` /
  `shifted_norm_to_physical_rad`。
- `engine/deim/deim_decoder.py`：`decoder_angle_encoding` 配置、`_resolved_angle_encoding`、
  站点 1/1b/3/4/6/7。
- `engine/deim/dfine_decoder.py`：`MSDeformableAttention.angle_encoding` 构造参数、
  站点 5。
- `engine/deim/denoising.py`：站点 2（编码模式需传入）。
- 配置入口（模型 config / yaml）：`decoder_angle_encoding` 键，默认 `proportional`。
- 测试：`test/test_obb_angle_contract.py`、`test/test_deimv2_obb_smoke.py`、
  注意力单测。
