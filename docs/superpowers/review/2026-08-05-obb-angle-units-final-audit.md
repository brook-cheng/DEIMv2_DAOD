# DEIMv2 OBB 角度量纲最终审计与修改建议

日期：2026-08-05  
状态：审计完成，结论不通过；待功能代码修复  
范围：dataset、transforms、geometry、decoder、denoising、criterion、matcher、postprocessor、eval/export、配置与测试

关联文档：

- 设计契约：`docs/superpowers/specs/2026-08-05-obb-angle-contract-simplification-design.md`
- 前次审计：`docs/superpowers/review/2026-08-04-obb-angle-units-audit.md`
- 代码综述：`docs/superpowers/review/OBB_CODE_REVIEW.md`

## 1. 执行摘要

本次审计确认，GT、主预测框 `pred_boxes`、denoising、criterion、matcher、postprocessor
等主边界已经基本完成以下契约迁移：

```text
theta_phys_rad in [0, pi)       公开物理域，单位为弧度
theta_norm in [0, 1)            decoder 私有无量纲域，theta_norm = theta_phys_rad / pi
theta_logit in (-inf, inf)      theta_norm 的 logit
theta_loss_rad in [-pi/4, 3pi/4) loss 内部规范域
```

但是，当前实现仍有两项阻断性量纲错误：

1. `angle_rep == 2`：`ref_points_initial` 已经是物理弧度，却在 decoder 内部和输出边界被再次乘以
   `pi`。实测 `outputs["ref_points"]` 的最大角度为 `6.302505 > pi`，实际可落入
   `[0, pi^2)`，并污染 `loss_local` 的 FGL target。
2. `angle_rep == 1`：`bbox2distance_obb` 产生的角残差 target 已是
   `delta_theta_rad * reg_scale`，decoder 解码前却又执行 `distance[..., 4] *= pi`，使
   encode/decode 相差精确的 `pi` 倍。

这两项均是迁移前已存在的 decoder 内部问题，不是本次 `[0, pi)` 契约调整新引入的；但它们仍
违反当前设计契约，因此当前审计结论为：

> **FAIL：主公开链路基本对齐，但 rep1 与 rep2 训练路径仍存在确定性量纲错误。**

推荐采用第 7 节的**方案 A：最小分支修复**。第 8 节的方案 B 作为后续架构收敛方向，不应与
本次最小修复混在同一轮实施。

## 2. QA 与证据

### 2.1 现有测试套件

明确运行以下 12 个测试文件，避免宽泛 `pytest -k` 收集造成超时：

```text
test/test_obb_angle_contract.py
test/test_obb_domain_audit.py
test/test_obb_roundtrip.py
test/test_obb_transforms.py
test/test_obb_adr_geometry.py
test/test_obb_adr_loss.py
test/test_obb_eval.py
test/test_obb_loss_integration.py
test/test_deimv2_obb_smoke.py
test/test_deim_criterion_obb_loss.py
test/test_yolo_obb_loss.py
test/test_matcher_obb_angle.py
```

结果：

```text
323 passed, 4 warnings in 95.86s
```

4 个 warning 均为 `torch.cuda.amp.GradScaler` 弃用提示，与角度量纲无关。

现有测试全部通过但未捕获本报告问题，原因见第 9 节。

### 2.2 四种 angle_rep 前向探针

最小 `DEIMTransformer` 前向结果：

| `angle_rep` | `pred_boxes[..., 4]` | `ref_points[..., 4]` | 公开域结论 |
| --- | --- | --- | --- |
| 0 | `[0.785398, 0.785398]` | `[0.785398, 0.785398]` | 通过 |
| 1 | `[0.785398, 0.785398]` | `[0.785398, 0.785398]` | 仅范围通过；残差互逆失败 |
| 2 | `[0.019320, 2.966980]` | **`[0.494540, 6.302505]`** | **失败：`ref_points > pi`** |
| 3 | `[0.785398, 0.785398]` | `[0.785398, 0.785398]` | 通过 |

### 2.3 rep1 encode/decode 专项探针

测试输入：

```text
theta_ref = 0.3 rad
theta_target = 0.5 rad
reg_scale = 4
signed_delta = 0.2 rad
encoded target = signed_delta * reg_scale = 0.8 rad * scale
```

结果：

```text
正确解码（不额外乘 pi） = 0.500000 rad
当前 rep1 解码             = 0.928318 rad
实际更新 / 正确更新         = 3.141592
```

该探针证明 rep1 更新被精确放大 `pi` 倍，不是浮点误差或 seam 边界问题。

### 2.4 rep2 二次缩放专项探针

`external_rect_to_oriented_box` 产生的角度：

```text
theta_phys = 3.041924 rad, valid in [0, pi)
```

当前输出边界再次乘 `pi` 后：

```text
theta_output = 9.556486 rad, invalid for [0, pi)
```

### 2.5 criterion、transform 与配置探针

- 非 seam 样例 `0.75pi` 与 `0.70pi`：周期与非周期消融均为 `0.05`。
- seam 样例 `0` 与 `pi - 1e-4`：周期距离约 `3.187e-5`，非周期 L1 约
  `0.99996811`。这是非周期消融的预期特征，不是实现错误。
- OBBFlip 在 `0、pi/4、pi/2、3pi/4、pi-eps` 上均输出 `[0, pi)`。
- `deimv2_obb_decouple.yml` 的 `angle_rep: True` 被 YAML 解析为 `bool`，Python 中
  `True == 1` 且 `True != 2`，因此实际进入 rep1 路径。
- `sp_ft_rep2.yml` 的 `angle_rep: 2` 被解析为整数 `2`，进入 ADR rep2 路径。

## 3. 逐环节量纲表

| 环节 | 当前表示 | 量纲/范围 | 审计 |
| --- | --- | --- | --- |
| DOTA/YOLO 顶点标注 | 4 个顶点 | 坐标，无角度 | 正确 |
| `xyxyxyxy_to_xywhr` | 5D OBB | `theta_phys_rad in [0, pi)` | 正确 |
| `external_rect_to_oriented_box` | 5D OBB | `theta_phys_rad in [0, pi)` | 正确 |
| `xywhr_to_xyxyxyxy` | 5D OBB 输入 | θ 为物理弧度 | 正确 |
| OBBFlip | 物理 OBB | `(pi - theta) % pi` | 正确 |
| affine/geometry | 物理 OBB | 重拟合后 `[0, pi)` | 正确 |
| anchor θ | decoder reference | `theta_norm in [0, 1)`；默认 `0.25` | 正确，对应 `pi/4` |
| `ref_points_unact` θ | decoder reference | `logit(theta_norm)` | 正确 |
| `dfine_decoder` attention θ | reference 第 5 维 | `theta_norm * pi` | 正确 |
| rep0 internal ref | decoder 私有 | `theta_norm in [0, 1)` | 正确 |
| rep1 internal ref | decoder 私有 | `theta_norm in [0, 1)` | ref 范围正确，残差单位错误 |
| rep2 initial ref | ADR 转 OBB 后 | 已是 `theta_phys_rad in [0, pi)` | 后续被误当 norm |
| rep3 internal ref | decoder 私有 | `theta_norm in [0, 1)` | 正确 |
| `distance2bbox_obb` 5D 输入 | points | `theta_phys_rad` | 正确契约 |
| `distance2bbox_obb` 5D distance | angle residual | `delta_theta_rad * reg_scale` | rep1 调用方错误乘 pi |
| `distance2bbox_obb` 6D 输入 | points | 5D 物理 OBB | rep2 调用方错误乘 pi |
| `inter_ref_bbox` 输出 | decoder 私有 | 函数产物物理 θ，随后 `/pi` 存为 norm | 转换本身正确 |
| `pred_boxes` 输出 | 5D OBB | `theta_phys_rad in [0, pi)` | 四种表示范围均正确 |
| rep2 `ref_points` 输出 | 5D OBB | 实际可达 `[0, pi^2)` | **错误** |
| denoising GT | 5D OBB → logits | 物理 θ `/pi` → norm | 正确 |
| criterion 周期路径 | matched OBB | `periodic_angle_distance / pi` | 正确 |
| criterion 非周期消融 | matched OBB | `theta_phys / pi` 后普通 L1 | 用户批准偏差 |
| YOLO angle loss | matched OBB | π 周期 wrap | 正确 |
| matcher angle cost | pairwise OBB | 消费物理弧度、周期惩罚 | 正确 |
| postprocessor | 5D OBB | θ factor `1.0`，物理弧度直通 | 主 `pred_boxes` 正确 |
| eval/export | 5D OBB | 消费物理弧度 | 主 `pred_boxes` 正确 |

## 4. 问题一：rep2 的物理角被当作归一角

严重度：**Critical**  
影响配置：显式 `angle_rep: 2`，例如 `configs/custom_obb/dlzdt/sp_ft_rep2.yml`  
影响路径：decoder ADR decode、`outputs["ref_points"]`、criterion `loss_local`/FGL target

### 4.1 根因

rep2 在 layer 0 使用：

```python
pre_bboxes = external_rect_to_oriented_box(
    ref_points_initial, dec_angle_initial
)
ref_points_initial = pre_bboxes.detach()
```

`external_rect_to_oriented_box` 已返回物理弧度 `[0, pi)`。但通用 decoder 路径仍执行：

```python
theta_scale[..., 4] *= torch.pi
ref_points_initial_scaled = ref_points_initial * theta_scale
```

该逻辑对 rep0/1/3 正确，因为这些路径的 `ref_points_initial[..., 4]` 是 `theta_norm`；对 rep2
错误，因为它已经是 `theta_phys_rad`。

此外，`dec_out_refs.append(ref_points_initial)` 保存的是 rep2 物理角，但输出边界又执行：

```python
norm_to_physical_rad(out_refs[..., 4:])
```

因此 `ref_points` 变成 `[0, pi^2)`。

### 4.2 影响

`outputs["ref_points"]` 被 `DEIMCriterion.loss_local()` 传给 `bbox2distance_obb()`。rep2 配置使用
`obbox_rep_dim: 6`，函数会调用 `oriented_box_to_external_rect(points)`，将超界 θ 直接送入
`cos/sin`。结果虽然通常保持 finite，但外接矩形、顶点偏移和 FGL target 的几何意义错误。

### 4.3 方案 A：最小分支修复（推荐）

保持当前 rep2 内部的混合域现状，但在两个边界显式分支：

1. `deim_decoder.py` 逐层 decode 前：

   ```text
   if angle_rep == 2:
       ref_points_initial_scaled = ref_points_initial
       # rep2 已是物理 OBB，不再乘 pi
   else:
       ref_points_initial_scaled = norm_to_physical_rad(ref_points_initial 的角度通道)
   ```

   空间四维保持原值，只对角度通道决定是否缩放。不要对完整 tensor 直接使用统一
   `theta_scale` 而忽略表示类型。

2. decoder 最终输出处：

   - `out_bboxes`：继续进行 norm → physical，因为 `inter_ref_bbox` 在 append 前统一做过 `/pi`。
   - `out_refs`：rep2 物理角直通；rep0/1/3 继续 `norm_to_physical_rad`。
   - `pre_bboxes`：rep2 物理角直通；rep0/1/3 继续 `norm_to_physical_rad`。

推荐使用局部辅助函数或清晰的条件分支，避免在三个输出变量上复制不一致的逻辑。例如概念上可
区分：

```text
decode_internal_norm_box(x)    # norm theta -> physical theta
passthrough_internal_phys_box(x) # physical theta unchanged
```

不要把 ADR 的 `(epsilon, eta)` 通道误当作 θ 通道处理；只有经过
`external_rect_to_oriented_box` 后得到的 5D OBB 才有 θ。

### 4.4 方案 A 验收条件

- `angle_rep=2`：`pred_boxes[...,4]` 与 `ref_points[...,4]` 均满足 `[0, pi)`。
- rep2 零残差 decode：输出 OBB 与输入 reference OBB 几何等价。
- rep2 `bbox2distance_obb` → `distance2bbox_obb` round-trip 通过。
- rep0/1/3 的既有输出域和 anchor 行为不变。
- rep2 的 `loss_local` 产出 finite，且 target 维数与 `pred_corners` 一致。

## 5. 问题二：rep1 角残差额外乘 pi

严重度：**Critical（主要 decouple 配置命中）**  
影响配置：`angle_rep: 1`，以及 YAML 中的 `angle_rep: True`  
影响路径：5D angle residual decode、matcher、box/KLD loss、FGL 互逆

### 5.1 根因

编码端 `bbox2distance_obb`：

```python
signed_delta = periodic_angle_distance(pred_theta, gt_theta, True)
angle_lens = signed_delta * reg_scale
```

因此 target 单位是：

```text
rad * reg_scale
```

解码端 `distance2bbox_obb`：

```python
theta_new = (theta_ref + distance_theta / reg_scale) % pi
```

这两个公式本来互逆。但 decoder 对 rep1 增加：

```python
if self.angle_rep == 1:
    distance[..., 4] *= torch.pi
```

最终变为：

```text
theta_new = theta_ref + delta_theta_rad * pi
```

与 target 相差 `pi` 倍。

### 5.2 方案 A：最小修复（推荐）

删除 rep1 的额外 `distance[..., 4] *= torch.pi`。不要改
`bbox2distance_obb` 或 `distance2bbox_obb` 的公式；当前两者在没有额外 `pi` 时已经互逆。

必须确认：

- rep1 的 `Integral` 输出与 FGL target 采用相同的 `rad * reg_scale` 单位。
- rep3 已经不执行该 `*pi`，可作为同类正确路径参考。
- rep0/rep2 使用 6D residual，不应受此修改影响。

### 5.3 方案 A 验收条件

- 已知 `theta_ref=0.3`、`theta_gt=0.5`，encode→decode 输出 `0.5`，周期误差接近 0。
- 测试至少覆盖正残差、负残差、跨 `0/pi` seam 残差。
- rep1 非零残差前向不能只检查范围，必须检查数值等于预期物理角。
- `pred_boxes`、matcher、criterion 中的 θ 均保持物理弧度 `[0, pi)`。

## 6. 配置与兼容性问题

### 6.1 `angle_rep: True` 的隐式语义

`True == 1`，因此：

```yaml
angle_rep: True
```

实际等价于：

```yaml
angle_rep: 1
```

建议所有配置显式使用整数 `0/1/2/3`，不要再使用布尔值。这样可以避免配置阅读者误以为 `True`
表示“启用 decoupled ADR”。

### 6.2 `obbox_rep_dim` 必须与 angle_rep 对齐

当前代码语义：

| `angle_rep` | decoder residual 维度 | criterion `obbox_rep_dim` |
| --- | --- | --- |
| 0 | 6：`alpha,beta,gamma,delta,epsilon,eta` | 6 |
| 1 | 5：`alpha,beta,gamma,delta,delta_theta` | 5 |
| 2 | 6：`alpha,beta,gamma,delta,epsilon,eta` | 6 |
| 3 | 5：`alpha,beta,gamma,delta,delta_theta` | 5 |

`deimv2_obb_decouple.yml` 当前使用 `angle_rep: True`（即 rep1），但没有显式配置
`obbox_rep_dim`，criterion 默认值为 `6`。这会使 decoder 的 5D residual 与 criterion 的 6D target
语义不一致。

建议将该配置明确写为：

```yaml
DEIMTransformer:
  angle_rep: 1

DEIMCriterion:
  obbox_rep_dim: 5
```

并对所有配置增加静态一致性检查。模型构建时建议断言：

```text
angle_rep in (0, 2) -> obbox_rep_dim == 6
angle_rep in (1, 3) -> obbox_rep_dim == 5
```

该断言可放在 workspace/config 组装层，或在构建 criterion/model 后进行一次跨组件校验；不要让 decoder
和 criterion 各自静默采用不一致默认值。

表 6.2 的维度映射已由 `configs/custom_obb/` 下全部 40+ 个配置文件的显式 `obbox_rep_dim` 逐项核实：
rep0=6、rep1=5、rep2=6、rep3=5，全部一致，无反例。

### 6.4 误导性注释与文档字符串（低严重度）

1. `engine/deim/deim_decoder.py:438` 的注释写作：

   ```python
   # 1:(α,β,γ,δ)(ε,η)->(α,β,γ,δ,ε,η) 2:(α,β,γ,δ)(deta_theta)->(α,β,γ,δ,deta_theta)
   ```

   该注释位于 `angle_rep in (2, 3)` 分支内，却使用 `1:` / `2:` 编号。它们与
   `angle_rep` 的编号（1 = 5D delta_theta，2 = 6D ε,η）直接冲突，极易误导读者以为
   rep1 是 6D、rep2 是 5D。建议改为显式 `rep2:(ε,η)->6D / rep3:(delta_theta)->5D`
   或 `6D:` / `5D:` 前缀，消除与 angle_rep 编号的歧义。

2. `engine/deim/dfine_utils.py:202` 的 docstring 声称 `deta_theta in [0,π]`。实际 5D
   encode 公式为 `signed_delta * reg_scale`，其中 `periodic_angle_distance(..., True)`
   返回有符号最短角差，量纲为 `rad * reg_scale`，范围约 `(-π/2 * reg_scale, π/2 * reg_scale)`，
   不是 `[0, π]`。建议把 docstring 修正为 `rad * reg_scale` 并去掉 `[0,π]` 域声明，
   避免与 6D 分支的 `(ε,η)` 域混淆。

### 6.3 checkpoint 兼容性

旧 checkpoint 的 head、anchor 和 reference logit 是按旧 shifted seam 语义训练的。本次迁移改变了：

- decoder 对外角度解释；
- GT 主值区间；
- anchor norm 值（`0.5 -> 0.25`，物理方向保持 `pi/4`）；
- denoising 角度 logit。

因此不建议直接用旧 shifted checkpoint 续训新等比契约。简单地对 checkpoint 中某个 tensor 做平移不能
保证所有隐状态、head 偏置和 optimizer momentum 一致。

建议：

1. 新契约实验使用新 checkpoint 从头训练或从不含 OBB angle head 的公共 backbone 初始化。
2. 配置或 README 明确标注旧 checkpoint 不兼容。
3. 若未来必须迁移 checkpoint，应作为独立设计与实验任务处理，不在本次最小修复中添加未经验证的
   兼容 shim。

## 7. 方案 A：本轮最小修改清单（推荐）

按以下顺序实施，每项完成后由测试锁定再进入下一项：

### A1. 修复 rep1 角残差单位

- 文件：`engine/deim/deim_decoder.py`
- 删除 rep1 的 `distance[..., 4] *= torch.pi`。
- 不修改 `bbox2distance_obb` 和 `distance2bbox_obb` 的公式。
- 新增 rep1 非零残差 encode/decode round-trip 测试。

### A2. 修复 rep2 decoder decode 输入域

- 文件：`engine/deim/deim_decoder.py`
- rep2 的 `ref_points_initial` 已是物理域，不参与通用 norm→physical `*pi`。
- rep0/1/3 保持当前 norm→physical 转换。
- 新增 rep2 零残差和非零 ADR residual decode 测试。

### A3. 修复 rep2 `out_refs` / `pre_bboxes` 输出域

- 文件：`engine/deim/deim_decoder.py`
- `out_bboxes` 仍统一 norm→physical。
- rep2 的 `out_refs` 与 `pre_bboxes` 物理角直通。
- rep0/1/3 的 `out_refs` 与 `pre_bboxes` 继续 norm→physical。
- 新增四种 angle_rep 的 `pred_boxes/ref_points/pre_bboxes` 域矩阵测试。

### A4. 显式修复配置映射

- 将 `angle_rep: True` 改为 `angle_rep: 1`。
- 对 rep1/3 配置显式设置 `obbox_rep_dim: 5`。
- 对 rep0/2 配置显式设置 `obbox_rep_dim: 6`。
- 增加跨组件配置断言或配置审计测试。

### A5. 清理低风险问题

- `matcher.py`：`angle_factor` 仅定义和赋值、无消费方。确认无外部配置传参后删除；或者标记 deprecated
  并在后续版本删除。
- `physical_rad_to_loss_rad`、`canonicalize_phys_rad`、`physical_rad_to_logit`、
  `logit_to_physical_rad` 当前无生产调用者。建议保留为批准设计中的契约 API，但在 docstring 或审计文档
  中明确“当前无运行时消费者”；不要为了消除死代码而强行增加没有语义价值的转换。
- 如果目标是让 `canonicalize_phys_rad` 成为唯一事实来源，可在后续独立清理中将 geometry/transforms 的
  等价 `remainder(..., pi)` 改为调用该函数；本轮不与 Critical 修复混合。

## 8. 方案 B：长期统一 decoder 内部域

长期建议让 decoder 内部所有 5D reference OBB 的 θ 永远保持 `theta_norm in [0,1)`：

```text
external_rect_to_oriented_box -> physical theta
                              -> immediately physical_rad_to_norm
                              -> decoder internal state
                              -> convert to physical only at geometry/decode boundary
```

该方案的目标是不再让 `ref_points_initial` 随 `angle_rep` 具有不同域：

- rep0/1/3 当前为 norm；
- rep2 当前为 physical；
- 长期统一后全部为 norm。

优点：

- `theta_scale[...,4] *= pi` 可以恢复为统一、无分支的几何边界转换；
- `dec_out_refs`、`pre_bboxes`、`inter_ref_bbox` 的内部契约一致；
- 减少未来新增 `angle_rep` 时的域分支错误。

代价与风险：

- 必须审计 `query_pos_head`、`query_angle_head`、`decouple_angle_layers`、`ref_dec_angle_detach`、
  `inverse_sigmoid` 的所有输入域；
- 必须同时覆盖 `use_angle_first` 与普通 decoupled 路径；
- 对 checkpoint 兼容和训练行为影响比方案 A 大；
- 不适合与当前最小修复在同一轮完成。

因此方案 B 应单独形成设计、实施计划和实验验证，不作为本轮阻断修复的前置条件。

## 9. 必须新增的测试

用户修改功能代码后，测试至少应包括：

1. **四表示公开域矩阵**
   - `angle_rep in [0,1,2,3]`；
   - `pred_boxes`、`ref_points`、`pre_bboxes` θ 均 `[0,pi)`；
   - rep2 必须使用真实 forward，而不是仅测试 helper。
2. **rep1 encode/decode inverse**
   - 正残差、负残差；
   - seam 两侧；
   - 断言周期距离小于容差，而不只是检查范围。
3. **rep2 ADR round-trip**
   - reference OBB → external rect/offset → residual encode/decode → OBB；
   - 零 residual 必须保持几何不变；
   - 非零 residual 与 target 几何等价。
4. **decoder 内外域检查**
   - geometry boundary 前 θ 是物理域；
   - decoder state θ 是 norm 域或按方案 A 的明确分支域；
   - `out_bboxes` 与 `out_refs` 不得出现同字段不同单位。
5. **配置一致性测试**
   - 禁止 `angle_rep` 使用 bool；
   - rep0/2 必须 `obbox_rep_dim=6`；
   - rep1/3 必须 `obbox_rep_dim=5`。
6. **回归测试**
   - 原 12 文件测试集必须继续 `323+` 全绿；
   - decoder→matcher→loss_boxes→backward smoke 继续通过；
   - rep1/rep2 的 `loss_local` 必须 finite，并确认 target shape 对齐。

## 10. 修复时不可误改的正确路径

以下位置当前量纲正确，不应随 rep1/rep2 修复被整体删除或改写：

- `obb_geometry.periodic_angle_distance`；
- GT 的 `torch.remainder(theta, torch.pi)`；
- anchor 默认 `r=0.25`；
- decoder `inter_ref_bbox[...,4:] / torch.pi`：`distance2bbox_obb` 返回物理角，该除法负责恢复内部 norm；
- `dfine_decoder.py` 的 `reference_points[...,4:5] * torch.pi`：其输入契约是 norm；
- denoising 的 `physical_rad_to_norm`；
- criterion 周期路径的 `periodic_angle_distance / torch.pi`；
- postprocessor 的 θ factor `1.0`；
- 非周期 L1 消融路径是用户明确保留项，不应在本轮偷偷改成周期距离。

## 11. 建议实施顺序与停止条件

推荐顺序：

```text
rep1 residual unit fix
  -> rep1 inverse test
  -> rep2 decoder input-domain fix
  -> rep2 output-domain fix
  -> rep2 forward/FGL tests
  -> config mapping fix
  -> full regression
  -> optional dead-parameter cleanup
```

若任一步出现以下情况，应停止继续修改并重新审计：

- 修复 rep2 后 rep0/1/3 的输出角发生变化；
- `pred_boxes` 与 `ref_points` 的角度单位不再一致；
- `bbox2distance_obb` 与 `distance2bbox_obb` 无法 round-trip；
- 为通过测试需要修改 `periodic_angle_distance`；
- 需要加入旧 shifted seam compatibility shim 才能运行新配置。

这些情况意味着修改已经超出方案 A 的最小边界，应转入方案 B 的独立设计流程。
