# DEIMv2 OBB 角度契约简化设计（修订版）

日期：2026-08-05（修订：2026-08-05，基于手工回退后的真实代码基线）

状态：设计已批准，决策完备，可直接生成实现计划

关联文档：

- 被拒绝的设计：`docs/superpowers/specs/2026-08-04-obb-angle-contract-unification-design.md`（9 任务迁移，中央化 shifted seam）
- 保留的设计：`docs/superpowers/specs/2026-08-04-obb-adr-loss-design.md`（ADR 表示与损失，不涉及角度 seam 语义）
- 审计记录：`docs/superpowers/review/2026-08-04-obb-angle-units-audit.md`

## 1. 当前基线（手工回退后）与问题诊断

2026-08-04 的 9 任务角度迁移已被用户手工回退。回退后工作区接近 HEAD：仅
`docs/superpowers/review/OBB_CODE_REVIEW.md` 与 `engine/deim/__init__.py` 有已跟踪修改，
未跟踪项为 `.superpowers/`、四份文档与 `engine/deim/obb_angle_contract.py`；
`engine/deim/obb_codecs.py` 已不存在。**无需任何回滚步骤。**

对当前代码逐行审计（见 §8 逐组件），结论：**当前代码是一个自洽的 shifted-seam 管线**，
seam 同时存在于 GT 规范化与 decoder 输出边界；而 decoder 逐层精修已是严格等比。

| 位置 | 当前实现 | 域 | 结论 |
| --- | --- | --- | --- |
| `obb_geometry.py:100`、`:238` GT 规范化 | `remainder(θ + π/4, π) − π/4` | **GT ∈ [−π/4, 3π/4)**（docstring 却写 `[0,π]`，文档与代码不符） | 需改等比 `remainder(θ, π)` |
| `deim_decoder.py:441-460` 逐层精修 | `θ_scale[...,4] *= π`，输出 `/π` | 严格等比 `[0,1] ↔ [0,π]` | **已正确，不改** |
| `deim_decoder.py:1100` `_get_decoder_input` 训练期 encoder 辅助输出 | `(x − 0.25)·π` | **aux pred ∈ [−π/4, 3π/4)** | 需改 `x·π` |
| `deim_decoder.py:1233-1241` 最终输出 | `(x − 0.25)·π` | **pred ∈ [−π/4, 3π/4)** | 需改 `x·π` |
| `deim_decoder.py:996` anchors 默认 | `r = 0.5` | 等比语义下对应物理 `π/2` | 需改 `r = 0.25`（保持迁移前物理 `π/4` 初始化方向） |
| `dfine_decoder.py:177` 旋转采样点 | `angle = ref[...,4:5] * π` | 等比 | **已正确，不改** |
| `dfine_utils.py` `distance2bbox_obb`/`bbox2distance_obb` | 物理域 `[0,π]` IO，`% π` | 物理 | **已正确，不改** |
| `denoising.py:110-111` | `(θ + π/4)/π` | shifted | 需改 `θ/π` |
| `deim_criterion.py:363-374` 周期路径 | `periodic_angle_distance(...) / π` | 域无关，数学正确 | **已正确，不改** |
| `deim_criterion.py:378-383` 非周期路径 | `(θ + π/4)/π`（无 wrap） | shifted，`θ > 3π/4` 时值超 1 | 需改为 `periodic_angle_distance` 度量 |
| `matcher.py` / `yolo_obb_loss.py` | 周期惩罚，域无关 | 物理 | **已正确，不改** |
| `postprocessor.py:62` | θ 通道 factor 1.0 直通 | 物理 | 只改注释 |
| `obb_angle_contract.py` | 未跟踪死代码，shifted 公式，无任何引用 | — | 重写为等比 + loss 规范辅助函数并接入（见 §8.1） |
| `__init__.py:25` | 残留 `obb_codecs` 注释导出 | — | 删除注释 |

被拒迁移的核心问题依旧成立：

1. **seam 进入网络边界**。`theta_norm = 0.0`（对应物理 `3π/4 ≡ −π/4`）处的平移 seam 使
   归一域内部出现不连续；任何未按 seam 语义处理的消费者（如 `dfine_decoder` 的
   `angle * π`）都会得到错误物理角。当前代码靠「GT 规范化与 decoder 输出同时平移」保持
   自洽，但 seam 语义仍渗入公开边界，与「公开物理域」契约冲突。
2. **文档与代码不符**。`obb_geometry.py` 的 GT 规范化实际输出 `[−π/4, 3π/4)`，
   而 docstring 声明 `[0, π]`；`postprocessor.py` 注释声明 `[0,π]` 而实际收到 shifted 角。
3. **criterion 非周期路径无 wrap**。`(θ + π/4)/π` 在 `θ > 3π/4` 时值超过 1，seam 两侧错配。
4. **契约模块是死代码**。`obb_angle_contract.py` 保留被拒的 shifted 公式且无人引用，
   是「唯一事实来源」的反面：事实散落在 9 个内联位置（GT 规范化 ×2、decoder 输出 ×4、
   denoising ×1、criterion 非周期 ×2），模块只是摆设。

结论：放弃 shifted seam 中央化，改为「公开物理域 / decoder 私有等比归一域 / loss 内部规范域」
的简单分层契约；**只改边界，不动 decoder 内部已正确的等比精修**。

## 2. 目标与非目标

### 2.1 目标

1. 建立简单的分层角度契约：公开物理域、decoder 私有归一域、loss 内部规范域，各域职责单一、转换公式显式。
2. decoder 私有归一角 `theta_norm` 与物理角**严格等比**：`theta_norm = theta_phys_rad / pi`。
   周期 seam 位于归一区间的自然边界 `0/1`，decoder 内不再引入位于区间内部的 shifted seam。
   loss 内部规范域的 seam 永远配合最短周期残差使用，不会产生虚假大梯度。
3. 公开边界一律物理域：GT 规范化、decoder 输出、criterion 输入、matcher、postprocessor、
   geometry、eval 全部 `theta_phys_rad ∈ [0, π)`。
4. `periodic_angle_distance`（`obb_geometry`）成为最短周期残差的单一事实来源；criterion
   周期/非周期路径统一委托。
5. 保持四种 `angle_rep`、组件边界与训练配置语义（`periodic_angle_flag`、`use_yolo_angle`、
   `lambda_angle` 等键名与默认值不变）。

### 2.2 非目标

1. 不改变 OBB 的 pi 周期几何定义。
2. 不改变类别预测结构、HBB 路径、postprocessor 的空间尺度变换、eval 度量。
3. 不引入 tensor 子类或 dataclass 包装，不影响训练、导出、TorchScript 与 ONNX 链路。
4. 不做新实验。anchor 角度初始化值（见 8.2.4）仅记录为可调旋钮，不在本设计中对比验证。
5. 不重构模块边界，不合并或删除 `angle_rep` 路径；不重建已被删除的 `obb_codecs.py`。
6. 不修改 decoder 逐层精修中已正确的等比换算（`deim_decoder.py:441-460`、`dfine_decoder.py:177`）。

## 3. 术语表

| 标识符 | 含义 | 数值域 | 可见范围 |
| --- | --- | --- | --- |
| `theta_phys_rad` | 物理角，弧度 | `[0, pi)` | 公开：dataset、transforms、geometry、criterion、matcher、postprocessor、eval、export |
| `theta_norm` | 归一化角，无量纲，与物理角严格等比 | `[0, 1)` | decoder 私有（含 denoising 生产者、dfine_decoder 消费者） |
| `theta_logit` | `logit(theta_norm)`，无界 | `(-inf, inf)` | decoder 各 head 的原始输出与 `ref_points_unact` 角度通道 |
| `theta_loss_rad` | loss 内部规范角，弧度 | `[-pi/4, 3pi/4)` | 仅 criterion 内部 |
| `delta_theta_rad` | 有符号最短 pi 周期残差，弧度 | `[-pi/2, pi/2)` | criterion 内部 |

域纪律：`theta_norm` 与 `theta_logit` 不得泄漏出 decoder 边界（见 §9.1 import 审计）。

## 4. 分层契约架构

```text
                       theta_phys_rad in [0, pi)         公开物理域
      ┌──────────────────────────┬──────────────────────────────┐
      │ dataset / transforms     │  geometry                    │
      │ criterion / matcher      │  postprocessor / eval / export│
      └─────────────┬────────────┴───────────────┬──────────────┘
                    │ theta_phys_rad             │ theta_phys_rad
                    │ (criterion 输入)            │ (几何 / 输出)
                    ▼                             ▲
      ┌─────────────────────────┐                │
      │  theta_loss_rad         │                │
      │  [-pi/4, 3pi/4)         │   loss 内部规范域（仅 criterion）
      │  (shift seam 只在此处)   │
      └─────────────────────────┘
                    ▲
                    │ 周期残差 delta_theta_rad in [-pi/2, pi/2)
                    │ （直接对物理角求亦可，结果相同，见第 6 节）
                    │
      ┌─────────────┴───────────────┐
      │  decoder 边界（一次转换）     │
      │  theta_norm = phys / pi     │   decoder 私有归一域
      │  theta_phys_rad = norm * pi │
      └─────────────┬───────────────┘
                    │ theta_norm（logit 化后）
                    ▼
      decoder 内部：query_pos_head、anchor 角度通道、deformable
      注意力旋转采样点、pre_bbox_head / dec_angle_head 输出
```

分层要点：

1. shifted seam 唯一出现的位置是 `theta_loss_rad` 的定义，且只被周期残差消费。
2. `theta_norm` 严格等比，周期 seam 只位于 `[0, 1)` 的自然边界，不在区间内部产生额外跳变；
   `dfine_decoder.py` 中 `angle * math.pi` 的旋转采样点语义天然正确（当前已如此）。
3. 公开边界（GT 规范化、decoder 输出、geometry、postprocessor、matcher、criterion 输入、eval）
   一律是 `theta_phys_rad`。当前代码的 shifted 边界是缺陷，不是特性。

## 5. 核心公式（正 / 逆变换）

所有 mod 运算使用 `torch.remainder`，结果落在半开区间。

| 名称 | 公式 | 输出域 |
| --- | --- | --- |
| 物理角规范化 | `canonicalize_phys_rad(theta) = theta mod pi` | `[0, pi)` |
| 物理角到归一角（正） | `theta_norm = theta_phys_rad / pi` | `[0, 1)` |
| 归一角到物理角（逆） | `theta_phys_rad = theta_norm * pi` | `[0, pi)` |
| 物理角到 logit（正） | `theta_logit = logit(clamp(theta_phys_rad / pi, eps, 1 - eps))` | ℝ |
| logit 到物理角（逆） | `theta_phys_rad = sigmoid(theta_logit) * pi` | `(0, pi)` |
| 物理角到 loss 规范角 | `theta_loss_rad = ((theta_phys_rad + pi/4) mod pi) - pi/4` | `[-pi/4, 3pi/4)` |

> 注：最短周期残差 `delta_theta_rad ∈ [-pi/2, pi/2)` 与施加残差操作不在此模块——前者由
> `obb_geometry.periodic_angle_distance`（含 `with_signal` 有符号变体）提供，后者已在
> `dfine_utils.distance2bbox_obb` 内联。本模块只管**绝对角度**的域转换，`obb_geometry` 管
> **相对角度**的距离/残差，职责分离。

`eps` 默认 `1e-4`，与现有一致。

## 6. 「先转 theta_loss_rad 再求残差」的等价性说明

**结论：将 pred 与 GT 先转换到 `theta_loss_rad` 再做周期残差，语义上有用，数学上与直接对物理角求残差产生相同结果。**

推导：

```text
pred_loss = ((pred_phys + pi/4) mod pi) - pi/4   ≡ pred_phys + pi/4   (mod pi)
gt_loss   = ((gt_phys   + pi/4) mod pi) - pi/4   ≡ gt_phys   + pi/4   (mod pi)
=> pred_loss - gt_loss ≡ pred_phys - gt_phys     (mod pi)
```

`delta_theta_rad` 公式只依赖 `(pred - gt)` 的模 pi 剩余类，因此：

```text
delta_theta_rad(pred_loss, gt_loss) == delta_theta_rad(pred_phys, gt_phys)
```

逐元素相等（至浮点舍入）。语义价值在于：

1. 两个角先归一到同一主值区间，「0」的含义无歧义，日志与调试更直观。
2. 未来若要在 loss 域直接做 L1 或分布量化，`theta_loss_rad` 已是现成的规范表示。
3. 实现上把「归一到规范区间」与「求周期残差」解耦，各一步可独立测试。

实现注意：

- 两条路径在精确 seam 边界（如 `pred_loss - gt_loss = pi/2` 的整数倍）上浮点舍入可能相差
  一个 `pi` 或出现 `-0.0` 与 `0.0`，但模 pi 剩余类相同，对损失值无影响；测试用 `allclose`
  并避开精确边界点。
- **禁止**在转换到 `theta_loss_rad` 后直接做减法而不做周期残差（见 §9.4）。

## 7. 边界示例

### 7.1 域映射表

| `theta_phys_rad` | `theta_norm = theta / pi` | `theta_loss_rad = ((theta + pi/4) mod pi) - pi/4` |
| --- | --- | --- |
| `0` | `0.0` | `0` |
| `pi/4` | `0.25` | `pi/4` |
| `pi/2` | `0.5` | `pi/2` |
| `3pi/4` | `0.75` | `-pi/4`（seam 处，等于 `3pi/4` 模 pi 的剩余类） |
| `pi - eps`（趋近 pi） | `1 - eps/pi`（趋近 1） | 趋近 `0`（负侧），因为 `pi ≡ 0 (mod pi)` |

要点：常见方向在归一域内等距分布（0、0.25、0.5、0.75）；归一域只保留 `0/1` 自然周期边界，
不再把 seam 平移到区间内部。shifted seam 只出现在 loss 规范域，且永远配合周期残差使用。

### 7.2 残差示例（验证 loss 域与物理域等价）

| pred（物理） | gt（物理） | loss 域 pred | loss 域 gt | delta（两域相同） |
| --- | --- | --- | --- | --- |
| `0` | `3pi/4` | `0` | `-pi/4` | `pi/4` |
| `pi/2` | `0` | `pi/2` | `0` | `-pi/2` |
| `pi - 1e-3` | `0` | 约 `-1e-3` | `0` | 约 `-1e-3`（趋近 0） |
| `0.75pi` | `0.70pi` | `-pi/4` | `0.70pi` | `-0.05pi`（直接相减会得 `0.95pi`，错误） |

### 7.3 logit 边界讨论

等比契约下 `theta_phys_rad = 0` 对应 `theta_norm = 0`，位于 logit 下界。这是可接受的：

1. sigmoid 输出天然不可能精确到达 0 或 1，实际预测永远略大于 0，对应物理角略大于 0，
   与 `theta_phys_rad = 0` 的 GT 损失极小，无病态梯度。
2. `physical_rad_to_logit` 的 `eps` clamp 保证 logit 有限（与空间坐标 `cx、cy` 处理 0/1
   边界的方式相同）。
3. 平移方案避免此边界，代价是在网络内部引入 seam 不连续，最坏产生约 `pi/2` 的错误残差；
   等比方案的最坏偏差仅为 `eps` 量级。两者权衡下等比方案更优。

## 8. 逐组件契约（对应当前基线）

### 8.1 `engine/deim/obb_angle_contract.py`（重写 + 接入）

保留模块与文件名，重写为下列接口（其余函数全部删除）：

```text
canonicalize_phys_rad(theta)          # theta mod pi, [0, pi)
physical_rad_to_norm(theta)           # theta / pi, 严格等比，无任何平移
norm_to_physical_rad(theta_norm)      # theta_norm * pi
physical_rad_to_logit(theta, eps)     # logit(clamp(theta / pi, eps, 1 - eps))
logit_to_physical_rad(theta_logit)    # sigmoid(theta_logit) * pi
physical_rad_to_loss_rad(theta)       # ((theta + pi/4) mod pi) - pi/4, [-pi/4, 3pi/4)
```

**职责边界**：本模块只管**绝对角度**的域转换（phys↔norm↔logit↔loss_rad）。最短周期残差
`delta_theta_rad ∈ [-pi/2, pi/2)` 由 `obb_geometry.periodic_angle_distance`（`with_signal=True`）
提供；施加残差已在 `dfine_utils.distance2bbox_obb` 内联（`(θ + delta/reg_scale) % π`）。
不在此模块重复实现，避免双事实来源。

**删除**旧 shifted seam 映射的平移语义（旧映射使常见方向落在 0.25/0.5/0.75/0.0），以及模块
docstring 中关于「平移优化区间」的说明。模块头注释改为本设计第 3 节术语表与第 4 节分层图。

**接入点**（让模块成为真实事实来源，而非死代码）：

- `deim_decoder.py:1233-1241` 最终输出：`(out_bboxes[..., 4:] - 0.25) * pi` →
  `norm_to_physical_rad(out_bboxes[..., 4:])`（即 `x * pi`）。
- `denoising.py:110-111`：`(θ + pi/4) / pi` → `physical_rad_to_norm(θ)`。
- `deim_criterion.py` 非周期路径：改为经 `periodic_angle_distance`（见 8.4.2）。
- `obb_geometry.py::periodic_angle_distance`：保持现有实现（已是周期距离单一事实来源）。
- `dfine_decoder.py:177`：`angle = reference_points[..., 4:5] * torch.pi` 可改为
  `norm_to_physical_rad(...)`（可选，语义等价；不改为保持最小 diff 亦可，测试锁定语义）。

### 8.2 decoder（`engine/deim/deim_decoder.py`、`engine/deim/dfine_decoder.py`）

#### 8.2.1 入口契约

`ref_points_unact` 角度通道是 `theta_norm` 的 logit，decoder 内部直接消费，无 shifted 往返。
当前代码的 `norm_to_physical_rad`/`physical_rad_to_norm` 往返已被回退清除；若审计确认入口
无往返，则保持现状。**只验证，不改。**

#### 8.2.2 层间状态

`inter_ref_bbox`（物理域 5D）由 `distance2bbox_obb` 产出（`deim_decoder.py:441-460`，
`dfine_utils.py:194-250`），`theta_phys_rad in [0, pi)`；下一层经 `theta_scale[...,4] *= pi`
与输出 `/ pi` 保持等比。当前已正确，**只验证，不改**。

#### 8.2.3 输出（唯一需改的 decoder 位置）

`deim_decoder.py` 共有**四处** shifted 输出换算，全部改为等比：

```text
# 现：x_angle = (x - 0.25) * torch.pi   → [-pi/4, 3pi/4)
# 改：x_angle = x * torch.pi            → [0, pi)   （即 norm_to_physical_rad）
```

- `:1100` `_get_decoder_input` 训练期 encoder 辅助输出 `enc_topk_bboxes`（`angle_rep != 2` 分支）；
- `:1237` 最终输出 `out_bboxes`；
- `:1240` 最终输出 `out_refs`；
- `:1243` 最终输出 `pre_bboxes`。

删除各处注释中「量纲为 [-pi/4, 3pi/4)」的过时说明。`pre_bboxes` 若在后续分支被再次消费为
`theta_norm`（例如 decoder 内部循环的 `ref_points_initial`），须保持内部视图仍为等比
`[0,1]`，仅对外出口乘以 `pi`（decoder 内部循环在第 369 行 `pre_bboxes.detach()` 处消费
sigmoid 后的 `[0,1]` 视图，该视图不得乘以 `pi`；以测试锁定）。

#### 8.2.4 anchors（`_generate_anchors`）

- 现默认 `r = 0.5`（`deim_decoder.py:996`），在旧 shifted 公式 `(r - 0.25) * pi` 下对应物理
  `pi/4`；新等比公式下 `0.5` 对应 `pi/2`。为保持迁移前物理初始化方向 `pi/4`，默认值改为
  **`r = 0.25`**（等比 `r * pi = pi/4`）。
- 不把默认值保留为 `0.5`，因为在新契约中它对应 `pi/2`，会将契约迁移与模型初始化行为变化
  混在一起。是否改用 `pi/2` 必须作为独立实验另行评估，不属于本设计。
- `angle_step > 0` 的多角度锚：`angle_candidates = arange(n_angles) * angle_step in [0, 1)`，
  同为 `theta_norm` 语义，保持等比。
- `valid_mask` 过滤 `theta_norm in (0, 1)` 之外的候选后 logit 化，逻辑不变。

#### 8.2.5 `_get_decoder_input`

- 训练期 encoder 辅助输出与 denoising 拼接中的角度通道：若仍残留 shifted 换算则改为等比
  （`* pi` / `norm_to_physical_rad`），否则保持现状。以审计为准，测试锁定。
- `angle_rep in (0, 1)` 路径直接拼接，不变。

#### 8.2.6 `dfine_decoder.py`（MSDeformableAttention）

- `reference_points` 第 5 维为 `theta_norm`：`angle = reference_points[..., 4:5] * torch.pi`
  语义已正确（等比）。可选改为 `norm_to_physical_rad` 调用；不改为保持最小 diff 亦可。
- 其余不变。

### 8.3 denoising 流（`engine/deim/denoising.py`）

- GT 角度进入 denoiser 时是 `theta_phys_rad ∈ [0, pi)`（GT 规范化修复后，见 8.6）。
- `denoising.py:110-111`：`input_query_bbox[..., 4] = (θ + pi/4) / pi` →
  `input_query_bbox[..., 4] = physical_rad_to_norm(θ)`（即 `θ / pi`，纯线性），再 `inverse_sigmoid`。
- 删除迁移注释中描述的 shifted 语义。原「legacy `(theta + pi/4)/pi` 在 `theta >= 3pi/4` 时被
  clip 到 1.0」的问题随等比化消失：`theta / pi in [0, 1)` 天然无越界，`theta = 3pi/4` 精确对应 `0.75`。

### 8.4 criterion（`engine/deim/deim_criterion.py`）损失流

输入 `src_boxes`、`target_boxes` 均为物理域 5D OBB（decoder 输出与 GT 修复后）。

1. **周期 L1 路径**（`periodic_angle_flag = True`，默认）：
   ```text
   angle_term = lambda_angle * periodic_angle_distance(src, tgt) / pi
   ```
   `periodic_angle_distance(pred, tgt)`（`obb_geometry.py:18`，`with_signal=False`）返回
   无符号最短 π 周期距离 `∈ [0, pi/2]`。当前 `deim_criterion.py:363-374` 已如此，**不改**。

2. **非周期路径**（`periodic_angle_flag = False`）：删除 `deim_criterion.py:378-383` 的
   无 wrap `(theta + pi/4) / pi` L1（`theta > 3pi/4` 时值超过 1，seam 两侧错配）。改为：
   ```text
   angle_term = lambda_angle * periodic_angle_distance(src_phys, tgt_phys) / pi
   ```
   与周期路径的 angle_term 数值完全一致——`periodic_angle_flag` 对角度项变为无副作用（两路径
   统一），这正是本简化的意图。

3. **显式禁止**：转换到 `theta_loss_rad` 后直接相减而不做周期距离。示例：`pred = 0.75pi,
   gt = 0.70pi` 时 loss 域值为 `-pi/4` 与 `0.70pi`，直接相减得 `0.95pi`，正确周期距离为
   `0.05pi`（见 7.2 表）。criterion 中任何角度差必须经 `periodic_angle_distance`。

4. **`yolo_angle_loss`（`yolo_obb_loss.py`）**：现有 wrap `delta - round(delta / pi) * pi`
   与 `periodic_angle_distance` 数学等价（`sin(2 * wrap)^2` 在边界点恒为 0，差异不影响损失值）。
   保留现状即可（`periodic_angle_distance` 已是单一事实来源）；以测试锁定两者一致性。

5. **`compute_angle_cost_matrix`（matcher 用）**：同周期惩罚语义，保留现状。

6. **KLD 损失**：输入物理域 OBB，不变。

### 8.5 dfine_utils（`engine/deim/dfine_utils.py`）

`distance2bbox_obb` / `bbox2distance_obb` 输入输出均为物理域 `[0, pi)` OBB，角度残差经
`% pi` 规范化。当前正确，**不改**。无需重建已删除的 `obb_codecs.py`。

### 8.6 geometry 与 transforms

- `engine/deim/obb_geometry.py`：
  - `xyxyxyxy_to_xywhr`（`:100`）与第二个变体（`:238`）的 GT 规范化：
    `remainder(theta + pi/4, pi) - pi/4` → **`remainder(theta, pi)`**（即 `canonicalize_phys_rad`），
    使 GT 物理角落在 `[0, pi)`。同步修正两处 docstring（现声明 `[0,pi]` 却输出 `[-pi/4,3pi/4)`）。
  - `periodic_angle_distance`（`:18`）：保持现有实现（已是周期距离单一事实来源，无需委托）。
  - `external_rect_to_oriented_box` / `oriented_box_to_external_rect`、`xywhr_to_xyxyxyxy` 等
    输入输出声明物理域，当前正确，不改（docstring 与实现核对即可）。
- `engine/data/transforms/obb_transforms.py`：保持物理域。水平翻转的 `canonicalize_phys_rad(pi - theta)`
  不变（纯物理操作，与 seam 无关）。该文件是允许 import `canonicalize_phys_rad` 的唯一
  transforms 位置。

### 8.7 postprocessor（`engine/deim/postprocessor.py`）

保持现状：角度通道 factor 为 `1.0` 直通，`theta_phys_rad` 逐位不变，不做任何换算。
删除/改写 `:62` 附近注释中对归一域的过时描述（现注释声明 `[0,π]` 而实际收到 shifted 角；
修复 GT/输出后注释变为事实）。

### 8.8 matcher（`engine/deim/matcher.py`）

- 输入物理域 5D OBB；角度代价经 `compute_angle_cost_matrix`（周期惩罚），不接触 `theta_norm`。
- 不变（`compute_angle_cost_matrix` 保留现状，`periodic_angle_distance` 已是单一事实来源）。

### 8.9 eval 与 export

- eval：消费物理域 OBB 与 `theta_phys_rad` 度量，无角度换算，不变。
- `convert_to_deploy` / ONNX：不涉及角度契约换算，不变。

### 8.10 配置文件

- **不新增任何配置键**。`periodic_angle_flag`、`use_yolo_angle`、`lambda_angle`、`angle_lambda`、
  `reg_scale` 等沿用现有键名与默认值。
- 迁移对配置文件的改动已被回退；无需配置改动。`angle_rep` 值（0/1/2/3 与 decouple 配置的
  `True` 语义）保持现状。

## 9. 错误处理与不变量

1. **域纪律（import 审计）**：`physical_rad_to_norm` / `norm_to_physical_rad` 只允许被
   `deim_decoder.py`、`denoising.py`、`dfine_decoder.py` 引用。criterion、matcher、
   postprocessor、obb_geometry、transforms、eval 只允许使用物理域与 loss 规范域辅助函数
   （`canonicalize_phys_rad`、`physical_rad_to_loss_rad`）与 `obb_geometry.periodic_angle_distance`。
   以验收测试的 grep 断言强制。
2. **半开区间纪律**：所有 mod 运算用 `torch.remainder`；`canonicalize_phys_rad` 结果恒在
   `[0, pi)`，`physical_rad_to_loss_rad` 恒在 `[-pi/4, 3pi/4)`，`periodic_angle_distance`
   （`with_signal=True`）恒在 `[-pi/2, pi/2)`。
3. **可微性**：全部转换是算术操作（除法、乘法、`remainder`、sigmoid、logit），梯度通畅；
   `physical_rad_to_logit` 用 `eps` clamp（默认 `1e-4`）保证有限值。
4. **Round-trip 不变量**（验收测试断言）：
   - `norm_to_physical_rad(physical_rad_to_norm(theta)) == theta`，对任意 `theta in [0, pi)`
     精确至浮点舍入（等比映射无需 mod）。
   - 等价性：`periodic_angle_distance(physical_rad_to_loss_rad(pred), physical_rad_to_loss_rad(gt))`
     == `periodic_angle_distance(pred, gt)`（spec §6）。
5. **数值说明**：loss 域与物理域求得的周期距离逐元素相等至浮点舍入；测试用 `allclose`，
   边界用例避开精确 seam 点（见 7.2 节）。
6. **seam 位置纪律**：字面量 `pi / 4`（或 `0.25` 相关换算）只允许出现在
   `physical_rad_to_loss_rad` 公式与 criterion 内部；decoder、denoising、geometry、
   postprocessor、matcher、eval 中不允许出现 seam 平移。

## 10. 实施策略（用户改功能性代码，助手编写并运行测试）

当前工作区无需回滚。按下列顺序实施：**用户负责每个 Task 的功能性代码改动；助手在每个
Task 后编写/更新测试脚本、运行测试并审核 diff**，确认通过后再进入下一 Task：

1. **契约模块**：用户重写 `obb_angle_contract.py`（等比公式 + loss 规范辅助函数）→ 助手重写
   `test/test_obb_angle_contract.py`（新公式、边界表、等价性、round-trip）并运行。
2. **geometry GT 规范化**：用户修复 GT 角度为 `[0, pi)`（`obb_geometry.py:100,238` + docstring）
   → 助手更新 `test/test_obb_roundtrip.py`、`test/test_obb_transforms.py` 中依赖 shifted 域的
   断言并运行。
3. **decoder 输出与 anchor**：用户改四处 `(x−0.25)·π` → `x·π`（`:1100,1237,1240,1243`），
   `:996` 改 `r = 0.25` → 助手更新/扩展 `test/test_deimv2_obb_smoke.py`（输出域 `[0,pi)`、
   anchor 物理 `pi/4`）并运行。
4. **denoising**：用户改 `denoising.py:110-111` 为 `physical_rad_to_norm` → 助手新增测试
   GT `3pi/4 → 0.75` 并运行。
5. **criterion**：用户改非周期路径为 `periodic_angle_distance` 度量；助手新增 seam 样例
   测试（`0.75pi` vs `0.70pi`）并运行。
6. **可选锁定**：`yolo_angle_loss` / `compute_angle_cost_matrix` 保留现状，助手以一致性
   测试锁定其与 `periodic_angle_distance` 等价。
7. **清理与审计**：用户删除 `__init__.py` 的 `obb_codecs` 注释、恢复 `OBB_CODE_REVIEW.md`
   等比描述 → 助手新增 import 审计测试（§9.1）、seam 字面量审计测试（§9.6）并运行。
8. **集成验证**：助手运行全量 OBB 测试套件（smoke、criterion、roundtrip、transforms、eval、
   adr）、短迭代训练 smoke、`convert_to_deploy` smoke。

## 11. 验收测试

### 11.1 单元测试（`test/test_obb_angle_contract.py` 重写）

1. 边界映射表（7.1 节：0、pi/4、pi/2、3pi/4、pi - 1e-3）精确断言 `theta_norm` 与 `theta_loss_rad`。
2. 等比 round-trip：随机 1000 个 `theta in [0, pi)`，`norm_to_physical_rad(physical_rad_to_norm(theta))`
   与 `theta` 满足 `allclose(rtol=1e-6)`。
3. 等价性：随机 1000 组 `(pred, gt)`，loss 域与物理域 `periodic_angle_distance`
   满足 `allclose`。
4. logit round-trip：远离 0 与 pi 的 `theta`，`logit_to_physical_rad(physical_rad_to_logit(theta))`
   收敛。
5. 残差示例表（7.2 节）逐项断言，含 seam 样例 `pred = 0.75pi, gt = 0.70pi` 周期距离 `0.05pi`。

### 11.2 geometry 测试（`test/test_obb_roundtrip.py`、`test/test_obb_transforms.py`）

1. `xyxyxyxy_to_xywhr` 输出 `theta in [0, pi)`（含 `3pi/4` 边界样例）。
2. 变换后 GT 角度仍 `in [0, pi)`；水平翻转 `canonicalize_phys_rad(pi - theta)` 正确。

### 11.3 decoder 契约测试（`test/test_deimv2_obb_smoke.py`）

1. 固定输入下 decoder 输出角度 `in [0, pi)` 且等于期望物理角（四种 `angle_rep` 各一组）。
2. anchors 默认 `r = 0.25` 对应物理角 `pi/4`，与迁移前物理初始化方向一致；`angle_step > 0`
   时候选角等比分布于 `[0, 1)`。
3. denoising：GT `theta = 3pi/4` 产出的归一角为 `0.75`（无 clip）；GT 接近 pi 时归一角接近
   1 且不越界。
4. `dfine_decoder` 旋转采样点：5 维输入的第 5 维按 `theta_norm * pi` 解释，旋转矩阵与物理角一致。

### 11.4 criterion 测试（`test/test_deim_criterion_obb_loss.py` 等）

1. seam 两侧样例：`pred = 0.75pi, gt = 0.70pi` 时非周期路径 angle_term 等于周期路径
   （`|delta|/pi` 度量），且不等于直接相减结果。
2. 非周期路径在 `theta > 3pi/4` 时无越界值（旧实现缺陷回归测试）。
3. `yolo_angle_loss` / `compute_angle_cost_matrix` 与 `periodic_angle_distance` 一致（`allclose`）。

### 11.5 域纪律审计（新增或并入现有测试）

1. grep 断言：`criterion`、`matcher`、`postprocessor`、`obb_geometry`、`obb_transforms` 中无
   `physical_rad_to_norm` / `norm_to_physical_rad` / `theta_norm` 引用。
2. grep 断言：seam 字面量（`pi / 4` 平移）只出现在 `physical_rad_to_loss_rad` 与 criterion 内部。

### 11.6 集成测试

1. `test/test_obb_roundtrip.py`、`test/test_obb_loss_integration.py`、`test/test_obb_eval.py`、
   `test/test_obb_transforms.py`、`test/test_deimv2_obb_smoke.py`、`test/test_deim_criterion_obb_loss.py`
   全绿。
2. 短迭代训练 smoke（现有入口）：四种 `angle_rep` 各跑少量 step，损失有限且正常下降。
3. `convert_to_deploy` 与 ONNX 导出 smoke 通过。

## 12. 明确排除项（不做的事）

1. 不保留旧 shifted seam 映射的任何语义（函数名可复用，公式必须为等比）。
2. 不保留 `theta_loss_rad` 直减路径：criterion 中所有角度差必须经 `periodic_angle_distance`。
3. 不在 decoder、denoising、geometry、postprocessor、matcher、eval 中出现 seam 平移。
4. 不改变模块边界，不合并 `angle_rep` 路径，不重建 `obb_codecs.py`。
5. 不做 anchor 角度初始化（`pi/4` vs `pi/2`）的对比实验；本设计只保持迁移前的物理初始化
   方向 `pi/4`。
6. 不修改 decoder 逐层精修中已正确的等比换算与 `dfine_decoder.py` 的旋转采样换算。
