# DEIMv2 OBB 角度量纲全链路审计报告（只读）

日期：2026-08-04
范围：`deimv2_daod/engine`（decoder / criterion / matcher / denoising / postprocessor / eval / data）
性质：**只读审计，未修改任何代码。** 本报告记录证据与结论，不提出修复方案。

---

## 1. 统一符号约定

同一物理方向（π 周期）在代码中存在四种仿射表示，本报告统一记为：

| 符号 | 定义 | 范围 |
|---|---|---|
| `θ_ext` | 对外几何弧度（canonical，数据/评估/输出契约） | `[-π/4, 3π/4)` |
| `θ_code` | 平移后的内部编码弧度，`θ_code = θ_ext + π/4` | `[0, π)` |
| `t` | 外部角的 turn 比例，`t = θ_ext/π` | `[-0.25, 0.75)` |
| `n` | sigmoid/logit 内部编码，`n = t + 0.25 = θ_code/π = (θ_ext+π/4)/π` | `[0, 1)` |

换算公式：

```
θ_code = θ_ext + π/4        θ_ext = θ_code - π/4
n      = θ_code / π         θ_code = n · π
n      = (θ_ext + π/4) / π  θ_ext  = (n - 0.25) · π
```

**关键风险语义**：`[0, π)` 与 `[0, 1)` 这两类"看起来无害"的范围，若同一 tensor 被不同消费者当成 `θ_code` 或 `n`，差一个 `π` 因子；若被当成 `θ_ext`，差一个 `π/4` 平移。审计的全部问题都源于此。

---

## 2. 全链路角度域总览表

| 环节 | 文件:行 | 角度域 | 备注 |
|---|---|---|---|
| 数据集 polygon→OBB | `dota_dataset.py` | `θ_ext ∈ [-π/4,3π/4)` | `xyxyxyxy_to_xywhr` 用 `remainder(θ+π/4,π)-π/4` canonicalize |
| 增强 `OBBFlip`（单独） | `obb_transforms.py` | 翻转后写 `[0,π)` | 单独执行时角度域跳变 |
| 增强 `OBBResize`/crop 等 | `obb_transforms.py` | `θ_ext ∈ [-π/4,3π/4)` | 经 affine→polygon→refit 重新 canonicalize |
| mosaic 增强 | `mosaic.py` | `θ_ext ∈ [-π/4,3π/4)` | polygon refit 重新 canonicalize |
| denoising GT 编码 | `denoising.py` | `θ_ext → n → inverse_sigmoid` | 编码侧正确 |
| anchor 生成 / top-k | `deim_decoder.py:1075-1138` | `n ∈ [0,1)` | sigmoid 输出 |
| decoder attention | `dfine_decoder.py` | `n`（用于 `MSDeformableAttention` 采样） | `reference_points[...,4:5] * π`，仅对 n 成立 |
| 几何 decode（5D） | `deim_decoder.py:441-463` | `θ_ref = n·π ∈ [0,π)`；残差按 rep 不同 | rep1 额外 `×π`（见 §4.2） |
| 几何 decode（6D） | `deim_decoder.py:441-463` | `θ_ext ∈ [-π/4,3π/4)` → 归一化 n | rep0/rep2 各自问题（见 §4.1/§4.3） |
| 最终输出转换 | `deim_decoder.py:1240-1250` | `n → θ_ext`：`(x-0.25)·π` | 对所有 rep 无条件执行 |
| criterion FGL | `deim_criterion.py:403-447` | `ref_points=θ_ext`、`GT=θ_ext` 弧度 | `out_refs` 已转 θ_ext，两侧一致 |
| criterion 其他损失 | `deim_criterion.py:363-392` | `θ_ext` 弧度；L1 分支转 n | `periodic_angle_distance` 弧度用法正确 |
| matcher | `matcher.py` | `θ_ext` 弧度 | ProbIoU / angle cost |
| postprocessor | `postprocessor.py` | 角度 ×1.0 原样 | 仅缩放空间维，正确 |
| eval | `obb_eval.py` / `dota_eval.py` | `θ_ext` 弧度 | 与 GT 直接 `batch_probiou` |

---

## 3. 各环节证据

### 3.1 数据与增强
- `xyxyxyxy_to_xywhr` 输出 `θ_ext ∈ [-π/4, 3π/4)`（`remainder(θ+π/4, π) - π/4`）。
- `OBBFlip` 单独执行后角度写为 `[0, π)`；常见 pipeline 后续 `OBBResize → affine_obb → xyxyxyxy_to_xywhr` 会重新 canonicalize 回 `[-π/4, 3π/4)`。
- 数值探针：输入 `-π/8`，flip 后 `π/8`；再经 resize/refit 仍为 canonical `π/8`。
- **风险**：任何只含 flip 的 pipeline（或单独调用）会留下 `[0,π)` 角度进入训练，与后续 canonical 消费者不一致。

### 3.2 denoising
- `denoising.py`：GT `θ_ext → n=(θ_ext+π/4)/π → inverse_sigmoid`，编码侧正确。
- **rep2 例外（已确认问题）**：`denoising_bbox_unact` 已是 logit，`deim_decoder.py:1124-1137` 对 rep2 直接送 `oriented_box_to_external_rect`，混淆 logit、归一化坐标与弧度三种量纲（探针：GT `[0.5,0.5,0.4,0.2,0]` 的角 logit `-1.0986` 被当弧度处理，输出不还原原框）。

### 3.3 anchor / top-k / encoder
- 锚点角度为 `n ∈ [0,1)`（sigmoid 输出）。
- `deim_decoder.py:1103-1108`：`angle_rep != 2` 时 top-k 框角度 `(x-0.25)·π` → `θ_ext`；rep2 保持原值。
- `angle_step>0` 时 `memory.repeat_interleave(n_angles)` 与锚点布局对齐（此前已修复的 mangle bug）。

### 3.4 decoder 内部参考点
- `ref_points_unact → sigmoid` 得到 `n ∈ [0,1)`（`deim_decoder.py:267-290`）。
- rep2/3 走独立角度 head 分支：rep3 的 `dec_angle_initial = sigmoid(...)` 为 `n`；rep2 的 `pre_bboxes = external_rect_to_oriented_box(...)` 为 `θ_ext` 弧度（`deim_decoder.py:408-415`）。

### 3.5 几何 decode（`deim_decoder.py:436-463`）
- `ref_points_initial_scaled = ref_points_initial * theta_scale`，`theta_scale[...,4] = π`：**对所有 rep 无条件执行**。
- 5D（rep1/rep3）：`distance2bbox_obb` 5D 分支 `θ_new = (θ_ref + distance/reg_scale) % π`，其中 `θ_ref = n·π`。
- 6D（rep0/rep2）：`θ_ref = n·π`（rep0）或 `θ_ext`（rep2）被 `×π` 后再进 6D 几何，见 §4。

### 3.6 criterion FGL（`deim_criterion.py:403-447`）
- `ref_points = outputs["ref_points"][idx]` = `out_refs[-1]`，**已在 `deim_decoder.py:1245-1247` 转换为 `θ_ext` 弧度**。
- `target_boxes = t["boxes"]` = `θ_ext` 弧度（数据侧 canonical）。
- 两侧均为弧度 → FGL 目标量纲一致，无 n/弧度混算问题（本审计修正了早前"criterion 混用 n 与弧度"的中间假设，证据见 §2 表与 §4）。
- `bbox2distance_obb` 5D 分支：`angle_lens = periodic_angle_distance(θ_ref, θ_gt, signed=True)·reg_scale`，**纯弧度、无 π**（`dfine_utils.py:306-314`）。
- 6D 分支：`oriented_box_to_external_rect` 几何一致（`dfine_utils.py:290-305`）。

### 3.7 matcher / 其他损失 / postprocessor / eval
- matcher 与 `periodic_angle_distance` 均为弧度正确用法。
- postprocessor 角度 `×1.0`，正确。
- eval 直接 `batch_probiou(θ_ext)`，与 GT 同域。

---

## 4. 逐 angle_rep 结论

### 4.0 总览

| rep | obbox_rep_dim | 解码路径 | 量纲一致性 | 严重度 |
|---|---|---|---|---|
| 0 | 6 | 6D 外接矩形 | **边界问题**：零残差固定 +π/4 物理旋转；ref 单位 n·π vs criterion θ_ext 不一致 | 高 |
| 1 | 5 | 5D 直接角度 | **单侧 ×π 单位错误（本轮实锤）** | 高 |
| 2 | 6 | 6D 外接矩形 | **ref 已为 θ_ext 却被再次 ×π**；denoising logit/弧度混淆 | 高 |
| 3 | 5 | 5D 直接角度 | **完整一致（唯一自洽路径）** | — |

### 4.1 rep0（基线，6D 解码，ref 为 n）
- `ref_points_initial[...,4]` 为 `n ∈ [0,1)`；`deim_decoder.py:443` 乘 `π` 得 `θ_ref = n·π`，而 criterion 的 ref 为 `θ_ext = (n-0.25)·π`。两者相差固定 `π/4`。
- 数值探针：`n=0.25`（外部应 `θ_ext=0`），零残差 6D 解码输出 `θ_ext = +π/4`（固定物理旋转）；`θ_ref = 0.25π` 的 ext-rect 与 criterion 假设的 ext-rect 不一致，ε,η 残差的缩放基准漂移。
- 结论：6D 解码把 `n` 直接当 `θ_code` 使用（`n·π = θ_code`），等价于用了平移后的编码角度而非外部角度。

### 4.2 rep1（5D 解码，`distance[...,4] *= π`）—— 本轮最终实锤
- **criterion 侧（无 π）**：FGL 目标 `angle_lens = Δθ(rad)·reg_scale`（`dfine_utils.py:306-314`；ref 与 GT 均为 `θ_ext` 弧度）。
- **decoder 侧（×π）**：`deim_decoder.py:446-447` 仅对 `angle_rep==1` 执行 `distance[...,4] *= torch.pi` → `θ_new = θ_ref + π·Δθ`。
- rep3 走完全相同的 FGL 与 5D 解码（`obbox_rep_dim=5`），**无此 π**，且解码正确。二者唯一差异即此 `×π`。
- 数值探针（`distance2bbox_obb` 实测）：
  ```
  target Δθ= 0.100000   rep3 decoded= 0.100000   rep1 decoded= 0.314159
  target Δθ=-0.200000   rep3 decoded=-0.200000   rep1 decoded=-0.628319
  target Δθ= 0.785398   rep3 decoded= 0.785398   rep1 decoded=-0.674191   (π·π/4 mod π)
  ```
- 后果：rep1 角度残差被放大 π 倍；DFL 目标 `π·Δθ·reg_scale` 易超 `reg_max` 被 `clamp(min=0,max=reg_max-eps)`（`dfine_utils.py:360`）截断，等效角度分辨率损失 1/π，且与同链路的 matcher/criterion 弧度代价不一致。网络可"学会"输出 Δθ/π 补偿，但属于不必要的退化。
- 结论：**rep1 单侧 `×π` 是确定的量纲单位错误**（此前 Oracle 标记"残差角可能额外乘 π"，本轮以代码对照 + 数值探针定案）。

### 4.3 rep2（6D 解码，ref 为 θ_ext）
- 非 angle-first 分支 `pre_bboxes = external_rect_to_oriented_box(...)`（`deim_decoder.py:408-411`）返回 `θ_ext ∈ [-π/4, 3π/4)`。
- 随后 `deim_decoder.py:443` 通用 `theta_scale[...,4] *= π` 把 `θ_ext` 再乘 π → `θ_ext·π`（范围 `[-0.785π, 2.356π)`），6D 几何解码的 cos/sin 输入量纲损坏。
- 数值探针：6D-derived `θ_ext=-0.3805` 被 `×π` 成 `-1.1954`，零残差解码得到 `1.9462 rad`。
- 另：denoising 侧 logit 与弧度混淆（§3.2）。
- 结论：rep2 的 `θ_ext` 被重复缩放，几何解码量纲损坏。

### 4.4 rep3（5D 解码，无 π）—— 唯一完整一致路径
- ref 为 `n`，`×π` 后 `θ_ref = n·π = θ_code ∈ [0,π)`；criterion ref/GT 为 `θ_ext` 弧度；FGL 5D 无 π；解码 `θ_new = (θ_ref + Δθ) % π`；归一化 `n' = θ_new/π`；输出 `(n'-0.25)·π = θ_ext`。全部自洽。
- 数值探针：目标 Δθ 精确还原（§4.2 表 rep3 列）。
- `use_angle_first=True` 仅与 rep3 兼容（`deim_decoder.py:561-565` 对 rep2 主动报错）。

---

## 5. 旧注释 / 文档 / 测试偏差清单

| 位置 | 内容 | 与实际不符处 |
|---|---|---|
| `dfine_utils.py:201,265` | docstring 声称 `θ in [0,π]` | 实际 canonical 输出/GT 为 `θ_ext ∈ [-π/4,3π/4)`；内部 5D 解码中间量才是 `[0,π)` |
| `dfine_utils.py:202` | `deta_theta in [0,π]` | 语义含混，易被误读为需乘 π |
| `obb_geometry.py:46` | `xywhr_to_xyxyxyxy` docstring `θ∈[0,π]` | 同 §5 第一行 |
| `obb_transforms.py` 文件头 | 旧 `[0,π)` 外部契约 | 与当前 `[-π/4,3π/4)` 不一致 |
| `docs/superpowers/review/OBB_CODE_REVIEW.md` | 旧 `[0,π)` 外部契约 | 同上 |
| `test_deimv2_obb_smoke.py:241-245` | 断言输出角 `[0,π]` | 实际输出 `θ_ext ∈ [-π/4,3π/4)`（含负角）；断言当前"通过"是因为合成输入零残差输出退化为恒定 π/4（见 §7），非量纲正确性证据；rep2 若接入同一断言会因负角失败 |
| 既有测试覆盖 | 仅形状/有限性/独立公式 | 未覆盖"零残差物理框不变性"与"非零残差还原"，故 rep0/rep1/rep2 的单位问题无法被现有测试捕获 |

---

## 6. 问题边界与严重度汇总

| # | 位置 | 问题 | 严重度 | 证据 |
|---|---|---|---|---|
| 1 | `deim_decoder.py:446-447` | rep1 单侧 `distance[...,4]*=π`，FGL 目标弧度 vs 解码 π× | 高（确定性单位错误） | §4.2 代码对照 + 数值探针 |
| 2 | `deim_decoder.py:443` | 通用 `theta_scale[...,4]*=π`：rep0 将 `n` 当 `θ_code`（零残差固定 +π/4）；rep2 将已为 `θ_ext` 的 ref 再乘 π | 高 | §4.1/§4.3 数值探针 |
| 3 | `deim_decoder.py:1124-1137` | rep2 denoising：logit 直接送 `oriented_box_to_external_rect`，logit/归一化坐标/弧度混淆 | 中-高 | §3.2 数值探针 |
| 4 | `dfine_decoder.py` `MSDeformableAttention` | `reference_points[...,4:5]*π` 仅对 n 输入成立；对 6D 转换后直接得到的 `θ_ext` 输入不成立 | 中 | 代码结构 |
| 5 | `OBBFlip` 单独执行 | 翻转后写 `[0,π)`，与 canonical 消费者不一致 | 低-中 | §3.1 探针 |
| 6 | docstring/文档/测试 | 多处旧 `[0,π)` 契约，且测试不覆盖量纲不变性 | 低 | §5 |

**结论**：四种角度表示中，**rep3 是唯一完整自洽路径**；rep0、rep1、rep2 各有可达的表示边界问题，其中 rep1 的 `×π` 为本轮最终确认的单侧量纲错误（编码无 π、解码乘 π），其余为解码 ref 单位选择与通用 `×π` 的相互作用。

---

## 7. 证据记录（本轮新增）

1. `dfine_utils.py:194-248` 5D/6D 解码；`251-362` 5D/6D FGL 编码——5D 编码无 π、解码无 π（除 rep1 显式 ×π）。
2. `deim_criterion.py:403-447` FGL 调用：`ref_points=outputs["ref_points"]`（= `out_refs[-1]`，已转 `θ_ext`），`target_boxes=t["boxes"]`（`θ_ext`）——两侧弧度一致。
3. `deim_decoder.py:1245-1247` `out_refs` 先转 `θ_ext` 再入 `outputs["ref_points"]`。
4. `deim_decoder.py:441-447` 唯一 rep 分叉：`distance[...,4] *= π` 仅 rep1。
5. 数值探针（rep1 vs rep3 同目标 Δθ）：rep1 解码 = π× 目标，rep3 = 精确目标。
6. 四 rep 同构合成冒烟配置实证（零残差/初始权重，4 种子）：

   | angle_rep | 输出 θ 范围（4 种子合并） | 解读 |
   |---|---|---|
   | 0 | `[0.0000, 1.5708]` = `[0, π/2]` | 真值参考 `θ_ext=(n−0.25)π∈[−π/4,π/4]` 整体 +π/4 平移：min=0 而非 −π/4——rep0 零残差 +π/4 伪影实证 |
   | 1 | 恒定 `0.7854` = π/4 | 零残差保留参考角（锚点 n=0.5 → θ_ext=π/4）；`×π` 在零残差下不可见（×0=0），需非零残差探针（见证据 5） |
   | 2 | `[-0.6305, 2.2877]` | 远宽于参考范围 `[−π/4, π/4]`——ref `θ_ext` 被 443 行通用 `×π` 重复缩放后 6D 几何输出被扭曲，实证 |
   | 3 | 恒定 `0.7854` = π/4 | 零残差正确保留参考角（自洽路径应有行为） |

7. 冒烟测试 `test_deimv2_obb_smoke.py:241-245` 的 `[0,π]` 断言：当前"通过"是因为合成零残差输出恰为恒定 π/4（证据 6 的 rep1 行），是退化解巧合而非量纲正确性证据；该断言亦与当前 `θ_ext ∈ [-π/4,3π/4)` 外部契约不符（rep2 接入即因负角失败）。
