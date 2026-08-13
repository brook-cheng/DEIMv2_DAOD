# 多视图规范坐标 DEIMv2-OBB 架构设计

> **状态**: Design Spec（定向文献检索 + 本地源码审计 + 决策完备设计；不含实现计划，无待办项）
> **日期**: 2026-08-13
> **范围**: 全局视图与局部裁剪视图在解码器之前的特征级融合（Route A）；Route C 列为研究后备；Route B 否决为生产主路线
> **相关工件**: `.omo/ulw-research/20260813-145641/`（研究日志，本设计为其结论）；`docs/superpowers/specs/2026-06-23-deimv2-obb-geometry-spec.md`；`docs/superpowers/plans/2026-07-31-deimv2-obb-five-proposals.md`
> **约束声明**: 本文档只做设计与研究决策，不修改任何生产代码；所有本地契约均以当前工作区 `engine/deim/` 源码为准，行号为撰写当日核对值

---

## 1. 执行摘要

本设计回答一个问题：如何让"整幅规范视图 + 若干局部裁剪视图"以**特征级**方式进入同一个 DEIMv2-OBB 检测器，同时保持锚点生成、去噪（DN）、匹配器、损失函数与 PostProcessor **一行不改**。

回答分四层：

1. **几何层**。规范画布 C 必须是对原始图像 O 的**相似性变换**（统一缩放、零平移、零填充）结果。唯有如此，C-归一化坐标才恒等于 O-归一化坐标，现有 GT 归一化、matcher/DN/损失与 PostProcessor 的 `[W,H,W,H,1]` 像素缩放契约（`postprocessor.py:60-67`）才能原样成立。各向异性缩放会剪切旋转矩形，使五参数 OBB 几何失真（θ 与 w/h 比均被破坏）；letterbox 填充引入非零偏移，PostProcessor 不含偏移项，输出坐标必然偏置。两者都被排除。

2. **架构层**。三路方案对比：Route A（解码器前特征级融合）推荐；Route C（规范查询、变换感知采样的多视图解码器，DETR3D/BEVFormer 式）列为研究后备；Route B（逐视图解码 + 预测级合并）否决为主路线，原因是 OBB 旋转 NMS 在跨视图合并时的不稳定性、单一匹配器语义被破坏、以及接缝处重复/漏检框问题，SAHI 与 RODM-NMS 文献正说明这类合并需要额外专门机制。

3. **融合层**。Route A 采用**归一化门控残差**融合：局部视图特征经相似性投影到规范三级网格，与全局基线 G 逐像素做残差，残差经 `active·coverage·taper·σ(gate)` 加权、归一化和（`ΣW·R / max(1, ΣW)`）、`tanh` 有界化后，以 `α = tanh(β)` 缩放加回 G。β 初值为 0，故**初始化即恒等**：训练开始时 Route A 与当前单视图基线逐位等价，这保证了可回退与可消融。

4. **契约层**。GT 仍是单一规范坐标集（每对象一个，不做逐视图 GT）；可见性只影响特征覆盖，不影响 GT 归属；DN/matcher/loss/输出全部保持现有语义。先实现固定宽高比模式（DOTA 切块场景，结构张量可跨场景复用），动态宽高比（随场景宽高比）作为无结构缓存扩展。

结论速览：

| 主题 | 结论 |
|---|---|
| 规范画布 | 相似性 + 零偏移 + 无填充，C-归一化 ≡ O-归一化 |
| 局部视图 | 相似性渲染的特征源，非预测源 |
| 融合方式 | 解码器前、逐层级、逐像素归一化门控残差；禁止原始 token 拼接、禁止稀疏唯一融合 |
| 推荐路线 | Route A（特征级融合），承认其成本不低 |
| 研究后备 | Route C（规范查询多视图解码器） |
| 否决 | Route B（预测级合并），含旋转 NMS 依赖 |
| 外部参考 | UHR-DETR 仅论文、无可用代码，仅作思想参考 |
| PostProcessor | 不变（`postprocessor.py:60-67`） |

---

## 2. 原始五步提案的形式化

用户的原始构想可形式化为五个步骤，每步附带一条不变式声明：

**第一步（规范画布）**。对原始图像 O（像素尺寸 W×H）施加保持宽高比的统一缩放，得到规范画布 C。不变式：C 与 O 宽高比相同，缩放为各向同性（相似性变换）。

**第二步（多视图）**。以 C 为全局视图，同时提供若干局部裁剪视图作为补充；各视图特征进入同一个检测流程。原始构想中全局视图允许非均匀缩放。不变式（待检验）：所有视图都贡献到同一个规范坐标系。

**第三步（特征直接混合）**。各视图特征"直接混合"进检测器。原始构想未规定坐标变换与可见性处理。不变式（待检验）：所有被混合的 token/特征共享同一个几何解释。

**第四步（规范训练）**。锚点、DN、matcher、损失函数与 head 输出全部保持在第一步定义的规范坐标系。不变式：规范坐标是唯一权威坐标。

**第五步（后处理不变）**。PostProcessor 保持现有实现与 `[W,H,W,H,1]` 缩放契约，不做任何修改。不变式：模型输出坐标为 C-归一化，乘原图尺寸即得原图像素坐标。

五步的内在张力在于：第二、三步的"多视图直接混合"与第四、五步的"单一规范坐标 + 后处理不变"之间，缺少一个显式的**变换感知融合**环节。这正是本设计要补上的部分。

---

## 3. 本地既定契约（file:line 引用）

以下契约全部来自当前工作区源码，是 Route A 必须保持不变的硬约束。

### 3.1 编码器契约：三级稠密 256 通道金字塔

- `engine/deim/hybrid_encoder.py:342-343`：`feat_strides=[8, 16, 32]`，`hidden_dim=256`。
- `engine/deim/hybrid_encoder.py:367`：`out_channels = [hidden_dim for _ in range(len(in_channels))]`，即三个输出层全部为 256 通道。
- `engine/deim/hybrid_encoder.py:396`：`input_dim = hidden_dim if self.fuse_op == 'sum' else hidden_dim * 2`，DEIM 采用 `fuse_op='sum'`（求和而非拼接）。
- `engine/deim/hybrid_encoder.py:484, 494`：FPN 路径的上下采样融合均为求和。

推论：全局视图 G 提供**稠密**三级基线特征（stride 8/16/32、256ch）。任何"稀疏唯一"的融合方案（如 UHR-DETR 的稀疏编码器风格）会破坏稠密金字塔契约，不在本设计范围内。

### 3.2 解码器契约：规则网格假设

- `engine/deim/deim_decoder.py:577-578`：构造默认值为 `num_levels=3, num_points=4`；活跃 OBB 配置 `configs/custom_obb/deimv2_obb_common.yml:108` 将每层采样点设为 `[3,6,3]`。本设计的硬约束是**三级规则网格与既有采样点配置保持不变**，不是把采样点强制固定为标量 4。
- `engine/deim/deim_decoder.py:990-1013`：`_get_encoder_input` 将每层 `[b,c,h,w]` 展平为 `[b,h*w,c]` 并沿第 1 维拼接（**行主序、层主序**），`spatial_shapes` 按层记录 `[h, w]`。memory 的几何解释完全由 `spatial_shapes` 决定。
- `engine/deim/deim_decoder.py:1015-1114`：`_generate_anchors` 在每层生成**规则网格**锚点：`grid_xy = (meshgrid + 0.5) / [w, h]`（x 按 w 归一化、y 按 h 归一化，`deim_decoder.py:1070-1072`），宽高 `wh = grid_size · 2^lvl`。
- `engine/deim/deim_decoder.py:1108-1112`：锚点边界掩码 `valid_mask`（锚点在 `(eps, 1-eps)` 内才有效），越界锚点置 `torch.inf`；`deim_decoder.py:882-886` 将其注册为 buffer，`deim_decoder.py:1182` 用它乘 memory。
- `engine/deim/deim_decoder.py:1186-1187`：`_select_topk` 从编码器输出选取 top-k 查询，作为解码器初始查询。
- `engine/deim/dfine_decoder.py:48, 123-163`：`MSDeformableAttention` 的 `reference_points` 为 `[bs, Lq, n_levels, 2]`、取值 `[0,1]`（`dfine_decoder.py:133`）；`spatial_shapes` 先由 `[h,w]` 翻转为 `[w,h]` 构造 `offset_normalizer`，再用于归一化 `sampling_offsets`（`dfine_decoder.py:154-161`）；OBB 路径把角度通道经 `shifted_norm_to_physical_rad` 解码为物理角（`dfine_decoder.py:185`）。

**关键结论：解码器 memory、锚点、可变形采样全部假设"规则稠密网格 + 规整 spatial_shapes"。任何不落在规则网格上的特征（裁剪视图原始特征、任意覆盖掩码）都不能直接进入现有解码器。** 这就是"原始 token 拼接"在几何上无效的根本原因，也是 Route C 需要解码器内部重写的根本原因。

### 3.3 训练契约：归一化规范坐标

- matcher 与损失在**归一化 cxcywhθ** 空间工作，θ 为物理角 `[0, π)`；`matcher.py:41` 的 `angle_factor = π`。
- 非周期角度 L1 路径经 `physical_rad_to_norm` 归一化：`deim_criterion.py:382, 389`；`engine/deim/obb_angle_contract.py:33, 45, 117` 定义 `physical_rad_to_norm` / `norm_to_physical_rad` / `shifted_norm_to_physical_rad`。
- DN 输入角度按配置编码为解码器内部归一化表示：默认 shifted 分支在 `denoising.py:116-118` 使用 `physical_rad_to_shifted_norm`，proportional 分支在 `denoising.py:120` 使用 `physical_rad_to_norm`。
- 锚点角度初始化为归一化值（shifted 编码下默认 `r=0.5`，`deim_decoder.py:1093-1096`）。

推论：若规范画布 C 满足"统一缩放 + 零偏移"，则 C-归一化与 O-归一化是同一坐标系，上述所有训练组件原样可用，GT 归一化方式与当前流水线完全一致。

### 3.4 推理契约：orig_size 与 PostProcessor

- 数据集将 `orig_size` 存为 `[w, h]`（宽在前）：`engine/data/dataset/coco_dataset.py:174`、`dota_dataset.py:229`、`voc_detection.py:74`、`obb_transforms.py:67, 97, 220`。
- `engine/deim/postprocessor.py:60-67`（OBB 分支）：`img_w = orig_target_sizes[:, 0:1]`，`img_h = orig_target_sizes[:, 1:2]`，`factor = [W, H, W, H, 1]`，`bbox_pred = boxes * factor`，注释明确"θ 不变（归一化到 [0,π)）"。
- 评估路径 `engine/eval/obb_eval.py:268-271` 直接堆叠 `t["orig_size"]` 传给 postprocessor。

推论：PostProcessor 是一个**纯逐轴缩放、无偏移项、角度不变**的映射。它能否原样工作，完全取决于模型输出的归一化坐标是否与"原图像素 ÷ 原图尺寸"一致。

### 3.5 变更纪律

`engine/core/workspace.py:182`：注册构造函数遇到多余 key 会直接抛 `TypeError`（签名过滤被注释）。因此本设计新增的任何组件都不能向现有已注册组件注入新构造参数，只能作为新注册组件出现。

---

## 4. 坐标空间证明

### 4.1 相似性变换 vs 各向异性缩放

OBB 五参数表示为 `(cx, cy, w, h, θ)`，θ 为从 x 轴正向到 w 方向的角度（le90 约定，`obb_geometry.py` 与 `2026-06-23-deimv2-obb-geometry-spec.md`）。

**相似性变换** `(x', y') = s·(x, y) + t`（s 为标量）：旋转矩形四个顶点经同一各向同性缩放后仍是旋转矩形，且 θ 不变、w/h 比不变、中心平移一致。对五参数而言：`(cx', cy', w', h', θ') = (s·cx + tx, s·cy + ty, s·w, s·h, θ)`。OBB 类在相似性变换下**封闭**。

**各向异性缩放** `(x', y') = diag(sx, sy)·(x, y) + t`：旋转矩形顶点经非均匀缩放后，边长方向不再互相垂直（一般情形下变成平行四边形）。用 `xyxyxyxy_to_xywhr`（`obb_geometry.py:72`）重新拟合得到的 θ′ 与 θ 不同、w′/h′ 与 w/h 不同；w≈h 的方形框会被剪切为菱形。OBB 类在各向异性缩放下**不封闭**。

本项目已有实证佐证：先前在线/离线 OBB 几何排查（affine refit 导致 AP75 下降）正是各向异性几何失真的体现（`.omo/ulw-research/20260813-145641/observation-manifest.md` O5）。

**结论**：全局视图必须是相似性渲染。各向异性全局视图只能作为"不承载几何解释的上下文"，而本设计选择更简单的路径：全局视图就是规范视图本身，不做各向异性。

### 4.2 [H, W] 与 [W, H] 约定

两处约定方向相反，必须严格区分：

- **数据/后处理侧**：`orig_size = [w, h]`（宽、高），`postprocessor.py:62-63` 取 `img_w = orig[:, 0]`、`img_h = orig[:, 1]`，因子 `[W, H, W, H, 1]`。
- **解码器侧**：`spatial_shapes = [h, w]`（高、宽，`deim_decoder.py:1009`）；锚点 x 用 `w` 归一化、y 用 `h` 归一化（`deim_decoder.py:1070-1072`）；`MSDeformableAttention` 在 `dfine_decoder.py:155-158` 将 `[h,w]` 翻转为 `[w,h]` 后构造 `offset_normalizer`，与采样坐标的 `(x,y)` 顺序对齐。

也就是说：归一化坐标 `(cx, cy)` 中 cx 的像素恢复是 `cx·W`，cy 是 `cy·H`。只要规范画布 C 与 O 宽高比一致（C = s·O），这套双向约定就自动自洽；若 C 采用与 O 不同的宽高比（转置、拉伸、letterbox），任何一侧约定都必须跟着改，PostProcessor 不变性随即失效。

### 4.3 零偏移无填充 C 是 PostProcessor 不变的充要条件

**定理**：设 O 像素尺寸 (W, H)。由 4.1 的 OBB 封闭性可知，几何变换首先必须满足 `s_x = s_y = s`（相似性）；在此前提下，令 C 为 O 的统一缩放与平移渲染（letterbox 填充等一切偏移均归入 t_x/t_y）。PostProcessor 保持 `postprocessor.py:60-67` 不变且对所有输入给出精确原图像素坐标的**充要条件**是 t_x = t_y = 0（零偏移无填充）。

**必要性（反证）**：C 的像素映射为 `x_c = s·x_o + t_x`、`y_c = s·y_o + t_y`，画布尺寸 `W_c = s·W_o`、`H_c = s·H_o`。模型在 C-归一化坐标中预测 `u_c = x_c / W_c = (s·x_o + t_x) / (s·W_o) = x_o/W_o + t_x/(s·W_o)`。stock PostProcessor 用**原图宽度** W_o 缩放（`postprocessor.py:62-67`）：`x_out = u_c·W_o = x_o + t_x/s`。要 x_out ≡ x_o 对一切输入成立，中心坐标必须满足 t_x = 0；同理 y 中心要求 t_y = 0。宽高是端点差，平移会相消：`w_c = s·w_o`、`u_w = w_c/W_c`、`w_out = u_w·W_o = w_o`，因此宽高通道本身不额外约束平移；**零偏移条件由中心坐标通道充分且必要地给出**。注意 s 在归一化中被消去：任何统一缩放 s > 0 都不影响结论。若中心偏移非零而 PostProcessor 不改，输出恒偏 `t/s`；要精确恢复必须让 PostProcessor 获知 s 与 t 并执行偏移校正，即违反“不变”。故必要性得证。

**充分性**：t_x = t_y = 0 时 `u_c = x_o/W_o`、`v_c = y_o/H_o`（与 s 无关），PostProcessor 的 `u_c·W_o = x_o`、`v_c·H_o = y_o`、`w_c_norm·W_o = w_o`、`h_c_norm·H_o = h_o`、θ 不变，精确恢复原图像素 OBB。充要性得证。

**推论链**：

1. C-归一化 ≡ O-归一化，GT 归一化方式与当前流水线逐位一致（`dota_dataset.py:229` 之后的所有变换不变）。
2. 锚点、DN、matcher、损失、输出头的坐标系全部保持现状。
3. DOTA 评估路径（`obb_eval.py:268-271`）不变，输出直接与原始多边形标注对齐。
4. **letterbox 不是合法路径**：letterbox 填充产生非零平移（t_x、t_y ≠ 0），违反必要性。若坚持 letterbox，必须改 PostProcessor（违背第五步）或接受偏置坐标（错误），两者皆不可接受。故此设计明确**无 letterbox**。

### 4.4 局部视图的几何地位

局部裁剪视图是**相似性渲染的特征源**，不是预测源。每个视图 v 由原始像素坐标系中的裁剪矩形 `(x0_v, y0_v, x1_v, y1_v)` 和视图像素密度（缩放）s_v 完全刻画；视图像素坐标到 C-归一化坐标的映射是相似性（统一缩放 + 平移），由矩形元数据解析确定。视图内对象的五参数几何在视图自身的像素坐标系中保持正确（相似性封闭性），融合时经投影回规范网格，几何解释统一于规范坐标。**可见性只决定特征覆盖，不决定 GT 归属**（见 8.6）。

---

## 5. 外部文献与仓库对比

| 来源 | 类型 | 与本设计的关系 | 引用结论 |
|---|---|---|---|
| Deformable DETR [1] | 论文 | 可变形注意力 + 规则层级网格 | 锚点/采样假设规则网格；裁剪视图不能作为普通层级追加 |
| RT-DETR [2] | 论文+代码 | 本地 PostProcessor 与编码器结构来源 | sum 融合先例；后处理纯缩放契约的出处 |
| SNIP / SNIPER [3][4] | 论文 | 尺度不变性与上下文区域 | 尺度一致性训练先例；上下文区域即"局部视图"的早期形态 |
| DETR3D [5] | 论文+代码 | 规范查询、变换感知采样 | Route C 的理论先例；每查询经外参投影到多视图采样 |
| BEVFormer [6] | 论文+代码 | BEV 稠密网格查询融合多相机特征 | Route A 的稠密规范网格 + 多源投影的强先例 |
| SAHI [7] | 论文+代码 | 切片推理 + 跨瓦片 NMS 融合 | 预测级合并路径的真实代价；Route A 欲消除的复杂度 |
| DOTA [8] | 数据集 | 大图 + OBB 评估 | 切块必要性；评估必须对齐原图多边形 |
| RODM-NMS [9] | 论文 | 旋转框切片合并 NMS 特例处理 | 旋转 NMS 在合并场景存在两类已知失败，需专门机制 |
| R3Det [14] | 论文+代码 | 旋转检测特征重对齐 | 特征级对齐（FRM）比预测级修补更稳健的先例 |
| LGI-DETR [10] | 论文 | 编码器内局部-全局交互 | 局部增强 + 全局注入的编码器级融合先例（UAV 小目标） |
| UHR-DETR [11] | 论文（仅） | 超高分遥感端到端检测 | **仅论文，代码未发布**；稀疏编码器思想与本设计稠密契约不兼容，仅作思想参考 |
| DEIM [12] / D-FINE [13] | 论文+代码 | 本仓库的上游架构 | 匹配器/损失/后处理设计的出处 |

**逐项分析**：

**Deformable DETR（2010.04159）**：多尺度可变形注意力把每个查询的采样点定义为"参考点 + 每层偏移"，参考点与偏移都在规则层级网格上归一化。这意味着解码器对 memory 的要求是"每个层级是一张规则稠密特征图"。把裁剪视图的特征图当作额外层级追加（原始第三步的朴素读法）会破坏该假设：裁剪视图的网格不是全局规则的，其 `spatial_shapes` 无法表达视图坐标系。这是 C2 判定的外部依据。

**RT-DETR（2304.08069）**：本仓库的 PostProcessor 直接复制自 RT-DETR（`postprocessor.py:1-3` 注明出处），编码器也继承其结构（DEIM 将 concat 融合改为 sum，`hybrid_encoder.py:396`）。RT-DETR 的整图推理 + 纯缩放后处理范式说明：检测器家族默认假设"输出归一化坐标 → 原图尺寸缩放"这一契约。Route A 不破坏该家族契约。

**SNIP / SNIPER（1711.07289 / 1805.09300）**：SNIP 论证了尺度一致性（同一对象只在合适尺度训练）的重要性；SNIPER 把全局图像分解为"上下文区域"（context regions），以区域为训练单元。两者是"全局上下文 + 局部高分辨率区域"组合的早期先例，与 Route A 的"全局基线 G + 局部视图特征"精神一致；但 SNIP/SNIPER 仍是预测级（区域各自检测后合并），其合并复杂度正是 Route A 要规避的。

**DETR3D（2110.06922）**：3D 检测中，每个规范查询维护一个 3D 参考点，经相机外参投影到各视图 2D 坐标，再采样特征。这是"规范查询 + 变换感知采样"的经典实现，Route C 的直接先例。Route A 不采用查询级采样（那样要改解码器），而是把同样的变换感知思想下沉到**特征投影**层：把视图特征投影回规范网格，等价于 DETR3D 采样的对偶操作（特征到网格，而非查询到特征）。

**BEVFormer（2203.17270）**：BEV 网格查询在固定 BEV 栅格上采样多相机特征，输出稠密 BEV 特征。Route A 的"规范三级稠密网格 + 多视图投影 + 逐像素融合"在结构上与 BEVFormer 的稠密网格范式同族，且 BEVFormer 证明了稠密网格融合多源的可行性。差异在于：Route A 不引入 BEV 空间变换，而是保持图像平面规范坐标，以兼容现有锚点/匹配器/后处理。

**SAHI（2202.06934）**：切片推理 + 跨片 NMS 融合的工程范式，广泛用于 DOTA 类大图。其代价正是 Route B 的代价：重叠切块产生重复框，接缝处产生截断框，跨片 NMS 需要调阈值；对旋转框，这个问题更严重（见 RODM-NMS）。

**DOTA（1711.10398）**：本仓库 OBB 评测的目标数据集。DOTA 图像巨大、对象密集，评测要求预测多边形与原图对齐（`obb_eval.py` 用 orig_size 直接缩放）。任何引入预测级合并的方案都必须自证其输出在 DOTA 评测语义下等价于单帧输出，Route A 因保持单帧输出语义而天然满足。

**RODM-NMS（IEEE 10701299）**：专门研究"高分辨率图像切片合并时旋转框 NMS 的失效情形"，归纳出两类 NMS 无法处理的情况（跨片截断框、旋转歧义框），需要专门的判定与处理流程。这篇工作本身就是"预测级合并对旋转框不可靠"的实证，直接支持 Route B 否决。

**LGI-DETR（2503.18785）**：基于 RT-DETR 的 UAV 小目标检测器，在编码器内部做局部空间增强（LSE）与全局信息注入（GII）的双向特征交互。它证明"编码器内局部-全局融合"是有效的，但其局部信息来自同一图像的特征层级，不涉及多视图坐标变换。Route A 借鉴其"局部增强残差注入全局特征"的形态，但把"局部"扩展为经过投影的异源视图。

**UHR-DETR（2604.21435，标记：论文仅、代码未发布）**：2026 年 4 月预印本，针对 STAR/SODA-A 等超高分遥感图，提出覆盖最大化稀疏编码器 + 全局-局部解耦解码器。论文声明代码"将发布"于 GitHub，截至本设计撰写，**无可用实现，无法验证其机制**；且其稀疏编码器与本仓库的稠密三级金字塔契约（3.1）不兼容。仅作"全局上下文 + 局部细节解耦"的思想参考，不作为路线依据。

---

## 6. 原始五步逐条裁定

| 步骤 | 原始表述 | 裁定 | 替代/精化 |
|---|---|---|---|
| 第一步：规范画布 | 保持宽高比的统一缩放 | **支持（形式化后保留）** | 必须同时满足相似性 + 零偏移 + 无填充（4.3 定理）；缺一即破坏后处理不变性 |
| 第二步：多视图 | 全局视图 + 局部裁剪视图；全局允许非均匀缩放 | **部分支持** | 全局 = 规范视图（相似性）；各向异性全局视图作为几何视图被反驳（4.1）；局部视图作为相似性渲染特征源成立 |
| 第三步：直接混合 | 特征直接混合进检测器 | **反驳（原样表述）** | 原始 token 拼接/朴素追加在几何上无效（3.2、5 中 Deformable DETR 分析）；替代为变换感知的归一化门控残差融合（8.5） |
| 第四步：规范训练 | 锚点/DN/matcher/loss 保持第一步坐标系 | **支持** | 这是保持全部训练契约不变的充要条件；由 C-归一化 ≡ O-归一化保证（4.3） |
| 第五步：后处理不变 | PostProcessor 不变 | **支持（附条件）** | 唯一前提是单一规范输出帧 + 零偏移无填充 C；满足后 PostProcessor 逐位不变 |

第二步中"各向异性全局视图"的否决依据：I4 期望"非均匀全图缩放可安全作为附加 OBB 特征视图"，但 4.1 证明各向异性剪切破坏 OBB 几何，且先前排查已观察到 affine refit 的 AP75 损失。若需要"被降采样的全局上下文"，正确做法是相似性降采样（仍是规范视图），而非各向异性拉伸。

第三步中"原始 token 拼接"的否决依据：解码器 memory 是行主序、层主序拼接，几何解释完全由 `spatial_shapes` 决定（`deim_decoder.py:1003-1013`）；把视图特征图直接 concat 进 memory 会让 `spatial_shapes` 无法表达视图坐标系，锚点、可变形采样、valid_mask 全部错位。

---

## 7. Route A / B / C 总对比

**Route A（特征级融合，推荐）**：规范画布 C 过编码器得稠密基线 G；各局部视图过同一编码器（共享权重）得视图特征，经相似性投影到规范三级网格，做归一化门控残差融合，得到仍为规则稠密金字塔的 F；解码器、匹配器、损失、后处理全部消费 F，语义与现状一致。

**Route B（逐视图解码 + 预测合并，否决）**：每个视图独立完成解码，得到各自预测，再合并（旋转 NMS）成最终输出。

**Route C（规范查询多视图解码器，研究后备）**：保持单一规范查询与解码器，但解码器内部每个查询经视图变换直接到各视图特征上采样（DETR3D 式），需要解码器内部重写并引入源覆盖掩码。

| 判据 | Route A | Route B | Route C |
|---|---|---|---|
| 解码器改动 | 无 | 无 | 大（内部重写） |
| 对规则网格/锚点/valid_mask 契约 | 完全保持 | 保持（各自视图内） | 破坏，需重写 |
| 匹配器语义 | 单一规范 matcher | 逐视图 matcher，语义分裂 | 单一规范 matcher |
| 重复预测 | 无（单输出帧） | 跨视图重复，需 NMS | 无 |
| 接缝（seam）风险 | 低（覆盖/taper 处理） | 高（截断框、漏检） | 低（源覆盖掩码） |
| 旋转 NMS 依赖 | 无 | 必须（且对 OBB 不稳定，见 [9]） | 无 |
| 端到端可微 | 完全可微 | 合并阶段不可微 | 完全可微 |
| PostProcessor 不变性 | 保持 | 破坏（多帧合并后无单一 orig_size 语义） | 保持 |
| 训练复杂度（GT/DN/loss 单一性） | 低（一切不变） | 高（每视图一套，需去重损失） | 中（解码器重写） |
| 理论上限 | 高（稠密多源特征） | 中（合并信息损失） | 最高（查询级最优采样） |
| 生产就绪度 | 高（增量组件） | 低（合并管线脆弱） | 低（研究原型） |

**排名：A > C >> B。**

- **A 胜出**：契约原生（解码器/匹配器/损失/后处理全不动），单一匹配器与 DN 语义完整保留，无预测重复、无接缝框、无旋转 NMS；BEVFormer 稠密网格融合与 LGI-DETR 编码器级局部-全局注入提供了外部可行性先例。
- **C 是明确的研究后备**：理论上最强（查询级变换感知采样，等价于把 DETR3D 的投影下沉到注意力内部），但要求解码器内部重写、引入源覆盖掩码（现有代码**没有**任意覆盖掩码概念），且需要重新验证锚点 top-k 与 valid_mask 语义，短期内无法生产化。
- **B 否决为主路线**：OBB 特定问题集中爆发：旋转 NMS 在跨视图合并时存在已知失效情形（[9]），单一匹配器语义被逐视图 matcher 破坏，接缝框难以处理；预测级重复对 OBB 尤其有害（重复框角度差异放大 IoU 敏感性）。B 的"预测级重复"与 A 的"特征级重复"性质不同：A 的特征级重复可微且被归一化和与 alpha 门控吸收，B 的预测级重复只能靠后处理消除。

---

## 8. Route A 决策完备设计

### 8.1 组件与接口

| 组件 | 职责 | 输入 | 输出 |
|---|---|---|---|
| 规范画布采样器 | O → C（相似性、零偏移、无填充） | 原图 O、规范形状 (H_c, W_c) | 画布 C（H_c×W_c×3） |
| 视图采样器 | 原图 → 相似性渲染的局部视图 | 原图 O、视图元数据 {rect, s_v} | 视图图块 V_v（H_v×W_v×3） |
| 共享编码器（现有 HybridEncoder） | C 与各视图各自编码 | C, V_1..V_K | G（三级 256ch）、F_v（三级 256ch） |
| 投影器（新） | 视图特征 → 规范网格 | F_v^l、视图元数据、规范网格 | P_v^l（B×256×h×w）、coverage_v^l |
| 融合器（新） | 归一化门控残差融合 | G、{P_v, coverage_v} | F（B×256×h×w×3），仍是规则稠密金字塔 |
| 解码器（现有） | 消费 F | F、spatial_shapes | 规范归一化输出 |
| 匹配器/DN/损失（现有） | 规范空间监督 | 输出、GT（C-归一化） | 损失 |
| PostProcessor（现有） | 像素恢复 | 输出、orig_size | 原图像素 OBB |

新增组件（投影器、融合器）是独立的、可注册的新组件（遵守 3.5 的 `workspace.py:182` 纪律，不向现有组件注入参数）。

### 8.2 数据流

**训练**：
1. O → C（相似性；先固定宽高比模式，见 8.7）。
2. 按视图元数据渲染局部视图 V_1..V_K（相似性）。
3. G = Encoder(C)；F_v = Encoder(V_v)（共享权重；视图数量 K 与元数据同批可变化）。
4. 对每个层级 l：P_v^l = Project(F_v^l, 元数据)；F_l = Fuse(G_l, {P_v^l})。
5. F 经 `_get_encoder_input`（现有路径 `deim_decoder.py:990-1013`）进入解码器。
6. GT 按 C-归一化（与当前流水线一致），DN/matcher/损失不变。

**推理**：
1. 同 1-4。
2. 解码器输出 C-归一化框 → PostProcessor × `[W,H,W,H,1]`（`postprocessor.py:60-67`）→ 原图像素 OBB → DOTA 评估（`obb_eval.py:268-271`，不变）。

### 8.3 最小视图元数据

每个局部视图 v 必须携带（全部由数据流水线在渲染时确定）：

- `active_v ∈ {0,1}`：本视图是否参与融合（视图缺帧/失败时为 0）。
- 齐次变换（3×3 相似性矩阵，按**像素坐标系**定义）：
  - `T_o_to_c`：原图像素 → 画布像素；其逆 `T_c_to_o = (T_o_to_c)⁻¹`；
  - `T_o_to_v`：原图像素 → 视图像素；其逆 `T_v_to_o = (T_o_to_v)⁻¹`；
  - `T_c_to_v = T_o_to_v · T_c_to_o`：画布像素 → 视图像素（由前两者导出，不单独存储）。
- 源边界与尺寸：原图像素中的源矩形 `(x0_v, y0_v, x1_v, y1_v)`（覆盖范围/合法性判定的边界，用于导出 coverage）；视图像素尺寸 `(H_v, W_v)`；画布像素尺寸 `(H_c, W_c)`。
- 覆盖率（按层级计算，不存储）：`coverage_v^l`。
- 门控初始 logit：0（见 8.5）。

**坐标映射约定**（投影器、8.4 与第 4 节证明共用）：C-归一化点 (u, v) 先乘画布尺寸转为画布像素，再经 `T_c_to_v` 到视图像素：

`p_v = T_c_to_v · [u·W_c, v·H_c, 1]ᵀ`

该式是唯一权威映射；投影器与 8.4 必须使用 `T_c_to_v` 齐次变换链，不允许以"视图源矩形坐标 ÷ 视图缩放"这类混合坐标系的简化表达式替代。

不需要的元数据：视图级 GT、视图级类别/目标列表。**没有逐视图 GT**。

### 8.4 三级网格 / 投影器 / 融合契约

**网格**：规范三级网格尺寸与编码器输出严格一致：`(H_c/8, W_c/8)`、`(H_c/16, W_c/16)`、`(H_c/32, W_c/32)`，与 `_generate_anchors` 的期望（`deim_decoder.py:1021-1022`，`eval_spatial_size / feat_strides`）一致。网格点为规范坐标 `(u, v) = ((j+0.5)/W_l, (i+0.5)/H_l)`。

**投影器**：对视图 v、层级 l，由 8.3 的映射构建采样网格（H_l × W_l × 2，视图像素坐标），对 F_v^l 做双线性采样得到 P_v^l，同时生成覆盖率图 coverage_v^l。覆盖判定在**视图像素坐标系**完成：采样点落在 `[0,W_v) × [0,H_v)` 内为 1，否则为 0；该范围等价于原图源矩形经 `T_o_to_v` 的像。边界半像素按双线性权重衰减。投影是**可微的**（grid_sample 及其梯度），视图变换以解析形式注入，不需要学习。

**融合器**：逐层级、逐像素执行 8.5 的公式，输出 F 保持规则稠密金字塔形状（B×256×h×w），随后进入现有 `_get_encoder_input`，spatial_shapes 不变。

### 8.5 归一化门控残差融合公式

对层级 l、视图 v，记 G_l 为全局基线特征，P_v^l 为投影后的视图特征。

1. **残差**：`R_v^l = P_v^l − G_l`（逐通道、逐像素）。
2. **有效权重**：`W_v^l = active_v · coverage_v^l · taper_v^l · σ(gate_v^l)`，其中：
   - `active_v` 为视图开关；
   - `coverage_v^l` 为投影覆盖率（[0,1]）；
   - `taper_v^l` 为**两单元 smoothstep 边缘过渡**：在覆盖区边界内 2 个网格单元内从 0 平滑升至 1。令 d 为到覆盖区边界的距离（单元数），`t = clamp(d/2, 0, 1)`，`taper = t²·(3 − 2t)`（smoothstep）。边界上 taper = 0，距边界 ≥ 2 单元处 taper = 1。作用：消除视图边缘的硬接缝，使融合权重在覆盖区边缘连续可微；
   - `σ(gate_v^l)` 为视图-层级门控的 sigmoid，gate 初始 logit 为 0。
3. **归一化和**：`R̄_l = Σ_v W_v^l ⊙ R_v^l / max(1, Σ_v W_v^l)`。分母取 `max(1, ·)` 保证无视图激活（ΣW = 0）时 R̄_l = 0，数值稳定；有重叠视图时按权重归一，避免重复视图叠加放大。
4. **有界残差注入**：`F_l = G_l + α_l · tanh(R̄_l)`，其中 `α_l = tanh(β_l)`，β_l 为可学习标量，**初值 0**。

**性质**：

- **恒等初始化**：β_l = 0 → α_l = 0 → F_l = G_l。训练第 0 步，Route A 与当前单视图基线逐位等价，任何融合缺陷都不可能污染初始状态；`tanh` 把残差幅度限制在 (−1, 1) 内（按层缩放语义），防止局部视图特征爆量淹没全局基线。
- **有界 alpha**：α_l ∈ (−1, 1)，融合强度被参数化约束，不会出现无界学习率下的发散。
- **可消融**：β 冻结为 0 即回到基线；去掉 taper 或 gate 各有独立的消融意义（见第 11 节）。
- **不引入原始 token 拼接**：所有跨视图信息交换发生在稠密网格上，解码器永远只看到规则金字塔 F。

### 8.6 GT / 可见性 / DN / matcher / 损失 / 输出语义

- **GT**：每对象一个 GT，位于 C-归一化坐标（cx, cy, w, h, θ 物理角 `[0, π)`）。由于 C-归一化 ≡ O-归一化，GT 归一化与当前流水线逐位一致。**没有逐视图 GT、没有按视图复制的 GT**。
- **可见性**：对象是否被某个视图覆盖只影响该视图的 `coverage_v^l`（特征覆盖），**不影响 GT 归属**。部分可见对象仍是规范坐标系中的完整目标；其跨视图信息由融合权重自然综合。
- **DN**：去噪查询在规范归一化空间生成；角度通道按 decoder 配置进入 shifted 编码（默认，`denoising.py:116-118`）或 proportional 编码（`denoising.py:120`），与现状一致。
- **matcher**：匈牙利匹配在规范归一化 cxcywhθ 空间进行，`angle_factor = π`（`matcher.py:41`），probiou/chamfer 成本不变。
- **损失**：非周期 L1 角度路径 `physical_rad_to_norm`（`deim_criterion.py:382, 389`）、probiou/kld、FGL/DDF 分布回归，全部在规范空间，不变。
- **输出**：解码器输出 C-归一化框，PostProcessor 经 `[W,H,W,H,1]` 因子恢复原图像素，θ 不变。**输出帧唯一**，无预测级合并。

### 8.7 固定宽高比优先契约与动态宽高比无结构缓存扩展

**模式一（固定宽高比，先行契约）**：规范画布形状固定（如 640×640，与当前 `eval_spatial_size` 机制一致），适用于输入已具备固定宽高比的场景（DOTA 切块为方形瓦片）。画布形状恒定 → 三级网格尺寸恒定，**形状相关的结构张量**可一次性生成并在场景间复用：位置编码（仅依赖形状）、解码器锚点与 valid_mask buffer（`deim_decoder.py:882-886` 的注册机制即属此类）。注意区分：**图像特征依赖具体场景内容，不能仅因形状相同而在不同场景间复用**；跨场景缓存仅限形状相关的结构张量，不包括任何图像内容特征。此模式下 C 仍是"全图在统一缩放 + 零偏移下的完整表示"（对固定宽高比输入成立），4.3 定理的充分性不变。

**模式二（动态宽高比，无结构缓存扩展）**：输入为任意宽高比的全图时，规范画布形状 = 原图宽高比（统一缩放 + 零偏移，无 letterbox）。画布形状随场景变化 → 网格尺寸变化 → 位置编码、锚点与 valid_mask 必须按实际运行时形状生成，无跨场景结构缓存可用。当前 `eval_spatial_size` 路径会注册固定形状的 anchor/valid_mask buffer，因此该扩展必须关闭固定 `eval_spatial_size` 缓存并走运行时 `_generate_anchors` 路径；它不是在同一已注册 buffer 上原地复用任意形状。几何契约与 PostProcessor 不变性在两种模式下完全一致；差异只在形状稳定性与结构张量的可复用性。

**明确排除**：letterbox/填充补边在任何模式下都不是合法路径（4.3 必要性）。

### 8.8 现有增强、尺寸与部署交互契约

当前 OBB 训练链包含 Mosaic、Mixup、CopyBlend、OBBZoomOut、OBBIoUCrop、OBBResize 与 collate 多尺度；它们不能在未定义场景所有权的情况下直接套到多视图路径上。生产首版采用以下**单一裁决**：

1. **先形成单一场景，再生成所有视图**。全局 C、局部视图、唯一 GT 集必须来自同一个场景坐标系。局部视图规划发生在场景级几何处理之后；任何允许的随机变换必须同步作用于场景图像、完整 GT 和后续视图变换元数据。
2. **生产首版禁用复合多场景增强**：Mosaic、Mixup、CopyBlend 在多视图路径中关闭。原因不是这些增强永远不可用，而是它们将多个源图像组合成一个训练样本，破坏 8.3 中单一 O、单一 `T_o_to_c` 与单一场景 GT 所有权。若未来恢复，必须把增强后的合成画布重新定义为新的场景 O′，并从 O′ 统一生成 C 与全部局部视图；不得保留各源图自己的视图或 GT 所有权。
3. **生产首版禁用改变可见区域的随机几何增强**：OBBZoomOut、OBBIoUCrop 在多视图路径中关闭。水平翻转等保持完整场景的一一映射仅可在 C 与局部视图生成之前统一执行，并同步更新 GT 与齐次变换。PhotometricDistort 等仅改变像素值的增强可保留，但同一场景各视图应使用一致的基础色彩变换，避免无意引入视图身份捷径。
4. **替换现有各向异性 OBBResize**：`engine/data/transforms/obb_transforms.py:81-106` 的逐轴 `sx/sy` resize + `affine_obb` refit 不进入多视图路径；由 8.1 的规范画布采样器执行统一缩放。固定宽高比首版不再使用 `BatchImageCollateFunction` 的图像-only 随机方形多尺度 resize（`engine/data/dataloader.py:513-520`），避免图像形状变化却不重建视图变换与结构缓存。
5. **尺寸必须满足层级整除性**：`H_c,W_c,H_v,W_v` 均必须是 32 的整数倍。DINOv3/STA 使用 patch-16 网格并由 stride 8/16/32 金字塔消费；显式 32 倍数约束避免 floor 除法造成投影网格、encoder 输出和 anchor shapes 不一致。
6. **导出首版为静态视图配置**：ONNX/部署首版固定 `[H_c,W_c]`、固定局部视图 `[H_v,W_v]`、固定最大视图数 K_max，并以 `active_v` 关闭未使用槽位。当前导出脚本只把 batch 设为动态轴，因此动态宽高比与动态视图列表属于后续扩展，不是首版承诺。
7. **旧检查点加载不是现状即兼容**：当前 resume 路径默认严格加载模型 state_dict；新增投影器/融合器后，旧检查点会出现预期 missing keys。实施计划必须显式选择迁移入口（例如 tuning/non-strict 路径或受控的 expected-missing-key 策略），并以 β=0 初始化新残差尺度。本文档只规定目标语义，不声称现有 resume 无改动即可成功。

---

## 9. 变更 vs 不变表

| 组件 | 变更 | 说明 |
|---|---|---|
| 数据流水线（OBB 路径） | **变更** | 相似性规范画布采样替代各向异性 OBBResize；首版关闭 Mosaic/Mixup/CopyBlend/ZoomOut/IoUCrop 与 collate 随机多尺度；视图采样器与元数据生成 |
| HybridEncoder | **不变**（复用） | C 与各视图共享同一编码器权重；`fuse_op='sum'` 等内部结构不动 |
| 投影器 | **新增** | 视图特征 → 规范网格的双线性投影 + 覆盖率 |
| 融合器 | **新增** | 归一化门控残差（8.5），含 taper 与 gate |
| 解码器（`deim_decoder.py`） | **不变** | num_levels=3、活跃配置 num_points=[3,6,3]、`_get_encoder_input`、`_generate_anchors`、top-k、valid_mask |
| 可变形注意力（`dfine_decoder.py`） | **不变** | reference_points [0,1] 归一化、offset_normalizer、旋转采样 |
| DN / matcher / 损失 | **不变** | 规范空间语义原样（`denoising.py:116-120`、`matcher.py:41`、`deim_criterion.py:382,389`） |
| PostProcessor | **不变** | `postprocessor.py:60-67` 的 `[W,H,W,H,1]` 因子、θ 不变 |
| orig_size 契约 | **不变** | `[w,h]` 序（`coco_dataset.py:174` 等） |
| 评估路径 | **不变** | `obb_eval.py:268-271` |
| HBB 路径 | **不变** | 多视图融合仅在 OBB/多视图流水线内生效 |
| 检查点兼容性 | **部分兼容（设计说明，非兼容代码）** | 主干/编码器/解码器的既有参数键保持兼容；state_dict **新增**投影器与融合器参数键（含 β、gate）。加载旧检查点时需按"预期缺失键"方式为新模块初始化（β=0 恒等），即从旧检查点续训在**参数语义**上等价，但**不宣称检查点格式/键集合完全一致**。本文档只作设计说明，不实现任何兼容代码 |
| ONNX / 应用层 | **首版变更** | 固定 canonical/view shapes 与 K_max；导出图新增投影/融合及 view metadata/active 输入；动态宽高比、动态视图列表后置 |

---

## 10. 风险

1. **计算成本（必须正视，不得声称 Route A 廉价）**：编码器是主要开销，Route A 需要 K+1 次编码（全局 + K 个视图），编码器 FLOPs 与显存随视图数量与视图面积线性增长；投影器与融合器增加少量开销。典型配置（1 全局 + 2 局部视图）下总成本约为基线的 1.5 至 3 倍，视图越大越接近上限。任何把 Route A 描述为"几乎免费"的说法都是错误的。缓解：共享编码器权重、**批量激活视图**（同批打包激活视图以摊薄调度开销）、**限制视图尺寸与数量**（显存/FLOPs 预算封顶），以及**后续计算共享研究**（如多视图共享中间特征，属明确研究项而非本期承诺）。
2. **特征域失配**：Encoder(C) 与 Encoder(V_v) 的统计分布不同（尺度、纹理密度），融合可能引入不一致特征。缓解：恒等初始化 + 残差形式 + 有界 alpha + 门控，让网络自行决定注入强度；β 从 0 逐步学习。
3. **BN 统计污染**：多尺度输入共享 BatchNorm 统计可能互相干扰。缓解：按视图统计追踪、必要时 SyncBN 或独立归一化分支（实现期决策）。
4. **边缘伪影**：视图边缘的采样不连续可能产生伪影。缓解：两单元 smoothstep taper + coverage 权重（8.5）。
5. **重叠视图的重复信息**：多视图重叠区域的残差叠加。缓解：归一化和分母 `max(1, ΣW)` 与 tanh 有界化。
6. **动态宽高比模式的工程成本**：形状变化导致位置编码、锚点、valid_mask 无法跨场景复用，需按场景重新生成，训练吞吐下降。缓解：先固定宽高比模式上线，动态模式作为显式扩展。
7. **Route C 尚未验证**：作为研究后备，其解码器重写与源覆盖掩码的成本未被任何实验证实，不进入生产路径。
8. **文献时效**：UHR-DETR 等 2026 年新作代码未发布，机制不可复现；本设计的外部依据以已发布可复现代码（Deformable DETR、RT-DETR、DETR3D、BEVFormer、SAHI）为主。
9. **增强收益回退**：首版关闭现有多场景/随机几何增强，可能降低数据多样性。该代价是为了先锁定坐标与所有权正确性；只有在 Route A 基线成立后，才按 8.8 的“合成场景 O′”规则逐项恢复增强并做独立消融。

---

## 11. 研究性验证与消融矩阵

以下矩阵为**研究专用**验证计划（本文档不实施任何训练；执行时按研究流程单独立项）：

| 实验 | 目的 | 预期结论 |
|---|---|---|
| A0 | 当前单视图基线（控制组） | 基准 AP/AP75/小目标 AP |
| A1 | Route A + 1 局部视图，β/gate 可学习 | 融合相对基线的最初增益或持平（恒等初始化保证不劣化起点） |
| A2 | Route A + 2 局部视图 | 视图数增益的边际效应 |
| A3 | Route A，β 冻结 0 | 隔离"恒等初始化 + 残差形式"之外的融合效应（应 ≈ A0） |
| A4 | Route A 去掉 taper | taper 对视图边缘 AP 的影响 |
| A5 | Route A 去掉 gate（W = active·coverage·taper） | 门控对融合稳健性的影响 |
| A6 | Route A，α 直通/固定为 1（即以足够大的**有限** β 近似全强度注入；不使用无穷参数） | 全强度注入的风险对照 |
| A7 | 动态宽高比模式 vs 固定宽高比模式 | 形状变化对收敛与精度的代价 |
| A8 | 视图缩放选择（1× vs 2× 视图像素密度） | 局部视图分辨率的最佳折中 |
| A9 | 接缝区/覆盖区专项 AP | 覆盖与 taper 的定量收益（按覆盖率分层统计） |
| A10 | 逐项恢复 photometric / flip / 合成场景增强 | 衡量关闭现有增强的精度代价，并验证 O′ 所有权规则 |
| C1（研究后备） | Route C 最小原型（解码器内部变换感知采样 + 源覆盖掩码） | 理论上限的实证；若显著优于 A 再考虑生产化 |
| B1（否决确认，可选） | Route B 对照（逐视图解码 + RODM-NMS 类合并） | 记录接缝/重复失败率，作为 A 的对照证据 |

度量：DOTA AP / AP50 / AP75、小目标 AP、按覆盖率的区域 AP、接缝区漏检率、重复框率、显存与延迟审计（含 K+1 编码开销）。

---

## 12. 结论

1. **几何**：规范画布必须满足"相似性 + 零偏移 + 无填充"。这是 PostProcessor 保持逐位不变的充要条件（4.3），也是 GT/DN/matcher/loss 全部保持现状的前提。
2. **路线**：Route A（解码器前特征级融合）为推荐路线：契约原生、单一匹配器语义、无预测级重复、无旋转 NMS 依赖、端到端可微，并有 BEVFormer 稠密网格融合与 LGI-DETR 编码器级局部-全局交互的外部先例。Route C 为研究后备（理论最强、工程最重）。Route B 否决：旋转 NMS 跨视图合并的不稳定性（RODM-NMS 文献）与接缝/重复问题使其不适合 OBB 生产。
3. **融合**：归一化门控残差（8.5）以恒等初始化 + 有界 alpha + smoothstep taper 保证安全接入现有流水线，不引入原始 token 拼接，不引入稀疏唯一融合。
4. **契约**：编码器三级稠密金字塔、解码器规则网格假设、训练与推理语义、PostProcessor 与 orig_size 契约全部保持不变；变更集中在数据流水线 + 两个新增组件。
5. **代价**：Route A 不廉价，编码成本随视图数量线性增长，风险 10.1 必须纳入预算。
6. **实施纪律**：先固定宽高比模式（结构张量可复用），动态宽高比作为无结构缓存扩展；研究验证矩阵（第 11 节）先行，生产实施另行立项。

---

## 13. 编号来源

1. Zhu X, et al. Deformable DETR: Deformable Transformers for End-to-End Object Detection. ICCV 2021. https://arxiv.org/abs/2010.04159
2. Lyu C, et al. RT-DETR: DETRs Beat YOLOs on Real-time Object Detection. CVPR 2024. https://arxiv.org/abs/2304.08069
3. Singh B, Davis L S. SNIP: An Analysis of Scale Invariance in Object Detection. CVPR 2018. https://arxiv.org/abs/1711.07289
4. Singh B, et al. SNIPER: Efficient Multi-Scale Training. ECCV 2018. https://arxiv.org/abs/1805.09300
5. Wang Y, et al. DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries. CoRL 2022. https://arxiv.org/abs/2110.06922
6. Li Z, et al. BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers. ECCV 2022. https://arxiv.org/abs/2203.17270
7. Akyon F C, et al. Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection. ICIP 2022. https://arxiv.org/abs/2202.06934
8. Xia G S, et al. DOTA: A Large-Scale Dataset for Object Detection in Aerial Images. CVPR 2018. https://arxiv.org/abs/1711.10398
9. Non-Maximum Suppression for Rotated Object Detection During Merging Slices of High-Resolution Images. IEEE, 2024. https://ieeexplore.ieee.org/document/10701299
10. Chen Z. LGI-DETR: Local-Global Interaction for UAV Object Detection. arXiv:2503.18785, 2025. https://arxiv.org/abs/2503.18785
11. Li J, et al. UHR-DETR: Efficient End-to-End Small Object Detection for Ultra-High-Resolution Remote Sensing Imagery. arXiv:2604.21435, 2026（**预印本；代码未发布**）. https://arxiv.org/abs/2604.21435
12. Wang W, et al. DEIM: DETR with Improved Matching for Fast Convergence. arXiv:2412.04234, 2024. https://arxiv.org/abs/2412.04234
13. Peng Y, et al. D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement. ICLR 2025. https://arxiv.org/abs/2410.13842
14. Yang X, et al. R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object. AAAI 2021. https://arxiv.org/abs/1908.05612

---

## 14. 方法论与局限

**方法论**：本设计基于一轮定向 librarian 检索（`.omo/ulw-research/20260813-145641/`）以及后续对 Deformable DETR、RT-DETR、SNIP/SNIPER、UHR-DETR、DETR3D/BEVFormer、SAHI/DOTA 与全局-局部检测器的补充核验；本地证据来自对 `engine/deim/`、数据增强、导出与 checkpoint 路径的逐行审计。研究日志中的 claim graph 仍保留 partial/pending 项，因此外部文献部分应理解为**覆盖关键先例的定向综述**，不是系统综述或穷尽性 meta-analysis。坐标空间结论（第 4 节）为解析证明，不依赖实验。

**局限**：

- 本文档为设计规范，**不包含任何训练实验**；第 11 节矩阵的全部 AP 数字均为"待研究"状态，本设计未断言任何精度增益。
- UHR-DETR（[11]）无可用代码，其机制不可复现，仅作思想参考。
- 本地契约行号随代码演进可能漂移，实施时须重新核对。
- Route C 的理论优势未被实证；Route B 的否决依据（旋转 NMS 合并失效）来自文献（[9]）与本仓库语义分析，未做本地对照实验。
- 动态宽高比模式（8.7 模式二）的收敛行为未经验证，属明确的后续研究项。
- 首版增强裁决（8.8）以正确性优先，会改变当前训练分布；恢复增强必须单独验证，本文档不预设其 AP 影响。
