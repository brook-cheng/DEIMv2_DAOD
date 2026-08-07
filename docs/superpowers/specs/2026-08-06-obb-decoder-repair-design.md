# DEIMv2 OBB decoder 两阶段修复设计

日期：2026-08-06

状态：阶段 1 与阶段 2 设计均已批准；职责调整为 AI 负责编写和运行测试、用户只实现生产代码，等待按三闸门流程推进。

关联文档：

- 角度分层契约：`docs/superpowers/specs/2026-08-05-obb-angle-contract-simplification-design.md`
- decoder 私有角度编码：`docs/superpowers/specs/2026-08-05-obb-decoder-shifted-angle-design.md`
- ADR 几何与损失：`docs/superpowers/specs/2026-08-04-obb-adr-loss-design.md`
- 最新审计结论：当前工作区中 `deim_decoder.py:1115` 与 `:426-430` 为阻断回归；rep2 还存在既有表示边界、denoising 与 anchor 语义错位。

本文优先级高于关联文档中与 rep2 6D 表示冲突或已经过时的描述。特别是，本文明确规定 rep2 decoder 内部 6D canonical representation 为 `cxcywh+offset`，而几何层继续使用 `xyxy+offset`。

## 1. 背景与问题分类

当前问题分为两类，必须分阶段处理：

1. **本轮角度契约修改引入的阻断回归**：
   - encoder auxiliary 输出使用 `[..., 4]` 丢失角度维，阻断 rep0/1/3 训练 forward；
   - rep2 活路径的 `torch.concat` 缺少 tensor list，同时使用 `[..., 4]` 丢失角度维，阻断 rep2 train/eval forward。
2. **长期存在的 rep2 ADR 表示错位**：
   - decoder 前四维按 `cxcywh` 生产，部分几何消费者却按 `xyxy` 解释；
   - denoising 将 logits 和 `theta_norm` 直接交给要求激活坐标与 `theta_phys_rad` 的几何函数；
   - rep2 anchors 的后两维来自网格位置，不是满足几何约束的顶点偏移；
   - encoder auxiliary、attention 与 layer-0 decode 没有共享同一套显式转换边界。

若一次性修复两类问题，运行时回归与表示迁移会互相掩盖，难以判断失败来源。因此采用两阶段方案：阶段 1 先恢复基线可运行性；阶段 2 再统一 rep2 ADR 语义。

## 2. 总体目标与非目标

### 2.1 目标

1. 阶段 1 以最小改动恢复 rep0、rep1、rep2、rep3 的 train/eval forward，并用回归测试锁定两个切片/concat 缺陷。
2. 阶段 2 将 rep2 decoder 内部 6D 表示统一为 `(cx,cy,w,h,epsilon,eta)`。
3. 保持几何函数的既有公开契约：
   - `oriented_box_to_external_rect`：`cxcywh+theta_phys_rad -> xyxy+offset`；
   - `external_rect_to_oriented_box`：`xyxy+offset -> cxcywh+theta_phys_rad`。
4. 所有 decoder/geometry 边界使用显式 `cxcywh <-> xyxy` 转换，禁止依赖变量名或上下文猜测格式。
5. rep2 denoising、anchors、attention、layer-0 decode、encoder auxiliary 输出共享同一套 canonical representation。
6. 每个修复单元可独立审核、测试和回滚，上一单元未通过不得进入下一单元。

### 2.2 非目标

1. 本轮不统一或修正 YAML 中遗留的 `angle_rep: True`、`angle_step: 10` 等配置；用户已明确要求将配置清理留到后续独立任务。
2. 不改变公开 OBB 输出契约：始终为 `(cx,cy,w,h,theta_phys_rad)`，`theta_phys_rad in [0,pi)`。
3. 不改变 `obb_geometry.py` 的函数签名和 `xyxy+offset` 契约。
4. 不改变 rep0/1/3 的表示设计、损失设计或 shifted/proportional 编码选择。
5. 不重构无关模块，不做 checkpoint 格式迁移，不改变 head 输出维度。
6. 不通过 clamp、异常吞掉或兼容分支掩盖错误表示；转换必须在正确边界完成。

### 2.3 职责分工与三闸门工作流

本轮修复按三个顺序闸门执行，AI 与用户职责互不重叠：

1. **AI Test Gate**：AI 负责所有 pytest 的编写、修正和运行。对于生产缺陷，先提供能因目标缺陷失败的 RED 证据；对于测试契约自身的修正，先记录旧测试的准确失败，再验证修正后的测试与当前生产契约一致。
2. **User Implementation Gate**：用户只修改当前审核单元批准的生产代码，提供该单元的生产 diff 和任何契约假设；不得修改或弱化 AI 提供的测试。
3. **AI Green/Review Gate**：AI 检查 tensor layout、激活域和角度域，运行定向测试与扩大回归，并返回明确 PASS 或包含文件、行号和契约违例的退回结论。

闸门顺序固定为 `AI Test Gate -> User Implementation Gate -> AI Green/Review Gate`。上一闸门未通过不得进入下一闸门。若某个单元仅修正测试契约而不需要生产代码，例如阶段 1 的 MSDA 等价测试，则跳过 User Implementation Gate，由 AI 完成测试修正和验证。

## 3. 术语与硬契约

### 3.1 角度域

| 标识符 | 数值域 | 使用位置 |
| --- | --- | --- |
| `theta_phys_rad` | 弧度 `[0, pi)` | geometry、criterion、matcher、postprocessor、公开输出 |
| `theta_norm` | 无量纲 `[0,1)` | decoder 内部显式角度 reference |
| `theta_logit` | 无界 | `inverse_sigmoid(theta_norm)` 后的 decoder reference |

阶段 2 不改变现有角度分层契约，只修正 rep2 在几何边界使用错误域的问题。

### 3.2 空间框格式

| 名称 | 张量布局 | 所属层 |
| --- | --- | --- |
| `obb_5d` | `(cx,cy,w,h,theta_phys_rad)` | 公开/几何层 |
| `rep2_6d_act` | `(cx_ext,cy_ext,w_ext,h_ext,epsilon,eta)` | decoder 内部激活态；前四维是 external rectangle 的 `cxcywh`，不是 OBB 自身的宽高 |
| `rep2_6d_unact` | `logit(rep2_6d_act)` | decoder reference logits |
| `external_6d_act` | `(x1,y1,x2,y2,epsilon,eta)` | 几何边界临时值 |

硬性规则：

1. rep2 decoder 内部存储、head residual、anchors 与 query reference 的前四维一律是 external rectangle 的 `cxcywh`；rep0/1/3 仍是 OBB 的 `cxcywh`。
2. `external_rect_to_oriented_box` 的前四维一律是 `xyxy`；调用前必须能在同一代码块中看到显式 `box_cxcywh_to_xyxy`，除非输入变量已由 `oriented_box_to_external_rect` 直接产生并命名为 `*_xyxy`。
3. `oriented_box_to_external_rect` 只接受激活态 `obb_5d`，不得传入 logits 或 `theta_norm`。
4. 几何层返回的 `xyxy` 不得直接写回 decoder 6D reference；必须先转回 `cxcywh`。
5. `inverse_sigmoid` 只能用于已经处于合法 `(0,1)` 激活域的 decoder reference。

## 4. 阶段 1：恢复运行基线

### 4.1 修改范围

阶段 1 只允许修改两个生产回归及其直接测试：

1. `engine/deim/deim_decoder.py` encoder auxiliary 角度切片：
   - 从 `enc_topk_bboxes[..., 4]` 恢复为 `enc_topk_bboxes[..., 4:]`。
2. 同文件 rep2 非 `use_angle_first` 活路径：
   - `torch.concat` 第一参数改为 tensor list；
   - 角度切片使用 `pre_bboxes[..., 4:]`；
   - 结果布局保持 5D `(cx,cy,w,h,theta_norm)`。
3. 更新 MSDA 旧等价测试：当 6D ADR 经几何函数产生 `theta_phys_rad` 后，构造等价 5D decoder reference 时必须先执行 `physical_rad_to_norm`，不能把物理角直接交给会内部乘 `pi` 的 5D attention 分支。该项由 AI Test Gate 直接编写和验证，不要求用户修改生产代码。
4. 新增或补强 train/eval smoke tests，覆盖 rep0/1/2/3。

### 4.2 禁止事项

阶段 1 不得：

- 修改 rep2 anchors；
- 修改 denoising 转换；
- 添加 `cxcywh <-> xyxy` 转换；
- 修改 PostProcessor；
- 修改配置文件；
- 顺手重构 decoder 分支或删除 `use_angle_first` 死代码。

### 4.3 阶段 1 验收标准

1. `test_obb_angle_contract.py`、`test_obb_adr_geometry.py`、`test_obb_domain_audit.py` 全部通过。
2. `test_deimv2_obb_smoke.py` 不再出现 `:1112` 维度错误或 `:426` concat `TypeError`。
3. rep0/1/3 training forward 返回 5D `pred_boxes`，值有限，角度为物理弧度 `[0,pi)`。
4. rep2 `use_angle_first=False` 的 train/eval forward 均能运行；`use_angle_first=True` 仍按现有构造函数约束拒绝，不在阶段 1 改变。
5. 任何新增失败必须证明与阶段 1 修改无关；不得以弱化测试通过验收。

## 5. 阶段 2：统一 rep2 `cxcywh+offset`

### 5.1 为什么不选择内部 `xyxy+offset`

现有 decoder 主干、anchor 空间位置、bbox head residual 和其他 angle representation 均以前四维 `cxcywh` 为基础。将 rep2 改成内部 `xyxy+offset` 会要求重新定义：

- anchor 前四维；
- encoder/decoder bbox head residual 语义；
- query reference 更新；
- attention 中心与尺寸提取；
- 既有 checkpoint 参数的数值语义。

选择 `cxcywh+offset` 可保持 tensor 维度、head 参数形状及公共 bbox residual 结构，只在 geometry 边界做格式转换，变更面更小且可分单元验证。

### 5.2 几何边界辅助

阶段 2 的第一个实现单元在 `engine/deim/obb_geometry.py` 新增两个集中、可测试的组合 helper；不得在调用点复制几何公式：

- `external_cxcywh_to_oriented_box(external_cxcywh, vertex_offsets) -> obb_5d`；
- `oriented_box_to_external_cxcywh(obbs) -> tuple[external_cxcywh, vertex_offsets]`。

两个 helper 只组合既有 `box_cxcywh_to_xyxy` / `box_xyxy_to_cxcywh` 与已有 ADR 几何函数，不改变已有函数签名。

正向 decode：

```text
rep2_6d_act: external-rect cxcywh + epsilon,eta
    -> box_cxcywh_to_xyxy(external-rect cxcywh)
external_6d_act: xyxy + epsilon,eta
    -> external_rect_to_oriented_box
obb_5d: cxcywh + theta_phys_rad
```

反向 encode：

```text
obb_5d: cxcywh + theta_phys_rad
    -> oriented_box_to_external_rect
external_6d_act: xyxy + epsilon,eta
    -> box_xyxy_to_cxcywh(xyxy)
rep2_6d_act: external-rect cxcywh + epsilon,eta
```

辅助逻辑必须保留最后一维，例如角度一律使用 `[..., 4:]` 或 `[..., 4:5]`。

### 5.3 decoder layer-0 与 attention

需要修正三类消费者：

1. rep2 layer-0 `external_rect_to_oriented_box` 调用：先将前四维 `cxcywh` 转成 `xyxy`。
2. `MSDeformableAttention` 的 6D reference：先按同一 helper 将 `cxcywh+offset` 解码为 `obb_5d`，再直接使用返回的 `theta_phys_rad` 计算旋转采样位置。
3. encoder auxiliary decode：将 encoder 6D 激活输出按同一 helper 转为公开 `obb_5d` 后再交给 criterion。

这些调用不得各自发明转换顺序。测试必须证明同一输入经三个边界得到相同 OBB。

### 5.4 rep2 denoising

通用 `denoising.py` 继续生产 5D OBB logits，不感知 `angle_rep`。rep2 私有转换固定在 `DEIMTransformer._get_decoder_input` 边界。

正确数据流：

```text
denoising_bbox_unact
    -> sigmoid
cxcywh + theta_norm
    -> norm_to_physical_rad(theta_norm)
cxcywh + theta_phys_rad
    -> oriented_box_to_external_rect
xyxy + epsilon,eta
    -> box_xyxy_to_cxcywh
external-rect cxcywh + epsilon,eta
    -> inverse_sigmoid
rep2_6d_unact
```

关键约束：

- 不得把 `denoising_bbox_unact` 直接传给 geometry；
- 不得把 `theta_norm` 或其 logit 当成物理弧度；
- 转回 logits 前必须验证六个激活值位于合法 `(0,1)` 域；
- 不新增训练路径 clamp 来掩盖非法 offset，除非 offset 定义本身要求且测试证明梯度语义正确。

### 5.5 rep2 anchors

现有 `grid_offset_wh` 是特征图网格位置，不是 `(epsilon,eta)` 顶点偏移。阶段 2 必须从一个明确、合法的初始 OBB 生成 external rectangle 与 offsets，而不是直接复用 grid 坐标。

推荐数据流：

```text
initial OBB cxcywh + theta_phys_rad = pi/4
    -> oriented_box_to_external_rect
xyxy + epsilon,eta
    -> box_xyxy_to_cxcywh
external-rect cxcywh + epsilon,eta
    -> inverse_sigmoid
anchor logits
```

初始角固定为 `theta_phys_rad = pi/4`，与现有 rep0/1/3 默认归一角 `0.25` 的物理方向一致。本轮不新增 anchor 角度配置旋钮。

anchors 必须满足：

- 对通过 `valid_mask` 的 anchors，`eps < cx_ext,cy_ext,w_ext,h_ext,epsilon,eta < 1-eps`（进入 logit 前）；
- external rectangle 宽高为正；
- `0 <= epsilon <= external_width`；
- `0 <= eta <= external_height`；
- anchor 经 rep2 decode 后得到有限、非退化的 `obb_5d`。

### 5.6 `box_noise_scale=0`

`input_query_bbox_unact = inverse_sigmoid(input_query_bbox)` 必须移出 `if box_noise_scale > 0` 条件：

- scale > 0：先应用空间噪声，再编码；
- scale == 0：直接编码原始框。

该修复与 rep2 几何语义无依赖，应作为独立小单元提交和测试。

### 5.7 PostProcessor 非 focal 分支

该修复放在阶段 2 末尾，并与 rep2 几何改动隔离：

1. `F.softmax(logits, dim=-1)` 显式指定类别维；
2. 选择框时使用已缩放的 `bbox_pred`，而不是原始归一化 `boxes`。

当前 OBB 配置均走 focal 路径，因此此项不是 decoder 修复的阻断条件，但属于已确认的生产缺陷。

## 6. 阶段 2 实现与审核单元

阶段 2 拆为六个顺序单元：

1. **边界 helper 与 round-trip tests**：锁定 `cxcywh+offset <-> obb_5d`。
2. **decoder/MSDA/encoder aux 边界**：三个消费者统一调用同一转换路径。
3. **rep2 denoising**：修复 logit、激活、角度与空间格式顺序。
4. **rep2 anchors**：从合法初始 OBB 生成 offsets，并验证几何不变量。
5. **共享 denoising 零噪声缺陷**：独立修复 `box_noise_scale=0`。
6. **PostProcessor 非 focal**：独立修复归一化维与框尺度。

每个单元的三闸门协作流程：

1. AI Test Gate：AI 编写或修正当前单元的 pytest，并运行确认 RED 失败原因准确。
2. User Implementation Gate：用户只实现当前单元的生产代码，不混入下一单元或无关重构，并提供生产 diff 与契约假设。
3. AI Green/Review Gate：AI 检查格式和数值域契约，运行定向测试与扩大回归，返回 PASS 或退回结论。
4. 当前单元通过后才开始下一单元。

## 7. 测试策略

### 7.1 阶段 1 测试

- 角度转换单元测试；
- ADR 几何 round-trip；
- domain audit；
- rep0/1/2/3 train/eval smoke；
- MSDA 5D norm 与 6D physical 等价输入测试。

### 7.2 阶段 2 单元测试

1. `cxcywh+offset -> OBB -> cxcywh+offset` round-trip，在非退化输入上满足容差。
2. helper 与原始几何函数组合结果一致。
3. 明确检测将 cxcywh 误传为 xyxy 的负例，确保测试会失败而非仅检查 finite。
4. denoising rep2 转换与直接从 GT OBB 生成的期望 6D reference 一致。
5. anchors 全量满足 offset 几何不变量。
6. `box_noise_scale=0` 与无噪声输入精确一致，且不抛异常。
7. PostProcessor 非 focal scores 沿类别维和为 1，boxes 为像素尺度。

### 7.3 阶段 2 集成矩阵

| 表示 | 模式 | denoising | 期望 |
| --- | --- | --- | --- |
| rep0 | train/eval | 0 / >0 | 无回归 |
| rep1 | train/eval | 0 / >0 | 无回归 |
| rep2 | train/eval | 0 / >0 | 运行成功、5D 公开输出、角度 `[0,pi)` |
| rep3 | train/eval | 0 / >0 | 无回归 |

rep2 额外检查：

- layer-0 6D reference 与后续 5D angle reference 的切换符合契约；
- MSDA sampling location 对等价 OBB 输入一致；
- encoder aux、pre outputs、dn outputs 均为公开 5D OBB；
- 所有输出有限，宽高为正，角度属于 `[0,pi)`。

## 8. 错误处理与禁止的修复方式

1. 不使用 `as any`、忽略类型、异常捕获或测试跳过隐藏失败。
2. 不通过 `reshape`/`unsqueeze` 机械消除错误而不确认通道语义。
3. 不在 geometry 函数内部自动猜测输入是 cxcywh 还是 xyxy。
4. 不为当前未发布的错误表示增加兼容分支。
5. 不将 `clamp_offsets=True` 作为默认训练修复；当前生产调用未启用该参数，且无梯度语义批准。
6. 不删除或弱化 smoke/contract tests。

## 9. 完成定义

阶段 1 完成必须同时满足：

- 两个阻断回归消失；
- smoke、angle、geometry、domain 测试通过；
- diff 不包含阶段 2 内容。

阶段 2 完成必须同时满足：

- rep2 canonical representation 在所有生产者/消费者处统一为 `cxcywh+offset`；
- geometry 边界均显式转换为 `xyxy+offset`；
- denoising 和 anchors 不再产生语义非法 reference；
- rep0/1/3 无回归；
- PostProcessor 非 focal 缺陷被独立修复；
- 配置文件遗留参数仍不在本轮修改范围，并在最终报告中明确列为后续工作。

## 10. 审核交付格式

用户在 User Implementation Gate 每个实现单元完成后提供：

1. 仅该单元的生产代码 diff；
2. 任何与本设计不同的实现选择及原因。

AI 在 AI Test Gate 和 AI Green/Review Gate 负责：

1. 编写、修正和维护当前单元所需测试；
2. 记录 RED 失败证据，并确认失败由目标生产缺陷或旧测试契约导致；
3. 逐调用点检查 tensor layout、激活域与角度域；
4. 运行变更文件 diagnostics、定向测试、smoke 和相关回归；
5. 汇总测试命令与结果，给出明确 PASS 或退回结论；退回时指出具体文件、行和契约违例。

测试脚本及测试输出均由 AI 维护，用户不得通过修改测试来适配生产实现。
