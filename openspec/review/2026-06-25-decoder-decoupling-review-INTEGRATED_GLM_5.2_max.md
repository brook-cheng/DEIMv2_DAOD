# DEIMv2-OBB Decoder 解耦方案 — 三方评审整合报告

> 整合日期：2026-06-25
> 整合来源：
> - `2026-06-25-decoder-decoupling-review_GLM_5.2_max.md`（GLM-5.2-max）
> - `2026-06-25-decoder-decoupling-review_Opus_4.8_max.md`（Opus 4.8）
> - `2026-06-25-decoder-decoupling_review_deepseekv4_pro_max.md`（DeepSeek-V4-Pro）
> - 对比文档 `2026-06-25-three-reviews-comparison_GLM_5.2_max.md`
> 整合目标：将三份评审中所有已被代码事实核实或互补的发现合并为单一可执行清单，删除冗余与已被反驳项，标注每条的来源、严重度与验证成本，给出统一的决策门控。
> 核心判定：**原方案"先于证据、指向错误、细节多处会崩"。在执行任何架构改造前，必须先通过廉价实验验证或证伪根因链。**

---

## 0. 三方共识（合并可信度高）

以下 4 条被三份评审独立指出、且方向一致，构成对原方案的最强反驳：

| # | 共识 | 来源 |
|---|------|------|
| C1 | **证据链反转诊断脚本自身的结论**：`diagnose_hungarian_matching.py:743-757` 判定逻辑输出"先修 matcher 再动 decoder"，但设计文档据 Q1=PASS / Q2="部分PASS" / Q3=FLAG 直接走向 decoder 解耦 | GLM §1.3 / Opus F3 / DeepSeek §2.2 |
| C2 | **5 项强耦合改动一次性上线、无消融、无回滚门槛**：解耦 + 2× decoder 层数 + 18× 多角度锚 + 正交旋转注意力 + Gated Softmax Fusion，2⁵=32 种组合无法归因 | GLM §4.7 / Opus F13 / DeepSeek §3.5 |
| C3 | **多角度生成锚点 θ=0 会产生 -inf**：`anchors = torch.log(anchors/(1-anchors))`（`deim_decoder.py:674`）在 θ=0 时无界；且 OBB 的 `valid_mask`（`:670-673`）只检查 `[..., :4]`，不查 θ 维度 | GLM §4.3 / Opus 间接 F15 / DeepSeek D1 |
| C4 | **HBB 兼容性验证严重不足**：计划 Task 6 Step 2 只 `YAMLConfig('...') + print('OK')`，无 forward、无数值一致性、无梯度一致性检查 | GLM 遗漏7 / Opus §3.5 F12 / DeepSeek §3.6 |

C1+C2 单独成立即可推翻方案的"立项理由"与"方法论"；C3+C4 是即便强行执行也立即触发的问题。

---

## 1. 根因假设完整集（按验证成本排序，必须先做的根因排除）

三份评审合计提出 6 个互不重叠的根因假设，覆盖"评测口径 / loss 目标 / matcher / LQE / 角度耦合 / query 多样性"6 个层面。**原方案直接跳到第 5 个、跳过了前 4 个更廉价的可能**。

| # | 假设 | 来源 | 验证成本 | 证据强度 |
|---|------|------|----------|---------|
| **H0** | **precision 0.0059 主要是评测口径假象**：`obb_eval.py _tpfp` 不施加分数阈值，postprocessor topk=300 全量返回 → precision ≈ N_tp/300 上界 | Opus F1（独家，已核实 `obb_eval.py:33-73` + `postprocessor.py:67-68`） | **1 min**（加阈值再算） | 🟢 代码级证据 |
| **H1** | **MAL loss 实际在收敛，看板"震荡"是 Kendall 加权产物**：单 batch loss_mal=3-6 与加权后波动并非"MAL 公式不收敛" | Opus F2（独家，已核对 `synthetic_training.log` ep15→30） | **5 min**（看 unweighted loss_mal 曲线） | 🟢 日志级证据 |
| **H2** | **IoU-aware matcher 第二阶段从未触发**：`matcher_change_epoch=45` > 实际训练 30 epoch，`change_matcher=true` 形同虚设；模型全程在无 score↔IoU 对齐的加性代价下训练 | Opus F6（独家，已核实 `synthetic_exp_020.yml:147,284` + `matcher.py:43-72,130-149`） | **30 min**（改 1→重训） | 🔴 最强证据 |
| **H3** | **`mal_iou_type: giou` 被静默忽略**：OBB 分支 `deim_criterion.py:191-192` 硬编码 `probiou()`，配置意图与实际行为脱节；任何基于 giou 的调参直觉都失效 | Opus F5（独家，已核实） | **10 min**（修静默 / 删冗余 key） | 🟢 代码级证据 |
| **H4** | **LQE 在 OBB 下被 angle 分布污染**：`LQE.forward` 对 OBB 把 6 条分布（α,β,γ,δ,ε,η）的统计量一起喂进 `reg_conf` MLP，ε,η（顶点偏移）分布与 ProbIoU 弱相关 → quality_score 失效 → MAL 失去 score↔IoU 的可学习桥梁 | GLM §2.3（独家，已核实 `dfine_decoder.py:334-359` + `deim_decoder.py:288`） | **1 h**（屏蔽后 2 条分布短训 500 iters） | 🟡 机制级推断（未实证） |
| **H5** | **300 query 全为 θ=0.5 且无多样性约束** → 自注意力早期无区分信号 → 推理多 query 聚到同一目标 → 重复检测被记为 FP | Opus §2.7（引用项目内 `analysis/self-attention-query-diversity.md`） | 0.5-1 d | 🟡 分析性证据 |
| **H6** | **decoder 中 θ 与 (cx,cy,w,h,label) 耦合**（原方案假设） | 原设计文档 | **数天-数周** | ❌ 未排除 H0-H5 即跳到 |

**决策门**：在 H6 之前，**H0/H1/H2/H3/H4 全部必须做完**。任意一条 ρ 提升到 ≥ 0.2 即可暂停架构解耦。预计总成本 < 3 GPU·小时，远低于 2× decoder 重写。

---

## 2. 阻断级技术 bug（计划按现状执行必崩，已全部代码核实）

三份评审中**只有 Opus 抓到硬 bug，但 GLM/DeepSeek 都未否认**，经独立代码核实确认：

### B1. `self.decoder_layer` 不存在 → AttributeError

- **来源**：Opus F7（独家，已核实）
- **代码事实**：`deim_decoder.py:391-414` 中 `decoder_layer` 与 `decoder_layer_wide` 是 `DEIMTransformer.__init__` 内的**局部变量**，构造后传入 `TransformerDecoder(...)` 内部存为 `self.layers` (ModuleList)。模型上根本无 `self.decoder_layer` 属性。
- **崩点**：计划 Task 4 Step 1 写
  ```python
  self.r_layers = nn.ModuleList([
      copy.deepcopy(self.decoder_layer) for _ in range(self.num_r_layers)
  ])  # AttributeError
  ```
- **修复前置**：要么在 `__init__` 把局部变量另存为 `self.decoder_layer = decoder_layer`，要么改为 `self.decoder.layers[k]` 取层。

### B2. box_head 输出维度差 8 倍 → 后续 Integral/LQE 全部维度错位

- **来源**：Opus F8（独家，已核实）
- **代码事实**：`deim_decoder.py:476-498` 真实的 `dec_bbox_head = MLP(..., self.num_reg_dist * (self.reg_max + 1), 3)`，OBB 下 = `6 × 33 = 198`。计划把"分布维度（DFL 33-bin）"误当成"标量维度"，写成：
  ```python
  self.bbox_head_xywh = MLP(hidden_dim, hidden_dim, num_reg_dist * 4, 3)   # 计划：24
  self.angle_heads = MLP(hidden_dim, hidden_dim, num_reg_dist * 2, 3)       # 计划：12
  ```
- **正确值**：XYWH 路径 = `4 × (reg_max+1) = 132`；R 路径 = `2 × (reg_max+1) = 66`。
- **下游影响**：Integral（softmax+投影，`dfine_decoder.py:309-331`）、LQE（`dfine_decoder.py:334-359`，对 OBB 取 `num_reg_dist=6`）全部 reshape 维度错位。

### B3. `distance2bbox_obb_xywh` 未创建

- **来源**：GLM 遗漏 1（独家）+ DeepSeek D5（命中但未升严重度）
- **代码事实**：计划 Task 3 Step 3 调用 `distance2bbox_obb_xywh(ref_xywh_scaled, integral(pred_corners_xywh[:4], project))`，但函数不存在。现有 `distance2bbox_obb`（`dfine_utils.py:190`）接 6 维 distance、出 5 维 OBB，无法直接复用。
- **影响**：Task 3 无法执行，核心解码断裂。
- **修复前置**：在 `dfine_utils.py` 新增 `distance2bbox_obb_xywh` 与 `bbox2distance_obb_xywh`，单测互逆。

### B4. 多角度锚点 θ=0 → -inf

- **来源**：GLM §4.3（三方都说，GLM 提供修复路径）
- **代码事实**：计划 Task 2 Step 1 `theta_val = (k * angle_step_deg) / 180.0`，k=0 时 θ=0；随后 `deim_decoder.py:674` `anchors = torch.log(anchors / (1 - anchors))` → -inf。OBB 的 `valid_mask`（`:670-673`）只检 `[..., :4]`，不查 θ → -inf 进入 `select_topk` gather 污染 enc_topk_anchors。
- **修复**：`theta_val = (k + 0.5) * angle_step_deg / 180.0` 避免边界；或 `valid_mask` 增 θ 维度检查。

---

## 3. 严重级技术 bug（不直接崩，但语义/形状错位）

### S1. Gated Softmax Fusion token 数不匹配

- **来源**：Opus F9（独家）
- **代码事实**：`GatedSoftmaxFusion.forward` 做 token 级 cat 与逐元素加权，要求所有 src 形状 `[B, num_tokens, d]` 一致。计划把 encoder memory 作为第三路源（设计 §3 src_A、架构图 "ENC → GF0/GF1"），但 encoder memory 是 `[B, ΣH_iW_i, d]`（成百上千 token），xywh/r 特征是 `[B, 300, d]`。**无法直接 cat**。
- **修复**：encoder memory 必须先通过 cross-attn/可变形注意力汇聚到 300 个 query 位置，再进 Fusion。计划/设计都没有这一步。

### S2. 角度 DFL 维度 / 语义混淆

- **来源**：Opus F10 + DeepSeek §3.2（两份独立命中）
- **代码事实**：计划 Task 4 Step 2
  ```python
  angle_delta = self.angle_heads[layer_idx](r_features)   # 66 维 DFL 分布
  r_current = r_current + angle_delta                      # r_current 是 1 维标量
  ```
  - DFL logits (66 维) vs 标量角度 (1 维) 维度不一致
  - 缺 Integral 解码这一步（softmax → 投影 → 标量）
  - (ε,η) 是 D-FINE 顶点偏移，本身不直接等于"角度增量 Δθ" → 把顶点偏移当角度增量是对 ADR 表示的误用
- **方案 vs 计划矛盾**（DeepSeek §3.2）：设计走 ADR (ε,η)，计划走直接弧度加法，**两者不可同时成立**。需择一而行、并同步两份文档。

### S3. `r % π` 引入边界不连续

- **来源**：Opus F11 + GLM §4.4（两份独立命中）
- **代码事实**：计划 `r_current = r_current % math.pi` 在 179°+2° 取模回 1°，边界处梯度断裂/病态。RIO-DETR 自己用 Shortest-Path Periodic Loss 回避——而设计文档把"周期性优化"列为核心。**与设计初衷自相矛盾**。
- **修复**：用 sin/cos 编码或周期平滑损失。

### S4. R layer 与现有 TransformerDecoderLayer 接口不匹配

- **来源**：GLM §4.2 + DeepSeek §4.6
- **代码事实**：现有 `TransformerDecoderLayer.forward`（`deim_decoder.py:199-308`）依赖：
  - `ref_points_unact` → `F.sigmoid` → `ref_points_detach` → `query_pos_head(ref_points_detach)` 期望 4-5 维输入
  - `MSDeformableAttention` 的 5 维分支需要 `reference_points[..., 4]`（θ）做旋转
  
  R 路径只有角度（1 维）。deepcopy 的 layer 无法直接吃 1 维 ref_points：
  - 无 1 维 `reference_points` 分支
  - `query_pos_head` 期望 ≥4 维
  
- **修复方向**：拼成 5 维 dummy xywh + 真实 θ；或在 R layer 前加 1→hidden_dim 的特征投影。

### S5. Gate Fusion 的并行/串行语义自相矛盾

- **来源**：GLM §4.5 + DeepSeek §3.3
- **代码事实**：设计 Mermaid 图（L45-60）显示所有 6 个 Gate Fusion 都从 L5（xywh 最终层）接收"xywh 特征"，同时 L1 → R1 暗示 R1 在 L1 之后运行。
  - **并行交错** (L0→R0→L1→R1...)：R0 时 L5 还没产出
  - **串行** (L0..L5 全跑完再 R0..R5)：每层 Fusion 拿同一个最终 xywh 特征（冗余且"偷看答案"）
- **副作用（DeepSeek §3.4 独家）**：R 路径在第一次角度预测前就看到了 xywh 最终预测特征 → 信息泄露、R 路径非真正独立。
- **修复前置**：二选一并更新图与计划：要么逐层 L_i → GF_i（无泄露），要么显式记录"L5 全特征注入"为有意设计并解释为何不对称。

---

## 4. 完备性遗漏（合并三方）

### 🔴 阻断级遗漏（不补则训练 crash 或行为错误）

| # | 遗漏 | 来源 |
|---|------|------|
| O1 | DN（去噪）路径与 R decoder 集成未定义。DN boxes 也含 θ，是 R decoder 也对 DN queries 跑？DN 的 angle ref_points 怎么拆？DN 的 angle 回归如何监督？全文未提 | GLM 遗漏2（独家） |
| O2 | aux_outputs 结构 + `loss_local`（FGL）适配未定义。现有 aux_outputs 每层带 `pred_corners` 6×33 维、`ref_points` 5 维。解耦后 xywh 路径 4 条分布、R 路径 2 条。`bbox2distance_obb`（`dfine_utils.py:224`）仍需 6 维 target | GLM 遗漏3（独家） |
| O3 | `deim_criterion.py:701-703` 的 `NotImplementedError`：`if self.box_mode == "obb": raise NotImplementedError()`。若启用 `boxes_weight_format` 配置，OBB 直接抛异常 | GLM 遗漏4（独家） |
| O4 | 参数 plumbing 未落实：`getattr(kwargs, "angle_step", 10)` 但 `__init__` 签名无 `kwargs`；`decouple_angle`、`angle_step`、`num_r_layers` 必须显式加 + 打通 YAML→register 反序列化链路 | GLM 遗漏5（独家） |
| O5 | LQE 在解耦后的归属未定义：xywh 路径 LQE 用 4 条分布？R 路径 score 怎么生成？是否复用 LQE？未说 | GLM 遗漏6（独家） |

### 🟡 严重级遗漏

| # | 遗漏 | 来源 |
|---|------|------|
| M1 | 预训练权重载入策略缺失：解耦使 `dec_bbox_head` 6×33 → 4×33 维度变更 + 新模块（r_layers / gate_fusions / angle_heads），现有 OBB checkpoint 无法 finetune | DeepSeek §4.4（独家） |
| M2 | 训练配方未调整：decoder 6→12 层改变梯度流。主 lr 应减半、warmup 翻倍、梯度裁剪 max_norm=1.0、Kendall σ_lr 监控、XYWH（预训练初始化）vs R（随机初始化）分路径 lr、新模块权重衰减配置 | DeepSeek §4.2（独家） |
| M3 | 延迟 / FPS 影响未评估：DEIMv2 实时检测器定位下 2× decoder 深度严重影响 FPS；部署优化（层融合/TensorRT）能否恢复未讨论 | DeepSeek §4.5（独家） |
| M4 | 失败模式预案缺失：模型训出更差 / xywh 路径退化 / Gate 权重坍缩 / 多角度锚偏好特定角度 / R 路径发散 / Kendall 与 12 层不良交互 —— 6 种失败模式均无应对方案 | DeepSeek §4.3（独家） |
| M5 | 训练条件下沉问题：项目前期已在修复"encoder memory 是否真到 GF0/GF1"等不一致；plan 直接假设了 encoder memory 双路注入，但 encoder memory 经 cross-attn 后的语义变化未建模 | DeepSeek §3.3 命中 |

---

## 5. 失败风险评估

### 5.1 "5 改动一次性"的不可归因风险（Opus F13 + DeepSeek M1 + GLM §4.7）

合并目标：消融 2⁵=32 组合才能归因，单训练成本 ~50 epoch × density_020 ≈ 30 min/run，理论上需 16 小时纯训练才能消融完——但每消融一项还需诊断脚本运行 + 看图。

**Opus 量化结论**：训练变好不知哪项起效；变坏不知哪项拖后腿。这构成"方法论存疑"而非"技术细节有 bug"的更高级风险。

### 5.2 18× 多角度锚点对召回的风险（Opus F15 独家）

设计 §5 多角度锚（angle_step=10°，每位置 18 个锚），但 `select_topk=300`：
- 锚点池扩大 18×，300/topk 意味着 **94% 角度假设在进 decoder 前被丢弃**
- 对于小旋转椭圆，encoder 对角度区分力本就弱 → 大概率把正确角度锚点丢掉 → **直接损害当前 0.97 的召回**
- 或同位置不同角度的近重复锚点挤占 topk → 空间覆盖下降 → 加剧重复检测

### 5.3 "正交注意力"前提已失效（Opus §3.3 独家）

设计 §4 称"当前旋转交叉注意力只沿长轴方向采样 → 容易特征坍塌"。但 `OBB_CODE_REVIEW.md #1` 已经**修复**了旋转注意力的数学错误：当前 `dfine_decoder.py:167-184` 是"先按半边尺寸缩放、再用 R(θ) 整体旋转"的正确实现，采样点 2D 按角度旋转，并非坍缩成一条线。

"沿长轴采样→特征坍塌"是**修复前**的旧 bug 现象。以已不存在的前提论证"必须加正交注意力"，论据失效。正交注意力作为增强或许仍有价值，**但需重新论证动机**。

---

## 6. 整合行动建议（按时间-成本排序）

### 阶段 0 — 廉价根因排除（强制前置，< 3 GPU·小时）

| 步骤 | 改动 | 验证假设 | 预期 |
|------|------|---------|------|
| **0a** | 1 min：`obb_eval._tpfp` 加分数阈值（0.05/0.1）后再算 precision；改看 AP50 + matched/unmatched 平均分差 | H0：precision 是评测假象 | precision 跳到合理量级 → 立项依据不成立 |
| **0b** | 5 min：直接检查 unweighted `loss_mal` 原始曲线（非 Kendall 加权） | H1：MAL 看板震荡是加权产物 | 验证"是否真震荡" |
| **0c** | 30 min：`synthetic_exp_020.yml` `matcher_change_epoch: 45 → 1`，重训 density_020 50 epoch | H2：IoU-aware 匹配缺失导致 score↔IoU 失联 | score↔ProbIoU 相关性显著上升 |
| **0d** | 10 min：清理 `mal_iou_type` 静默失活：让 OBB 也尊重该 key 或确认用 ProbIoU；必要时降 γ 从 1.5→1.0 缓解正样本目标被压低 | H3：目标偏低导致 score 塌缩 | score 直方图分离 |
| **0e** | 1 h：临时让 OBB `LQE.forward` 只用前 4 条分布（屏蔽 ε,η），短训 500 iters | H4：LQE 被 angle 分布污染 | Q3 的 r 显著回升 → **改 1 行解决** |

**统合决策门**：0a-0e 任一让 Q3 ρ ≥ 0.2，**暂停或放弃 decoder 解耦方案**。三方评审都预测：步骤 0c+0e 很可能让 ρ>0.3，**使架构重构变得不必要**。

### 阶段 1 — 修复评测口径（无论后续是否解耦，必做）

- 修改 `obb_evaluate` + `postprocessor`：评测时按分数阈值过滤再算 precision/recall
- 把"matched vs unmatched query 平均 score"作为新的辅助指标
- 同时修复 `deim_criterion.py:701-703` NotImplementedError（GLM O3）

### 阶段 2 — 若 0x 全部排除（ρ 仍 < 0.1），按最小化路径启动解耦

依据三方共识，**禁止 5 项改动一次性上线**，按以下顺序逐项消融：

**子阶段 2A（基础设施，并行）**
- T0a：新增 `distance2bbox_obb_xywh` + `bbox2distance_obb_xywh`，单测互逆（修 B3）
- T0b：`DEIMTransformer.__init__` 显式加 `decouple_angle/angle_step/num_r_layers`，打通 YAML→register（修 O4）
- T0c：修复 B1 `self.decoder_layer` 不存在
- T0d：修 `deim_criterion.py:701` NotImplementedError

**子阶段 2B（XYWH 路径，不引入 R decoder）**
- T1：拆 ref_points 为 xywh(4) + r(1)；box_head 输出 `4*(reg_max+1)`（修 B2）；`distance2bbox_obb_xywh` 解码
- T1-验证：**保持 R 路径用现有逻辑**，确认 xywh-only 路径不 crash
- T1-回归：HBB forward 数值一致性测试 < 1e-6（补 C4 / M3）

**子阶段 2C（R 路径，明确串行 / 并行二选一）**
- T2a：定义 R layer 接口（建议拼 5 维 dummy xywh + 真 θ，复用现有 layer，不 deepcopy 后改接口——修 S4）
- T2b：实现 R decoder forward。**关键决策**：先修 B4 的 θ=0 -inf、S2 的 Integral 缺失、S3 的 `%π` 边界，再写 R forward
- T2c：DN 路径集成（修 O1）
- T2d：aux_outputs 重构 + `loss_local` 适配（修 O2）；LQE 归属定义（修 O5）

**子阶段 2D（多角度锚，单独消融）**
- T3：修 B4，单独验证 anchor 数量与 valid_mask
- T3-消融：**先单独上多角度锚（不解耦）**，看 Q2 bbox/chamfer norm_sep 与 Q3 r 变化
- T3-回滚门：若 0.97 召回落到 < 0.92（Opus F15），回滚

**子阶段 2E（正交注意力 + Gated Fusion）**
- T4a：正交注意力前提重新论证（Opus §3.3）
- T4b：GatedSoftmaxFusion 接入前先通过 cross-attn 把 encoder memory 汇聚到 300 query（修 S1）
- T4c：明确"L5 → 全部 GF" vs "逐层 L_i → GF_i"，二选一并消除信息泄露（修 S5 / DeepSeek §3.4）

**阶段 3 — 联合验证**

每加一个组件跑一次 density_020 短训 + 诊断，记录 Q1/Q2/Q3 + **加阈值 precision** + AP50 + AP75 变化。明确回滚门槛：
- Q1 一对一 < 95% → 回滚
- Q3 ρ 比上一阶段差 → 回滚
- 召回 < 0.92 → 回滚
- 加阈值 precision 不升 → 回滚

### 阶段 4 — 工程化补丁（DeepSeek 独家，编码前或并行）

- 训练配方补充：主 lr 0.0001 → 0.00005（M2）；warmup 2000→4000 steps；梯度裁剪 `max_norm=1.0`；Kendall σ_lr 监控；新模块（r_layers / gate_fusions / angle_heads）独立 lr/weight decay 待评估
- 预训练权重载入策略（M1）：定义 strict=False、`encoder/decoder/dec_bbox_head/dec_score_head` 分别 migrate、新模块 Xavier 初始化
- FPS / 部署评估（M3）：上线前测 FPS；评估 TensorRT 层融合能否恢复；考虑 XYWH / R 路径是否并行化
- 失败模式预案（M4）：6 种失败模式逐条定义回滚策略

---

## 7. 完整三方共识对照表

| 维度 | GLM | Opus | DeepSeek | 整合 |
|------|-----|------|----------|------|
| 立项证据是否成立 | ⚠️ 有保留 | ❌ 不成立 | ⚠️ 关键缺口 | **❌ 不成立** |
| 根因假设是否唯一 | ❌（漏 LQE / matcher / IoU-aware） | ❌（漏 LQE） | ❌（漏绝大多数） | **❌ 至少 6 个未排除** |
| 方案对症状是否对症 | ⚠️（Mode A 不解分类） | ❌（Mode A 自相矛盾） | ⚠️（设计 vs 实施矛盾） | **❌ 默认 Mode A 不触及分类** |
| 计划是否可执行 | ❌（B1/B3/O1-O5 未提） | ❌（B1/B2/B3 独家抓） | ⚠️（B3 单列但 B1/B2 漏） | **❌ 至少 3 个 AttributeError + 1 个核心函数缺失** |
| 测试是否充分 | ❌（HBB 测试弱） | ❌（验证指标错位） | ❌（各 task 无数值检查） | **❌ 需补 6 类数值单测** |
| 5 项改动是否消融 | ❌ 无 | ❌ 无且归因不可能 | ❌ 应分阶段 | **❌ 必须分 5 阶段** |

---

## 8. 三方独家价值点速览

每份评审都有另两份没有的独特贡献，整合时都已保留：

### GLM-5.2 独家
- **H4 LQE 污染假设**（1 行代码改动 + 500 iters，性价比最高的对照实验）
- **O1 DN 路径集成遗漏**（不补即 crash）
- **O5 LQE 在解耦后的归属问题**（反证解耦可能恰是 H4 想做的事）
- 多角度锚 θ=0 -inf bug 的具体修复路径

### Opus 4.8 独家
- **H0 评测口径假象**（代码级证据 `obb_eval.py` + training log）
- **H1 MAL 看板震荡是 Kendall 加权产物**（核对 training log ep15→30 单调下降）
- **H2 IoU-aware matcher 从未触发**（最强证据，`matcher_change_epoch=45` > 实际 30 epoch）
- **H3 mal_iou_type 静默失活**（代码级证据）
- **B1 self.decoder_layer 不存在**（AttributeError 阻断）
- **B2 box_head 维度差 8 倍**（分布式语义混淆）
- **F15 18× 锚点损害召回的量化推理**
- **正交注意力前提已失效**（旧 bug 已修，论证失效）

### DeepSeek-V4-Pro 独家
- **方案 vs 计划角度预测方式矛盾**（ADR ε,η vs 直接弧度加法）最清晰表述
- **M1 预训练权重载入**（6×33→4×33 维度变更）
- **M2 训练配方缺失**（lr/warmup/梯度裁剪/分路径 lr/权重衰减）
- **M3 FPS 延迟评估**（DEIMv2 实时定位）
- **M4 6 种失败模式预案缺失**
- **3 阶段 ABC 分阶段消融方案**（最直观）

---

## 9. 最终结论

> "方案立项所依赖的三条证据——precision 低、MAL 不收敛、诊断指向 decoder——经三方合并核验分别是**评测假象、与日志矛盾、与诊断脚本自身结论相反**。真实存在的 score-IoU 解耦，证据更指向评测口径 + matcher 第二阶段从未启用 + LQE 污染 + 静默配置失活——四者均可通过 < 3 GPU·小时的 config / 1 行代码改动 验证或排除。在这些廉价实验未能证伪之前，不应启动这次数天-数周的 2× decoder 大手术。"

行动顺序：
1. **阶段 0（强制前置，< 3 h）**：0a-0e 廉价根因排除
2. **阶段 1（无论后续是否解耦）**：修复评测口径 + 解锁 O3
3. **阶段 2（仅当 ρ 仍 < 0.1）**：按 2A→2B→2C→2D→2E 子阶段逐项消融推进，每补一项修对应 B/S/O bug，并设回滚门
4. **阶段 3（每子阶段后）**：联合验证 + 回滚门控
5. **阶段 4（与阶段 2 并行）**：补足 DeepSeek 的工程化维度

整体方法论：**先排除最廉价的，再改最贵的；多证据交叉验证，单一证据不动手；一次只改一项，每项必消融。**

---

## 附：本整合的方法声明

- 已核实的关键代码事实（独立阅读）：`obb_eval.py`、`postprocessor.py`、`synthetic_training.log`、`synthetic_exp_020.yml`、`matcher.py`、`deim_criterion.py:155-231`、`deim_decoder.py:391-498`、`dfine_decoder.py:167-184,309-359`
- 三方共识（C1-C4）合并为最高可信度
- 阻塞级 bug（B1-B4）Opus 独家但经核实通过；GLM/DeepSeek 未否认
- 根因假设 H0/H2/H3 Opus 独家代码级证据；H4 GLM 独家机制推断（性价比最高实验）
- 工程维度（M1-M4）DeepSeek 独家，与根因判断正交但实施必要
- 删除冗余：三方都说"C2 5 项捆绑无消融"，整合只保留一次
- 未削弱任何评审的独家发现（H1/LQE/DN/FPS 均保留）