# DEIMv2-OBB Decoder 解耦方案与计划评审

> 评审日期：2026-06-25
> 评审模型：GLM-5.2-max
> 评审对象：
> - 设计：`openspec/2026-06-25-decoder-decoupling-design.md`
> - 计划：`openspec/plans/2026-06-25-decoder-decoupling-plan.md`
> - 诊断证据：`test/diagnose_hungarian_matching.py` 及其输出 `test/outputs/matching_diag/matching_report.txt`
> - 代码基线：`engine/deim/{deim_decoder,dfine_decoder,dfine_utils,deim_utils,deim_criterion}.py`
> 评审维度：合理性（hypothesis→solution 因果链）、完备性（任务覆盖与遗漏）、可靠性（可实现性与风险）

---

## 0. 评审结论速览

| 维度 | 评级 | 一句话结论 |
|------|------|-----------|
| **合理性** | ⚠️ 有保留 | 根因假设与诊断证据存在错位；未排除更廉价的替代解释（LQE 机制）就跳到架构重构 |
| **完备性** | ❌ 不足 | 缺少 `distance2bbox_obb_xywh` 创建任务、DN 路径集成、aux_outputs 重构、criterion NotImplementedError 修复等关键任务 |
| **可靠性** | ❌ 高风险 | 4 个重大改动（解耦 + 多角度锚 + 正交注意力 + Gated Fusion）同时落地，无消融、无回归、无回滚门槛；多处接口与现有代码不匹配 |

**总判定**：**不建议按当前计划直接进入实施**。建议先做 1 个低成本对照实验排除 LQE 假设，再决定是否需要架构解耦；若确认需要，计划需补齐 5 处遗漏任务并拆分为可独立验证的阶段。

---

## 1. 证据复核：诊断报告说了什么 vs 设计文档引用了什么

### 1.1 诊断报告原文（`matching_report.txt`）

```
Q1: One-to-Many        → PASS (100% 一对一)
Q2: Cost Discriminability
  class    norm_sep: 2.83  [OK]
  bbox     norm_sep: 0.82  [FLAG]
  chamfer  norm_sep: 0.77  [FLAG]
  probiou  norm_sep: 1.22  [OK]
Q3: Score-IoU Correlation
  Pearson r: 0.0525     [FLAG]
  Spearman ρ: -0.0470
CONCLUSION: Matching issues found -> fix matcher before decoder.
```

### 1.2 设计文档的引用（design.md L9）

> "经过匈牙利匹配诊断（Q1=PASS, **Q2=部分PASS**, Q3=FLAG r=0.05）..."

### 1.3 错位

| 项 | 报告原意 | 设计文档转述 | 偏差 |
|----|---------|-------------|------|
| Q2 | bbox/chamfer **FLAG**（< 1.0 阈值） | "部分 PASS" | **弱化**了 FLAG 信号 |
| Q2 结论 | 代价函数对 bbox/chamfer **区分力不足** | 未提及 | 回避了 matcher 侧问题 |
| 总结论 | "fix matcher **before** decoder" | 直接 "fix decoder" | **反转**了诊断给出的优先级 |

**这是合理性的第一处裂痕**：诊断脚本的判定逻辑（`diagnose_hungarian_matching.py:744-757`）明确写着 `if min(q2.values()) <= 1.0: "Q2 flagged: cost function lacks discriminability"`，并且综合结论是"匹配有问题 → 先修 matcher 再考虑 decoder"。设计文档把这个优先级倒置了，却没有给出为什么推翻诊断结论的论证。

---

## 2. 合理性评审：根因假设是否站得住脚

### 2.1 设计文档的根因假设

> "θ（角度/r）与 (cx,cy,w,h,label) 共用同一个 decoder layer 导致空间-语义-角度信息耦合。"
> → 推论：把 θ 拆到独立 R decoder 即可让分数重新与 IoU 相关。

### 2.2 诊断证据实际支持的解释

Q3 FLAG（r=0.05）说明"分类分数与 IoU 无关"。在 DEIMv2 中，分数与 IoU 的相关性由 **LQE（Local Quality Estimator）** 机制建立，不是由"decoder 是否解耦"建立。让我追踪这条链路：

**链路 1 — LQE 如何让分数携带 IoU 信息**（`dfine_decoder.py:334-359`，`deim_decoder.py:288`）：

```python
# dfine_decoder.py LQE.forward
prob = F.softmax(pred_corners.reshape(B, L, num_reg_dist, reg_max+1), dim=-1)
prob_topk, _ = prob.topk(self.k, dim=-1)           # 取每个分布的 top-k 概率
stat = torch.cat([prob_topk, prob_topk.mean(...)], dim=-1)
quality_score = self.reg_conf(stat.reshape(B, L, -1))
return scores + quality_score                       # 把"分布锐度"加到分类分上
```

- HBB 模式：`num_reg_dist=4`，`pred_corners` 是 4 条边距分布（α,β,γ,δ）。分布越锐（top-k 概率越高）→ 框越确定 → 质量越高。这是 GFLv2 的经典机制，与 IoU 正相关。
- OBB 模式：`num_reg_dist=6`，`pred_corners` 是 6 条分布（α,β,γ,δ,**ε,η**）。后 2 条是**外接矩形顶点偏移**（`dfine_utils.py:190-221`），不是边距。

**关键问题**：ε,η 的分布锐度**是否与 IoU 正相关**？没有证据。顶点偏移的分布形状反映的是"角度/顶点预测的确定性"，但 ProbIoU 对角度误差和尺寸误差的敏感度不同。把 6 条分布的 stat 一起喂进 `reg_conf` MLP，angle 分布的噪声可能**淹没** xywh 分布的信号。

**这是一个比"decoder 耦合"更具体、更廉价可验证的假设**：LQE 的 quality_score 因为混入了不相关的 angle 分布统计量而失效。

**链路 2 — MAL loss 如何训练 LQE**（`deim_criterion.py:155-231`）：

```python
target_score_o[idx] = ious                    # target = IoU（OBB 用 probiou）
target_score = target_score_o.unsqueeze(-1) * target   # target = IoU * one_hot(cls)
target_score = target_score.pow(self.gamma)  # γ=1.5
weight = pred_score.pow(γ) * (1 - target) + target
loss = BCE(src_logits, target_score, weight=weight)
```

MAL 把 `sigmoid(logits) + LQE_quality` 推向 `IoU^γ`。如果 LQE 的 quality_score 与 IoU 不相关（因为 angle 分布污染），那么 MAL 的 target 和 prediction 之间缺少可学习的桥梁，loss 自然震荡不下降——**这正是 Comet 看板上观察到的现象**。

### 2.3 替代假设：LQE 污染假设

> **假设 H_LQE**：OBB 模式下 LQE 把 6 条分布的统计量统一送入 `reg_conf`，其中 ε,η（顶点偏移）分布与 ProbIoU 质量弱相关，污染了 α,β,γ,δ（边距）分布的质量信号，导致 quality_score 无法反映 IoU，进而 MAL loss 无法把分数与 IoU 绑定。

**验证成本**：~30 分钟，零架构改动：
1. 临时修改 `LQE.forward`，在 OBB 模式下只用前 4 条分布：`pred_corners[..., :4*(reg_max+1)]`
2. 在 density_020 上短训，重跑 `diagnose_hungarian_matching.py`
3. 观察 Q3 的 r 是否显著回升

### 2.4 设计假设 vs 替代假设的对比

| 维度 | 设计假设 H_dec（decoder 耦合） | 替代假设 H_LQE（LQE 污染） |
|------|------------------------------|---------------------------|
| 解释 Q3 r=0.05 | ✓（间接：耦合导致 label/angle 互相干扰） | ✓（直接：quality_score 计算源被污染） |
| 解释 MAL 震荡 | ✓（间接） | ✓（直接：target 与 pred 缺桥梁） |
| 解释 Q2 bbox/chamfer FLAG | ⚠️（需额外假设：角度回归差拖累整体回归） | ⚠️（同左，独立于 LQE） |
| 验证成本 | 12-layer decoder + fusion + 多锚 + 正交 attn，数天 | 1 行代码 + 短训 30 分钟 |
| 可证伪性 | 难（4 个改动同时上，无法归因） | 强（只用 4 条分布 vs 6 条分布，对照清晰） |

### 2.5 合理性结论

**根因尚未收敛**。设计文档从"分数与 IoU 无关"直接跳到"decoder 耦合"是因果跨度过大。在没有排除 H_LQE 的情况下投入大规模架构重构，存在"改了半天发现根因在 LQE"的返工风险。

**建议**：在启动 decoder 解耦之前，先执行 H_LQE 验证实验（见 §5 建议 A）。若 H_LQE 成立，可用极小改动解决问题；若不成立，再进入 decoder 解耦，且此时证据更硬。

---

## 3. 完备性评审：计划任务覆盖与遗漏

### 3.1 计划现有 7 个 Task

| Task | 内容 | 评价 |
|------|------|------|
| T1 | GatedSoftmaxFusion 模块 + 单测 | ✓ 模块本身清晰 |
| T2 | Anchor 多角度生成 | ⚠️ 有 bug（见 §4.3） |
| T3 | XYWH Decoder 路径调整 | ⚠️ 行号/命名不准（见 §4.1） |
| T4 | R Decoder 路径 | ⚠️ 接口与现有 layer 不匹配（见 §4.2） |
| T5 | 正交旋转注意力 | ✓ 改动位置准确 |
| T6 | 配置与集成测试 | ⚠️ 测试过于薄弱（见 §4.6） |
| T7 | 合成数据集验证 | ⚠️ 缺消融对照（见 §4.7） |

### 3.2 遗漏的关键任务（按严重度排序）

#### 🔴 遗漏 1：`distance2bbox_obb_xywh` 未创建

计划 Task 3 Step 3 调用 `distance2bbox_obb_xywh(ref_xywh_scaled, integral(pred_corners_xywh[:4], project))`，但该函数**不存在**。现有 `distance2bbox_obb`（`dfine_utils.py:190`）接收 6 维 distance 并输出 5 维 OBB。File Structure 部分提到 `dfine_utils.py: MODIFY: xywh-only bbox ops` 但**没有任何 Task 落实这个修改**。

**影响**：Task 3 无法执行，核心解码逻辑断裂。

#### 🔴 遗漏 2：DN（去噪）路径未处理

现有 `forward`（`deim_decoder.py:782-814, 850-871`）对 denoising queries 做完整 split 与 aux_loss 构造。DN boxes 也含 θ。新增 R decoder 是否对 DN queries 也跑？如果是，DN 的 angle ref_points 怎么拆？如果否，DN 的 angle 回归怎么监督？

计划全文未提及 DN 与 R decoder 的集成。**这会导致训练 crash 或 DN 失效**。

#### 🔴 遗漏 3：aux_outputs 结构与 loss_local 兼容

现有 `aux_outputs` 每层带 `pred_corners`（6*(reg_max+1) 维）和 `ref_points`（5 维）。`loss_local`（FGL，`deim_criterion.py:270-318`）依赖 `pred_corners` reshape 为 `(N, reg_max+1)` 并按 `num_reg_dist` 分组。

解耦后 xywh 路径输出 4 条分布、R 路径输出 2 条分布。aux_outputs 如何拼装？`loss_local` 的 `bbox2distance_obb`（`dfine_utils.py:224`）仍需 6 维 target。**计划未说明 loss 侧如何适配新的 corners 结构**。

#### 🟡 遗漏 4：`deim_criterion.py:701-703` 的 NotImplementedError

```python
if self.box_mode == "obb":
    raise NotImplementedError()   # get_loss_meta_info, boxes_weight_format
```

若启用 `boxes_weight_format`（部分 baseline 配置启用），OBB 模式直接抛异常。计划声明"保持 criterion 兼容"但未修复此已知障碍。

#### 🟡 遗漏 5：参数 plumbing 未落实

计划多处用 `getattr(self, "decouple_angle", False)` 和 `getattr(self, "angle_step", 10)`，但 `DEIMTransformer.__init__`（`deim_decoder.py:315-346`）的签名**没有这些参数**，也没有 `**kwargs`。Task 2 Step 2 写 `self.angle_step = getattr(kwargs, "angle_step", 10)` —— 但 `kwargs` 不存在。

需要显式在 `__init__` 加参数，并在 YAML config → `register` 反序列化链路中打通。计划未覆盖这条链路。

#### 🟡 遗漏 6：LQE 在解耦后的归属

LQE 当前用 `pred_corners`（6 条分布）算 quality_score。解耦后：
- xywh 路径只有 4 条分布 → LQE 用 4 条（这恰好是 H_LQE 验证想做的事）
- R 路径有 2 条分布 → R 路径的 score 怎么来？是否也需要 LQE？

计划未说明 LQE 如何拆分。如果 xywh 路径的 LQE 只用 4 条分布，那**解耦方案无意中验证了 H_LQE**——但这恰恰说明 H_LQE 可能才是根因，不需要整个 R decoder。

#### 🟢 遗漏 7：HBB 回归测试

计划 Task 6 Step 2 只 `YAMLConfig('...deimv2_obb_sp.yml')` 然后 print OK。没有实际跑 HBB forward 确认输出 shape/数值不变。需要至少 1 个 HBB 模式 forward smoke test + 数值一致性检查。

### 3.3 完备性结论

7 个 Task 中有 3 处🔴级遗漏（会导致实施中断或 crash）、3 处🟡级遗漏（会导致训练异常或无法配置）、1 处🟢级遗漏（测试不足）。**计划无法按现有形态端到端跑通**。

---

## 4. 可靠性评审：可实现性与风险

### 4.1 代码引用准确性

| 计划引用 | 实际位置 | 偏差 |
|---------|---------|------|
| "OBB 分支第 646-662 行" | `deim_decoder.py:647-663` | ±1 行，可接受 |
| "box_head 定义处约 840 行" | `dec_bbox_head` 定义在 `:476-483`；`:840` 是 forward 里 θ 量纲调整 | **严重偏差**，会误导实施者 |
| "MSDeformableAttention OBB 5 维分支约 167-184 行" | `dfine_decoder.py:167-184` | ✓ 准确 |
| "Gate 类 `deim_utils.py:70`" | `deim_utils.py:70-83` | ✓ 准确 |
| 计划用 `self.bbox_head` | 实际是 `self.dec_bbox_head`（ModuleList） | **命名不一致** |
| 计划用 `self.bbox_head[i]` 在 layer forward 里 | 实际 layer forward 接收 `bbox_head` 参数（外部传入 `self.dec_bbox_head`） | **作用域误解** |

### 4.2 R Decoder 与现有 layer 的接口不匹配

计划 Task 4 Step 1：`self.r_layers = nn.ModuleList([copy.deepcopy(self.decoder_layer) ...])`。

现有 `TransformerDecoderLayer.forward`（`deim_decoder.py:199-308`）依赖：
- `ref_points_unact` → `F.sigmoid` → `ref_points_detach` → `query_pos_head(ref_points_detach)`
- `MSDeformableAttention` 的 5 维分支需要 `reference_points[..., 4]`（θ）做旋转

R 路径只有角度（1 维）。deepcopy 的 layer 无法直接吃 1 维 ref_points：
- `query_pos_head` 期望 4 或 5 维输入
- `MSDeformableAttention` 没有 1 维 reference_points 分支

**计划未定义 R layer 的 ref_points 结构**（是拼成 5 维用 dummy xywh？还是改 layer 接 1 维？）。这是实施时第一个会撞的墙。

### 4.3 多角度锚点的数值 bug

计划 Task 2 Step 1 生成 `theta_val = (k * angle_step_deg) / 180.0`，k=0 时 `theta_val = 0`。

随后 `deim_decoder.py:674`：
```python
anchors = torch.log(anchors / (1 - anchors))   # inverse sigmoid
```

`anchors[..., 4] = 0` → `log(0 / 1) = -inf`。虽然 `valid_mask` 用 `anchors > eps` 过滤，但 `valid_mask` 对 OBB 只检查 `[..., :4]`（`:670-673`），**不检查 θ 维度**。于是 `theta=0` 的锚点会带着 `-inf` 进入后续 `select_topk` 的 gather，污染 enc_topk_anchors。

**修复**：`theta_val` 应从 `eps` 起步，或 `valid_mask` 加上 θ 检查，或用 `(k+0.5)*angle_step/180` 避免边界。

### 4.4 角度参数化不一致

现有代码精心维护两套 θ 表示：
- **sigmoid 空间 [0,1]**：decoder 内部、ref_points、anchor 存储
- **弧度 [0,π]**：criterion/matcher/postprocessor 接口（在 `forward` 末尾 `:839-848` 转换）

计划 Task 4 Step 2 的 R decoder forward：
```python
r_current = r_current + angle_delta
r_current = r_current % math.pi     # ← 弧度空间
```

这混用了弧度空间更新与 sigmoid 空间约定。`% math.pi` 在 0 和 π 边界**不可导**（梯度断裂），且与现有"sigmoid 空间 residual + 边界处 distance2bbox_obb 转换"的范式冲突。`r_init = anchors[..., 4:5] * math.pi`（Task 4 Step 3）又把 sigmoid 空间 anchor 转成弧度——**两条空间混用，梯度链不清晰**。

### 4.5 GatedSoftmaxFusion 的架构定位模糊

Mermaid 图（design.md L45-60）显示：所有 6 个 Gate Fusion（GF0~GF5）都从 `L5`（xywh 最后层）接收"xywh 特征"。同时 `L1 -->|"inter_ref_bbox 作参考点"| R1` 暗示 R1 在 L1 之后运行。

**矛盾**：如果 R0 在 L0 之后运行（R 路径独立迭代），那 R0 时 L5 还没产出，GF0 拿不到"xywh 最终特征"。如果 R 路径在 xywh 全部跑完后才运行，那"inter_ref_bbox 作参考点 → R1"的逐层反馈不存在，R 路径就是纯串行 6 层，Gate Fusion 的"每层融合"失去意义（每层融合的都是同一个最终 xywh 特征）。

**需澄清**：R decoder 与 xywh decoder 是**并行交错**（L0→R0→L1→R1...）还是**串行**（L0..L5→R0..R5）？两者实现完全不同，计划同时画出了两种语义。

### 4.6 测试薄弱

- Task 1 单测：只测 shape，不测数值正确性（如权重和为 1、梯度回传）
- Task 6 Step 2：`YAMLConfig(...)` + print，**不跑 forward**，无法发现接口不匹配
- Task 6 Step 3：`python train.py` 启动训练，"不应 crash"——但崩溃点大概率在第一次 backward（DN split、aux_loss、loss_local reshape），启动通过不代表训练通过
- **无中间层输出数值检查**：解耦是深层架构改动，需要 unit test 验证每层 ref_points/corners 的 shape 与数值范围

### 4.7 无消融、无回滚门槛

4 个改动（decoder 解耦 + 18 倍多角度锚 + 正交注意力 + Gated Fusion）**同时上线**：
- 若 Q3 的 r 回升到 0.3，不知道是哪个组件起作用
- 若 r 没回升，不知道是哪个组件拖后腿
- 没有"若短训后 r < X 则回滚"的明确止损条件

Task 7 Step 2 预期"Q3 的 r 值显著提升（从 0.05 到 ≥ 0.3）"——但没有定义"未达 0.3 怎么办"。

### 4.8 可靠性结论

5 处技术风险（接口不匹配、数值 bug、参数化冲突、架构语义模糊、测试薄弱）叠加 4 改动同时上线，**实施失败概率高**，且失败时无法定位。

---

## 5. 综合建议

### 建议 A（必做，先于一切）：H_LQE 对照实验

**目的**：在投入架构重构前，用最小成本排除 LQE 污染假设。

**做法**：
1. 在 `dfine_decoder.py` 的 `LQE.forward` 加分支：OBB 模式下 `pred_corners = pred_corners[..., :4*(self.reg_max+1)]`，`num_reg_dist=4`
2. 用 `configs/custom_obb/synthetic_configs/synthetic_exp_020.yml` 短训 ~500 iters
3. 跑 `test/diagnose_hungarian_matching.py`，看 Q3 的 r

**决策门**：
- 若 r ≥ 0.2 → H_LQE 成立，根因在 LQE 不在 decoder 耦合。改 1 行代码 + 重训即可，**放弃当前解耦计划**。
- 若 r < 0.1 → H_LQE 不成立，根因更可能在 decoder。**进入建议 B 的修订计划**。
- 若 0.1 ≤ r < 0.2 → 部分成立。LQE 是贡献因子之一，但仍需解耦；解耦计划中务必让 xywh 路径 LQE 只用 4 条分布（见遗漏 6）。

**成本**：~1 小时（含短训）。**收益**：可能省下数天的架构重构，或为重构提供更硬的证据。

### 建议 B（若建议 A 排除 LQE 后仍需解耦）：修订计划

按以下顺序补齐与拆分：

**阶段 0 — 基础设施（并行）**
- T0a：在 `dfine_utils.py` 新增 `distance2bbox_obb_xywh` 与 `bbox2distance_obb_xywh`（对应遗漏 1），单测验证互逆
- T0b：在 `DEIMTransformer.__init__` 显式加 `decouple_angle`、`angle_step`、`num_r_layers` 参数，打通 YAML→register 链路（遗漏 5）
- T0c：修复 `deim_criterion.py:701-703` NotImplementedError（遗漏 4）

**阶段 1 — XYWH 路径（不引入 R decoder）**
- T1：拆 ref_points 为 xywh(4) + r(1)，xywh 路径 box_head 输出 4 维，`distance2bbox_obb_xywh` 解码
- T1-验证：**保持 R 路径用现有逻辑（6 维 corners 的后 2 维）**，确认 xywh-only 路径不 crash，HBB 不变
- T1-回归：HBB forward 数值一致性测试（遗漏 7）

**阶段 2 — R 路径**
- T2a：定义 R layer 的 ref_points 结构（建议拼成 5 维 dummy xywh + 真实 θ，复用现有 layer，不 deepcopy 后改接口——见 §4.2）
- T2b：实现 R decoder forward，明确**串行**（xywh 全跑完再跑 R）还是**交错**，二选一，不要画两种（见 §4.5）
- T2c：DN 路径集成（遗漏 2）：R decoder 对 DN queries 的处理
- T2d：aux_outputs 重构（遗漏 3）：定义 corners 拼装方式，修 `loss_local` 适配

**阶段 3 — 多角度锚点**
- T3：修复 θ=0 的 -inf bug（§4.3），单独验证 anchor 数量与 valid_mask
- T3-消融：先单独上多角度锚（不解耦），看 Q2 的 bbox/chamfer norm_sep 是否改善

**阶段 4 — 正交注意力 + Gated Fusion**
- T4a：正交注意力（Task 5），独立验证
- T4b：GatedSoftmaxFusion（Task 1），接入 R 路径

**阶段 5 — 联合验证**
- 每"上 1 个组件"跑一次 density_020 短训 + 诊断，记录 Q1/Q2/Q3 变化
- 明确回滚门槛：任一组件使 Q1 退步（< 95% 一对一）或 Q3 r < 上一阶段值 → 回滚该组件

### 建议 C（无论 A/B 都应做）：补强测试

1. **数值单测**：每个新模块测 shape + 数值范围 + 梯度回传（`.backward()` 不报错 + grad 非 NaN）
2. **HBB 回归**：`box_mode="hbb"` 时 forward 输出与 baseline checkpoint 数值差 < 1e-6
3. **DN smoke test**：`num_denoising > 0` 时训练 1 iter 不 crash
4. **criterion 集成 test**：`loss_local`、`loss_labels_mal` 在新 corners 结构下产出有限 loss

### 建议 D：澄清 Q2 FLAG

Q2 的 bbox/chamfer norm_sep < 1.0 是 matcher 侧问题。即使解耦 decoder 让回归更准，**代价函数本身的区分力**仍可能不足。建议：
- 单独消融 matcher 权重（`weight_dict` 里 `cost_bbox: 5, cost_chamfer: 5`）是否需要调整
- 或诊断"匹配到的 query 的 bbox 代价分布"与"该 query 实际 IoU"的关系——如果代价低但 IoU 也低，说明 L1/chamfer 在 OBB 下不是好的 IoU 代理

这部分与 decoder 解耦正交，可并行排查。

---

## 6. 设计文档的其他观察

### 6.1 Mode A/B 与 YAGNI

设计文档提出 Mode A（label 留 xywh）和 Mode B（独立 label decoder），计划选 Mode A。这是正确的 YAGNI 取舍。但文档里 Mode B 细节、MoE TODO、`gate_spatial_semantic` 等占了不少篇幅却暂不实施——建议把 Mode B/MoE 移到单独的"未来工作"文档，保持当前 spec 聚焦。

### 6.2 自检清单未勾选

`design.md:198-203` 的自检 4 项全未勾选。评审后可勾选状态：
- [x] 所有设计要点对应明确的代码改动位置 → **不通过**（`distance2bbox_obb_xywh` 无对应 task）
- [x] angle_step 有硬件降级方案 → 通过（提到增大 step）
- [x] 双模式 label 解耦有明确的配置控制 → 通过
- [x] MoE 作为独立 TODO，不影响 baseline → 通过

### 6.3 RIO-DETR / STD 引用

设计文档引用了 RIO-DETR 的 Content-Driven θ 预测、正交旋转注意力，以及 STD 的空间侧分支。这些是合理的设计灵感，但：
- RIO-DETR 是从零设计的 OBB 检测器，其 Content-Driven 机制与 DEIMv2 的 D-FINE refinement 深度耦合。直接嫁接到 DEIMv2 的 FDR pipeline 上，兼容性论证不足。
- 建议在实施前，先读 RIO-DETR 的 angle decoder 具体结构，确认"每层 Gate Fusion 融合 xywh + r + enc"是否真的是 RIO-DETR 的做法，还是混合了 STD 的侧分支思路。两者的融合时机不同。

---

## 7. 最终建议行动顺序

```
1. 建议 A：H_LQE 对照实验（1h）—— 决定是否需要 decoder 解耦
   ├─ r ≥ 0.2 → 改 1 行，重训，结束
   └─ r < 0.1 → 进入 2
2. 建议 D：matcher 权重消融（0.5h，与 3 并行）—— 排查 Q2 FLAG
3. 建议 B：按 5 阶段修订计划实施，每阶段独立验证
   ├─ 阶段 0：基础设施（T0a/T0b/T0c 并行）
   ├─ 阶段 1：XYWH 路径 + HBB 回归
   ├─ 阶段 2：R 路径 + DN + aux_outputs
   ├─ 阶段 3：多角度锚（单独消融）
   ├─ 阶段 4：正交 attn + Gated Fusion
   └─ 阶段 5：联合验证 + 回滚门槛
4. 建议 C：全程补强测试
```

---

## 8. 评审者声明

本评审基于对以下材料的完整阅读：
- 设计与计划文档全文
- `engine/deim/{deim_decoder,dfine_decoder,dfine_utils,deim_utils,deim_criterion}.py` 关键段落
- `test/diagnose_hungarian_matching.py` 全文
- 诊断报告输出 `matching_report.txt`
- `OBB_CODE_REVIEW.md` 前 80 行（已知 bug 清单）
- `configs/custom_obb/` 配置文件 grep 结果
- git log 近 30 条

未独立验证的部分（需实施时留意）：
- `matcher.py` 的完整代价计算逻辑（仅看了 diagnose 脚本里的复现）
- `obb_geometry.py` / `obb_ops.py` 的完整几何运算
- `postprocessor.py` 的 OBB 后处理
- RIO-DETR / STD 原论文（建议实施者自行核对嫁接兼容性）

本评审为只读评审，未修改任何源码或文档。
