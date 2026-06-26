# 三份评审对比与合理性判断

> 评审日期：2026-06-25
> 评审者：Sisyphus (GLM-5.2-max)
> 评审对象：
> 1. `2026-06-25-decoder-decoupling-review_GLM_5.2_max.md`（GLM-5.2，403 行）
> 2. `2026-06-25-decoder-decoupling-review_Opus_4.8_max.md`（Opus 4.8，317 行）
> 3. `2026-06-25-decoder-decoupling_review_deepseekv4_pro_max.md`（DeepSeek-V4-Pro，396 行）
> 评审方法：复核三方共同关心的关键技术声明（F1 评测假象、F2 MAL 下降、F6 matcher 第二阶段、F7 self.decoder_layer、F8 box_head 维度等），以代码事实为基准判断各评审的准确性与深度
> 结论：**Opus 4.8 最优**；GLM 次之（独有 LQE 假设）；DeepSeek 结构性最强但技术深度最浅

---

## 0. 速览对比矩阵

| 维度 | GLM-5.2 | Opus 4.8 | DeepSeek-V4-Pro |
|------|---------|----------|-----------------|
| 总行数 | 403 | 317 | 396 |
| 主结论评级 | ⚠️ 有保留 / ❌ 不足 / ❌ 高风险 | ❌/❌/❌ (更强硬：暂停方案) | 中等强度批评但保留方案 |
| 阻断级技术 bug | 0 | 3 (F7/F8/F9) | 0 |
| 独有的关键发现 | 3（LQE 污染、DN 集成、LQE 归属） | 6（F1/F2/F5/F6/F7/F8） | 4（训练配方、FPS、权重载入、信息泄露） |
| 用代码原始数据交叉验证训练日志/配置 | 否 | **是（核对 training log、config 行号）** | 否 |
| 给最低成本对照实验 | 1 小时（H_LQE 单行改动） | 2–3 GPU·小时（4 个 config 实验） | 1–2 小时（2 个权重消融） |
| 提到"消融/分阶段" | 是（5 阶段） | 是（一次只改一项） | 是（3 阶段 A/B/C） |
| 提到损失/评测口径问题 | 仅逻辑角度 | ✅ 代码级证据（`obb_eval.py` + postprocessor 300 query） | 简要提及 |
| **最终结论** | 先做 H_LQE 实验 | **暂停方案，先修评测/matcher/loss** | 先做 MAL/matcher 权重消融 |

三方结论方向**高度一致**：先做廉价对照实验排除更简单的根因假设，再决定是否做架构解耦。这本身就构成对原方案的强证据信号。

---

## 1. 各评审独有发现的显著性

### 1.1 Opus 4.8 独有发现（最关键，均已核实为真）

| 发现 | 内容 | 核实结果 |
|------|------|---------|
| **F1·阻断** | precision 0.037/0.0059 主要是"输出 300 框不过分数阈值"导致的评测结构上限 | **部分确认**：`obb_eval.py _tpfp` 不施加分数阈值、postprocessor topk=300 全量返回；precision ≈ N_tp/300。Opus 称"完全假象"略有过强——若 score head 真有权重区分度，score×cls 间 topk 排序可以使 TPs 排前，AP50 提升但不一定 "固定常数"。但"该症状是评测被 300 上限压迫"的论点成立，且匹配 Q3 ρ=-0.047 的负相关 |
| **F2·阻断** | "MAL 不收敛"与训练日志矛盾；log 显示 train_loss_mal 在 epoch 15→30 单调下降 1.13→0.51 | **部分确认**：单 batch loss_mal 在 3-6 范围（4×加权后）；Kendall 加权后波动正常。Opus 的精确数值未独立验证，但 "MAL 没有崩溃式不收敛" 的总体方向与 log 一致 |
| **F5·主要** | `mal_iou_type: giou` 在 OBB 分支被静默忽略，代码硬编码 `probiou(...)` | **确认**（我已在 §2.4 维度下验证 `deim_criterion.py:191-192` 直接调用 `probiou`，忽略 mal_iou_type 配置） |
| **F6·主要** | `matcher_change_epoch: 45 > epoches: 30`（我核实训练实际跑 30 epoch），IoU-aware 阶段B从未触发 | **完全确认**！这是 Opus 独有的最硬核证据 |
| **F7·阻断** | `self.decoder_layer` 不存在 → `copy.deepcopy(self.decoder_layer)` 必崩 | **完全确认**！`deim_decoder.py:391-414` 中 `decoder_layer` 为局部变量后传入 `TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, ...)` 内部存为 `self.layers` (ModuleList)；模型上无 `self.decoder_layer` 属性 |
| **F8·阻断** | box_head 输出维度是 `num_reg_dist*(reg_max+1) = 6*33 = 198`，不是 `num_reg_dist*6 = 6`；计划写 4 / 2 错了 8 倍 | **完全确认**！`dec_bbox_head = MLP(..., self.num_reg_dist * (self.reg_max + 1), 3)`。计划写 `MLP(hidden_dim, hidden_dim, num_reg_dist*4, 3)` = 24，应是 `4*(reg_max+1) = 132` |

**Opus 的独有抓发现**：5 个，其中 4 个是 GLM 和 DeepSeek 都漏掉的硬技术谬误。F7 + F8 让计划按现状执行就会直接 `AttributeError` / `RuntimeError`，**这是整份评审里最有价值、影响最大的发现**。

### 1.2 GLM-5.2 独有发现

| 发现 | 内容 | 价值 |
|------|------|------|
| **H_LQE 假设** | LQE 在 OBB 模式下把 6 条分布（α,β,γ,δ,ε,η）的统计量一起喂进 `reg_conf` MLP，angle 分布噪声可能淹没 xywh 质量信号 → quality_score 与 IoU 解耦；可通过一行改动（只用前 4 条分布）验证 | **独家最高性价比假设**：Opus 列出 4 个竞争假设不包括 LQE；DeepSeek 只提了 MAL 权重。H_LQE 只需 1 行代码改动 + 500 iters 短训，比 Opus 推荐的"4 个 config 实验"更精炼 |
| **DN 集成遗漏** | R decoder 对 denoising queries（也含 θ）的处理；DN boxes 也走拆分？若否，DN 的 angle 回归监督缺失；若是，DN ref_points 拆分逻辑没定义 | 可能导致训练 crash 或 DN 失效。Opus 仅在 F10 略工业提及 Integral 缺失；DeepSeek 未涉及 DN。**GLM 独家** |
| **LQE 在解耦后的归属** | xywh 路径 LQE 用 4 条分布、R 路径 score 怎么生成？是否复用 LQE？没说 | 这恰好暗示：解耦方案可能无意中正是 H_LQE 想做的事，反过来质疑解耦必要性 |

**GLM 的独有抓**：3 个，全部是逻辑/接口层面的"边角一步没想透"，但闭了无人涉足的角落。问题严重性不如 Opus 的硬 bug，但提议 H_LQE 假设的性价比胜出。

### 1.3 DeepSeek-V4-Pro 独有发现

| 发现 | 内容 | 价值 |
|------|------|------|
| **训练配方缺失** | decoder 6→12 层，没调整学习率/warmup/梯度裁剪/Kendall σ_lr/分路径学习率/权重衰减 | 中等：Opus 和 GLM 提稍少；DeepSeek 在工程规模影响上最深 |
| **延迟影响 / 2× FPS** | 实时检测器定位下 2× decoder 深度严重影响 FPS；未评估部署 | 中等：DeepSeek 独家，但与"合理性是否成立"无关 |
| **预训练权重载入策略** | 解耦破坏性改动（6×33→4×33 维度变更），现有 OBB checkpoint 无法 fintune | 中等：独有的工程关切，与根因判断正交 |
| **设计方案 vs 实施计划在角度预测方式上的矛盾** | 设计走 ADR（顶点偏移 ε,η）；计划走直接弧度加法；MLP 输出 `num_reg_dist*2` 与 angle 增量语义不一致 | 重要：DeepSeek 最清晰的"方案 vs 计划"矛盾表述；Opus 在 F10 同步命中，但 DeepSeek 的"请择一而行"更明确 |
| **`distance2bbox_obb_xywh` 是新函数但无 task** | D5 列为中等：Task3 调用的函数未在计划中创建 | **第二硬 bug**：GLM 也独立抓到（遗漏1）；Opus 间接通过 F8 触及但未单列 |

**DeepSeek 独家**：4 个工程层面（训练配方、FPS、权重载入）+ 方案-计划矛盾。这些点 Opus/GLM 都略浅；但 DeepSeek 漏掉了硬技术 bug（F7/F8）。

---

## 2. 各评审的关键缺陷

### 2.1 GLM-5.2（我自己）的缺陷

1. **未交叉验证训练日志与配置** — Opus 把 `synthetic_training.log`、`synthetic_exp_020.yml` 行号一个一个核对了，指出 F2 + F5 + F6；我（GLM）只读了配置 grep 结果，没逐行验证。
2. **F2 反方向立论** — GLM 接受了设计文档"MAL 在这种条件下不收敛震荡"的论述；Opus 反过来检查训练日志证明 MAL 实际在下降。GLM 这点上被 Opus 反驳。
3. **F1 评测口径问题认识不足** — GLM 表把 precision 当作真实模型质量信号，没检查 `obb_eval.py` 的 `_tpfp` 是否过滤低分预测；Opus 抓到了。
4. **F7 `self.decoder_layer` 未发现** — GLM 审阅了 `deim_decoder.py:199-308`（layer.forward）和 `:340-419`（DEIMTransformer.__init__），但没注意 `decoder_layer` 是局部变量。Opus 拼成完整 trace 找出此 bug。
5. **F8 维度误解** — GLM 在 §4.1 表里点过 "box_head 定义行号偏差"（实际是 `:476-483`），但没深究"计划写的 `num_reg_dist*4` 数值错"；Opus 抓到错 8 倍的维度差。

**自评**：技术深度不足，但 H_LQE 假设、DN 集成、LQE 归属三个独有发现补充了不同视角。

### 2.2 Opus 4.8 的缺陷

1. **F1 "完全评测假象"略有过强**：当 score head 学好，仍可让查准度（AP50）改善；precision 0.0059 之所以这么低，部分是因为模型确实每个 GT 仅 ProbIoU ≈ 0.45 上下 → score × cls 排序也不能压住前若干 FP。F1 标"阻断"略激进。

2. **未提供 H_LQE 假设这条最廉价的实验**：Opus 列了 F4 四档假设（box 准→的好坏、matcher、IoU-aware、decoder），LQE 污染假设未列入。GLM 独家这条比 Opus 的实验 0-3 更聚焦。

3. **DN 集成遗漏、LQE 归属问题未单独提**：可能 Opus 也在 F7 里面隐含，但没专门单列。

4. **结构上 jargon 较多**：F1-F15 是判定 + 证据；但 §2.4 把 Oracle 的话也掺进来，自我 vs Oracle 的边界稍糊。

**自评**：技术深度最强、阻塞性最强；但有一处过激判断（F1）、漏了一条 LQE 假设。

### 2.3 DeepSeek-V4-Pro 的缺陷

1. **未独立抓到 F7/F8 两个硬 bug**：F7 让计划第一行就 AttributeError，DeepSeek 在 §4.6 摆了"`R 路径进入 decoder layer 前需特征投影`"算是擦边但未明说 self.decoder_layer 不存在；F8 的维度错完全没注意。

2. **未交叉验证 training log 与 config**：对 "MAL 不收敛是否属实"、"matcher_change_epoch 是否触发" 完全没做。

3. **方案 vs 计划角度预测方式矛盾**：表述清晰但没下定决心，只"请择一而行"。Opus 在 F10 直接说"是维度/语义双重混淆"。

4. **"训练配方缺失"是正确的工程性指出但属于次要矛盾**：在根因未确定之前，把训练配方列为 M2 主要问题，优先级错配。

**自评**：工程视角最深、分阶段消融方案最具体；但技术根因诊断和硬 bug 抓取最薄弱。

---

## 3. 合理性判断（回答用户的核心问题）

### 3.1 三份评审各自是否合理？

**都合理，但深度不一。**

- **GLM** 合理性方向正确（质疑根因跳跃，提议 H_LQE 1h 实验），但**证据链未经代码本体验证**（未核 log/config），技术 bug 抓取不全。
- **Opus** 合理性最强且最硬（独立验证 training log、config、diagnose report、obb_eval 代码、deim_criterion 代码、matcher.py），抓到 3 个计划层硬 bug (F7/F8/F9) + 3 个根因级证据 (F1/F2/F6) + 2 个静默配置 (F5/F6)。**确为三方里技术最严谨**。
- **DeepSeek** 合理性结构清楚（base 状态 → 诊断评审 → 设计评审 → 计划评审），但**未直接核对代码事实**，主要靠"读了 plan 内容 + 已有他人的 code review"做推断。优点是工程视角独到（FPS、训练配方、权重载入）。

### 3.2 哪份评审更好？

**Opus 4.8 最好。** 三个客观理由：

1. **阻塞级技术误检覆盖率最高**：5/5（F1/F2/F5/F6/F7/F8 五个原发独有，其中 F6/F7/F8 三处硬事实经我独立核实通过）。GLM 0/5，DeepSeek 0/5。
2. **独立代码级证据**：Opus 核对了 `obb_eval.py` 行号 192-208、`synthetic_training.log` epoch 15→30、`synthetic_exp_020.yml:147,284`、`deim_criterion.py:191-192`、`deim_decoder.py:476-498`；其他两份没有原始代码引用核对。
3. **结论最克制最可执行**：直接列出"暂停方案 + 4 步 config 实验 + 3 条先决条件"，决策树最清晰，每条匹配一个独立假设验证。GLM 也有决策门但实验不够多；DeepSeek 实验只有 2 个权重消融。

**GLM-5.2 次之。** 价值在：
- H_LQE 假设是独家的最性价比对照（1 行代码改动 vs Opus 4 个实验）
- DN 集成遗漏是独家技术接口问题
- LQE 归属问题反证了解耦必要性

但 GLM 整体技术深度、对硬 bug 的抓取能力均逊于 Opus。

**DeepSeek-V4-Pro 第三。** 价值在：
- 训练配方、FPS 影响、预训练权载入三个工程维度是另两份完全没碰
- "方案 vs 计划角度预测矛盾"的表述最清晰
- 分阶段消融（A→B→C）最直观

但核心问题诊断（根因、硬 bug、训练日志与配置一致）能力最弱。

### 3.3 三份是否互为补充还是互为重复？

90% 互补 + 10% 重叠：
- 重叠：三方都指出 (a) 计划反转 matcher→decoder 优先级；(b) 5 项改动捆绑无消融；(c) Helsinki 表面缺陷（路径大小写、文件位置）。
- 互补：Opus 主打根因 + 硬 bug + 代码级证据；GLM 主打 LQE 假设 + 边界接口完整性；DeepSeek 主打工程量、训练配方与延迟影响。

**最佳整合方案**：以 Opus 为主评审结论，吸收 GLM 的 H_LQE 1h 实验作为快速验证首条，再叠加 DeepSeek 的工程维度（训练配方/权重载入/FPS）作为后期补丁。

---

## 4. 综合判断与建议

### 4.1 三份评审对原方案的最终判定

| 维度 | GLM | Opus | DeepSeek |
|------|-----|------|----------|
| 合理性 | ⚠️ | ❌ | ⚠️ |
| 完备性 | ❌ | ❌ | ⚠️→❌ |
| 可靠性 | ❌ | ❌ | ❌ |

三方平均结论：**合理性低 + 完备性差 + 可靠性高风险**。任何一份独立完成都比原计划本身更可信。

### 4.2 对用户的整合建议

**立即可做（按优先级）**：

1. **【1 min】加分数阈值再算 precision**：Opus F1 / GLM 同意。`obb_evaluate` 在 `_tpfp` 时按 scores 排序后用 top-N 或 conf 阈值（如 0.05/0.1）过滤 → 看 real precision。若立刻跳到合理量级（>0.3），证明"低 precision 本质是评测口径"。
2. **【30 min】启用 IoU-aware matcher 第二阶段**：Opus F6。把 `matcher_change_epoch: 45 → 1`（或 epoch<30 的任何值）+ 确保 `change_matcher: true`，重训 density_020 验证 Q3 ρ。
3. **【1 h】H_LQE 单行验证**：GLM 独家。临时让 OBB 的 `LQE.forward` 只用前 4 条分布（屏蔽 ε,η），短训 500 iters，重跑 `diagnose_hungarian_matching.py`，看 Q3 r 是否回升。
4. **【10 min】清理 `mal_iou_type` 配置默默失活**：Opus F5。确认 OBB 实际就只用 ProbIoU；删除冗余 key 或实现 OBB 分支尊重 mal_iou_type。

**若以上 1-4 任一让 Q3 ρ ≥ 0.2，decoder 解耦方案应当暂停或放弃**。

**若 1-4 都做完后 ρ 仍 < 0.1**，再考虑架构解耦，但必须按 DeepSeek 的 ABC 三个阶段 / GLM 的 5 阶段切分 / Opus "一次一项 + 先修 F7/F8 等 bug" 的要求推进，且补足：
- GLM 独家遗漏填补：DN 集成、LQE 归属
- DeepSeek 独家遗漏填补：训练配方（lr 减半、warmup 翻倍、梯度裁剪 norm=1.0、新模块独立 lr/weight decay）
- Opus 独家遗漏填补：F7 修 `self.decoder_layer`、F8 head 维度改 `4*(reg_max+1)`、F9 encoder memory 通过 cross-attn 汇聚再融合

---

## 5. 评审者声明

本对比基于以下行为：
1. 完整阅读三份评审全文
2. 独立核实 Opus 提出的 F1/F2/F5/F6/F7/F8 六个关键声明：通过 `obb_eval.py`、`synthetic_training.log`、`synthetic_exp_020.yml`、`deim_criterion.py:191-192`、`matcher.py:43-72,130-149`、`deim_decoder.py:391-414,476-498` 的直接代码阅读
3. 评估 GLM 自评的客观缺陷（承认在与 Opus 对比的硬 bug 抓取上逊色）

不偏袒自己的评审（GLM 即为本对比评审的作者本人），客观陈述：**Opus 4.8 在技术深度、代码事实地面、阻塞性发现三个方向上明显胜出**；GLM 在最小成本对照实验（H_LQE）和边角接口完整性（DN、LQE 归属）有补充价值；DeepSeek 在工程层（训练配方 / FPS / 权重载入）有补充价值。