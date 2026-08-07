# DEIMv2-OBB Decoder 解耦方案 / 计划 评审报告

> 评审日期：2026-06-25
> 评审对象：
> - 设计：`docs/superpowers/design/2026-06-25-decoder-decoupling-design.md`（canonical）
> - 计划：`docs/superpowers/plans/2026-06-25-decoder-decoupling-plan.md`（canonical）
> 评审维度：合理性（root cause 是否成立）、完备性（计划是否可执行）、可靠性（技术细节是否正确）
> 评审方法：阅读设计/计划/诊断脚本/历史代码审查/训练日志/配置/criterion 与 decoder 源码 + Oracle 独立对抗式复核
> 评审人：Opus 4.8 (max)
> 结论一句话：**当前方案"先于证据"且"指向错误"。在没有先排除更廉价、证据更直接的根因（评测口径、loss 目标、matcher）之前，不应启动这次 2× decoder 的大手术。**

---

## 0. 结论速览

| # | 类别 | 发现 | 严重度 | 证据 |
|---|------|------|--------|------|
| F1 | 根因·致命 | **"precision=0.037"主要是评测假象**，不是模型质量信号，decoder 改造无法移动它 | 🔴 阻断 | `obb_eval.py:192-208` 对全部 300 query 不设分数阈值；precision ≈ N_gt/N_pred ≈ 20/300 |
| F2 | 根因·致命 | **设计文档"MAL loss 不下降且震荡"与训练日志矛盾**：MAL 单调下降 1.13→0.51 | 🔴 阻断 | `synthetic_training.log` ep15→ep30 |
| F3 | 根因·严重 | **诊断脚本自身的结论是"先修 matcher 再动 decoder"**，与设计文档结论相反 | 🔴 阻断 | `matching_report.txt` CONCLUSION |
| F4 | 根因·严重 | score-IoU 解耦真实存在（ρ=-0.047），但根因更可能在 **loss 目标 + matcher**，而非 decoder 耦合 | 🟠 主要 | 见 §2.4 |
| F5 | 根因·中 | **配置与代码不一致**：`mal_iou_type: giou` 对 OBB 被静默忽略，代码硬编码 ProbIoU | 🟠 主要 | `deim_criterion.py:191-192` |
| F6 | 根因·中 | **IoU-aware matcher 第二阶段从未触发**（`matcher_change_epoch:45` > `epoches:30`） | 🟠 主要 | `synthetic_exp_020.yml:147,284` |
| F7 | 完备·致命 | **`self.decoder_layer` 不存在**，计划的 `copy.deepcopy(self.decoder_layer)` 必崩 | 🔴 阻断 | `deim_decoder.py` 仅有 `self.decoder.layers`(ModuleList) |
| F8 | 可靠·致命 | **box_head 输出维度理解错误**：实际 `num_reg_dist*(reg_max+1)=198`，非 `num_reg_dist*4/2` | 🔴 阻断 | `deim_decoder.py:476-498` |
| F9 | 可靠·严重 | **GatedSoftmaxFusion 形状不匹配**：encoder_memory 的 token 数 ≠ 300 query | 🟠 主要 | 计划 Task1 + 设计 §3 |
| F10 | 可靠·严重 | **角度 DFL 维度/语义混淆**：`r_current + angle_delta`，缺 Integral 解码 | 🟠 主要 | 计划 Task4 Step2 |
| F11 | 可靠·中 | **`r % π` 重新引入角度边界不连续**，与"周期性优化"初衷自相矛盾 | 🟡 中 | 计划 Task4 Step2 |
| F12 | 完备·中 | 文件/路径多处错误：`synthetic_exp_020.yml` 实际在 `synthetic_configs/`；`/mnt/d/.../deimv2_daod` vs `/home/cx/.../DEIMv2_DAOD` | 🟡 中 | 见 §3.5 |
| F13 | 方法·严重 | **5 项改动捆绑、无消融**：解耦+2×层+18×锚点+正交注意力+融合，结果不可归因 | 🟠 主要 | 计划全文 |
| F14 | 方法·中 | **验证指标错位**：成功判据"Q3 r≥0.3"，而 Q3 本身是被污染的代理指标 | 🟡 中 | 计划 Task7 |
| F15 | 方法·中 | **18× 多角度锚点 + topk=300 可能损害召回**（94% 角度假设在进入 decoder 前被丢弃） | 🟡 中 | 设计 §5 |

> **Oracle 独立复核结论**（节选）：*"The plan is misdirected... precision 0.037 is structurally an evaluation artifact... fix the matcher cost function and the quality-loss IoU type FIRST — cheap config changes — before a multi-thousand-line decoder rebuild that bundles 5 untested hypotheses with at least 2 genuine shape-mismatch bugs."*

---

## 1. 评审范围与方法

本次评审不仅核对了设计/计划文档本身，还交叉验证了它们所依据的"证据链"：

- 诊断脚本 `test/diagnose_hungarian_matching.py`（811 行）及其设计文档；
- 诊断真实输出 `test/outputs/matching_diag/matching_report.txt`、`test/outputs/infer_diag/*/score_dist.txt`；
- 训练日志 `synthetic_training.log`（epoch 15–30 的逐项 loss + test 指标）；
- 历史代码审查 `openspec/review/OBB_CODE_REVIEW.md`（11 项 bug，10 项已修）；
- 配置 `configs/custom_obb/synthetic_configs/synthetic_exp_020.yml`、`deimv2_obb_sp.yml`；
- 源码：`deim_decoder.py` / `dfine_decoder.py` / `deim_criterion.py` / `matcher.py` / `obb_eval.py` / `postprocessor.py`；
- 同组的竞争性分析 `2026-06-24-deimv2-obb-self-attention-query-diversity.md`；
- Oracle 高算力模型对"根因 + 方案 + 技术 bug"的独立对抗式复核。

**核心判断标准**：一个架构改造方案要成立，必须满足三个前提——(a) 它针对的"症状"是真实的模型问题而非测量假象；(b) 它针对的"根因"已被证据指向，且竞争性假设已被排除；(c) 它本身在技术上可执行、可消融。当前方案在这三点上**全部不达标**。

---

## 2. 合理性评审：根因诊断存在根本性问题

### 2.1 【F1·阻断】"precision 0.037"主要是评测口径造成的假象

这是本次评审最重要的发现。整个项目的驱动力是"precision 低（0.037）"，但这个数字主要由评测代码的口径决定，**与 decoder 架构几乎无关**。

证据链（`engine/eval/obb_eval.py`）：

```python
# Stage 5: precision / recall at IoU=0.5  (line 192-208)
for cls_id in range(num_classes):
    for img_idx in range(n_imgs):
        det = all_dets[cls_id][img_idx]   # ← postprocessor 输出的全部 300 个 query
        ...
        tp, fp = _tpfp(ious, det[:, 5], 0.5)   # ← 不施加任何分数阈值
        all_tp_sum += tp.sum(); all_fp_sum += fp.sum()
results_dict["precision"] = all_tp_sum / max(all_tp_sum + all_fp_sum, 1)
```

- `_tpfp`（`obb_eval.py:33-73`）对**传入的每一个预测**都标 TP 或 FP，没有分数过滤；
- `postprocessor.py:68` `torch.topk(..., self.num_top_queries=300)` 返回全部 300 个 query，也无置信度阈值；
- 因此每张图：TP ≤ N_gt ≈ 20，FP ≈ 300 − TP ≈ 280；
- **precision ≈ 20/300 ≈ 0.067（理论上界）**，实测 0.037（部分 GT 在 ProbIoU≥0.5 下未匹配）。

**这解释了训练日志中最反常的现象**：`test_precision` 在 epoch 15→30 全程死锁在 0.0370–0.0377，而同期 `test_AP50` 从 0.261 爬升到 0.337、`test_recall` 从 0.953 升到 0.972。AP50 是基于排序的 PR 曲线积分，会随排序改善而上升；而"取全部 300 个 query 算 precision"的数字结构上 ≈ N_gt/N_pred，是个**几乎与模型无关的常数**。

> **直接推论**：把 6 层 decoder 拆成 12 层、加正交注意力、加融合模块，**也无法把 0.037 移动到 0.1**——因为该数字由"输出 300 个框、不过滤"决定，而不是由角度耦合决定。用它作为立项依据和成功判据都是错误的。

**正确做法**：评测时加入置信度阈值（如 0.05/0.1）后再算 precision，或直接用已经在改善的 AP50；若想直接衡量"分数可分性"，记录 matched vs unmatched query 的平均分数差。

### 2.2 【F2·阻断】"MAL loss 不下降且反复震荡"与训练日志矛盾

设计文档背景第 2 条写明立项动机：*"MAL loss 设计意图无法实现，Comet 看板上 MAL loss 不下降且反复震荡"*。但训练日志显示 `train_loss_mal` **单调下降**：

| epoch | 15 | 17 | 20 | 23 | 26 | 28 | 29 | 30 |
|-------|----|----|----|----|----|----|----|----|
| train_loss_mal | 1.126 | 1.084 | 1.040 | 1.015 | 1.004 | 1.001 | 0.549 | 0.507 |

（ep29 起 `no_aug_epoch` 关闭增广，loss 进一步骤降。）原始 MAL loss 没有"不下降/震荡"。**看板上看到的"震荡"极可能是 Kendall 不确定性加权后的*加权* loss**——`KendallWeighting`（`design.md:77-114`，配置 `enabled:true`）会在训练中动态调整 σ_i，导致 MAL 的有效权重随训练波动。把 Kendall 加权产物的波动误读为"MAL 本身不收敛"，进而推断"分数与 IoU 无法正相关 ⇒ decoder 耦合"，是一条断裂的因果链。

### 2.3 【F3·阻断】诊断脚本自身的结论与设计文档相反

设计文档称诊断结果为"Q1=PASS, Q2=部分PASS, Q3=FLAG"，据此走向 decoder 解耦。但诊断脚本真实产出的 `matching_report.txt` 写的是：

```
Q2: Cost Discriminability
  class    norm_sep: 2.83  [OK]
  bbox     norm_sep: 0.82  [FLAG]      ← 不是"部分PASS"，是 box 代价不合格
  chamfer  norm_sep: 0.77  [FLAG]
  probiou  norm_sep: 1.22  [OK]
Q3: Pearson r: 0.0525 / Spearman rho: -0.0470  → FLAG

CONCLUSION
  - Q2 flagged: cost function lacks discriminability
  - Q3 flagged: score unrelated to match quality
  Matching issues found -> fix matcher before decoder.   ← 脚本结论
```

脚本依据自己的判定逻辑（`diagnose_hungarian_matching.py:743-757`）明确输出 **"先修 matcher 再动 decoder"**。设计文档却跳过这个结论，直接走向 decoder。这是对自有诊断证据的**选择性采纳**：

- Q2 的两个 box 代价（bbox=0.82、chamfer=0.77）**不合格**，却在 matcher 中占了 16 份权重里的 10 份（`cost_bbox:5 + cost_chamfer:5`，见 `synthetic_exp_020.yml:274-279`）。即匹配代价的 62.5% 押在了区分力最差的两项上——这是直接的 matcher 设计问题，方案完全没有触及。
- Q3 的 Spearman ρ = **-0.047（负相关）**，比"零相关"更糟，说明 IoU 越高的框分数反而略低，更像 loss/匹配信号问题，而非单纯特征耦合。

### 2.4 【F4·主要】score-IoU 解耦真实存在，但证据指向 loss/matcher 而非 decoder

需要肯定的是：score 与 ProbIoU 几乎不相关（甚至负相关）是**真实信号**，值得解决。但"真问题"不等于"方案对的"。从 criterion 源码看，MAL 的设计本身是正确的：

```python
# deim_criterion.py:191-192, 210-221（OBB 分支）
ious = probiou(target_boxes, src_boxes).squeeze(-1).detach()  # 目标用 ProbIoU
target_score = (ious * one_hot).pow(self.gamma)               # q = ProbIoU^1.5
weight = pred_score.pow(gamma)*(1-y) + y                       # 标准 quality-focal
```

MAL 的正样本目标 = ProbIoU^γ，**与 Q3 测量的 ProbIoU 是同一度量**。也就是说，loss 在数学上就是要把分数训练成 ProbIoU 的单调函数。既然公式正确而相关性仍缺失，问题必然出在**喂给 loss 的输入**（decoder 特征 / matcher 分配 / 目标数值），而非 loss 公式或角度是否独立成路。最可能的机制（按可证伪、可廉价验证排序）：

1. **正样本目标本身偏低 + γ=1.5 进一步压低**：早期 box/角度不准 ⇒ 匹配对 ProbIoU 偏低（如 0.5）⇒ 目标 0.5^1.5≈0.35；叠加 300:20 的正负极度不平衡，模型用"处处输出≈0"即可最小化绝大多数负样本损失（与 `score_dist.txt` 中"数千个预测分数≈0"完全吻合）。这是 quality-focal 的经典退化解。
2. **matcher 区分力不足（F3）**：分配错配 ⇒ 正样本 query 对应错 GT ⇒ 目标分数加到错的 query 上 ⇒ 训练信号是噪声。
3. **IoU-aware matcher 从未触发（F6）**：见下。
4. **decoder 角度-空间耦合**：排在最后，且最贵。即使角度完美独立，只要前三项不修，分数仍会塌缩。

> 这里同时修正 Oracle 的一处判断：Oracle 把 `mal_iou_type: giou` 列为"头号 smoking gun"（认为旋转目标用水平 GIoU 当质量目标会让正样本目标≈0）。但**实际代码对 OBB 硬编码了 ProbIoU、忽略了 `mal_iou_type` 配置**（见 F5）。因此"GIoU 目标"这一具体机制不成立；真正的机制是"ProbIoU 目标偏低 + γ 压低 + 类别不平衡"。结论方向一致、且更强：根因在 loss/matcher，不在 decoder。

### 2.5 【F5·主要】配置与代码不一致：`mal_iou_type: giou` 被静默忽略

`synthetic_exp_020.yml:287` 设了 `mal_iou_type: giou`，但 `deim_criterion.py` 的 OBB 分支（`loss_labels_mal`，line 191-192）**直接调用 `probiou(...)`，根本不读 `mal_iou_type`**（该开关只在 HBB 分支经 `self.local_iou_type` 生效，line 163-189）。后果：

- 这是一个**死配置 / 静默覆盖**：作者以为 MAL 用 giou，实际用 ProbIoU。任何基于"giou"的调参直觉都会落空。
- 它也说明项目里存在"配置意图 ≠ 代码行为"的隐患，建议在动架构前先把这类静默不一致清掉（要么让 OBB 也尊重 `mal_iou_type`，要么删除该 key 并注释）。

### 2.6 【F6·主要】IoU-aware matcher 第二阶段从未触发

`matcher.py` 有两套代价（见 explore 复核）：

- 阶段A（默认）：`C = 2·class + 5·bbox + 2·probiou + 5·chamfer`（加性）；
- 阶段B（`change_matcher=True 且 epoch≥matcher_change_epoch`）：`C = -(class_score · ProbIoU^iou_order_alpha)`，`iou_order_alpha=4.0`（乘性、IoU-aware）。

阶段B 正是业界（Align-DETR / Stable-DINO / RT-DETRv2 系）用来**强化 score↔IoU 对齐**的机制。但 `synthetic_exp_020.yml` 中 `matcher_change_epoch: 45` 而 `epoches: 30`——**阶段B 永远不会启动**。换言之，模型自始至终都在"无 IoU 感知"的匹配下训练，从未经历过会产生 score-IoU 正相关的训练 regime。这是一个**改一行配置就能验证**的竞争性假设，方案却未提及。

### 2.7 还存在另一条同组的竞争性根因（未被纳入对比）

`analysis/self-attention-query-diversity.md` 是项目内自己的分析，结论是：300 个 query 在初始化时角度全为 0.5、空间无多样性约束 ⇒ 自注意力早期无区分信号 ⇒ 推理时多 query 聚到同一目标 ⇒ **"高召回但低精度（重复检测被记为 FP）"**。这条假设与本方案的"decoder 耦合"是**不同的根因**，且指向更轻的修法（query 去重/多样性、或 §2.1 的评测/NMS 口径）。方案没有把它列为竞争假设加以排除。

**小结（合理性）**：方案立项所依赖的三条证据——"precision 低""MAL 不收敛""诊断指向 decoder"——经核验分别是**评测假象、与日志矛盾、与脚本自身结论相反**。真实存在的 score-IoU 解耦，证据更指向 loss 目标与 matcher。**合理性不成立。**

---

## 3. 方案（设计）评审

即便假设"decoder 耦合"是真根因，设计本身也存在若干硬伤。

### 3.1 角度解耦无法直接解决"分类分数"问题，且 Mode A 自相矛盾

问题症状是**分类分数**不可靠（score-IoU 解耦）。但方案默认的 **Mode A** 明确"label 留在 xywh 路径"（设计 §2 表格 + 计划 Global Constraints）。也就是说：

- 分类 head 仍从 xywh 路径特征读分数；
- 把角度拆走，**并没有改变分类 head 的输入与监督**；
- 设计 §2 表格自己也承认 Mode A 下"θ 解耦后 label 仍与空间耦合"。

于是默认方案在逻辑上**没有触及它声称要解决的分类问题**。真正声称能解分类耦合的是 Mode B（独立 Label Decoder），却被推迟到"后续实现"。这是方案内部的目标-手段错位。

（更底层：DETR decoder 的 cross-attention 本就通过注意力机制对不同语义维度做软解耦，是否需要"物理拆路"来让分类忽略角度维度，本身就需要先用证据证明——而非默认。）

### 3.2 【F15】18× 多角度锚点 + topk=300 可能损害召回

设计 §5 把每个网格位置的角度锚点从 1 个增到 18 个（angle_step=10°），但 `select_topk` 仍只取 300。后果：

- 锚点池扩大 18×，topk=300 意味着**约 94% 的角度假设在进入 decoder 前就被丢弃**；
- 模型必须仅凭 encoder 分数在"进 decoder 之前"猜对角度；对小旋转椭圆，encoder 对角度的区分力本就弱 ⇒ 大概率把正确角度锚点丢掉 ⇒ **直接损害召回**（而当前召回 0.97 是少数没坏的指标，有变差风险）；
- 或者，同位置不同角度的近重复锚点挤占 topk 名额 ⇒ 空间覆盖下降，反而**加剧 §2.7 的重复检测**。

### 3.3 正交注意力的前提，对"已修复"的当前代码不成立

设计 §4 称"当前 DEIMv2-OBB 的旋转交叉注意力只沿长轴方向采样 → 容易特征坍塌"。但 `OBB_CODE_REVIEW.md` 的 #1 已经**修复**了旋转注意力的数学错误：当前 `dfine_decoder.py:167-184` 是"先按半边尺寸缩放、再用 R(θ) 整体旋转"的正确实现（`rotated = einsum("bqij,bqhpj->bqhpi", rot, scaled)`），采样点在 2D 平面按角度旋转，并非坍缩成一条线。"沿长轴采样→特征坍塌"的描述更像是**修复前**的旧 bug 现象。以一个已不存在的前提去论证"必须加正交注意力"，论据失效（正交注意力作为增强或许仍有价值，但需重新论证动机）。

### 3.4 参数翻倍的代价 vs 未经证实的收益

decoder 6→12 层、外加每层 Gated Fusion + 多套 head，参数与显存显著增加。在 256×256、3 类、合成椭圆的小数据上，2× 容量更可能带来**过拟合**与训练不稳定，而非解决一个本质上属于 loss/评测口径的问题。投入产出比在"根因未证实"的前提下不成立。

### 3.5 【F12】文件 / 路径引用错误（完备性）

- 计划 Task2 Step3 / 诊断默认值引用 `configs/custom_obb/synthetic_exp_020.yml`，实际该文件在 **`configs/custom_obb/synthetic_configs/synthetic_exp_020.yml`**（根目录下不存在）。
- 计划多处 bash 用 `/mnt/d/cx/thired/deimv2_daod`（小写），而工程实际在 `/home/cx/win_dir/thired/DEIMv2_DAOD`（大小写不同）。即便 WSL drvfs 大小写不敏感可侥幸跑通，文档也应统一，否则他人/未来的你会困惑。
- `configs/custom_obb/deimv2_obb_decouple.yml`、`test/test_infer_diag.py` 中前者尚不存在（计划要创建，可接受），但 §Task7 直接调用 `test/test_infer_diag.py` 时它确实存在——已确认 OK。

---

## 4. 计划（实施）可靠性评审

计划给出了具体到行号与代码片段的实现，这点值得肯定。但其中多处与真实代码结构**不符**，按现状执行会直接报错或行为错误。

### 4.1 【F7·阻断】`self.decoder_layer` 不存在 → 计划代码必崩

计划 Task4 Step1：

```python
self.r_layers = nn.ModuleList([
    copy.deepcopy(self.decoder_layer) for _ in range(self.num_r_layers)   # ← AttributeError
])
```

经源码核验（`deim_decoder.py`）：`decoder_layer` / `decoder_layer_wide` 是 `DEIMTransformer.__init__` 里的**局部变量**，构造后立即传入 `TransformerDecoder(...)` 并在其内部被 deepcopy 进 `self.decoder.layers`（一个 `nn.ModuleList`）。**模型上根本没有 `self.decoder_layer` 属性**。计划这行会抛 `AttributeError`。要可行，必须先在 `__init__` 里把局部变量另存为 `self.decoder_layer`，或改为从 `self.decoder.layers[k]` 取层。

### 4.2 【F8·阻断】box_head 输出维度理解错误（198 vs 6）

计划 Task3 Step2 / Task4 Step1 把 box/angle head 写成：

```python
self.bbox_head_xywh = MLP(hidden_dim, hidden_dim, num_reg_dist * 4, 3)   # 计划
self.angle_heads    = MLP(hidden_dim, hidden_dim, num_reg_dist * 2, 3)   # 计划
```

但真实的 `dec_bbox_head`（`deim_decoder.py:476-498`）输出维度是 **`num_reg_dist * (reg_max + 1) = 6 × 33 = 198`**——是"每个自由度一套 33-bin 分布（DFL）"，不是 6 个标量。`(α,β,γ,δ,ε,η)` 这 6 个值是 `Integral`（softmax+投影，`dfine_decoder.py:309-331`）**之后**才得到的。计划把"分布维度"误当成"标量维度"：

- XYWH 路径应是 4 个分布 × 33 bins = `4*(reg_max+1)=132`；
- R 路径应是 2 个分布（ε,η）× 33 = `2*(reg_max+1)=66`；
- 计划写的 `num_reg_dist*4`、`num_reg_dist*2` 数值与语义都错。

这会让 Integral、LQE（`dfine_decoder.py:334-359`，对 OBB 取 `num_reg_dist=6`、`k=4`）等下游全部维度错位。

### 4.3 【F9·严重】GatedSoftmaxFusion 形状不匹配

计划 Task1 的 `GatedSoftmaxFusion.forward` 对各源做 token 级拼接与加权：

```python
cat = torch.cat([query] + srcs, dim=-1)         # 要求所有 src 形状 [B, num_tokens, d] 完全一致
fused = Σ weights[..., i:i+1] * src
```

单测里三源都是 `[2,100,256]`，能跑。但设计/计划把 **encoder memory 作为第三路源**（设计 §3 src_A、架构图"ENC → GF0/GF1"）。encoder memory 的形状是 `[B, ΣH_iW_i, d]`（多尺度特征图展平，token 数成百上千），而 xywh/r 特征是 `[B, 300, d]`。两者 token 数不同，**无法 token 级拼接/逐元素加权**，会广播错误或直接崩。要把 encoder memory 纳入，必须先经 cross-attention/可变形注意力把它"汇聚"到 300 个 query 位置上——计划/设计都没有这一步。**这是设计层面的结构性缺陷，不是改个数字能解决的。**

### 4.4 【F10·严重】角度 DFL 维度/语义混淆

计划 Task4 Step2：

```python
angle_delta = self.angle_heads[layer_idx](r_features)   # 输出 (ε,η) 的 DFL 分布，维度 2*(reg_max+1)=66
...
r_current = r_current + angle_delta                     # r_current 是标量角度 [..,1]
r_current = r_current % math.pi
```

`r_current` 是 1 维角度，而 `angle_delta` 若是 (ε,η) 的 DFL 分布（66 维），二者**维度不一致**，相加非法。正确链路应是：DFL logits → softmax → Integral 解码成标量偏移 → 再做残差更新。计划缺了 Integral 这步，是维度/语义双重混淆。（且 (ε,η) 是 D-FINE 的"顶点偏移"分布，本身也不直接等于"角度增量 Δθ"——把顶点偏移当角度增量是对 ADR 表示的误用。）

### 4.5 【F11·中】`r % π` 重新引入角度边界不连续

计划 `r_current = r_current % math.pi` 在迭代细化中对角度取模。OBB 角度的周期性边界（0/π 处）正是旋转检测里著名的"边界不连续"难题：179°+2° 经取模回到 1°，在边界处梯度零/病态。RIO-DETR 自己用 **Shortest-Path Periodic Loss** 来回避它——而设计 §RIO-DETR 也把"周期性优化"列为核心。计划却用最朴素的 `%π` 把这个不连续又加了回来，**与设计初衷自相矛盾**。应改用 sin/cos 编码或周期平滑损失处理。

### 4.6 【F13·严重】方法论：5 项改动捆绑、无消融

计划一次性引入：(1) XYWH/R 双路解耦、(2) decoder 层数翻倍、(3) 18× 多角度锚点、(4) 正交旋转注意力、(5) 每层 Gated Softmax Fusion。这 5 项**互相纠缠**，2⁵=32 种组合，每种都要一次完整训练才能消融。后果：训练若变好，不知是哪项起效（甚至可能是某项变好、另一项变坏后的净值）；若变差（鉴于上面的 bug，很可能），更无法定位。**这是评审中仅次于"根因错"的第二大风险。**

### 4.7 【F14·中】验证指标错位

计划 Task7 把成功判据定为"Q3 的 r 从 0.05 提升到 ≥0.3"。但 Q3 这个指标本身有方法学缺陷：

- 只在**被匹配的 ~20 个 query**上算相关（范围受限/range restriction），方差被压缩，相关系数天然偏低；
- 用 ProbIoU 高斯近似而非精确多边形 IoU；
- 诊断脚本重建的 `total_cost` 还漏掉了真实 matcher 里的 `cost_giou:2`（`synthetic_exp_020.yml:277`），与线上代价不一致。

把一个被污染的代理指标当唯一成功判据，可能"指标动了但真实精度没动"或反之。应以"加阈值后的 precision / AP"为主判据。

---

## 5. 建议：先做廉价的"config 实验"，再决定是否动架构

Oracle 与本评审一致建议：**在动 decoder 之前，先用只改配置/评测的廉价实验，验证或证伪"decoder 耦合"假设**。预计总成本 2–3 GPU·小时，远低于这次重写。

| 步骤 | 改动（仅配置/评测） | 验证的假设 | 预期 |
|------|--------------------|-----------|------|
| 0 | 评测加置信度阈值（0.05/0.1）后再算 precision；或改看 AP50 + matched/unmatched 平均分差 | F1：precision 是评测假象 | precision 立刻"恢复正常量级"，证明它本不该是立项依据 |
| 1 | `matcher_change_epoch: 45 → 1`（让 IoU-aware 阶段B 从头生效）；或直接缩短到 ≤ epoch 数 | F6：IoU-aware 匹配缺失 | score↔ProbIoU 相关性上升 |
| 2 | 降 `cost_bbox`/`cost_chamfer` 权重、提高 `cost_class`/`cost_probiou` 权重 | F3：box 代价区分力不足主导匹配 | 匹配更准、训练更快 |
| 3 | 清理 `mal_iou_type` 静默忽略（让 OBB 也尊重该 key，或确认就用 probiou）；必要时调 γ（1.5→1.0）缓解正样本目标被压低 | F4/F5：目标偏低导致分数塌缩 | score 直方图分离 |
| 4 | 训 5 epoch，测 ρ(score, ProbIoU)（在匹配对上，并加测加阈值后的 precision） | 综合 | **若 ρ≥0.3 → 本 decoder 方案可被数据"否决"** |
| 5 | 仅当 1–3 后 ρ 仍 <0.1：冻结 backbone、只训 decoder+head 10 epoch，看现有 decoder 是否有能力学到对齐 | 决定 decoder 是否真的是瓶颈 | 若能学到 → 耦合假设证伪 |

**门控判据**：在"仅配置改动 + 冻结实验"未能把 ρ 顶到 ≥0.3 之前，不启动任何 decoder 架构改动。本评审与 Oracle 都预测——**步骤 1–3 很可能就让 ρ>0.3，从而使本方案变得不必要。**

---

## 6. 如果证据最终仍指向 decoder：最小化、可消融的路径

若上述实验做完，ρ 依旧贴近 0、且冻结实验证明现有 decoder 确无能力学到对齐，那么"角度需要独立处理"才成为有依据的下一步。即便如此，也应：

1. **一次只改一项**，每项单独消融（先只做"角度独立 head + Integral 残差更新"，不要同时上 18× 锚点/正交注意力/三源融合）。
2. **先解决 §4 的 5 个技术 bug**：补 `self.decoder_layer`；head 用 `*(reg_max+1)`；角度更新走 Integral 解码 + 周期损失（弃 `%π`）；融合若要用 encoder memory，先经 cross-attn 汇聚到 query；多角度锚点要同步放大 topk 或改为"角度由 decoder 细化而非锚点枚举"。
3. **优先考虑比"拆 decoder"更轻的解法**：角度先验注入 query 初始化（呼应 query-diversity 分析）、或在 encoder 侧增强角度特征，而非 2× 参数的物理拆路。
4. **成功判据换成端到端指标**（加阈值 precision / AP50 / AP75），而非被污染的 Q3 r。

---

## 7. 评审结论

| 维度 | 结论 |
|------|------|
| **合理性** | ❌ 不成立。立项三大依据（precision 低、MAL 不收敛、诊断指向 decoder）分别是评测假象（F1）、与日志矛盾（F2）、与诊断脚本自身结论相反（F3）。真实的 score-IoU 解耦，证据更指向 loss 目标 + matcher（F4/F5/F6）与 query 多样性，而非 decoder 角度耦合。 |
| **完备性** | ❌ 不达标。关键代码假设与真实结构不符（`self.decoder_layer` 不存在 F7、head 维度错 F8），文件/路径多处错位（F12），且把 5 项强耦合改动捆绑、无消融计划（F13）。 |
| **可靠性** | ❌ 不达标。至少 2 个会直接崩（F7/F8）、2 个结构性错误（融合形状 F9、角度维度 F10），1 个与设计初衷自相矛盾（`%π` 边界 F11），验证指标被污染（F14）。 |

**总评**：当前 decoder 解耦方案是**对一个评测假象 + 误读的 loss 现象的过度工程化响应**，且其实施细节在多处与真实代码冲突、会直接报错。**建议暂停本方案**，先执行 §5 的廉价 config/评测实验。绝大概率，问题在"评测口径 + matcher + loss 目标/不平衡"层面就能显著缓解；只有当这些都排除后，才谈得上 decoder 的架构改造，且要以最小化、可消融、先修 bug 的方式推进。

> 附：本评审与 Oracle 独立复核在主结论上完全一致（"premature / misdirected，先修 matcher 与 loss 目标，再谈 decoder"）。两处对 Oracle 的修正：(i) `mal_iou_type:giou` 实际被代码忽略、OBB 硬编码 ProbIoU（F5），故"GIoU 目标"这一具体机制不成立，但"目标偏低导致分数塌缩"的结论方向不变且更强；(ii) 正交注意力的"沿长轴坍塌"前提对当前已修复代码不成立（§3.3）。
