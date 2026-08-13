# Superpowers 工作流制品索引

> 维护说明：当新增或删除 `docs/superpowers/**` 下的 Markdown 文件时，请更新本索引。
> 本索引覆盖 `docs/superpowers/**/*.md` 所有制品（含原 openspec 合并而来的提案/设计/规范/分析）。

---

## specs/ — 已批准规范

[2026-08-13-obb-evaluation-coordinate-fixes-design](specs/2026-08-13-obb-evaluation-coordinate-fixes-design.md) — OBB 评估坐标修复设计：纠正推理/后处理中的 width/height 因子顺序与坐标约定，统一 hw ordering。

[2026-08-10-rep2-nan-diagnostic-runner-design](specs/2026-08-10-rep2-nan-diagnostic-runner-design.md) — DEIMv2-OBB rep2 NaN 远程诊断 runner：单 GPU 短程恢复训练、autograd anomaly 定位、rep2 几何探针与完整失败现场回传设计。

[2026-08-10-rep2-stable-atan2-design](specs/2026-08-10-rep2-stable-atan2-design.md) — DEIMv2-OBB rep2 atan2 数值稳定性修复设计：保持 forward 几何语义不变，仅稳定 backward 分母奇点（`_StableAtan2` 自定义 autograd，`(x²+y²).clamp_min(eps)`，FP16/BF16 临时 FP32），并定义失败现场回放工具的验收契约。

[2026-08-12-focused-application-layer-design](specs/2026-08-12-focused-application-layer-design.md) ★ **canonical** — DEIMv2 聚焦型工程应用层设计：以简化参数设置和统一推理应用为首版目标，通过稳定 AppConfig、任务适配器和结构化 HBB/OBB Prediction 隔离持续变化的底层算法。

[2026-06-02-deimv2-obb-proposal](specs/2026-06-02-deimv2-obb-proposal.md) — DEIMv2-OBB 变更提案：通过 `box_mode` 闸门为 DEIMv2 增加定向边界框支持，使用 ADR 6-distribution DDF。

[2026-06-02-deimv2-obb-design](specs/2026-06-02-deimv2-obb-design.md) — DEIMv2-OBB 架构设计：HBB/OBB 双路径，组件变更与向后兼容保证。

[2026-06-23-synthetic-ellipse-obb-dataset-design](specs/2026-06-23-synthetic-ellipse-obb-dataset-design.md) — 合成椭圆 OBB 数据集密度对照实验设计（H1 假设验证）。

[2026-06-23-deimv2-obb-detection-spec](specs/2026-06-23-deimv2-obb-detection-spec.md) — OBB 检测能力规范：定向边界框检测的接口与行为规范。

[2026-06-23-deimv2-obb-geometry-spec](specs/2026-06-23-deimv2-obb-geometry-spec.md) — OBB 几何操作能力规范：旋转框几何运算（IoU、转换等）规范。

[2026-06-23-deimv2-obb-evaluation-spec](specs/2026-06-23-deimv2-obb-evaluation-spec.md) — OBB 评估能力规范：定向检测评估指标与流程规范。

[2026-06-24-hungarian-matching-diagnosis-design](specs/2026-06-24-hungarian-matching-diagnosis-design.md) — 匈牙利匹配正确性诊断实验设计（Q1-Q3 验证）。

[2026-06-25-decoder-decoupling-design](specs/2026-06-25-decoder-decoupling-design.md) ★ **canonical** — DEIMv2-OBB Decoder 解耦设计：空间-语义-角度三层解耦，解决分类分数与 IoU 质量无关（MAL loss 不收敛）问题。2026-06-26 修订版，已纳入整合评审修正。

[2026-07-01-training-monitoring-refactor-design](specs/2026-07-01-training-monitoring-refactor-design.md) — 训练监控体系重构设计：det_engine.py 日志/监控结构性问题（硬编码 loss key、Comet 双重上报、TensorBoard 冗余、无梯度监控）的重构设计。

[2026-07-07-deimv2-obb-representation-refinement-design](specs/2026-07-07-deimv2-obb-representation-refinement-design.md) — OBB 表示与精细化设计：保留 Ding 式 ADR（外接矩形 + 顶点偏移 ε,η）为主路径，闭合周期角几何、offset 有效性、query 解耦的已知缺口，使 external-rect+offset 设计几何自洽、可测试、可消融。

[2026-07-09-obb-eval-speed-optimization](specs/2026-07-09-obb-eval-speed-optimization.md) — OBB 评估速度优化记录：评估管线性能优化记录。

[2026-07-16-deimv2-obb-loss-refinement-design](specs/2026-07-16-deimv2-obb-loss-refinement-design.md) — DEIMv2-OBB Loss 精细化设计：OBB loss 公式调优设计。

[2026-08-03-dota-dataset-cache-design](specs/2026-08-03-dota-dataset-cache-design.md) — DOTA 数据集分层缓存设计：为 DOTA 数据集引入分层缓存以加速训练数据加载。

[2026-08-04-obb-adr-loss-design](specs/2026-08-04-obb-adr-loss-design.md) — OBB ADR 分解 Loss 设计：ADR（外接矩形 + 偏移）分解式 loss 设计。

[2026-08-04-obb-angle-contract-unification-design](specs/2026-08-04-obb-angle-contract-unification-design.md) — DEIMv2 OBB 角度量纲统一设计：统一全链路角度量纲契约。

[2026-08-05-engineering-platform-refactor-design](specs/2026-08-05-engineering-platform-refactor-design.md) — HBB/OBB 工程化平台兼容式分层重构设计：统一 CLI、运行契约、训练控制器、推理后端、checkpoint 和测试迁移路线。

[2026-08-05-obb-angle-contract-simplification-design](specs/2026-08-05-obb-angle-contract-simplification-design.md) — DEIMv2 OBB 角度契约简化设计（修订版）：对早期角度量纲统一设计的简化修订。

[2026-08-05-obb-decoder-shifted-angle-design](specs/2026-08-05-obb-decoder-shifted-angle-design.md) — DEIMv2 OBB decoder 私有 shifted 角度编码设计：将 decoder 内部绝对角 reference 的 sigmoid 参数化从边界饱和区移到中心（seam 移至 135°），保持公开物理角契约 [0,π) 与 proportional 模式逐位不变。

[2026-08-06-obb-decoder-repair-design](specs/2026-08-06-obb-decoder-repair-design.md) — DEIMv2 OBB decoder 两阶段修复设计：阶段 1 修复角度契约修改引入的阻断回归；阶段 2 统一 rep2 ADR 表示语义（cxcywh+offset canonical）。

[2026-08-06-obb-training-matrix-design](specs/2026-08-06-obb-training-matrix-design.md) — DEIMv2 OBB Stage 1/2 功能稳定性训练矩阵设计：Stage 1/2 功能稳定性训练矩阵实验设计。

[2026-08-07-early-stopping-best-checkpoint-design](specs/2026-08-07-early-stopping-best-checkpoint-design.md) — EMA early-stopping 与 best-checkpoint 设计：监控 EMA mAP50_95，双 best 值（observed/significant）+ patience，best.pth/last.pth 分离契约，三退出路径，首实验 ES-Base 不动现有 schedule。

[2026-08-07-bf16-forward-fp32-loss-design](specs/2026-08-07-bf16-forward-fp32-loss-design.md) — BF16 forward + FP32 loss 当前设计：`use_amp` 选择 BF16 autocast，嵌套浮点输出通过 `tree_map` 转 FP32，训练循环不使用 GradScaler；保留 2026-08-07 精度接口清理结果。

---

## plans/ — 实施计划（Agent 执行用）

[2026-01-17-model-correctness-tests](plans/2026-01-17-model-correctness-tests.md) — 模型输出正确性测试套件：验证可视化、分布、匹配、refinement、loss 单调性等 7 维度。

[2026-06-02-deimv2-obb-tasks](plans/2026-06-02-deimv2-obb-tasks.md) — DEIMv2-OBB 实施任务分解。

[2026-06-23-synthetic-ellipse-obb-implementation-plan](plans/2026-06-23-synthetic-ellipse-obb-implementation-plan.md) — 合成椭圆 OBB 数据集实施计划（基于密度对照设计）。

[2026-06-24-cdn-input-viz](plans/2026-06-24-cdn-input-viz.md) — CDN 生成输入可视化：`test_cdn_generation_visualization()` 单元测试，画正/负噪声 query vs GT 椭圆。

[2026-06-24-hungarian-matching-diagnosis-plan](plans/2026-06-24-hungarian-matching-diagnosis-plan.md) — 匈牙利匹配诊断脚本实施计划：编写 `diagnose_hungarian_matching.py`，验证 Q1-Q3。

[2026-06-25-decoder-decoupling-plan](plans/2026-06-25-decoder-decoupling-plan.md) ★ **canonical** — Decoder 解耦实施计划：拆分 XYWH 路径 + R 路径，Gated Softmax Fusion 桥接，2026-06-26 修订版包含 F7-F9/S1-S3/O1-O5 修复。  

[2026-06-26-h4-lqe-ablation-plan](plans/2026-06-26-h4-lqe-ablation-plan.md) — H4 LQE Angle-Distribution 消融实施计划：可配置 `lqe_num_dist` 排除角度分布污染。

[2026-06-27-decoder-decoupling-plan-v2](plans/2026-06-27-decoder-decoupling-plan-v2.md) — Decoder 解耦实施计划 v2：拆为 XYWH 路径（6 层）+ R 路径（6 层），Gate Fusion 桥接，统一 (ε,η) 角度表示；源自 Hyperplan 对抗分析两轮蒸馏。

[2026-06-29-decoder-decoupling-plan-v3-r-interleaved](plans/2026-06-29-decoder-decoupling-plan-v3-r-interleaved.md) — Decoder 解耦实施计划 v3：R Decoder 交错式修正版。

[2026-07-01-training-monitoring-refactor](plans/2026-07-01-training-monitoring-refactor.md) — 训练监控体系重构实现计划：重构 det_engine.py 日志体系——通用 parse_loss_key() 替代硬编码 loss key、Comet 分层树结构、梯度监控、移除 TensorBoard writer。

[2026-07-09-obb-eval-speed-optimization](plans/2026-07-09-obb-eval-speed-optimization.md) — OBB 评估速度优化实现计划：评估管线性能优化的实施计划。

[2026-07-16-deimv2-obb-loss-refinement-plan](plans/2026-07-16-deimv2-obb-loss-refinement-plan.md) — DEIMv2-OBB Loss 精细化实现计划：OBB loss 公式调优的实施计划。

[2026-07-29-wh-swap-angle-error-experiments](plans/2026-07-29-wh-swap-angle-error-experiments.md) — w/h 互换角度误差放大实验套件实现计划：研究 w/h 互换对角度误差放大的影响。

[2026-07-31-deimv2-obb-five-proposals](plans/2026-07-31-deimv2-obb-five-proposals.md) — DEIMv2-OBB 五项改进实施计划：通过五项改进提升角度预测精度——启用 GatedSoftmaxFusion、ADR 去角度损失、角度范围调整、多角度锚点、先角度后位置；每项独立可消融。

[2026-08-03-dota-dataset-cache](plans/2026-08-03-dota-dataset-cache.md) — DOTA 数据集分层缓存实现计划：分层缓存加速训练数据加载的实施计划。

[2026-08-04-obb-adr-loss-design](plans/2026-08-04-obb-adr-loss-design.md) — OBB ADR 分解 Loss 实现计划：ADR 分解式 loss 的实施计划。

[2026-08-04-obb-angle-contract-unification](plans/2026-08-04-obb-angle-contract-unification.md) — DEIMv2 OBB 角度量纲统一实现计划：角度量纲契约统一的实施计划。

[2026-08-05-engineering-platform-refactor-roadmap](plans/2026-08-05-engineering-platform-refactor-roadmap.md) — HBB/OBB 工程化平台兼容式分层重构实施计划：12 个 TDD 任务、G0-G6 验收门、统一 CLI/推理/训练生命周期迁移顺序。

[2026-08-05-obb-angle-contract-simplification](plans/2026-08-05-obb-angle-contract-simplification.md) — DEIMv2 OBB 角度契约简化实现计划（修订版）：角度契约简化（修订版）的实施计划。

[2026-08-06-obb-decoder-stage1-runtime-recovery](plans/2026-08-06-obb-decoder-stage1-runtime-recovery.md) — OBB Decoder Stage 1 运行时恢复实现计划：恢复基线可运行性的阶段 1 实施。

[2026-08-06-obb-decoder-stage2-rep2-contract](plans/2026-08-06-obb-decoder-stage2-rep2-contract.md) — OBB Decoder Stage 2 Rep2 契约实现计划：统一 rep2 ADR 表示语义的阶段 2 实施。

[2026-08-06-obb-training-matrix](plans/2026-08-06-obb-training-matrix.md) — DEIMv2 OBB Stage 1/2 功能稳定性训练矩阵执行计划：训练矩阵实验的执行计划。

[2026-08-07-ema-early-stopping-best-checkpoint](plans/2026-08-07-ema-early-stopping-best-checkpoint.md) — EMA early-stopping 与 best.pth 恢复实现计划：训练循环加入双 best 值 + patience 状态机，退出时恢复 best.pth 并再验证；与 specs/2026-08-07-early-stopping-best-checkpoint-design 配对。

[2026-08-07-obb-decoder-shifted-angle](plans/2026-08-07-obb-decoder-shifted-angle.md) — DEIMv2 OBB decoder 私有 shifted 角度编码实现计划：6 个 Review Unit 的 TDD 流程，覆盖 contract 函数、MSDeformableAttention、配置传播、anchor/encoder、denoising、geometry decode 站点。

[2026-08-07-bf16-forward-fp32-loss](plans/2026-08-07-bf16-forward-fp32-loss.md) — 精度路径最终实施记录：说明 FP16-only 历史决策被 NaN/Inf 结果推翻，并记录 BF16 forward + FP32 loss、无 GradScaler 的实现与验证证据。

[2026-08-10-rep2-nan-diagnostic-runner](plans/2026-08-10-rep2-nan-diagnostic-runner.md) — rep2 NaN 远程诊断 runner 实施计划：7 个 TDD 任务实现 `test/tool_diagnose_rep2_nan.py` 与纯 CPU 测试，覆盖 checkpoint 分类/恢复、BF16+FP32 诊断循环、atan2 几何探针、失败现场持久化；与 specs/2026-08-10-rep2-nan-diagnostic-runner-design 配对。

[2026-08-10-rep2-stable-atan2](plans/2026-08-10-rep2-stable-atan2.md) — rep2 atan2 数值稳定性修复实施计划：TDD 稳定 atan2 算子与退化几何测试、`_StableAtan2` 实现与 rep2 解码调用替换、失败现场回放工具契约与实现、本地回归与 seed 控制敏感性验收；与 specs/2026-08-10-rep2-stable-atan2-design 配对。

[2026-08-12-focused-application-layer](plans/2026-08-12-focused-application-layer.md) ★ **canonical** — DEIMv2 聚焦型工程应用层首版实施计划：10 个 TDD 任务实现可继承应用 YAML、HBB COCO/OBB DOTA/YOLO-OBB 映射、DEIM adapter、结构化预测、共享 Python API/CLI 和 PyTorch 推理。

[2026-08-13-obb-offline-hw-order-fix](plans/2026-08-13-obb-offline-hw-order-fix.md) — OBB 离线 hw-order 修复实施计划：TDD 修正推理工具与 postprocessor 的 width/height 因子顺序，统一坐标约定。

---

## review/ — 评审与实验结果

[OBB_CODE_REVIEW](review/OBB_CODE_REVIEW.md) — DEIMv2-OBB 代码审查报告：逐行审查 OBB 相关代码，11 项发现（10 项已修复）。  

[2026-06-24-deimv2-obb-self-attention-query-diversity](review/2026-06-24-deimv2-obb-self-attention-query-diversity.md) — 自注意力 query 多样性分析：检查 300 个 query 进入自注意力前的区分度。

[2026-06-25-decoder-decoupling_review_deepseekv4_pro_max](review/2026-06-25-decoder-decoupling_review_deepseekv4_pro_max.md) — Decoder 解耦方案评审（DeepSeek-V4-Pro），评审设计方案与实施计划。  

[2026-06-25-decoder-decoupling-review_GLM_5.2_max](review/2026-06-25-decoder-decoupling-review_GLM_5.2_max.md) — Decoder 解耦方案评审（GLM-5.2-max），评审合理性/完备性/可靠性。  

[2026-06-25-decoder-decoupling-review_Opus_4.8_max](review/2026-06-25-decoder-decoupling-review_Opus_4.8_max.md) — Decoder 解耦方案评审（Opus 4.8 max），结论：当前方案"先于证据"，应先排除廉价根因。  

[2026-06-25-decoder-decoupling-review-INTEGRATED_GLM_5.2_max](review/2026-06-25-decoder-decoupling-review-INTEGRATED_GLM_5.2_max.md) — 三方评审整合报告（GLM-5.2-max），综合 GLM/Opus/DeepSeek 三份评审。  

[2026-06-25-three-reviews-comparison_GLM_5.2_max](review/2026-06-25-three-reviews-comparison_GLM_5.2_max.md) — 三份评审对比与合理性判断（GLM-5.2-max）。  

[2026-06-26-h0-h4-experiment-results](review/2026-06-26-h0-h4-experiment-results.md) — H0-H4 根因实验结果补充：记录整合评审提出的 5 个廉价根因实验的实际执行结果。  

[2026-06-27-workflow-docs-cleanup-note](review/2026-06-27-workflow-docs-cleanup-note.md) — 工作流文档清理记录：所有权契约、canonical 路径、变更文件清单。

[2026-06-29-decoder-interleaved-review](review/2026-06-29-decoder-interleaved-review.md) — Decoder 交错式 R Decoder 代码评审：交错式 R decoder 实现的代码审查。

[2026-06-29-decoder-interleaved-review-v2](review/2026-06-29-decoder-interleaved-review-v2.md) — Decoder 交错式 R Decoder 代码评审 v2：交错式 R decoder 代码审查的第二轮。

[2026-07-01-det-engine-logging-review](review/2026-07-01-det-engine-logging-review.md) — det_engine.py 日志/监控代码审查报告：训练引擎日志与监控代码的审查。

[2026-08-04-obb-angle-units-audit](review/2026-08-04-obb-angle-units-audit.md) — DEIMv2 OBB 角度量纲全链路审计报告（只读）：角度量纲跨全链路的只读审计。

[2026-08-05-obb-angle-units-final-audit](review/2026-08-05-obb-angle-units-final-audit.md) — DEIMv2 OBB 角度量纲最终审计与修改建议：角度量纲最终审计结论与修改建议。

---

## engineering/ — 工程化应用层用户文档

> 这些文档位于 `docs/engineering/`（不在 `docs/superpowers/` 下），是面向终端用户的工程化应用层（`deim_app`）使用指南，与 SDD/MEMO 制品体系互补。此处仅作索引登记。

[application-config](../engineering/application-config.md) — `deim_app` 应用 YAML 配置指南首版：六段公共字段表、继承与覆盖优先级、HBB COCO / OBB DOTA / OBB YOLO-OBB 三例、类别元数据来源、preset 拥有的参数类别、`pretrained` vs `resume` 语义、`remap_mscoco_category` 自动检测规则。

[inference-api](../engineering/inference-api.md) — `deim_app` 推理 Python API 与统一 CLI 指南：`DetectionModel` 快速上手、四个子命令（train/eval/infer/export）的批准 flag 表、JSON/DOTA/visualization 三种输出格式、端到端 smoke 命令。

---

## 约定

- ★ **canonical** 标记的文档是当前技术路线的主参考文档
- 所有评审文档为历史记录，保留原文不加修改（仅路径引用可更新）
- 正式技术决策（提案、设计、规范）置于 `specs/`；任务分解置于 `plans/`；深度分析置于 `review/`
- 本索引必须覆盖 `docs/superpowers/**/*.md` 所有 Markdown 文件；故意排除的文件需在此注明原因
