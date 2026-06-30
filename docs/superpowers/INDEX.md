# Superpowers 工作流制品索引

> 维护说明：当新增或删除 `docs/superpowers/**` 下的 Markdown 文件时，请更新本索引。
> 正式规范与变更文档不在本索引范围内，见 `openspec/INDEX.md`。

---

## design/ — 脑暴与设计草稿

[2026-06-25-decoder-decoupling-design](design/2026-06-25-decoder-decoupling-design.md) ★ **canonical** — DEIMv2-OBB Decoder 解耦设计：空间-语义-角度三层解耦，解决分类分数与 IoU 质量无关（MAL loss 不收敛）问题。2026-06-26 修订版，已纳入整合评审修正。

---

## plans/ — 实施计划（Agent 执行用）

[2026-06-25-decoder-decoupling-plan](plans/2026-06-25-decoder-decoupling-plan.md) ★ **canonical** — Decoder 解耦实施计划：拆分 XYWH 路径 + R 路径，Gated Softmax Fusion 桥接，2026-06-26 修订版包含 F7-F9/S1-S3/O1-O5 修复。  

[2026-06-24-hungarian-matching-diagnosis-plan](plans/2026-06-24-hungarian-matching-diagnosis-plan.md) — 匈牙利匹配诊断脚本实施计划：编写 `diagnose_hungarian_matching.py`，验证 Q1-Q3。  

[2026-06-24-cdn-input-viz](plans/2026-06-24-cdn-input-viz.md) — CDN 生成输入可视化：`test_cdn_generation_visualization()` 单元测试，画正/负噪声 query vs GT 椭圆。  

[2026-06-26-h4-lqe-ablation-plan](plans/2026-06-26-h4-lqe-ablation-plan.md) — H4 LQE Angle-Distribution 消融实施计划：可配置 `lqe_num_dist` 排除角度分布污染。  

[2026-01-17-model-correctness-tests](plans/2026-01-17-model-correctness-tests.md) — 模型输出正确性测试套件：验证可视化、分布、匹配、refinement、loss 单调性等 7 维度。

---

## review/ — 评审与实验结果

[OBB_CODE_REVIEW](review/OBB_CODE_REVIEW.md) — DEIMv2-OBB 代码审查报告：逐行审查 OBB 相关代码，11 项发现（10 项已修复）。  

[2026-06-25-decoder-decoupling_review_deepseekv4_pro_max](review/2026-06-25-decoder-decoupling_review_deepseekv4_pro_max.md) — Decoder 解耦方案评审（DeepSeek-V4-Pro），评审设计方案与实施计划。  

[2026-06-25-decoder-decoupling-review_GLM_5.2_max](review/2026-06-25-decoder-decoupling-review_GLM_5.2_max.md) — Decoder 解耦方案评审（GLM-5.2-max），评审合理性/完备性/可靠性。  

[2026-06-25-decoder-decoupling-review_Opus_4.8_max](review/2026-06-25-decoder-decoupling-review_Opus_4.8_max.md) — Decoder 解耦方案评审（Opus 4.8 max），结论：当前方案"先于证据"，应先排除廉价根因。  

[2026-06-25-decoder-decoupling-review-INTEGRATED_GLM_5.2_max](review/2026-06-25-decoder-decoupling-review-INTEGRATED_GLM_5.2_max.md) — 三方评审整合报告（GLM-5.2-max），综合 GLM/Opus/DeepSeek 三份评审。  

[2026-06-25-three-reviews-comparison_GLM_5.2_max](review/2026-06-25-three-reviews-comparison_GLM_5.2_max.md) — 三份评审对比与合理性判断（GLM-5.2-max）。  

[2026-06-26-h0-h4-experiment-results](review/2026-06-26-h0-h4-experiment-results.md) — H0-H4 根因实验结果补充：记录整合评审提出的 5 个廉价根因实验的实际执行结果。  

[2026-06-27-decoder-decoupling-design-review_sisyphus](review/2026-06-27-decoder-decoupling-design-review_sisyphus.md) — Decoder 解耦设计评估（Sisyphus），评估 `docs/superpowers/design/2026-06-25-decoder-decoupling-design.md` 的合理性、可靠性和可实施性，并纳入 H3 用户补充判断。  

[2026-06-27-workflow-docs-cleanup-note](review/2026-06-27-workflow-docs-cleanup-note.md) — 工作流文档清理记录：所有权契约、canonical 路径、变更文件清单。

---

## 约定

- ★ **canonical** 标记的文档是当前技术路线的主参考文档
- 所有评审文档为历史记录，保留原文不加修改（仅路径引用可更新）
- 正式技术决策应升格为 `openspec/` 文档
- 本索引必须覆盖 `docs/superpowers/**/*.md` 所有 Markdown 文件；故意排除的文件需在此注明原因
