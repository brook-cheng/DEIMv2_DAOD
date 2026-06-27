# Superpowers Workflow Docs — Agent 工作流制品

## 这是什么

`docs/superpowers/` 是 DEIMv2-OBB 项目的 **agent 工作流文档区**。这里的文档由 OpenCode/Sisyphus agent 在探索、设计、实施和评审过程中自动或半自动生成。

## Belongs here（放在这里）

以下类型的文档属于 `docs/superpowers/`：

| 文档类型 | 目录 | 示例 |
|---------|------|------|
| 脑暴/设计 | `design/` | `2026-06-25-decoder-decoupling-design.md`（decoder 解耦设计） |
| 实施计划 | `plans/` | `2026-06-25-decoder-decoupling-plan.md`（decoder 解耦实施计划） |
| 代码评审 | `review/` | `OBB_CODE_REVIEW.md`（OBB 代码审查） |
| 方案评审 | `review/` | `*decoder-decoupling-review*.md`（decoder 解耦方案评审） |
| 实验结果 | `review/` | `2026-06-26-h0-h4-experiment-results.md`（根因实验结果） |
| 流程笔记 | `review/` | `2026-06-27-workflow-docs-cleanup-note.md`（清理记录） |

## Does not belong here（不要放在这里）

以下类型的文档应该放在 `openspec/`：

- 正式变更提案 → `openspec/changes/<name>/proposal.md`
- 正式架构设计 → `openspec/changes/<name>/design.md`
- 正式任务列表 → `openspec/changes/<name>/tasks.md`
- 正式规范（spec） → `openspec/changes/<name>/specs/` 或 `openspec/specs/`
- 深度分析报告 → `openspec/changes/<name>/analysis/`

## 目录结构

```
docs/superpowers/
├── README.md          ← 你正在看
├── INDEX.md           ← 所有制品索引
├── design/            ← 脑暴/设计草稿
│   └── 2026-06-25-decoder-decoupling-design.md  ← decoder 解耦设计（canonical）
├── plans/             ← 实施计划（agent 执行用）
│   ├── 2026-06-25-decoder-decoupling-plan.md     ← decoder 解耦计划（canonical）
│   ├── 2026-06-24-hungarian-matching-diagnosis-plan.md
│   ├── 2026-06-24-cdn-input-viz.md
│   ├── 2026-06-26-h4-lqe-ablation-plan.md
│   └── 2026-01-17-model-correctness-tests.md
└── review/            ← 评审与实验结果
    ├── OBB_CODE_REVIEW.md
    ├── 2026-06-25-decoder-decoupling_review_deepseekv4_pro_max.md
    ├── 2026-06-25-decoder-decoupling-review_GLM_5.2_max.md
    ├── 2026-06-25-decoder-decoupling-review_Opus_4.8_max.md
    ├── 2026-06-25-decoder-decoupling-review-INTEGRATED_GLM_5.2_max.md
    ├── 2026-06-25-three-reviews-comparison_GLM_5.2_max.md
    ├── 2026-06-26-h0-h4-experiment-results.md
    └── 2026-06-27-workflow-docs-cleanup-note.md
```

## 与 OpenSpec 的关系

- **OpenSpec** 是"决策系统" — 记录已接受的正式决策和规范
- **Superpowers** 是"工作区" — 记录探索过程、脑暴输出、评审意见和实验记录

两条线不互相替代。一个正式决策必须有对应的 `openspec/` 文档，但它的探索过程（评审、计划迭代）可以放在 `docs/superpowers/`。
