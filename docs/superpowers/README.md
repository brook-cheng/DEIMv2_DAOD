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

## 历史说明：OpenSpec 已合并

本项目曾设 `openspec/` 作为"正式决策系统"，与 `docs/superpowers/`（工作区）双轨并行。现 `openspec/` 已全部合并至 `docs/superpowers/`：

- 原 `openspec/changes/<name>/proposal.md`、`design.md`、`specs/` → `docs/superpowers/specs/`（按日期+主题命名）
- 原 `openspec/changes/<name>/tasks.md` → `docs/superpowers/plans/`
- 原 `openspec/changes/<name>/analysis/` → `docs/superpowers/review/`
- 原 `openspec/specs/` 独立规范 → 按实际类型分发至 `specs/` 或 `plans/`

所有新文档统一置于 `docs/superpowers/` 对应子目录。

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

## 文档体系（单一）

`docs/superpowers/` 是本项目唯一的文档目录，承载所有制品类型：设计草稿（`design/`）、实施计划（`plans/`）、设计规范与变更提案（`specs/`）、评审与实验结果（`review/`）。

历史双轨制（`openspec/` + `docs/superpowers/`）已于 2026-08 合并为单一体系，详见 [docs/MAINTENANCE.md](../MAINTENANCE.md)。
