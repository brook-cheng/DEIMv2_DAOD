# OpenSpec — 正式规范与变更生命周期

## OpenSpec 是什么

OpenSpec 是 DEIMv2-OBB 项目的**正式规范管理系统**。这里的文档代表项目已接受或审阅中的技术决策、架构设计和变更提案。

## Belongs here（放在这里）

以下类型的文档属于 `openspec/`：

| 文档类型 | 目录 | 示例 |
|---------|------|------|
| 变更提案 | `changes/<name>/proposal.md` | `deimv2-obb/proposal.md` |
| 变更设计 | `changes/<name>/design.md` | `deimv2-obb/design.md` |
| 变更任务 | `changes/<name>/tasks.md` | `deimv2-obb/tasks.md` |
| 规范（spec） | `changes/<name>/specs/` | `obb-detection/spec.md` |
| 分析报告 | `changes/<name>/analysis/` | `self-attention-query-diversity.md` |
| 已接受规范 | `specs/` | `2026-06-24-hungarian-matching-diagnosis-design.md` |

## Does not belong here（不要放在这里）

以下类型的文档应该放在 `docs/superpowers/`：

- 脑暴/设计草稿 → `docs/superpowers/design/`
- 实施计划（agent 执行计划） → `docs/superpowers/plans/`
- 代码/计划评审 → `docs/superpowers/review/`
- 实验结果记录 → `docs/superpowers/review/`
- Agent 工作流制品和笔记 → `docs/superpowers/review/`

## 目录结构

```
openspec/
├── README.md          ← 你正在看
├── INDEX.md           ← 所有制品索引
├── changes/           ← 变更提案及其设计/任务/规范
│   ├── deimv2-obb/
│   │   ├── proposal.md
│   │   ├── design.md
│   │   ├── tasks.md
│   │   ├── specs/
│   │   │   ├── obb-detection/spec.md
│   │   │   ├── obb-geometry/spec.md
│   │   │   └── obb-evaluation/spec.md
│   │   └── analysis/
│   │       └── self-attention-query-diversity.md
│   └── deimv2-obb-eval-opt/
│       └── proposal.md
└── specs/             ← 独立已接受规范
    ├── 2026-06-23-synthetic-ellipse-obb-dataset-design.md
    ├── 2026-06-23-synthetic-ellipse-obb-implementation-plan.md
    └── 2026-06-24-hungarian-matching-diagnosis-design.md
```

## 工作流

1. **提案** → 在 `changes/<name>/proposal.md` 提出变更动机与范围
2. **设计** → 在 `changes/<name>/design.md` 编写架构设计
3. **任务** → 在 `changes/<name>/tasks.md` 列出实施任务
4. **规范** → 将正式规范放在 `changes/<name>/specs/` 或 `specs/`
5. **分析** → 将深度分析放在 `changes/<name>/analysis/`

评审和实验发现应记录在 `docs/superpowers/review/`，正式决策和接受的规范保留在 `openspec/`。
