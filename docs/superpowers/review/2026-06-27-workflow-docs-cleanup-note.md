# Workflow Docs Cleanup — 清理记录

> 日期：2026-06-27
> 原因：`b2db979 doc: 调整文档位置` 将 decoder-decoupling 的 design/plan 文档从 `openspec/` 顶层迁移到 `docs/superpowers/`，但未明确定义两套系统的所有权边界，导致部分评审文档仍引用旧路径，新文档不知道该放入哪个目录。

## 最终所有权契约

| 系统 | 目录 | 职责 |
|------|------|------|
| **OpenSpec** | `openspec/` | 正式规范与变更生命周期：提案、设计、任务、已接受规范、分析 |
| **Superpowers** | `docs/superpowers/` | Agent 工作流制品：脑暴设计、实施计划、代码/方案评审、实验结果 |

**规则：** OpenSpec 是"决策系统"，Superpowers 是"工作区"。正式决策必须有 `openspec/` 文档，但探索过程（评审、计划迭代）放在 `docs/superpowers/`。

## Canonical 路径

Decoder-decoupling 的当前 canonical 文档：

- 设计：`docs/superpowers/design/2026-06-25-decoder-decoupling-design.md`
- 计划：`docs/superpowers/plans/2026-06-25-decoder-decoupling-plan.md`

旧路径（已废弃，不应再引用）：

- `openspec/2026-06-25-decoder-decoupling-design.md`
- `openspec/plans/2026-06-25-decoder-decoupling-plan.md`

## 变更文件清单

### 新增
1. `openspec/README.md` — OpenSpec 所有权说明
2. `docs/superpowers/README.md` — Superpowers 所有权说明
3. `openspec/INDEX.md` — OpenSpec 制品索引（11 个 Markdown 文件）
4. `docs/superpowers/INDEX.md` — Superpowers 制品索引（14 个 Markdown 文件）
5. `docs/superpowers/review/2026-06-27-workflow-docs-cleanup-note.md` — 本文件

### 修改
6. `docs/superpowers/review/2026-06-25-decoder-decoupling_review_deepseekv4_pro_max.md` — 更新 stale 路径引用
7. `docs/superpowers/review/2026-06-25-decoder-decoupling-review_GLM_5.2_max.md` — 更新 stale 路径引用
8. `docs/superpowers/review/2026-06-25-decoder-decoupling-review_Opus_4.8_max.md` — 更新 stale 路径引用

### 未变更
- 所有历史评审、设计、计划文档的内容保持不变
- `openspec/changes/**` 和 `openspec/specs/**` 下的 OpenSpec 生命周期文档不变
- 模型代码、配置、测试脚本不变

## 验证

- `grep -r 'openspec/2026-06-25-decoder-decoupling' docs/ openspec/` → 零结果
- 所有 INDEX 文件引用的路径均指向存在的文件
- 两个 README 定义了明确的 Belongs/Does not belong 规则
