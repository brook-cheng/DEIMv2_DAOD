# OpenSpec 制品索引

> 维护说明：当新增或删除 `openspec/changes/**` 或 `openspec/specs/**` 下的 Markdown 文件时，请更新本索引。
> 工作流制品（设计草稿、实施计划、评审）不在本索引范围内，见 `docs/superpowers/INDEX.md`。

---

## changes/ — 变更提案

### deimv2-obb
[proposal](changes/deimv2-obb/proposal.md) — DEIMv2-OBB 变更提案，通过 `box_mode` 闸门为 DEIMv2 增加定向边界框支持，使用 ADR 6-distribution DDF。  
[design](changes/deimv2-obb/design.md) — DEIMv2-OBB 架构设计，HBB/OBB 双路径，组件变更与向后兼容保证。  
[tasks](changes/deimv2-obb/tasks.md) — 实施任务分解。  
[specs/obb-detection](changes/deimv2-obb/specs/obb-detection/spec.md) — OBB 检测规范。  
[specs/obb-geometry](changes/deimv2-obb/specs/obb-geometry/spec.md) — OBB 几何操作规范。  
[specs/obb-evaluation](changes/deimv2-obb/specs/obb-evaluation/spec.md) — OBB 评估规范。  
[analysis/self-attention-query-diversity](changes/deimv2-obb/analysis/self-attention-query-diversity.md) — 自注意力 query 多样性分析，检查 300 个 query 进入自注意力前的区分度。

### deimv2-obb-eval-opt
[proposal](changes/deimv2-obb-eval-opt/proposal.md) — OBB 评估速度优化提案，用 `batch_probiou` 替代 `poly_iou` 实现 ~5000x 加速。

---

## specs/ — 独立规范

[2026-06-23-synthetic-ellipse-obb-dataset-design](specs/2026-06-23-synthetic-ellipse-obb-dataset-design.md) — 合成椭圆 OBB 数据集密度对照实验设计（H1 假设验证）。  
[2026-06-23-synthetic-ellipse-obb-implementation-plan](specs/2026-06-23-synthetic-ellipse-obb-implementation-plan.md) — 合成椭圆数据集实施计划（基于上述设计）。  
[2026-06-24-hungarian-matching-diagnosis-design](specs/2026-06-24-hungarian-matching-diagnosis-design.md) — 匈牙利匹配正确性诊断实验设计（Q1-Q3 验证）。

---

## 约定

- OpenSpec 文档记录**正式技术决策与规范**，不记录 agent 工作流过程
- 评审、脑暴、实施计划等 agent 制品见 [docs/superpowers/INDEX.md](../docs/superpowers/INDEX.md)
- 本索引必须覆盖 `openspec/changes/**/*.md` 和 `openspec/specs/**/*.md` 所有 Markdown 文件；故意排除的文件需在此注明原因
