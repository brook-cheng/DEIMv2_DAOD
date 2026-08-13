[English](./2026-08-13-obb-ablation-cleanup-design.md) | [简体中文](./2026-08-13-obb-ablation-cleanup-design_CN.md)

# OBB 消融清理设计

## 目标

把活跃的 OBB 算法面收敛到确实提升过效果的设置：

- 保留 `angle_rep=0` 与 `angle_rep=3`；
- 使 shifted 角度编码成为解码器唯一的角度编码；
- 移除 rep1、rep2、gate 融合、angle-first 预测、调整后偏移缩放与多角度锚点。

公开 OBB 契约保持为 `(cx, cy, w, h, theta)`，其中 `theta` 以弧度表示，取值 `[0, pi)`。

## 职责划分

生产代码、配置与文档的改动由用户负责。Sisyphus 负责 `test/` 下的全部改动：在对应的生产改动之前先写测试，审查每一批生产改动，并运行最终验证套件。

## 保留的架构

`angle_rep=0` 保留 5D 参考加 6D ADR 残差路径。其外接矩形与顶点偏移辅助函数虽然 rep2 也曾使用，仍然保留。

`angle_rep=3` 保留 5D 直接角度残差路径与 `decouple_angle_layers`。其 5D 几何辅助函数虽然 rep1 也曾使用，仍然保留。

shifted 编码在所有解码器私有边界处变为无条件启用：锚点初始化、去噪输入、可变形注意力参考解码、解码器细化、编码器辅助输出转换与最终公开角度转换。

`physical_rad_to_norm` 与 `norm_to_physical_rad` 不会全局删除。criterion 中保留的非周期角度损失只用比例归一化来缩放物理角度 L1 损失；这与解码器私有编码是两回事。

## 移除的架构

以下构造参数与行为是移除，而非固定为可配置默认值：

- `use_gate_fusion` 与 `engine/deim/gated_fusion.py`；
- `use_angle_first`；
- `angle_step` 与候选扩展；
- `offset_scale_source`，保留的几何固定为调整前（pre-adjustment）缩放；
- `decoder_angle_encoding`，shifted 行为固定于内部；
- 所有 rep1 与 rep2 专属的 head、锚点、去噪转换、辅助转换、诊断与配置文件。

移除之后，被接受的 OBB 表示取值恰好为 `0` 与 `3`。非法值在构造阶段直接失败。

## 配置覆盖范围

清理范围不止 `configs/custom_obb/dlzdt/ablation/`，还包括 `configs/custom_obb/synthetic_configs/` 下的直接与继承引用、应用预设、测试工具，以及任何提供被移除构造键的活跃 YAML。

通过 include 引用被废弃基础文件的依赖配置，要先于基础文件本身被移除或迁移。`synthetic_exp_020_dec.yml` 使用 `angle_rep: True`，数值上等价于 rep1，被视为过时的遗留配置。

过时的构造键会在 `engine/core/workspace.py:182` 处大声失败：第 176-178 行的可选签名过滤器已被注释掉，因此调用已注册的构造器时 Python 会抛出 `TypeError`。仓库中的配置契约测试把这一失败提前暴露在 CI 中，并报告确切的 YAML 区块与键名。

## 检查点策略

旧的比例归一化（proportional）OBB 检查点与清理前的 rep3 检查点是有意不兼容的。它们不得静默通过 `strict=False` 微调或应用适配器。

新的 OBB 检查点携带 `meta.obb_angle_contract = "shifted_v1"`。OBB 恢复训练与推理都要求该标记。OBB 微调遵循以下规则：

- 带 `shifted_v1` 标记的 OBB 检查点被接受；
- 无标记、且回归头可识别为 4D HBB 模型的检查点，作为 HBB 预训练被接受；
- 无标记或标记不同的 5D/6D OBB 检查点，以专用兼容性错误被拒绝。

这样就区分了合法的 HBB 到 OBB 初始化与语义含糊的旧 OBB 权重。

## 测试契约

Sisyphus 将在生产改动之前锁定以下场景：

1. Rep0 + shifted 能完成构造并跑通 CPU 前向传播，输出有限的 5D 公开 OBB，且 `theta` 保持在 `[0, pi)`。
2. Rep3 + shifted 在无 angle-first、无 gate 融合的条件下能完成构造，跑通 CPU 前向传播，并保持有限的解耦角度参考与 5D 公开 OBB。
3. 活跃的 OBB YAML 只包含被接受的构造键，且仅使用 rep0/rep3。
4. 被移除的构造参数与 rep1/rep2 一律被拒绝。
5. 旧 OBB 检查点明确报错；带标记的 shifted OBB 检查点与可识别的 HBB 预训练检查点在各自的预期加载路径中被接受。
6. Matcher、criterion、postprocessor、应用预测与物理角度边界保持不变。

仅 rep2 使用的测试与诊断工具，要在其生产路径被移除后才删除。共享几何、stable-atan2、matcher、损失与角度契约测试保留。

## 验证

每个移除切片都遵循 red、green、review、regression 的节奏推进。最终验证包括：

- 针对 rep0 与 rep3 的聚焦 CPU 测试；
- 对每个保留的 OBB 配置执行配置解析与模型构造；
- 明确的检查点策略探针；
- 完整的 OBB 与应用测试套件；
- 对改动的源文件运行诊断；
- 一次仓库搜索，证明被移除的符号不再出现在活跃的生产/配置代码中。

历史规范与已完成计划作为先前实验的记录，保持不变。
