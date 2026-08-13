# OBB 消融清理实施计划

> **语言切换 / Language:** [English](2026-08-13-obb-ablation-cleanup.md) | [简体中文](2026-08-13-obb-ablation-cleanup_CN.md)

> **致代理执行者：** 必备子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实施本计划。步骤使用复选框（`- [ ]`）语法跟踪进度。

**目标：** 仅保留采用 shifted 解码器编码的 rep0/rep3，删除已被否决的 OBB 消融分支，并在不破坏 HBB→OBB 预训练的前提下拒绝语义不兼容的旧 OBB 检查点。

**架构：** 以可独立评审的切片、自叶到根删除被否决的行为。Sisyphus 先编写并运行测试；用户负责修改生产代码/配置/文档；Sisyphus 逐批评审与验证。物理 5D OBB API 保持不变。

**技术栈：** Python、PyTorch、pytest、YAML 配置注册表、CPU 模型冒烟测试。

## 全局约束

- 用户负责 `engine/`、`configs/`、现行文档与生产工具。
- Sisyphus 负责 `test/` 下的所有修改、生产代码评审与验证。
- 不得修改公开的 `(cx, cy, w, h, theta_rad)` 契约，或 matcher/postprocessor/评估语义。
- 即使被移除的表征曾共享它们，仍须保留 rep0 6D ADR 辅助函数与 rep3 5D 直接角度辅助函数。
- 保留 criterion 侧 `physical_rad_to_norm` 用于被保留的非周期损失；仅移除比例解码器选择。
- 在同一个已验证增量内删除构造函数键与所有活跃 YAML 出现位置。
- 不得改写历史已完成计划/规格。
- 除非用户明确要求，不得提交（commit）。

## 场景契约

| ID | 场景 | 二进制通过条件 | 自动化证据 | 真实表面证据 |
|---|---|---|---|---|
| S1 | rep0 shifted 前向 | 有限的 5 通道 `pred_boxes`，`0 <= theta < pi` | `test/test_obb_retained_representations.py::test_rep0_shifted_forward_contract` | CPU 驱动打印 rep0 形状与 theta 最小/最大值，随后打印 `PASS` |
| S2 | rep3 shifted 前向 | 有限的 5D 输出与有限的逐层引用；无融合键 | `test/test_obb_retained_representations.py::test_rep3_shifted_forward_contract` | CPU 驱动打印 `rep3 PASS`，状态字典融合键数量为 `0` |
| S3 | 过期配置拒绝 | 被保留的配置可正常构建；被移除的键与表征均不存在 | `test/test_obb_config_contract.py` | 配置加载器将每个被保留的配置打印为 `OK` |
| S4 | 检查点兼容性 | 已标记的 shifted OBB 被接受；未标记的旧 OBB 被拒绝；HBB 调优被接受 | `test/test_obb_checkpoint_contract.py` | 检查点探针打印三种预期结果 |
| S5 | 相邻公开 API | matcher、损失、postprocessor 与应用测试保持 5D 弧度输出 | 任务 11 中列出的既有聚焦测试套件 | 应用/CPU 冒烟输出保持 5D 物理 OBB |

## 任务 1：添加被保留表征测试

**文件：**
- 新建：`test/test_obb_retained_representations.py`
- 修改：`test/test_deimv2_obb_smoke.py`

**接口：**
- 消费当前 `DEIMTransformer` 测试工厂模式。
- 产出后续每个任务所使用的显式 rep0/rep3 shifted 契约。

- [ ] Sisyphus 在不改动生产代码的前提下，从既有冒烟测试构造中抽取一个小型 CPU 模型工厂。
- [ ] 添加 `test_rep0_shifted_forward_contract`：断言 5D 公开 boxes 有限且 theta 位于 `[0, pi)`。
- [ ] 添加 `test_rep3_shifted_forward_contract`：使用 `angle_rep=3`、`use_angle_first=False`、shifted 编码；断言 5D boxes 与引用均有限。
- [ ] 运行 `pytest -q test/test_obb_retained_representations.py`；两个被保留的前向测试必须在当前实现上通过，并建立表征证据。

## 任务 2：添加配置契约测试

**文件：**
- 新建：`test/test_obb_config_contract.py`
- 修改：`test/deim_app/test_legacy_parity.py`

**接口：**
- 消费仓库 YAML 加载器与已注册的构造函数模式。
- 产出过期键、被拒绝表征与断裂 include 链的清单。

- [ ] 对活跃的 `configs/custom_obb/**/*.yml` 与应用 OBB 预设做递归清单。
- [ ] 通过既有 YAML 加载器解析 `__include__`，而非重新实现合并行为。
- [ ] 对已注册的区段，将解析后的键与构造函数参数对比，并报告 `path:section.key`。
- [ ] 断言清理后活跃 OBB 配置中的 `angle_rep` 取值仅为 `0` 或 `3`。
- [ ] 断言被移除的键不存在：`offset_scale_source`、`use_gate_fusion`、`angle_step`、`use_angle_first`、`decoder_angle_encoding`。
- [ ] 更新 legacy-parity 预期，使固定行为不必继续以 YAML 键形式存在。
- [ ] 在生产清理之前运行新测试，并记录预期中的过期键失败。

## 任务 3：移除 offset-post 行为

**文件：**
- 用户修改：`engine/deim/dfine_utils.py`、`engine/deim/deim_criterion.py`、`engine/deim/deim_decoder.py`
- 用户修改/删除：baseline/preset offset 键、`abl_offset_post.yml`、合成 offset-post 配置
- Sisyphus 修改：`test/test_obb_adr_geometry.py`

- [ ] Sisyphus 添加签名断言，证明 offset 来源不再可配置，随后记录 RED。
- [ ] 用户移除 `offset_scale_source` 参数与校验；仅保留预调整几何。
- [ ] 用户在同一个批次中移除对应的 YAML 键与被否决的 post 配置。
- [ ] Sisyphus 仅移除 post/mismatch 测试；保留 pre、inverse、clamp、degenerate 与 stable-atan2 测试。
- [ ] 运行 `pytest -q test/test_obb_adr_geometry.py test/test_obb_config_contract.py` 并评审 diff。

## 任务 4：移除多角度锚点

**文件：**
- 用户修改：`engine/deim/deim_decoder.py`
- 用户删除/修改：`abl_mangle.yml`、合成多角度配置、baseline/preset `angle_step`
- Sisyphus 修改：`test/test_deimv2_obb_smoke.py`

- [ ] Sisyphus 为被移除的 `angle_step` 添加构造函数签名断言，随后记录 RED。
- [ ] 用户移除候选扩展、内存重复、构造函数状态与 YAML 键。
- [ ] Sisyphus 移除仅以多角度生成为契约的两个测试。
- [ ] 运行聚焦锚点、被保留表征与配置契约测试。

## 任务 5：移除 angle-first 与门控融合

**文件：**
- 用户修改：`engine/deim/deim_decoder.py`
- 用户删除：`engine/deim/gated_fusion.py`
- 用户删除/修改：AFP/fused 配置与 baseline/preset 键
- Sisyphus 修改：`test/test_deimv2_obb_smoke.py`、`test/test_obb_retained_representations.py`

- [ ] Sisyphus 将 rep3 矩阵改为 `use_angle_first=False`，并记录构造函数/状态字典 RED 测试。
- [ ] Sisyphus 添加 `test_rep3_has_no_gate_fusion_state_after_cleanup`；单独运行并记录 RED，因为当前 rep3 状态字典包含 `gate_fusions` 键。
- [ ] 用户移除 angle-first 查询构造、特殊首层流程、融合构造/调用与两个构造函数参数。
- [ ] 保留 `decouple_angle_layers`；rep3 依赖它。
- [ ] 仅在准备删除 AFP/fused 配置与 `gated_fusion.py` 时执行 `rg -n 'GatedSoftmaxFusion|gate_fusions' engine`；命令必须显示已无调用方。
- [ ] 运行被保留表征测试；状态字典必须包含零个 `gate_fusions` 键。

## 任务 6：移除 rep1

**文件：**
- 用户修改：`engine/deim/deim_decoder.py`
- 用户删除：`abl_rep1.yml`、rep1 合成基线与所有 include 依赖项
- Sisyphus 修改：`test/test_deimv2_obb_smoke.py`、`test/test_exp_020_obb_compare.py`、`test/` 下受影响的分析工具

- [ ] Sisyphus 添加 `angle_rep=1` 构造拒绝，并将被保留矩阵改为 `{0,3}`。
- [ ] 将误导性的 `angle_rep=True` 表征重指向显式 rep3（其真实目的是保留 ADR/引用行为）。
- [ ] 用户移除 rep1 头部维度与表征分支。
- [ ] 在删除其基座配置之前，先移除/迁移 rep1 include 依赖项。
- [ ] 运行冒烟、对比与配置契约测试。

## 任务 7：移除 rep2 及其诊断

**文件：**
- 用户修改：`engine/deim/deim_decoder.py`、`engine/deim/dfine_decoder.py`
- 用户删除：rep2 配置与合成依赖项
- Sisyphus 删除：`test/test_deimv2_obb_rep2_eval.py`、`test/test_rep2_nan_diagnostic.py`、`test/test_rep2_nan_failure_replay.py`、`test/tool_diagnose_rep2_nan.py`、`test/tool_replay_rep2_nan_failure.py`
- Sisyphus 修改：`test/` 下推理/调试/对比工具中的 rep2 条目

- [ ] Sisyphus 添加显式 rep2 构造拒绝并记录 RED。
- [ ] 用户移除 rep2 去噪转换、6D 引用/头部/锚点/辅助分支与 6D 注意力转换。
- [ ] Sisyphus 在删除 rep2 专属文件前先评审它们；保留共享几何与 stable-atan2 覆盖。
- [ ] 在删除其基座配置之前，先移除/迁移 rep2 include 依赖项。
- [ ] 运行被保留表征、ADR 几何、冒烟与配置契约测试套件。

## 任务 8：使 shifted 编码无条件生效

**文件：**
- 用户修改：`engine/deim/deim_decoder.py`、`engine/deim/denoising.py`、`engine/deim/dfine_decoder.py`
- 用户修改/删除：baseline/preset shifted 键与现已多余的 `abl_shifted.yml`
- Sisyphus 修改：`test/test_obb_angle_contract.py`、`test/test_deimv2_obb_smoke.py`、被保留表征测试

- [ ] Sisyphus 添加签名测试，证明 `decoder_angle_encoding` 不再被接受，并记录 RED。
- [ ] 用户仅在解码器精化、锚点、去噪、注意力、编码器辅助输出与公开转换点保留 shifted 转换。
- [ ] 用户移除编码参数、校验、传播与比例分支。
- [ ] 不得删除 criterion 侧 `physical_rad_to_norm`/`norm_to_physical_rad` 辅助函数。
- [ ] 删除 `abl_shifted.yml`，因为 shifted 现为基线行为而非消融。
- [ ] 运行角度契约、去噪、被保留表征、冒烟、criterion 损失与配置契约测试。

## 任务 9：简化表征守卫

**文件：**
- 用户修改：`engine/deim/deim_decoder.py`、相关注释/文档

- [ ] 将存留的多表征谓词替换为显式 rep0/rep3 分支。
- [ ] 添加构造函数 `ValueError`，列出可接受集合 `{0, 3}`。
- [ ] 从活跃源码/配置文档中移除描述 rep1/rep2 或被移除开关的注释。
- [ ] Sisyphus 对照 rep0/rep3 契约评审每个变更的分支，并重跑聚焦测试。

## 任务 10：添加 OBB 检查点兼容性契约

**文件：**
- 用户修改：`engine/solver/_solver.py`
- 用户修改：`tools/inference/torch_inf.py`、`tools/inference/torch_inf_vis.py`、`tools/visualization/fiftyone_vis.py`、`tools/deployment/export_onnx.py`
- Sisyphus 新建：`test/test_obb_checkpoint_contract.py`

**接口：**
- 产出 `OBB_ANGLE_CONTRACT = "shifted_v1"`。
- 产出类型化兼容性错误与一个辅助函数，用于分类已标记 OBB、旧 OBB 与 HBB 预训练状态字典。

- [ ] Sisyphus 编写 RED 测试：已标记 shifted OBB 被接受；错误标记被拒绝；未标记 5D/6D OBB 被拒绝；可识别的 4D HBB 仅允许用于调优。
- [ ] 用户在保存 OBB solver 状态时添加 `meta.obb_angle_contract`。
- [ ] 用户在 OBB 恢复与推理的 `load_state_dict` 之前检查标记。
- [ ] 用户检查调优输入：允许已标记 shifted OBB 检查点或未标记 4D HBB 检查点；在非严格 `_matched_state` 加载之前拒绝有歧义的旧 OBB 检查点。
- [ ] 确保 HBB 恢复/推理行为不变。
- [ ] 运行检查点测试并做一次内存内状态往返。

## 任务 11：修复过期测试并完成配置清理

**文件：**
- Sisyphus 修改：`test/test_kendall.py`、`test/test_model_correctness.py`、`test/test_model_output.py`、`test/test_obb_loss_integration.py`、`test/test_obb_transforms.py`、`test/test_early_stopping_configs.py`
- 用户修改/删除：任务 2 发现的其余活跃 OBB 配置

- [ ] 将引用已删除 `configs/custom_obb/deimv2_obb_sp.yml` 的位置替换为适合各测试的被保留权威配置。
- [ ] 在不发明兼容性别名的前提下，解决缺失的 `sp_fz_rep0_nloss_amp.yml` 预期。
- [ ] 删除或迁移 `synthetic_exp_020_dec.yml`，因为 `angle_rep: True` 属于 rep1。
- [ ] 运行 `pytest -q test/test_obb_config_contract.py`，直至每个被保留配置都能在无过期键的情况下解析并构建。
- [ ] 运行相邻 matcher、criterion、postprocessor、transforms、应用与 CLI 测试。

## 任务 12：评审与全面验证

**文件：** 所有变更文件。

- [ ] Sisyphus 通读每个生产/配置 diff，并对照已批准的设计进行检查。
- [ ] 对每个变更的 Python 文件运行诊断。
- [ ] 先运行聚焦套件，再运行 `pytest -q test`（或仓库文档中记载的完整测试命令）。
- [ ] 为 S1/S2 运行 CPU 驱动，并捕获形状、theta 最小/最大值、有限性与融合键数量。
- [ ] 加载并构建每个被保留的 OBB 配置；捕获每行 `OK <path>`。
- [ ] 运行检查点探针：已标记 shifted OBB、旧 OBB 拒绝与 HBB 调优接受。
- [ ] 运行 `rg -n 'use_gate_fusion|use_angle_first|angle_step|offset_scale_source|decoder_angle_encoding|GatedSoftmaxFusion' engine configs/custom_obb` 并对每个残留匹配分类；活跃实现/配置匹配属于失败。
- [ ] 在不暂存或不提交的前提下运行 `git diff --check`。
- [ ] 在交接之前调用实施后评审工作流，因为本次清理改动超过三个文件并移除了检查点兼容状态。

## 执行顺序

任务严格按序执行：`1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12`。测试工作始终先于其所规定的生产变更。构造函数参数删除与对应的活跃 YAML 清理须保持在同一个已验证增量内。

若之后要求提交，任务 3-10 各使用一个原子提交，测试与生产变更放在一起。撰写提交信息前先研究仓库历史；规划期间不得创建提交。

---

# 附录 A：生产代码参考操作（任务 3-10，用户侧）

> 行号取自 2026-08-13 的代码快照。由于逐任务修改会使行号漂移，**动手前先按内容定位（`rg -n` 或搜索关键行），再参照下面的「当前 → 目标」对照执行**。每段目标代码均已考虑与后续任务的衔接。

## A3 任务 3 参考：移除 offset-post

### A3.1 `engine/deim/dfine_utils.py` — `distance2bbox_obb`（194-250 行）

当前签名（195 行）：

```python
def distance2bbox_obb(
    points, distance, reg_scale, offset_scale_source: str = "pre", layer_idx=0
):
```

目标签名：

```python
def distance2bbox_obb(points, distance, reg_scale, layer_idx=0):
```

删除校验段（214-218 行）：

```python
    if offset_scale_source not in ("pre", "post"):
        raise ValueError(
            f"offset_scale_source must be 'pre' or 'post', "
            f"got {offset_scale_source!r}"
        )
```

当前偏移缩放选择（229-235 行）：

```python
        offset_scale_wh = (
            ext_rect_cxcywh[..., 2:]
            if offset_scale_source == "pre"
            else ext_adj_cxcywh[..., 2:]
        )
```

目标（固定为 pre）：

```python
        offset_scale_wh = ext_rect_cxcywh[..., 2:]
```

### A3.2 `engine/deim/dfine_utils.py` — `bbox2distance_obb`（253-316 行）

当前签名（253-262 行）中删除 `offset_scale_source: str = "pre",`；目标：

```python
def bbox2distance_obb(
    points,
    bbox,
    reg_max,
    reg_scale,
    up,
    eps=0.1,
    obbox_rep_dim=6,
):
```

删除校验段（284-288 行，与 A3.1 相同的 `if offset_scale_source not in ... raise`）。

当前偏移缩放选择（300-304 行）：

```python
        offset_scale_wh = (
            rect_cxcywh_pred[..., 2:]
            if offset_scale_source == "pre"
            else rect_cxcywh_gt[..., 2:]
        )
```

目标（固定为 pre）：

```python
        offset_scale_wh = rect_cxcywh_pred[..., 2:]
```

### A3.3 `engine/deim/deim_criterion.py`

- 67 行签名：删除 `offset_scale_source="pre",`。
- 105 行：删除 `self.offset_scale_source = offset_scale_source`。
- 426-434 与 445-453 两处 `bbox2distance_obb(...)` 调用：删除参数行 `offset_scale_source=self.offset_scale_source,`。

### A3.4 `engine/deim/deim_decoder.py`

- `TransformerDecoder.__init__`：184 行删除 `offset_scale_source="pre",`；198 行删除 `self.offset_scale_source = offset_scale_source`。
- `DEIMTransformer.__init__`：603 行删除 `offset_scale_source="pre",`；631 行删除 `self.offset_scale_source = offset_scale_source`。
- 728 行：删除 `offset_scale_source=self.offset_scale_source,`。
- 解码循环内两处 `distance2bbox_obb(...)` 调用（509-514、529-534）：删除末尾的 `offset_scale_source=self.offset_scale_source,` 行。当前（509-514 行）：

```python
                    inter_ref_bbox = distance2bbox_obb(
                        ref_phys,
                        distance,
                        reg_scale,
                        offset_scale_source=self.offset_scale_source,
                    )
```

目标：

```python
                    inter_ref_bbox = distance2bbox_obb(
                        ref_phys,
                        distance,
                        reg_scale,
                    )
```

529-534 行（proportional 分支）同样删除该 kwarg；该分支整体将在任务 8 删除。

### A3.5 YAML

删除保留配置中的 `offset_scale_source` 键：

- `configs/custom_obb/dlzdt/sp_fz_common.yml:193`（DEIMTransformer 节）与 `:272`（DEIMCriterion 节）。
- `configs/app/presets/deimv2_dinov3_sp_obb.yml:64` 与 `:104`。
- `configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep0_offset_per.yml:259,305`、`synthetic_exp_020_anrep3_offset_per.yml:258,303`、`synthetic_exp_020_undec_offset_per.yml:257,302`。

删除配置：`dlzdt/ablation/abl_offset_post.yml`、`synthetic_exp_020_undec_offset_post.yml`、`synthetic_exp_020_anrep2_offset_post.yml`（anrep2 配置本属任务 7，此处提前删除以避免残留 post 键）。

**验收**：`rg -n 'offset_scale_source' engine configs` 只剩注释或历史文档；`test_obb_adr_geometry.py` 保留 pre/inverse/clamp/degenerate/stable-atan2 测试通过。

## A4 任务 4 参考：移除多角度锚点

### A4.1 `engine/deim/deim_decoder.py` — 构造参数

- 605 行：删除 `angle_step=0.0,`。
- 633 行：删除 `self.angle_step = angle_step`。

### A4.2 `_generate_anchors` OBB 分支（1039-1105 行）

当前结构：

```python
        elif self.box_mode == "obb":
            if self.angle_rep == 2:
                ...  # rep2 6D 锚点（任务 7 删除，本次不动）
            else:
                for lvl, (h, w) in enumerate(spatial_shapes):
                    ...
                    if self.angle_step > 0:
                        n_angles = int(1.0 / self.angle_step)
                        ...  # 1075-1091 多角度候选展开
                    else:
                        default_r = (
                            0.5 if self.decoder_angle_encoding == "shifted" else 0.25
                        )
                        r = default_r * torch.ones(...)
                        lvl_anchors = torch.concat([grid_xy, wh, r], dim=-1).reshape(
                            -1, h * w, self._num_box_dof
                        )
```

目标（删除 `if self.angle_step > 0:` 分支，else 体直接提升；`default_r` 三元在任务 8 简化为 `0.5`）：

```python
        elif self.box_mode == "obb":
            if self.angle_rep == 2:
                ...  # 保持不变
            else:
                for lvl, (h, w) in enumerate(spatial_shapes):
                    ...
                    default_r = (
                        0.5 if self.decoder_angle_encoding == "shifted" else 0.25
                    )
                    r = default_r * torch.ones(
                        *grid_xy.shape[:-1],
                        1,
                        dtype=grid_xy.dtype,
                        device=grid_xy.device,
                    )
                    lvl_anchors = torch.concat([grid_xy, wh, r], dim=-1).reshape(
                        -1, h * w, self._num_box_dof
                    )
```

### A4.3 `_get_decoder_input`（1173-1180 行）

删除整个多角度 memory 复制块：

```python
        # Multi-angle anchors (angle_step > 0) expand the candidate pool to
        # (num_spatial_positions * n_angles). Replicate each spatial memory
        # token n_angles times so it aligns with the anchor layout
        # (position-major, angle-minor within each level). This only affects
        # query selection; the decoder still receives the original memory.
        if self.box_mode == "obb" and self.angle_rep != 2 and self.angle_step > 0:
            n_angles = int(1.0 / self.angle_step)
            memory = memory.repeat_interleave(n_angles, dim=1)
```

### A4.4 YAML

删除 `angle_step` 键：`sp_fz_common.yml:203`、`app/presets/deimv2_dinov3_sp_obb.yml:66`。删除配置：`dlzdt/ablation/abl_mangle.yml`、`synthetic_configs/ablation/syn_ablation_mangle.yml`。

## A5 任务 5 参考：移除 angle-first 与门控融合

### A5.1 `TransformerDecoder.__init__`（168-239 行）

- 185 行删除 `use_gate_fusion=False,`；186 行删除 `use_angle_first=False,`。
- 199-200 行删除两个 `self.xxx = xxx` 存储。
- 221-239 行，当前：

```python
        if self.angle_rep != 0 and self.angle_rep != 1:
            from .gated_fusion import GatedSoftmaxFusion

            decouple_layer_template = self.layers[-1]
            self.decouple_angle_layers = nn.ModuleList(
                [
                    copy.deepcopy(decouple_layer_template)
                    for _ in range(self.num_decouple_layers)
                ]
            )

            self.gate_fusions = nn.ModuleList(
                [
                    GatedSoftmaxFusion(
                        d_model=hidden_dim, n_sources=2, hidden_dim=hidden_dim
                    )
                    for _ in range(self.num_decouple_layers - 1)
                ]
            )
```

目标（保留 `decouple_angle_layers`，删除 lazy import 与 `gate_fusions`；guard 在任务 9 简化为 `== 3`）：

```python
        if self.angle_rep != 0 and self.angle_rep != 1:
            decouple_layer_template = self.layers[-1]
            self.decouple_angle_layers = nn.ModuleList(
                [
                    copy.deepcopy(decouple_layer_template)
                    for _ in range(self.num_decouple_layers)
                ]
            )
```

### A5.2 `TransformerDecoder.forward` 查询构造（304-331 行）

当前 305-328 行含 `use_angle_first` 三分支。目标（rep2/3 分支中删除 319-328 的条件判断，保留 else 内容）：

```python
        if self.box_mode == "obb":
            if self.angle_rep == 0 or self.angle_rep == 1:
                ref_points_detach = F.sigmoid(ref_points_unact)
                query_pos_embed = query_pos_head(ref_points_detach).clamp(
                    min=-10, max=10
                )
            elif self.angle_rep == 2 or self.angle_rep == 3:
                dec_angle_output = target
                dec_angle_output_detach = 0
                dec_angle_pred_corners_undetach = 0
                ref_points_detach = F.sigmoid(ref_points_unact[..., :4])
                ref_dec_angle_detach = F.sigmoid(ref_points_unact)
                query_dec_angle_embed = query_angle_head(
                    F.sigmoid(ref_points_unact[..., 4:])
                )
                query_pos_embed = query_pos_head(ref_points_detach).clamp(
                    min=-10, max=10
                )
```

### A5.3 层循环内 angle-first 块（335-377 行）

删除整个 `if (self.use_angle_first and ...)` 块（335-375），`else` 体无条件保留：

```python
            ref_points_input = ref_points_detach.unsqueeze(2)
```

注意：被删块内的 `decouple_angle_layers` 调用只是 angle-first 路径的调用；标准路径在 A5.4 保留，`decouple_angle_layers` 模块本身（A5.1）不受影响。

### A5.4 层循环内 OBB 细化（420-492 行）

当前是 `if self.angle_rep == 2 or self.angle_rep == 3:` 内套 `if self.use_angle_first:`（422-442）与 `else:`（443-492）。目标：删除 `if self.use_angle_first:` 分支（含 rep2 子分支 424-434、rep3 子分支 435-438），else 体提升为直接正文；再删除融合调用 484-487：

```python
            if self.box_mode == "obb":
                if self.angle_rep == 2 or self.angle_rep == 3:
                    ref_dec_angle_input = ref_dec_angle_detach.unsqueeze(2)
                    dec_angle_output = self.decouple_angle_layers[layer_idx](
                        dec_angle_output,
                        ref_dec_angle_input,
                        value,
                        spatial_shapes,
                        attn_mask,
                        query_dec_angle_embed,
                    )
                    if layer_idx == 0:
                        dec_angle_initial = torch.sigmoid(
                            pre_angle_head(dec_angle_output)
                            + inverse_sigmoid(ref_dec_angle_detach)[..., 4:]
                        )
                        if self.angle_rep == 2:
                            ...  # rep2 子分支（任务 7 删除，本次保留）
                        elif self.angle_rep == 3:
                            pre_bboxes = torch.concat(
                                [pre_bboxes, dec_angle_initial], dim=-1
                            )
                        ref_points_initial = pre_bboxes.detach()
                    dec_angle_pred_corners = (
                        dec_angle_head[layer_idx](
                            dec_angle_output + dec_angle_output_detach
                        )
                        + dec_angle_pred_corners_undetach
                    )
                    dec_angle_output_detach = dec_angle_output.detach()
                    dec_angle_pred_corners_undetach = dec_angle_pred_corners
                    pred_corners = torch.concat(
                        [pred_corners, dec_angle_pred_corners], dim=-1
                    )
```

被删除的融合调用（484-487 行）：

```python
                        if self.use_gate_fusion and layer_idx < len(self.gate_fusions):
                            dec_angle_output = self.gate_fusions[layer_idx](
                                [output, dec_angle_output], query=dec_angle_output
                            )
```

### A5.5 构造期守卫（645-650 行）

删除整个 `if use_angle_first and angle_rep == 2: raise ValueError(...)` 段。

### A5.6 head 维度（784-793 行）

rep3 分支（789-793 行）：

```python
            elif self.angle_rep == 3:
                pre_bbox_head_out_dim = 4
                num_query_pos_in = 5 if self.use_angle_first else 4
                num_reg_dist_xywh = 4
                num_angle_describer = 1
```

目标：`num_query_pos_in = 4`（其余不变）。

### A5.7 删除 `engine/deim/gated_fusion.py`

前提：`rg -n 'GatedSoftmaxFusion|gate_fusions' engine` 输出为空。

### A5.8 YAML

删除 `use_gate_fusion` 键（`sp_fz_common.yml:198`、`app/presets:65`）与 `use_angle_first` 键（`sp_fz_common.yml:208`、`app/presets:67`）。删除配置：`dlzdt/ablation/abl_rep2_fused.yml`、`abl_rep3_fused.yml`、`abl_rep3_afp.yml`、`synthetic_configs/ablation/syn_ablation_fused.yml`、`syn_ablation_afp.yml`。

**验收**：`test_rep3_has_no_gate_fusion_state_after_cleanup` 转绿；rep3 状态字典零个 `gate_fusions` 键。

## A6 任务 6 参考：移除 rep1

### A6.1 dof/reg_dist 分支（654-666 行）

当前：

```python
        if self.box_mode == "obb":
            if self.angle_rep == 0:
                self._num_box_dof = 5  # (cx,cy,w,h,θ)
                self.num_reg_dist = 6  # (α,β,γ,δ,ε,η)
            elif self.angle_rep == 1:
                self._num_box_dof = 5  # (cx,cy,w,h,θ)
                self.num_reg_dist = 5  # (α,β,γ,δ,deta_θ)
            elif self.angle_rep == 2:
                self._num_box_dof = 6  # (cx,cy,w,h,ε,η)
                self.num_reg_dist = 6  # (α,β,γ,δ,ε,η)
            elif self.angle_rep == 3:
                self._num_box_dof = 5  # (cx,cy,w,h,θ)
                self.num_reg_dist = 5  # (α,β,γ,δ,deta_θ)
```

目标（删除 658-660 的 rep1 分支；rep2 分支任务 7 删除）：

```python
        if self.box_mode == "obb":
            if self.angle_rep == 0:
                self._num_box_dof = 5  # (cx,cy,w,h,θ)
                self.num_reg_dist = 6  # (α,β,γ,δ,ε,η)
            elif self.angle_rep == 2:
                self._num_box_dof = 6  # (cx,cy,w,h,ε,η)
                self.num_reg_dist = 6  # (α,β,γ,δ,ε,η)
            elif self.angle_rep == 3:
                self._num_box_dof = 5  # (cx,cy,w,h,θ)
                self.num_reg_dist = 5  # (α,β,γ,δ,deta_θ)
```

### A6.2 head 维度（775-793 行）

删除 rep1 分支（780-783 行）：

```python
            elif self.angle_rep == 1:
                pre_bbox_head_out_dim = 5  # (cx,cy,w,h,θ)
                num_query_pos_in = 5
                num_reg_dist_xywh = 5  # (α,β,γ,δ,deta_theta)
```

### A6.3 YAML

删除 `dlzdt/ablation/abl_rep1.yml`、`synthetic_exp_020_anrep1_offset_per.yml` 及其 include 依赖（先处理依赖再删基座）：

- `synthetic_exp_020_loss_kld.yml`（include anrep1_offset_per）
- `synthetic_exp_020_loss_prob_kld.yml`
- `synthetic_exp_020_loss_prob_angle_kld.yml`
- `synthetic_configs/ablation/syn_ablation_loss_prob_kld.yml`
- `synthetic_configs/provenance/synthetic_exp_020_loss_kld.completed.yml`
- `synthetic_configs/provenance/synthetic_exp_020_loss_prob_kld.completed.yml`

## A7 任务 7 参考：移除 rep2 及其诊断

### A7.1 删除 helper（47-62 行）

删除整个 `_obb_denoising_unact_to_rep2_unact` 函数。

### A7.2 分支收敛

- 654-666 行：删除 rep2 分支（661-663）。
- 775-793 行：删除 rep2 分支（784-788，含 `num_angle_describer = 2`）。
- 310 行：`elif self.angle_rep == 2 or self.angle_rep == 3:` → `elif self.angle_rep == 3:`。
- 390、421 行：`== 2 or == 3` → `== 3`。
- A5.4 目标代码中的 rep2 子分支（`if self.angle_rep == 2: ...`）删除，仅留 `elif self.angle_rep == 3:` 分支：

```python
                    if layer_idx == 0:
                        dec_angle_initial = torch.sigmoid(
                            pre_angle_head(dec_angle_output)
                            + inverse_sigmoid(ref_dec_angle_detach)[..., 4:]
                        )
                        if self.angle_rep == 3:
                            pre_bboxes = torch.concat(
                                [pre_bboxes, dec_angle_initial], dim=-1
                            )
                        ref_points_initial = pre_bboxes.detach()
```

（也可保留 `if self.angle_rep == 3:` 显式形式，执行时二选一保持风格一致。）

### A7.3 锚点生成（1039-1063 行）

删除 OBB 分支里的 `if self.angle_rep == 2:` 块（1040-1063），else 体直接作为 OBB 分支内容（即 A4.2 的目标代码去掉外层 if）。

### A7.4 enc aux 角度转换（1198-1220 行）

删除外层 `if self.angle_rep != 2:` 判断与 `else: enc_topk_bboxes = enc_topk_bboxes`（1218-1220），内层编码判断直接提升（任务 8 再删比例分支）。

### A7.5 denoising 拼接（1232-1241 行）

当前：

```python
        if denoising_bbox_unact is not None:
            if self.angle_rep != 2:
                enc_topk_bbox_unact = torch.concat(
                    [denoising_bbox_unact, enc_topk_bbox_unact], dim=1
                )
            else:
                dn_bbox_unact = _obb_denoising_unact_to_rep2_unact(denoising_bbox_unact)
                enc_topk_bbox_unact = torch.concat(
                    [dn_bbox_unact, enc_topk_bbox_unact], dim=1
                )
            content = torch.concat([denoising_logits, content], dim=1)
```

目标：

```python
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat(
                [denoising_bbox_unact, enc_topk_bbox_unact], dim=1
            )
            content = torch.concat([denoising_logits, content], dim=1)
```

### A7.6 enc aux 输出转换（1415-1422 行）

删除整个块：

```python
            if self.angle_rep == 2:
                enc_topk_bboxes_list = [
                    external_xywh_rect_to_oriented_box(
                        enc_topk_bboxes[..., :4],
                        enc_topk_bboxes[..., 4:],
                    )
                    for enc_topk_bboxes in enc_topk_bboxes_list
                ]
```

### A7.7 `engine/deim/dfine_decoder.py` 注意力 6D 分支（173-187 行）

当前：

```python
        elif reference_points.shape[-1] == 5 or reference_points.shape[-1] == 6:
            if reference_points.shape[-1] == 6:
                reference_points = external_xywh_rect_to_oriented_box(
                    reference_points[..., :4], reference_points[..., 4:]
                )
                angle = reference_points[..., 4:5]
            else:
                if self.angle_encoding == "shifted":
                    angle = shifted_norm_to_physical_rad(reference_points[..., 4:5])
                else:
                    angle = reference_points[..., 4:5] * torch.pi
```

目标：

```python
        elif reference_points.shape[-1] == 5:
            if self.angle_encoding == "shifted":
                angle = shifted_norm_to_physical_rad(reference_points[..., 4:5])
            else:
                angle = reference_points[..., 4:5] * torch.pi
```

若第 26 行 `from .obb_geometry import external_xywh_rect_to_oriented_box` 无其他调用，一并删除。

### A7.8 YAML

删除 `dlzdt/ablation/abl_rep2.yml`、`synthetic_exp_020_anrep2_offset_per.yml`、`anrep2_bn0.yml`、`anrep2_dn0.yml`、`synthetic_configs/ablation/syn_ablation_noangle.yml`（`anrep2_offset_post.yml` 已在 A3.5 删除）。

## A8 任务 8 参考：shifted 编码无条件化

### A8.1 解码循环（498-538 行）

当前为 `if self.decoder_angle_encoding == "shifted":` 与 `else:` 双分支。目标（只留 shifted 体，删除 523-538 的 else）：

```python
            elif self.box_mode == "obb":
                # [0,1)→[-pi/4,3*pi/4)
                ref_phys = torch.cat(
                    [
                        ref_points_initial[..., :4],
                        shifted_norm_to_physical_rad(ref_points_initial[..., 4:5]),
                    ],
                    dim=-1,
                )
                distance = integral(pred_corners, project)
                inter_ref_bbox = distance2bbox_obb(
                    ref_phys,
                    distance,
                    reg_scale,
                )
                # [-pi/4,3*pi/4)→[0,1)
                inter_ref_bbox = torch.cat(
                    [
                        inter_ref_bbox[..., :4],
                        physical_rad_to_shifted_norm(inter_ref_bbox[..., 4:]),
                    ],
                    dim=-1,
                )
```

### A8.2 锚点 default_r（1093-1095 行）

`default_r = (0.5 if self.decoder_angle_encoding == "shifted" else 0.25)` → `default_r = 0.5`。

### A8.3 enc aux 转换（1199-1217 行，任务 7 后为内层判断）

当前：

```python
                if self.decoder_angle_encoding == "shifted":
                    # 内部 θ_shift 还原为物理角 [0, π)
                    enc_topk_bboxes = torch.cat(
                        [
                            enc_topk_bboxes[..., :4],
                            shifted_norm_to_physical_rad(enc_topk_bboxes[..., 4:]),
                        ],
                        dim=-1,
                    )
                else:
                    # 角度量纲 [0,1]->[0, pi)
                    enc_topk_bboxes = torch.cat(
                        [
                            enc_topk_bboxes[..., :4],
                            norm_to_physical_rad(enc_topk_bboxes[..., 4:]),
                        ],
                        dim=-1,
                    )
```

目标（删除 else 分支）：

```python
                # 内部 θ_shift 还原为物理角 [0, π)
                enc_topk_bboxes = torch.cat(
                    [
                        enc_topk_bboxes[..., :4],
                        shifted_norm_to_physical_rad(enc_topk_bboxes[..., 4:]),
                    ],
                    dim=-1,
                )
```

### A8.4 公开输出转换（1346-1350 行）

当前：

```python
        if self.box_mode == "obb":
            if self.decoder_angle_encoding == "shifted":
                theta_decode = shifted_norm_to_physical_rad
            else:
                theta_decode = norm_to_physical_rad
```

目标：

```python
        if self.box_mode == "obb":
            theta_decode = shifted_norm_to_physical_rad
```

（后续 1351-1371 的 `theta_decode(out_bboxes[..., 4:])` 等保持不变。）

### A8.5 参数与开关删除（`engine/deim/deim_decoder.py`）

- 44 行：删除 `_VALID_DECODER_ANGLE_ENCODINGS = ("proportional", "shifted")`。
- `TransformerDecoderLayer.__init__`：79 行删除 `angle_encoding="proportional",`；96-103 行改为：

```python
        self.cross_attn = MSDeformableAttention(
            d_model,
            n_head,
            n_levels,
            n_points,
            method=cross_attn_method,
        )
```

- `TransformerDecoder.__init__`：187 行删除 `angle_encoding="proportional",`；202 行删除 `self.decoder_angle_encoding = angle_encoding`。
- `DEIMTransformer.__init__`：607 行删除 `decoder_angle_encoding="proportional",`；635-642 行整段删除（校验 635-639、rep2 强制三元 640-642）。
- 696、710 行：删除 `angle_encoding=self.decoder_angle_encoding,`。
- 731 行：删除 `angle_encoding=self.decoder_angle_encoding,`。
- 1301 行：删除 `angle_encoding=self.decoder_angle_encoding,`。

### A8.6 `engine/deim/denoising.py`（21 行、113-121 行）

- 21 行：删除 `angle_encoding="proportional",`。
- 当前 113-121 行：

```python
    elif box_mode == "obb":
        # [0,pi) → decoder 私有编码 [0,1)
        if angle_encoding == "shifted":
            input_query_bbox[..., 4] = physical_rad_to_shifted_norm(
                input_query_bbox[..., 4]
            )
        else:
            input_query_bbox[..., 4] = physical_rad_to_norm(input_query_bbox[..., 4])
        input_query_bbox = torch.cat([noise_spatial, input_query_bbox[..., 4:]], dim=-1)
```

目标：

```python
    elif box_mode == "obb":
        # [0,pi) → decoder 私有 shifted 编码 [0,1)
        input_query_bbox[..., 4] = physical_rad_to_shifted_norm(
            input_query_bbox[..., 4]
        )
        input_query_bbox = torch.cat([noise_spatial, input_query_bbox[..., 4:]], dim=-1)
```

- 第 9 行 import：若 `physical_rad_to_norm` 在本文件已无调用，从 import 移除（保留 `physical_rad_to_shifted_norm`）。

### A8.7 `engine/deim/dfine_decoder.py`

- 58 行：删除 `angle_encoding="proportional",`。
- 100 行：删除 `self.angle_encoding = angle_encoding`。
- 184-187 行（任务 7 后）：

```python
                if self.angle_encoding == "shifted":
                    angle = shifted_norm_to_physical_rad(reference_points[..., 4:5])
                else:
                    angle = reference_points[..., 4:5] * torch.pi
```

目标：

```python
                angle = shifted_norm_to_physical_rad(reference_points[..., 4:5])
```

### A8.8 保留不动

`engine/deim/obb_angle_contract.py` 整文件保留：`physical_rad_to_norm`/`norm_to_physical_rad` 仍被 criterion 非周期 L1 损失（`deim_criterion.py:382,389`）使用，属损失侧归一化，与解码器私有编码无关。

### A8.9 YAML

删除 `decoder_angle_encoding` 键：`sp_fz_common.yml:215`、`app/presets/deimv2_dinov3_sp_obb.yml:68`、`synthetic_exp_020_anrep0_offset_per.yml:260`。删除 `dlzdt/ablation/abl_shifted.yml`（shifted 已成基线行为）。

## A9 任务 9 参考：守卫简化与非法值拒绝

### A9.1 guard 简化清单（此时仅剩 rep0/rep3）

- `deim_decoder.py:305`：`if self.angle_rep == 0 or self.angle_rep == 1:` → `if self.angle_rep == 0:`。
- `deim_decoder.py:310`：`elif self.angle_rep == 2 or self.angle_rep == 3:` → `elif self.angle_rep == 3:`。
- `deim_decoder.py:390,421`：`== 2 or == 3` → `== 3`。
- `deim_decoder.py:551`：`if self.angle_rep != 0 and self.angle_rep != 1:` → `if self.angle_rep == 3:`。
- `deim_decoder.py:832,925`：`!= 0 and != 1` → `== 3`。
- `deim_decoder.py:221`（TransformerDecoder）：`!= 0 and != 1` → `== 3`。

### A9.2 非法值构造期拒绝（`DEIMTransformer.__init__`，dof 分支之前）

在 `if self.box_mode == "obb":` 的 if/elif 链之前插入：

```python
        if self.box_mode == "obb" and self.angle_rep not in (0, 3):
            raise ValueError(
                f"angle_rep must be 0 or 3 for box_mode='obb', got {self.angle_rep!r}"
            )
```

### A9.3 注释清理

删除活跃源码/配置中描述 rep1/rep2、`use_gate_fusion`、`use_angle_first`、`angle_step`、`offset_scale_source`、`decoder_angle_encoding` 的注释（历史计划/规格不动）。

## A10 任务 10 参考：OBB 检查点兼容契约

### A10.1 `engine/solver/_solver.py`

在 `remove_module_prefix` 之后新增常量、异常与分类/校验辅助：

```python
OBB_ANGLE_CONTRACT = "shifted_v1"


class CheckpointIncompatibleError(RuntimeError):
    """Raised when a checkpoint cannot be loaded under the current OBB contract."""


def classify_checkpoint_kind(state: Dict) -> str:
    """按编码器 box head 输出维度区分 'hbb' / 'obb' / 'unknown'。

    ``enc_bbox_head.layers.2.bias`` 长度等于 ``_num_box_dof``:
    HBB=4，rep0/rep3=5，rep2=6。执行时先打印一次实际 state_dict 键名，
    若封装不同（DataParallel/EMA）则取对应前缀下的同一键。
    """
    try:
        model_state = state.get("model", {})
        if "module" in model_state:
            model_state = model_state["module"]
        dof = model_state["enc_bbox_head.layers.2.bias"].shape[0]
    except (KeyError, AttributeError):
        return "unknown"
    return "hbb" if dof == 4 else ("obb" if dof in (5, 6) else "unknown")


def assert_checkpoint_compat(state: Dict, expected: str = OBB_ANGLE_CONTRACT) -> None:
    """OBB 检查点必须携带匹配的 meta.obb_angle_contract 标记。"""
    marker = (state.get("meta") or {}).get("obb_angle_contract")
    if marker != expected:
        raise CheckpointIncompatibleError(
            "OBB checkpoint is incompatible with the current decoder: "
            f"expected meta.obb_angle_contract={expected!r}, got {marker!r}. "
            "Pre-cleanup OBB checkpoints (proportional encoding or gate-fusion "
            "state) must be retrained under the shifted-only contract."
        )
```

`state_dict`（202-215 行）：在 `return state` 前加入：

```python
        if getattr(self.model, "box_mode", None) == "obb":
            state["meta"] = {"obb_angle_contract": OBB_ANGLE_CONTRACT}
```

`load_resume_state`（240-248 行）：在 `self.load_state_dict(state)` 之前加入：

```python
        if getattr(self.model, "box_mode", None) == "obb":
            assert_checkpoint_compat(state)
```

`load_tuning_state`（250-275 行）：在头部调整逻辑之前加入：

```python
        if getattr(self.model, "box_mode", None) == "obb":
            kind = classify_checkpoint_kind(state)
            if kind != "hbb":
                assert_checkpoint_compat(state)
            else:
                print("Load unmarked 4D HBB checkpoint as OBB pretraining")
```

（规则：标记为 `shifted_v1` 的 OBB 检查点接受；无标记但可识别为 4D HBB 的检查点作为预训练接受；其余无标记/标记不符的 5D/6D OBB 检查点在非严格 `_matched_state` 加载之前拒绝。）

### A10.2 推理/导出工具（四处，逻辑一致）

`tools/inference/torch_inf.py:120-130`、`tools/inference/torch_inf_vis.py:117-127`、`tools/visualization/fiftyone_vis.py:238-248`、`tools/deployment/export_onnx.py:31-39`。在各文件 `cfg.model.load_state_dict(...)` 之前加入（变量名按各文件实际加载结果调整）：

```python
        from engine.solver._solver import OBB_ANGLE_CONTRACT, CheckpointIncompatibleError

        if getattr(cfg.model, "box_mode", None) == "obb":
            marker = (checkpoint.get("meta") or {}).get("obb_angle_contract")
            if marker != OBB_ANGLE_CONTRACT:
                raise CheckpointIncompatibleError(
                    "OBB checkpoint is incompatible with the current decoder: "
                    f"expected meta.obb_angle_contract={OBB_ANGLE_CONTRACT!r}, "
                    f"got {marker!r}"
                )
```

HBB 推理/导出路径不受影响。`deim_app` 应用层若已有 `strict=False` 适配路径，同样在其前插入该检查（执行时按实际入口核对）。

**验收**：`test/test_obb_checkpoint_contract.py` 全绿；内存内状态往返一次（保存 → 分类 → 校验 → 加载）成功。
