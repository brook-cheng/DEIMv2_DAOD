# DEIMv2 OBB 角度契约简化实现计划（修订版）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**协作模式（用户指定）：** 用户**只负责功能性代码改动**；**测试脚本由助手编写并运行**。每个 Task 流程：
1. 用户完成功能性代码改动（见「用户改动」清单）；
2. 助手编写/更新对应测试脚本（见「助手测试」清单）；
3. 助手运行测试并审核 diff；
4. 确认通过后进入下一 Task。

**Goal:** 在当前（手工回退后的）代码基线上落地已批准的「公开物理域 / decoder 私有等比归一域 / loss 内部规范域」分层角度契约。**无需任何回滚**：只改边界（GT 规范化、decoder 输出、anchor 默认值、denoising、criterion 非周期路径、死代码契约模块、注释清理），不动 decoder 内部已正确的等比精修。

**Architecture:** 先重写 `engine/deim/obb_angle_contract.py` 为等比公式 + loss 规范辅助函数，再修复 GT 规范化、decoder 输出与 anchor、denoising、criterion，最后做 import 审计、注释清理与全量验证。每个 Task 由用户改动功能性代码，助手负责全部测试编写与运行。

**Tech Stack:** Python 3.12, PyTorch, pytest（测试自插 ROOT 到 `sys.path`，命令统一 `python -m pytest <file> -q`）。

## Global Constraints

- 公式（本计划唯一批准形式，逐字采用）：`theta_norm = theta_phys_rad / pi`；逆 `theta_phys_rad = theta_norm * pi`。`theta_norm` 只有自然的 `0/1` 周期边界，归一区间内部**不允许**任何 shifted seam。
- 域纪律（spec §3）：`theta_norm` / `theta_logit` 只存在于 decoder 私有边界（`deim_decoder.py`、`denoising.py`、`dfine_decoder.py`）；criterion、matcher、postprocessor、obb_geometry、transforms、eval 只允许物理域与 loss 规范域辅助函数。
- 半开区间：所有 mod 用 `torch.remainder`；`canonicalize_phys_rad` 结果恒在 `[0, pi)`，`physical_rad_to_loss_rad` 恒在 `[-pi/4, 3pi/4)`，`periodic_angle_distance`（`with_signal=True`）恒在 `[-pi/2, pi/2)`。
- seam 字面量纪律（spec §9.6）：`pi / 4` 平移只允许出现在 `physical_rad_to_loss_rad` 公式与 criterion 内部；decoder、denoising、geometry、postprocessor、matcher、eval 中不允许出现。
- criterion 中任何角度差必须经 `periodic_angle_distance`；**禁止**转换到 `theta_loss_rad` 后直接相减。
- 保留四种 `angle_rep`（0/1/2/3）路径；**不重建**已删除的 `obb_codecs.py`；decoder 逐层精修（`deim_decoder.py:441-460`、`dfine_decoder.py:177`）已正确，**不改公式**（可改为调用 `norm_to_physical_rad`，语义等价，测试锁定）。
- **不新增任何配置键**（spec §8.10）：配置保持现状。
- 默认 anchor 物理方向保持迁移前的 `pi/4`：`_generate_anchors` 默认 `r = 0.5` → `r = 0.25`。
- **禁止**：`git reset --hard`、`git clean -fdx`、整树 restore、提交/推送/改写历史。当前工作区无需回滚，全程不执行任何 git 破坏性命令。
- 工作目录：`/mnt/d/cx/thired/deimv2_daod`（下文所有相对路径以此为准）。

## Current Baseline（Task 0 锁定的起始状态）

- `git status`：已跟踪修改仅 `docs/superpowers/review/OBB_CODE_REVIEW.md`、`engine/deim/__init__.py`；未跟踪 `.superpowers/`、四份文档、`engine/deim/obb_angle_contract.py`。
- 关键现状（审计记录，见 spec §1 表）：
  - `obb_geometry.py:100,238` GT 规范化：`remainder(θ + π/4, π) − π/4` → 输出 `[−π/4, 3π/4)`（docstring 声称 `[0,π]`）。
  - `deim_decoder.py:1100`（`_get_decoder_input` 训练期 encoder 辅助输出）与 `:1233-1241` 最终输出：`(x − 0.25)·π` → `[−π/4, 3π/4)`（共四处，见 Task 3）。
  - `deim_decoder.py:996` anchor 默认 `r = 0.5`。
  - `denoising.py:110-111`：`(θ + π/4)/π`。
  - `deim_criterion.py:363-374` 周期路径：`periodic_angle_distance(...)/π`（已正确）。
  - `deim_criterion.py:378-383` 非周期路径：`(θ + π/4)/π`。
  - `obb_angle_contract.py`：未跟踪、无引用、shifted 公式死代码。
  - `__init__.py:25`：残留 `obb_codecs` 注释。
- 测试基线：`test/test_deimv2_obb_smoke.py` 2 passed；`test/test_deim_criterion_obb_loss.py` 13 passed；较宽 OBB 集合 198 passed / 2 failed（两项为环境问题：`test_deim_vs_ultralytics` 内存、`test_mixup` 外部权重）。

## File Structure（改动清单）

**用户改动（功能性代码）**

- `engine/deim/obb_angle_contract.py`（重写：等比公式 + loss 规范辅助函数）
- `engine/deim/obb_geometry.py`（GT 规范化 `:100,238`、docstring；`periodic_angle_distance` 保持现有实现）
- `engine/deim/deim_decoder.py`（最终输出 `:1233-1241` 与 `_get_decoder_input` `:1100` 共四处 `(x−0.25)·π` → `x·π`、anchor `:996`、注释清理）
- `engine/deim/denoising.py`（`:110-111`）
- `engine/deim/deim_criterion.py`（非周期路径 `:378-383`）
- `engine/deim/yolo_obb_loss.py`（保留现状；助手以一致性测试锁定与 `periodic_angle_distance` 等价）
- `engine/deim/postprocessor.py`（`:62` 注释）
- `engine/deim/__init__.py`（`:25` 删除残留注释）
- `docs/superpowers/review/OBB_CODE_REVIEW.md`（恢复等比描述）

**助手编写/更新（测试脚本）**

- `test/test_obb_angle_contract.py`（重写为新公式，Task 1）
- `test/test_obb_roundtrip.py`（`test_angle_range`、`test_decoder_output_scaling`、`test_decoder_input_scaling` 改等比断言，Task 2）
- `test/test_deimv2_obb_smoke.py`（输出域 `[0,pi)`、anchor `r=0.25`、denoising `3pi/4→0.75`，Task 3/4）
- `test/test_deim_criterion_obb_loss.py`（seam 样例、非周期越界回归，Task 5）
- 新增域纪律审计测试（grep 断言，Task 6）

**不修改**（已正确）：`dfine_utils.py`、`matcher.py`、`dfine_decoder.py` 公式（可只做 docstring/调用等价替换）、`obb_transforms.py`、`obb_ops.py`、`engine/eval/*`、配置文件。

---

## Task 0：基线锁定（助手操作）

**用户无需改动。助手执行：**

- [ ] 运行 `git status --short`，确认与上述「Current Baseline」一致。
- [ ] 运行 `python -m pytest test/test_deimv2_obb_smoke.py test/test_deim_criterion_obb_loss.py -q`，确认 2 + 13 passed。
- [ ] 向用户报告基线核对结果，等待用户开始 Task 1 功能性改动。

**验收：** 基线状态与审计记录一致；两个测试文件通过。

---

## Task 1：契约模块重写

**用户改动**（`engine/deim/obb_angle_contract.py` 重写）：

- [ ] 保留模块与文件名；重写 6 个函数（spec §8.1）：`canonicalize_phys_rad`、`physical_rad_to_norm`（`theta/pi`）、`norm_to_physical_rad`（`theta*pi`）、`physical_rad_to_logit`、`logit_to_physical_rad`、`physical_rad_to_loss_rad`（`((θ+π/4) mod π) − π/4`）。**不实现** `periodic_delta_rad`/`apply_delta_rad`（前者由 `obb_geometry.periodic_angle_distance` 提供，后者已内联于 `dfine_utils`）。
- [ ] **删除**旧 shifted 平移语义与 docstring 中「平移优化区间」说明；模块头注释改为 spec §3 术语表与 §4 分层图。
- [ ] 全部 mod 用 `torch.remainder`。

**助手测试**（重写 `test/test_obb_angle_contract.py`，覆盖 spec §11.1）：

- [ ] 边界映射表（7.1 节）：`theta_phys_rad ∈ {0, pi/4, pi/2, 3pi/4, pi-1e-3}` → 精确断言 `theta_norm`（`/pi`）与 `theta_loss_rad`（`((θ+π/4) mod π) − π/4`）。
- [ ] 等比 round-trip：随机 1000 个 `theta ∈ [0, pi)`，`norm_to_physical_rad(physical_rad_to_norm(theta)) ≈ theta`（`allclose(rtol=1e-6)`）。
- [ ] 等价性：随机 1000 组 `(pred, gt)`，loss 域与物理域 `periodic_angle_distance` 结果 `allclose`（spec §6）。
- [ ] logit round-trip：远离 0/pi 的 theta，`logit_to_physical_rad(physical_rad_to_logit(theta))` 收敛。
- [ ] 残差示例表（7.2 节）：含 seam 样例 `pred=0.75pi, gt=0.70pi` 周期距离 `0.05pi`；`pred=0, gt=3pi/4` 周期距离 `pi/4`。
- [ ] 运行 `python -m pytest test/test_obb_angle_contract.py -q`，全绿；grep 确认模块无 shifted 平移残留。

**验收：** 测试全绿；模块无 shifted 平移残留。

---

## Task 2：geometry GT 规范化 + 委托

**用户改动**（`engine/deim/obb_geometry.py`）：

- [ ] `:100` 与 `:238` 两处 GT 规范化：`remainder(theta + pi/4, pi) - pi/4` → `remainder(theta, pi)`（或 `canonicalize_phys_rad(theta)`）。
- [ ] 同步修正两处 docstring（`θ belongs to [0,pi]` 现与实现一致）。
- [ ] `periodic_angle_distance`（`:18`）：保持现有实现（已是周期距离单一事实来源，无需委托）。

**助手测试**（更新 `test/test_obb_roundtrip.py` 等）：

- [ ] `test_angle_range`：改为断言输出 `theta ∈ [0, pi)`（`(thetas >= 0)` 且 `(thetas < pi)`）。
- [ ] `test_decoder_output_scaling`：改为验证等比 `x * pi` 映射 `[0,1] → [0, pi)`（`x=0 → 0`，`x=0.25 → pi/4`，`x=0.5 → pi/2`，`x=0.75 → 3pi/4`）。
- [ ] `test_decoder_input_scaling`：改为验证等比 `theta / pi` 映射 `[0, pi) → [0,1)`（`theta=0 → 0`，`pi/4 → 0.25`，`3pi/4 → 0.75`）。
- [ ] 新增：`xyxyxyxy_to_xywhr` 对 `3pi/4` 边界样例输出 `3pi/4`（旧实现折叠到 `-pi/4`）。
- [ ] 运行 `python -m pytest test/test_obb_roundtrip.py test/test_obb_transforms.py test/test_obb_adr_geometry.py -q`，全绿。

**验收：** 三个测试文件全绿；GT 输出域 `[0, pi)`。

---

## Task 3：decoder 最终输出 + anchor

**用户改动**（`engine/deim/deim_decoder.py`）：

- [ ] **四处** `(x - 0.25) * pi` → `x * pi`（`norm_to_physical_rad`）：
  - `:1100` `_get_decoder_input` 训练期 encoder 辅助输出 `enc_topk_bboxes`；
  - `:1237` 最终输出 `out_bboxes`；
  - `:1240` 最终输出 `out_refs`；
  - `:1243` 最终输出 `pre_bboxes`。
  删除各处「量纲为 [-pi/4, 3pi/4)」过时注释。
- [ ] `:996`：anchor 默认 `r = 0.5` → `r = 0.25`。`angle_step > 0` 分支不动。
- [ ] 核对 `pre_bboxes` 在 `_get_decoder_input` 等后续分支中的消费语义：若内部视图需保持等比 `[0,1]`，仅对外出口乘以 `pi`。

**助手测试**（更新 `test/test_deimv2_obb_smoke.py`、`test/test_obb_roundtrip.py`）：

- [ ] `test/test_deimv2_obb_smoke.py` 现有输出域断言（`theta in [0, pi]`，`:241-245`）保持并强化为严格 `[0, pi)`（含边界样例）。
- [ ] 新增：四种 `angle_rep` 各一组，固定输入下 decoder 输出角度 `in [0, pi)` 且等于期望物理角。
- [ ] 新增：anchor 默认 `r = 0.25` 对应物理角 `pi/4`（`_generate_anchors` 输出 logit 后 sigmoid 还原核对）；`angle_step > 0` 时候选角等比分布于 `[0,1)`。
- [ ] 运行 `python -m pytest test/test_deimv2_obb_smoke.py test/test_obb_roundtrip.py -q`，全绿。

**验收：** 相关测试全绿；decoder 输出域 `[0, pi)`；anchor 物理方向 `pi/4`。

---

## Task 4：denoising 等比化

**用户改动**（`engine/deim/denoising.py`）：

- [ ] `:110-111`：`(input_query_bbox[..., 4] + torch.pi / 4) / torch.pi` → `physical_rad_to_norm(input_query_bbox[..., 4])`（即 `theta / pi`）。
- [ ] 更新该处注释（删除 shifted 语义描述）。

**助手测试**（更新 `test/test_deimv2_obb_smoke.py`）：

- [ ] 新增：GT `theta = 3pi/4` 时 denoiser 产出归一角 `0.75`（无 clip）；GT 接近 `pi` 时归一角接近 1 且不越界。
- [ ] 运行 `python -m pytest test/test_deimv2_obb_smoke.py -q`，全绿。

**验收：** 相关 smoke 测试全绿；denoising 纯线性无 seam。

---

## Task 5：criterion 非周期路径统一

**用户改动**（`engine/deim/deim_criterion.py`）：

- [ ] `deim_criterion.py:378-383` 非周期路径：`(θ + pi/4)/pi` L1 → `angle_term = lambda_angle * periodic_angle_distance(src_phys, tgt_phys) / pi`。
- [ ] criterion 周期路径 `:363-374` 公式不变（已是 `periodic_angle_distance(...)/π`）。两路径 angle_term 数值完全一致——`periodic_angle_flag` 对角度项变为无副作用，这是本简化的意图。
- [ ] `yolo_obb_loss.py::yolo_angle_loss` 与 `compute_angle_cost_matrix` **保留现状**（`periodic_angle_distance` 已是单一事实来源；以一致性测试锁定）。

**助手测试**（更新 `test/test_deim_criterion_obb_loss.py` 等）：

- [ ] 新增 seam 样例：`pred = 0.75pi, gt = 0.70pi` 时非周期路径 angle_term 等于周期路径（`periodic_angle_distance/pi`），且不等于直接相减结果（`0.95pi/pi`）。
- [ ] 新增回归：非周期路径在 `theta > 3pi/4` 时无越界值（旧 `(θ+π/4)/pi` 缺陷）。
- [ ] 保持现有 13 个测试通过。
- [ ] `test/test_yolo_obb_loss.py`、`test/test_matcher_obb_angle.py` 锁定 `yolo_angle_loss`/`compute_angle_cost_matrix` 与 `periodic_angle_distance` 一致（`allclose`）。
- [ ] 运行 `python -m pytest test/test_deim_criterion_obb_loss.py test/test_yolo_obb_loss.py test/test_matcher_obb_angle.py -q`，全绿。

**验收：** 相关测试全绿；criterion 无 seam 平移、无裸相减。

---

## Task 6：清理与审计

**用户改动**：

- [ ] `engine/deim/__init__.py:25`：删除 `# from .obb_codecs import DirectAngleCodec, ExternalRectOffsetCodec` 残留注释。
- [ ] `engine/deim/postprocessor.py:62`：更新注释（θ 直通 `[0, pi)` 物理域，不再描述归一域）。
- [ ] `docs/superpowers/review/OBB_CODE_REVIEW.md`：恢复等比描述（当前仍宣称 shifted seam 正确），同步 spec 修订内容。

**助手测试**（新增域纪律审计测试，spec §11.5）：

- [ ] grep 断言：`criterion`、`matcher`、`postprocessor`、`obb_geometry`、`obb_transforms` 中无 `physical_rad_to_norm` / `norm_to_physical_rad` / `theta_norm` 引用。
- [ ] grep 断言：seam 字面量（`pi / 4` 平移）只出现在 `physical_rad_to_loss_rad` 与 criterion 内部。
- [ ] 运行审计测试与全量回归，确认无死代码/过时注释残留。

**验收：** 审计测试通过；无死代码/过时注释残留。

---

## Task 7：集成验证（助手执行）

**助手操作：**

- [ ] `python -m pytest test/test_obb_roundtrip.py test/test_obb_transforms.py test/test_obb_eval.py test/test_obb_adr_geometry.py test/test_obb_adr_loss.py test/test_obb_loss_integration.py test/test_yolo_obb_loss.py test/test_matcher_obb_angle.py test/test_deimv2_obb_smoke.py test/test_deim_criterion_obb_loss.py -q` 全绿（环境相关两项已知失败除外）。
- [ ] 短迭代训练 smoke（现有入口）：四种 `angle_rep` 各跑少量 step，损失有限且正常下降。
- [ ] `convert_to_deploy` smoke 通过（如有现成入口）。
- [ ] 与 Task 0 基线对比：除已知环境失败外无新增失败。

**验收：** 全量测试通过；无回归；契约模块被真实引用（不再是死代码）。

---

## 明确排除项（不做的事）

1. 不回滚、不清理工作区（回退已完成）。
2. 不重建 `obb_codecs.py`。
3. 不改 `dfine_utils.py`、`matcher.py` 公式与配置。
4. 不做 anchor `pi/4 vs pi/2` 对比实验。
5. 不修改 decoder 逐层精修与 `dfine_decoder` 旋转采样换算（已正确）。
6. 不提交、不推送。
7. 用户不编写测试代码；测试脚本全部由助手编写并运行。
