# DEIMv2 HBB/OBB 工程化平台重构设计

日期：2026-08-05

状态：设计已批准，待用户复核书面规范

## 1. 背景与结论

当前代码已经完成主要模型研究，HBB/OBB 模型、损失函数、角度契约、AMP、EMA、学习率调度和评估链路包含大量已验证行为。下一阶段目标不是继续重塑模型，而是把代码建设成可长期维护的单机 GPU 训练平台，并统一训练、恢复、评估、导出和推理流程。

结论：采用兼容式分层重构，不重写模型核心，不一次性迁移配置系统。现有 YAML、命令行和 checkpoint 必须继续可用；新架构通过兼容层逐步接管外围生命周期。

## 2. 已确认需求

- 平台范围：完整训练平台。
- 第一阶段环境：单机单卡和单机多卡 GPU。
- 任务范围：HBB 与 OBB 均为正式能力。
- 首要目标：流程自动化。
- 正式入口：统一 CLI。
- 兼容要求：现有 YAML、命令行和 checkpoint 完全兼容。
- 迁移方式：增量迁移。第一阶段禁止修改 backbone、decoder、criterion、matcher、OBB geometry、postprocessor 和 evaluator 的数学公式；若回归测试暴露独立缺陷，必须作为单独变更处理。

## 3. 当前问题

### 3.1 训练职责耦合

`engine/solver/det_engine.py::train_one_epoch()` 同时处理 forward、loss、AMP、梯度诊断、裁剪、optimizer、EMA、scheduler、终端输出和 Comet 日志。分支复杂度已经导致过 AMP 下 EMA 不更新、scheduler 不推进等回归。

`engine/solver/det_solver.py::DetSolver.fit()` 同时负责任务生命周期、阶段切换、checkpoint、best 指标、验证和日志。阶段语义与文件名绑定，难以独立测试。

### 3.2 工具链重复

PyTorch、ONNX、TensorRT 和 OpenVINO 推理脚本分别维护 checkpoint 加载、预处理、尺寸还原、结果过滤和输出逻辑。脚本使用 `sys.path.append` 并直接依赖工程内部对象，尚无稳定的 Python 推理接口。

### 3.3 配置缺少启动前验证

YAML registry 具有较强灵活性，但任务类型、box mode、类别数、scheduler 组合、AMP/device、数据路径和 checkpoint 兼容性主要在运行时才暴露错误。

### 3.4 自动化测试缺口

项目目前没有正式 `tests/` 测试套件。模型研究阶段积累了规范和诊断文档，但 AMP、EMA、scheduler、resume、角度契约和导出一致性缺少持续回归护栏。

### 3.5 依赖与运行产物未分层

训练、评估、可视化和部署依赖集中在同一 requirements 文件。实验目录中的配置、指标、checkpoint、导出文件和诊断快照没有统一 manifest 和生命周期约定。

## 4. 目标与非目标

### 4.1 目标

1. 统一 `train`、`resume`、`eval`、`export`、`infer` 和 checkpoint 检查入口。
2. 保持旧 YAML、CLI 和 checkpoint 可用，重构前后模型结果在定义的浮点容差内一致。
3. 将训练计算、精度控制、优化推进、checkpoint 和日志拆成可独立测试的组件。
4. 为 HBB/OBB 建立统一、显式的推理输入输出契约。
5. 建立可复现的运行目录、结构化指标和环境记录。
6. 在不引入集群基础设施的前提下稳定支持单机单卡与单机 DDP。

### 4.2 非目标

1. 不重写 backbone、decoder、criterion、matcher、OBB geometry 或 evaluator。
2. 不替换现有 YAML registry，不要求用户迁移旧配置。
3. 第一阶段不引入 Web 控制台、数据库、Kubernetes、Slurm 或任务队列。
4. 不设计多租户、权限、远程制品库或分布式实验数据库。
5. 不改变已发布 checkpoint 的张量命名和数值语义。
6. 不把未来分割、姿态任务的抽象提前加入第一阶段。

## 5. 总体架构

```text
new CLI / legacy wrappers
            |
            v
application services
train | resume | evaluate | export | infer | inspect-checkpoint
            |
            v
runtime contracts and controllers
config | checkpoint | events | prediction | precision | optimization
            |
            v
existing engine
model | criterion | dataset | geometry | evaluator | postprocessor
```

建议新增顶层包 `deim_app/`，不移动现有 `engine/`。依赖方向只能从 `deim_app` 指向 `engine`；模型层不得反向 import 应用层。

### 5.1 CLI 层

正式入口：

```text
python -m deim_app train -c CONFIG [--resume CHECKPOINT]
python -m deim_app resume -c CONFIG -r CHECKPOINT
python -m deim_app eval -c CONFIG -r CHECKPOINT
python -m deim_app export -c CONFIG -r CHECKPOINT --format onnx|tensorrt|openvino
python -m deim_app infer -c CONFIG -r CHECKPOINT --input INPUT
python -m deim_app inspect-checkpoint -r CHECKPOINT
```

`train --resume` 与 `resume` 语义一致；独立子命令用于提高可发现性。CLI 只负责解析参数、调用应用服务、格式化成功或失败结果，不包含模型构建与业务计算。

兼容入口继续存在：

- `train.py` 保留全部现有参数并委托新应用服务。
- `tools/inference/*.py` 保留现有调用方式并委托统一推理服务。
- `tools/deployment/*.py` 保留现有调用方式并委托统一导出服务。

旧入口在第一阶段不输出弃用警告。

### 5.2 应用服务层

- `TrainApplication`：解析运行配置，创建训练会话，执行新训练或 resume。
- `EvaluateApplication`：加载模型资产并运行 HBB/OBB evaluator。
- `ExportApplication`：构建部署模型，导出目标后端并执行一致性验证。
- `InferApplication`：创建推理 pipeline，支持图片、目录和视频输入。
- `InspectCheckpointApplication`：检查 schema、任务、box mode、类别、epoch 和可恢复组件。

应用服务负责编排，不直接实现 AMP、几何换算或后端推理。

## 6. 配置契约

现有 `YAMLConfig` 仍是对象实例化入口。新层在实例化训练运行时前产生只读 `ResolvedRunConfig`，包含：

- 原始配置路径和内容摘要。
- CLI override 后的最终值。
- task、box mode、类别数和类别名称。
- train/eval spatial size。
- device、AMP 和 distributed 设置。
- optimizer、scheduler 和 warmup 模式摘要。
- 数据集路径、输出目录和 resume/tuning 互斥关系。
- checkpoint 元数据及兼容检查结果。

启动前验证必须覆盖：

1. 配置文件和数据路径存在。
2. HBB/OBB task、postprocessor、evaluator 和 box mode 一致。
3. 类别数与 checkpoint model signature 兼容。
4. `resume` 与 `tuning` 不可同时启用。
5. AMP 仅在支持的设备上启用。
6. iteration scheduler 与 epoch scheduler 的配置组合合法。
7. 输出目录可创建且不会意外覆盖不兼容实验。

验证层不得改变旧 YAML 的默认值。无法从旧配置确定的字段保留现有行为，并在 manifest 中明确记录。

## 7. 训练生命周期

### 7.1 TrainingSession

`TrainingSession` 接管 epoch 循环与高层状态机：

```text
setup -> optional resume -> optional baseline eval
      -> train epoch -> validate -> checkpoint
      -> stage transition -> next epoch -> finalize
```

职责包括 start epoch、停止条件、单机 DDP sampler epoch、阶段切换、验证频率和最终结果。现有 `DetSolver.fit()` 第一阶段作为适配器调用该会话。

### 7.2 TrainStepExecutor

单个训练 step 的固定顺序：

```text
zero gradients
forward
loss computation
finite-loss check
backward through PrecisionController
gradient inspection policy
gradient clipping
optimizer decision
OptimizationController commit
emit structured event
```

它返回结构化 `StepResult`，字段固定包含 loss 明细、total loss、`data_step`、`optimizer_step`、optimizer 是否成功 step、AMP scale、是否溢出、gradient norm、样本数和耗时。

### 7.3 PrecisionController

统一 AMP 与非 AMP 行为：

- 管理 autocast 和 GradScaler。
- unscale 后判断非有限梯度。
- 溢出时跳过 optimizer step 并更新 scale。
- 明确返回 `optimizer_step_succeeded`。
- 不直接更新 EMA 或 scheduler。

### 7.4 OptimizationController

唯一负责 optimizer 之后的状态推进：

- optimizer 成功 step 后才更新 EMA。
- 为保持现有训练曲线兼容，iteration scheduler 按每个已消费的 dataloader iteration 推进；AMP 溢出导致 optimizer 跳步时仍推进 scheduler。
- epoch scheduler 在 epoch 完成后推进。
- warmup 与主 scheduler 的切换集中管理。
- checkpoint 恢复后验证 optimizer、scheduler、scaler 与 epoch 的连续性。

当前 FlatCosine scheduler 保留无状态实现，但由控制器使用 `data_step` 计算学习率，避免调用位置再次分叉。`data_step` 表示已消费的训练 batch 数；`optimizer_step` 仅在参数成功更新时递增。对外日志中的 `global_step` 等同于 `data_step`，checkpoint 同时保存两者。

### 7.5 DiagnosticsPolicy

提供 `off`、`standard`、`debug` 三档：

- `off`：仅有限性检查和必要异常。
- `standard`：记录溢出、gradient norm 和有限数量快照。
- `debug`：启用逐参数零梯度/非有限梯度诊断和详细快照。

生产默认 `standard`。诊断逻辑不得改变优化语义。

## 8. Checkpoint 契约

旧 checkpoint 保持可读。新 checkpoint 在现有 key 基础上新增：

```text
schema_version
task
box_mode
config_digest
model_signature
class_names
epoch
global_step
optimizer_step
runtime_versions
```

现有 `model`、`ema`、`optimizer`、`scaler`、`lr_scheduler`、`lr_warmup_scheduler` 和 `last_epoch` key 保持不变。

`CheckpointManager` 职责：

- 兼容加载本地路径和现有 URL checkpoint。
- 旧 checkpoint 无 manifest 时，仅从现有 key 和用户提供的 YAML 推断 task、box mode、epoch 与可恢复组件；无法确认的字段标记为 `unknown`，不得猜测类别名称或模型 signature。
- 保存到同目录临时文件，flush 后原子 rename。
- 管理 `last.pth`、周期 checkpoint 和按指标命名的 best checkpoint；现有 `best_stg1.pth`、`best_stg2.pth` 继续作为兼容别名生成和读取。
- 加载前检查 task、box mode、类别和模型 signature。
- 提供仅加载模型的 tuning 模式和完整 resume 模式，两者不得混用。

resume 验收必须检查下一 epoch、data/global step、optimizer step、optimizer buffer、AMP scale、EMA updates 和 lr 连续。旧 checkpoint 没有 step 字段时，`data_step` 按 `next_epoch * len(train_dataloader)` 重建，`optimizer_step` 标记为 unknown，直到新训练产生首个成功 optimizer step。

## 9. 统一推理与导出

### 9.1 Pipeline

```text
InputAdapter -> Preprocessor -> RuntimeBackend
             -> OutputDecoder -> PredictionBatch -> ResultWriter
```

- `InputAdapter`：图片路径、目录、视频帧或内存图像。
- `Preprocessor`：resize、normalize、batch 和原图元数据。
- `RuntimeBackend`：PyTorch、ONNX Runtime、TensorRT 或 OpenVINO。
- `OutputDecoder`：调用统一 postprocessor，完成阈值、类别和坐标还原。
- `ResultWriter`：JSON、可视化图片或视频；不得参与模型计算。

### 9.2 PredictionBatch

预测输出不得再依赖张量最后一维猜测 box 类型。契约固定包含：

- `task` 与 `box_mode`。
- `class_ids`、`scores`。
- HBB 的 `xyxy` 或 OBB 的内部物理角表示。
- OBB polygon 与 external-rect 的显式转换方法。
- 原图尺寸、缩放信息、image id。
- preprocess、inference、postprocess 耗时。

内部 OBB 角度契约以重构开始时实际通过回归测试的当前实现为准，并记录在行为基线中。本重构不迁移角度数值域、不改变 seam、不调整宽高交换规则；任何角度契约变更必须作为独立项目执行。

### 9.3 导出一致性

导出服务必须：

1. 使用与推理服务相同的 checkpoint 选择和预处理。
2. 记录输入尺寸、动态轴、opset、类别和 box mode。
3. 使用固定样例执行 PyTorch 与目标后端输出一致性测试。
4. HBB 比较 boxes/scores/classes；OBB 同时比较周期角度距离和 polygon。
5. 一致性失败时不将产物标记为可发布。

## 10. 事件、日志与实验目录

统一实验目录：

```text
runs/<experiment>/
  resolved_config.yml
  manifest.json
  checkpoints/
    last.pth
    best-<metric>.pth
    epoch-0001.pth
  metrics/
    train.jsonl
    eval.jsonl
  exports/
  artifacts/
```

JSONL 是本地权威指标记录。终端、TensorBoard 和 Comet 实现为 `EventSink`；任一远程 sink 异常只能产生警告，不能中断训练。

训练事件固定包括 epoch、data/global step、optimizer step、lr、各项 loss、total loss、gradient norm、AMP scale、AMP skipped step、EMA update、吞吐和显存；验证与 checkpoint 事件分别包含验证指标和 checkpoint 路径。

`manifest.json` 记录配置摘要、命令行、代码版本标识、Python/PyTorch/CUDA 版本、设备、随机种子、数据路径摘要和 checkpoint lineage。第一阶段不要求自动上传制品。

## 11. 错误处理

应用层定义稳定错误类别：

- `ConfigurationError`
- `CheckpointCompatibilityError`
- `DatasetError`
- `NumericalTrainingError`
- `ExportError`
- `BackendRuntimeError`

CLI 对已知错误输出简短原因、相关路径和修复建议并返回非零退出码。完整 traceback 写入运行日志。模型内部不可吞掉异常；监控 sink 异常可降级为警告。

## 12. 测试策略

### 12.1 行为基线

重构前固定一套 HBB 小配置和一套 OBB 小配置，各自保存：

- 固定输入预测。
- 单 step loss 和梯度摘要。
- 一个短 epoch 的 lr、AMP scale、EMA updates。
- save/resume 前后的状态。
- 评估指标和导出结果。

### 12.2 单元测试

- 配置解析和启动前验证。
- checkpoint 新旧 schema 与兼容错误。
- AMP 成功、溢出跳步和 scale 更新。
- optimizer、EMA、iteration/epoch scheduler 推进矩阵。
- HBB/OBB prediction contract。
- OBB 角度与编解码已有数学契约。

### 12.3 组件测试

固定 synthetic batch 覆盖：非 AMP、AMP 成功和 AMP overflow。测试必须断言参数是否变化、EMA 是否变化、lr 是否变化和 scaler 是否变化。

### 12.4 集成测试

微型 HBB/OBB 数据执行：

```text
train -> save -> resume -> eval -> export -> infer
```

同时运行旧入口和新入口，比较 resolved config、checkpoint 关键状态与预测。

### 12.5 后端测试

PyTorch 是参考后端。ONNX Runtime 为第一阶段必测部署后端；TensorRT/OpenVINO 在环境可用时运行。后端不可用必须明确 skip 原因，不得假装通过。

## 13. 迁移阶段

### 阶段 0：冻结基线

建立 HBB/OBB 行为样例、测试目录、数值容差和当前命令快照。不移动生产代码。

### 阶段 1：运行契约

引入 resolved config、manifest、checkpoint compatibility 和统一实验目录。旧训练流程仍负责实际训练。

### 阶段 2：统一 CLI

实现应用服务和新 CLI；旧入口改为薄包装。train/resume/eval/export/infer 都从同一配置与 checkpoint 服务进入。

### 阶段 3：统一推理

抽取预处理、PredictionBatch、PyTorch backend 和结果 writer，再迁移 ONNX、TensorRT 与 OpenVINO。

### 阶段 4：训练 step 拆分

在测试保护下抽取 PrecisionController、TrainStepExecutor、OptimizationController 和 DiagnosticsPolicy。每次只迁移一项职责并运行数值回归。

### 阶段 5：训练 session 拆分

抽取 TrainingSession、CheckpointManager 和 EventSink，简化 `DetSolver.fit()`。保留现有 stage1/stage2 行为的兼容适配。

### 阶段 6：依赖与文档整理

分离 train、deploy 和 dev 依赖，补齐运维手册、checkpoint 说明和故障排查文档。旧入口在本设计覆盖的全部阶段内保留；是否弃用不属于本项目范围。

## 14. 验收标准

1. 现有 YAML、`train.py`、推理/导出脚本和 checkpoint 无需修改即可使用。
2. 新 CLI 完成 train、resume、eval、export、infer 和 inspect-checkpoint。
3. HBB/OBB 重构前后固定样例结果满足批准的数值容差，评估指标无非预期回退。
4. AMP 成功和溢出路径均有自动测试；EMA 与 scheduler 推进不再依赖重复分支。
5. resume 后 epoch、global step、optimizer、lr、AMP scale 和 EMA updates 连续。
6. 新 checkpoint 原子保存，损坏或不兼容 checkpoint 在启动前失败。
7. PyTorch 与 ONNX 的固定样例一致性测试通过；其他后端按环境明确通过或跳过。
8. Comet/TensorBoard 不可用时训练仍可运行，JSONL 和本地 checkpoint 完整。
9. 单机 DDP 仍使用现有启动方式并通过最小 smoke test。

## 15. 风险与控制

- 数值回归：先测试后拆分，模型数学文件不纳入无关重构。
- 兼容层长期存在：旧入口只做参数转发，不复制业务逻辑，并用等价测试限制分叉。
- 配置验证改变旧默认：验证只拒绝明确冲突；未知旧行为记录 warning 而非擅自修正。
- 平台范围膨胀：第一阶段不加入 Web、集群、数据库和未来任务插件。
- 后端输出漂移：统一后处理并要求导出一致性测试。
- checkpoint 损坏：原子写入、schema 标识和加载前兼容检查。

## 16. 实施纪律

1. 每个阶段都必须可以独立交付和回滚。
2. 不以大规模移动文件作为重构起点。
3. 不同时修改模型数学和训练生命周期。
4. 新旧入口必须调用同一实现，不允许长期双份逻辑。
5. 未通过 HBB/OBB 回归测试，不进入下一迁移阶段。
