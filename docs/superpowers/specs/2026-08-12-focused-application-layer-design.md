# DEIMv2 聚焦型工程应用层设计

日期：2026-08-12

状态：设计及首版公开参数已批准

## 1. 背景

DEIMv2 的底层 HBB/OBB 算法仍在持续优化。工程化首版不以重构训练引擎或建设完整平台为目标，而是优先解决两个直接影响使用的问题：

1. 简化训练、评估和推理参数的设置。
2. 提供便于业务代码和命令行使用的统一推理入口。

工程应用层必须与底层算法实现解耦。decoder、loss、输出字典、完整 YAML 结构或模型构建方式发生变化时，应用接口应尽量保持稳定，变化集中在算法适配器内。

## 2. 设计结论

首版采用“稳定应用契约 + 任务适配器”的轻量架构：

```text
用户 / 业务系统
        |
        v
deim_app
  简化 YAML | CLI | Python API | Prediction
        |
        v
DetectionAdapter
  配置映射 | 模型构建 | 训练/评估调用 | 推理解码 | 导出
        |
        v
engine
  当前 DEIM HBB/OBB 算法与完整 YAMLConfig
```

首版只实现当前 DEIM 检测算法的 `DeimDetectionAdapter`，不建设动态插件发现系统。未来增加其他算法时，通过实现相同适配器契约接入。

依赖方向固定为 `deim_app -> adapter -> engine`。`engine` 禁止导入 `deim_app`，应用层禁止直接依赖 decoder、criterion、matcher、solver 具体实现和底层原始输出结构。

## 3. 首版目标

### 3.1 必须实现

1. 稳定、精简且支持 `__include__` 的应用 YAML 配置。
2. 应用 YAML 加常用 CLI 参数覆盖。
3. 应用基础 YAML 到现有完整算法 YAML preset 的映射。
4. 一个覆盖训练、评估、模型加载、推理和导出的 DEIM 检测适配器。
5. HBB/OBB 显式区分的结构化预测结果。
6. 共用同一核心实现的 Python 推理 API 和 CLI。
7. 图片和目录输入，以及 JSON、DOTA 和可视化结果输出。
8. PyTorch 推理后端。
9. 旧完整 YAML、`train.py` 和现有工具继续可用。

### 3.2 后续迭代

以下能力不作为首版验收条件：

- ONNX、TensorRT 和 OpenVINO 后端。
- 视频、流式和异步批处理。
- 动态插件发现与第三方算法注册。
- Web UI、服务化 API、数据库或任务队列。
- 训练 step、训练 session、事件系统和 checkpoint 生命周期的大规模拆分。
- 任意底层配置透传、配置版本迁移器和复杂 preset 继承。

后续能力必须在首版稳定契约上迭代，不应推翻用户配置和推理结果接口。

## 4. 应用配置

### 4.1 配置格式与继承

应用配置继续使用 `.yml`，并沿用现有 `__include__` 递归字典合并语义。首版提供两个应用基础文件：

- `hbb_app.yml`：固定 HBB COCO 数据契约和 HBB 算法 preset。
- `obb_app.yml`：固定 OBB 数据契约和 OBB 算法 preset，同时支持 `DOTA` 与 `YOLO-OBB`。

算法 preset 由应用基础 YAML 固定。普通用户通过更换 `__include__` 选择任务和模型方案，不再额外设置 `model.preset`。应用基础 YAML 可以关联 `sp_fz_common.yml` 一类完整算法配置，但普通用户的配置只覆盖本设计批准的公开字段。

### 4.2 OBB 配置示例

```yaml
__include__:
  - ../base/obb_app.yml

project:
  name: dlzdt
  output_dir: ./outputs/dlzdt

runtime:
  input_size: [640, 640]
  seed: 0

data:
  format: DOTA
  train_images: /data/images/train
  train_annotations: /data/labelTxt/train
  val_images: /data/images/val
  val_annotations: /data/labelTxt/val
  classes_file: /data/classes.txt
  num_workers: 2
  cache_images: disk

train:
  epochs: 100
  batch_size: 8
  learning_rate: 0.0005
  device: cuda
  amp: true
  pretrained: ./ckpts/pretrained.pth
  resume: null
  early_stopping:
    enabled: false
    patience: 40

evaluation:
  batch_size: 2
  device: cuda

inference:
  checkpoint: null
  device: cuda
  batch_size: 1
  score_threshold: 0.25
  top_k: 300
  class_filter: null
  output_formats: [json, dota, visualization]
```

`data.format` 可取 `DOTA` 或 `YOLO-OBB`。两种格式都使用图像目录、标注目录和 `classes_file`；适配器仅切换当前 `DotaDataset` 的解析模式。

### 4.3 HBB COCO 配置差异

HBB 配置继承 `hbb_app.yml`，数据部分使用 COCO 图像目录和 annotation JSON：

```yaml
__include__:
  - ../base/hbb_app.yml

data:
  format: COCO
  train_images: /data/train2017
  train_annotations: /data/annotations/instances_train2017.json
  val_images: /data/val2017
  val_annotations: /data/annotations/instances_val2017.json
  num_workers: 4
  cache_images: none
```

HBB 类别名称和类别数从 COCO annotation 的 `categories` 推导。OBB 类别名称和类别数从 `classes_file` 推导。用户不得手工设置 `num_classes`。

### 4.4 首版公开字段

首版应用 YAML 只允许覆盖以下稳定字段：

- `project.name`、`project.output_dir`。
- `runtime.input_size`、`runtime.seed`。
- `data.format`、训练/验证图像与标注路径、OBB `classes_file`、`data.num_workers`、`data.cache_images`。
- `train.epochs`、`train.batch_size`、`train.learning_rate`、`train.device`、`train.amp`。
- `train.pretrained`、`train.resume`。
- `train.early_stopping.enabled`、`train.early_stopping.patience`。
- `evaluation.batch_size`、`evaluation.device`。
- `inference.checkpoint`、`inference.device`、`inference.batch_size`。
- `inference.score_threshold`、`inference.top_k`、`inference.class_filter`、`inference.output_formats`。

字段应使用强类型、显式默认值和启动前验证。应用配置不复制底层完整 YAML 的全部字段。

### 4.5 不公开的底层参数

以下参数由应用基础 YAML 和算法 preset 管理，不属于首版应用契约：

- `angle_rep`、`offset_scale_source`、gate fusion 等 OBB 消融参数。
- backbone、encoder、decoder 结构参数。
- optimizer 类型、参数分组、betas 和 weight decay。
- warmup、flat、no-aug 阶段参数。
- Mosaic、Mixup、CopyBlend 等增强概率和阶段。
- EMA、gradient clipping、checkpoint 和日志频率。
- loss、matcher 权重和 DDP 内部参数。
- `num_classes` 和应用配置之外的任意点路径 override。

底层新增实验参数默认留在 preset 中，不自动进入应用 YAML。研究人员仍可直接使用现有完整算法 YAML。

### 4.6 参数映射

适配器必须按以下规则映射公开参数：

- `runtime.input_size` 同步映射训练 resize、collate base size、验证 resize、`eval_spatial_size` 和推理预处理。
- `train.learning_rate` 只覆盖 optimizer 主学习率。backbone 分组学习率、比例和显式分组值保留 preset 设置。
- `train.batch_size` 和 `evaluation.batch_size` 都表示跨 GPU 的总 batch size。
- `data.cache_images` 只接受 `none`、`disk`、`ram`，适配器转换为当前 dataset 字段。
- `train.pretrained` 只加载模型初始化权重；`train.resume` 恢复完整训练状态，两者互斥。
- early stopping 首版只开放 `enabled` 和 `patience`；metric、mode、min epochs、min delta 和 restore-best 由 preset 固定。
- 推理的 score threshold、Top-K 和类别过滤在结构化结果层执行，不改变适配器保留的全量解码结果。
- `inference.output_formats` 只接受当前 box mode 支持的 writer；HBB 不接受 DOTA 输出。

适配器生成只读 resolved 配置摘要，并调用现有 `YAMLConfig` 完成底层对象构建。

### 4.7 参数优先级

最终配置优先级固定为：

```text
CLI 覆盖 > 应用 YAML > 应用基础 YAML > 算法 preset 默认值
```

CLI 仅覆盖本节白名单中的频繁运行参数，包括训练/评估/推理设备、checkpoint、resume、输入、输出、推理 batch size、置信度阈值、Top-K 和类别过滤。复杂模型参数不通过 CLI 暴露。

## 5. 适配器契约

首版适配器提供以下逻辑能力：

```text
resolve_config
train
evaluate
load_model
predict
export
```

适配器是唯一同时理解应用契约和底层算法实现的模块，其职责包括：

- 映射简化配置与底层 YAML。
- 构建当前 solver、模型和 postprocessor。
- 选择普通模型权重或 EMA 权重。
- 调用现有训练和评估入口，不复制训练循环。
- 将底层推理输出转换为标准预测对象。
- 复用现有 HBB/OBB 几何与后处理函数，不复制数学公式。
- 隔离 checkpoint、输出字典和构建流程的算法特定细节。

应用层不得通过 tensor 最后一维猜测 box 类型。`box_mode` 必须来自已验证的任务配置或模型元数据。

## 6. Python API 与 CLI

### 6.1 Python API

首版提供面向业务使用的简单入口：

```python
from deim_app import DetectionModel

model = DetectionModel.from_config("configs/app/cable_obb.yml")
model.load("runs/cable-obb/best.pth")

results = model.predict(
    source="images/",
    score_threshold=0.3,
)

results.save_images("outputs/vis")
results.export_dota("outputs/dota")
```

`DetectionModel` 是应用 facade，不包含算法逻辑。它根据配置选择适配器，并委托适配器完成模型构建、加载和推理。

### 6.2 CLI

首版命令：

```text
python -m deim_app train -c APP_CONFIG
python -m deim_app eval -c APP_CONFIG -r CHECKPOINT
python -m deim_app infer -c APP_CONFIG -r CHECKPOINT --input INPUT --output OUTPUT
python -m deim_app export -c APP_CONFIG -r CHECKPOINT --format FORMAT
```

CLI 与 Python API 必须调用同一应用服务和适配器，禁止维护第二套模型加载、预处理或后处理逻辑。

首版 `export` 可以先委托现有导出实现；支持的 format 由当前环境和适配器明确报告。后端尚未实现时必须返回清晰错误，不得生成不可用产物。

## 7. 推理数据流

```text
图片路径 / 目录 / 内存图像
        |
        v
InputAdapter
        |
        v
Preprocessor
        |
        v
DeimDetectionAdapter + PyTorch Runtime
        |
        v
现有 postprocessor / OBB geometry
        |
        v
PredictionCollection
        |
        +-> Python 业务逻辑
        +-> JSON
        +-> DOTA txt
        +-> 可视化图片
```

预处理元数据至少记录原图尺寸、模型输入尺寸、缩放信息和 image id。坐标还原只执行一次。

置信度阈值属于结果过滤。后端原始解码结果与导出结果不得因可视化阈值被提前截断。

## 8. 预测结果契约

### 8.1 Detection

单个检测结果包含：

- `class_id`。
- `class_name`。
- `score`。
- 显式 `box_mode`。
- HBB 的 `xyxy`，或 OBB 的当前标准物理表示。

HBB 和 OBB 使用不同的 box 数据类型或显式判别联合，避免同一个无语义 tensor 承载两种几何。

### 8.2 Prediction

单张图像结果包含：

- `image_id` 和输入来源。
- 原图尺寸。
- 检测结果集合。
- preprocess、inference 和 postprocess 耗时。

### 8.3 PredictionCollection

批量结果提供：

- 按阈值和类别过滤。
- 保存可视化图片。
- 导出 JSON。
- OBB 导出 DOTA txt。
- HBB 的通用 JSON 表示。

OBB polygon 转换必须委托现有已验证几何函数。首版不改变角度范围、宽高交换、周期 seam 或其他数学契约。

## 9. 错误处理

应用层定义以下稳定错误：

- `AppConfigError`：简化配置缺失、冲突或类型错误。
- `AdapterConfigurationError`：preset 或映射无法满足当前算法。
- `CheckpointCompatibilityError`：checkpoint 与任务、box mode 或类别不兼容。
- `InputSourceError`：图片或目录输入无效。
- `InferenceBackendError`：推理后端不可用或执行失败。
- `ExportError`：导出失败或格式暂不支持。

CLI 输出简短原因、相关路径和修复建议，并返回非零退出码。Python API 保留原始异常作为 cause。模型内部异常不得被静默吞掉。

## 10. 兼容策略

- 现有完整 YAML 和 `YAMLConfig` 保持可用。
- 现有 `train.py`、推理脚本和导出脚本首版不删除。
- 首版新入口可以委托现有训练、评估和导出实现，但不得通过 subprocess 调用自身仓库脚本。
- 新旧推理入口应通过固定样例验证结果一致。
- 底层算法优化不要求同步修改应用层；只有适配器或 preset 映射需要适应算法变化。
- 不为尚未发布的应用配置草案增加兼容迁移器；首版稳定后再按实际需求版本化。

## 11. 测试策略

### 11.1 配置测试

- 应用 YAML 的必填字段、默认值、类型和 `__include__` 继承。
- CLI、应用 YAML、应用基础 YAML 和算法 preset 的覆盖优先级。
- HBB COCO、OBB DOTA、OBB YOLO-OBB 的映射快照。
- 数据路径、格式、类别、box mode 和 output format 的启动前校验。
- `input_size` 多站点同步映射。
- pretrained/resume 互斥和总 batch size 校验。

### 11.2 依赖边界测试

- `engine` 不得导入 `deim_app`。
- 应用 facade、CLI 和预测对象不得导入 decoder、criterion 或 matcher。
- 只有 adapter 包允许接触底层具体构建和输出结构。

### 11.3 推理契约测试

- HBB 和 OBB box 类型显式且不可混用。
- 固定输入下，新旧 PyTorch 推理结果在批准容差内一致。
- OBB polygon 与现有几何函数一致。
- 可视化阈值不影响全量结果导出。
- Python API 与 CLI 产生等价的结构化结果。

### 11.4 最小集成测试

使用微型 HBB 和 OBB 配置分别执行：

```text
resolve config -> build -> load -> infer -> write results
```

训练侧至少执行一次配置解析和现有训练入口的 smoke test。首版不要求通过重构训练循环来完成该测试。

## 12. 首版实施顺序

```text
1. 应用契约和依赖守卫
2. 应用 YAML schema、继承与 preset 映射
3. DeimDetectionAdapter
4. Prediction 数据类型与 writer
5. Python 推理 API
6. infer CLI
7. train / eval / export 轻量应用入口
8. HBB/OBB 新旧路径回归验证
```

首版完成后，再根据真实使用反馈决定 ONNX、多后端、视频、服务化或训练生命周期重构的优先级。

## 13. 首版验收标准

1. 用户能通过一份可继承的应用 YAML 配置 HBB COCO、OBB DOTA 或 OBB YOLO-OBB 任务。
2. 用户无需理解底层完整 YAML 即可运行训练、评估和 PyTorch 推理。
3. 首版公开参数白名单和 CLI 覆盖优先级有自动测试，算法内部字段无法从应用入口任意覆盖。
4. Python API 和 CLI 共用同一推理实现。
5. Python API 返回结构化、显式区分 HBB/OBB 的预测对象。
6. 推理结果支持 JSON、OBB DOTA txt 和可视化输出。
7. 固定 HBB/OBB 输入的新旧 PyTorch 推理结果满足批准容差。
8. 底层 `engine` 不依赖应用层，算法变化集中在 adapter 和 preset 映射内。
9. 旧完整 YAML、`train.py` 和现有工具仍可使用。
10. 首版没有引入训练内部大重构、动态插件系统或多后端复杂度。

## 14. 与原平台设计的关系

`2026-08-05-engineering-platform-refactor-design.md` 保留为完整工程平台的远期参考。本设计覆盖并替代其首期实施优先级：

- 简化配置和推理应用成为主线。
- 任务适配器成为应用层与算法层的正式边界。
- 训练控制器、事件系统、复杂 checkpoint 管理和依赖拆分延后。
- 后续迭代只有在真实需求出现时才从原设计中选取能力实施。

若两份文档在首版范围或顺序上冲突，以本设计为准。
