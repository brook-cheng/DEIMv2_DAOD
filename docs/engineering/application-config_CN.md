# 应用配置指南（`deim_app`）

[English](application-config.md) | 简体中文

本文档说明 `deim_app` **第一版**应用配置契约，包括公开 YAML schema、继承与覆盖优先级、归预设管理的参数，以及 `pretrained` 与 `resume` 的语义。

推理的 Python API 与 CLI 见 [inference-api_CN.md](inference-api_CN.md)。

---

## 1. 公开字段表

应用 YAML **只接受**以下六个顶层节。任何其他键（如 `DEIMTransformer`、`optimizer`、`angle_rep`、`num_classes`、`epoches` 等算法节）在构造引擎对象之前就会被加载器拒绝。该白名单由 `deim_app.config.loader.validate_public_keys` 强制校验。

### `project`

| 键           | 类型   | 默认值          | 说明                                                     |
|--------------|--------|-----------------|----------------------------------------------------------|
| `name`       | string | `"deim-app"`    | 自由格式的项目标签（仅供参考）。                       |
| `output_dir` | string | `"outputs"`     | checkpoint / 日志的输出目录。映射到引擎顶层 `output_dir`。|

### `runtime`

| 键          | 类型         | 默认值       | 说明                                                                                     |
|-------------|--------------|--------------|------------------------------------------------------------------------------------------|
| `input_size`| `[int, int]` | `[640, 640]` | 推理/训练分辨率 `[H, W]`，会同步到每个 `Resize`/`OBBResize` 算子、`collate_fn.base_size` 以及 `eval_spatial_size`。 |
| `seed`      | integer      | `42`         | 引擎 `seed`。                                                                            |

### `data`

| 键                 | 类型         | 默认值 | 说明                                                                                       |
|--------------------|--------------|--------|--------------------------------------------------------------------------------------------|
| `format`           | enum         | `COCO` | `COCO`、`DOTA`、`YOLO-OBB` 之一。选择数据集类与元数据加载器。                              |
| `train_images`     | string       | `""`   | 训练图像目录（COCO/HBB）或图像文件夹（OBB）。                                              |
| `train_annotations`| string       | `""`   | COCO JSON 文件（COCO）**或**标签文件夹（DOTA/YOLO-OBB）。                                  |
| `val_images`       | string       | `""`   | 验证图像目录。                                                                             |
| `val_annotations`  | string       | `""`   | 验证标注路径（文件或文件夹，视 `format` 而定）。                                            |
| `classes_file`     | string/null | `null` | **OBB 必填**。每行一个类别名。COCO 不使用此项（类别来自 JSON）。                              |
| `num_workers`      | non-neg int  | `4`    | DataLoader 工作进程数。                                                                    |
| `cache_images`     | enum         | `none` | `none` / `disk` / `ram`。**COCO 必须为 `none`**；OBB 三种均可。                             |

### `train`

| 键                            | 类型         | 默认值  | 说明                                                                                  |
|-------------------------------|--------------|---------|---------------------------------------------------------------------------------------|
| `epochs`                      | positive int | `100`   | 总训练轮数。映射到引擎 `epoches`。                                                     |
| `batch_size`                  | positive int | `8`     | 映射到 `train_dataloader.total_batch_size`。                                           |
| `learning_rate`               | float        | `1.0e-4`| **仅**映射到 `optimizer.lr`。各参数组的学习率归预设管理，用户配置不会改动。                     |
| `device`                      | string       | `"cuda"`| 训练时模型所在的设备。                                                                  |
| `amp`                         | bool         | `True`  | 混合精度开关（映射到 `use_amp`）。                                                      |
| `pretrained`                  | string/null  | `null`  | 用于初始化主干的权重路径。与 `resume` 互斥。见 §6。                                            |
| `resume`                      | string/null  | `null`  | 用于恢复完整训练状态的 checkpoint。与 `pretrained` 互斥。见 §6。                             |
| `early_stopping.enabled`      | bool         | `False` | 早停开关（覆盖预设）。                                                                  |
| `early_stopping.patience`     | pos int      | `10`    | 早停 patience（覆盖预设）。其余早停字段归预设管理。                                        |

### `evaluation`

| 键          | 类型         | 默认值  | 说明                                      |
|-------------|--------------|---------|-------------------------------------------|
| `batch_size`| positive int | `8`     | 映射到 `val_dataloader.total_batch_size`。|
| `device`    | string       | `"cuda"`| 评估阶段使用的设备。                      |

### `inference`

| 键                | 类型                  | 默认值                    | 说明                                                                 |
|-------------------|-----------------------|---------------------------|----------------------------------------------------------------------|
| `checkpoint`      | string/null           | `null`                    | 默认 checkpoint 路径（仅供参考；实际通过 `load()` 或 `-r` 传入）。    |
| `device`          | string                | `"cuda"`                  | 推理设备。                                                           |
| `batch_size`      | positive int          | `1`                       | 推理批大小。                                                         |
| `score_threshold` | float ∈ `[0, 1]`      | `0.25`                    | `predict_filtered` 应用的分数阈值。                                  |
| `top_k`           | positive int          | `300`                     | `predict_filtered` 应用的每图 top-k 上限。                           |
| `class_filter`    | list[str]/null        | `null`                    | 可选的类别名白名单（按元数据校验有效性）。                             |
| `output_formats`  | list[str]             | `["json","visualization"]`| `json`、`dota`、`visualization` 的子集。                             |

---

## 2. 继承与覆盖优先级

解析顺序如下（越靠后优先级越高）：

```
1. 算法预设          (configs/app/presets/*.yml)
2. 应用基础 YAML     (configs/app/base/{hbb,obb}_app.yml)
3. 用户应用 YAML     (用户配置文件，通过 __include__ 引入基础文件)
4. CLI 覆盖          (-r / --score-threshold 等)
```

即：**`CLI > 用户 YAML > 应用基础 > 算法预设`**。

规则：

* 用户 YAML **必须**声明且只能声明一个 `__include__:`，指向两个受支持的应用基础文件之一（`configs/app/base/hbb_app.yml` 或 `obb_app.yml`）。直接引入预设会被拒绝。
* 用户、基础、CLI 输入中只允许出现这六个公开节。
* 深度合并：嵌套键（如 `train.early_stopping.patience`）只替换对应的叶子节点，其余兄弟节点保持不变。
* 算法预设通过引擎的 `__include__` 链只加载一次，随后成为不可变的算法契约；解析器只修改 §1 中列出的字段，其余内容原样保留。

---

## 3. 示例

### 3.1 HBB / MS COCO

```yaml
# configs/app/examples/hbb_coco.yml
__include__:
  - ../base/hbb_app.yml

project:
  name: deimv2-coco
  output_dir: ./outputs/deimv2-coco

runtime:
  input_size: [640, 640]
  seed: 42

data:
  format: COCO
  train_images: /data/COCO/train2017/
  train_annotations: /data/COCO/annotations/instances_train2017.json
  val_images: /data/COCO/val2017/
  val_annotations: /data/COCO/annotations/instances_val2017.json
  num_workers: 4
  cache_images: none

train:
  epochs: 68
  batch_size: 4
  learning_rate: 5.0e-4
  device: cuda
  amp: True
```

### 3.2 OBB / DOTA

```yaml
# configs/app/examples/obb_dota.yml
__include__:
  - ../base/obb_app.yml

project:
  name: deimv2-dota
  output_dir: ./outputs/deimv2-dota

runtime:
  input_size: [640, 640]
  seed: 42

data:
  format: DOTA
  train_images: /data/DOTA/train/images/
  train_annotations: /data/DOTA/train/labels/
  val_images: /data/DOTA/val/images/
  val_annotations: /data/DOTA/val/labels/
  classes_file: /data/DOTA/classes.txt
  num_workers: 2
  cache_images: disk

train:
  epochs: 200
  batch_size: 4
  learning_rate: 5.0e-4
  device: cuda
  amp: True
```

### 3.3 OBB / YOLO-OBB

```yaml
# configs/app/examples/obb_yolo.yml
__include__:
  - ../base/obb_app.yml

project:
  name: deimv2-yolo-obb
  output_dir: ./outputs/deimv2-yolo-obb

runtime:
  input_size: [640, 640]
  seed: 42

data:
  format: YOLO-OBB
  train_images: /data/yolo_obb/train/images/
  train_annotations: /data/yolo_obb/train/labels/
  val_images: /data/yolo_obb/val/images/
  val_annotations: /data/yolo_obb/val/labels/
  classes_file: /data/yolo_obb/classes.txt
  num_workers: 2
  cache_images: disk

train:
  epochs: 200
  batch_size: 4
  learning_rate: 5.0e-4
  device: cuda
  amp: True
```

---

## 4. 类别元数据来源

`num_classes` **始终**由标注文件的元数据推导得出，**绝不手工填写**。具体来源取决于 `data.format`：

| 格式       | 来源                                           | `num_classes`               | 类别名                                  |
|------------|------------------------------------------------|-----------------------------|-----------------------------------------|
| `COCO`     | `data.train_annotations`（一个 `instances_*.json`）| `len(categories)`（在 remap 决策之后）| `categories[*].name`（按 id 排序）      |
| `DOTA`     | `data.classes_file`                            | 行数                        | 每个非空行一个名称（行序 → 标签）        |
| `YOLO-OBB` | `data.classes_file`                            | 行数                        | 每个非空行一个名称（行序 → 标签）        |

推导结果写入引擎顶层 `num_classes`，并同步到模型的类别预测头。

### 4.1 `remap_mscoco_category` 自动检测

对于 COCO，解析器会自动判断是否需要应用标准的 MS COCO `1..90 → 0..79` 重映射：

* 若 JSON 的类别**名称集合**等于标准 MS COCO 80 类集合**且**其 **id 集合**等于 `{1, 2, …, 90}`（存在空隙），则自动应用重映射，引擎侧的 `remap_mscoco_category` 会被置为 `True`。
* 否则解析器要求 id 从 0 开始且连续（`0, 1, …, N-1`），遇到不连续的 id 会抛出可操作的错误并终止。

这避免了一类静默的标签错乱 bug：过去 id 与 MS COCO 范围重叠的自定义数据集会被意外重映射。如需覆盖自动检测结果，可在算法预设中显式设置 `remap_mscoco_category`（极少需要）。

---

## 5. 归预设管理的参数

以下类别**绝不**出现在公开 schema 中——它们归算法预设管理，不受用户/基础/CLI 输入影响：

| 类别                  | 代表性键                                                                                                      |
|-----------------------|---------------------------------------------------------------------------------------------------------------|
| 主干 / 编码器 / 解码器结构 | `DEIM.backbone`、`DINOv3STAs.*`、`DINOv3STAsResAtten.*`、`HybridEncoder.{in_channels,hidden_dim,dim_feedforward}`、`DEIMTransformer.{feat_channels,hidden_dim,num_layers,dim_feedforward,eval_idx}` |
| OBB 角度契约          | `DEIMTransformer.{angle_rep,box_mode}`、`PostProcessor.box_mode`、`DEIMCriterion.{box_mode,obbox_rep_dim}`——`angle_rep` 仅接受整数 `0` 或 `3`（布尔/浮点在解码器构造时被拒绝）|
| 优化器结构            | `optimizer.type`、`optimizer.betas`、`optimizer.weight_decay`、每个 `optimizer.params[*]` 组（正则 + 每组 `lr`/`weight_decay`）|
| 调度器 / 训练阶段     | `lrsheduler`、`lr_gamma`、`warmup_iter`、`flat_epoch`、`no_aug_epoch`、`epoches`（镜像 `train.epochs`）、`clip_max_norm`、`use_ema`、`ema.*` |
| 数据增强              | Mosaic / Mixup / CopyBlend 算子 + 概率 + 策略轮数、`collate_fn.{base_size_repeat,mixup_epochs,copyblend_epochs,…}` |
| 损失 / 匹配器权重     | `DEIMCriterion.weight_dict.*`、`DEIMCriterion.{reg_max,gamma,alpha,angle_lambda,…}`、`DEIMCriterion.matcher.*` |
| 派生字段              | `num_classes`（从元数据推导——见 §4）、`eval_spatial_size`（镜像 `runtime.input_size`）、`checkpoint_freq` |
| 早停预设字段          | `early_stopping.{metric,mode,min_epochs,min_delta,restore_best}`——只有 `enabled` 和 `patience` 可由用户覆盖 |

主学习率覆盖（`train.learning_rate`）是公开 schema 中**唯一**的优化器字段；它只修改 `optimizer.lr`，不会改动各参数组的学习率。

---

## 6. `pretrained` 与 `resume` 的区别

两个字段都接收 checkpoint 路径，但语义不同且**互斥**——同时设置两者会在加载时抛出 `AppConfigError`。

| 字段         | 引擎键     | 语义                                                                                                                                                                |
|--------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `pretrained` | `tuning`   | **仅初始化主干。** 只加载主干初始化所需的权重（通常是 ImageNet 权重），其余部分（优化器、EMA 状态等）均不恢复，训练从头开始。适用于用预训练主干开启一次全新训练。 |
| `resume`     | `resume`   | **完整训练状态。** 恢复模型 + 优化器 + EMA + 调度器 + epoch 计数器，训练从 checkpoint 的断点处继续。                                                                |

这一互斥校验避免了常见错误：在同一次运行中同时要求主干初始化和完整状态恢复——否则二者的执行顺序未定义且不会报错。

---

## 7. 验证

这些结构性约束由 `test/deim_app/test_legacy_parity.py` 和 `test/deim_app/` 测试套件校验。运行命令：

```bash
pytest test/deim_app/ -v
```
