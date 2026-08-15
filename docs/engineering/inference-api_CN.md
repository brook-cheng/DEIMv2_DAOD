# 推理 API 与 CLI 指南（`deim_app`）

[English](inference-api.md) | 简体中文

本文档说明 **第一版** `deim_app` 推理 API、CLI、输出格式和手动冒烟测试。

应用 YAML schema 见 [application-config_CN.md](application-config_CN.md)。

---

## 1. Python API 快速上手

```python
from deim_app import DetectionModel

model = DetectionModel.from_config("configs/app/examples/obb_yolo.yml")
model.load("outputs/model.pth")
full = model.predict("images/")
filtered = full.filter(score_threshold=0.25).top_k(300)
filtered.export_json("outputs/predictions.json")
filtered.export_dota("outputs/dota")
filtered.save_images("outputs/visualization")
```

* `from_config` **不会**加载 checkpoint —— 之后需调用 `load`。它返回 `self`，
  因此推荐链式写法 `DetectionModel.from_config(...).load(...)`。
* `predict` 返回**完整未过滤**的集合；`predict_filtered` 依次应用
  `score_threshold` → `class_filter` → `top_k`（默认取自加载配置的 `inference`
  段），并返回不可变视图。
* 过滤操作不会修改完整集合 —— 两个视图并存。
* `class_filter` 中的类别名在推理前依据数据集元数据校验，未知名称会抛出
  `AppConfigError`。

### 1.1 使用配置默认值过滤推理

```python
model = DetectionModel.from_config("configs/app/examples/obb_yolo.yml").load("model.pth")

# Uses inference.score_threshold / top_k / class_filter from the YAML by default.
results = model.predict_filtered("images/")
results.export_dota("outputs/dota")
```

### 1.2 调用时显式覆盖

```python
results = model.predict_filtered(
    "images/",
    score_threshold=0.5,
    top_k=100,
    class_filter=("car", "truck"),
)
```

### 1.3 读取类型化预测结果

```python
full = model.predict("images/one.jpg")
img_pred = full.predictions[0]
for det in img_pred.detections:
    print(det.class_id, det.class_name, det.score)
    if full.box_mode == "obb":
        cx, cy, w, h, theta = det.xywhr
    else:
        x1, y1, x2, y2 = det.xyxy
```

---

## 2. CLI 参考

入口为 `python -m deim_app`，提供四个子命令：`train`、`eval`、`infer`、
`export`。未知 flag 由 argparse 以退出码 2 拒绝。

### 2.1 `infer`

```bash
python -m deim_app infer \
  -c configs/app/examples/obb_yolo.yml \
  -r /path/to/model.pth \
  -i /path/to/images \
  -o /tmp/deim-app-output \
  --score-threshold 0.25 \
  --format json dota visualization
```

Flags：

| Flag                    | 用途                                                                |
|-------------------------|---------------------------------------------------------------------|
| `-c` / `--config`       | 应用 YAML 路径（必填）。                                            |
| `-r` / `--checkpoint`   | 模型 checkpoint 路径（必填）。                                      |
| `-i` / `--input`        | 图片文件或目录（必填）。                                            |
| `-o` / `--output`       | 输出目录（必填）。                                                  |
| `--device`              | 推理设备（默认取自 YAML `inference.device`）。                      |
| `--batch-size`          | 推理批次大小（默认取自 YAML `inference.batch_size`）。              |
| `--score-threshold`     | 置信度阈值（默认取自 YAML `inference.score_threshold`）。           |
| `--top-k`               | Top-k 上限（默认取自 YAML `inference.top_k`）。                     |
| `--class-filter`        | 可选的类别名称白名单（可重复）。                                    |
| `--format`              | 输出格式：`json`、`dota`、`visualization` 的子集（可重复）。        |

### 2.2 `train`

```bash
python -m deim_app train \
  -c configs/app/examples/hbb_coco.yml \
  --device cuda \
  [--resume /path/to/checkpoint.pth] \
  [--output-dir ./outputs/my-run]
```

训练调用引擎 solver 的 `fit()` —— 训练循环与 `tools/train/train.py` 入口保持一致。
`--resume` 与 YAML 中的 `train.pretrained` 互斥。

### 2.3 `eval`

```bash
python -m deim_app eval \
  -c configs/app/examples/hbb_coco.yml \
  -r /path/to/model.pth \
  [--device cuda]
```

评估调用引擎 solver 的 `val()`。checkpoint 通过 solver 的 resume 路径加载，
评估对验证集完整跑一遍。

### 2.4 `export`

```bash
python -m deim_app export \
  -c configs/app/examples/hbb_coco.yml \
  -r /path/to/model.pth \
  -o /tmp/export-output \
  --format onnx \
  [--device cpu]
```

> **注意：** 首个应用层版本未启用任何导出格式，`export` 始终抛出
> `ExportError`。该命令预留给后续的 ONNX / OpenVINO / TensorRT 支持。

---

## 3. 输出格式

三个 writer 共享同一个不可变 `PredictionCollection`，可在单次调用中任意组合。

| 格式           | Writer                         | 输出                                                                                          |
|----------------|--------------------------------|-----------------------------------------------------------------------------------------------|
| `json`         | `write_json`                   | 一个 `predictions.json` 文件，每张图片一条记录（labels、boxes、scores、class_names）。        |
| `dota`         | `write_dota`                   | 每张图片一个 `<image_stem>.txt`，每行一个 OBB 检测结果：`x1 y1 x2 y2 x3 y3 x4 y4 class score`。仅限 OBB。 |
| `visualization`| `write_visualization`          | 每张图片一个 `<image_stem>.{png,jpg}`，在原图上叠加框/多边形。                               |

`json` 和 `visualization` 同时适用于 HBB 与 OBB；`dota` 仅限 OBB。

---

## 4. 手动端到端冒烟测试

使用以下命令检查 OBB 管线。命令只运行一次推理，并输出全部三种格式：

```bash
python -m deim_app infer \
  -c configs/app/examples/obb_yolo.yml \
  -r /path/to/model.pth \
  -i /path/to/images \
  -o /tmp/deim-app-output \
  --score-threshold 0.25 \
  --format json dota visualization
```

`/tmp/deim-app-output/` 下的预期输出：

```
/tmp/deim-app-output/
├── predictions.json        # JSON writer
├── dota/
│   ├── image_001.txt       # one per image
│   └── image_002.txt
└── visualization/
    ├── image_001.png       # one per image
    └── image_002.png
```

验证项：

1. `predictions.json` 可解析，每张输入图片一条记录，且每个检测结果的
   `class_name` 均来自 `classes.txt`。
2. 每个 `dota/*.txt` 行的格式为 `x1 y1 x2 y2 x3 y3 x4 y4 class score`
   （以空白分隔，共 10 个字段），坐标采用原图像素坐标。
3. 每个 `visualization/*.png` 是叠加了旋转多边形的原图，且分数高于
   `--score-threshold`。

这项**手动**冒烟测试需要本地 checkpoint 与数据集。该路径与旧工具
（`tools/inference/torch_inf.py`、
`tools/compare/core.py`）的数值一致性，由 `test/deim_app/test_legacy_parity.py`
在 `DEIM_APP_PARITY_*` 环境变量就绪时自动校验。
