# DEIMv2 中文文档

[English](README.md) | 简体中文

DEIMv2 基于 [DEIM](https://github.com/ShihuaHuang95/DEIM)，使用 DINOv3 作为主干，覆盖从轻量到高精度的多个尺寸。S 尺寸在 COCO 上 AP 50.9。论文：[arXiv:2509.20787](https://arxiv.org/abs/2509.20787)，主页：[DEIMv2](https://intellindust-ai-lab.github.io/projects/DEIMv2/)。

权重下载表、更新记录、引擎级用法与引用信息见 [English README](README.md)。

## 1. 模型系列一览

### 轻量系列（HGNetv2）

Atto、Femto、Pico、N 基于 HGNetv2 主干，参数量 0.5M–3.6M，适合资源受限或实时场景。

### 精度系列（DINOv3）

S、M、L、X 基于 DINOv3 主干，参数量 9.7M–50.3M，精度更高。

### 应用预设（HBB / OBB）

`configs/app/presets/` 提供两个算法预设：

| 预设 | 主干 | 框类型 | 数据集格式 |
| :--- | :--- | :--- | :--- |
| `deimv2_dinov3_sp_hbb.yml` | DINOv3 ViT-S/16 + STA | HBB 水平框 | COCO |
| `deimv2_dinov3_sp_obb.yml` | DINOv3 ViT-S/16+ + STA | OBB 旋转框 | DOTA / YOLO-OBB |

- HBB 路径面向 COCO 格式数据集，输出水平框（`CocoEvaluator`）。
- OBB 路径面向 DOTA 与 YOLO-OBB 格式数据集，输出旋转框（`box_mode: obb`）。

## 2. Model Zoo 摘要（COCO）

| 型号 | 系列 | AP | Params | GFLOPs | 延迟 (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Atto** | HGNetv2 | **23.8** | 0.5M | 0.8 | 1.10 |
| **Femto** | HGNetv2 | **31.0** | 1.0M | 1.7 | 1.45 |
| **Pico** | HGNetv2 | **38.5** | 1.5M | 5.2 | 2.13 |
| **N** | HGNetv2 | **43.0** | 3.6M | 6.8 | 2.32 |
| **S** | DINOv3 | **50.9** | 9.7M | 25.6 | 5.78 |
| **M** | DINOv3 | **53.0** | 18.1M | 52.2 | 8.80 |
| **L** | DINOv3 | **56.0** | 32.2M | 96.7 | 10.47 |
| **X** | DINOv3 | **57.8** | 50.3M | 151.6 | 13.75 |

以上均为 COCO 指标。含配置文件、权重与日志下载链接的完整表格见 [English README · Model Zoo](README.md#3-model-zoo)。OBB 预设暂无公开的官方 AP 数值，请以实际评测为准。

## 3. 快速开始

### 环境安装

```shell
conda create -n deimv2 python=3.11 -y
conda activate deimv2
pip install -r requirements.txt
```

DINOv3 骨干预训练权重需放在 `./ckpts/`（按 [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) 官方指引下载），预设中已引用 `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`（HBB）与 `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`（OBB）。

### 选择并修改配置

`configs/app/examples/` 下有三个可直接使用的示例：

| 示例配置 | 用途 | 标注格式 |
| :--- | :--- | :--- |
| `hbb_coco.yml` | HBB + COCO | COCO JSON |
| `obb_dota.yml` | OBB + DOTA | DOTA txt |
| `obb_yolo.yml` | OBB + YOLO-OBB | YOLO-OBB txt |

复制一份示例，将 `data` 下的路径替换为本地数据集路径，再按需修改 `runtime`、`train`、`inference` 等公共字段。字段说明见 [应用配置指南](docs/engineering/application-config_CN.md)。

注意事项：

- OBB 数据必须提供 `classes_file`（每行一个类别名，行序即标签编号），`num_classes` 由标注元数据自动推导，无需手动填写。
- COCO 需 `cache_images: none`（示例已配置）；OBB 支持 `none` / `disk` / `ram`。
- 从零训练可用 `train.pretrained` 初始化骨干权重；断点续训用 `train.resume` 或 CLI `--resume`，两者互斥。

### 训练

```bash
python -m deim_app train \
  -c configs/app/examples/hbb_coco.yml \
  --device cuda \
  [--resume /path/to/checkpoint.pth] \
  [--output-dir ./outputs/my-run]
```

### 评估

```bash
python -m deim_app eval \
  -c configs/app/examples/hbb_coco.yml \
  -r /path/to/model.pth \
  [--device cuda]
```

### 推理

HBB（JSON + 可视化）：

```bash
python -m deim_app infer \
  -c configs/app/examples/hbb_coco.yml \
  -r /path/to/model.pth \
  -i /path/to/images \
  -o /tmp/deim-app-output \
  --score-threshold 0.25 \
  --format json visualization
```

OBB（JSON + DOTA + 可视化）：

```bash
python -m deim_app infer \
  -c configs/app/examples/obb_yolo.yml \
  -r /path/to/model.pth \
  -i /path/to/images \
  -o /tmp/deim-app-output \
  --score-threshold 0.25 \
  --format json dota visualization
```

`json` 与 `visualization` 同时支持 HBB 与 OBB；`dota` 仅用于 OBB，每张图一个 txt，每行格式为 `x1 y1 x2 y2 x3 y3 x4 y4 class score`。

### Python API

```python
from deim_app import DetectionModel

model = DetectionModel.from_config("configs/app/examples/obb_yolo.yml")
model.load("outputs/model.pth")
results = model.predict_filtered("images/", score_threshold=0.25)
results.export_json("outputs/predictions.json")
results.export_dota("outputs/dota")
results.save_images("outputs/visualization")
```

`from_config` 不会自动加载权重，需显式调用 `load`（返回 `self`，可链式调用）。`predict_filtered` 默认使用配置中的 `score_threshold`、`class_filter` 与 `top_k`，也可在调用时覆盖筛选参数。

## 4. 模型导出

应用层导出（`python -m deim_app export`）为预留接口，v1 版本尚未启用，调用会抛出 `ExportError`。需要 ONNX / TensorRT 请使用仓库现有的部署工具（基于引擎配置）：

```shell
pip install onnx onnxsim
python tools/deployment/export_onnx.py --check -c configs/deimv2/deimv2_dinov3_${model}_coco.yml -r model.pth

# TensorRT
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

## 5. 更多文档

- [application-config_CN.md](docs/engineering/application-config_CN.md)：应用层 YAML 配置指南（公共字段、继承优先级、类元数据来源）
- [inference-api_CN.md](docs/engineering/inference-api_CN.md)：推理 Python API 与 CLI 完整参考（子命令、参数、输出格式）
- [English README](README.md)：Model Zoo 完整表格与权重下载、引擎级训练/调优用法、工具、引用与致谢
