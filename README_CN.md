# deimv2_obb 中文文档

[English](README.md) | 简体中文

`deimv2_obb` 是基于 [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) 的 fork，专注于**旋转框检测（OBB）**与统一的**应用层（`deim_app`）**——训练、评估与推理。

上游 [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) 提供基础框架：基于 DINOv3 / HGNetv2 主干的实时检测、官方模型系列与引擎级训练配方。本 fork ：

- **加固 OBB 旋转框流水线**：角度表示契约、稳定 atan2 反向传播、hw-order 一致的坐标约定；
- **新增数据集无关的应用层（`deim_app`）**：统一的 `train / eval / infer` CLI 与 `DetectionModel` Python API，覆盖精选应用配置预设（HBB COCO、OBB DOTA / YOLO-OBB）；
- **重组工具**并按领域划分 pytest 套件（`contracts / obb / eval / engine / deim_app`）。

> **上游 DEIMv2。** 官方模型库、预训练权重、引擎级训练 / 测试 / 调优配方、部署与基准工具、上游变更历史均在
> [原始 DEIMv2 仓库](https://github.com/Intellindust-AI-Lab/DEIMv2) 中。本仓库专注于上述 OBB / 应用层新增内容，其余部分以上游为准。

## Release Notes — v1.0.0

本 fork 的首个生产版本。工程线亮点（基于上游 DEIMv2）：

- **OBB 流水线加固**：旋转框训练/评估采用 `shifted_v1` checkpoint 契约（所有加载路径显式拒绝无标记的历史 checkpoint）、严格的 `angle_rep ∈ {0, 3}` 构造，以及按配置的契约测试套件。
- **应用层（`deim_app`）**：统一的 `train / eval / infer` CLI + `DetectionModel` Python API，覆盖精选应用配置预设（HBB COCO、OBB DOTA/YOLO），含类别数校验、checkpoint 门禁与 JSON / DOTA / 可视化输出。
- **工具重组**：`tools/` 下的可执行入口（`train`、`inference`、`compare` 研究流水线、`analysis`），pytest 套件按领域分组（`contracts / obb / eval / engine / deim_app`）。
- **安全加固**：应用层使用 `weights_only=True` checkpoint 反序列化；脚本中的凭据移除（仅环境变量）。
- **多 GPU 诊断支持**：环境门控的逐 rank 迭代心跳、SIGUSR1 栈转储、通过 `scripts/diag_2gpu.sh` 打包 NCCL flight-recorder。
- **真实数据验收通过**（单 GPU 硬件）：17 用例矩阵（训练、评估、推理、API 一致性、负路径）；验收期间发现并修复 7 个真实集成缺陷。

---

## 快速开始

`deim_app` 通过每个数据集一个 YAML 配置运行训练、评估与推理。完整 CLI 与 Python API 见 [推理 API 与 CLI 指南](docs/engineering/inference-api_CN.md)；YAML 结构见 [应用配置指南](docs/engineering/application-config_CN.md)。

### 环境安装

```shell
conda create -n deimv2 python=3.11 -y
conda activate deimv2
pip install -r requirements.txt
```

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

### 模型导出

应用层导出（`python -m deim_app export`）为预留接口，v1 版本尚未启用，调用会抛出 `ExportError`。需要 ONNX / TensorRT 请使用上游
[DEIMv2 仓库](https://github.com/Intellindust-AI-Lab/DEIMv2) 的部署工具（`tools/deployment/export_onnx.py` 与 `trtexec`）。

## 应用层（deim_app）

`deim_app` 工作流见 [快速开始](#快速开始)。完整参考见 [应用配置指南](docs/engineering/application-config_CN.md) 与 [推理 API 与 CLI 指南](docs/engineering/inference-api_CN.md)。

## 项目工具

本 fork 重组的工具：

- `tools/train`、`tools/inference` —— 可执行入口
- `tools/compare` —— 模型对比研究流水线
- `tools/analysis` —— 分析工具（如 OBB 误差分类、rep2 NaN 诊断）
- pytest 套件按领域分组：`contracts / obb / eval / engine / deim_app`

部署（ONNX / TensorRT）、基准测试与可视化工具继承自上游——见
[DEIMv2 仓库](https://github.com/Intellindust-AI-Lab/DEIMv2)。

## 上游 DEIMv2

本项目基于 [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)。官方模型库与权重、引擎级训练 / 测试 / 调优配方、数据准备指南与上游变更历史，请参考原始仓库：
**[github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)**。

## 引用

如果您在研究中使用了 `DEIMv2` 或其方法，请引用以下 BibTeX：

<details open>
<summary> bibtex </summary>

```latex
@article{huang2025deimv2,
  title={Real-Time Object Detection Meets DINOv3},
  author={Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
  journal={arXiv},
  year={2025}
}

```
</details>

## 致谢

我们的工作基于 [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)、
[D-FINE](https://github.com/Peterande/D-FINE)、
[RT-DETR](https://github.com/lyuwenyu/RT-DETR)、
[DEIM](https://github.com/ShihuaHuang95/DEIM) 与
[DINOv3](https://github.com/facebookresearch/dinov3)。感谢他们的杰出工作！

✨ 欢迎贡献，有任何问题请随时联系我们！✨
