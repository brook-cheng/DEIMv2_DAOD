# deimv2_obb

Real-Time Oriented Object Detection on DEIMv2

<p align="center">
    English | <a href="./README_CN.md">简体中文</a>
</p>

`deimv2_obb` is a fork of [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)
focused on **oriented object detection (OBB)** and a unified application layer
(`deim_app`) for training, evaluation, and inference.

The upstream [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) provides the
base framework — real-time detection built on DINOv3 / HGNetv2 backbones, the
official model families, and the engine-level training recipes. This fork:

- **hardens the oriented-box (OBB) pipeline** for rotated-box training and
  evaluation (angle-representation contract, stable atan2 backward, hw-order
  consistent coordinates);
- **adds a dataset-neutral application layer** (`deim_app`) with a unified
  `train / eval / infer` CLI and a `DetectionModel` Python API over curated
  app-config presets (HBB COCO, OBB DOTA / YOLO-OBB);
- **reorganizes tooling** and groups the pytest suite by domain
  (`contracts / obb / eval / engine / deim_app`).

> **Upstream DEIMv2.** The official model zoo, pretrained weights, engine-level
> training / testing / tuning recipes, deployment and benchmark tools, and the
> upstream change history all live in the
> [original DEIMv2 repository](https://github.com/Intellindust-AI-Lab/DEIMv2).
> This repository focuses on the OBB / application-layer additions listed
> above; everything else is delegated to upstream.

---

## Release Notes — v1.0.0

First production release of this fork. Highlights of the engineering track
(built on top of upstream DEIMv2):

- **OBB pipeline hardened**: rotated-box training/eval with the
  `shifted_v1` checkpoint contract (unmarked legacy checkpoints are
  explicitly rejected on every load path), strict `angle_rep ∈ {0, 3}`
  construction, and per-config contract test suites.
- **Application layer** (`deim_app`): unified `train / eval / infer` CLI +
  `DetectionModel` Python API over curated app-config presets (HBB COCO and
  OBB DOTA/YOLO), with class-count validation, checkpoint gating, and JSON /
  DOTA / visualization writers.
- **Tooling reorganized**: executable entry points under `tools/`
  (`train`, `inference`, `compare` research pipeline, `analysis`), pytest
  suite grouped by domain (`contracts / obb / eval / engine / deim_app`).
- **Security hardening**: `weights_only=True` checkpoint deserialization in
  the application layer; credentials removed from scripts (environment
  only).
- **Multi-GPU diagnosis support**: env-gated per-rank iteration heartbeat,
  SIGUSR1 stack dumps, NCCL flight-recorder bundling via
  `scripts/diag_2gpu.sh`.
- **Real-data acceptance passed** on single-GPU hardware: 17-case matrix
  (training, evaluation, inference, API parity, negative paths); seven
  real integration defects found and fixed during acceptance.

---

## Quick Start

`deim_app` runs training, evaluation, and inference from one YAML config per
dataset. For the full CLI and Python API, see the
[Inference API & CLI Guide](docs/engineering/inference-api.md); for the YAML
schema, see the [Application Config Guide](docs/engineering/application-config.md).

### Setup

```shell
conda create -n deimv2 python=3.11 -y
conda activate deimv2
pip install -r requirements.txt
```

### Choose a config

Three examples cover the supported formats:

| Example | Format | Box mode |
| :--- | :--- | :---: |
| [hbb_coco.yml](./configs/app/examples/hbb_coco.yml) | COCO | HBB |
| [obb_dota.yml](./configs/app/examples/obb_dota.yml) | DOTA | OBB |
| [obb_yolo.yml](./configs/app/examples/obb_yolo.yml) | YOLO-OBB | OBB |

Copy the example that matches your dataset and edit its `data` section: point
`train_images`, `train_annotations`, `val_images`, and `val_annotations` at
your paths. OBB examples also require `classes_file` (one class name per
line); COCO class names come from the annotation JSON instead.

### Train

```shell
python -m deim_app train \
  -c configs/app/examples/hbb_coco.yml \
  --device cuda \
  [--resume /path/to/checkpoint.pth] \
  [--output-dir ./outputs/my-run]
```

`--resume` continues an interrupted run from a full training-state checkpoint
and is mutually exclusive with `train.pretrained` in the YAML.

### Evaluate

```shell
python -m deim_app eval \
  -c configs/app/examples/hbb_coco.yml \
  -r /path/to/model.pth \
  [--device cuda]
```

### Infer

HBB, JSON and visualization:

```shell
python -m deim_app infer \
  -c configs/app/examples/hbb_coco.yml \
  -r /path/to/model.pth \
  -i /path/to/images \
  -o /tmp/deim-app-output \
  --score-threshold 0.25 \
  --format json visualization
```

OBB, JSON plus DOTA labels and visualization:

```shell
python -m deim_app infer \
  -c configs/app/examples/obb_yolo.yml \
  -r /path/to/model.pth \
  -i /path/to/images \
  -o /tmp/deim-app-output \
  --score-threshold 0.25 \
  --format json dota visualization
```

`json` and `visualization` work for both HBB and OBB; `dota` is OBB-only.

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

`from_config` does not load weights; call `load` next. `predict_filtered`
applies `score_threshold`, `class_filter`, and `top_k` from the config by
default.

### Export

The `deim_app export` subcommand is not enabled and always raises
`ExportError`. For ONNX and TensorRT export, use the deployment tools from the
upstream [DEIMv2 repository](https://github.com/Intellindust-AI-Lab/DEIMv2)
(`tools/deployment/export_onnx.py` and `trtexec`).

## Application Layer (deim_app)

The `deim_app` workflow is covered in
[Quick Start](#quick-start). For the full
reference, see the [Application Config Guide](docs/engineering/application-config.md)
and the [Inference API & CLI Guide](docs/engineering/inference-api.md).

## Project Tools

Tooling reorganized for this fork:

- `tools/train`, `tools/inference` — executable entry points
- `tools/compare` — research pipeline for model comparison
- `tools/analysis` — analysis utilities (e.g., OBB error classification,
  rep2 NaN diagnostics)
- pytest suite grouped by domain: `contracts / obb / eval / engine / deim_app`

Deployment (ONNX / TensorRT), benchmarking, and visualization tools are
inherited from upstream — see the
[DEIMv2 repository](https://github.com/Intellindust-AI-Lab/DEIMv2).

## Upstream DEIMv2

This project is built on [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2).
For the official model zoo and weights, engine-level training / testing /
tuning recipes, data preparation guides, and upstream change history, please
refer to the original repository:
**[github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)**.

## Citation

If you use `DEIMv2` or its methods in your work, please cite the following
BibTeX entries:

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

## Acknowledgement

Our work is built upon [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2),
[D-FINE](https://github.com/Peterande/D-FINE),
[RT-DETR](https://github.com/lyuwenyu/RT-DETR),
[DEIM](https://github.com/ShihuaHuang95/DEIM), and
[DINOv3](https://github.com/facebookresearch/dinov3). Thanks for their great work!

✨ Feel free to contribute and reach out if you have any questions! ✨
