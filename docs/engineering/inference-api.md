# Inference API & CLI Guide (`deim_app`)

This document is the canonical reference for performing inference through the
`deim_app` application layer in the **first version**. It covers the Python
API, the unified CLI, output formats, and a manual smoke command.

For the application YAML schema see [application-config.md](application-config.md).

---

## 1. Python API quick-start

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

Key points:

* `from_config` **does not** load the checkpoint — call `load` next. It returns
  `self`, so the chain `DetectionModel.from_config(...).load(...)` is idiomatic.
* `predict` returns the **full unfiltered** collection; `predict_filtered`
  applies `score_threshold` → `class_filter` → `top_k` (defaults from the
  loaded config's `inference` section) and returns an immutable view.
* The full collection is never mutated by filtering — both views coexist.
* Class-filter names are validated against dataset metadata before any
  inference runs; unknown names raise `AppConfigError`.

### 1.1 Filtered inference with config defaults

```python
model = DetectionModel.from_config("configs/app/examples/obb_yolo.yml").load("model.pth")

# Uses inference.score_threshold / top_k / class_filter from the YAML by default.
results = model.predict_filtered("images/")
results.export_dota("outputs/dota")
```

### 1.2 Explicit overrides at call time

```python
results = model.predict_filtered(
    "images/",
    score_threshold=0.5,
    top_k=100,
    class_filter=("car", "truck"),
)
```

### 1.3 Reading typed predictions

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

## 2. CLI reference

The unified entry point is `python -m deim_app`. Four subcommands are
supported: `train`, `eval`, `infer`, `export`. The flag whitelist is exhaustive
— unknown flags are rejected by argparse at exit code 2.

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

Flags:

| Flag                    | Purpose                                                              |
|-------------------------|----------------------------------------------------------------------|
| `-c` / `--config`       | Application YAML path (required).                                    |
| `-r` / `--checkpoint`   | Model checkpoint path (required).                                    |
| `-i` / `--input`        | Image file or directory (required).                                  |
| `-o` / `--output`       | Output directory (required).                                         |
| `--device`              | Inference device (default from YAML `inference.device`).             |
| `--batch-size`          | Inference batch size (default from YAML `inference.batch_size`).     |
| `--score-threshold`     | Score threshold (default from YAML `inference.score_threshold`).     |
| `--top-k`               | Top-k cap (default from YAML `inference.top_k`).                     |
| `--class-filter`        | Optional class-name whitelist (repeatable).                          |
| `--format`              | Output formats: subset of `json`, `dota`, `visualization` (repeatable). |

### 2.2 `train`

```bash
python -m deim_app train \
  -c configs/app/examples/hbb_coco.yml \
  --device cuda \
  [--resume /path/to/checkpoint.pth] \
  [--output-dir ./outputs/my-run]
```

Delegates to the engine solver's `fit()` — the training loop itself is
unchanged from the legacy `train.py` entry point. `--resume` is mutually
exclusive with `train.pretrained` from the YAML.

### 2.3 `eval`

```bash
python -m deim_app eval \
  -c configs/app/examples/hbb_coco.yml \
  -r /path/to/model.pth \
  [--device cuda]
```

Delegates to the engine solver's `val()`. The checkpoint is loaded via the
solver's resume path; evaluation runs one full pass over the validation set.

### 2.4 `export`

```bash
python -m deim_app export \
  -c configs/app/examples/hbb_coco.yml \
  -r /path/to/model.pth \
  -o /tmp/export-output \
  --format onnx \
  [--device cpu]
```

> **Note:** no export format is enabled in the first application-layer version;
> `export` always raises `ExportError`. The subcommand exists so the CLI
> surface is stable — a future version will wire ONNX / OpenVINO / TensorRT
> export through this entry point.

---

## 3. Output formats

All three writers consume one immutable `PredictionCollection`; they can be
combined freely in a single invocation.

| Format         | Writer                         | Output                                                                                       |
|----------------|--------------------------------|----------------------------------------------------------------------------------------------|
| `json`         | `write_json`                   | One `predictions.json` file with one entry per image (labels, boxes, scores, class_names).   |
| `dota`         | `write_dota`                   | One `<image_stem>.txt` per image with one OBB detection per line: `x1 y1 x2 y2 x3 y3 x4 y4 class score`. OBB only. |
| `visualization`| `write_visualization`          | One `<image_stem>.{png,jpg}` per image with boxes/polygons overlaid on the source image.    |

`json` and `visualization` work for both HBB and OBB; `dota` is OBB-only.

---

## 4. Manual end-to-end smoke test

This is the recommended one-command verification that an OBB pipeline is wired
correctly end-to-end. It runs one inference call and asks for all three output
formats from a single invocation:

```bash
python -m deim_app infer \
  -c configs/app/examples/obb_yolo.yml \
  -r /path/to/model.pth \
  -i /path/to/images \
  -o /tmp/deim-app-output \
  --score-threshold 0.25 \
  --format json dota visualization
```

Expected outputs under `/tmp/deim-app-output/`:

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

Verify:

1. `predictions.json` parses, has one entry per input image, and every
   detection has a `class_name` from `classes.txt`.
2. Each `dota/*.txt` line is `x1 y1 x2 y2 x3 y3 x4 y4 class score` (whitespace-
   separated, 10 fields), with coordinates in original-image pixels.
3. Each `visualization/*.png` is the source image with rotated polygons
   overlaid, scored above `--score-threshold`.

This is a **manual** smoke step — it is not automated because it depends on a
locally-available checkpoint and dataset. The numerical parity between this
new path and the legacy tools (`tools/inference/torch_inf.py`,
`test/tool_deimv2_obb_infer.py`) is gated automatically by
`test/deim_app/test_legacy_parity.py` when the `DEIM_APP_PARITY_*` env vars are
populated.
