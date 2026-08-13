# Application Config Guide (`deim_app`)

This document is the canonical reference for the **first-version** application
config contract exposed by `deim_app`. It covers the public YAML schema,
inheritance & override priority, the three bundled examples (HBB COCO, OBB
DOTA, OBB YOLO-OBB), class-metadata sources, parameters intentionally owned
by presets, and `pretrained` vs `resume` semantics.

For the inference Python API and CLI see [inference-api.md](inference-api.md).

---

## 1. Public field table

The application YAML accepts **only** the six top-level sections below. Every
other key (algorithm sections such as `DEIMTransformer`, `optimizer`,
`angle_rep`, `num_classes`, `epoches`, …) is rejected by the loader before any
engine object is constructed. The whitelist is enforced by
`deim_app.config.loader.validate_public_keys`.

### `project`

| Key         | Type   | Default        | Notes                                             |
|-------------|--------|----------------|---------------------------------------------------|
| `name`      | string | `"deim-app"`   | Free-form project label (informational).          |
| `output_dir`| string | `"outputs"`    | Where checkpoints / logs are written. Maps to the engine's top-level `output_dir`. |

### `runtime`

| Key          | Type            | Default       | Notes                                                                         |
|--------------|-----------------|---------------|-------------------------------------------------------------------------------|
| `input_size` | `[int, int]`    | `[640, 640]`  | Inference/training resolution `[H, W]`. Propagated to every `Resize`/`OBBResize` op, `collate_fn.base_size`, and `eval_spatial_size`. |
| `seed`       | integer         | `42`          | Engine `seed`.                                                                |

### `data`

| Key                  | Type         | Default | Notes                                                                                          |
|----------------------|--------------|---------|------------------------------------------------------------------------------------------------|
| `format`             | enum         | `COCO`  | One of `COCO`, `DOTA`, `YOLO-OBB`. Selects the dataset class and metadata loader.              |
| `train_images`       | string       | `""`    | Directory of training images (COCO/HBB) or image folder (OBB).                                |
| `train_annotations`  | string       | `""`    | COCO JSON file (COCO) **or** labels folder (DOTA/YOLO-OBB).                                   |
| `val_images`         | string       | `""`    | Validation image directory.                                                                   |
| `val_annotations`    | string       | `""`    | Validation annotation path (file or folder, see `format`).                                    |
| `classes_file`       | string/null  | `null`  | **Required for OBB**. One class name per line. Ignored for COCO (categories come from the JSON). |
| `num_workers`        | non-neg int  | `4`     | DataLoader workers.                                                                            |
| `cache_images`       | enum         | `none`  | `none` / `disk` / `ram`. **COCO requires `none`**; OBB accepts all three.                     |

### `train`

| Key             | Type            | Default   | Notes                                                                                       |
|-----------------|-----------------|-----------|---------------------------------------------------------------------------------------------|
| `epochs`        | positive int    | `100`     | Total training epochs. Maps to engine `epoches`.                                            |
| `batch_size`    | positive int    | `8`       | Maps to `train_dataloader.total_batch_size`.                                                |
| `learning_rate` | float           | `1.0e-4`  | Maps to **`optimizer.lr` only**. Per-param-group LRs are preset-owned and never touched.   |
| `device`        | string          | `"cuda"`  | Device the solver places the model on.                                                      |
| `amp`           | bool            | `True`    | Mixed-precision toggle (maps to `use_amp`).                                                 |
| `pretrained`    | string/null     | `null`    | Backbone init weights path. Mutually exclusive with `resume`. See §6.                       |
| `resume`        | string/null     | `null`    | Full-training-state checkpoint to resume from. Mutually exclusive with `pretrained`. See §6.|
| `early_stopping.enabled`  | bool     | `False`   | Early-stopping toggle (overrides preset).                                                   |
| `early_stopping.patience` | pos int  | `10`      | Early-stopping patience (overrides preset). Other ES fields are preset-owned.               |

### `evaluation`

| Key          | Type         | Default  | Notes                                       |
|--------------|--------------|----------|---------------------------------------------|
| `batch_size` | positive int | `8`      | Maps to `val_dataloader.total_batch_size`.  |
| `device`     | string       | `"cuda"` | Device for the evaluation pass.             |

### `inference`

| Key               | Type              | Default                | Notes                                                                 |
|-------------------|-------------------|------------------------|-----------------------------------------------------------------------|
| `checkpoint`      | string/null       | `null`                 | Default checkpoint path (informational; pass via `load()` or `-r`).   |
| `device`          | string            | `"cuda"`               | Device for inference.                                                 |
| `batch_size`      | positive int      | `1`                    | Inference batch size.                                                 |
| `score_threshold` | float ∈ `[0, 1]`  | `0.25`                 | Score threshold applied by `predict_filtered`.                        |
| `top_k`           | positive int      | `300`                  | Top-k per-image cap applied by `predict_filtered`.                    |
| `class_filter`    | list[str]/null    | `null`                 | Optional class-name whitelist (validated against metadata).           |
| `output_formats`  | list[str]         | `["json","visualization"]` | Subset of `json`, `dota`, `visualization`.                        |

---

## 2. Inheritance and override priority

Resolution order, **highest precedence last**:

```
1. algorithm preset          (configs/app/presets/*.yml)
2. application base YAML     (configs/app/base/{hbb,obb}_app.yml)
3. user application YAML     (your file, including a base via __include__)
4. CLI overrides             (-r / --score-threshold / etc.)
```

Equivalently: **`CLI > user YAML > application base > algorithm preset`**.

Rules:

* The user YAML **must** declare exactly one `__include__:` pointing at one of
  the two approved application bases (`configs/app/base/hbb_app.yml` or
  `obb_app.yml`). Including a preset directly is rejected.
* Only the six public sections may appear in user, base, or CLI inputs.
* Deep-merge: nested keys (e.g. `train.early_stopping.patience`) replace only
  that leaf; siblings are preserved.
* The algorithm preset is loaded once via the engine's `__include__` chain and
  becomes the immutable algorithm contract; the resolver only mutates the
  fields listed in §1 (everything else is preserved verbatim).

---

## 3. Examples

### 3.1 HBB on MS COCO

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

### 3.2 OBB on DOTA

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

### 3.3 OBB on YOLO-OBB

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

## 4. Class metadata sources

`num_classes` is **always derived** from on-disk annotation metadata, never
hand-entered. The derivation depends on `data.format`:

| Format     | Source                                            | `num_classes`              | Class names                                |
|------------|---------------------------------------------------|----------------------------|--------------------------------------------|
| `COCO`     | `data.train_annotations` (a `instances_*.json`)   | `len(categories)` (after remap decision) | `categories[*].name` (sorted by id)        |
| `DOTA`     | `data.classes_file`                               | line count                 | One name per non-empty line (line order → label) |
| `YOLO-OBB` | `data.classes_file`                               | line count                 | One name per non-empty line (line order → label) |

The derived value flows into the top-level engine `num_classes` and is propagated
to the model's class-prediction heads.

### 4.1 `remap_mscoco_category` auto-detection

For COCO the resolver auto-detects whether to apply the standard MS COCO
`1..90 → 0..79` remap:

* If the JSON's category **names set** equals the standard MS COCO 80-class set
  **and** its **id set** equals `{1, 2, …, 90}` (with gaps), the remap is applied
  automatically — `remap_mscoco_category` is set to `True` on the engine side.
* Otherwise the resolver requires **contiguous zero-based ids** (`0, 1, …, N-1`)
  and refuses non-contiguous ids with an actionable error.

This eliminates a class of silent label-corruption bugs that occurred when
custom datasets whose ids overlapped the MS COCO range were accidentally
remapped. To override the auto-detection, set `remap_mscoco_category` explicitly
in the algorithm preset (rarely needed).

---

## 5. Parameters intentionally owned by presets

These categories are **never** exposed in the public schema — they belong to the
algorithm preset and survive unchanged across user/base/CLI inputs:

| Category                | Representative keys                                                                                          |
|-------------------------|--------------------------------------------------------------------------------------------------------------|
| Backbone / encoder / decoder structure | `DEIM.backbone`, `DINOv3STAs.*`, `DINOv3STAsResAtten.*`, `HybridEncoder.{in_channels,hidden_dim,dim_feedforward}`, `DEIMTransformer.{feat_channels,hidden_dim,num_layers,dim_feedforward,eval_idx}` |
| OBB angle contract      | `DEIMTransformer.{angle_rep,offset_scale_source,use_gate_fusion,angle_step,use_angle_first,decoder_angle_encoding}`, `PostProcessor.box_mode`, `DEIMCriterion.{box_mode,obbox_rep_dim,offset_scale_source}` |
| Optimizer structure     | `optimizer.type`, `optimizer.betas`, `optimizer.weight_decay`, every `optimizer.params[*]` group (regex + per-group `lr`/`weight_decay`) |
| Scheduler / training stages | `lrsheduler`, `lr_gamma`, `warmup_iter`, `flat_epoch`, `no_aug_epoch`, `epoches` (mirrors `train.epochs`), `clip_max_norm`, `use_ema`, `ema.*` |
| Augmentation            | Mosaic / Mixup / CopyBlend ops + probabilities + policy epochs, `collate_fn.{base_size_repeat,mixup_epochs,copyblend_epochs,…}` |
| Loss / matcher weights  | `DEIMCriterion.weight_dict.*`, `DEIMCriterion.{reg_max,gamma,alpha,angle_lambda,…}`, `DEIMCriterion.matcher.*` |
| Derived fields          | `num_classes` (derived from metadata — see §4), `eval_spatial_size` (mirrors `runtime.input_size`), `checkpoint_freq` |
| Early-stopping preset fields | `early_stopping.{metric,mode,min_epochs,min_delta,restore_best}` — only `enabled` and `patience` are user-overridable |

The main learning-rate override (`train.learning_rate`) is the **only** optimizer
field exposed in the public schema; it modifies `optimizer.lr` only and never
touches per-param-group LRs.

---

## 6. `pretrained` versus `resume`

Both fields accept a checkpoint path string, but they have distinct semantics
and are **mutually exclusive** — specifying both raises `AppConfigError` at
load time.

| Field        | Engine key   | Semantics                                                                                                                                                          |
|--------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `pretrained` | `tuning`     | **Backbone init only.** Loads just enough to initialise the backbone (typically ImageNet weights). Training starts from scratch elsewhere (no optimizer / EMA state restored). Use this when starting a fresh run with a pretrained backbone. |
| `resume`     | `resume`     | **Full training state.** Restores model + optimizer + EMA + scheduler + epoch counter, so training continues exactly where the checkpoint left off.                |

The mutual-exclusion guard prevents the common mistake of asking for both
backbone init and full-state resume in the same run, which would silently
produce undefined ordering.

---

## 7. Verification

The structural invariants in this document are enforced continuously by
`test/deim_app/test_legacy_parity.py` (always-run) and the broader
`test/deim_app/` suite. Run them with:

```bash
pytest test/deim_app/ -v
```
