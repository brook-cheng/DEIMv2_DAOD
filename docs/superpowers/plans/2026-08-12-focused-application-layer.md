# DEIMv2 Focused Application Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first usable `deim_app` layer that simplifies YAML configuration and provides a shared Python/CLI PyTorch inference path for HBB COCO, OBB DOTA, and OBB YOLO-OBB without coupling application code to unfinished algorithm internals.

**Architecture:** Add a new `deim_app/` package above the existing `engine/`. Application YAML files inherit clean application presets through the existing `__include__` mechanism; a DEIM adapter maps the approved public fields into the current full YAML structure and normalizes model outputs into explicit HBB/OBB prediction types. Existing training, evaluation, model, postprocessor, geometry, and writer logic is reused rather than rewritten.

**Tech Stack:** Python 3.11, frozen `dataclasses`, PyYAML through the existing `engine.core.yaml_utils`, PyTorch 2.5.1, torchvision, Pillow, pytest.

## Global Constraints

- Preserve dependency direction `deim_app -> engine`; `engine` must never import `deim_app`.
- Do not change backbone, encoder, decoder, criterion, matcher, OBB geometry, postprocessor, evaluator, or training-loop mathematics.
- Do not edit or remove existing full YAML files, `train.py`, or existing inference/deployment tools in the first version.
- The first version supports PyTorch inference only. ONNX, TensorRT, and OpenVINO remain later iterations.
- The first version does not add dynamic plugin discovery, a Web/API service, or training-controller/session refactors.
- Application YAML uses `.yml` and the existing recursive `__include__` merge semantics.
- Only the approved public parameter whitelist may appear in user application YAML or CLI overrides.
- Parameter priority is `CLI > user application YAML > application base YAML > algorithm preset`.
- `runtime.input_size` must map to every training, validation, model-cache, and inference size location as one atomic operation.
- `train.pretrained` and `train.resume` are mutually exclusive.
- `train.learning_rate` changes only `optimizer.lr`; explicit parameter-group learning rates remain owned by the algorithm preset.
- HBB COCO class metadata comes from the annotation JSON. OBB DOTA/YOLO-OBB class metadata comes from `classes_file`.
- OBB DOTA and YOLO-OBB share one application base YAML and differ only through `data.format`.
- The existing postprocessor receives original image sizes and is the only component that restores HBB/OBB boxes to original-image coordinates. Do not rescale its outputs again.
- Threshold, Top-K, and class filters operate on immutable structured predictions; the retained full prediction collection is never mutated.
- Keep each new Python file focused and below 250 pure lines where practical.
- The worktree contains unrelated changes. Stage or modify only files named by the active task.
- Do not create Git commits unless the user explicitly requests commits during execution.

---

## File Structure

```text
deim_app/
  __init__.py                       public DetectionModel and prediction exports
  __main__.py                       python -m deim_app entry
  api.py                            DetectionModel facade
  cli.py                            train/eval/infer/export argument parsing
  errors.py                         stable application exception types
  config/
    __init__.py
    schema.py                       frozen application config types
    loader.py                       raw whitelist validation + include resolution
    mapping.py                      application-to-engine YAML mapping
    metadata.py                     COCO/classes.txt metadata loading
  adapters/
    __init__.py
    base.py                         DetectionAdapter protocol
    deim.py                         DEIM implementation and thin solver wrappers
    checkpoint.py                   model/EMA checkpoint selection and loading
    geometry.py                     adapter-owned wrappers over existing OBB geometry/drawing
  predictions/
    __init__.py
    types.py                        HBB/OBB discriminated prediction types
    collection.py                   immutable filtering and batch collection
  inference/
    __init__.py
    inputs.py                       image and directory source enumeration
    preprocessing.py                canonical resize/tensor/normalize path
    torch_backend.py                batched PyTorch inference and normalization
  writers/
    __init__.py
    json_writer.py                  generic structured JSON output
    dota_writer.py                  OBB per-image DOTA output
    visualization.py                HBB/OBB image rendering
configs/app/
  base/
    hbb_app.yml                     public HBB defaults and preset include
    obb_app.yml                     public OBB defaults and preset include
  presets/
    deimv2_dinov3_sp_hbb.yml        dataset-neutral HBB algorithm preset
    deimv2_dinov3_sp_obb.yml        dataset-neutral OBB preset derived from sp_fz_common
  examples/
    hbb_coco.yml                    documented HBB COCO example
    obb_dota.yml                    documented native DOTA example
    obb_yolo.yml                    documented YOLO-OBB example
test/deim_app/
  conftest.py
  test_dependency_boundaries.py
  config/
    test_loader.py
    test_mapping.py
  predictions/
    test_collection.py
    test_writers.py
  inference/
    test_inputs.py
    test_torch_backend.py
  adapters/
    test_deim_adapter.py
    test_solver_wrappers.py
  test_api.py
  test_cli.py
  test_legacy_parity.py
```

## Approved Public YAML Contract

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

## Task Dependencies

```text
Task 1 -> Task 2 -> Task 3 -> Task 5 -> Task 8 --\
   \-> Task 4 ---------> Task 6 -> Task 7 -------> Task 9 -> Task 10
```

Tasks 2 and 4 can be implemented in parallel after Task 1. Task 6 requires both Tasks 4 and 5. Task 8 can be implemented in parallel with Task 6 after Task 5. Task 9 requires Tasks 7 and 8, and Task 10 is the final release gate.

---

### Task 1: Create the Package, Errors, and Dependency Guards

**Files:**
- Create: `deim_app/__init__.py`
- Create: `deim_app/errors.py`
- Create: `deim_app/adapters/__init__.py`
- Create: `test/deim_app/conftest.py`
- Create: `test/deim_app/test_dependency_boundaries.py`

**Interfaces:**
- Produces `AppConfigError`, `AdapterConfigurationError`, `CheckpointCompatibilityError`, `InputSourceError`, `InferenceBackendError`, and `ExportError`.
- Establishes the import rule used by every later task.

- [ ] **Step 1: Write the dependency-guard tests**

```python
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_engine_never_imports_deim_app() -> None:
    violations = []
    for path in (ROOT / "engine").rglob("*.py"):
        if any(name == "deim_app" or name.startswith("deim_app.") for name in imported_modules(path)):
            violations.append(path.relative_to(ROOT))
    assert violations == []


def test_only_adapters_import_engine_solver_or_model_internals() -> None:
    violations = []
    for path in (ROOT / "deim_app").rglob("*.py"):
        if "adapters" in path.parts:
            continue
        forbidden = {
            name
            for name in imported_modules(path)
            if name == "engine"
            or name.startswith("engine.")
            or name.startswith("tools.model_compare")
        }
        if forbidden:
            violations.append((path.relative_to(ROOT), sorted(forbidden)))
    assert violations == []
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `pytest test/deim_app/test_dependency_boundaries.py -v`

Expected: collection or import failure because `deim_app` does not exist.

- [ ] **Step 3: Add the package and stable errors**

```python
class DeimApplicationError(Exception):
    """Base class for user-facing application failures."""


class AppConfigError(DeimApplicationError):
    pass


class AdapterConfigurationError(DeimApplicationError):
    pass


class CheckpointCompatibilityError(DeimApplicationError):
    pass


class InputSourceError(DeimApplicationError):
    pass


class InferenceBackendError(DeimApplicationError):
    pass


class ExportError(DeimApplicationError):
    pass
```

`deim_app/__init__.py` initially exports only the exception types. `deim_app/adapters/__init__.py` is an empty package boundary. Later tasks add `DetectionModel`, adapter exports, and prediction types.

- [ ] **Step 4: Run the focused test and diagnostics**

Run: `pytest test/deim_app/test_dependency_boundaries.py -v`

Run: `python -m py_compile deim_app/__init__.py deim_app/errors.py`

Expected: PASS and exit code 0.

---

### Task 2: Implement the Typed Application YAML Loader

**Files:**
- Create: `deim_app/config/__init__.py`
- Create: `deim_app/config/schema.py`
- Create: `deim_app/config/loader.py`
- Create: `test/deim_app/config/test_loader.py`

**Interfaces:**
- Produces frozen types `ProjectConfig`, `RuntimeConfig`, `DataConfig`, `EarlyStoppingConfig`, `TrainConfig`, `EvaluationConfig`, `InferenceConfig`, and `AppConfig`.
- Produces `LoadedAppConfig(app: AppConfig, engine_base: dict[str, object], source: Path, app_base: Path)`.
- Produces `load_app_config(path: str | Path, cli_overrides: Mapping[str, object] | None = None) -> LoadedAppConfig`.

- [ ] **Step 1: Write tests for the whitelist and merge priority**

```python
def test_rejects_algorithm_key_in_user_yaml(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "bad.yml",
        {"__include__": ["base.yml"], "DEIMTransformer": {"angle_rep": 2}},
    )
    write_yaml(tmp_path / "base.yml", valid_base_dict())
    with pytest.raises(AppConfigError, match="DEIMTransformer"):
        load_app_config(path)


def test_rejects_direct_include_of_algorithm_yaml(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "bad.yml",
        {"__include__": ["../../configs/custom_obb/dlzdt/sp_fz_common.yml"]},
    )
    with pytest.raises(AppConfigError, match="application base"):
        load_app_config(path)


def test_cli_device_overrides_user_and_base_yaml(tmp_path: Path) -> None:
    base = write_yaml(tmp_path / "base.yml", valid_base_dict(device="cpu"))
    user = write_yaml(
        tmp_path / "user.yml",
        {"__include__": [base.name], "train": {"device": "cuda:0"}},
    )
    loaded = load_app_config(user, {"train": {"device": "cuda:1"}})
    assert loaded.app.train.device == "cuda:1"


def test_pretrained_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "bad.yml",
        {
            **valid_user_dict(),
            "train": {"pretrained": "init.pth", "resume": "last.pth"},
        },
    )
    with pytest.raises(AppConfigError, match="pretrained.*resume"):
        load_app_config(path)
```

Also cover:

- `data.format` accepts only `COCO`, `DOTA`, and `YOLO-OBB`.
- `data.cache_images` accepts only `none`, `disk`, and `ram`.
- `runtime.input_size` is exactly two positive integers.
- batch sizes and worker counts are non-negative integers, with train/evaluation/inference batch sizes greater than zero.
- score threshold is within `[0, 1]`; Top-K is positive.
- unknown keys at every public section raise `AppConfigError` with the full dotted path.
- CLI overrides are checked against the same whitelist instead of passing arbitrary dotted keys to the engine.
- a user YAML has exactly one direct `__include__`, resolving to `configs/app/base/hbb_app.yml` or `configs/app/base/obb_app.yml`; direct algorithm-preset includes, additional includes, and traversal to another file fail before recursive loading.
- HBB rejects `cache_images: disk` and `cache_images: ram` because the current `CocoDetection` has no image-cache contract; OBB accepts all three values.

- [ ] **Step 2: Run the loader tests and confirm RED**

Run: `pytest test/deim_app/config/test_loader.py -v`

Expected: FAIL because the config package is absent.

- [ ] **Step 3: Implement frozen schema types**

Use `@dataclass(frozen=True, slots=True)`. Define defaults in one place, for example:

```python
@dataclass(frozen=True, slots=True)
class InferenceConfig:
    checkpoint: Path | None = None
    device: str = "cuda"
    batch_size: int = 1
    score_threshold: float = 0.25
    top_k: int = 300
    class_filter: tuple[str, ...] | None = None
    output_formats: tuple[str, ...] = ("json", "visualization")
```

Do not store the full algorithm YAML in these types.

- [ ] **Step 4: Implement the two-stage loader**

The loader order must separate trusted engine content from the public application object:

```python
user_raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
validate_public_keys(user_raw)
app_base = validate_single_application_base(source, user_raw["__include__"])
engine_base = yaml_utils.load_config(str(source), cfg={})
public_merged = extract_public_sections(engine_base)
merge_dict(public_merged, validated_cli_overrides)
app = AppConfig.from_mapping(public_merged)
return LoadedAppConfig(
    app=app,
    engine_base=engine_base,
    source=source,
    app_base=app_base,
)
```

`validate_public_keys` applies to the user file and CLI overrides, not to trusted algorithm keys inherited through the approved application base. `AppConfig.from_mapping` receives only the six public sections, never the complete engine dictionary. Pass a fresh `cfg={}` to `load_config`; do not rely on its mutable default argument.

- [ ] **Step 5: Verify focused tests and diagnostics**

Run: `pytest test/deim_app/config/test_loader.py -v`

Run: `python -m py_compile deim_app/config/schema.py deim_app/config/loader.py`

Expected: PASS.

---

### Task 3: Add Dataset-Neutral Application Presets and Mapping

**Files:**
- Create: `deim_app/config/metadata.py`
- Create: `deim_app/config/mapping.py`
- Create: `configs/app/presets/deimv2_dinov3_sp_obb.yml`
- Create: `configs/app/presets/deimv2_dinov3_sp_hbb.yml`
- Create: `configs/app/base/obb_app.yml`
- Create: `configs/app/base/hbb_app.yml`
- Create: `configs/app/examples/obb_dota.yml`
- Create: `configs/app/examples/obb_yolo.yml`
- Create: `configs/app/examples/hbb_coco.yml`
- Create: `test/deim_app/config/test_mapping.py`

**Interfaces:**
- Produces `DatasetMetadata(box_mode, num_classes, class_names_by_label, output_names_by_id)`.
- Produces `ResolvedAlgorithmConfig(config_path, overrides, metadata, app)`.
- Produces `resolve_algorithm_config(loaded: LoadedAppConfig) -> ResolvedAlgorithmConfig`.
- Downstream construction is always `YAMLConfig(str(config_path), **overrides)`; do not add an alternate engine config class.

- [ ] **Step 1: Write metadata tests**

```python
def test_obb_metadata_comes_from_classes_file(tmp_path: Path) -> None:
    classes = tmp_path / "classes.txt"
    classes.write_text("cable\nclamp\n", encoding="utf-8")
    metadata = load_obb_metadata(classes)
    assert metadata.num_classes == 2
    assert metadata.class_names_by_label == {0: "cable", 1: "clamp"}


def test_hbb_custom_coco_requires_contiguous_zero_based_ids(tmp_path: Path) -> None:
    ann = write_coco(tmp_path / "instances.json", categories=[
        {"id": 1, "name": "one"}, {"id": 3, "name": "three"}
    ])
    with pytest.raises(AppConfigError, match="contiguous"):
        load_coco_metadata(ann, remap_mscoco_category=False)
```

For `remap_mscoco_category=True`, allow the standard MS COCO IDs and retain both mappings:

- contiguous model label -> class name;
- emitted category ID -> class name after postprocessor remapping.

- [ ] **Step 2: Write mapping tests for every exposed field**

Assert the exact target paths:

| Public field | Engine YAML target |
|---|---|
| `project.output_dir` | `output_dir` |
| `runtime.seed` | `seed` |
| `runtime.input_size` | `eval_spatial_size`, every train/val resize op `size`, train `collate_fn.base_size` |
| `train.epochs` | `epoches` |
| `train.batch_size` | `train_dataloader.total_batch_size` |
| `evaluation.batch_size` | `val_dataloader.total_batch_size` |
| `train.learning_rate` | `optimizer.lr` only |
| `train.device` | `device` for train wrapper |
| `train.amp` | `use_amp` |
| `train.pretrained` | `tuning` |
| `train.resume` | `resume` |
| early-stopping public fields | matching fields under `early_stopping` while preserving all preset-owned fields |
| data image paths | train/val `dataset.img_folder` |
| OBB annotation paths | train/val `dataset.ann_folder` |
| HBB annotation paths | train/val `dataset.ann_file` |
| OBB `classes_file` | both dataset `classes_file` |
| `data.format` | OBB datasets `format` |
| `data.num_workers` | both dataloaders `num_workers` |
| `data.cache_images` | HBB accepts only `none`; OBB `none -> cache_images: none, cache_ram: 0`, `disk -> cache_images: disk, cache_ram: 0`, `ram -> cache_images: none, cache_ram: number of validated images in that split` |
| derived class count | top-level `num_classes` and shared model/postprocessor consumers through YAML registry |

The resolved application object, rather than engine overrides, owns these runtime-only fields:

- `project.name` identifies the run in the read-only resolved summary.
- `evaluation.device` is applied by the evaluation wrapper when constructing the engine config.
- `inference.checkpoint`, `inference.device`, and `inference.batch_size` are consumed by the adapter/backend.
- `inference.score_threshold`, `inference.top_k`, `inference.class_filter`, and `inference.output_formats` are consumed only by the facade, CLI, and immutable result/writer layer.

Include this invariant:

```python
before_group_lr = loaded.engine_base["optimizer"]["params"][0]["lr"]
resolved = resolve_algorithm_config(loaded)
assert resolved.overrides["optimizer"]["lr"] == 1e-3
assert resolved.overrides["optimizer"]["params"][0]["lr"] == before_group_lr
```

Build `resolved.overrides` from a deep copy of `loaded.engine_base`; mapping must never mutate `LoadedAppConfig` or the process-global YAML registry.

- [ ] **Step 3: Run mapping tests and confirm RED**

Run: `pytest test/deim_app/config/test_mapping.py -v`

Expected: FAIL because metadata and mapping modules do not exist.

- [ ] **Step 4: Create clean algorithm presets**

`deimv2_dinov3_sp_obb.yml` must be derived from `configs/custom_obb/dlzdt/sp_fz_common.yml`, but remove dataset paths, project names, output paths, and ablation-only comments. Preserve the validated baseline model and optimization defaults, including:

- DINOv3 ViT-S/16 + STA adapter configuration;
- HybridEncoder and DEIMTransformer dimensions;
- `angle_rep: 0`, `offset_scale_source: pre`, gate fusion disabled, `angle_step: 0`, angle-first disabled, proportional encoding;
- current optimizer parameter groups, loss/matcher weights, EMA, augmentation policy, and scheduler defaults.

`deimv2_dinov3_sp_hbb.yml` must be a dataset-neutral extraction of `configs/deimv2/deimv2_dinov3_l_coco.yml`: retain its DINOv3 ViT-S/16 `DINOv3STAs`, HybridEncoder, DEIMTransformer, optimizer, scheduler, augmentation, and matcher defaults; include only `configs/runtime.yml`, `configs/base/dataloader.yml`, `configs/base/optimizer.yml`, and `configs/base/deimv2.yml`. Do not include `configs/dataset/coco_detection.yml`, `dataset_coco128_detection_train.yml`, or concrete data/output paths; Task 3 mapping supplies the COCO dataset objects and paths.

- [ ] **Step 5: Create application base YAMLs**

Each base file includes exactly one algorithm preset and provides public defaults. Example:

```yaml
__include__:
  - ../presets/deimv2_dinov3_sp_obb.yml

project:
  name: deimv2-obb
  output_dir: ./outputs/deimv2-obb

runtime:
  input_size: [640, 640]
  seed: 0

data:
  format: DOTA
  train_images: null
  train_annotations: null
  val_images: null
  val_annotations: null
  classes_file: null
  num_workers: 2
  cache_images: none
```

Do not put concrete dataset paths in base or preset files.

- [ ] **Step 6: Implement mapping and run end-to-end config tests**

Run: `pytest test/deim_app/config/test_loader.py test/deim_app/config/test_mapping.py -v`

Also run:

```bash
python -c "from deim_app.config import load_app_config, resolve_algorithm_config; print(resolve_algorithm_config(load_app_config('configs/app/examples/obb_dota.yml')).metadata)"
```

Expected: all tests pass; the command either prints valid metadata for configured fixture paths or fails with a precise missing-path `AppConfigError` before model construction.

---

### Task 4: Add Immutable HBB/OBB Predictions and Writers

**Files:**
- Create: `deim_app/predictions/__init__.py`
- Create: `deim_app/predictions/types.py`
- Create: `deim_app/predictions/collection.py`
- Create: `deim_app/adapters/geometry.py`
- Create: `deim_app/writers/__init__.py`
- Create: `deim_app/writers/json_writer.py`
- Create: `deim_app/writers/dota_writer.py`
- Create: `deim_app/writers/visualization.py`
- Create: `test/deim_app/predictions/test_collection.py`
- Create: `test/deim_app/predictions/test_writers.py`

**Interfaces:**
- `HBBDetection(class_id, class_name, score, xyxy)` with `box_mode == "hbb"`.
- `OBBDetection(class_id, class_name, score, xywhr)` with `box_mode == "obb"`.
- `ImagePrediction(image_id, source, original_image, original_size, detections, timings)` retains an immutable RGB copy for visualization; JSON and DOTA writers omit pixel data.
- `PredictionCollection(box_mode, predictions)` with immutable `filter()` and `top_k()` methods.
- Writers consume a collection; they never mutate it.
- `deim_app.adapters.geometry.obb_to_polygon(xywhr)` and `draw_obb_detections(...)` are the only application-layer wrappers allowed to import existing engine geometry or `tools/model_compare/obb_utils.py`; writers call these wrappers and never import those concrete modules directly.

- [ ] **Step 1: Write prediction filtering tests**

```python
def test_filter_returns_new_collection_without_mutating_full_predictions() -> None:
    full = make_obb_collection(scores=(0.9, 0.2))
    filtered = full.filter(score_threshold=0.5)
    assert len(full.predictions[0].detections) == 2
    assert len(filtered.predictions[0].detections) == 1


def test_top_k_is_per_image_and_score_ordered() -> None:
    full = make_hbb_collection(scores=(0.3, 0.9, 0.7))
    limited = full.top_k(2)
    assert [d.score for d in limited.predictions[0].detections] == [0.9, 0.7]
```

Also cover class-name and class-ID filtering and empty predictions.

- [ ] **Step 2: Write writer tests**

- JSON contains explicit `box_mode` and either `xyxy` or `xywhr`.
- OBB DOTA polygon coordinates equal `engine.deim.obb_geometry.xywhr_to_xyxyxyxy` through `deim_app.adapters.geometry.obb_to_polygon`; do not reproduce trigonometry.
- HBB DOTA export raises `ExportError`.
- OBB visualization delegates through `deim_app.adapters.geometry.draw_obb_detections`, whose implementation calls `draw_obb_polygons` from `tools/model_compare/obb_utils.py`.
- Applying a visualization threshold does not mutate the collection subsequently exported to JSON or DOTA.

- [ ] **Step 3: Run tests and confirm RED**

Run: `pytest test/deim_app/predictions -v`

- [ ] **Step 4: Implement prediction types and writers**

Use tuples and frozen dataclasses for public objects. Import writer functions inside `PredictionCollection.export_*` methods to avoid unnecessary import cycles.

- [ ] **Step 5: Verify tests and diagnostics**

Run: `pytest test/deim_app/predictions -v`

Run: `python -m py_compile deim_app/predictions/*.py deim_app/writers/*.py`

Expected: PASS.

---

### Task 5: Implement the DEIM Adapter and Checkpoint Loading

**Files:**
- Create: `deim_app/adapters/__init__.py`
- Create: `deim_app/adapters/base.py`
- Create: `deim_app/adapters/checkpoint.py`
- Create: `deim_app/adapters/deim.py`
- Create: `test/deim_app/adapters/test_deim_adapter.py`

**Interfaces:**
- `DetectionAdapter` protocol defines `resolve_config`, `load`, `predict`, `train`, `evaluate`, and `export`.
- `select_model_state(checkpoint, prefer_ema=True) -> Mapping[str, Tensor]` prefers `ema.module`, then `model`, and strips a leading `module.` prefix.
- `DeimDetectionAdapter.from_config(path, cli_overrides=None) -> DeimDetectionAdapter`.
- `load(checkpoint: str | Path | None = None, prefer_ema: bool = True) -> None` builds current engine objects and stores deployed model/postprocessor.

- [ ] **Step 1: Write checkpoint selection tests**

```python
def test_select_model_state_prefers_ema_module() -> None:
    checkpoint = {
        "model": {"weight": torch.tensor([1.0])},
        "ema": {"module": {"weight": torch.tensor([2.0])}},
    }
    state = select_model_state(checkpoint, prefer_ema=True)
    assert state["weight"].item() == 2.0


def test_select_model_state_falls_back_to_model() -> None:
    checkpoint = {"model": {"module.weight": torch.tensor([1.0])}}
    state = select_model_state(checkpoint, prefer_ema=True)
    assert set(state) == {"weight"}
```

- [ ] **Step 2: Write adapter construction tests with engine seams monkeypatched**

Assert that the adapter:

- calls `YAMLConfig(str(source_path), **resolved.overrides)`;
- disables backbone pretrained download when loading a checkpoint;
- loads state before calling `.deploy()`;
- stores explicit box mode and metadata;
- raises `CheckpointCompatibilityError` with missed/unmatched head keys when class counts are incompatible.

- [ ] **Step 3: Run tests and confirm RED**

Run: `pytest test/deim_app/adapters/test_deim_adapter.py -v`

- [ ] **Step 4: Implement the adapter boundary**

Only `deim_app/adapters/` may import `engine.core.YAMLConfig`, `engine.solver.TASKS`, concrete engine modules, or `tools.model_compare` helpers. Keep checkpoint normalization in `checkpoint.py` and geometry reuse in `geometry.py` so `deim.py` remains orchestration-focused.

- [ ] **Step 5: Verify tests, dependency guard, and diagnostics**

Run: `pytest test/deim_app/adapters/test_deim_adapter.py test/deim_app/test_dependency_boundaries.py -v`

Run: `python -m py_compile deim_app/adapters/*.py`

Expected: PASS.

---

### Task 6: Build the Shared PyTorch Inference Pipeline

**Files:**
- Create: `deim_app/inference/__init__.py`
- Create: `deim_app/inference/inputs.py`
- Create: `deim_app/inference/preprocessing.py`
- Create: `deim_app/inference/torch_backend.py`
- Create: `test/deim_app/inference/test_inputs.py`
- Create: `test/deim_app/inference/test_torch_backend.py`
- Modify: `deim_app/adapters/deim.py`

**Interfaces:**
- `InputSource = str | Path | PIL.Image.Image` and `list_inputs(source: InputSource) -> tuple[InputImage, ...]`.
- `InputImage(image_id, source, image)` stores a stable source label plus the loaded RGB image; paths are loaded lazily during enumeration and in-memory images receive deterministic IDs such as `memory-000001`.
- `PreparedImage(image_id, source, original_image, original_size_hw, tensor)` retains the original image needed by visualization without reopening or re-resizing it.
- `Preprocessor(input_size, normalize) -> PreparedImage`.
- `TorchBackend.predict(inputs, batch_size) -> PredictionCollection` returns the full structured collection.
- `DeimDetectionAdapter.predict(source, checkpoint=None, device=None, batch_size=None) -> PredictionCollection` delegates to the backend.

- [ ] **Step 1: Write input enumeration tests**

- A supported image file returns one input.
- A directory returns supported images sorted by filename and ignores unrelated files.
- An in-memory `PIL.Image.Image` returns one RGB input with a deterministic memory image ID.
- Missing paths and empty image directories raise `InputSourceError`.
- Directory enumeration is non-recursive in v1.

- [ ] **Step 2: Write backend tests using a stub model and postprocessor**

The stub postprocessor must return the current deploy tuple `(labels, boxes, scores)`. Test both modes:

```python
labels = torch.tensor([[0, 1]])
scores = torch.tensor([[0.9, 0.4]])
hbb_boxes = torch.tensor([[[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 8.0, 9.0]]])
obb_boxes = torch.tensor([[[5.0, 6.0, 7.0, 8.0, 0.5], [1.0, 2.0, 3.0, 4.0, 1.0]]])
```

Assert:

- HBB creates `HBBDetection`; OBB creates `OBBDetection`.
- Class names are resolved through metadata mappings, including remapped COCO category IDs.
- Original image sizes are passed to the postprocessor as `[width, height]`, matching `PostProcessor.forward`.
- Returned boxes equal postprocessor boxes exactly. No OBB rescale helper is called after postprocessing.
- `batch_size` splits inputs deterministically.
- The full collection retains all postprocessor results.
- `ImagePrediction.timings` contains non-negative preprocess, inference, and postprocess durations, and its retained source image can be visualized without reading a second input path.

- [ ] **Step 3: Run tests and confirm RED**

Run: `pytest test/deim_app/inference -v`

- [ ] **Step 4: Implement canonical preprocessing**

For the initial DINOv3 presets, use:

```python
transforms.Compose([
    transforms.Resize(input_size),
    ConvertPILImage(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

Record `(height, width)` in public metadata, but construct the postprocessor tensor as `[width, height]` because the current postprocessor scales x coordinates by column 0 and y coordinates by column 1.

- [ ] **Step 5: Implement backend normalization**

The backend performs:

```text
load PIL -> preprocess -> batch tensor -> model -> deployed postprocessor
           -> tuple(labels, boxes, scores) -> typed detections -> full collection
```

No result filtering or file writing occurs inside the backend.

- [ ] **Step 6: Verify tests and adapter integration**

Run: `pytest test/deim_app/inference test/deim_app/adapters/test_deim_adapter.py -v`

Expected: PASS.

---

### Task 7: Add the Python Facade and Result Filtering

**Files:**
- Create: `deim_app/api.py`
- Modify: `deim_app/__init__.py`
- Create: `test/deim_app/test_api.py`

**Interfaces:**
- `DetectionModel.from_config(config_path, **cli_overrides) -> DetectionModel`.
- `load(checkpoint=None, prefer_ema=True) -> DetectionModel` returns `self` for chaining.
- `predict(source, *, batch_size=None) -> PredictionCollection` returns full predictions.
- `predict_filtered(source, *, score_threshold=None, top_k=None, class_filter=None, batch_size=None) -> PredictionCollection` calls `predict()` and returns an immutable filtered view.

- [ ] **Step 1: Write facade delegation tests**

```python
def test_predict_returns_full_collection_and_filtered_view_is_separate(fake_adapter) -> None:
    model = DetectionModel(fake_adapter)
    full = model.predict("images")
    filtered = model.predict_filtered("images", score_threshold=0.5, top_k=1)
    assert len(full.predictions[0].detections) == 3
    assert len(filtered.predictions[0].detections) == 1
```

Also verify:

- predicting before loading raises `InferenceBackendError`;
- config defaults feed `predict_filtered` when arguments are omitted;
- class filters accept configured class names and fail early for unknown names.

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest test/deim_app/test_api.py -v`

- [ ] **Step 3: Implement the facade as pure delegation**

The facade may import `DeimDetectionAdapter`, but it must not import engine modules or perform tensor operations.

- [ ] **Step 4: Verify API tests and dependency guards**

Run: `pytest test/deim_app/test_api.py test/deim_app/test_dependency_boundaries.py -v`

Expected: PASS.

---

### Task 8: Add Thin Train, Evaluate, and Export Wrappers

**Files:**
- Modify: `deim_app/adapters/deim.py`
- Create: `test/deim_app/adapters/test_solver_wrappers.py`

**Interfaces:**
- `train() -> None` builds `TASKS[cfg.yaml_cfg["task"]](cfg)` and calls `solver.fit()`.
- `evaluate(checkpoint=None) -> None` builds the solver, applies `evaluation.device`, and calls `solver.val()` with `cfg.resume` set.
- `supported_export_formats() -> tuple[str, ...]` returns `()` in v1.
- `export(checkpoint, format, output) -> Path` always raises `ExportError("No export format is enabled in the first application-layer version")` before creating output.

- [ ] **Step 1: Write wrapper tests with a fake solver registry**

```python
def test_train_delegates_to_existing_solver_fit(monkeypatch, adapter) -> None:
    calls = []
    monkeypatch.setitem(TASKS, "detection", lambda cfg: FakeSolver(cfg, calls))
    adapter.train()
    assert calls == ["fit"]


def test_evaluate_delegates_to_val_with_checkpoint(monkeypatch, adapter, tmp_path) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"fixture")
    adapter.evaluate(checkpoint)
    assert adapter.last_built_cfg.resume == str(checkpoint)
```

- [ ] **Step 2: Lock first-version export behavior**

The existing `tools/deployment/export_onnx.py` is a script-oriented entry point that owns config/model construction and optional ONNX checks. Wrapping it would duplicate the new adapter path, while refactoring it is outside v1. Therefore assert exactly:

```python
assert adapter.supported_export_formats() == ()
with pytest.raises(ExportError, match="No export format is enabled"):
    adapter.export(checkpoint, "onnx", output)
assert not output.exists()
```

Do not implement or expose ONNX, TensorRT, OpenVINO, or any other model export format in this task.

- [ ] **Step 3: Run tests and confirm RED**

Run: `pytest test/deim_app/adapters/test_solver_wrappers.py -v`

- [ ] **Step 4: Implement thin wrappers without touching solver code**

Do not call repository scripts through subprocess. Call Python functions or current solver methods directly.

- [ ] **Step 5: Verify wrapper tests**

Run: `pytest test/deim_app/adapters/test_solver_wrappers.py -v`

Expected: PASS.

---

### Task 9: Add the Shared CLI and Output Selection

**Files:**
- Create: `deim_app/cli.py`
- Create: `deim_app/__main__.py`
- Create: `test/deim_app/test_cli.py`

**Interfaces:**
- Subcommands: `train`, `eval`, `infer`, `export`.
- `main(argv: Sequence[str] | None = None) -> int` returns an exit code and is invoked by `__main__.py`.
- CLI calls `DetectionModel`/adapter methods; it contains no model construction, preprocessing, postprocessing, or geometry.

- [ ] **Step 1: Write parser and whitelist tests**

Approved flags:

```text
all:    -c/--config
train:  --device --resume --output-dir
eval:   -r/--checkpoint --device
infer:  -r/--checkpoint -i/--input -o/--output --device
        --batch-size --score-threshold --top-k --class-filter --format
export: -r/--checkpoint -o/--output --format --device
```

Assert that a flag such as `--angle-rep` or arbitrary `-u DEIMTransformer.angle_rep=2` is rejected by argparse.

- [ ] **Step 2: Write CLI/API equivalence tests**

Monkeypatch `DetectionModel.from_config` with a fake facade, then assert:

- `infer` requests one full prediction collection;
- applies score/Top-K/class filtering through collection methods;
- writes only requested formats, using `inference.output_formats` when `--format` is omitted;
- `format=dota` on HBB exits non-zero through `ExportError`;
- user-facing errors print the concise message and return non-zero without swallowing the original exception in Python API use.

- [ ] **Step 3: Run CLI tests and confirm RED**

Run: `pytest test/deim_app/test_cli.py -v`

- [ ] **Step 4: Implement the CLI**

Use `argparse`. Keep `__main__.py` minimal:

```python
from deim_app.cli import main

raise SystemExit(main())
```

- [ ] **Step 5: Verify CLI help and tests**

Run: `pytest test/deim_app/test_cli.py -v`

Run: `python -m deim_app --help`

Run: `python -m deim_app infer --help`

Expected: tests pass and help lists only approved fields.

---

### Task 10: Add Real HBB/OBB Parity Gates and User Documentation

**Files:**
- Create: `test/deim_app/test_legacy_parity.py`
- Create: `docs/engineering/application-config.md`
- Create: `docs/engineering/inference-api.md`
- Modify: `README.md`
- Modify: `docs/superpowers/INDEX.md`

**Interfaces:**
- No new runtime interface. This task is the release gate for the first version.

- [ ] **Step 1: Add an always-running structural parity test**

Assert for each example application YAML:

- resolved box mode, classes, data paths, total batch sizes, input sizes, AMP, learning rate, and early-stopping fields equal the expected full YAML values;
- algorithm-only fields from the preset survive unchanged;
- no application file contains forbidden algorithm keys.

- [ ] **Step 2: Add HBB and OBB numerical parity tests**

For available local fixture checkpoints and one fixed image per mode:

1. Run the new adapter backend.
2. Independently construct the current reference path from `tools/inference/torch_inf.py` for HBB or `test/tool_deimv2_obb_infer.py` for OBB.
3. Pass original image sizes to both postprocessors.
4. Assert labels equal and boxes/scores satisfy:

```python
torch.testing.assert_close(new_scores, legacy_scores, rtol=1e-5, atol=1e-6)
torch.testing.assert_close(new_boxes, legacy_boxes, rtol=1e-5, atol=1e-4)
```

If a fixture checkpoint is absent, skip only the numerical test with the exact missing path. Structural, mapping, prediction, CLI, and dependency tests must always run.

- [ ] **Step 3: Add an end-to-end manual inference smoke command**

Run when a checkpoint and image are available:

```bash
python -m deim_app infer \
  -c configs/app/examples/obb_yolo.yml \
  -r /path/to/model.pth \
  -i /path/to/images \
  -o /tmp/deim-app-output \
  --score-threshold 0.25 \
  --format json dota visualization
```

Verify JSON, per-image DOTA txt, and visualized images are produced from one inference run.

- [ ] **Step 4: Document the first-version contract**

`application-config.md` must include:

- the approved public field table;
- inheritance and override priority;
- HBB COCO, OBB DOTA, and OBB YOLO-OBB examples;
- class metadata sources;
- parameters intentionally owned by presets;
- pretrained versus resume semantics.

`inference-api.md` must include:

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

- [ ] **Step 5: Run the complete first-version verification matrix**

Run: `pytest test/deim_app -v`

Run the focused legacy suites affected by reused utilities:

```bash
pytest test/test_obb_utils.py test/test_deim_postprocessor.py test/test_dota_dataset_cache.py -v
```

Run: `python -m py_compile $(printf '%s ' deim_app/**/*.py deim_app/*.py)`

Run: `git diff --check`

Expected: all available tests pass; numerical parity tests either pass or skip with named missing fixture paths; no unrelated file is modified.

---

## Execution Order and Gates

```text
G0: Task 1
    Package imports and dependency rules are enforced.

G1: Tasks 2-3
    Application YAML accepts only approved fields and resolves all three dataset formats into current engine YAML.

G2: Task 4
    HBB/OBB result types and outputs are stable before inference code depends on them.

G3: Tasks 5-6 and Task 8
    A checkpoint can be loaded once and full PyTorch predictions are returned without duplicate geometry logic.
    Train/eval wrappers delegate to the current solver, and model export is explicitly unsupported in v1.

G4: Tasks 7 and 9
    Python API and CLI share the same adapter/backend; train/eval remain thin wrappers.

G5: Task 10
    New and current paths are structurally and numerically equivalent for available HBB/OBB fixtures.
```

Do not begin a gate until the preceding gate passes. Tasks 2 and 4 may run in parallel after G0. Task 8 can be prepared in parallel with Task 6 after Task 5 establishes adapter construction.

## Completion Checklist

- [ ] User YAML files use `.yml` and `__include__`.
- [ ] HBB COCO, OBB DOTA, and OBB YOLO-OBB resolve successfully.
- [ ] Unknown application fields and arbitrary algorithm overrides fail before engine construction.
- [ ] `runtime.input_size` updates every required location consistently.
- [ ] Class names/counts are derived, never hand-entered as `num_classes`.
- [ ] `train.pretrained` and `train.resume` are distinct and mutually exclusive.
- [ ] Main learning-rate override leaves parameter-group learning rates unchanged.
- [ ] PyTorch backend returns explicit HBB/OBB structured results.
- [ ] Original-size coordinate restoration happens once in the existing postprocessor.
- [ ] Full predictions remain available while filtering returns immutable views.
- [ ] JSON, OBB DOTA, and visualization writers work from one prediction collection.
- [ ] Python API and CLI call the same inference implementation.
- [ ] Train and eval call current solver `fit()` and `val()` without training-loop edits.
- [ ] Unsupported export formats fail before producing files.
- [ ] `engine` has no `deim_app` import.
- [ ] Existing full YAMLs, `train.py`, and existing tools remain unchanged and usable.
- [ ] First-version tests and relevant legacy tests pass.
