# DEIMv2 Engineering Platform Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-machine HBB/OBB engineering platform with unified train, resume, eval, export, and infer workflows while preserving every existing YAML file, legacy command, checkpoint, and validated model behavior.

**Architecture:** Add a top-level `deim_app/` application package above the existing `engine/`. New contracts and services call existing model code; model math never imports the application layer. Migration proceeds behind regression tests, and legacy entry points become thin adapters only after equivalent new paths pass.

**Tech Stack:** Python 3.11, PyTorch 2.5.1, torchvision 0.20.1, PyYAML, pytest, ONNX Runtime for the mandatory deployment consistency gate, existing TensorBoard/Comet integrations.

## Global Constraints

- Preserve existing YAML semantics, CLI flags, checkpoint keys, HBB behavior, OBB behavior, and single-machine DDP startup.
- Do not change formulas in backbone, decoder, criterion, matcher, OBB geometry, postprocessor, or evaluator during this refactor.
- Keep dependency direction `deim_app -> engine`; `engine` must never import `deim_app`.
- Existing `train.py` and `tools/{inference,deployment}/*.py` remain supported throughout the project.
- Iteration schedulers advance for every consumed dataloader batch, including AMP-overflow batches where optimizer step is skipped.
- EMA advances only after a successful optimizer step.
- `global_step` is an alias for consumed-batch `data_step`; `optimizer_step` increments only after parameter updates.
- Legacy checkpoints with no `optimizer_step` expose it as `None`, not a guessed integer.
- New formal tests live in `tests/`; the existing `test/` directory remains unchanged and must continue to pass.
- New checkpoints preserve existing keys and add metadata without requiring migration.
- No Web UI, database, Kubernetes, Slurm, task queue, remote artifact store, or future-task plugin system.

---

## File Structure

```text
deim_app/
  __init__.py                 package marker
  __main__.py                 unified CLI parser and exit-code mapping
  errors.py                   stable application errors
  config.py                   ResolvedRunConfig and startup validation
  run_layout.py               run directory and manifest creation
  checkpoint.py               checkpoint inspection and compatibility metadata
  prediction.py               explicit HBB/OBB prediction contract
  applications/
    train.py                  train/resume orchestration entry
    evaluate.py               evaluation entry
    export.py                 export entry and release gate
    infer.py                  inference entry
    inspect_checkpoint.py     checkpoint inspection entry
  pipeline/
    types.py                  prepared input and timing contracts
    input_adapter.py          image/directory/video input iteration
    preprocessor.py           resize, normalize, batching, metadata
    backend.py                RuntimeBackend protocol
    torch_backend.py          PyTorch runtime
    onnx_backend.py           ONNX Runtime runtime
    tensorrt_backend.py       TensorRT runtime
    openvino_backend.py       OpenVINO runtime
    output_decoder.py         postprocessor to PredictionBatch conversion
    result_writer.py          JSON/image/video output
    inference.py              pipeline composition
  training/
    types.py                  StepResult and training events
    precision.py              AMP/non-AMP execution decisions
    optimization.py           optimizer/EMA/scheduler state transitions
    diagnostics.py            off/standard/debug policy
    step.py                   one training-step executor
    checkpoint_manager.py     atomic save/load and aliases
    events.py                 JSONL/TensorBoard/Comet sinks
    session.py                epoch lifecycle and stage transitions
tests/
  unit/                       pure contract and policy tests
  component/                  synthetic train-step tests
  integration/                CLI, resume, and end-to-end tests
  backend/                    runtime consistency tests
  baselines/                  approved HBB/OBB regression artifacts
```

## Approved Decisions

1. `TrainApplication` drives `TrainingSession`; `DetSolver.fit()` does not import `deim_app`. Once parity passes, legacy `train.py` delegates to `TrainApplication`.
2. Existing inference scripts retain their current preprocessing behavior through explicit compatibility options. The new CLI uses one canonical preprocessing configuration recorded in `manifest.json`; behavior changes are not mixed into this refactor.
3. New tests use `tests/`; existing `test/` is a separate legacy gate.
4. Legacy checkpoint `optimizer_step` remains `None` until the first successful optimizer step after resume.

---

### Task 1: Freeze HBB and OBB Behavioral Baselines

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/helpers/baseline.py`
- Create: `tests/integration/test_legacy_entry_smoke.py`
- Create: `tests/integration/test_resume_baseline.py`
- Create: `tests/baselines/README.md`
- Create after approved execution: `tests/baselines/hbb-baseline.pt`
- Create after approved execution: `tests/baselines/obb-baseline.pt`

**Interfaces:**
- Produces: `BaselineRecord` with `prediction`, `losses`, `lr_trace`, `amp_scale_trace`, `ema_updates`, and resumable state summaries.
- Consumes: existing `train.py`, `BaseSolver.state_dict()`, HBB config `configs/custom/deimv2_dinov3_vits16p_coco128_freeze.yml`, and OBB config `configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep0_offset_per.yml`.

- [ ] **Step 1: Add deterministic test configuration and baseline record helpers**

```python
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class BaselineRecord:
    prediction: Mapping[str, torch.Tensor]
    losses: Mapping[str, float]
    lr_trace: tuple[float, ...]
    amp_scale_trace: tuple[float, ...]
    ema_updates: int
    last_epoch: int
```

- [ ] **Step 2: Write smoke tests that fail because baseline artifacts do not exist**

```python
def test_hbb_baseline_artifact_exists(baseline_dir):
    assert (baseline_dir / "hbb-baseline.pt").is_file()


def test_obb_baseline_artifact_exists(baseline_dir):
    assert (baseline_dir / "obb-baseline.pt").is_file()
```

- [ ] **Step 3: Run the tests and confirm RED**

Run: `pytest tests/integration/test_legacy_entry_smoke.py -v`

Expected: FAIL because both baseline artifacts are absent.

- [ ] **Step 4: Generate baselines from the unrefactored code**

Use fixed seed `0`, fixed mini datasets, one short epoch, and `max_optimizer_steps` to bound runtime. Record the exact command, CUDA/PyTorch versions, tolerances, and source checkpoint in `tests/baselines/README.md`. Never regenerate artifacts automatically during normal tests.

- [ ] **Step 5: Add numeric comparison assertions**

```python
torch.testing.assert_close(actual_boxes, expected_boxes, rtol=1e-4, atol=1e-5)
torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-4, atol=1e-5)
assert actual.lr_trace == pytest.approx(expected.lr_trace, rel=1e-7, abs=1e-10)
assert actual.ema_updates == expected.ema_updates
```

For OBB angles, compare the existing periodic angle distance and polygon coordinates rather than raw angle scalars.

- [ ] **Step 6: Verify GREEN and legacy tests**

Run: `pytest tests/integration/test_legacy_entry_smoke.py tests/integration/test_resume_baseline.py -v`

Run: `pytest test/ -q`

Expected: all available tests PASS; environment-dependent tests explicitly SKIP with reasons.

- [ ] **Step 7: Commit**

```bash
git add tests
git commit -m "test: 冻结 HBB 和 OBB 工程化重构基线"
```

**Gate G0:** No production files have moved; HBB and OBB fixed-input predictions, short training traces, and resume state are reproducible.

---

### Task 2: Add Runtime Configuration and Run Manifest Contracts

**Files:**
- Create: `deim_app/__init__.py`
- Create: `deim_app/errors.py`
- Create: `deim_app/config.py`
- Create: `deim_app/run_layout.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/unit/test_run_layout.py`
- Create: `tests/integration/test_manifest_creation.py`

**Interfaces:**
- Produces: `ResolvedRunConfig`, `RunConfigIssue`, `resolve_run_config()`, `validate_run_config()`, `RunLayout`, `create_run_layout()`, `write_manifest()`.
- Consumes: `engine.core.YAMLConfig` and `engine.core.yaml_utils.parse_cli()` without changing them.

- [ ] **Step 1: Write failing validation tests**

```python
def test_resume_and_tuning_are_mutually_exclusive(tmp_path):
    cfg = resolve_run_config(CONFIG, {"resume": "a.pth", "tuning": "b.pth"})
    with pytest.raises(ConfigurationError, match="resume.*tuning"):
        validate_run_config(cfg)


def test_config_digest_is_stable():
    first = resolve_run_config(CONFIG, {"seed": 0})
    second = resolve_run_config(CONFIG, {"seed": 0})
    assert first.config_digest == second.config_digest
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest tests/unit/test_config.py tests/unit/test_run_layout.py -v`

Expected: FAIL with `ModuleNotFoundError: deim_app`.

- [ ] **Step 3: Implement typed errors and frozen resolved config**

```python
@dataclass(frozen=True)
class ResolvedRunConfig:
    config_path: Path
    config_digest: str
    raw_config: Mapping[str, object]
    cli_overrides: Mapping[str, object]
    task: str
    box_mode: Literal["hbb", "obb"]
    num_classes: int
    output_dir: Path
    resume: Path | None
    tuning: Path | None
    device: str | None
    use_amp: bool
```

Implement the seven validation groups from the approved spec. Validation rejects only explicit contradictions; unknown legacy fields are recorded as warnings.

- [ ] **Step 4: Implement run layout and manifest**

`create_run_layout()` creates `checkpoints/`, `metrics/`, `exports/`, and `artifacts/`, then atomically writes `resolved_config.yml` and `manifest.json`. Manifest records command, config digest, Python/PyTorch/CUDA versions, device, seed, dataset paths, and checkpoint lineage.

- [ ] **Step 5: Run unit and integration tests**

Run: `pytest tests/unit/test_config.py tests/unit/test_run_layout.py tests/integration/test_manifest_creation.py -v`

Expected: PASS.

- [ ] **Step 6: Verify old config construction remains unchanged**

Run a representative HBB and OBB config through both `YAMLConfig` directly and `resolve_run_config()` followed by `YAMLConfig`; compare all existing effective fields used by training.

- [ ] **Step 7: Commit**

```bash
git add deim_app tests/unit tests/integration/test_manifest_creation.py
git commit -m "feat: 增加训练运行配置与 manifest 契约"
```

---

### Task 3: Add Versioned Checkpoint Inspection and Atomic Saving

**Files:**
- Create: `deim_app/checkpoint.py`
- Create: `tests/unit/test_checkpoint.py`
- Create: `tests/integration/test_checkpoint_compatibility.py`
- Modify: `engine/solver/_solver.py:202-248`

**Interfaces:**
- Produces: `CheckpointInfo`, `inspect_checkpoint(path, yaml_cfg=None)`, `augment_checkpoint_state(state, metadata)`, `atomic_torch_save(state, path)`.
- Preserves: existing `model`, `ema`, `optimizer`, `scaler`, `lr_scheduler`, `lr_warmup_scheduler`, `last_epoch` keys.

- [ ] **Step 1: Write failing old/new/corrupt checkpoint tests**

```python
def test_legacy_checkpoint_marks_optimizer_step_unknown(legacy_checkpoint):
    info = inspect_checkpoint(legacy_checkpoint, yaml_cfg=OBB_CONFIG)
    assert info.optimizer_step is None
    assert "optimizer_step" in info.unknown_fields


def test_corrupt_checkpoint_fails_before_model_setup(tmp_path):
    path = tmp_path / "broken.pth"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointCompatibilityError):
        inspect_checkpoint(path)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest tests/unit/test_checkpoint.py -v`

Expected: FAIL because checkpoint contracts do not exist.

- [ ] **Step 3: Implement metadata inspection without model construction**

`CheckpointInfo` includes schema version, task, box mode, config digest, model signature, class names, epoch, data/global step, optimizer step, runtime versions, present keys, and unknown fields. For old checkpoints, infer only facts supported by keys and user YAML.

- [ ] **Step 4: Implement atomic save and state augmentation**

Write to `<name>.tmp-<pid>`, flush and `fsync`, then `os.replace()`. Preserve all old state keys; add metadata at top level.

- [ ] **Step 5: Route `BaseSolver.state_dict()` and load through compatible helpers**

Keep public `state_dict()`, `load_state_dict()`, and `load_resume_state()` signatures unchanged. Do not import `deim_app` from `engine`; inject metadata/save handling from the application layer and retain current behavior when no metadata is supplied.

- [ ] **Step 6: Run compatibility and corruption tests**

Run: `pytest tests/unit/test_checkpoint.py tests/integration/test_checkpoint_compatibility.py -v`

Run: `pytest tests/integration/test_resume_baseline.py -v`

Expected: old checkpoints load unchanged, new metadata round-trips, broken files fail early.

- [ ] **Step 7: Commit**

```bash
git add deim_app/checkpoint.py tests engine/solver/_solver.py
git commit -m "feat: 增加版本化 checkpoint 检查与原子保存"
```

**Gate G1:** Existing configs and checkpoints still work, while every new run has a resolved config and manifest.

---

### Task 4: Introduce the Unified CLI and Application Services

**Files:**
- Create: `deim_app/__main__.py`
- Create: `deim_app/applications/__init__.py`
- Create: `deim_app/applications/train.py`
- Create: `deim_app/applications/evaluate.py`
- Create: `deim_app/applications/export.py`
- Create: `deim_app/applications/infer.py`
- Create: `deim_app/applications/inspect_checkpoint.py`
- Create: `tests/integration/test_cli.py`
- Modify: `train.py`

**Interfaces:**
- Produces: six CLI subcommands and `Application.run() -> int` services.
- Consumes: Tasks 2-3 contracts and existing `TASKS[cfg.task]` solver construction.

- [ ] **Step 1: Write parser and exit-code tests**

```python
@pytest.mark.parametrize("command", ["train", "resume", "eval", "export", "infer", "inspect-checkpoint"])
def test_cli_exposes_command(cli_runner, command):
    result = cli_runner("--help")
    assert command in result.stdout


def test_train_resume_and_resume_subcommand_resolve_equally(cli_runner):
    left = cli_runner("train", "-c", CONFIG, "--resume", CKPT, "--dry-run")
    right = cli_runner("resume", "-c", CONFIG, "-r", CKPT, "--dry-run")
    assert resolved_payload(left) == resolved_payload(right)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest tests/integration/test_cli.py -v`

Expected: FAIL because `python -m deim_app` has no entry point.

- [ ] **Step 3: Implement the parser and error mapping**

Map known application errors to stable nonzero exit codes; full traceback goes to the run log. CLI contains no model logic.

- [ ] **Step 4: Implement train/eval applications as adapters over existing solvers**

At this phase, applications call existing `solver.fit()` and `solver.val()`. Export and infer call current tools through internal functions, not subprocesses. `inspect-checkpoint` uses Task 3.

- [ ] **Step 5: Convert `train.py` to a compatibility wrapper**

Retain every current flag and default. Its `main(args)` translates arguments into `TrainApplication`; command-line behavior and printed resolved configuration remain covered by parity tests.

- [ ] **Step 6: Run CLI and legacy parity tests**

Run: `pytest tests/integration/test_cli.py tests/integration/test_legacy_entry_smoke.py -v`

Run: `python -m deim_app inspect-checkpoint -r <baseline-checkpoint>`

Expected: new and old paths resolve the same config and checkpoint state.

- [ ] **Step 7: Commit**

```bash
git add deim_app train.py tests/integration/test_cli.py
git commit -m "feat: 统一训练评估导出推理命令入口"
```

**Gate G2:** All six subcommands work; legacy `train.py` is behaviorally equivalent.

---

### Task 5: Establish Explicit HBB/OBB Prediction and Preprocessing Contracts

**Files:**
- Create: `deim_app/prediction.py`
- Create: `deim_app/pipeline/types.py`
- Create: `deim_app/pipeline/input_adapter.py`
- Create: `deim_app/pipeline/preprocessor.py`
- Create: `deim_app/pipeline/output_decoder.py`
- Create: `tests/unit/test_prediction.py`
- Create: `tests/unit/test_preprocessor.py`
- Create: `tests/component/test_output_decoder.py`

**Interfaces:**
- Produces: `PredictionBatch`, `PreprocessingConfig`, `PreparedBatch`, `InputAdapter`, `Preprocessor`, `OutputDecoder`.
- Consumes: existing deployed postprocessor outputs `(labels, boxes, scores)`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_box_mode_is_explicit_not_inferred():
    with pytest.raises(ValueError, match="box_mode"):
        PredictionBatch(class_ids=IDS, scores=SCORES, boxes=BOXES)


def test_obb_polygon_conversion_matches_existing_geometry(obb_prediction):
    expected = existing_obb_to_polygon(obb_prediction.boxes)
    torch.testing.assert_close(obb_prediction.to_polygons(), expected)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest tests/unit/test_prediction.py tests/unit/test_preprocessor.py -v`

Expected: FAIL because contracts do not exist.

- [ ] **Step 3: Implement immutable preprocessing metadata**

Record original size, resized size, scale, padding, color format, normalization, and image ID. Compatibility wrappers explicitly select the legacy resize mode; the new CLI records the canonical mode in manifest.

- [ ] **Step 4: Implement `PredictionBatch` validation and conversions**

HBB requires `xyxy`; OBB requires current validated physical representation. Delegate polygon/external-rect math to existing geometry functions rather than duplicating formulas.

- [ ] **Step 5: Implement the common output decoder**

Use existing postprocessor output, preserve EMA checkpoint selection, threshold after decoding, and retain timing metadata.

- [ ] **Step 6: Run unit/component and baseline tests**

Run: `pytest tests/unit/test_prediction.py tests/unit/test_preprocessor.py tests/component/test_output_decoder.py -v`

Run: `pytest tests/integration/test_legacy_entry_smoke.py -v`

- [ ] **Step 7: Commit**

```bash
git add deim_app/prediction.py deim_app/pipeline tests
git commit -m "feat: 统一 HBB 与 OBB 推理数据契约"
```

---

### Task 6: Unify PyTorch and ONNX Inference Pipelines

**Files:**
- Create: `deim_app/pipeline/backend.py`
- Create: `deim_app/pipeline/torch_backend.py`
- Create: `deim_app/pipeline/onnx_backend.py`
- Create: `deim_app/pipeline/result_writer.py`
- Create: `deim_app/pipeline/inference.py`
- Create: `tests/backend/test_torch_backend.py`
- Create: `tests/backend/test_onnx_consistency.py`
- Modify: `tools/inference/torch_inf.py`
- Modify: `tools/inference/torch_inf_vis.py`
- Modify: `tools/inference/onnx_inf.py`
- Modify: `tools/deployment/export_onnx.py`

**Interfaces:**
- Produces: `RuntimeBackend.run(prepared) -> BackendOutputs`, `InferencePipeline.run(source) -> list[PredictionBatch]`.
- Consumes: Task 5 contracts and existing model/postprocessor `.deploy()` methods.

- [ ] **Step 1: Write failing backend protocol and parity tests**

```python
def test_torch_backend_uses_deploy_model(model_factory):
    backend = TorchBackend(model_factory)
    assert backend.model_is_deployed


def test_onnx_matches_torch_for_hbb(fixed_hbb_input):
    assert_hbb_close(torch_result, onnx_result)


def test_onnx_matches_torch_for_obb(fixed_obb_input):
    assert_obb_periodic_and_polygon_close(torch_result, onnx_result)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest tests/backend/test_torch_backend.py tests/backend/test_onnx_consistency.py -v`

- [ ] **Step 3: Implement backend protocol, PyTorch backend, and pipeline**

Checkpoint loading, EMA preference, preprocessing, postprocessing, timing, and result writing exist only once. PyTorch backend calls both model and postprocessor in deploy mode.

- [ ] **Step 4: Implement ONNX backend and export release gate**

Export metadata records input size, dynamic axes, opset, task, box mode, and classes. Run fixed-sample consistency before marking an export releasable.

- [ ] **Step 5: Convert legacy PyTorch/ONNX scripts to wrappers**

Keep current flags, default output filenames, and legacy preprocessing selection. Remove duplicated model/checkpoint/postprocess logic only after parity tests pass.

- [ ] **Step 6: Run backend and CLI integration tests**

Run: `pytest tests/backend/test_torch_backend.py tests/backend/test_onnx_consistency.py tests/integration/test_cli.py -v`

Expected: PyTorch and ONNX pass fixed HBB/OBB tolerances; ONNX Runtime absence gives an explicit skip.

- [ ] **Step 7: Commit**

```bash
git add deim_app/pipeline tools/inference tools/deployment/export_onnx.py tests/backend
git commit -m "feat: 统一 PyTorch 与 ONNX 推理导出链路"
```

---

### Task 7: Add TensorRT and OpenVINO Backend Adapters

**Files:**
- Create: `deim_app/pipeline/tensorrt_backend.py`
- Create: `deim_app/pipeline/openvino_backend.py`
- Create: `tests/backend/test_optional_backends.py`
- Modify: `tools/inference/trt_inf.py`
- Modify: `tools/inference/openvino_inf.py`

**Interfaces:**
- Produces implementations of the Task 6 `RuntimeBackend` protocol.
- Consumes exactly the same `PreparedBatch`, `BackendOutputs`, and `OutputDecoder` as PyTorch/ONNX.

- [ ] **Step 1: Write explicit availability and skip tests**

```python
def test_tensorrt_unavailable_reports_reason():
    with pytest.raises(BackendRuntimeError, match="TensorRT"):
        TensorRTBackend.require_available()
```

- [ ] **Step 2: Implement adapters without preprocessing or postprocessing duplication**

- [ ] **Step 3: Convert legacy scripts to compatibility wrappers**

- [ ] **Step 4: Run optional backend tests**

Run: `pytest tests/backend/test_optional_backends.py -v`

Expected: PASS when installed; otherwise SKIP/expected error with the missing dependency named.

- [ ] **Step 5: Commit**

```bash
git add deim_app/pipeline tools/inference tests/backend/test_optional_backends.py
git commit -m "feat: 接入 TensorRT 与 OpenVINO 推理后端"
```

**Gate G3:** All inference/export entries share one preprocessing/output contract; PyTorch and ONNX consistency is mandatory.

---

### Task 8: Extract Precision and Optimization Controllers

**Files:**
- Create: `deim_app/training/__init__.py`
- Create: `deim_app/training/types.py`
- Create: `deim_app/training/precision.py`
- Create: `deim_app/training/optimization.py`
- Create: `tests/unit/test_precision.py`
- Create: `tests/unit/test_optimization.py`
- Create: `tests/component/test_training_state_matrix.py`

**Interfaces:**
- Produces: `StepResult`, `PrecisionController`, `OptimizationController`.
- Consumes: existing optimizer, EMA, FlatCosine scheduler, epoch scheduler, and warmup objects. Existing scaler checkpoint state remains a compatibility concern outside the active training controller.

- [ ] **Step 1: Write the full state-transition matrix first**

```python
@pytest.mark.parametrize("mode", ["fp32", "amp_bf16"])
def test_training_state_transition_matrix(
    mode,
    training_state_factory,
):
    state = training_state_factory(mode=mode)
    before = state.snapshot()
    result = state.run_one_batch()
    after = state.snapshot()

    assert after.parameters_changed_from(before)
    assert after.ema_changed_from(before)
    assert after.lr_changed_from(before)
    assert result.optimizer_step_succeeded
    if mode == "amp_bf16":
        assert result.model_autocast_dtype is torch.bfloat16
        assert result.criterion_input_dtype is torch.float32
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest tests/unit/test_precision.py tests/unit/test_optimization.py tests/component/test_training_state_matrix.py -v`

- [ ] **Step 3: Implement `PrecisionController`**

It owns BF16 autocast selection, nested floating-output conversion to FP32, finite-loss checks, direct backward, and returns `optimizer_step_succeeded`. It does not use GradScaler and never updates EMA or scheduler.

- [ ] **Step 4: Implement `OptimizationController`**

`on_data_step()` advances the iteration scheduler. `on_optimizer_success()` increments optimizer step and updates EMA. `on_epoch_end()` advances the epoch scheduler after warmup completion.

- [ ] **Step 5: Run state matrix and regression baselines**

Run: `pytest tests/component/test_training_state_matrix.py tests/integration/test_resume_baseline.py -v`

- [ ] **Step 6: Commit**

```bash
git add deim_app/training tests/unit tests/component
git commit -m "feat: 抽取 AMP 与优化状态控制器"
```

---

### Task 9: Extract Diagnostics Policy and TrainStepExecutor

**Files:**
- Create: `deim_app/training/diagnostics.py`
- Create: `deim_app/training/step.py`
- Create: `tests/unit/test_diagnostics_policy.py`
- Create: `tests/component/test_train_step.py`
- Modify: `engine/solver/det_engine.py:171-514`

**Interfaces:**
- Produces: `DiagnosticsPolicy(level)` and `TrainStepExecutor.run_step(samples, targets, *, epoch: int, data_step: int, optimizer_step: int) -> StepResult`.
- Consumes: Task 8 controllers and existing criterion/model interfaces.

- [ ] **Step 1: Write failing executor tests for FP32 and BF16-forward/FP32-loss modes**

Assert total loss, parameter changes, EMA, lr, model autocast dtype, criterion input dtype, gradient norm, `data_step`, and `optimizer_step` for every branch.

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest tests/component/test_train_step.py -v`

- [ ] **Step 3: Implement the executor using existing diagnostics functions**

The order is zero-grad, precision-controlled forward, FP32 output conversion when AMP is enabled, loss, finite checks, direct backward, diagnostics, clip, optimizer decision, optimization commit, event result. `standard` is default; monitoring/diagnostics cannot alter optimization semantics.

- [ ] **Step 4: Replace duplicated body in `train_one_epoch()` with executor calls**

Preserve the function signature and returned metrics. Keep tqdm/metric aggregation temporarily in `train_one_epoch()`; only compute/optimization leaves the function in this task.

- [ ] **Step 5: Run all historical regression gates**

Run: `pytest tests/component/test_train_step.py tests/integration/test_legacy_entry_smoke.py tests/integration/test_resume_baseline.py -v`

Run: `pytest test/ -q`

Expected: predictions, loss traces, model/criterion dtype traces, EMA updates, and lr traces match Task 1 baselines.

- [ ] **Step 6: Commit**

```bash
git add deim_app/training engine/solver/det_engine.py tests
git commit -m "refactor: 拆分单步训练计算与优化控制流"
```

**Gate G4:** BF16-forward/FP32-loss, EMA, and scheduler regressions are covered automatically; `train_one_epoch()` no longer duplicates precision/optimization branches.

---

### Task 10: Add Atomic Checkpoint Manager and Event Sinks

**Files:**
- Create: `deim_app/training/checkpoint_manager.py`
- Create: `deim_app/training/events.py`
- Create: `tests/unit/test_checkpoint_manager.py`
- Create: `tests/unit/test_event_sinks.py`
- Create: `tests/integration/test_resume_continuity.py`

**Interfaces:**
- Produces: `CheckpointManager.save_last/save_epoch/save_best/load_for_resume/load_model_only`, `EventSink`, `EventHub`, JSONL/TensorBoard/Comet sinks.
- Consumes: Tasks 2-3 run layout/checkpoint contracts.

- [ ] **Step 1: Write failing atomic-save, alias, resume, and sink-failure tests**

```python
def test_remote_sink_failure_does_not_abort(jsonl_sink, failing_sink):
    hub = EventHub([jsonl_sink, failing_sink])
    hub.emit(STEP_EVENT)
    assert jsonl_sink.contains(STEP_EVENT)


def test_best_stage_aliases_point_to_compatible_checkpoint(manager):
    state = {"model": {"weight": torch.tensor([1.0])}, "last_epoch": 0}
    path = manager.save_best(
        state=state,
        metric_name="mAP50_95",
        metric_value=0.5,
        legacy_alias="best_stg1.pth",
    )
    assert (manager.checkpoint_dir / "best_stg1.pth").exists()
    assert path.name == "best-mAP50_95.pth"
```

- [ ] **Step 2: Implement atomic checkpoint lifecycle**

Preserve old key names and stage aliases. Incompatible task/box mode/class/model signature fails before solver setup. `tuning` loads weights only; `resume` restores the complete state.

- [ ] **Step 3: Implement local-authoritative events**

JSONL always writes locally. TensorBoard and Comet are optional sinks; exceptions become warnings. Events have distinct step, validation, and checkpoint schemas.

- [ ] **Step 4: Verify resume continuity**

Run: `pytest tests/unit/test_checkpoint_manager.py tests/unit/test_event_sinks.py tests/integration/test_resume_continuity.py -v`

Assert epoch, data/global step, optimizer step, optimizer buffers, lr, AMP scale, and EMA updates.

- [ ] **Step 5: Commit**

```bash
git add deim_app/training tests
git commit -m "feat: 统一 checkpoint 生命周期与训练事件输出"
```

---

### Task 11: Extract TrainingSession and Complete Legacy Delegation

**Files:**
- Create: `deim_app/training/session.py`
- Create: `engine/solver/det_lifecycle.py`
- Create: `tests/helpers/imports.py`
- Create: `tests/unit/test_dependency_direction.py`
- Create: `tests/component/test_training_session.py`
- Create: `tests/integration/test_train_resume_eval_flow.py`
- Modify: `deim_app/applications/train.py`
- Modify: `engine/solver/det_solver.py:24-342`

**Interfaces:**
- Produces: `TrainingSession.run() -> TrainingResult`.
- Consumes: TrainStepExecutor, CheckpointManager, EventHub, existing evaluator/dataloaders/stage fields.

- [ ] **Step 1: Write a failing lifecycle-order test**

```python
def test_resume_session_order(training_session, fake_session_dependencies):
    result = training_session.run()
    assert fake_session_dependencies.calls == [
        "setup", "resume", "baseline_eval", "train_epoch", "validate", "checkpoint", "finalize"
    ]
```

- [ ] **Step 2: Write stage-transition compatibility tests**

Cover current stop epoch behavior, `best_stg1.pth` reload, EMA decay restart, validation cadence, periodic checkpoint cadence, and DDP sampler `set_epoch()`.

- [ ] **Step 3: Implement `TrainingSession`**

The session owns the epoch loop and lifecycle; it receives already constructed engine objects. `TrainApplication` constructs the existing solver components, then calls the session directly.

- [ ] **Step 4: Preserve `DetSolver.fit()` as an engine-local compatibility adapter**

Extract the existing lifecycle operations into `engine/solver/det_lifecycle.py` as engine-local functions with no `deim_app` dependency. `TrainingSession` calls those functions. `DetSolver.fit()` remains public and delegates to the same engine-local functions, preserving out-of-repository callers while preventing duplicate lifecycle logic.

Add the dependency guard before modifying `DetSolver`:

```python
def test_engine_never_imports_application_layer(project_root):
    violations = imports_from(project_root / "engine", "deim_app")
    assert violations == []
```

`tests/helpers/imports.py::imports_from(root: Path, package: str) -> list[Path]` parses Python files with `ast` and returns files containing either `import deim_app` or `from deim_app ...`.

- [ ] **Step 5: Run complete train-resume-eval flow**

Run: `pytest tests/component/test_training_session.py tests/integration/test_train_resume_eval_flow.py tests/integration/test_resume_continuity.py -v`

Run: `pytest tests/integration/test_legacy_entry_smoke.py -v`

Run: `pytest test/ -q`

- [ ] **Step 6: Commit**

```bash
git add deim_app/training/session.py deim_app/applications/train.py engine/solver/det_lifecycle.py engine/solver/det_solver.py tests/helpers/imports.py tests/unit/test_dependency_direction.py tests/component/test_training_session.py tests/integration/test_train_resume_eval_flow.py
git commit -m "refactor: 抽取训练会话与阶段生命周期"
```

**Gate G5:** New CLI and legacy entry both complete train-save-resume-eval with identical HBB/OBB behavior and continuous state.

---

### Task 12: Split Dependencies, Add Dependency Guards, and Complete Documentation

**Files:**
- Create: `requirements-train.txt`
- Create: `requirements-deploy.txt`
- Create: `requirements-dev.txt`
- Modify: `requirements.txt`
- Modify: `tests/unit/test_dependency_direction.py`
- Create: `docs/engineering/cli.md`
- Create: `docs/engineering/checkpoints.md`
- Create: `docs/engineering/run-layout.md`
- Create: `docs/engineering/troubleshooting.md`
- Modify: `README.md`
- Modify: `docs/superpowers/INDEX.md`

**Interfaces:**
- Produces: documented installation profiles and an automated `engine`-must-not-import-`deim_app` guard.

- [ ] **Step 1: Extend dependency tests for optional deployment imports**

```python
def test_core_package_does_not_import_optional_backends(project_root):
    violations = imports_from(
        project_root / "deim_app",
        {"tensorrt", "openvino", "onnxruntime"},
        exclude={"pipeline/tensorrt_backend.py", "pipeline/openvino_backend.py", "pipeline/onnx_backend.py"},
    )
    assert violations == []
```

- [ ] **Step 2: Split dependencies without changing pinned core versions**

`requirements.txt` remains a compatibility aggregate. Training adds dataset/evaluation/monitoring dependencies; deploy adds ONNX Runtime and optional backend notes; dev adds pytest and static tooling.

- [ ] **Step 3: Document the new and legacy workflows**

Include exact train/resume/eval/export/infer commands, run layout, checkpoint fields, legacy behavior, BF16-forward/FP32-loss semantics, and recovery procedures for corrupt/incompatible checkpoints.

- [ ] **Step 4: Run the complete verification matrix**

Run: `pytest tests/unit -v`

Run: `pytest tests/component -v`

Run: `pytest tests/integration -v`

Run: `pytest tests/backend -v`

Run: `pytest test/ -q`

Run HBB and OBB fixed baseline commands, then a single-machine DDP smoke test. Backend tests must either pass or explicitly skip with a named missing dependency.

- [ ] **Step 5: Commit**

```bash
git add requirements*.txt tests/unit/test_dependency_direction.py docs README.md
git commit -m "docs: 完成工程化平台依赖与运维说明"
```

**Gate G6:** The full platform is documented, dependency direction is enforced, all new and legacy gates pass, and no model-math file changed as part of the refactor.

---

## Execution Order and Review Gates

```text
Task 1 (G0)
  -> Tasks 2-3 (G1)
  -> Task 4 (G2)
  -> Tasks 5-7 (G3)
  -> Tasks 8-9 (G4)
  -> Tasks 10-11 (G5)
  -> Task 12 (G6)
```

Do not begin a task group until the preceding gate passes. Tasks 6 and 7 may be implemented in parallel after Task 5. Tasks 8 and 10 may be developed in parallel after checkpoint contracts exist, but Task 11 requires both.

## Completion Checklist

- [ ] Existing YAML files load without migration.
- [ ] Existing `train.py` and inference/deployment commands remain usable.
- [ ] Existing checkpoints load for inference, tuning, and resume.
- [ ] HBB and OBB baseline predictions and metrics remain within approved tolerance.
- [ ] BF16-forward/FP32-loss, EMA, scheduler, and resume continuity have automated tests.
- [ ] New CLI supports train, resume, eval, export, infer, and inspect-checkpoint.
- [ ] PyTorch and ONNX fixed-sample consistency passes for HBB and OBB.
- [ ] TensorRT/OpenVINO pass or explicitly skip based on installed runtime.
- [ ] JSONL remains complete when TensorBoard/Comet fail.
- [ ] New checkpoint writes are atomic and retain `best_stg1/2` compatibility aliases.
- [ ] `engine` has no dependency on `deim_app`.
- [ ] Single-machine DDP smoke test passes.
