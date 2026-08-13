# DEIM Application Review Blocker Repair Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair every confirmed application-layer merge blocker with regression tests, while preserving unrelated work and leaving merge, commit, and push actions to the user.

**Architecture:** Fix defects at their owning boundaries: adapter load owns module device placement and checkpoint compatibility, inference execution owns autograd suppression, config parsing owns writer validation and trusted include resolution, input enumeration owns output-identity uniqueness, and test configuration owns fixture discovery. Each behavioral repair follows an isolated red-green cycle before broad verification.

**Tech Stack:** Python, PyTorch, pytest, basedpyright, YAML application configuration.

## Global Constraints

- Work on `feat/deim-app-v1` at reviewed baseline `8cf3454`; do not merge, commit, or push.
- Preserve `configs/custom_obb/dlzdt/ablation/sp_fz_rect_576x1024.yml` and `out/` untouched.
- Add a failing regression test before every production-code change.
- Do not weaken `strict=False` checkpoint loading for non-class-head parameters.
- Do not limit pytest discovery with `testpaths` to hide legacy tests.
- Do not add compatibility shims or unrelated refactors.

---

### Task 1: Inference Device And Autograd Safety

**Files:**
- Modify: `test/deim_app/adapters/_stubs.py`
- Modify: `test/deim_app/adapters/test_deim_adapter.py`
- Modify: `test/deim_app/inference/test_torch_backend.py`
- Modify: `deim_app/adapters/deim.py`
- Modify: `deim_app/inference/torch_backend.py`

**Interfaces:**
- Consumes: `nn.Module.deploy() -> self`, `InferenceConfig.device`.
- Produces: deployed model and postprocessor on the configured device; forward and postprocessing run under `torch.inference_mode()`.

- [ ] Add tests proving `load()` calls `.to(inference.device)` after deploy on both modules.
- [ ] Run the adapter tests and verify failure because the stub has no recorded `.to()` calls.
- [ ] Add a backend test whose model and postprocessor assert `torch.is_inference_mode_enabled()`.
- [ ] Run the backend test and verify failure because inference mode is disabled.
- [ ] Move both deployed modules to `loaded.app.inference.device` and wrap model plus postprocessor execution in `torch.inference_mode()`.
- [ ] Run both focused test modules and verify success.

### Task 2: Canonical Application-Base Trust Boundary

**Files:**
- Modify: `test/deim_app/config/conftest.py`
- Modify: `test/deim_app/config/test_loader.py`
- Modify: `deim_app/config/loader.py`

**Interfaces:**
- Consumes: user YAML source path and one `__include__` entry.
- Produces: a resolved base path that is either an approved repository base or an explicitly test-approved path.

- [ ] Add tests proving an arbitrary same-named `hbb_app.yml` and a symlink to an outside same-named file are rejected.
- [ ] Run the tests and verify current basename-only validation accepts the malicious base.
- [ ] Replace raw-string directory checks with membership in resolved approved paths. Keep a monkeypatchable tuple of exact approved base paths for synthetic tests.
- [ ] Run all config-loader tests and verify legitimate sibling includes still work.

### Task 3: Checkpoint Class-Head Completeness

**Files:**
- Modify: `test/deim_app/adapters/test_deim_adapter.py`
- Modify: `deim_app/adapters/deim.py`

**Interfaces:**
- Consumes: model and checkpoint state mappings.
- Produces: `CheckpointCompatibilityError` before `load_state_dict` when any model class-head parameter is absent or shape-incompatible.

- [ ] Add a test with a model class-head key absent from the checkpoint.
- [ ] Run it and verify current code proceeds to `load_state_dict`.
- [ ] Extend `_verify_class_count_compatibility` to report missing model class-head keys as incompatibilities while retaining checkpoint-only-key tolerance.
- [ ] Run all adapter tests and verify matching and checkpoint-only cases remain valid.

### Task 4: Writer Format Validation At Configuration Boundary

**Files:**
- Modify: `test/deim_app/config/test_loader.py`
- Modify: `deim_app/config/schema.py`

**Interfaces:**
- Consumes: `data.format` and `inference.output_formats`.
- Produces: only `json`, `visualization`, and compatible `dota` writer selections.

- [ ] Replace the existing permissive `csv` tuple test with separate tuple-coercion and unknown-format rejection tests.
- [ ] Add HBB+DOTA rejection and OBB+DOTA acceptance tests.
- [ ] Run focused tests and verify unknown/HBB-incompatible formats currently pass.
- [ ] Validate known values in `_build_inference`, then perform the cross-section DOTA/box-mode check in `AppConfig.from_mapping`.
- [ ] Run all config tests and CLI tests.

### Task 5: Duplicate Output Identity Prevention

**Files:**
- Modify: `test/deim_app/inference/test_inputs.py`
- Modify: `deim_app/inference/inputs.py`

**Interfaces:**
- Consumes: non-recursive supported image files in one directory.
- Produces: unique stem-based `image_id` values or `InputSourceError` before inference.

- [ ] Add a test for `same.jpg` plus `same.png` in one directory.
- [ ] Run it and verify both inputs currently receive `image_id == "same"`.
- [ ] Detect duplicate stems during directory enumeration and raise `InputSourceError` naming the collision.
- [ ] Run all input and writer tests.

### Task 6: Pytest 9 Fixture Registration

**Files:**
- Modify: `test/deim_app/conftest.py`
- Modify: `test/deim_app/test_api.py` or add a focused fixture-collection test only if needed.

**Interfaces:**
- Consumes: facade fixture functions in `_facade_fakes.py`.
- Produces: fixtures discoverable without a nested `pytest_plugins` declaration.

- [ ] Run `python -m pytest test/deim_app/test_api.py --collect-only -q` with pytest 9 and capture the nested-plugin failure when reproducible.
- [ ] Import and re-export the fixture functions directly from `_facade_fakes.py` in `conftest.py`, using an explicit file-loaded module only if package import rules require it.
- [ ] Remove `pytest_plugins` and update stale module documentation.
- [ ] Verify focused facade collection and `python -m pytest test/deim_app -q`.

### Task 7: Static Type Errors

**Files:**
- Modify: `deim_app/predictions/collection.py`
- Modify: `deim_app/writers/visualization.py`
- Modify writer imports only where the fresh basedpyright output requires it.

**Interfaces:**
- Produces: fully parameterized color tuples and explicit drawing helper parameter types without circular runtime imports.

- [ ] Run fresh basedpyright and record the exact current errors.
- [ ] Add `tuple[int, int, int]` annotations and `ImagePrediction` helper annotations.
- [ ] Resolve writer/collection cycles using `TYPE_CHECKING` imports if the checker reports them; do not add suppressions.
- [ ] Run basedpyright on all `deim_app` and require zero errors.

### Task 8: Final Verification And Scope Review

**Files:**
- Verify all changed files and the final working-tree diff.

- [ ] Run `python -m compileall -q deim_app test/deim_app`.
- [ ] Run focused tests for every repaired boundary.
- [ ] Run `python -m pytest test/deim_app -q`.
- [ ] Run the three legacy regression modules used by the review.
- [ ] Run basedpyright and LSP diagnostics for every changed Python file.
- [ ] Perform manual CPU inference-mode and malicious-include smoke checks.
- [ ] Run `git diff --check`, inspect `git status --short`, and confirm only intended files changed.
- [ ] Confirm no commit, merge, or push occurred.
