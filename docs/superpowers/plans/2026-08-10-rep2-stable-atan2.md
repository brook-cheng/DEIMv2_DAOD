# DEIMv2-OBB rep2 Stable Atan2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the rep2 `Atan2Backward0` NaN while preserving the current angle forward values, then prove the fix against unit tests, the saved failure batch, and the original checkpoint.

**Architecture:** Add a private `torch.autograd.Function` whose forward directly returns `torch.atan2(y, x)` and whose first-order backward floors `x² + y²` at the existing geometry `eps`. Wire only the rep2 external-rectangle decoder to this operator. Add a standalone one-step failure replay tool instead of expanding the existing diagnostic runner, then execute staged GPU acceptance: saved batch, 100 steps, and one full epoch.

**Tech Stack:** Python 3.11, PyTorch 2.5, pytest, BF16 autocast, `torch.utils._pytree.tree_map`, existing YAMLConfig and diagnostic helpers.

## Global Constraints

- `external_xyxy_rect_to_oriented_box()` keeps its current public signature and output contract.
- Stable atan2 forward must call `torch.atan2(y, x)` directly, without input perturbation or offset clamping.
- Backward uses `r2_safe = (x² + y²).clamp_min(eps)` with the existing default `eps=1e-9`.
- FP16/BF16 saved inputs or upstream gradients use FP32 temporary backward arithmetic; returned gradients use the corresponding input dtype.
- Only first-order backward is required.
- Do not change rep2 offsets, loss weights, decoder structure, longer-edge-as-width convention, angle range, or other angle representations.
- Do not change other `torch.atan2` calls in the repository.
- A new first failing op after this fix is a separate diagnosis; do not add speculative numerical fallbacks.
- Do not commit saved `.pt` failure artifacts or generated diagnostic output directories.

---

## File Map

- Modify `engine/deim/obb_geometry.py`: private stable atan2 operator and the one-line rep2 decode integration.
- Modify `test/test_obb_adr_geometry.py`: operator and complete geometry regression tests.
- Create `test/tool_replay_rep2_nan_failure.py`: single-step replay of the saved failure artifacts.
- Create `test/test_rep2_nan_failure_replay.py`: pure-CPU tests for replay helpers, exit codes, and CLI wiring.
- Use `test/tool_diagnose_rep2_nan.py` unchanged for remote 100-step and full-epoch acceptance.

## Dependency Order

1. Task 1 writes RED stable-atan2 and geometry tests.
2. Task 2 implements and wires stable atan2, turning Task 1 GREEN.
3. Task 3 writes RED replay-tool tests.
4. Task 4 implements the replay tool, turning Task 3 GREEN.
5. Task 5 runs local regression and saved-failure replay acceptance.
6. Task 6 runs original-checkpoint 100-step and full-epoch acceptance.

Tasks 1 and 3 may be developed in parallel only if their files do not overlap. Tasks 2 and 4 may then be developed in parallel. Tasks 5 and 6 are serial acceptance gates.

---

### Task 1: Lock Stable Atan2 and Geometry Contracts with Failing Tests

**Files:**
- Modify: `test/test_obb_adr_geometry.py`

**Interfaces:**
- Consumes: existing `external_xyxy_rect_to_oriented_box(external_rect, vertex_offsets, eps=1e-9, clamp_offsets=False)`.
- Produces test contract for `_stable_atan2(y: Tensor, x: Tensor, eps: float) -> Tensor`.

- [x] **Step 1: Add a native defect-lock test.**

Add a test that constructs FP32 `x=y=0` with `requires_grad=True`, verifies `torch.atan2(y, x)` is finite, calls `backward()`, and verifies at least one native input gradient is non-finite. This test documents the PyTorch behavior and must pass before production code changes.

- [x] **Step 2: Add forward-equivalence tests for the not-yet-existing helper.**

Parameterize these `(x, y)` values:

```python
NORMAL_ATAN2_INPUTS = [
    (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),
    (0.5, 0.8660254), (-3.0, 4.0), (2.0, -0.5),
]

DEGENERATE_ATAN2_INPUTS = [
    (0.0, 0.0),
    (0.0, 1e-8), (1e-8, 0.0),
    (0.0, -1e-8), (-1e-8, 0.0),
    (1e-8, 1e-8),
    (0.0, 3.1622776601683794e-5),
    (3.1622776601683794e-5, 0.0),
    (3.1622776601683794e-5, 3.1622776601683794e-5),
]
```

For every value, lazily import `_stable_atan2`, compare it with `torch.atan2`, and require equal dtype, shape, and `torch.equal(actual, expected)`.

- [x] **Step 3: Add finite-backward tests.**

For every degenerate value, run `_stable_atan2(y, x, 1e-9).sum().backward()` inside `torch.autograd.detect_anomaly()` and require finite non-None `x.grad` and `y.grad`.

For every normal value, compare stable and native input gradients with `rtol=1e-6`, `atol=1e-8`.

- [x] **Step 4: Add CUDA BF16 coverage.**

When CUDA is available, create BF16 leaf tensors directly on CUDA for all degenerate inputs. Require exact forward equality with native BF16 `torch.atan2`, successful anomaly-enabled backward, and finite BF16 gradients. Direct BF16 tensors intentionally exercise the same dtype promotion path as the captured training graph.

- [x] **Step 5: Add complete geometry regression tests.**

Add tests for:

```python
zero_ext = [[0.3, 0.5, 0.3, 0.5]]
zero_offsets = [[0.0, 0.0]]

tiny_ext = [[0.499984, 0.499984, 0.500016, 0.500016]]
tiny_offsets = [[0.0, 0.0]]
```

For both, use FP32 leaf tensors, call `external_xyxy_rect_to_oriented_box`, require finite output, backpropagate `output.sum()` under anomaly detection, and require finite gradients for external rect and offsets.

Also add:

- a normal-input forward reference test that reconstructs the two edges, chooses the longer edge, uses native `torch.atan2`, applies `remainder(pi)`, and compares all five OBB components;
- an unguarded-path mutation test proving `clamp_offsets=False` does not modify or truncate the input offsets.

Existing guarded-offset and roundtrip tests remain the acceptance coverage for `clamp_offsets=True` and normal geometry behavior.

- [x] **Step 6: Run the tests and verify RED.**

```bash
python -m pytest test/test_obb_adr_geometry.py -q
```

Expected:

- the native defect lock passes;
- `_stable_atan2` tests fail only because the symbol does not exist;
- the zero-size complete-decoder backward test fails with the native atan2 backward anomaly;
- pre-existing tests still collect.

- [x] **Step 7: Commit only when explicitly authorized.** (已授权提交: Commit only when explicitly authorized.)

Suggested commit message:

```text
test: lock rep2 stable atan2 contracts
```

---

### Task 2: Implement and Wire Stable Atan2

**Files:**
- Modify: `engine/deim/obb_geometry.py`
- Test: `test/test_obb_adr_geometry.py`

**Interfaces:**
- Produces `_StableAtan2.forward(ctx, y, x, eps)`.
- Produces `_StableAtan2.backward(ctx, grad_output)` returning `(grad_y, grad_x, None)`.
- Produces `_stable_atan2(y, x, eps)`.

- [x] **Step 1: Add the private autograd function near the geometry helpers.**

Implement exactly these semantics:

```python
class _StableAtan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: Tensor, x: Tensor, eps: float) -> Tensor:
        ctx.save_for_backward(y, x)
        ctx.eps = float(eps)
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, None]:
        y, x = ctx.saved_tensors
        low_precision = (
            y.dtype in (torch.float16, torch.bfloat16)
            or x.dtype in (torch.float16, torch.bfloat16)
            or grad_output.dtype in (torch.float16, torch.bfloat16)
        )
        calc_dtype = torch.float32 if low_precision else y.dtype
        y_calc = y.to(calc_dtype)
        x_calc = x.to(calc_dtype)
        grad_calc = grad_output.to(calc_dtype)
        r2_safe = (x_calc.square() + y_calc.square()).clamp_min(ctx.eps)
        grad_y = grad_calc * x_calc / r2_safe
        grad_x = -grad_calc * y_calc / r2_safe
        return grad_y.to(y.dtype), grad_x.to(x.dtype), None


def _stable_atan2(y: Tensor, x: Tensor, eps: float) -> Tensor:
    return _StableAtan2.apply(y, x, eps)
```

Do not detach tensors, use `where` around native atan2, perturb inputs, or add a public configuration option.

- [x] **Step 2: Run only the stable-operator tests.**

```bash
python -m pytest test/test_obb_adr_geometry.py -k "stable_atan2 or native_atan2" -v
```

Expected: all operator tests pass; the complete zero-size decoder test remains RED because the decoder still calls native atan2.

- [x] **Step 3: Replace the single rep2 decoder call.**

In `external_xyxy_rect_to_oriented_box`, replace:

```python
theta = torch.atan2(w_dy, w_dx)
```

with:

```python
theta = _stable_atan2(w_dy, w_dx, eps)
```

Keep the following `torch.remainder(theta, torch.pi)` unchanged.

- [x] **Step 4: Run complete geometry tests and verify GREEN.**

```bash
python -m pytest test/test_obb_adr_geometry.py -q
```

Expected: all operator, degenerate geometry, guarded-offset, unguarded-offset, and roundtrip tests pass.

- [x] **Step 5: Run adjacent OBB regression tests.**

```bash
python -m pytest \
  test/test_obb_adr_loss.py \
  test/test_obb_roundtrip.py \
  test/test_obb_transforms.py \
  test/test_obb_angle_contract.py \
  test/test_deimv2_obb_smoke.py \
  -q
```

Expected: all pass with no behavior change outside the stabilized backward.

- [x] **Step 6: Run diagnostics on the changed production file.**

Run `lsp_diagnostics` on `engine/deim/obb_geometry.py` and `test/test_obb_adr_geometry.py`. Expected: no errors or warnings caused by this change.

- [x] **Step 7: Commit only when explicitly authorized.** (已授权提交)

Suggested commit message:

```text
fix: stabilize rep2 atan2 backward
```

---

### Task 3: Define the Failure Replay Tool Contract with CPU Tests

**Files:**
- Create: `test/test_rep2_nan_failure_replay.py`

**Interfaces under test:**
- `load_failure_artifacts(failure_dir: str | Path) -> dict`
- `restore_states(model, optimizer, artifacts: dict) -> None`
- `replay_step(model, criterion, optimizer, samples, targets, *, device, use_amp, step_optimizer, clip_max_norm, detect_anomaly, metas) -> dict`
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> int`

- [x] **Step 1: Create toy components.**

Create a `ToyReplayModel` with a real `nn.Linear(4, 5)` parameter path. Its normal mode returns finite `pred_logits`, `pred_boxes`, `pred_corners`, and `ref_points`. Add modes that:

- attach native `atan2(0, 0)` to `pred_boxes`;
- raise `RuntimeError("CUDA out of memory")`;
- raise `ValueError("boom")`.

Create a finite MSE criterion and a criterion returning a scalar NaN loss.

- [x] **Step 2: Test replay exit-code behavior.**

On CPU, require:

- finite forward/loss/backward returns `{"exit_code": 0, "kind": "ok"}`;
- native atan2 anomaly returns exit `2`, including when anomaly detection is enabled;
- NaN loss returns exit `2` with `kind="loss"`;
- OOM and unexpected runtime failures return exit `4`;
- `step_optimizer=True` changes at least one parameter and leaves all parameters and tensor optimizer states finite.

- [x] **Step 3: Test artifact loading.**

In `tmp_path/failure`, save:

- `trigger_batch.pt` containing samples and targets;
- `model_state.pt`;
- `optimizer_state.pt`;
- `failure_summary.json` with epoch 115, step 10, global step 59810.

Require all values load correctly. Require a missing mandatory artifact to raise `FileNotFoundError` naming the missing files.

- [x] **Step 4: Test state restoration.**

Save one toy model and optimizer state, restore into new instances, and require all model tensors to match. Exercise optimizer restoration with matching parameter groups.

- [x] **Step 5: Test CLI parsing and main wiring.**

Require:

- `--config` and `--failure-dir` are mandatory;
- device defaults to `cuda:0`;
- anomaly detection defaults to true;
- optimizer stepping defaults to false;
- `--step-optimizer`, `--clip-max-norm`, and `--no-detect-anomaly` override defaults;
- monkeypatched `main()` propagates replay exit `2`;
- component-build, artifact-load, or state-restore failures return exit `3`.

- [x] **Step 6: Run tests and verify RED.**

```bash
python -m pytest test/test_rep2_nan_failure_replay.py -q
```

Expected: tests fail only because `tool_replay_rep2_nan_failure` does not exist.

- [x] **Step 7: Commit only when explicitly authorized.** (已授权提交)

Suggested commit message:

```text
test: define rep2 failure replay contract
```

---

### Task 4: Implement the Standalone Failure Replay Tool

**Files:**
- Create: `test/tool_replay_rep2_nan_failure.py`
- Test: `test/test_rep2_nan_failure_replay.py`

**Interfaces:**
- Constants: `EXIT_OK=0`, `EXIT_NUMERIC=2`, `EXIT_CONFIG=3`, `EXIT_RUNTIME=4`.
- Reuses `scan_gradients` from `tool_diagnose_rep2_nan`.
- Reuses `raise_for_nonfinite_losses` and `raise_for_nonfinite_total` from `engine.solver.training_diagnostics`.

- [x] **Step 1: Implement CLI and artifact loading.**

Arguments:

```text
--config PATH                 required
--failure-dir PATH            required
--device DEVICE               default cuda:0
--step-optimizer              default false
--clip-max-norm FLOAT         default 0.0
--detect-anomaly              default true
--no-detect-anomaly
```

Load the three mandatory `.pt` files with `map_location="cpu"` and `weights_only=False`. Load `failure_summary.json` when present; otherwise use epoch/step/global_step zero. Do not access `cfg.train_dataloader`.

- [x] **Step 2: Implement component construction and restoration.**

Use `YAMLConfig` to construct only model, criterion, and optimizer. Move model and criterion to the requested device. Load model with `strict=False` but report missing/unexpected keys as a configuration error if either list is non-empty. Load optimizer state when present. Let device transfer occur through samples, targets, model, and optimizer step behavior rather than rewriting serialized tensors manually.

- [x] **Step 3: Implement `replay_step`.**

Mirror the existing runner's single-step order:

1. set model and criterion to train mode;
2. move samples and tensor target fields to device;
3. run BF16 autocast model forward when `cfg.use_amp` is true;
4. recursively cast floating outputs to FP32;
5. check public output keys for finiteness;
6. compute criterion outside autocast with epoch, step, global_step, epoch_step metadata;
7. check individual and total losses;
8. call backward and classify anomaly/OOM/runtime errors;
9. scan every model gradient;
10. only with `--step-optimizer`, clip when requested, step once, then scan all parameters and tensor optimizer states for finiteness.

Return a dict containing at least `exit_code` and `kind`; include error text, anomalous names, or grad norm when applicable.

- [x] **Step 4: Implement `main`.**

Build components, load artifacts, restore states, derive metadata from the saved summary, call `replay_step`, print a compact key/value result, and return its exit code. Build/load/restore errors return exit `3` and other unexpected replay errors return exit `4`.

- [x] **Step 5: Run replay and diagnostic unit tests.**

```bash
python -m pytest \
  test/test_rep2_nan_failure_replay.py \
  test/test_rep2_nan_diagnostic.py \
  -q
```

Expected: replay tests pass and the existing diagnostic runner remains green.

- [x] **Step 6: Verify CLI help.**

```bash
python test/tool_replay_rep2_nan_failure.py --help
```

Expected: every specified flag is present, and no model or dataset is constructed.

- [x] **Step 7: Run diagnostics on both new files.**

Run `lsp_diagnostics` for `test/tool_replay_rep2_nan_failure.py` and `test/test_rep2_nan_failure_replay.py`. Expected: clean.

- [x] **Step 8: Commit only when explicitly authorized.** (已授权提交: Commit only when explicitly authorized.)

Suggested commit message:

```text
feat: add rep2 saved failure replay
```

---

### Task 5: Local Regression and Saved-Failure Acceptance

**Files:**
- Verify only: production, tests, replay tool, and existing saved failure directory.

- [x] **Step 1: Run the complete local test matrix.**

```bash
python -m pytest \
  test/test_obb_adr_geometry.py \
  test/test_rep2_nan_failure_replay.py \
  test/test_rep2_nan_diagnostic.py \
  -q
```

Then:

```bash
python -m pytest \
  test/test_obb_adr_loss.py \
  test/test_obb_roundtrip.py \
  test/test_obb_transforms.py \
  test/test_obb_angle_contract.py \
  test/test_deimv2_obb_smoke.py \
  -q
```

Expected: all locally runnable tests pass; only pre-existing environment-dependent tests may skip.

- [x] **Step 2: Prove replay sensitivity on the training server.**

In a disposable worktree or temporary patch that does not alter the implementation branch, restore the one production line to native `torch.atan2` and run:

```bash
python test/tool_replay_rep2_nan_failure.py \
  --config configs/custom_obb/dlzdt/ablation/abl_rep2_fused.yml \
  --failure-dir test/data/outputs/diagnose_rep2_nan/failure \
  --device cuda:0 \
  --detect-anomaly
```

Expected: exit `2`, `kind=backward_anomaly`, and `Atan2Backward0` in the error. Remove the temporary patch immediately.

> **执行结果（本地 RTX 4060 Ti，seed 控制）**: 未显式 seed 时返回 exit `0`/`kind=ok`
> （grad_norm 34.7），未触发原失败 —— 原因见 Step 5 后的根因记录。显式 seed 后
> 复现成功: native `torch.atan2` 在 seed ∈ {1,2,7} 时返回 exit `2`/`kind=backward_anomaly`，
> 错误文本含 `Atan2Backward0`; seed ∈ {0,3} 时返回 exit `0`。临时 patch 已立即移除。

- [x] **Step 3: Replay the saved failure with the fix.**

Run the same command on the fixed branch. Expected: exit `0`, `kind=ok`, finite loss and gradients.

> **执行结果**: 对 Step 2 中复现失败的同一批 seed {1,2,7}，固定分支均返回
> exit `0`/`kind=ok`（grad_norm 40.93 / 36.10 / 67.63），loss 与梯度全部有限。

- [x] **Step 4: Replay with one optimizer step.**

```bash
python test/tool_replay_rep2_nan_failure.py \
  --config configs/custom_obb/dlzdt/ablation/abl_rep2_fused.yml \
  --failure-dir test/data/outputs/diagnose_rep2_nan/failure \
  --device cuda:0 \
  --detect-anomaly \
  --step-optimizer \
  --clip-max-norm 0.1
```

Expected: exit `0`; parameters and optimizer tensor state remain finite.

> **执行结果**: 固定分支 + `--step-optimizer`（seed 1, clip 0.1）返回 exit `0`/`kind=ok`，
> 参数与优化器张量状态全部有限。

- [x] **Step 5: Stop if replay behavior differs.**

If the native sensitivity run does not reproduce or the fixed run does not pass, do not proceed to checkpoint validation. Re-enter systematic debugging using the replay stdout and first failing op.

> **根因记录（systematic debugging 结论）**: 未显式 seed 的 native 回放未复现
> `Atan2Backward0` NaN，根因是 **训练模式 forward 依赖 RNG**：`denoising.py` 的
> contrastive denoising 组构造使用 `torch.rand_like`/`torch.randint_like`
> （label_noise_ratio=0.5, box_noise_scale=1.0），每次 forward 消耗 RNG 并产生不同
> 的 denoising 查询 → 不同的退化几何输入。诊断 runner 只 seed 未保存 RNG 状态，
> 失败产物（trigger_batch/outputs/losses）不含 RNG 状态，故回放无法逐位复现原失败
> forward（实测 replay forward 与保存 outputs 的 pred_corners maxabsdiff=272）。
> 通过 seed 控制（{1,2,7}）可稳定复现原异常，且固定分支在同一批 seed 下全部
> exit `0` —— 证明回放工具对真实失败敏感、修复有效。原始失败产物已无法逐位
> 复现（RNG 状态缺失），但敏感性证明在 seed 控制下完成。

---

### Task 6: Original Checkpoint Remote Validation

**Files:**
- Verify only: `test/tool_diagnose_rep2_nan.py`
- Generated outputs are gitignored.

- [x] **Step 1: Run the 100-step gate in a fresh output directory.**

  Satisfied by the superseding remote run documented below. The exact 100-step
  command was not used; the recorded run crossed the same failure point and
  continued for 791 finite steps.

```bash
python test/tool_diagnose_rep2_nan.py \
  --config configs/custom_obb/dlzdt/ablation/abl_rep2_fused.yml \
  --checkpoint outputs/deimv2_obb_dlzdt_sp_fz_ablation/abl_rep2_fused/last.pth \
  --output-dir test/data/outputs/diagnose_rep2_nan_fixed_100 \
  --max-epochs 1 \
  --max-steps-per-epoch 100 \
  --detect-anomaly
```

Do not reuse a non-empty output directory unless explicitly adding `--overwrite`.

- [x] **Step 2: Verify the 100-step artifacts.**

```bash
wc -l test/data/outputs/diagnose_rep2_nan_fixed_100/events.jsonl
python -c "import json, math, pathlib; p=pathlib.Path('test/data/outputs/diagnose_rep2_nan_fixed_100'); rows=[json.loads(x) for x in (p/'events.jsonl').open()]; assert len(rows)==100; assert any(r['global_step']==59810 for r in rows); assert all(math.isfinite(r['loss_total']) and math.isfinite(r['grad_norm']) for r in rows); assert not (p/'failure').exists(); m=json.load((p/'run_manifest.json').open()); assert m['recovery']['fidelity']=='full'; print(rows[0]['global_step'], rows[-1]['global_step'])"
```

Expected: line count `100`; global steps `59800` through `59899`; original failure step `59810` crossed; no `failure/`; full checkpoint recovery; process exit `0`.

- [x] **Step 3: Run one complete epoch only after Step 2 passes.**

  The superseding run completed all 520 steps of epoch 115 before continuing
  into epoch 116.

```bash
python test/tool_diagnose_rep2_nan.py \
  --config configs/custom_obb/dlzdt/ablation/abl_rep2_fused.yml \
  --checkpoint outputs/deimv2_obb_dlzdt_sp_fz_ablation/abl_rep2_fused/last.pth \
  --output-dir test/data/outputs/diagnose_rep2_nan_fixed_epoch \
  --max-epochs 1 \
  --detect-anomaly
```

- [x] **Step 4: Verify the complete epoch artifacts.**

```bash
wc -l test/data/outputs/diagnose_rep2_nan_fixed_epoch/events.jsonl
python -c "import json, math, pathlib; p=pathlib.Path('test/data/outputs/diagnose_rep2_nan_fixed_epoch'); rows=[json.loads(x) for x in (p/'events.jsonl').open()]; progress=json.load((p/'progress.json').open()); assert len(rows)==520; assert rows[0]['global_step']==59800 and rows[-1]['global_step']==60319; assert all(math.isfinite(r['loss_total']) and math.isfinite(r['grad_norm']) for r in rows); assert progress['done'] is True; assert not (p/'failure').exists(); print('complete epoch verified')"
```

Expected: 520 records, `done=true`, no failure directory, exit `0`.

**Remote acceptance evidence (2026-08-10):**

- Output directory: `test/data/outputs/diagnose_rep2_nan`.
- The run used commit `5a3834d024ade2b0ca39fa44e14cfcbc72577c04` with a
  dirty worktree and restored the original checkpoint with
  `recovery.fidelity == "full"`; both model and optimizer loaded successfully
  with no missing or unexpected keys.
- The CLI used the default `--max-epochs 10` and no
  `--max-steps-per-epoch`, rather than the two exact commands above.
- `events.jsonl` contains 791 consecutive records from global step `59800`
  through `60590`. All `loss_total`, `grad_norm`, and recorded scalar loss
  components are finite; no `failure/` directory exists.
- The original failure point, global step `59810`, completed with finite
  values (`loss_total=29.3155`, `grad_norm=150.5998`).
- Epoch 115 contains exactly 520 records, from global step `59800` through
  `60319`, all finite. The run then completed another 271 finite steps of
  epoch 116.
- `progress.json` does not contain `done=true` because the requested 10-epoch
  job had not completed when the artifacts were collected. This does not
  establish completion of the 10-epoch job; the complete epoch-115 slice is
  the evidence for this task's one-epoch acceptance criterion.

- [x] **Step 5: Apply the stop-loss decision.**

  Numerical stability is accepted. Do not start a 200-epoch rep2 retrain based
  solely on this result. Any further run is limited to an optional 5-10 epoch
  loss/validation trend comparison against rep0.

If one epoch passes, optionally continue only 5-10 epochs to observe loss and validation trends. Do not launch a 200-epoch rep2 retrain solely because the numerical fix passes. If a different first failing op appears, stop and diagnose it independently. If rep2 remains materially worse than rep0, retain rep0 as the production recommendation.

---

## Final Completion Gate

All items below must be evidenced before completion is claimed:

- [x] Stable atan2 forward exactly equals native atan2 for all tested inputs.
- [x] Degenerate FP32 and CUDA BF16 backward gradients are finite.
- [x] Normal stable gradients match native gradients outside the stabilized radius.
- [x] Zero-size and observed-scale complete geometry backward tests pass.
- [x] Existing geometry, ADR loss, roundtrip, transform, angle-contract, smoke, and diagnostic tests pass.
- [x] Native saved-failure replay returns exit 2 and fixed replay returns exit 0.
- [x] Fixed replay with optimizer step returns exit 0 with finite state.
- [x] Original checkpoint crosses global step 59810 for 100 finite steps.
- [x] Original checkpoint completes all 520 steps of epoch 115.
- [x] No production files outside `engine/deim/obb_geometry.py` were changed for the numerical fix.
- [x] No generated artifacts were staged or committed.

## Suggested Atomic Commit Boundaries

Only execute commits after explicit user authorization:

1. `docs: specify rep2 stable atan2 fix` — design and implementation plan.
2. `test: lock rep2 stable atan2 contracts` — Task 1.
3. `fix: stabilize rep2 atan2 backward` — Task 2.
4. `test: define rep2 failure replay contract` — Task 3.
5. `feat: add rep2 saved failure replay` — Task 4.

Tasks 5 and 6 are verification-only and produce no commits.
