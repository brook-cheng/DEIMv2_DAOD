# OBB Offline H/W Ordering Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the standalone offline OBB inference entry pass canvas dimensions to `PostProcessor` in `[W, H]` order while preserving the public `imgsz=(H, W)` convention.

**Architecture:** Keep all existing size conventions unchanged and convert only at the confirmed faulty boundary in `test/tool_deimv2_obb_infer.py`. Lock the behavior with an entry-point regression test and a PostProcessor characterization test. Do not modify coordinate recovery or OBB geometry in this phase.

**Tech Stack:** Python 3.11, PyTorch, torchvision, Pillow, pytest.

## Global Constraints

- `imgsz` remains `(H, W)`.
- `PostProcessor.orig_target_sizes` remains `[W, H]`.
- Do not modify `engine/deim/postprocessor.py`, online evaluation, `rescale_obb_to_original()`, `affine_obb()`, configs, checkpoints, or datasets.
- Write and run the rectangular failing test before changing the inference boundary.
- The workspace is not a git repository; do not commit.

---

### Task 1: Add the offline inference boundary regression

**Files:**
- Create: `test/test_tool_deimv2_obb_infer_hw_order.py`

**Interfaces:**
- Consumes: `infer_obb_and_export(..., imgsz: tuple, device: str)`.
- Produces: behavioral proof that the model wrapper receives `[[W, H]]`.

- [x] **Step 1: Create a lightweight recording model test harness**

The test imports `test/tool_deimv2_obb_infer.py`, replaces only expensive model/config/checkpoint seams, creates one temporary image, and records the `orig_target_sizes` argument passed by the real `infer_obb_and_export()` loop.

- [x] **Step 2: Add the rectangular scenario**

Test name:

```python
def test_infer_obb_passes_width_height_order_for_rectangular_imgsz(
    tmp_path, monkeypatch
) -> None:
```

Given `imgsz=(576, 1024)`, assert the captured tensor is `[[1024, 576]]`.

- [x] **Step 3: Run the rectangular test and capture RED**

Run:

```bash
python -m pytest test/test_tool_deimv2_obb_infer_hw_order.py::test_infer_obb_passes_width_height_order_for_rectangular_imgsz -v
```

Expected failure:

```text
assert [[576, 1024]] == [[1024, 576]]
```

- [x] **Step 4: Add and run the square regression**

Test name:

```python
def test_infer_obb_preserves_square_target_size(tmp_path, monkeypatch) -> None:
```

Given `imgsz=(640, 640)`, assert the captured tensor is `[[640, 640]]`.

Run:

```bash
python -m pytest test/test_tool_deimv2_obb_infer_hw_order.py::test_infer_obb_preserves_square_target_size -v
```

Expected: PASS before and after the fix.

---

### Task 2: Lock the PostProcessor dimension contract

**Files:**
- Modify: `test/test_deim_postprocessor.py`

**Interfaces:**
- Consumes: `PostProcessor.forward(outputs, orig_target_sizes)`.
- Produces: characterization proof for `[W, H, W, H, 1]` scaling.

- [x] **Step 1: Add the rectangular scaling characterization**

Test name:

```python
def test_obb_postprocessor_scales_with_width_height_factor_order() -> None:
```

Use a normalized box `[0.5, 0.5, 0.2, 0.1, pi/4]` and
`orig_target_sizes=[[1024, 576]]`. Assert the pixel box is
`[512, 288, 204.8, 57.6, pi/4]`.

- [x] **Step 2: Run the characterization**

```bash
python -m pytest test/test_deim_postprocessor.py::test_obb_postprocessor_scales_with_width_height_factor_order -v
```

Expected: PASS because `PostProcessor` is already correct.

---

### Task 3: Apply the minimal boundary fix

**Files:**
- Modify: `test/tool_deimv2_obb_infer.py:200`
- Modify: `test/deim_app/test_legacy_parity.py` only if its legacy helper mirrors the same boundary contract.

**Interfaces:**
- Consumes: `imgsz=(H, W)`.
- Produces: `dst_sz=[[W, H]]`.

- [x] **Step 1: Change the confirmed faulty call site**

Replace:

```python
dst_sz = torch.tensor([imgsz[0], imgsz[1]], device=device)[None, :]
```

with:

```python
dst_sz = torch.tensor([imgsz[1], imgsz[0]], device=device)[None, :]
```

- [x] **Step 2: Keep the legacy parity helper faithful**

If `_obb_legacy_tool_infer()` constructs the same tensor from `size=(H, W)`, swap it to `[size[1], size[0]]`. Do not alter unrelated parity logic.

- [x] **Step 3: Verify GREEN**

```bash
python -m pytest test/test_tool_deimv2_obb_infer_hw_order.py -v
```

Expected: both tests PASS.

- [x] **Step 4: Run focused regression coverage**

```bash
python -m pytest test/test_tool_deimv2_obb_infer_hw_order.py test/test_deim_postprocessor.py test/deim_app/test_legacy_parity.py -v
```

Expected: all collected tests PASS or existing fixture-dependent parity tests SKIP.

- [x] **Step 5: Run diagnostics on changed Python files**

Run LSP diagnostics for each changed file and the repository's available Python lint/type checks scoped to those files. Record unrelated pre-existing findings separately.

---

### Task 4: Real-surface rectangular inference QA

**Files:** none modified.

**Interfaces:**
- Consumes: real checkpoint, real config, one real image, `imgsz=(576, 1024)`.
- Produces: exported DOTA predictions from the corrected entry point.

- [x] **Step 1: Create temporary input and output directories**

Copy one validation image into a unique temporary image directory. Use a unique temporary output directory. Record both paths for cleanup.

- [x] **Step 2: Run `infer_obb_and_export()` with the real model**

Use:

```text
checkpoint: outputs/dlzdt_ablation/abl_rep0_last.pth
config: configs/custom_obb/dlzdt/sp_fz_common.yml
classes: /mnt/d/project_data/model_test/deimv2_obb_train_data/dlzdt_obb_val/classes.txt
imgsz: (576, 1024)
device: cuda:0
```

Expected: the image is processed and a DOTA prediction file is emitted.

- [x] **Step 3: Prove the corrected boundary on the real entry path**

Run the entry-point recording test against `(576, 1024)` and retain its
`[[1024, 576]]` assertion as the binary axis-order artifact. Separately inspect
the real DOTA file for finite numeric coordinates and valid 10-field rows.
Do not use simple image-bound checks as the sole proof because valid OBB
vertices may extend outside image bounds.

- [x] **Step 4: Run the square regression surface**

Run the same one-image inference with `imgsz=(640, 640)` and confirm it completes
without changing the square dimension contract.

- [x] **Step 5: Clean up all temporary resources**

Remove temporary image/output directories and confirm no inference process remains.

## Completion Gate

Phase 1 is complete only when:

1. The rectangular boundary test was observed RED for the exact swapped-axis value.
2. The production edit is one boundary conversion, with an optional faithful test mirror.
3. Rectangular and square boundary tests are GREEN.
4. PostProcessor characterization is GREEN.
5. Focused regressions pass or legitimately skip.
6. Real rectangular and square inference surfaces complete successfully.
7. Temporary resources are removed.
8. No Phase 2 geometry or evaluation-space code changed.
