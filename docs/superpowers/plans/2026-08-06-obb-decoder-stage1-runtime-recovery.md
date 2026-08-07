# OBB Decoder Stage 1 Runtime Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and finish the minimal runtime-recovery patch so rep0/1/3 training and the supported rep2 path execute without the two angle-channel shape failures.

**Architecture:** Keep the current angle-domain refactor intact, but constrain this stage to rank-preserving slices, the supported rep2 concat path, and tests that encode the new 5D-normalized/6D-physical attention boundary. Do not introduce the Stage 2 `cxcywh+offset` geometry migration here.

**Tech Stack:** Python 3, PyTorch, pytest, existing DEIMv2 OBB decoder and geometry utilities.

## Global Constraints

- Work from the current dirty tree; never revert unrelated user changes.
- Do not modify YAML configuration files.
- Do not modify rep2 anchors, denoising conversion, geometry helpers, or PostProcessor behavior.
- Preserve public OBB output `(cx,cy,w,h,theta_phys_rad)` with `theta_phys_rad in [0,pi)`.
- Preserve decoder angle channels with `[..., 4:]` or `[..., 4:5]`; never use `[..., 4]` when concatenating.
- `angle_rep=2, use_angle_first=True` remains rejected by the existing constructor guard.
- Do not commit until the reviewer approves the completed stage.

## Three-Gate Ownership

- **AI Test Gate**：AI 负责所有 pytest 编写、修正和运行。Task 1 的 MSDA 测试契约修正由 AI 立即执行；Task 3 的 rep2 eval smoke 也由 AI 编写。
- **User Implementation Gate**：用户只负责 Task 2 指定的生产代码，提交限定路径的生产 diff；不得修改测试。
- **AI Green/Review Gate**：AI 负责 Task 4 的完整验收，并在每个生产实现单元后运行定向测试、扩大回归和契约审核。

顺序固定为 `AI Test Gate -> User Implementation Gate -> AI Green/Review Gate`。Task 1 是测试契约修正，预期在当前候选生产修复上直接 PASS，因此不需要 User Implementation Gate。

---

## Current Working-Tree Baseline

The current disk already contains candidate fixes at both decoder sites:

```python
pre_bboxes = torch.concat(
    [
        pre_bboxes[..., :4],
        physical_rad_to_norm(pre_bboxes[..., 4:]),
    ],
    dim=-1,
)
```

and the encoder auxiliary site currently shows:

```python
norm_to_physical_rad(enc_topk_bboxes[..., 4:])
```

Therefore this plan verifies and tests those edits rather than blindly reapplying them. The current `postprocessor.py` comment deletion is outside Stage 1 and must not be included in the Stage 1 review package; do not revert or modify it as part of this stage.

### Task 1: Lock the MSDA angle-domain equivalence test

**Owner:** AI（AI Test Gate；立即执行，预期在当前磁盘候选修复上 PASS）

**Files:**
- Modify: `test/test_deimv2_obb_smoke.py:28-31,102-119`

**Interfaces:**
- Consumes: `external_rect_to_oriented_box(external_rect, vertex_offsets) -> Tensor[...,5]` returning physical radians.
- Consumes: `physical_rad_to_norm(theta_phys_rad) -> theta_norm`.
- Produces: a test proving 6D physical-angle attention and equivalent 5D normalized-angle attention use the same rotation.

- [ ] **Step 1: Update test imports**

Add the angle conversion import next to the existing geometry import:

```python
from engine.deim.obb_angle_contract import physical_rad_to_norm
from engine.deim.obb_geometry import external_rect_to_oriented_box
```

- [ ] **Step 2: Change the equivalent 5D reference construction**

Replace the direct physical-angle 5D attention input with an explicit normalized decoder reference:

```python
ref_6d_as_obb_phys = external_rect_to_oriented_box(
    ref_6d[..., :4], ref_6d[..., 4:]
)
ref_6d_as_obb_norm = torch.cat(
    [
        ref_6d_as_obb_phys[..., :4],
        physical_rad_to_norm(ref_6d_as_obb_phys[..., 4:]),
    ],
    dim=-1,
)
out_6d_as_5d = attn(query, ref_6d_as_obb_norm, value, spatial_shapes)
```

Keep the existing `torch.allclose(..., atol=1e-6)` assertion.

- [ ] **Step 3: Run only the equivalence test**

Run:

```bash
pytest -q test/test_deimv2_obb_smoke.py::test_msdeform_attn_decouple_angle_reference_consumes_theta
```

Expected: PASS. A failure means either the 6D branch still multiplies physical radians by `pi`, or the 5D branch is not receiving `theta_norm`.

- [ ] **Step 4: Submit Task 1 for review**

Provide the test diff and command output. Do not modify production code in this task.

### Task 2: Verify the two decoder shape fixes on disk

**Owner:** User（User Implementation Gate；仅在当前磁盘不满足指定代码时修改生产文件）

**Files:**
- Verify/Modify only if different: `engine/deim/deim_decoder.py:388-397,423-432,1112-1120`

**Interfaces:**
- Consumes: `external_rect_to_oriented_box(...) -> (cx,cy,w,h,theta_phys_rad)`.
- Produces: `(cx,cy,w,h,theta_norm)` with rank preserved.

- [ ] **Step 1: Verify both rep2 physical-to-normalized blocks are identical**

Both blocks must exactly use:

```python
pre_bboxes = torch.concat(
    [
        pre_bboxes[..., :4],
        physical_rad_to_norm(pre_bboxes[..., 4:]),
    ],
    dim=-1,
)
```

If the disk already matches, make no edit.

- [ ] **Step 2: Verify encoder auxiliary angle slicing**

The non-rep2 branch must exactly contain:

```python
enc_topk_bboxes = torch.cat(
    [
        enc_topk_bboxes[..., :4],
        norm_to_physical_rad(enc_topk_bboxes[..., 4:]),
    ],
    dim=-1,
)
```

If the disk already matches, make no edit.

- [ ] **Step 3: Inspect the Stage 1 production diff**

Run:

```bash
git diff -- engine/deim/deim_decoder.py engine/deim/dfine_decoder.py
```

Expected production behavior represented by the diff:

- rep2 geometry results are converted from physical radians back to normalized decoder references;
- `theta_scale[...,4] *= pi` applies after that normalization;
- final `out_refs` and `pre_bboxes` convert normalized angles to physical radians;
- MSDA 6D branch uses geometry-returned physical radians directly;
- no `[...,4]` rank-dropping concatenation remains.

- [ ] **Step 4: Submit Task 2 for review**

Provide the production diff only. The reviewer checks every changed angle boundary before tests continue.

### Task 3: Add a supported rep2 eval smoke test

**Owner:** AI（AI Test Gate；AI 编写并运行测试，用户不修改测试文件）

**Files:**
- Create: `test/test_deimv2_obb_rep2_eval.py`

**Interfaces:**
- Consumes: `DEIMTransformer(... angle_rep=2, use_angle_first=False)`.
- Produces: an eval assertion over public `pred_boxes` only.

- [ ] **Step 1: Write the eval test**

Create the following focused test in `test/test_deimv2_obb_rep2_eval.py`. Keep it separate because the legacy smoke module already exceeds 250 pure LOC:

```python
def test_angle_rep2_eval_forward_returns_public_physical_obb() -> None:
    torch.manual_seed(0)
    model = DEIMTransformer(
        num_classes=5,
        hidden_dim=32,
        num_queries=4,
        feat_channels=[32, 32],
        feat_strides=[4, 8],
        num_levels=2,
        num_points=2,
        nhead=4,
        num_layers=3,
        dim_feedforward=64,
        dropout=0.0,
        activation="relu",
        num_denoising=0,
        learn_query_content=False,
        eval_spatial_size=(16, 16),
        eval_idx=-1,
        eps=1e-2,
        aux_loss=False,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=4,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
        box_mode="obb",
        angle_rep=2,
        offset_scale_source="pre",
        use_angle_first=False,
    )
    model.eval()
    feats = [torch.randn(1, 32, 4, 4), torch.randn(1, 32, 2, 2)]

    with torch.no_grad():
        outputs = model(feats)

    pred_boxes = outputs["pred_boxes"]
    assert pred_boxes.shape[-1] == 5
    assert torch.isfinite(pred_boxes).all()
    assert (pred_boxes[..., 4] >= 0).all()
    assert (pred_boxes[..., 4] < math.pi).all()
```

The feature sizes deliberately match `eval_spatial_size=(16,16)` and strides `[4,8]`; do not use the training-only `[8x8,4x4]` feature sizes here.

- [ ] **Step 2: Run the new eval test**

Run:

```bash
pytest -q test/test_deimv2_obb_rep2_eval.py
```

Expected: PASS. It must not raise the former concat `TypeError`.

- [ ] **Step 3: Run the existing train matrix**

Run:

```bash
pytest -q test/test_deimv2_obb_smoke.py::test_angle_rep_forward_theta_in_proportional_domain
```

Expected: all four parameter combinations PASS, including `(angle_rep=2, use_angle_first=False)`.

- [ ] **Step 4: Submit Task 3 for review**

Provide the focused test-file diff and both outputs.

### Task 4: Isolate the Stage 1 review package and run the gate

**Owner:** AI（AI Green/Review Gate）

**Files:**
- Verify: all Stage 1 changed files

**Interfaces:**
- Produces: a Stage 1 diff containing decoder/attention changes and direct tests only.

- [ ] **Step 1: Exclude unrelated dirty-tree changes from the review package**

Run `git status --short` and record all pre-existing changes. Do not edit or revert `engine/deim/postprocessor.py`, untracked specs, logs, or any other file outside the Stage 1 file list. The Stage 1 review diff must be generated with explicit path arguments.

- [ ] **Step 2: Run focused tests**

Run:

```bash
pytest -q test/test_obb_angle_contract.py test/test_obb_adr_geometry.py test/test_obb_domain_audit.py
```

Expected: all tests PASS.

- [ ] **Step 3: Run the complete decoder smoke suite**

Run:

```bash
pytest -q test/test_deimv2_obb_smoke.py
```

Expected: all tests PASS; no stale MSDA equivalence failure, encoder dimension error, or rep2 concat error.

- [ ] **Step 4: Run syntax and diagnostics checks**

Run:

```bash
python -m py_compile engine/deim/deim_decoder.py engine/deim/dfine_decoder.py test/test_deimv2_obb_smoke.py test/test_deimv2_obb_rep2_eval.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 5: Submit the complete Stage 1 review package**

AI 汇总并提供：

- `git diff -- engine/deim/deim_decoder.py engine/deim/dfine_decoder.py test/test_deimv2_obb_smoke.py test/test_deimv2_obb_rep2_eval.py`;
- focused-test summary;
- full smoke summary;
- syntax/diff-check summary.

Do not start Stage 2 until the reviewer returns an explicit PASS for Stage 1.
