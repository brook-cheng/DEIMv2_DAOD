# OBB Decoder Stage 2 Rep2 Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make rep2 consistently use external-rectangle `cxcywh+offset` inside the decoder while preserving `xyxy+offset` geometry APIs and public 5D physical-radian OBB outputs.

**Architecture:** Add two composition helpers at the geometry boundary, then migrate decoder/MSDA/encoder-aux consumers, denoising, and anchors in separate review units. Finish with two independent existing-defect fixes: zero-noise denoising and non-focal PostProcessor behavior.

**Tech Stack:** Python 3, PyTorch, pytest, DEIMv2 decoder, existing ADR geometry functions.

## Global Constraints

- Stage 1 must have an explicit reviewer PASS before any task in this plan begins.
- Rep2 decoder 6D activated layout is `(cx_ext,cy_ext,w_ext,h_ext,epsilon,eta)` where the first four values describe the external rectangle in `cxcywh` format.
- Geometry APIs remain `OBB cxcywh+theta_phys -> external xyxy+offset` and `external xyxy+offset -> OBB cxcywh+theta_phys`.
- Public outputs remain 5D `(cx,cy,w,h,theta_phys_rad)` with `theta_phys_rad in [0,pi)`.
- Do not modify YAML configuration files.
- Do not enable `use_angle_first=True` for rep2.
- Do not use `clamp_offsets=True` as a training-path fix.
- Do not copy geometry formulas into decoder or attention modules.
- Do not commit until the reviewer approves each review unit.
- Every review unit follows `AI Test Gate -> User Implementation Gate -> AI Green/Review Gate`: AI writes and runs every pytest and records RED evidence; the user implements production code only; AI runs targeted and regression verification and submits the review result.

---

### Task 1: Add external-rectangle `cxcywh` composition helpers

**Ownership:** AI Test Gate writes Steps 1-2 tests and records RED; User Implementation Gate implements Step 3; AI Green/Review Gate runs Steps 4-5 and returns PASS/FAIL.

**Files:**
- Modify: `engine/deim/obb_geometry.py`
- Modify: `test/test_obb_adr_geometry.py`

**Interfaces:**
- Produces: `external_cxcywh_to_oriented_box(external_cxcywh: Tensor, vertex_offsets: Tensor, eps: float = 1e-9, clamp_offsets: bool = False) -> Tensor`.
- Produces: `oriented_box_to_external_cxcywh(obbs: Tensor) -> Tuple[Tensor, Tensor]`.
- Later tasks must consume these helpers instead of directly composing coordinate conversions.

- [ ] **Step 1: Write helper round-trip tests**

Add imports in `test/test_obb_adr_geometry.py`:

```python
from engine.deim.obb_geometry import (
    external_cxcywh_to_oriented_box,
    oriented_box_to_external_cxcywh,
)
```

Add tests using an unambiguous non-square OBB:

```python
def test_external_cxcywh_helpers_roundtrip_obb_geometry():
    obb = torch.tensor(
        [[[0.55, 0.45, 0.30, 0.12, math.pi / 4]]], dtype=torch.float32
    )
    external_cxcywh, offsets = oriented_box_to_external_cxcywh(obb)
    reconstructed = external_cxcywh_to_oriented_box(external_cxcywh, offsets)

    assert external_cxcywh.shape == (1, 1, 4)
    assert offsets.shape == (1, 1, 2)
    assert reconstructed.shape == (1, 1, 5)
    assert obb_vertex_error(obb, reconstructed) < ROUNDTRIP_TOL


def test_external_cxcywh_helper_matches_existing_xyxy_composition():
    obb = torch.tensor(
        [[[0.42, 0.58, 0.28, 0.10, math.pi / 6]]], dtype=torch.float32
    )
    ext_xyxy, offsets = oriented_box_to_external_rect(obb)
    ext_cxcywh = box_xyxy_to_cxcywh(ext_xyxy)

    via_helper = external_cxcywh_to_oriented_box(ext_cxcywh, offsets)
    via_primitives = external_rect_to_oriented_box(ext_xyxy, offsets)

    assert obb_vertex_error(via_helper, via_primitives) < ROUNDTRIP_TOL
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pytest -q test/test_obb_adr_geometry.py -k external_cxcywh_helper
```

Expected: collection/import failure because the two helpers do not exist.

- [ ] **Step 3: Implement the composition helpers**

In `engine/deim/obb_geometry.py`, import the existing conversions:

```python
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
```

Add the helpers immediately after `external_rect_to_oriented_box`:

```python
def external_cxcywh_to_oriented_box(
    external_cxcywh: Tensor,
    vertex_offsets: Tensor,
    eps: float = 1e-9,
    clamp_offsets: bool = False,
) -> Tensor:
    external_xyxy = box_cxcywh_to_xyxy(external_cxcywh)
    return external_rect_to_oriented_box(
        external_xyxy,
        vertex_offsets,
        eps=eps,
        clamp_offsets=clamp_offsets,
    )


def oriented_box_to_external_cxcywh(obbs: Tensor) -> Tuple[Tensor, Tensor]:
    external_xyxy, vertex_offsets = oriented_box_to_external_rect(obbs)
    return box_xyxy_to_cxcywh(external_xyxy), vertex_offsets
```

Do not add validation, clamping, or duplicated geometry formulas.

- [ ] **Step 4: Run helper and full geometry tests**

Run:

```bash
pytest -q test/test_obb_adr_geometry.py -k external_cxcywh_helper
pytest -q test/test_obb_adr_geometry.py
```

Expected: both commands PASS.

- [ ] **Step 5: Submit Review Unit 1**

Provide only the geometry helper/test diff and both test outputs.

### Task 2: Migrate decoder, MSDA, and encoder auxiliary boundaries

**Ownership:** AI Test Gate writes Steps 1-2 and Step 6 assertions and records RED; User Implementation Gate performs Steps 3-5; AI Green/Review Gate runs Steps 7-8 and returns PASS/FAIL.

**Files:**
- Modify: `engine/deim/deim_decoder.py:32,388-397,422-432,1318-1325`
- Modify: `engine/deim/dfine_decoder.py:26,169-178`
- Modify: `test/test_deimv2_obb_smoke.py:28-31,108-119,315-364`

**Interfaces:**
- Consumes: helpers created by Task 1.
- Produces: identical OBB decoding at layer-0, MSDA 6D reference handling, and encoder auxiliary output.

- [ ] **Step 1: Rewrite the MSDA equivalence fixture around valid rep2 input**

Replace random 6D ADR construction with a valid OBB encoded through the Task 1 helper:

```python
from engine.deim.obb_geometry import oriented_box_to_external_cxcywh

theta_phys = torch.rand(bs, n_queries, n_ref_levels, 1) * math.pi
obb_phys = torch.cat(
    [
        torch.rand(bs, n_queries, n_ref_levels, 2) * 0.4 + 0.3,
        torch.rand(bs, n_queries, n_ref_levels, 2) * 0.2 + 0.1,
        theta_phys,
    ],
    dim=-1,
)
external_cxcywh, offsets = oriented_box_to_external_cxcywh(obb_phys)
ref_6d = torch.cat([external_cxcywh, offsets], dim=-1)
out_6d = attn(query, ref_6d, value, spatial_shapes)
```

Replace the equivalence-conversion block (the Stage 1 version calls `external_rect_to_oriented_box`) with the new cxcywh helper so the 5D reference is normalized:

```python
ref_6d_as_obb_phys = external_cxcywh_to_oriented_box(
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

Retain the existing `torch.allclose(out_6d, out_6d_as_5d, atol=1e-6)` assertion.

- [ ] **Step 2: Run the MSDA test before production migration**

Run:

```bash
pytest -q test/test_deimv2_obb_smoke.py::test_msdeform_attn_decouple_angle_reference_consumes_theta
```

Expected: FAIL because the current 6D attention branch still interprets external `cxcywh` as `xyxy`.

- [ ] **Step 3: Replace direct decoder geometry calls**

Import `external_cxcywh_to_oriented_box` in `deim_decoder.py` and replace both layer-0 calls:

```python
pre_bboxes = external_cxcywh_to_oriented_box(
    ref_points_initial, dec_angle_initial
)
```

Keep the existing physical-to-normalized concat immediately after each call.

- [ ] **Step 4: Replace the MSDA 6D conversion**

In `dfine_decoder.py`, import `external_cxcywh_to_oriented_box` and replace:

```python
reference_points = external_cxcywh_to_oriented_box(
    reference_points[..., :4], reference_points[..., 4:]
)
angle = reference_points[..., 4:5]
```

The 5D branch must continue multiplying `theta_norm` by `pi`.

- [ ] **Step 5: Replace encoder auxiliary rep2 decode**

In `DEIMTransformer.forward`, replace the rep2 list conversion with:

```python
enc_topk_bboxes_list = [
    external_cxcywh_to_oriented_box(
        enc_topk_bboxes[..., :4],
        enc_topk_bboxes[..., 4:],
    )
    for enc_topk_bboxes in enc_topk_bboxes_list
]
```

- [ ] **Step 6: Add encoder auxiliary assertions to the rep2 train smoke test**

Inside `test_angle_rep_forward_theta_in_proportional_domain`, when `angle_rep == 2`, assert:

```python
assert outputs["enc_aux_outputs"][0]["pred_boxes"].shape[-1] == 5
assert torch.isfinite(outputs["enc_aux_outputs"][0]["pred_boxes"]).all()
```

- [ ] **Step 7: Run the review-unit tests**

Run:

```bash
pytest -q test/test_obb_adr_geometry.py -k external_cxcywh_helper
pytest -q test/test_deimv2_obb_smoke.py -k "msdeform_attn or angle_rep_forward or angle_rep2_eval"
```

Expected: PASS.

- [ ] **Step 8: Submit Review Unit 2**

Provide only decoder/MSDA/encoder-aux and direct test changes.

### Task 3: Convert rep2 denoising at the decoder boundary

**Ownership:** AI Test Gate writes Steps 1-2 and Step 5 tests and records RED; User Implementation Gate performs Steps 3-4; AI Green/Review Gate runs Steps 6-7 and returns PASS/FAIL.

**Files:**
- Modify: `engine/deim/deim_decoder.py:30-32,1029-1150`
- Modify: `test/test_deimv2_obb_smoke.py`

**Interfaces:**
- Consumes: `oriented_box_to_external_cxcywh(obbs)` from Task 1.
- Produces: `_obb_denoising_unact_to_rep2_unact(denoising_bbox_unact: Tensor) -> Tensor` with last dimension 6.

- [ ] **Step 1: Write a direct conversion test**

Import the private boundary helper and `norm_to_physical_rad`:

```python
from engine.deim.deim_decoder import _obb_denoising_unact_to_rep2_unact
from engine.deim.obb_angle_contract import norm_to_physical_rad
```

Add:

```python
def test_rep2_denoising_boundary_converts_logit_obb_to_logit_external_cxcywh():
    obb_norm = torch.tensor(
        [[[0.5, 0.5, 0.3, 0.1, 0.25]]], dtype=torch.float32
    )
    obb_unact = torch.logit(obb_norm.clamp(1e-4, 1 - 1e-4))

    rep2_unact = _obb_denoising_unact_to_rep2_unact(obb_unact)
    rep2_act = torch.sigmoid(rep2_unact)

    obb_phys = torch.cat(
        [obb_norm[..., :4], norm_to_physical_rad(obb_norm[..., 4:])],
        dim=-1,
    )
    expected_external_cxcywh, expected_offsets = oriented_box_to_external_cxcywh(
        obb_phys
    )
    expected = torch.cat([expected_external_cxcywh, expected_offsets], dim=-1)

    assert rep2_unact.shape[-1] == 6
    assert torch.allclose(rep2_act, expected, atol=1e-6)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q test/test_deimv2_obb_smoke.py::test_rep2_denoising_boundary_converts_logit_obb_to_logit_external_cxcywh
```

Expected: import failure because the helper does not exist.

- [ ] **Step 3: Implement the private decoder-boundary helper**

Add near the module imports in `deim_decoder.py`:

```python
def _obb_denoising_unact_to_rep2_unact(
    denoising_bbox_unact: torch.Tensor,
) -> torch.Tensor:
    denoising_bbox_act = torch.sigmoid(denoising_bbox_unact)
    denoising_obb_phys = torch.cat(
        [
            denoising_bbox_act[..., :4],
            norm_to_physical_rad(denoising_bbox_act[..., 4:]),
        ],
        dim=-1,
    )
    external_cxcywh, vertex_offsets = oriented_box_to_external_cxcywh(
        denoising_obb_phys
    )
    rep2_bbox_act = torch.cat([external_cxcywh, vertex_offsets], dim=-1)
    return inverse_sigmoid(rep2_bbox_act)
```

Use the existing `inverse_sigmoid` implementation already imported by the module. Do not clamp offsets here.

- [ ] **Step 4: Use the helper in `_get_decoder_input`**

Replace the current rep2 geometry block with:

```python
dn_bbox_unact = _obb_denoising_unact_to_rep2_unact(
    denoising_bbox_unact
)
enc_topk_bbox_unact = torch.concat(
    [dn_bbox_unact, enc_topk_bbox_unact], dim=1
)
```

- [ ] **Step 5: Add a rep2 denoising integration test**

Build the same small model used by the existing angle-rep test, but set `angle_rep=2`, `use_angle_first=False`, `num_denoising=4`, and provide:

```python
targets = [
    {
        "labels": torch.tensor([1], dtype=torch.int64),
        "boxes": torch.tensor(
            [[0.5, 0.5, 0.3, 0.1, math.pi / 4]], dtype=torch.float32
        ),
    }
]
```

Assert `pred_boxes`, `dn_outputs[-1]["pred_boxes"]`, and `dn_pre_outputs["pred_boxes"]` are finite 5D OBB tensors with angles in `[0,pi)`.

- [ ] **Step 6: Run denoising tests**

Run:

```bash
pytest -q test/test_deimv2_obb_smoke.py -k "rep2_denoising or denoising_theta"
```

Expected: PASS.

- [ ] **Step 7: Submit Review Unit 3**

Provide only the private conversion, `_get_decoder_input` call, and denoising tests.

### Task 4: Generate valid rep2 anchors from an explicit OBB

**Ownership:** AI Test Gate writes Steps 1-2 tests and records RED; User Implementation Gate performs Step 3; AI Green/Review Gate runs Steps 4-5 and returns PASS/FAIL.

**Files:**
- Modify: `engine/deim/deim_decoder.py:941-1027`
- Modify: `test/test_deimv2_obb_smoke.py`

**Interfaces:**
- Consumes: `oriented_box_to_external_cxcywh` from Task 1.
- Produces: activated rep2 anchors `(cx_ext,cy_ext,w_ext,h_ext,epsilon,eta)` before the existing logit transform.

- [ ] **Step 1: Write anchor invariant tests**

Add:

```python
def test_rep2_generated_anchors_are_valid_external_rect_offsets():
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
    anchors_unact, valid_mask = model._generate_anchors(
        [[4, 4], [2, 2]], device="cpu"
    )
    anchors = torch.sigmoid(anchors_unact)
    valid = valid_mask.squeeze(-1)
    valid_anchors = anchors[valid]

    assert valid_anchors.shape[-1] == 6
    assert valid_anchors.numel() > 0
    assert torch.isfinite(valid_anchors).all()
    assert (valid_anchors > model.eps).all()
    assert (valid_anchors < 1 - model.eps).all()
    assert (valid_anchors[..., 4] <= valid_anchors[..., 2]).all()
    assert (valid_anchors[..., 5] <= valid_anchors[..., 3]).all()
```

- [ ] **Step 2: Run the invariant test and verify RED**

Run:

```bash
pytest -q test/test_deimv2_obb_smoke.py::test_rep2_generated_anchors_are_valid_external_rect_offsets
```

Expected: FAIL because current `grid_offset_wh` is unrelated to external rectangle width and height.

- [ ] **Step 3: Replace rep2 anchor offset construction**

Inside `_generate_anchors`, retain `grid_xy` and `wh`, then generate a physical OBB at `pi/4`:

```python
theta = torch.full(
    (*grid_xy.shape[:-1], 1),
    torch.pi / 4,
    dtype=grid_xy.dtype,
    device=grid_xy.device,
)
initial_obb = torch.cat([grid_xy, wh, theta], dim=-1)
external_cxcywh, vertex_offsets = oriented_box_to_external_cxcywh(initial_obb)
lvl_anchors = torch.cat(
    [external_cxcywh, vertex_offsets], dim=-1
).reshape(-1, h * w, self._num_box_dof)
```

Delete `grid_offset_wh`; do not retain it as a fallback.

- [ ] **Step 4: Run anchor and forward tests**

Run:

```bash
pytest -q test/test_deimv2_obb_smoke.py -k "rep2_generated_anchors or angle_rep2_eval or angle_rep_forward or rep2_denoising"
```

Expected: PASS.

- [ ] **Step 5: Submit Review Unit 4**

Provide only anchor generation and anchor/forward tests.

### Task 5: Support `box_noise_scale=0`

**Ownership:** AI Test Gate writes Steps 1-3 tests and records RED; User Implementation Gate performs Step 4; AI Green/Review Gate runs Steps 5-6 and returns PASS/FAIL.

**Files:**
- Modify: `engine/deim/denoising.py:90-117`
- Modify: `test/test_deimv2_obb_smoke.py:375-417`

**Interfaces:**
- Produces: valid denoising logits for both zero and positive noise scales.

- [ ] **Step 1: Extend the denoising test helper**

Change `_run_denoising` to accept `box_noise_scale` and return the complete activated boxes when requested:

```python
def _run_denoising(
    gt_theta_rad,
    num_classes=5,
    hidden_dim=8,
    box_noise_scale=1.0,
):
    # existing target/embed setup
    _, dn_bbox_unact, _, _ = get_contrastive_denoising_training_group(
        targets=[target],
        num_classes=num_classes,
        num_queries=4,
        class_embed=class_embed,
        num_denoising=10,
        label_noise_ratio=0.0,
        box_noise_scale=box_noise_scale,
        box_mode="obb",
    )
    return torch.sigmoid(dn_bbox_unact)
```

Update existing callers to use `[...,4]` on the returned activated boxes.

- [ ] **Step 2: Add the zero-noise test**

```python
def test_denoising_box_noise_scale_zero_preserves_original_obb():
    boxes = _run_denoising(math.pi / 4, box_noise_scale=0.0)
    expected = torch.tensor([0.5, 0.5, 0.3, 0.2, 0.25])
    assert torch.allclose(
        boxes,
        expected.reshape(1, 1, 5).expand_as(boxes),
        atol=1e-4,
    )
```

- [ ] **Step 3: Run the test and verify RED**

Run:

```bash
pytest -q test/test_deimv2_obb_smoke.py::test_denoising_box_noise_scale_zero_preserves_original_obb
```

Expected: FAIL with the current `UnboundLocalError`.

- [ ] **Step 4: Refactor the noise branch without changing positive-noise behavior**

Implement this control flow in `denoising.py`:

```python
spatial_bbox = input_query_bbox[..., :4]
if box_noise_scale > 0:
    known_bbox = box_cxcywh_to_xyxy(spatial_bbox)
    diff = torch.tile(spatial_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
    rand_sign = torch.randint_like(spatial_bbox, 0, 2) * 2.0 - 1.0
    rand_part = torch.rand_like(spatial_bbox)
    rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (
        1 - negative_gt_mask
    )
    known_bbox += rand_sign * rand_part * diff
    known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
    noise_spatial = box_xyxy_to_cxcywh(known_bbox)
    noise_spatial[noise_spatial < 0] *= -1
else:
    noise_spatial = spatial_bbox

if box_mode == "hbb":
    input_query_bbox = noise_spatial
elif box_mode == "obb":
    theta_norm = physical_rad_to_norm(input_query_bbox[..., 4:])
    input_query_bbox = torch.cat([noise_spatial, theta_norm], dim=-1)

input_query_bbox_unact = inverse_sigmoid(input_query_bbox)
```

- [ ] **Step 5: Run all denoising tests**

Run:

```bash
pytest -q test/test_deimv2_obb_smoke.py -k denoising
```

Expected: PASS for zero noise and existing proportional-angle cases.

- [ ] **Step 6: Submit Review Unit 5**

Provide only `denoising.py` and denoising-test changes.

### Task 6: Fix the non-focal PostProcessor branch

**Ownership:** AI Test Gate writes Steps 1-2 tests and records RED; User Implementation Gate performs Step 3; AI Green/Review Gate runs Steps 4-5 and returns PASS/FAIL.

**Files:**
- Modify: `engine/deim/postprocessor.py:79-87`
- Create: `test/test_deim_postprocessor.py`

**Interfaces:**
- Produces: per-query class softmax over the last dimension and pixel-scaled selected boxes.

- [ ] **Step 1: Write the failing PostProcessor test**

Create `test/test_deim_postprocessor.py`:

```python
import math

import torch

from engine.deim.postprocessor import PostProcessor


def test_non_focal_obb_postprocessor_uses_class_dim_and_scaled_boxes():
    processor = PostProcessor(
        num_classes=2,
        use_focal_loss=False,
        num_top_queries=1,
        remap_mscoco_category=False,
        box_mode="obb",
    )
    logits = torch.tensor(
        [[[4.0, 0.0, -2.0], [0.0, 2.0, -2.0]]], dtype=torch.float32
    )
    pred_boxes = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.2, math.pi / 4],
          [0.25, 0.25, 0.1, 0.1, math.pi / 6]]],
        dtype=torch.float32,
    )
    orig_sizes = torch.tensor([[200.0, 100.0]])

    result = processor(
        {"pred_logits": logits, "pred_boxes": pred_boxes}, orig_sizes
    )[0]

    expected_score = torch.softmax(logits, dim=-1)[0, 0, 0]
    expected_box = torch.tensor([100.0, 50.0, 40.0, 20.0, math.pi / 4])
    assert result["labels"].tolist() == [0]
    assert torch.allclose(result["scores"], expected_score.reshape(1))
    assert torch.allclose(result["boxes"][0], expected_box, atol=1e-6)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q test/test_deim_postprocessor.py
```

Expected: FAIL because implicit softmax chooses the wrong normalization dimension and the branch gathers unscaled `boxes`.

- [ ] **Step 3: Implement the two-line behavior fix**

Change the non-focal branch to:

```python
scores = F.softmax(logits, dim=-1)[:, :, :-1]
scores, labels = scores.max(dim=-1)
if scores.shape[1] > self.num_top_queries:
    scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
    labels = torch.gather(labels, dim=1, index=index)
    boxes = torch.gather(
        bbox_pred,
        dim=1,
        index=index.unsqueeze(-1).tile(1, 1, bbox_pred.shape[-1]),
    )
else:
    boxes = bbox_pred
```

The explicit `else` is required so non-focal calls with `num_queries <= num_top_queries` also return scaled boxes.

- [ ] **Step 4: Run PostProcessor tests**

Run:

```bash
pytest -q test/test_deim_postprocessor.py
```

Expected: PASS.

- [ ] **Step 5: Submit Review Unit 6**

Provide only PostProcessor behavior and its new test file.

### Task 7: Run the complete Stage 2 integration gate

**Owner:** AI（AI Green/Review Gate；用户不修改测试或验收脚本）

**Files:**
- Verify all Stage 2 changed files.

**Interfaces:**
- Produces: evidence that all review units compose without rep0/1/3 regressions.

- [ ] **Step 1: Run geometry and angle contracts**

Run:

```bash
pytest -q test/test_obb_angle_contract.py test/test_obb_adr_geometry.py test/test_obb_domain_audit.py
```

Expected: PASS.

- [ ] **Step 2: Run decoder and denoising smoke tests**

Run:

```bash
pytest -q test/test_deimv2_obb_smoke.py
```

Expected: PASS for rep0/1/2/3 train/eval and rep2 denoising.

- [ ] **Step 3: Run PostProcessor tests**

Run:

```bash
pytest -q test/test_deim_postprocessor.py
```

Expected: PASS.

- [ ] **Step 4: Run syntax and diff checks**

Run:

```bash
python -m py_compile \
  engine/deim/obb_geometry.py \
  engine/deim/deim_decoder.py \
  engine/deim/dfine_decoder.py \
  engine/deim/denoising.py \
  engine/deim/postprocessor.py \
  test/test_obb_adr_geometry.py \
  test/test_deimv2_obb_smoke.py \
  test/test_deim_postprocessor.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 5: Submit the final Stage 2 review package**

Provide:

- diff grouped by the six review units;
- all test summaries;
- confirmation that YAML files were not changed;
- confirmation that `use_angle_first=True` remains rejected for rep2;
- any deviations from the approved helper signatures or data-flow ordering.

The reviewer performs a final contract audit over every production call to `external_rect_to_oriented_box` and `oriented_box_to_external_rect` before returning PASS.
