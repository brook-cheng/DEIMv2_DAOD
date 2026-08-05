# DEIMv2 OBB Angle Contract Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify DEIMv2 OBB angle semantics so every public 5D OBB uses physical radians in `[0, pi)`, network-only references use a shifted normalized encoding in `[0, 1)`, and all direct angle residuals use signed radians in `[-pi/2, pi/2)`.

**Architecture:** Add one centralized angle-contract module and two stateless OBB Codecs. Dataset, transforms, geometry, decoder outputs, criterion, matcher, postprocessor, eval, and export exchange only physical 5D OBBs. `angle_rep` remains decoder-local: rep0/rep2 use `ExternalRectOffsetCodec`, rep1/rep3 use `DirectAngleCodec`; decoder calls `decode`, criterion calls the equivalent configured Codec's `encode_fgl_target`.

**Tech Stack:** Python 3.12, PyTorch, pytest, project registry/YAML configuration, existing D-FINE FGL utilities.

## Global Constraints

- Public OBB angle: `theta_phys_rad in [0, pi)`.
- Private network angle: `theta_norm in [0, 1)`, using the shifted optimization interval `[-pi/4, 3*pi/4)`.
- Direct angle residual: `delta_theta_rad in [-pi/2, pi/2)`.
- Geometry functions accept physical radians only.
- `theta_norm` and `theta_logit` must not cross decoder/denoising optimization boundaries.
- Criterion, matcher, postprocessor, eval, and export must not branch on `angle_rep`.
- No raw `* pi`, `/ pi`, or `+/- 0.25` angle conversion outside the centralized angle-contract module and Codec implementation.
- HBB behavior must remain unchanged.
- Four OBB representations remain supported.
- Do not add tensor wrapper classes that interfere with training, TorchScript, or ONNX.
- Do not execute git operations unless the user explicitly requests them. Each task ends with a verification checkpoint instead of a commit step.

## File Structure

**Create**

- `engine/deim/obb_angle_contract.py`: the only public absolute-angle, normalized-angle, logit, and periodic-residual conversion functions.
- `engine/deim/obb_codecs.py`: `OBBCodec`, `ExternalRectOffsetCodec`, `DirectAngleCodec`, and Codec construction/config validation.
- `test/test_obb_angle_contract.py`: helper invariants and boundary coverage.
- `test/test_obb_codecs.py`: 5D/6D Codec identity, round-trip, and cross-representation geometry tests.
- `test/test_obb_public_contract.py`: dataset/transform/decoder/criterion/postprocessor public-boundary tests and source-policy scans.

**Modify**

- `engine/deim/__init__.py`: register/import Codec classes.
- `engine/deim/obb_geometry.py`: restore physical `[0, pi)` canonical output and use centralized physical canonicalization.
- `engine/data/dataset/dota_dataset.py`: assert/retain physical OBB output.
- `engine/data/transforms/obb_transforms.py`: keep every transform in the physical domain and update stale comments.
- `engine/data/transforms/mosaic.py`: preserve physical canonical output after polygon refit.
- `engine/deim/dfine_utils.py`: delegate OBB encode/decode semantics to the Codecs or expose dimension-neutral spatial primitives; remove public `obbox_rep_dim` branching from criterion-facing use.
- `engine/deim/deim_decoder.py`: inject/select Codec, keep explicit physical reference state per layer, isolate normalized/logit references, and output physical OBBs.
- `engine/deim/dfine_decoder.py`: make rotated attention consume an explicitly normalized reference and convert through the central contract before geometry.
- `engine/deim/denoising.py`: encode physical angle to shifted normalized logit through helpers.
- `engine/deim/deim_criterion.py`: inject Codec, remove `obbox_rep_dim`, and call `encode_fgl_target`.
- `engine/deim/matcher.py`: document/assert physical angle input; no representation branch.
- `engine/deim/postprocessor.py`: correct the angle contract comment and preserve angle bitwise.
- `engine/eval/obb_eval.py`, `engine/eval/dota_eval.py`: document/assert physical angle input and output.
- OBB YAML files under `configs/deimv2_obb/` and `configs/custom_obb/`: configure the correct Codec and remove `obbox_rep_dim`.
- Existing OBB tests: replace stale `[-pi/4, 3*pi/4)` public assertions and `[0, pi]` inclusive assertions with the approved half-open physical contract.
- `docs/superpowers/review/OBB_CODE_REVIEW.md`: update only lines 9, 303, 304 (stale `[0, pi]` / old denoising claims) to the new contract; retain shifted interval only as a private optimization encoding. No other docs are modified: historical docs stay as-is (they document their era's state).

---

### Task 1: Lock the Current Failures with Contract Tests

**Files:**
- Create: `test/test_obb_angle_contract.py`
- Create: `test/test_obb_codecs.py`
- Create: `test/test_obb_public_contract.py`
- Reference: `docs/superpowers/specs/2026-08-04-obb-angle-contract-unification-design.md`

**Interfaces:**
- Consumes: current production functions from `obb_geometry.py`, `dfine_utils.py`, `deim_decoder.py`, and `denoising.py`.
- Produces: failing tests that define the approved public and private contracts before production code changes.

- [ ] **Step 1: Add failing helper-contract tests**

Create `test/test_obb_angle_contract.py` with these exact cases, importing the not-yet-created helpers from `engine.deim.obb_angle_contract`:

```python
import math

import pytest
import torch

from engine.deim.obb_angle_contract import (
    apply_delta_rad,
    canonicalize_phys_rad,
    logit_to_physical_rad,
    norm_to_physical_rad,
    periodic_delta_rad,
    physical_rad_to_logit,
    physical_rad_to_norm,
)


def test_physical_norm_mapping_places_common_axes_away_from_seam():
    theta = torch.tensor([0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
    expected = torch.tensor([0.25, 0.5, 0.75, 0.0])
    assert torch.allclose(physical_rad_to_norm(theta), expected, atol=1e-6)


def test_norm_physical_mapping_is_periodically_inverse():
    theta = torch.tensor([0.0, 1e-6, math.pi / 4, math.pi / 2, math.pi - 1e-6])
    restored = norm_to_physical_rad(physical_rad_to_norm(theta))
    assert torch.allclose(periodic_delta_rad(restored, theta), torch.zeros_like(theta), atol=1e-6)


def test_periodic_delta_is_signed_shortest_radian_rotation():
    ref = torch.tensor([math.pi - 0.01, 0.01, 0.0])
    target = torch.tensor([0.01, math.pi - 0.01, math.pi / 2])
    expected = torch.tensor([0.02, -0.02, -math.pi / 2])
    assert torch.allclose(periodic_delta_rad(target, ref), expected, atol=1e-6)


def test_apply_delta_recovers_target_across_zero_pi_seam():
    ref = torch.tensor([math.pi - 0.01, 0.01])
    target = torch.tensor([0.01, math.pi - 0.01])
    restored = apply_delta_rad(ref, periodic_delta_rad(target, ref))
    assert torch.allclose(periodic_delta_rad(restored, target), torch.zeros(2), atol=1e-6)


def test_logit_round_trip_recovers_physical_angle():
    theta = torch.tensor([0.0, 0.3, math.pi / 2, math.pi - 1e-4])
    restored = logit_to_physical_rad(physical_rad_to_logit(theta))
    assert torch.allclose(periodic_delta_rad(restored, theta), torch.zeros_like(theta), atol=1e-5)


@pytest.mark.parametrize("value", [-4 * math.pi, -0.1, 0.0, math.pi, 5 * math.pi + 0.2])
def test_canonicalize_phys_rad_returns_half_open_physical_range(value):
    result = canonicalize_phys_rad(torch.tensor(value))
    assert 0.0 <= result.item() < math.pi
```

- [ ] **Step 2: Add failing Codec invariants**

Create `test/test_obb_codecs.py` with helper fixtures and tests that import:

```python
from engine.deim.obb_codecs import DirectAngleCodec, ExternalRectOffsetCodec
from engine.deim.obb_geometry import periodic_angle_distance, xywhr_to_xyxyxyxy
```

Required cases:

```python
REF = torch.tensor([[0.50, 0.50, 0.40, 0.20, 0.0]])
TARGET = torch.tensor([[0.55, 0.45, 0.35, 0.25, math.pi - 0.10]])
REG_SCALE = 4.0


@pytest.mark.parametrize("codec", [DirectAngleCodec(), ExternalRectOffsetCodec()])
def test_zero_continuous_residual_preserves_reference_geometry(codec):
    residual = torch.zeros((1, codec.residual_dim))
    decoded = codec.decode_continuous(REF, residual, REG_SCALE)
    assert torch.allclose(decoded[:, :4], REF[:, :4], atol=1e-6)
    assert torch.allclose(periodic_angle_distance(decoded[:, 4], REF[:, 4]), torch.zeros(1), atol=1e-6)


@pytest.mark.parametrize("codec", [DirectAngleCodec(), ExternalRectOffsetCodec()])
def test_encode_continuous_then_decode_recovers_target_geometry(codec):
    residual = codec.encode_continuous(REF, TARGET, REG_SCALE)
    decoded = codec.decode_continuous(REF, residual, REG_SCALE)
    ref_vertices = xywhr_to_xyxyxyxy(TARGET)
    decoded_vertices = xywhr_to_xyxyxyxy(decoded)
    assert _unordered_vertex_error(ref_vertices, decoded_vertices) < 1e-5


def test_direct_angle_codec_uses_radians_without_extra_pi():
    codec = DirectAngleCodec()
    residual = torch.zeros((1, 5))
    residual[:, 4] = 0.1 * REG_SCALE
    decoded = codec.decode_continuous(REF, residual, REG_SCALE)
    assert torch.allclose(decoded[:, 4], torch.tensor([0.1]), atol=1e-6)
```

Implement `_unordered_vertex_error` in the test by computing bidirectional minimum squared vertex distances, matching `test/test_obb_roundtrip.py`.

- [ ] **Step 3: Add failing public-boundary tests**

Create `test/test_obb_public_contract.py` with:

```python
def assert_physical_angles(boxes: torch.Tensor) -> None:
    assert boxes.shape[-1] == 5
    assert (boxes[..., 4] >= 0).all()
    assert (boxes[..., 4] < torch.pi).all()
```

Add tests for:

- `xyxyxyxy_to_xywhr` output on angles `-0.1`, `0`, `pi/2`, and `pi+0.1`;
- `external_rect_to_oriented_box` output;
- `OBBFlip`, `OBBResize`, and `affine_obb` output;
- `PostProcessor` preserving the input angle exactly;
- a source scan that fails if `distance[..., 4] *= torch.pi` remains in `deim_decoder.py`.

- [ ] **Step 4: Run the new tests and capture the expected failures**

Run:

```bash
python -m pytest \
  test/test_obb_angle_contract.py \
  test/test_obb_codecs.py \
  test/test_obb_public_contract.py -q
```

Expected: collection fails because `obb_angle_contract.py` and `obb_codecs.py` do not exist. After temporary import guards are avoided, current geometry range and decoder source-policy cases must also fail.

- [ ] **Step 5: Verification checkpoint**

Record the failing test names in the implementation session. Do not weaken these assertions during later tasks.

---

### Task 2: Implement the Central Angle Contract

**Files:**
- Create: `engine/deim/obb_angle_contract.py`
- Modify: `engine/deim/obb_geometry.py:18-39`
- Test: `test/test_obb_angle_contract.py`

**Interfaces:**
- Consumes: PyTorch tensor operations and the approved formulas.
- Produces:
  - `canonicalize_phys_rad(theta_rad: Tensor) -> Tensor`
  - `physical_rad_to_norm(theta_phys_rad: Tensor) -> Tensor`
  - `norm_to_physical_rad(theta_norm: Tensor) -> Tensor`
  - `physical_rad_to_logit(theta_phys_rad: Tensor, eps: float = 1e-4) -> Tensor`
  - `logit_to_physical_rad(theta_logit: Tensor) -> Tensor`
  - `periodic_delta_rad(target_phys_rad: Tensor, ref_phys_rad: Tensor) -> Tensor`
  - `apply_delta_rad(ref_phys_rad: Tensor, delta_theta_rad: Tensor) -> Tensor`

- [ ] **Step 1: Implement the exact helper formulas**

Create `engine/deim/obb_angle_contract.py`:

```python
import torch
from torch import Tensor


def canonicalize_phys_rad(theta_rad: Tensor) -> Tensor:
    return torch.remainder(theta_rad, torch.pi)


def physical_rad_to_norm(theta_phys_rad: Tensor) -> Tensor:
    theta_opt_rad = torch.remainder(theta_phys_rad + torch.pi / 4, torch.pi) - torch.pi / 4
    return (theta_opt_rad + torch.pi / 4) / torch.pi


def norm_to_physical_rad(theta_norm: Tensor) -> Tensor:
    theta_opt_rad = (theta_norm - 0.25) * torch.pi
    return canonicalize_phys_rad(theta_opt_rad)


def physical_rad_to_logit(theta_phys_rad: Tensor, eps: float = 1e-4) -> Tensor:
    theta_norm = physical_rad_to_norm(theta_phys_rad).clamp(min=eps, max=1.0 - eps)
    return torch.logit(theta_norm)


def logit_to_physical_rad(theta_logit: Tensor) -> Tensor:
    return norm_to_physical_rad(torch.sigmoid(theta_logit))


def periodic_delta_rad(target_phys_rad: Tensor, ref_phys_rad: Tensor) -> Tensor:
    return torch.remainder(target_phys_rad - ref_phys_rad + torch.pi / 2, torch.pi) - torch.pi / 2


def apply_delta_rad(ref_phys_rad: Tensor, delta_theta_rad: Tensor) -> Tensor:
    return canonicalize_phys_rad(ref_phys_rad + delta_theta_rad)
```

- [ ] **Step 2: Make the legacy distance helper delegate to the contract**

In `obb_geometry.py`, replace the signed and unsigned formula duplication with calls to `periodic_delta_rad`. Preserve the public signature of `periodic_angle_distance`:

```python
from .obb_angle_contract import periodic_delta_rad


def periodic_angle_distance(pred: Tensor, target: Tensor, with_signal=False) -> Tensor:
    signed = periodic_delta_rad(target, pred)
    return signed if with_signal else signed.abs()
```

- [ ] **Step 3: Run helper tests**

Run:

```bash
python -m pytest test/test_obb_angle_contract.py -q
```

Expected: all helper tests pass.

- [ ] **Step 4: Run legacy periodic-loss tests**

Run:

```bash
python -m pytest test/test_yolo_obb_loss.py test/test_matcher_obb_angle.py -q
```

Expected: pass without behavior changes.

- [ ] **Step 5: Verification checkpoint**

Run `lsp_diagnostics` on `obb_angle_contract.py` and `obb_geometry.py`; require zero errors.

---

### Task 3: Restore the Physical Geometry and Data Contract

**Files:**
- Modify: `engine/deim/obb_geometry.py:42-283`
- Modify: `engine/data/dataset/dota_dataset.py`
- Modify: `engine/data/transforms/obb_transforms.py`
- Modify: `engine/data/transforms/mosaic.py`
- Modify: `test/test_obb_roundtrip.py`
- Modify: `test/test_obb_transforms.py`
- Test: `test/test_obb_public_contract.py`

**Interfaces:**
- Consumes: `canonicalize_phys_rad` from Task 2.
- Produces: every polygon/refit/transform OBB angle in `[0, pi)`.

- [ ] **Step 1: Change polygon and external-rectangle reconstruction to physical canonicalization**

In both `xyxyxyxy_to_xywhr` and `external_rect_to_oriented_box`, replace:

```python
torch.remainder(theta + torch.pi / 4, torch.pi) - torch.pi / 4
```

with:

```python
canonicalize_phys_rad(theta)
```

Update docstrings to state `theta_phys_rad in [0, pi)`.

- [ ] **Step 2: Keep every transform in the public physical domain**

- `OBBFlip`: replace the inline modulo expression with `canonicalize_phys_rad(torch.pi - b[:, 4])`.
- `OBBZoomOut`, `OBBResize`, `OBBIoUCrop`, affine helpers, and mosaic polygon refit: ensure returned boxes are canonicalized by the geometry conversion, without norm/logit conversion.
- `OBBConvertBoxes`: continue normalizing only `cxcywh`; angle remains physical radians.
- Remove stale `[-pi/4, 3*pi/4)` comments from active data code.

- [ ] **Step 3: Update geometry tests to the public physical range**

In `test_obb_roundtrip.py`:

- replace `test_angle_range` with `[0, pi)` assertions;
- remove tests that describe `(x - 0.25) * pi` as a decoder public-output conversion;
- retain those formulas only in `test_obb_angle_contract.py` as private optimization encoding tests;
- retain vertex-set round-trip checks for `w<h` and square boxes.

- [ ] **Step 4: Strengthen transform range tests**

In `test_obb_transforms.py`, after each transform call, assert:

```python
assert (boxes[:, 4] >= 0).all()
assert (boxes[:, 4] < torch.pi).all()
```

Add an explicit flip case for physical angles `0.01` and `pi - 0.01`.

- [ ] **Step 5: Run geometry/data tests**

Run:

```bash
python -m pytest \
  test/test_obb_roundtrip.py \
  test/test_obb_transforms.py \
  test/test_obb_adr_geometry.py \
  test/test_obb_public_contract.py -q
```

Expected: geometry and transform cases pass; decoder-related public-contract cases may remain failing until Task 5.

- [ ] **Step 6: Verification checkpoint**

Require clean diagnostics on all modified data/geometry files.

---

### Task 4: Implement the Two OBB Codecs

**Files:**
- Create: `engine/deim/obb_codecs.py`
- Modify: `engine/deim/__init__.py`
- Modify: `engine/deim/dfine_utils.py:194-362`
- Test: `test/test_obb_codecs.py`

**Interfaces:**
- Consumes: angle-contract helpers, current spatial `distance2bbox`, `translate_gt`, external-rectangle geometry.
- Produces:

```python
class OBBCodec(nn.Module):
    residual_dim: int
    def decode_continuous(self, ref_phys_obb, residual, reg_scale): ...
    def encode_continuous(self, ref_phys_obb, target_phys_obb, reg_scale): ...
    def encode_fgl_target(self, ref_phys_obb, target_phys_obb, reg_max, reg_scale, up): ...

class ExternalRectOffsetCodec(OBBCodec):  # residual_dim = 6
class DirectAngleCodec(OBBCodec):        # residual_dim = 5
```

- [ ] **Step 1: Write the abstract stateless interface and registry classes**

Use `@register()` for both concrete Codecs. They have no trainable parameters and may subclass `nn.Module` for registry/config compatibility.

`OBBCodec.encode_fgl_target` must call `translate_gt` on `encode_continuous(...).reshape(-1)`, return the same `(indices, weight_right, weight_left)` tuple currently consumed by `loss_local`, and clamp indices to `reg_max - 0.1` exactly once.

- [ ] **Step 2: Implement DirectAngleCodec**

- Spatial residual encoding/decoding must preserve current D-FINE `alpha,beta,gamma,delta` formulas.
- Angle encode:

```python
angle_residual = periodic_delta_rad(target[..., 4:5], ref[..., 4:5]) * reg_scale
```

- Angle decode:

```python
theta = apply_delta_rad(ref[..., 4:5], residual[..., 4:5] / reg_scale)
```

- No multiplication by pi is allowed.

- [ ] **Step 3: Implement ExternalRectOffsetCodec**

- Convert both reference and target physical OBBs with `oriented_box_to_external_rect`.
- Preserve the current `pre`/`post` offset scale semantics as a constructor argument:

```python
ExternalRectOffsetCodec(offset_scale_source: str = "pre")
```

- Encode and decode must use the same scale source.
- Zero continuous residual must reconstruct the same reference geometry.

- [ ] **Step 4: Retire dimension-driven public branching in dfine_utils**

Keep `distance2bbox_obb` and `bbox2distance_obb` as thin wrappers ONLY because `test/test_obb_adr_geometry.py` (lines 363-558) calls them directly with the existing `offset_scale_source` keyword. Each wrapper constructs the matching Codec internally and delegates:

```python
def distance2bbox_obb(ref, dist, reg_scale, offset_scale_source="pre"):
    codec = ExternalRectOffsetCodec(offset_scale_source=offset_scale_source)
    return codec.decode_continuous(ref, dist, reg_scale)
```

`dist` is the existing Integral continuous residual and is passed unchanged. The
Codec owns the single `/ reg_scale` conversion; dividing in the wrapper would
apply the scale twice.

Selection is by the explicit `offset_scale_source` argument only. Do not select by tensor last-dimension or `obbox_rep_dim`; do not add a `codec` parameter (would break the existing ADR tests). New production code must call the Codec directly, never these wrappers.

Mark wrappers for later removal in documentation, not with runtime warnings that would pollute training logs. `test_obb_adr_geometry.py` keeps passing unchanged.

- [ ] **Step 5: Register Codec imports**

Add to `engine/deim/__init__.py`:

```python
from .obb_codecs import DirectAngleCodec, ExternalRectOffsetCodec
```

- [ ] **Step 6: Run Codec tests**

Run:

```bash
python -m pytest test/test_obb_codecs.py test/test_obb_adr_geometry.py -q
```

Expected: all zero-residual, nonzero-radian-residual, encode/decode, and cross-Codec target-reconstruction tests pass.

- [ ] **Step 7: Verification checkpoint**

Require clean diagnostics on `obb_codecs.py`, `dfine_utils.py`, and `engine/deim/__init__.py`.

---

### Task 5: Refactor Decoder State and Attention Boundaries

**Files:**
- Modify: `engine/deim/deim_decoder.py`
- Modify: `engine/deim/dfine_decoder.py`
- Modify: `test/test_deimv2_obb_smoke.py`
- Test: `test/test_obb_public_contract.py`
- Test: `test/test_obb_codecs.py`

**Interfaces:**
- Consumes: concrete `OBBCodec`, `physical_rad_to_norm`, `norm_to_physical_rad`.
- Produces: physical `pred_boxes`, `ref_points`, `pre_boxes`, auxiliary boxes, and denoising boxes for every rep.

- [ ] **Step 1: Inject a Codec into DEIMTransformer and TransformerDecoder**

Add `obb_codec=None` to both constructors. Add `"obb_codec"` to `DEIMTransformer.__inject__`. For `box_mode="obb"`, reject `None`. Validate:

```python
expected_dim = {0: 6, 1: 5, 2: 6, 3: 5}[angle_rep]
if obb_codec.residual_dim != expected_dim:
    raise ValueError(...)
```

Pass the Codec object to `TransformerDecoder`; remove independent residual-semantic branching based on `angle_rep` where the Codec owns the behavior.

- [ ] **Step 2: Separate physical state from network reference encoding**

Inside each decoder layer, maintain:

```text
ref_phys_obb     # public physical geometry, [0, pi)
ref_theta_norm   # temporary attention/head encoding, [0, 1)
```

Initial anchor/DN logits still enter through sigmoid, but convert the angle immediately:

```python
ref_theta_norm = torch.sigmoid(ref_angle_logit)
ref_theta_phys = norm_to_physical_rad(ref_theta_norm)
ref_phys_obb = torch.cat([ref_xywh, ref_theta_phys], dim=-1)
```

Do not retain `theta_code_rad = theta_norm * pi` as a physical reference.

- [ ] **Step 3: Route attention through explicit normalized references**

Build `ref_points_input` from `cxcywh` plus `physical_rad_to_norm(ref_phys_obb[..., 4:5])` only where rotated attention requires a fifth normalized dimension.

In `dfine_decoder.py`, rename local variables to `theta_norm` and convert through `norm_to_physical_rad` before `cos`/`sin` or rotated sampling geometry. Remove direct `reference_points[..., 4:5] * torch.pi`.

- [ ] **Step 4: Route all residual decoding through the Codec**

Replace the shared `theta_scale`, rep1 `distance[..., 4] *= torch.pi`, and 5D/6D shape branch with:

```python
continuous_residual = integral(pred_corners, project)
inter_ref_phys_obb = self.obb_codec.decode_continuous(
    ref_phys_obb,
    continuous_residual,
    reg_scale,
)
```

The Codec output is the next layer's physical state.

- [ ] **Step 5: Make every public decoder output physical**

Remove the final unconditional `(angle - 0.25) * pi` block. Stack/return physical boxes directly for:

- main output;
- auxiliary layers;
- encoder top-k output;
- pre-output;
- denoising output;
- `ref_points`.

Encoder top-k and anchor angles must be converted from private normalized encoding to physical radians exactly once before becoming public outputs.

- [ ] **Step 6: Preserve head architecture differences only**

Keep rep0/rep1/rep2/rep3 head dimensions, angle-first sequencing, and feature fusion behavior. Restrict `angle_rep` conditionals to head construction, private feature flow, and Codec compatibility checks.

- [ ] **Step 7: Replace stale smoke assertions**

In `test_deimv2_obb_smoke.py`:

- parametrize `angle_rep` over `0,1,2,3`;
- inject the matching Codec;
- assert physical `[0, pi)` output;
- assert nonzero direct-angle residual `0.1 rad` remains `0.1 rad` for rep1 and rep3;
- assert zero continuous residual preserves the physical reference for all reps.

- [ ] **Step 8: Run decoder tests**

Run:

```bash
python -m pytest \
  test/test_deimv2_obb_smoke.py \
  test/test_obb_codecs.py \
  test/test_obb_public_contract.py -q
```

Expected: all four rep forward paths pass and source scan confirms removal of the rep1 `* pi` line.

- [ ] **Step 9: Manual forward/backward gate**

Run the existing small CPU smoke construction for reps 0-3 in training mode; sum finite `pred_boxes`, `pred_logits`, and `pred_corners`, call backward, and require finite gradients.

- [ ] **Step 10: Verification checkpoint**

Require clean diagnostics on both decoder files and updated tests.

---

### Task 6: Migrate Denoising to Explicit Physical/Normalized Conversion

**Files:**
- Modify: `engine/deim/denoising.py`
- Modify: `engine/deim/deim_decoder.py` denoising input handling
- Test: `test/test_obb_public_contract.py`
- Modify: `test/test_obb_roundtrip.py` denoising tests

**Interfaces:**
- Consumes: physical targets and angle-contract helpers.
- Produces: denoising angle logits only; geometry consumers receive decoded physical angles.

- [ ] **Step 1: Add failing denoising round-trip tests**

Test physical angles `0`, `0.01`, `pi/2`, and `pi-0.01` through:

```text
physical_rad_to_norm -> inverse_sigmoid -> sigmoid -> norm_to_physical_rad
```

Assert periodic recovery.

- [ ] **Step 2: Replace inline denoising arithmetic**

In `get_contrastive_denoising_training_group`, replace `(theta + pi/4) / pi` with `physical_rad_to_norm(theta)` before `inverse_sigmoid`.

- [ ] **Step 3: Remove rep2 logit-to-geometry misuse**

In decoder denoising input handling, never pass `denoising_bbox_unact` to `oriented_box_to_external_rect`. Convert spatial logits with sigmoid and angle logit with `logit_to_physical_rad`, construct a physical OBB, then call the configured Codec/geometry.

- [ ] **Step 4: Run denoising and decoder tests**

Run:

```bash
python -m pytest test/test_obb_roundtrip.py test/test_deimv2_obb_smoke.py test/test_obb_public_contract.py -q
```

Expected: all denoising round trips and rep2 denoising forward paths pass.

- [ ] **Step 5: Verification checkpoint**

Require clean diagnostics and a source scan proving no geometry call receives a variable named `*_logit` or `*_unact`.

---

### Task 7: Inject the Codec into Criterion and Migrate FGL Targets

**Files:**
- Modify: `engine/deim/deim_criterion.py`
- Modify: `engine/deim/matcher.py`
- Modify: `test/test_deim_criterion_obb_loss.py`
- Modify: `test/test_obb_adr_loss.py`
- Modify: `test/test_obb_loss_integration.py`
- Modify: OBB YAML configs under `configs/deimv2_obb/` and `configs/custom_obb/`

**Interfaces:**
- Consumes: `OBBCodec.encode_fgl_target`, public physical OBBs.
- Produces: representation-neutral criterion/matcher behavior and a deterministic Codec configuration rule.

- [ ] **Step 1: Inject `obb_codec` into DEIMCriterion**

Add `"obb_codec"` to `DEIMCriterion.__inject__` and add constructor argument `obb_codec=None`. For OBB mode, reject `None`. Remove `obbox_rep_dim`, `self.obbox_rep_dim`, and `self.num_reg_dist` derivation from representation shape; use `obb_codec.residual_dim`.

- [ ] **Step 2: Route FGL targets through the Codec**

Replace both denoising and normal calls to `bbox2distance_obb` with:

```python
self.obb_codec.encode_fgl_target(
    ref_points,
    target_boxes,
    self.reg_max,
    outputs["reg_scale"],
    outputs["up"],
)
```

Both `ref_points` and `target_boxes` must already be public physical OBBs.

- [ ] **Step 3: Keep matcher representation-neutral**

Do not add Codec or `angle_rep` to matcher. Update docs/debug checks to state that `out_bbox` and `tgt_bbox` are physical `[0, pi)` OBBs. Periodic costs continue to use radians.

- [ ] **Step 4: Define the YAML configuration shape**

Each resolved OBB config must contain one named Codec node and inject that name into both components:

```yaml
obb_codec:
  type: ExternalRectOffsetCodec  # rep0/rep2
  offset_scale_source: pre

DEIMTransformer:
  obb_codec: obb_codec

DEIMCriterion:
  obb_codec: obb_codec
```

For rep1/rep3:

```yaml
obb_codec:
  type: DirectAngleCodec
```

NOTE (Task 7 implementation correction): the shared-node `obb_codec: {type: ...}` + string-reference shape above does NOT resolve in this codebase — the inject path requires the referenced node to be a registered class-name node carrying `_name` (see `engine/core/workspace.py`, inject path 1). Use the registered class-name top-level node + string reference instead, e.g. `ExternalRectOffsetCodec:` / `DirectAngleCodec:` nodes referenced as `obb_codec: ExternalRectOffsetCodec` by both `DEIMTransformer` and `DEIMCriterion`. This is also deep-merge-safe: a 5D leaf's `DirectAngleCodec: {}` cannot inherit a 6D base's `offset_scale_source`.

Before mass migration, add a YAMLConfig construction test proving that both injected objects have the same concrete class and constructor settings. Identity equality is not required because the Codecs are stateless.

- [ ] **Step 5: Migrate all active OBB configs deterministically**

For every OBB YAML under `configs/deimv2_obb/` and `configs/custom_obb/`:

- rep0 or rep2 → `ExternalRectOffsetCodec`;
- rep1 or rep3 → `DirectAngleCodec`;
- remove `DEIMCriterion.obbox_rep_dim`;
- preserve `offset_scale_source` by moving it to the 6D Codec node;
- add matching `obb_codec` references to decoder and criterion;
- do not change weights, loss switches, output directories, or training hyperparameters.

Use a config test to enumerate resolved configs and assert the mapping instead of relying on a manual file count. Exclude `configs/custom_obb/synthetic_configs/provenance/` archive files (`.completed.yml` and `completed_runs.yml`) from migration and from the config test — they are historical run records, not active configs.

- [ ] **Step 6: Update direct criterion fixtures**

Every direct `DEIMCriterion(...)` construction in tests must pass the Codec dictated by the rep→Codec mapping from Step 5 (rep0/rep2 → `ExternalRectOffsetCodec`, rep1/rep3 → `DirectAngleCodec`). Keep ADR loss tests on physical OBBs; ADR geometry is independent of the private normalized encoding.

- [ ] **Step 7: Run criterion/config tests**

Run:

```bash
python -m pytest \
  test/test_deim_criterion_obb_loss.py \
  test/test_obb_adr_loss.py \
  test/test_obb_loss_integration.py \
  test/test_obb_loss_experiment_configs.py \
  test/test_matcher_obb_angle.py -q
```

Expected: criterion constructs for every config, FGL dimensions come from Codec, matcher remains unchanged, and all losses are finite.

- [ ] **Step 8: Verification checkpoint**

Source scans must show no `obbox_rep_dim` in `deim_criterion.py` or active OBB YAML files, and no `angle_rep` in criterion/matcher/postprocessor/eval.

---

### Task 8: Lock Postprocessor, Eval, Export, and Documentation Contracts

**Files:**
- Modify: `engine/deim/postprocessor.py`
- Modify: `engine/eval/obb_eval.py`
- Modify: `engine/eval/dota_eval.py`
- Modify: `docs/superpowers/review/OBB_CODE_REVIEW.md` lines 9, 303, 304
- Modify: `test/test_obb_eval.py`
- Modify: `test/test_obb_public_contract.py`
- Verify (no change expected): `tools/deployment/export_onnx.py`, `tools/deployment/export_yolo_w_nms.py`, `tools/inference/onnx_inf.py` — audited as generic exporters with no OBB angle documentation; do not add any unless a source scan proves otherwise.

**Interfaces:**
- Consumes: public physical OBBs.
- Produces: unchanged physical angle through postprocessing, evaluation, and export.

- [ ] **Step 1: Correct downstream contracts without adding representation branches**

- Postprocessor factor remains `[W, H, W, H, 1]`.
- Update its comment to `theta_phys_rad in [0, pi)`.
- Eval and DOTA conversion code must canonicalize only at explicit polygon-to-OBB boundaries.
- `tools/deployment/export_onnx.py` and `tools/deployment/export_yolo_w_nms.py`: confirmed generic exporters with no OBB angle claims; leave unchanged (do not add angle documentation).

- [ ] **Step 2: Add bitwise postprocessor test**

Use input angles including `0`, `pi/2`, and `pi-1e-6`; assert output angle tensor is `torch.equal` to the input angle tensor after selection/scaling.

- [ ] **Step 3: Add representation-independence source checks**

Scan these files for `angle_rep` and require no matches:

```text
engine/deim/deim_criterion.py
engine/deim/matcher.py
engine/deim/postprocessor.py
engine/eval/obb_eval.py
engine/eval/dota_eval.py
```

- [ ] **Step 4: Update documentation**

In `docs/superpowers/review/OBB_CODE_REVIEW.md`, update exactly:
- line 9: keep the public physical contract phrasing (`theta in [0, pi)` half-open) and remove any suggestion that the decoder-internal `[0,1]` is public;
- line 303: replace "输出/criterion/matcher/postprocessor 统一转 [0,π]" with the new contract: decoder output, criterion, matcher, postprocessor all exchange physical `[0, pi)`; decoder-internal `[0,1]` is a private normalized encoding only;
- line 304: replace "θ/π 归一化后再 inverse_sigmoid" with "`physical_rad_to_norm` 后 inverse_sigmoid（shifted seam）".

State clearly:

- public physical angle is `[0, pi)`;
- shifted `[-pi/4, 3*pi/4)` exists only as the private optimization seam;
- direct residual is signed radians;
- geometry never consumes norm/logit.

- [ ] **Step 5: Run downstream tests**

Run:

```bash
python -m pytest test/test_obb_eval.py test/test_obb_public_contract.py -q
```

Expected: pass with no downstream `angle_rep` dependency.

- [ ] **Step 6: Verification checkpoint**

Require clean diagnostics and no stale active comments claiming public `[-pi/4, 3*pi/4)` output.

---

### Task 9: Full Regression and Runtime QA

**Files:**
- Verify: all files changed in Tasks 1-8
- Update only tests/docs needed to correct failures caused by this migration

**Interfaces:**
- Consumes: completed unified angle contract.
- Produces: evidence that all four reps train, decode, postprocess, and evaluate under one public contract.

- [ ] **Step 1: Run focused OBB tests**

Run:

```bash
python -m pytest \
  test/test_obb_angle_contract.py \
  test/test_obb_codecs.py \
  test/test_obb_public_contract.py \
  test/test_obb_roundtrip.py \
  test/test_obb_transforms.py \
  test/test_obb_adr_geometry.py \
  test/test_obb_adr_loss.py \
  test/test_deim_criterion_obb_loss.py \
  test/test_matcher_obb_angle.py \
  test/test_yolo_obb_loss.py \
  test/test_obb_loss_integration.py \
  test/test_deimv2_obb_smoke.py \
  test/test_obb_eval.py -q
```

Expected: all pass.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
python -m pytest -q
```

Expected: pass, or report only pre-existing unrelated failures with exact names and evidence.

- [ ] **Step 3: Run four-rep config construction**

Resolve and construct representative configs:

```text
configs/custom_obb/dlzdt/sp_fz_rep0_nloss.yml
configs/custom_obb/dlzdt/sp_fz_rep1_nloss.yml
configs/custom_obb/dlzdt/sp_fz_rep2_nloss.yml
configs/custom_obb/dlzdt/sp_fz_rep3_nloss.yml
```

Assert decoder/criterion Codec classes are correct for each rep.

- [ ] **Step 4: Run four-rep forward/backward smoke**

For each representative config or equivalent small CPU model:

- training forward with no denoising;
- training forward with denoising;
- auxiliary output shape/physical range checks;
- scalar loss backward;
- finite gradient checks;
- eval forward and postprocessor;
- physical angle `[0, pi)` checks at every public output.

- [ ] **Step 5: Run source-policy scans**

Require:

- no `distance[..., 4] *= torch.pi`;
- no geometry call with logit/unactivated angle input;
- no public `(theta - 0.25) * pi` conversion;
- angle conversion arithmetic outside `obb_angle_contract.py` and `obb_codecs.py` is either absent or explicitly justified as non-angle arithmetic;
- no `angle_rep` branches outside decoder/head construction and config validation.

- [ ] **Step 6: Review the original audit findings**

Re-run numeric probes for:

- rep0 zero residual: no fixed `+pi/4` rotation;
- rep1 `0.1 rad` residual: decoded increment is `0.1`, not `0.314159`;
- rep2 reference: no extra `* pi` before geometry;
- rep2 denoising: no logit passed into geometry;
- rep3 direct-angle round trip remains exact.

- [ ] **Step 7: Final diagnostics and completion report**

Run diagnostics on every changed Python file. Report:

- changed modules;
- focused/full test results;
- four-rep runtime evidence;
- source-policy scan results;
- any unrelated pre-existing failure.

Do not claim completion without command output from this execution session.
