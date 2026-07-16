# DEIMv2-OBB Loss Refinement Design

Date: 2026-07-16

## 1. Purpose

Refine DEIMv2-OBB regression so Hungarian-positive predictions receive useful center and scale gradients even when they do not overlap their ground-truth boxes, while aligning the criterion and both Hungarian matcher phases with OBB representation equivalence.

The recommended criterion combines:

- center and canonical side-length L1;
- ProbIoU;
- periodic angle loss;
- KLD.

The matcher uses the same center and canonical side-length L1 before and after `matcher_change_epoch`, keeps ProbIoU and angle geometry, and retains Chamfer in the early additive phase.

## 2. Scope and Constraints

### 2.1 In scope

- OBB-only box losses in `engine/deim/deim_criterion.py`.
- OBB-only costs in `engine/deim/matcher.py`.
- A focused OBB loss utility module.
- Explicit OBB configuration migration, including Kendall-managed loss names.
- Unit, gradient, matcher, configuration, Kendall, and integration tests.
- A controlled convergence comparison between the legacy, ProbIoU-only-geometry, and recommended loss combinations.

### 2.2 Out of scope

- HBB criterion or matcher behavior.
- ADR/FGL/DDF representation and decoder changes.
- MAL, CDN, encoder auxiliary outputs, postprocessing, evaluation, or dataset changes.
- Replacing Hungarian assignment with TaskAlignedAssigner.
- Adding Chamfer distance as a differentiable criterion loss.
- Claiming faster convergence or higher mAP before controlled experiments provide evidence.

## 3. Current Problems

1. The current OBB `loss_bbox` applies L1 directly to `(cx, cy, w, h)` and adds a parameter-space angle term. Equivalent `(w, h, theta)` and `(h, w, theta + pi/2)` representations can therefore receive different penalties.
2. The criterion and matcher do not use the same primary geometry throughout training.
3. `1 - ProbIoU` remains differentiable for moderately separated boxes but approaches its upper bound for distant boxes. Its center gradient can become negligibly small and eventually zero at the implementation's upper Bhattacharyya-distance clamp.
4. The periodic angle loss has no center gradient and cannot pull a distant prediction toward its GT.
5. The late matching-aware branch currently discards all L1 information, so distant candidate queries can be poorly separated when their class and ProbIoU qualities are similar.
6. Kendall configuration must explicitly include every produced OBB loss family.

## 4. Final Decisions

| Area | Decision |
|---|---|
| Criterion L1 | Keep `loss_bbox`, but redefine it as center L1 plus canonical short/long-side L1. Remove angle from this key. |
| Primary box loss | Add equal-positive `loss_probiou = sum(1 - ProbIoU) / normalizer`. |
| Angle loss | Add equal-positive `loss_angle = sum(scale_weight * sin²(2Δtheta)) / normalizer`. |
| KLD | Retain `loss_kld`; it complements L1 and ProbIoU with distribution geometry. |
| Positive weighting | Do not multiply the four geometry losses by detached current ProbIoU or `boxes_weight`. |
| Early matcher L1 | Redefine `cost_bbox` as pairwise center L1 plus pairwise canonical side L1; baseline weight 2. |
| Early ProbIoU | Use `cost_probiou=4`. |
| Early angle | Add `cost_angle=3` using the criterion's periodic angle geometry. |
| Late matcher L1 | Add the same `cost_bbox` additively with independent `late_cost_bbox=0.25`. |
| Late angle | Multiply matching quality by angle quality. |
| Chamfer | Keep matcher-only in the early branch, weight 5. |
| Compatibility | New switches default off; unmigrated OBB configs retain the legacy OBB branch. No silent key conversion. |

## 5. OBB Representation and Canonical Side L1

DEIMv2 decodes OBBs with the longer-edge-as-`w` convention in its geometry path, but loss and matcher code must not assume every input is canonical. External annotations, transforms, and near-square boxes can still expose equivalent `w/h` parameterizations.

For every prediction and target, define:

```python
pred_sides = torch.sort(pred_boxes[..., 2:4], dim=-1).values
target_sides = torch.sort(target_boxes[..., 2:4], dim=-1).values
```

The two columns are `[short_side, long_side]`. The matched-pair criterion loss is:

```text
loss_bbox_i =
    |pred_cx - target_cx|
  + |pred_cy - target_cy|
  + |pred_short - target_short|
  + |pred_long - target_long|
```

This loss is invariant to swapping `w` and `h`. It contains no angle term. Near `w == h`, `sort` can switch branches, but PyTorch supplies a valid subgradient and both sides are geometrically close.

The loss does not modify inputs in place and does not silently repair invalid widths or heights. Existing decode and OBB numerical safeguards remain responsible for valid box production.

## 6. ProbIoU and No-Overlap Gradient Behavior

ProbIoU models OBBs as Gaussian distributions, so zero polygon intersection does not by itself imply zero loss gradient. Moderately separated boxes still receive center, size, and covariance gradients.

For distant boxes, however, the current implementation computes a bounded Hellinger-style quantity from `exp(-bd)` and clamps `bd` to a maximum of 100. Consequently:

- `loss_probiou` approaches 1;
- its center gradient decays exponentially;
- after the upper clamp, its center gradient can be zero.

The recommended loss therefore assigns explicit roles:

- `loss_bbox`: non-saturating far-field center and side-length direction;
- `loss_kld`: continuous center, scale, and covariance geometry;
- `loss_probiou`: primary near-field and overlap-aware whole-box geometry;
- `loss_angle`: explicit periodic direction supervision.

The matcher L1 costs improve discrete assignment only. Hungarian matching is non-differentiable and must not be described as a source of criterion gradients.

## 7. Angle Geometry

DEIMv2 decodes final angles as:

```python
theta = torch.atan2(w_dy, w_dx) % torch.pi
```

Therefore `theta` belongs to `[0, pi)`. Define:

```python
delta = pred_theta - target_theta
delta = delta - torch.round(delta / torch.pi) * torch.pi
angle_penalty = torch.sin(2 * delta).square()
```

The penalty is zero at `delta=0` and `delta=±pi/2`, matching `w/h` interchange symmetry.

The aspect-ratio factor follows the current Ultralytics form:

```python
log_ar = torch.log((target_w + eps) / (target_h + eps))
scale_weight = torch.exp(-log_ar.square() / angle_lambda**2)
```

It is strongest near a square and decreases for extreme aspect ratios. ProbIoU and KLD remain responsible for distribution geometry when the explicit angle term is at a stationary point.

## 8. New Utility Module

Create `engine/deim/yolo_obb_loss.py` with four pure functions:

```python
def canonical_side_l1_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    normalizer: float | torch.Tensor,
) -> torch.Tensor:
    """Return center plus canonical short/long-side L1 over matched pairs."""


def yolo_angle_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    normalizer: float | torch.Tensor,
    lambda_val: float = 3.0,
) -> torch.Tensor:
    """Return scalar periodic angle loss over matched OBB pairs."""


def yolo_probiou_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    normalizer: float | torch.Tensor,
) -> torch.Tensor:
    """Return scalar (1 - ProbIoU) loss over matched OBB pairs."""


def compute_angle_cost_matrix(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    lambda_val: float = 3.0,
) -> torch.Tensor:
    """Return pairwise angle cost with shape (num_queries, num_targets)."""
```

The matcher constructs canonical pairwise L1 explicitly:

```python
pred_center = pred_bboxes[..., :2]
target_center = target_bboxes[..., :2]
pred_sides = torch.sort(pred_bboxes[..., 2:4], dim=-1).values
target_sides = torch.sort(target_bboxes[..., 2:4], dim=-1).values

cost_center = torch.cdist(pred_center, target_center, p=1)
cost_sides = torch.cdist(pred_sides, target_sides, p=1)
cost_bbox = cost_center + cost_sides
```

Empty matched tensors return scalar zero on the same device/dtype. Empty matcher targets return `(num_queries, 0)` matrices without NaN. Every utility rejects a final dimension other than 5 with `ValueError`.

## 9. Criterion Design

### 9.1 Constructor switches

Add:

```python
use_yolo_probiou: bool = False
use_yolo_angle: bool = False
keep_kld: bool = True
angle_lambda: float = 3.0
```

Defaults preserve unmigrated configs. The existing condition that both new switches are false selects the complete legacy OBB branch, including its original parameter-space `loss_bbox` behavior.

### 9.2 Configuration validation

For `box_mode == "obb"` in new mode, defined as either new switch being enabled:

- `weight_dict` must contain `loss_bbox` because canonical L1 is the non-saturating center/scale channel;
- enabled ProbIoU requires `loss_probiou`;
- enabled angle requires `loss_angle`;
- enabled KLD requires `loss_kld`;
- no key is automatically renamed, copied, removed, or assigned a fallback weight.

Legacy OBB mode continues to require its existing `loss_bbox` and, when enabled, `loss_kld`.

### 9.3 New OBB `loss_boxes` branch

For already-matched pairs:

```python
normalizer = num_boxes

losses["loss_bbox"] = canonical_side_l1_loss(
    src_boxes, target_boxes, normalizer
)

if self.use_yolo_probiou:
    losses["loss_probiou"] = yolo_probiou_loss(
        src_boxes, target_boxes, normalizer
    )

if self.use_yolo_angle:
    losses["loss_angle"] = yolo_angle_loss(
        src_boxes, target_boxes, normalizer,
        lambda_val=self.angle_lambda,
    )

if self.keep_kld:
    losses["loss_kld"] = (
        kld_loss(src_boxes, target_boxes, reduction="none").sum()
        / normalizer
    )
```

All Hungarian positives are equally weighted. `boxes_weight` remains accepted for HBB and existing meta-info compatibility but is ignored by these new OBB geometry losses.

`loss_bbox`, `loss_probiou`, and `loss_angle` remain outputs of the existing `boxes` dispatch family. No new dispatch names are added to `self.losses`.

## 10. Matcher Design

### 10.1 Constructor

Add:

```python
self.cost_angle = weight_dict.get("cost_angle", 0.0)
self.late_cost_bbox = weight_dict.get("late_cost_bbox", 0.0)
self.angle_order_alpha = angle_order_alpha
```

with `angle_order_alpha: float = 1.0`. Defaults preserve existing matcher configurations.

### 10.2 Early additive matcher

Before `matcher_change_epoch`:

```text
C = 2 * canonical_cost_bbox
  + 2 * cost_class
  + 4 * cost_probiou
  + 3 * cost_angle
  + 5 * cost_chamfer
```

`canonical_cost_bbox` is the sum of `cost_center` and `cost_sides` shown in Section 8. The old periodic angle component is removed from `cost_bbox`; angle geometry is represented only by `cost_angle`.

### 10.3 Late matching-aware matcher

At and after `matcher_change_epoch`:

```python
if self.cost_angle != 0:
    angle_cost = compute_angle_cost_matrix(out_bbox, target_bbox)
    angle_quality = (1.0 - angle_cost).clamp(0.0, 1.0)
else:
    angle_quality = 1.0

geometry_quality = (
    class_score
    * bbox_iou.pow(self.iou_order_alpha)
    * angle_quality.pow(self.angle_order_alpha)
)

C = -geometry_quality + self.late_cost_bbox * cost_bbox
```

The late L1 term is additive rather than converted into a multiplicative quality. This avoids introducing an arbitrary L1-to-quality mapping and temperature.

The independent baseline `late_cost_bbox=0.25` is lower than the early `cost_bbox=2` because `geometry_quality` is dimensionless and typically bounded by 1. Separate weights also permit independent ablation.

To reproduce the legacy late formula, set both `cost_angle=0` and `late_cost_bbox=0`.

## 11. Kendall Weighting

Do not change HBB/global defaults in `engine/solver/det_solver.py`. Migrated OBB configs that enable Kendall explicitly declare:

```yaml
KendallWeighting:
  enabled: true
  sigma_lr: 0.001
  init_log_sigma: 0.0
  loss_names:
    - loss_mal
    - loss_bbox
    - loss_probiou
    - loss_angle
    - loss_kld
    - loss_fgl
```

Main, aux, dn, enc, and pre variants of each family share one Kendall parameter through the existing suffix aggregation logic.

## 12. Configuration Baseline

Migrate:

- `configs/custom_obb/deimv2_obb_common.yml`
- `configs/custom_obb/deimv2_obb_sp.yml`
- `configs/deimv2_obb/deimv2_obb.yml`

Criterion baseline:

```yaml
DEIMCriterion:
  weight_dict:
    loss_mal: 1
    loss_bbox: 2
    loss_probiou: 5
    loss_angle: 3
    loss_kld: 1
    loss_fgl: 0.15
  losses: [mal, boxes, local]
  use_yolo_probiou: true
  use_yolo_angle: true
  keep_kld: true
  angle_lambda: 3.0
```

Config-specific MAL/FGL overrides remain unchanged.

Matcher baseline:

```yaml
matcher:
  weight_dict:
    cost_class: 2
    cost_bbox: 2
    cost_probiou: 4
    cost_angle: 3
    cost_chamfer: 5
    late_cost_bbox: 0.25
  change_matcher: true
  iou_order_alpha: 4.0
  angle_order_alpha: 1.0
```

Other experiment configs are not bulk-migrated and remain on the legacy branch because the new switches default to false.

## 13. Ablations

| Experiment | Criterion | Matcher |
|---|---|---|
| Legacy | Existing parameter-space `loss_bbox + loss_kld` | Existing costs |
| New geometry without canonical L1 | `loss_probiou + loss_angle + loss_kld` in a test-only comparison | New angle costs; L1 disabled |
| Recommended | Canonical `loss_bbox + loss_probiou + loss_angle + loss_kld` | Early canonical L1 and late canonical L1 |
| No KLD | Recommended without KLD; canonical L1 remains mandatory | Recommended matcher |
| No late L1 | Recommended criterion | `late_cost_bbox=0` |
| Legacy late matcher | Recommended criterion | `cost_angle=0`, `late_cost_bbox=0` |
| No Chamfer | Recommended criterion | `cost_chamfer=0` |

The no-canonical-L1 row exists for controlled convergence comparison, not as a supported migrated production configuration.

## 14. Testing

### 14.1 Utility geometry

- Identical boxes: canonical L1 and angle loss near zero; ProbIoU loss finite and below `1e-3`.
- Swapping prediction `w/h` leaves canonical L1 unchanged.
- A geometrically equivalent `(w,h,theta)` to `(h,w,theta+pi/2)` conversion leaves canonical L1 unchanged.
- Oversized and undersized predicted sides receive gradients toward target side lengths.
- `delta=pi/2`: angle loss/cost near zero.
- `delta=pi/4`, square GT: angle loss near one.
- Extreme-aspect-ratio angle weight is lower than the square case.
- Cost matrix shapes, empty inputs, dtype/device preservation, and malformed final dimensions.

### 14.2 No-overlap gradients

Construct a valid GT and a prediction that is fully disjoint and located to its lower right. Under the recommended criterion:

- all four geometry losses are finite;
- total `dL/dpred_cx > 0` and `dL/dpred_cy > 0`, so gradient descent moves the prediction toward the GT;
- canonical `loss_bbox` alone has finite, nonzero center gradients at moderate and far distances;
- increasing separation may reduce ProbIoU gradient, but total center gradient remains finite, nonzero, and correctly directed;
- disabling KLD still leaves the canonical L1 far-field channel intact;
- tests assert direction and nonzero gradients, not a fragile exact magnitude.

### 14.3 Criterion compatibility

- Recommended mode returns `loss_bbox`, `loss_probiou`, `loss_angle`, and `loss_kld`.
- New `loss_bbox` is canonical center/side L1 and contains no angle term.
- Invalid switch/key combinations fail before training.
- Legacy switches retain the old OBB `loss_bbox + loss_kld` formula.
- HBB outputs remain unchanged.

### 14.4 Matcher phases

- Early canonical L1 is invariant to raw `w/h` swaps.
- Early nonzero `cost_angle` changes assignment in a controlled case.
- Late `late_cost_bbox=0.25` prefers the center/scale-closer query when class, ProbIoU, and angle qualities are tied or nearly tied.
- `late_cost_bbox=0` removes late L1 influence.
- `cost_angle=0` and `late_cost_bbox=0` reproduce the legacy late formula.
- Empty targets and random valid OBB matrices remain finite and safe.

### 14.5 Kendall, configuration, and integration

- OBB Kendall names contain all six recommended families.
- Main and aux/dn/enc/pre variants aggregate under their base family.
- Every configured Kendall family has a produced criterion key.
- All three migrated configs parse and instantiate.
- One synthetic criterion forward with supported auxiliary outputs is finite.
- Existing OBB smoke/evaluation consistency and HBB regression tests pass.

### 14.6 Convergence experiment

Using the same dataset split, initialization policy, augmentation, optimizer, schedule, random seed policy, and training budget, compare:

1. legacy `loss_bbox + loss_kld`;
2. `loss_probiou + loss_angle + loss_kld` without canonical L1;
3. recommended canonical `loss_bbox + loss_probiou + loss_angle + loss_kld`.

Record per-loss curves, total regression loss, validation OBB metrics, and final mAP. The implementation is accepted based on correctness and stability; faster convergence or improved mAP is reported only if the controlled results demonstrate it.

## 15. Files Changed

| File | Responsibility |
|---|---|
| `engine/deim/yolo_obb_loss.py` | Canonical L1, ProbIoU/angle losses, and angle cost. |
| `engine/deim/deim_criterion.py` | Switches, validation, legacy/new OBB branches. |
| `engine/deim/matcher.py` | Canonical early/late L1 and angle geometry. |
| `configs/custom_obb/deimv2_obb_common.yml` | Recommended common baseline. |
| `configs/custom_obb/deimv2_obb_sp.yml` | Model weights and Kendall names. |
| `configs/deimv2_obb/deimv2_obb.yml` | Standalone baseline migration. |
| `test/test_yolo_obb_loss.py` | Utility and gradient tests. |
| `test/test_deim_criterion_obb_loss.py` | Criterion, validation, and compatibility tests. |
| `test/test_matcher_obb_angle.py` | Early/late matcher tests. |
| `test/test_kendall.py` | Loss-family aggregation tests. |

## 16. Acceptance Criteria

1. Recommended configs produce canonical `loss_bbox`, `loss_probiou`, `loss_angle`, and `loss_kld` for OBB positives.
2. Fully disjoint and far-field matched predictions receive finite, nonzero, correctly directed total center gradients.
3. Canonical L1 is invariant to `w/h` exchange and supplies center/scale gradients independently of ProbIoU saturation.
4. Early and late matchers both use canonical center/side L1; late L1 has independent weight `0.25`.
5. `cost_angle=0` and `late_cost_bbox=0` reproduce the legacy late matching-aware formula.
6. Kendall manages all and only configured OBB loss families.
7. Unmigrated legacy OBB configs and HBB behavior remain unchanged.
8. Existing OBB and HBB regression tests pass.
9. Convergence claims are based on controlled experiment results, not assumed from loss design.
