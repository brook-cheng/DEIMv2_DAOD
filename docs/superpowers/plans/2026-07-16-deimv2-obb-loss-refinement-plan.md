# DEIMv2-OBB Loss Refinement Implementation Plan

> **For agentic workers:** Execute task by task using TDD. Run every command from `/home/cx/win_dir/thired/DEIMv2_DAOD`. Do not commit unless the user explicitly requests commits.

**Goal:** Give disjoint and far-field Hungarian-positive OBB predictions reliable center and scale gradients while preserving representation equivalence, ProbIoU geometry, periodic angle supervision, KLD, and both matcher phases.

**Architecture:** Add pure OBB loss helpers for canonical center/side L1, ProbIoU, periodic angle loss, and angle cost. Keep the existing `boxes` criterion family and `loss_bbox` key, but select either the untouched legacy OBB formula or the new canonical formula through explicit switches. Use canonical center/side L1 in both matcher phases, with an independent late weight. Migrate only the three approved OBB baselines and their Kendall configuration.

**Source of truth:** `docs/superpowers/specs/2026-07-16-deimv2-obb-loss-refinement-design.md` at commit `dc82980`.

## Global constraints

- Restrict behavior changes to `box_mode == "obb"`; HBB behavior must remain unchanged.
- Preserve constructor defaults `use_yolo_probiou=False`, `use_yolo_angle=False`, `keep_kld=True` so unmigrated configs use the complete legacy OBB branch.
- In new mode, `loss_bbox` means center L1 plus sorted `[short, long]` side L1 and contains no angle term.
- Apply equal weighting to all Hungarian positives. Never multiply new OBB geometry losses by detached ProbIoU or `boxes_weight`.
- Do not silently rename, remove, copy, or synthesize loss keys.
- Keep `loss_bbox`, `loss_probiou`, and `loss_angle` as outputs of the existing `boxes` dispatch family.
- Early matcher `cost_bbox` is `cost_center + cost_sides`, each computed with `torch.cdist(..., p=1)`.
- Late matcher cost is `-geometry_quality + late_cost_bbox * cost_bbox`; baseline `late_cost_bbox=0.25`.
- Do not change global/HBB Kendall defaults in `engine/solver/det_solver.py`.
- Put tests under `test/`, not `tests/`.
- Do not modify ADR/FGL/DDF, MAL, CDN, Chamfer semantics, postprocessing, evaluation, or datasets.

## Recommended baseline

```yaml
DEIMCriterion:
  weight_dict:
    loss_mal: 1
    loss_bbox: 2
    loss_probiou: 5
    loss_angle: 3
    loss_kld: 1
    loss_fgl: 0.15
  use_yolo_probiou: true
  use_yolo_angle: true
  keep_kld: true
  angle_lambda: 3.0

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

Preserve intentional config-specific MAL/FGL weights.

---

## Task 1: Lock canonical L1 and angle geometry with failing tests

**Files**
- Create: `test/test_yolo_obb_loss.py`
- Create later: `engine/deim/yolo_obb_loss.py`

### Steps

- [ ] Add imports for `canonical_side_l1_loss`, `yolo_probiou_loss`, `yolo_angle_loss`, and `compute_angle_cost_matrix`.
- [ ] Test identical valid OBB pairs: canonical L1 and angle loss are zero; ProbIoU loss is finite and below `1e-3`.
- [ ] Test canonical L1 invariance when only prediction `w/h` are exchanged.
- [ ] Test invariance for the equivalent representation `(w, h, theta)` versus `(h, w, theta + pi/2)`.
- [ ] Test oversized and undersized sides: after backward, gradient descent moves sorted sides toward target sorted sides.
- [ ] Test `delta=pi/2` gives near-zero angle loss and cost; `delta=pi/4` with square GT gives angle loss near one.
- [ ] Test equal angle errors on square and extreme-ratio GTs; the extreme-ratio penalty must be lower.
- [ ] Test angle cost shape `(10, 3)`, empty target shape `(10, 0)`, no NaN, and device/dtype preservation.
- [ ] Parameterize all helpers with final dimensions other than 5 and require `ValueError` naming the invalid shape.
- [ ] Run and confirm failure is specifically the absent utility module:

```bash
python -m pytest test/test_yolo_obb_loss.py -q
```

**Acceptance:** tests collect correctly and fail only because production helpers do not yet exist.

---

## Task 2: Implement stateless OBB loss helpers

**Files**
- Create: `engine/deim/yolo_obb_loss.py`
- Test: `test/test_yolo_obb_loss.py`
- Reuse: `engine/deim/obb_ops.py` (`probiou`)

### Required interfaces

```python
def canonical_side_l1_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    normalizer: float | torch.Tensor,
) -> torch.Tensor: ...

def yolo_probiou_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    normalizer: float | torch.Tensor,
) -> torch.Tensor: ...

def yolo_angle_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    normalizer: float | torch.Tensor,
    lambda_val: float = 3.0,
) -> torch.Tensor: ...

def compute_angle_cost_matrix(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    lambda_val: float = 3.0,
) -> torch.Tensor: ...
```

### Steps

- [ ] Add a private shape guard requiring final dimension 5.
- [ ] Implement canonical L1 as center absolute error plus `torch.sort(...[..., 2:4]).values` side absolute error, summed and divided by `normalizer`.
- [ ] Implement ProbIoU as `sum(1 - probiou(pred, target)) / normalizer` without masks, quality weights, or detach.
- [ ] Implement wrapped angle delta as `delta - round(delta / pi) * pi` and penalty `sin(2 * delta).square()`.
- [ ] Implement target aspect-ratio weight `exp(-log((w+eps)/(h+eps)).square() / lambda_val**2)` with dtype-safe epsilon.
- [ ] Return same-device/dtype scalar zero for empty matched pairs and `(num_queries, 0)` for empty matcher targets before reductions.
- [ ] Keep prediction gradients intact; do not modify inputs in place or clamp valid predictions in the helper.
- [ ] Run:

```bash
python -m pytest test/test_yolo_obb_loss.py -q
```

**Acceptance:** all utility, symmetry, empty-input, malformed-shape, and gradient tests pass.

---

## Task 3: Lock criterion modes and no-overlap gradients with failing tests

**Files**
- Create: `test/test_deim_criterion_obb_loss.py`
- Modify later: `engine/deim/deim_criterion.py`

### Steps

- [ ] Add a minimal repository-style fixture for `DEIMCriterion` and its matcher.
- [ ] Test legacy OBB mode (`False, False, True`) with legacy keys; assert the existing parameter-space `loss_bbox + loss_kld` values and keys remain unchanged.
- [ ] Test recommended mode; assert exact geometry keys are `loss_bbox`, `loss_probiou`, `loss_angle`, and `loss_kld`.
- [ ] Verify new `loss_bbox` is invariant to `w/h` exchange and unchanged when only angles differ.
- [ ] Test new mode without KLD; canonical L1 remains present and finite.
- [ ] Test empty matches return finite scalar zeros for every enabled loss.
- [ ] Add validation tests for missing `loss_bbox`, `loss_probiou`, `loss_angle`, and enabled `loss_kld` keys.
- [ ] Prove validation does not mutate `weight_dict` and performs no automatic key conversion.
- [ ] Add HBB regression coverage for existing box and IoU keys and values.
- [ ] Construct a fully disjoint prediction lower-right of its GT. Backpropagate the recommended geometry sum and assert finite `dL/dcx > 0`, `dL/dcy > 0`.
- [ ] Repeat at a farther distance; allow ProbIoU gradient decay but require finite, nonzero, correctly directed total center gradients.
- [ ] Disable KLD and prove canonical `loss_bbox` alone preserves the far-field center direction.
- [ ] Run and confirm only new-mode expectations fail before implementation:

```bash
python -m pytest test/test_deim_criterion_obb_loss.py -q
```

**Acceptance:** the red phase distinguishes missing new behavior from already-working legacy behavior.

---

## Task 4: Implement explicit legacy and canonical OBB criterion branches

**Files**
- Modify: `engine/deim/deim_criterion.py`
- Reuse: `engine/deim/yolo_obb_loss.py`
- Test: `test/test_deim_criterion_obb_loss.py`

### Steps

- [ ] Add constructor fields with defaults `False`, `False`, `True`, and `3.0` for ProbIoU, angle, KLD, and angle lambda.
- [ ] Define new mode as either new loss switch enabled; both switches false select the untouched legacy OBB formula.
- [ ] Validate new mode: `loss_bbox` is mandatory; enabled optional losses require their matching keys; no key mutation or fallback.
- [ ] Preserve the entire old parameter-space OBB `loss_bbox` formula in legacy mode, including periodic-angle behavior.
- [ ] In new mode, always produce canonical `loss_bbox`, then produce each enabled optional loss using matched `src_boxes`, `target_boxes`, and `normalizer=num_boxes`.
- [ ] Ignore `boxes_weight` for all new OBB geometry losses while retaining the parameter and HBB behavior.
- [ ] Keep KLD calculation on the existing `kld_loss(..., reduction="none").sum() / num_boxes` path.
- [ ] Do not alter `self.losses`, `get_loss_meta_info()`, `use_uni_set`, suffix generation, or HBB code.
- [ ] Run:

```bash
python -m pytest test/test_deim_criterion_obb_loss.py -q
```

**Acceptance:** legacy, canonical, no-KLD, validation, empty, no-overlap gradient, and HBB tests pass.

---

## Task 5: Lock early and late matcher behavior with failing tests

**Files**
- Create: `test/test_matcher_obb_angle.py`
- Modify later: `engine/deim/matcher.py`

### Steps

- [ ] Build controlled synthetic OBBs where raw `w/h` exchange would alter old L1 but canonical sides must preserve assignment.
- [ ] Before `matcher_change_epoch`, verify `cost_bbox` is invariant to equivalent `w/h` exchange.
- [ ] Before the epoch, show nonzero `cost_angle` changes assignment in an angle-controlled case.
- [ ] At/after the epoch, tie or nearly tie class, ProbIoU, and angle qualities; show `late_cost_bbox=0.25` selects the center/scale-closer query.
- [ ] Show `late_cost_bbox=0` removes late L1 influence.
- [ ] Show `cost_angle=0` and `late_cost_bbox=0` reproduce the exact legacy late assignment formula.
- [ ] Test empty targets return empty index tensors and random valid OBB costs remain finite.
- [ ] Run and confirm canonical early and late-L1 expectations fail before implementation:

```bash
python -m pytest test/test_matcher_obb_angle.py -q
```

**Acceptance:** failures isolate the absent canonical and late matching behavior.

---

## Task 6: Implement canonical L1 in both matcher phases

**Files**
- Modify: `engine/deim/matcher.py`
- Reuse: `engine/deim/yolo_obb_loss.py`
- Test: `test/test_matcher_obb_angle.py`

### Steps

- [ ] Add `angle_order_alpha: float = 1.0` and store it.
- [ ] Read `cost_angle` and `late_cost_bbox` from `weight_dict` with defaults `0.0`.
- [ ] In the OBB path, construct pairwise canonical L1 exactly as:

```python
pred_center = out_bbox[..., :2]
tgt_center = tgt_bbox[..., :2]
pred_sides = torch.sort(out_bbox[..., 2:4], dim=-1).values
tgt_sides = torch.sort(tgt_bbox[..., 2:4], dim=-1).values
cost_center = torch.cdist(pred_center, tgt_center, p=1)
cost_sides = torch.cdist(pred_sides, tgt_sides, p=1)
cost_bbox = cost_center + cost_sides
```

- [ ] Before the change epoch, replace the old OBB spatial-plus-angle `cost_bbox` with canonical `cost_bbox`; add independent `cost_angle`; preserve class, ProbIoU, and Chamfer terms.
- [ ] At/after the epoch, compute angle quality only when `cost_angle != 0`; otherwise use unit quality.
- [ ] Compute `geometry_quality = class_score * bbox_iou.pow(iou_order_alpha) * angle_quality.pow(angle_order_alpha)`.
- [ ] Set final late cost to `-geometry_quality + late_cost_bbox * cost_bbox`.
- [ ] Do not multiply late L1 by early `cost_bbox`; the late coefficient is independent.
- [ ] Preserve HBB matching unchanged.
- [ ] Run:

```bash
python -m pytest test/test_matcher_obb_angle.py -q
```

**Acceptance:** both phases pass canonical symmetry, angle, late-L1, legacy-ablation, finite-cost, and empty-target tests.

---

## Task 7: Migrate the three approved OBB baselines

**Files**
- Modify: `configs/custom_obb/deimv2_obb_common.yml`
- Modify: `configs/custom_obb/deimv2_obb_sp.yml`
- Modify: `configs/deimv2_obb/deimv2_obb.yml`

### Steps

- [ ] In each criterion `weight_dict`, retain/add `loss_bbox: 2`, add `loss_probiou: 5`, `loss_angle: 3`, and `loss_kld: 1`; preserve intentional MAL/FGL overrides.
- [ ] Explicitly enable ProbIoU, angle, and KLD with `angle_lambda: 3.0`.
- [ ] In each matcher, set `cost_class: 2`, `cost_bbox: 2`, `cost_probiou: 4`, `cost_angle: 3`, `cost_chamfer: 5`, and `late_cost_bbox: 0.25`.
- [ ] Set `angle_order_alpha: 1.0`; preserve each file's matcher-change epoch and unrelated settings.
- [ ] Do not modify any other experiment config.
- [ ] Parse all three files and assert exact geometry keys and values:

```bash
python - <<'PY'
from engine.core.yaml_config import YAMLConfig

for path in (
    "configs/custom_obb/deimv2_obb_common.yml",
    "configs/custom_obb/deimv2_obb_sp.yml",
    "configs/deimv2_obb/deimv2_obb.yml",
):
    cfg = YAMLConfig(path).yaml_cfg["DEIMCriterion"]
    assert cfg["weight_dict"]["loss_bbox"] == 2
    assert cfg["weight_dict"]["loss_probiou"] == 5
    assert cfg["weight_dict"]["loss_angle"] == 3
    assert cfg["weight_dict"]["loss_kld"] == 1
    matcher = cfg["matcher"]
    assert matcher["weight_dict"]["late_cost_bbox"] == 0.25
    assert matcher["angle_order_alpha"] == 1.0
    print(path, "OK")
PY
```

**Acceptance:** all target configs parse, instantiate supported components, and unrelated configs remain untouched.

---

## Task 8: Migrate Kendall families and test suffix aggregation

**Files**
- Modify: `configs/custom_obb/deimv2_obb_sp.yml`
- Modify: `test/test_kendall.py`
- Must not modify: `engine/solver/det_solver.py`

### Steps

- [ ] Set OBB Kendall `loss_names` to `loss_mal`, `loss_bbox`, `loss_probiou`, `loss_angle`, `loss_kld`, and `loss_fgl`.
- [ ] Extend tests so main and repository-supported aux/dn/enc/pre suffix keys aggregate under the matching base family.
- [ ] Add a contract test proving every configured Kendall family has a produced criterion key in a synthetic main-plus-auxiliary result.
- [ ] Preserve and verify HBB/global default behavior.
- [ ] Run:

```bash
python -m pytest test/test_kendall.py -q
```

**Acceptance:** exactly six OBB families are managed, suffix aggregation passes, and global defaults are unchanged.

---

## Task 9: Run integration and regression gates

**Files**
- All files changed by Tasks 1-8
- Existing relevant tests under `test/`

### Steps

- [ ] Instantiate model, criterion, matcher, and Kendall weighting from `configs/custom_obb/deimv2_obb_sp.yml` through the existing configuration entry point.
- [ ] Run one synthetic criterion forward with main and supported auxiliary/dn/enc/pre outputs; assert every produced loss is finite.
- [ ] Assert recommended suffix families include canonical `loss_bbox`, ProbIoU, angle, and KLD.
- [ ] Instantiate an unmigrated or explicit legacy OBB fixture and prove its output remains legacy `loss_bbox + loss_kld`.
- [ ] Run focused tests:

```bash
python -m pytest \
  test/test_yolo_obb_loss.py \
  test/test_deim_criterion_obb_loss.py \
  test/test_matcher_obb_angle.py \
  test/test_kendall.py -q
```

- [ ] Run existing OBB smoke/evaluation consistency tests and the smallest HBB criterion/matcher regression suite identified under `test/`.
- [ ] Run repository lint/type/static checks on every changed Python file; if no project command exists, run `python -m compileall engine/deim test`.
- [ ] Run `git diff --check` and verify the diff is restricted to the approved file list.

**Acceptance:** new focused tests, existing OBB/HBB regressions, configuration instantiation, and static checks pass.

---

## Task 10: Run the controlled convergence comparison

**Files**
- Prefer existing experiment config inheritance; add only narrowly scoped experiment configs/scripts if the repository requires them.
- Record outputs in the repository's existing experiment tracking format.

### Steps

- [ ] Define three runs with identical dataset split, initialization policy, augmentation, optimizer, schedule, random seed policy, hardware policy, and training budget:
  1. legacy `loss_bbox + loss_kld`;
  2. `loss_probiou + loss_angle + loss_kld` without canonical L1, as a test-only comparison;
  3. recommended canonical `loss_bbox + loss_probiou + loss_angle + loss_kld`.
- [ ] Record per-loss curves, total regression loss, validation OBB metrics, final mAP, run config, seed, and checkpoint identifiers.
- [ ] Compare early convergence at predeclared equal steps/epochs and final metrics at equal budget.
- [ ] Report measured results without claiming improvement when confidence or run parity is insufficient.

**Acceptance:** all three runs are reproducible and directly comparable; any convergence claim cites recorded evidence.

---

## Dependency order

| Order | Task | Depends on |
|---:|---|---|
| 1 | Utility failing tests | None |
| 2 | Utility implementation | 1 |
| 3 | Criterion failing tests | 2 |
| 4 | Criterion implementation | 3 |
| 5 | Matcher failing tests | 2 |
| 6 | Matcher implementation | 5 |
| 7 | Config migration | 4, 6 |
| 8 | Kendall migration/tests | 7 |
| 9 | Integration/regressions | 1-8 |
| 10 | Convergence comparison | 9 |

Tasks 3-4 and 5-6 may run in parallel after Task 2. Do not migrate configs before both constructor surfaces accept the new fields.

## Expected implementation files

```text
engine/deim/yolo_obb_loss.py
engine/deim/deim_criterion.py
engine/deim/matcher.py
configs/custom_obb/deimv2_obb_common.yml
configs/custom_obb/deimv2_obb_sp.yml
configs/deimv2_obb/deimv2_obb.yml
test/test_yolo_obb_loss.py
test/test_deim_criterion_obb_loss.py
test/test_matcher_obb_angle.py
test/test_kendall.py
```

## Must not have

- No raw `w`-to-`w`, `h`-to-`h` L1 in the new OBB branch.
- No angle term inside new `loss_bbox` or canonical `cost_bbox`.
- No detached ProbIoU self-weighting.
- No automatic loss-key migration.
- No new criterion dispatch family.
- No reuse of early `cost_bbox=2` as the late coefficient; use `late_cost_bbox=0.25`.
- No changes to global Kendall defaults or HBB behavior.
- No performance claim without controlled-run evidence.
- No tests under `tests/`.

## Completion criteria

1. Disjoint and far-field positives receive finite, nonzero, correctly directed center gradients.
2. Canonical L1 is invariant to `w/h` exchange and supplies center/scale gradients independently of ProbIoU.
3. Recommended OBB output includes `loss_bbox`, `loss_probiou`, `loss_angle`, and `loss_kld`.
4. Early and late matcher phases use `cost_center + cost_sides`; late L1 uses independent weight `0.25`.
5. `cost_angle=0` plus `late_cost_bbox=0` restores the legacy late formula.
6. Kendall manages exactly the configured OBB families and suffixes.
7. Unmigrated OBB and HBB behavior remain unchanged.
8. Focused and existing regression tests pass.
9. Convergence conclusions are backed by reproducible equal-budget experiments.
