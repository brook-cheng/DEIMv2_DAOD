# DEIMv2 OBB Representation and Refinement Design

Date: 2026-07-07

## 1. Purpose

This design keeps the current Ding-style ADR representation as the main oriented-box refinement path and closes the known gaps around periodic angle geometry, offset validity, and query decoupling.

The goal is not to replace the current DEIMv2 OBB branch with RiO-DETR. The goal is to make the existing external-rectangle-plus-offset design geometrically consistent, testable, and suitable for ablation.

## 2. Background

The current OBB branch follows a D-FINE/DEIM-style distribution refinement pattern. Instead of directly refining only `(cx, cy, w, h, theta)`, it converts a reference OBB into:

- an axis-aligned external rectangle; and
- two vertex offsets `(epsilon, eta)`.

The decoder predicts six distribution-refined residuals:

```text
(alpha, beta, gamma, delta, epsilon, eta)
```

where `alpha/beta/gamma/delta` refine the external rectangle edges and `epsilon/eta` refine the oriented vertices on the external rectangle boundary.

This is aligned with Ding et al. 2026 ADR: direct scalar angle regression is avoided during the distribution-refinement step, while the final public box representation remains the standard 5D OBB:

```text
(cx, cy, w, h, theta), theta in [0, pi)
```

RiO-DETR uses a different design. It keeps the standard 5D OBB representation, removes angle from the positional query, and refines angle with a bounded periodic update plus shortest-path periodic angle loss. This design borrows RiO-DETR's periodic-consistency and query-decoupling principles without discarding the existing ADR-style representation.

## 3. Design Goals

1. Preserve the current ADR-style internal representation because it matches DEIM/D-FINE distribution refinement.
2. Keep the external interface as standard 5D OBB for loss, evaluation, visualization, and export.
3. Prevent ordinary Euclidean angle L1 from over-penalizing equivalent angles near the `0/pi` seam.
4. Keep unreliable angle estimates out of the main positional query.
5. Define offset semantics and validity constraints clearly.
6. Add targeted tests for round-trip geometry, decode/refine consistency, periodic-angle behavior, and degenerate cases.
7. Leave room for ablations rather than hard-coding unverified assumptions.

## 4. Non-Goals

- Do not replace the whole branch with pure RiO-style direct angle refinement in this design.
- Do not change the public output format away from `(cx, cy, w, h, theta)`.
- Do not perform unrelated model-architecture refactors.
- Do not remove KLD or ProbIoU geometry-aware supervision.

## 5. Representation

### 5.1 Internal Representation

The internal refinement representation remains:

```text
external_rect = (x1, y1, x2, y2)
vertex_offsets = (epsilon, eta)
```

The four reconstructed vertices are:

```text
v_top    = (x2 - epsilon, y1)
v_right  = (x2, y2 - eta)
v_bottom = (x1 + epsilon, y2)
v_left   = (x1, y1 + eta)
```

The OBB is recovered from these vertices by selecting the longer adjacent edge as width and computing:

```text
theta = atan2(width_edge_y, width_edge_x) mod pi
```

### 5.2 Output Representation

Every downstream consumer receives standard OBBs:

```text
(cx, cy, w, h, theta)
```

This keeps compatibility with KLD, ProbIoU, DOTA export, visualization, and existing evaluation code.

## 6. Decoder and Query Design

### 6.1 Main Localization Query

The main bbox/class decoder positional query must remain angle-free:

```text
Q_pos = phi(cx, cy, w, h)
```

or an equivalent external-rectangle spatial representation. It must not encode scalar `theta` directly.

Rationale: RiO-DETR shows that orientation is content-driven and should not contaminate early spatial attention. This also matches the current direction where the main query uses only the first four dimensions.

### 6.2 Offset Branch Query

The offset branch should avoid treating an unreliable scalar angle estimate as a positional prior. Its query embedding should prefer spatial reference data:

```text
query_offset_input = external_rect or (cx, cy, w, h)
```

If angle-related information is introduced later, it must be periodic-aware and should be justified by ablation.

## 7. Refinement Flow

Each decoder layer follows this conceptual flow:

```text
current OBB reference
  -> oriented_box_to_external_rect()
  -> current external rectangle + current offsets
  -> predict six residual distributions
  -> integral projection to residual distances
  -> refine external rectangle edges
  -> refine epsilon/eta offsets
  -> external_rect_to_oriented_box()
  -> next OBB reference/output
```

### 7.1 Offset Residual Scale

The preferred default is to scale offset residuals by the pre-adjustment external rectangle size. This keeps all residuals in the current reference coordinate frame and avoids same-layer coupling between edge refinement and offset refinement.

The alternative, scaling by the post-adjustment external rectangle size, remains an explicit ablation.

Required ablation:

```text
A. offset residual scale = pre-adjustment external rectangle
B. offset residual scale = post-adjustment external rectangle
```

## 8. Loss and Matching

### 8.1 Geometry-Aware Terms

KLD and ProbIoU remain the primary geometry-aware supervision terms. They help align optimization with actual OBB geometry and should not be removed as part of this design.

### 8.2 Periodic Angle L1

If `loss_bbox` includes an angle term, the angle component must use shortest-path periodic distance:

```text
d = abs(theta_pred - theta_tgt)
L_angle = min(d, pi - d)
```

The spatial dimensions keep ordinary L1:

```text
L_xywh = L1(pred[..., :4], target[..., :4])
```

The combined loss should use a normalized angle component:

```text
loss_bbox = L_xywh + lambda_angle * L_angle / pi
```

### 8.3 Matcher Consistency

The matcher bbox cost must use the same periodic angle distance if it includes `theta`. Otherwise, matching and training loss will disagree near the `0/pi` seam.

Matcher cost should conceptually be:

```text
cost_bbox = L1_xywh + lambda_angle * periodic_angle_distance / pi
```

with ProbIoU cost retained.

## 9. Validity and Degenerate Cases

### 9.1 Offset Validity

The valid offset range is:

```text
0 <= epsilon <= external_width
0 <= eta <= external_height
```

Training should avoid unnecessary hard clamping that destroys gradients, but decoding and evaluation should guard against invalid geometry. The exact enforcement mechanism should be tested and, if needed, ablated:

- soft penalty for invalid offsets;
- sigmoid ratio parameterization;
- decode-time clamp only.

### 9.2 Near-Square Instability

External-rectangle-plus-offset representation does not eliminate long-side instability. When adjacent edge lengths are close, long-side selection can flip and induce an approximate `pi/2` angle jump.

This must be tested explicitly for nearly square boxes.

### 9.3 Vertex Assignment Seam

The conversion from OBB to offsets selects top and right vertices using `argmin(y)` and `argmax(x)`. This can jump near axis-aligned or square-like cases. The first implementation should keep the current behavior but add tests and diagnostics before considering smoother multi-candidate selection.

## 10. Test Plan

### 10.1 Geometry Round-Trip Tests

Test:

```text
OBB -> external rectangle + offsets -> OBB
```

Use vertex-level geometry error rather than raw parameter error because equivalent OBB parameterizations may differ.

Cases:

- theta near 0;
- theta near pi;
- theta near pi/2;
- axis-aligned boxes;
- thin long boxes;
- square-like boxes;
- random valid OBBs.

### 10.2 Decode/Target Inversion Tests

Test:

```text
bbox2distance_obb(reference, target)
distance2bbox_obb(reference, distance)
```

Expected result: reconstructed target geometry matches the original target within tolerance.

### 10.3 Periodic Loss Tests

Construct seam cases:

```text
theta_pred = pi - epsilon
theta_tgt = epsilon
```

Expected result:

```text
ordinary L1 is large
periodic L1 is small
```

### 10.4 Matcher Tests

Construct two candidate predictions where ordinary angle L1 chooses the wrong seam-side match and periodic angle distance chooses the geometry-consistent match.

### 10.5 Offset Validity Tests

Cover invalid or boundary offsets:

- negative epsilon/eta;
- epsilon greater than external width;
- eta greater than external height;
- zero-width or zero-height external rectangles.

## 11. Ablation Plan

Minimum ablations:

1. Current ADR-style baseline.
2. Baseline + periodic angle L1.
3. Baseline + periodic matcher angle cost.
4. Baseline + offset query without scalar angle prior.
5. Offset residual scale: pre-adjustment vs post-adjustment external rectangle.
6. Pure RiO-style direct angle refinement baseline, only if a larger comparison branch is desired.

## 12. Acceptance Criteria

The design is considered implemented when:

1. The ADR-style representation remains the primary OBB refinement path.
2. Main positional query remains angle-free.
3. Angle L1 and matcher angle cost use periodic shortest-path distance wherever angle L1/cost is used.
4. Round-trip, decode/target inversion, seam, near-square, and offset-validity tests exist and pass.
5. Existing KLD/ProbIoU behavior is preserved.
6. Any change to offset scaling is documented and backed by an ablation result or a clear default-plus-ablation plan.

## 13. Recommended Implementation Order

1. Add geometry tests for current behavior without changing model code.
2. Add periodic angle distance utility and tests.
3. Update `loss_bbox` angle component to use periodic distance.
4. Update matcher angle cost to use the same periodic distance.
5. Audit offset branch query input and remove scalar angle positional prior if present.
6. Add offset validity guards or diagnostics.
7. Run ablations for offset residual scale and query input.

## 14. Key Decision

Proceed with a Ding-style ADR core and a RiO-style consistency layer:

```text
Keep external rectangle + epsilon/eta distribution refinement.
Keep final output as standard 5D OBB.
Keep main positional query angle-free.
Make angle loss and matcher cost periodic-aware.
Validate offset geometry with targeted tests before deeper architectural changes.
```
