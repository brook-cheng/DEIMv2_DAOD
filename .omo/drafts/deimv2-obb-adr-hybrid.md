---
slug: deimv2-obb-adr-hybrid
status: plan-written
intent: clear
pending-action: write .omo/plans/deimv2-obb-adr-hybrid.md
approach: Keep Ding-style ADR as the primary OBB representation, add tests first, then add periodic angle utility, wire loss and matcher through it, resolve offset-scale and offset-validity gaps, and audit/fix the decoupled offset reference path without replacing the architecture with pure RiO.
---

# Draft: deimv2-obb-adr-hybrid

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
| C1 | ADR geometry and conversion tests prove existing external-rect + epsilon/eta representation remains valid | active | docs/superpowers/specs/2026-07-07-deimv2-obb-representation-refinement-design.md:51-85; engine/deim/obb_geometry.py:79-169 |
| C2 | Periodic angle distance is shared by loss and matcher | active | docs/superpowers/specs/2026-07-07-deimv2-obb-representation-refinement-design.md:140-177; engine/deim/deim_criterion.py:233-267; engine/deim/matcher.py:169-180 |
| C3 | ADR offset scaling and validity are explicit, tested, and do not silently diverge | active | docs/superpowers/specs/2026-07-07-deimv2-obb-representation-refinement-design.md:127-138; engine/deim/dfine_utils.py:190-291 |
| C4 | Decoupled angle branch uses a deliberate reference representation and is smoke-tested with synthetic_exp_020_dec.yml | active | engine/deim/deim_decoder.py:322-388; configs/custom_obb/synthetic_configs/synthetic_exp_020_dec.yml:247-249 |

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->
| Angle loss weight | `lambda_angle: 1.0` for criterion and matcher | Preserves current away-from-seam scale while only changing seam topology | reversible config |
| Offset residual scale | `offset_scale_source: pre` | Matches current active code and design default | reversible config/ablation |
| Offset validity | Clamp only detached references / eval decode surfaces, never the loss-bearing tensor | Prevents invalid reference propagation without killing gradients | reversible |
| Pure RiO baseline | Out of scope for this implementation plan | User approved preserving Ding-style ADR core | reversible future branch |
| Full 80-epoch training | Not a verification gate | User offered config for tests; fast agent-executed QA is required | reversible after implementation |

## Findings (cited - path:lines)
- Current OBB `loss_bbox` divides theta by pi and applies ordinary L1, so it is not periodic: `engine/deim/deim_criterion.py:255-264`.
- Current OBB matcher uses `torch.cdist` with angle scaled by `1/pi`, so it is also ordinary non-periodic L1: `engine/deim/matcher.py:169-180`.
- ADR decode uses pre-adjustment external-rectangle scale for epsilon/eta and contains a TODO about pre/post scale choice: `engine/deim/dfine_utils.py:210-219`.
- ADR target encoding also uses pre-adjustment external-rectangle scale: `engine/deim/dfine_utils.py:236-287`.
- `external_rect_to_oriented_box` currently accepts offsets directly and reconstructs vertices without validity guarding: `engine/deim/obb_geometry.py:115-169`.
- The approved synthetic test config runs `box_mode: obb` and `decouple_angle: True`: `configs/custom_obb/synthetic_configs/synthetic_exp_020_dec.yml:247-249`.
- The config uses OBB losses and matcher costs that exercise `loss_bbox`, `loss_kld`, `loss_fgl`, `cost_bbox`, `cost_chamfer`, and `cost_probiou`: `configs/custom_obb/synthetic_configs/synthetic_exp_020_dec.yml:258-288`.

## Decisions (with rationale)
- Add `periodic_angle_distance` once in `engine/deim/obb_geometry.py` and import it from both criterion and matcher. No inline duplicate formulae.
- Add `lambda_angle=1.0` to both `DEIMCriterion` and `HungarianMatcher`, plus config entries under `DEIMCriterion` and its nested matcher.
- Keep `offset_scale_source: pre` as default, implement a shared helper path for `distance2bbox_obb` and `bbox2distance_obb`, and expose `post` only as an ablation switch.
- Do not change `argmin`/`argmax` vertex assignment in this iteration; add tests that document seam/near-square behavior.
- Do not enable `GatedSoftmaxFusion`; it remains out of scope.
- Do not require full training or mAP improvement to pass the implementation. Unit tests plus a short decouple-angle smoke path are enough.

## Scope IN
- Pytest coverage for ADR geometry round-trip, decode/target inversion, near-square behavior, periodic angle seam, matcher seam, and offset validity.
- Shared periodic angle utility and wiring into `loss_bbox` and OBB matcher cost.
- Configurable angle weight and offset scale source with defaults matching current behavior except at angle seam.
- Offset validity guards that protect detached references/eval paths without suppressing training gradients.
- Decoder offset-reference audit and the smallest safe correction if the active decouple-angle path feeds scalar theta where the branch expects spatial ADR reference values.
- Smoke verification using `configs/custom_obb/synthetic_configs/synthetic_exp_020_dec.yml` or a synthetic batch derived from the same model settings.

## Scope OUT (Must NOT have)
- No pure RiO-style replacement of ADR.
- No broad model architecture rewrite.
- No full 80-epoch training gate.
- No enabling commented-out gated fusion.
- No changing vertex assignment from `argmin`/`argmax` to multi-candidate smoothing.
- No removal of KLD/ProbIoU/chamfer costs.
- No commit; user manages git.

## Open questions
- None blocking. Defaults above are deliberately reversible and recorded for veto.

## Approval gate
status: approved-by-request
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
User explicitly requested: "请开始实现计划，如果需要测试的话，能够使用 configs/custom_obb/synthetic_configs/synthetic_exp_020_dec.yml 中信息". Treat this as approval to write the plan file, not approval to implement.
