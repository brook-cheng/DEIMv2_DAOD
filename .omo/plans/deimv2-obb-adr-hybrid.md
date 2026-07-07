# deimv2-obb-adr-hybrid - Work Plan

## TL;DR (For humans)

**What you'll get:** The current external-rectangle-plus-offset OBB design stays as the main representation, but its weak points are made explicit and test-covered: periodic angle distance, consistent matching, offset validity, and decoupled-angle smoke coverage.

**Why this approach:** It preserves the Ding-style ADR core already built into the model while borrowing RiO-DETR's most important consistency fix: angles near `0` and `pi` must be compared by shortest periodic distance, not ordinary scalar L1.

**What it will NOT do:** It will not replace the model with pure RiO-DETR, enable unrelated gated-fusion code, change vertex assignment strategy, or require a full 80-epoch training run.

**Effort:** Medium
**Risk:** Medium - the loss/matcher changes are small, but the decoupled offset-reference path must be audited carefully because `decouple_angle: True` is active in the supplied config.
**Decisions to sanity-check:** Defaults are `lambda_angle=1.0`, `offset_scale_source="pre"`, no full-training gate, and no pure RiO branch.

Your next move: approve execution with `$start-work` / "start work", or ask for a high-accuracy plan review first. Full execution detail follows below.

---

> TL;DR (machine): Medium-risk test-first implementation plan to keep ADR OBB representation while adding periodic angle loss/matcher consistency, explicit offset-scale/validity behavior, decouple-angle smoke verification, and no architecture replacement.

## Scope

### Must have

- Preserve the Ding-style ADR core: external rectangle + `(epsilon, eta)` offsets remain the primary internal OBB refinement representation.
- Keep public outputs as standard 5D OBB `(cx, cy, w, h, theta)` with `theta` in `[0, pi)`.
- Add a single shared periodic angle distance utility and use it from both OBB `loss_bbox` and the OBB matcher bbox cost.
- Add configurable defaults:
  - `lambda_angle: 1.0` for criterion and matcher.
  - `offset_scale_source: pre` for ADR encode/decode.
- Keep KLD, ProbIoU, and chamfer paths active and unchanged except for surrounding bbox-cost composition.
- Promote geometry checks into pytest and add tests for periodic angle, matcher seam behavior, decode/target inversion, near-square behavior, and offset validity.
- Audit the decoupled offset-reference path under `decouple_angle: True`; apply only the smallest safe correction needed to keep the offset branch spatial/ADR-consistent.
- Use `configs/custom_obb/synthetic_configs/synthetic_exp_020_dec.yml` as the model/settings reference for smoke verification.

### Must NOT have (guardrails, anti-slop, scope boundaries)

- Do not replace ADR with pure RiO-style direct angle refinement.
- Do not enable `GatedSoftmaxFusion` or uncomment the gated-fusion block.
- Do not change top/right vertex assignment from current `argmin(y)` / `argmax(x)` behavior.
- Do not remove KLD, ProbIoU, chamfer, FGL, or DDF losses/costs.
- Do not require full 80-epoch training or mAP improvement as a basic verification gate.
- Do not commit; user manages git.
- Do not use type suppressions or broad exception swallowing.

## Verification strategy

> Zero human intervention - all verification is agent-executed.

- Test decision: TDD + pytest. Add failing tests for each behavioral gap before production changes.
- Evidence path: `.omo/evidence/task-<N>-deimv2-obb-adr-hybrid.txt` for each task command output.
- Changed Python files require `lsp_diagnostics` clean before done.
- Fast gates:
  - targeted pytest geometry/loss/matcher tests;
  - existing adjacent tests (`test_model_correctness.py`, `test_obb_transforms.py`) where runnable;
  - a short `decouple_angle=True` smoke test based on `synthetic_exp_020_dec.yml` settings or synthetic batch construction.
- No full-training gate. If a 1-2 step smoke path is too heavy in the local environment, document the blocker and still run all pure/unit tests.

## Execution strategy

### Parallel execution waves

- **Wave 1: Lock current behavior with tests.** Add pytest coverage for geometry round-trip, periodic distance expectation, matcher seam, decode/target inversion, and offset-validity behavior. These tests should fail where production code is currently missing behavior.
- **Wave 2: Implement shared primitives and wire loss/matcher.** Add `periodic_angle_distance`, `lambda_angle`, and matcher cost decomposition.
- **Wave 3: ADR offset consistency and decoder audit.** Resolve `offset_scale_source`, offset validity guards, and the decouple-angle reference representation path.
- **Wave 4: Smoke/diagnostics and cleanup.** Run targeted tests, smoke config path, LSP diagnostics, and final verification review.

### Dependency matrix

| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | none | 2, 3, 4, 5, 6 | none |
| 2 | 1 | 3, 8 | 5 |
| 3 | 1, 2 | 8 | 4, 5 |
| 4 | 1, 2 | 8 | 3, 5 |
| 5 | 1 | 6, 8 | 2, 3, 4 |
| 6 | 1, 5 | 7, 8 | none |
| 7 | 1, 6 | 8 | none |
| 8 | 2, 3, 4, 5, 6, 7 | final verification | none |

## Todos

> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->

- [x] 1. Add OBB geometry/periodic test scaffold before production changes
  What to do / Must NOT do: Create pytest coverage for the approved spec's geometry and seam requirements. Promote the useful logic from `engine/deim/obb_geometry.py`'s `__main__` self-test into a normal test file, but do not change production geometry behavior yet. Include helpers for vertex-level error so tests compare geometry rather than raw parameter identity. Must not change `argmin`/`argmax` vertex selection.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 2, 3, 4, 5, 6
  References (executor has NO interview context - be exhaustive): `docs/superpowers/specs/2026-07-07-deimv2-obb-representation-refinement-design.md:206-266`; `engine/deim/obb_geometry.py:18-169`; `engine/deim/obb_geometry.py:218-519` existing self-test logic.
  Acceptance criteria (agent-executable): Add tests under `test/` (recommended `test/test_obb_adr_geometry.py`) covering: OBB round-trip for theta near `0`, `pi/2`, `pi-1e-3`, `1e-6`, `pi/4`; axis-aligned boxes; thin boxes; square-like boxes; 2000 random valid OBBs. Round-trip geometry must assert vertex error `< 1e-5`. Add tests for periodic seam expectation using a local helper in test first; these should fail to import production `periodic_angle_distance` until Todo 2.
  QA scenarios (name the exact tool + invocation): Happy: `pytest test/test_obb_adr_geometry.py -k "roundtrip or near_square" -v | tee .omo/evidence/task-1-deimv2-obb-adr-hybrid.txt`. Failure: temporarily set a seam expectation for ordinary theta L1 and confirm it demonstrates the current non-periodic problem; preserve the evidence in the same file.
  Commit: N | test(obb): lock ADR geometry and angle seam expectations

- [x] 2. Add shared periodic angle distance utility
  What to do / Must NOT do: Add one vectorized, differentiable utility in `engine/deim/obb_geometry.py`: `periodic_angle_distance(pred: Tensor, target: Tensor) -> Tensor`. It returns `min(abs(pred-target) % pi, pi - (abs(pred-target) % pi))` in radians, preserving broadcast semantics. Use this utility in tests. Must not duplicate the formula inline in criterion or matcher.
  Parallelization: Wave 2 | Blocked by: 1 | Blocks: 3, 8
  References: `docs/superpowers/specs/2026-07-07-deimv2-obb-representation-refinement-design.md:146-165`; `engine/deim/obb_geometry.py:13-15` import style; `engine/deim/deim_criterion.py:255-264`; `engine/deim/matcher.py:169-180`.
  Acceptance criteria: `periodic_angle_distance(torch.tensor([torch.pi - 0.01]), torch.tensor([0.01])) < 0.021`; ordinary normalized L1 for the same pair is `> 0.9`; output shape follows PyTorch broadcasting; gradient exists for non-boundary points.
  QA scenarios: Happy: `pytest test/test_obb_adr_geometry.py -k periodic -v | tee .omo/evidence/task-2-deimv2-obb-adr-hybrid.txt`. Failure: add an assertion that `periodic_angle_distance(0, pi)` is zero/tiny and verify the utility handles the seam rather than returning pi.
  Commit: N | feat(obb): add shared periodic angle distance

- [x] 3. Wire periodic angle distance into DEIMCriterion OBB loss
  What to do / Must NOT do: Modify `DEIMCriterion.__init__` to accept `lambda_angle: float = 1.0`. In `loss_boxes` OBB branch, compute spatial L1 over `src_boxes[..., :4]` and `target_boxes[..., :4]`, compute `periodic_angle_distance(src_boxes[..., 4:], target_boxes[..., 4:]) / torch.pi`, multiply the angle term by `lambda_angle`, concatenate/sum consistently with existing `loss_bbox` shape, and keep `loss_kld` unchanged. Update config `configs/custom_obb/synthetic_configs/synthetic_exp_020_dec.yml` under `DEIMCriterion` with `lambda_angle: 1.0`.
  Parallelization: Wave 2 | Blocked by: 1, 2 | Blocks: 8 | Can parallelize with: 4, 5
  References: `engine/deim/deim_criterion.py:43-59`; `engine/deim/deim_criterion.py:233-267`; `configs/custom_obb/synthetic_configs/synthetic_exp_020_dec.yml:258-290`; `docs/superpowers/specs/2026-07-07-deimv2-obb-representation-refinement-design.md:140-165`.
  Acceptance criteria: A unit test constructs matched `outputs/targets/indices` with theta seam pair and proves `loss_bbox` angle contribution is small near `0/pi`; `loss_kld` is still returned in OBB mode; HBB branch behavior remains unchanged.
  QA scenarios: Happy: `pytest test/test_obb_adr_geometry.py -k "periodic_loss or criterion" -v | tee .omo/evidence/task-3-deimv2-obb-adr-hybrid.txt`. Failure: set `lambda_angle=0.0` in a targeted test and confirm angle-only seam contribution becomes zero while spatial contribution remains.
  Commit: N | feat(criterion): use periodic OBB angle L1

- [x] 4. Wire periodic angle distance into HungarianMatcher OBB bbox cost
  What to do / Must NOT do: Modify `HungarianMatcher.__init__` to accept `lambda_angle: float = 1.0`. Replace OBB `torch.cdist(out_bbox * factor, tgt_bbox * factor, p=1)` with `torch.cdist(out_bbox[..., :4], tgt_bbox[..., :4], p=1) + lambda_angle * periodic_angle_distance(out_bbox[:, None, 4:], tgt_bbox[None, :, 4:]).squeeze(-1) / angle_factor`. Keep `cost_probiou`, `cost_chamfer`, and the `change_matcher` post-epoch branch unchanged because that branch already uses ProbIoU only. Update nested matcher config with `lambda_angle: 1.0`.
  Parallelization: Wave 2 | Blocked by: 1, 2 | Blocks: 8 | Can parallelize with: 3, 5
  References: `engine/deim/matcher.py:37-80`; `engine/deim/matcher.py:130-180`; `configs/custom_obb/synthetic_configs/synthetic_exp_020_dec.yml:273-288`; `docs/superpowers/specs/2026-07-07-deimv2-obb-representation-refinement-design.md:167-177`.
  Acceptance criteria: A matcher seam test with two predictions and one target proves periodic matcher chooses the geometry-consistent seam-side box; a non-periodic reference calculation in the test demonstrates the old cost would differ. `change_matcher=True` with `epoch >= matcher_change_epoch` still executes the ProbIoU-only branch without periodic angle cost.
  QA scenarios: Happy: `pytest test/test_obb_adr_geometry.py -k matcher_seam -v | tee .omo/evidence/task-4-deimv2-obb-adr-hybrid.txt`. Failure: force `lambda_angle=0.0` in the test and confirm the seam distinction no longer comes from angle cost.
  Commit: N | feat(matcher): use periodic OBB angle cost

- [x] 5. Resolve ADR offset scale source with shared configurable default
  What to do / Must NOT do: Add `offset_scale_source: str = "pre"` to the relevant ADR encode/decode utilities. Keep current behavior as default. `distance2bbox_obb` must use pre-adjustment external rect size when `pre`, post-adjustment size when `post`. `bbox2distance_obb` must use the same semantic default and document that `post` is an ablation mode. Add config key under `DEIMTransformer` or the narrowest config owner that can feed both decoder/criterion paths. Must not leave the existing TODO unresolved.
  Parallelization: Wave 2 | Blocked by: 1 | Blocks: 6, 8 | Can parallelize with: 2, 3, 4
  References: `engine/deim/dfine_utils.py:190-221`; `engine/deim/dfine_utils.py:224-291`; `engine/deim/deim_decoder.py:366-374`; `engine/deim/deim_criterion.py:293-317`; `configs/custom_obb/synthetic_configs/synthetic_exp_020_dec.yml:219-249`; `docs/superpowers/specs/2026-07-07-deimv2-obb-representation-refinement-design.md:127-138`.
  Acceptance criteria: Tests prove `offset_scale_source="pre"` preserves current decode/target behavior; `offset_scale_source="post"` is accepted and produces finite OBBs; decode/target inversion uses one consistent scale source without mixed semantics.
  QA scenarios: Happy: `pytest test/test_obb_adr_geometry.py -k "inversion or offset_scale" -v | tee .omo/evidence/task-5-deimv2-obb-adr-hybrid.txt`. Failure: intentionally mismatch encode/decode scale in a local test helper and prove inversion error exceeds the pinned tolerance, documenting why a shared setting matters.
  Commit: N | feat(obb): make ADR offset scale source explicit

- [x] 6. Add offset validity guards without gradient-destructive training clamps
  What to do / Must NOT do: Add a small validity helper for `(external_rect, vertex_offsets)` that clamps offsets into `[0, ext_w]` / `[0, ext_h]` only when explicitly requested. Use it for detached references/eval-safe paths, not for the loss-bearing `inter_ref_bbox` tensor. If the decoder reference path needs protection, clamp only tensors after `.detach()` before they become next-layer references. Do not add a soft penalty or sigmoid reparameterization in this iteration.
  Parallelization: Wave 3 | Blocked by: 1, 5 | Blocks: 7, 8
  References: `engine/deim/obb_geometry.py:115-169`; `engine/deim/deim_decoder.py:393-398`; `engine/deim/postprocessor.py:60-67`; `docs/superpowers/specs/2026-07-07-deimv2-obb-representation-refinement-design.md:179-204`.
  Acceptance criteria: Offset-validity tests cover negative epsilon/eta, epsilon greater than external width, eta greater than external height, and zero-size external rectangles. Guarded decode returns finite OBBs and valid clamped offsets; unguarded training decode remains available for gradient-bearing outputs.
  QA scenarios: Happy: `pytest test/test_obb_adr_geometry.py -k offset_validity -v | tee .omo/evidence/task-6-deimv2-obb-adr-hybrid.txt`. Failure: run the same invalid inputs through unguarded helper and confirm the guarded path is the one responsible for finite/clamped behavior.
  Commit: N | feat(obb): guard invalid ADR offsets safely

- [x] 7. Audit and minimally fix decouple-angle offset reference path
  What to do / Must NOT do: Inspect `engine/deim/dfine_decoder.py` MSDeformableAttention handling of `reference_points` dimensions, then audit `engine/deim/deim_decoder.py` decouple-angle branch. Decide from code evidence whether layer>0 `ref_offset_detach = inter_ref_bbox.detach()` passes scalar theta into a branch that expects spatial ADR offsets. If the attention path ignores dims beyond xywh, document that with tests and leave code unchanged. If it consumes the fifth/sixth dimensions semantically, convert next-layer offset references back to spatial ADR `(cx, cy, w, h, epsilon, eta)` or the exact representation expected by that attention implementation. Must not rename broad APIs or enable gated fusion.
  Parallelization: Wave 3 | Blocked by: 1, 6 | Blocks: 8
  References: `engine/deim/deim_decoder.py:269-279`; `engine/deim/deim_decoder.py:322-388`; `engine/deim/deim_decoder.py:393-398`; `engine/deim/deim_decoder.py:611-645`; `configs/custom_obb/synthetic_configs/synthetic_exp_020_dec.yml:247-249`; `docs/superpowers/specs/2026-07-07-deimv2-obb-representation-refinement-design.md:87-109`.
  Acceptance criteria: Add a focused test or smoke assertion that the `decouple_angle=True` forward path keeps reference-point dimensionality consistent across decoder layers and produces finite `pred_boxes` / `pred_corners`. Evidence must state whether the code was left unchanged due to attention semantics or changed to keep spatial ADR references.
  QA scenarios: Happy: `pytest test/test_deimv2_obb_smoke.py -k decouple_angle_reference -v | tee .omo/evidence/task-7-deimv2-obb-adr-hybrid.txt` or an equivalent targeted pytest using a minimal synthetic batch. Failure: inject a deliberately wrong reference dimension in a test helper and confirm the smoke test catches the mismatch or non-finite outputs.
  Commit: N | fix(decoder): keep decoupled OBB references consistent

- [x] 8. Run final targeted verification and diagnostics
  What to do / Must NOT do: Run all targeted tests and diagnostics after implementation. Use the supplied synthetic config as the reference for decouple-angle smoke coverage. Do not broaden into full training. Do not claim success from partial output; capture command evidence.
  Parallelization: Wave 4 | Blocked by: 2, 3, 4, 5, 6, 7 | Blocks: final verification
  References: `configs/custom_obb/synthetic_configs/synthetic_exp_020_dec.yml:1-313`; `docs/superpowers/specs/2026-07-07-deimv2-obb-representation-refinement-design.md:279-310`; all edited files from prior todos.
  Acceptance criteria: The following pass or produce documented environment blockers unrelated to code changes: `pytest test/test_obb_adr_geometry.py -v`; `pytest test/test_model_correctness.py test/test_obb_transforms.py -v`; `pytest test/test_deimv2_obb_smoke.py -k decouple_angle -v` if added. `lsp_diagnostics` reports zero errors on each edited Python file. `grep`/codegraph confirms `periodic_angle_distance` is called by both criterion and matcher.
  QA scenarios: Happy: run commands above and save combined output to `.omo/evidence/task-8-deimv2-obb-adr-hybrid.txt`. Failure: if smoke test cannot run due missing weights/data, replace it with synthetic-batch instantiation from config dimensions and record the original blocker plus synthetic fallback evidence.
  Commit: N | test(obb): verify ADR hybrid implementation

## Final verification wave

> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.

- [x] F1. Plan compliance audit
  Verify every Must Have is implemented and every Must NOT Have is untouched. Evidence: `.omo/evidence/f1-plan-compliance-deimv2-obb-adr-hybrid.txt`.
- [x] F2. Code quality review
  Review edited files for type suppressions, duplicated periodic-distance formulae, broad exception handling, oversized opportunistic refactors, and hidden config defaults. Evidence: `.omo/evidence/f2-code-quality-deimv2-obb-adr-hybrid.txt`.
- [x] F3. Real manual QA
  Agent-run real QA only: targeted pytest + synthetic decouple-angle smoke from Todo 8. No human visual confirmation. Evidence: `.omo/evidence/f3-real-qa-deimv2-obb-adr-hybrid.txt`.
- [x] F4. Scope fidelity
  Confirm no pure RiO branch, no gated fusion, no vertex-assignment replacement, no KLD/ProbIoU removal, and no unrelated model architecture changes. Evidence: `.omo/evidence/f4-scope-fidelity-deimv2-obb-adr-hybrid.txt`.

## Commit strategy

- Do not commit automatically. User manages git.
- If the user later asks for commits, use `/git-master` and split into logical commits:
  1. `test(obb): add ADR geometry and periodic seam coverage`
  2. `feat(obb): add periodic angle loss and matcher cost`
  3. `feat(obb): make ADR offset scale and validity explicit`
  4. `fix(decoder): verify decoupled OBB reference consistency`

## Success criteria

- `engine/deim/obb_geometry.py` exposes one shared periodic angle distance utility.
- `engine/deim/deim_criterion.py` and `engine/deim/matcher.py` both use the shared utility for OBB angle L1/cost.
- `lambda_angle: 1.0` and `offset_scale_source: pre` are present in the relevant config path and default behavior is documented.
- ADR geometry tests prove round-trip, near-square, seam, inversion, matcher, and offset-validity behavior.
- `decouple_angle=True` path is smoke-tested using the supplied synthetic config settings or an equivalent synthetic batch.
- KLD, ProbIoU, chamfer, FGL, DDF, current vertex assignment, and gated-fusion disabled status are preserved.
- No full 80-epoch run is required to call the implementation ready for review.
