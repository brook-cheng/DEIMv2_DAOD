# OBB Ablation Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain only rep0/rep3 with shifted decoder encoding, remove rejected OBB ablations, and reject semantically incompatible old OBB checkpoints without breaking HBB-to-OBB pretraining.

**Architecture:** Delete rejected behavior leaf-to-root in independently reviewable slices. Sisyphus writes and runs tests first; the user changes production/config/docs; Sisyphus reviews and verifies each batch. The physical 5D OBB API remains unchanged.

**Tech Stack:** Python, PyTorch, pytest, YAML configuration registry, CPU model smoke tests.

## Global Constraints

- User owns `engine/`, `configs/`, active documentation, and production tools.
- Sisyphus owns every modification under `test/`, production code review, and verification.
- Do not modify the public `(cx, cy, w, h, theta_rad)` contract or matcher/postprocessor/evaluation semantics.
- Keep rep0 6D ADR helpers and rep3 5D direct-angle helpers even where removed representations shared them.
- Keep criterion-side `physical_rad_to_norm` use for the retained non-periodic loss; remove proportional decoder selection only.
- Delete constructor keys and every active YAML occurrence in the same verified increment.
- Do not rewrite historical completed plans/specs.
- Do not commit unless the user explicitly requests it.

## Scenario Contract

| ID | Scenario | Binary pass condition | Automated evidence | Real-surface evidence |
|---|---|---|---|---|
| S1 | rep0 shifted forward | finite 5-channel `pred_boxes`, `0 <= theta < pi` | `test/test_obb_retained_representations.py::test_rep0_shifted_forward_contract` | CPU driver prints the rep0 shape and theta minimum/maximum followed by `PASS` |
| S2 | rep3 shifted forward | finite 5D output and finite per-layer references; no fusion keys | `test/test_obb_retained_representations.py::test_rep3_shifted_forward_contract` | CPU driver prints `rep3 PASS`, state-dict fusion-key count `0` |
| S3 | stale config rejection | retained configs construct; removed keys and reps are absent | `test/test_obb_config_contract.py` | config loader prints every retained config as `OK` |
| S4 | checkpoint compatibility | marked shifted OBB accepted; unmarked old OBB rejected; HBB tuning accepted | `test/test_obb_checkpoint_contract.py` | checkpoint probe prints three expected outcomes |
| S5 | adjacent public API | matcher, loss, postprocessor, and app tests retain 5D radian output | existing focused suites named in Task 11 | application/CPU smoke output remains 5D physical OBB |

## Task 1: Add retained-representation tests

**Files:**
- Create: `test/test_obb_retained_representations.py`
- Modify: `test/test_deimv2_obb_smoke.py`

**Interfaces:**
- Consumes current `DEIMTransformer` test factory patterns.
- Produces explicit rep0/rep3 shifted contracts used by every later task.

- [ ] Sisyphus extracts a small CPU model factory from existing smoke-test construction without changing production code.
- [ ] Add `test_rep0_shifted_forward_contract`: assert 5D finite public boxes and theta in `[0, pi)`.
- [ ] Add `test_rep3_shifted_forward_contract`: use `angle_rep=3`, `use_angle_first=False`, shifted encoding; assert finite 5D boxes and references.
- [ ] Run `pytest -q test/test_obb_retained_representations.py`; the two retained-forward tests must pass on the current implementation and establish characterization evidence.

## Task 2: Add configuration-contract tests

**Files:**
- Create: `test/test_obb_config_contract.py`
- Modify: `test/deim_app/test_legacy_parity.py`

**Interfaces:**
- Consumes the repository YAML loader and registered constructor schemas.
- Produces a manifest of stale keys, rejected reps, and broken include chains.

- [ ] Add a recursive inventory over active `configs/custom_obb/**/*.yml` and application OBB presets.
- [ ] Resolve `__include__` through the existing YAML loader rather than reimplementing merge behavior.
- [ ] For registered sections, compare resolved keys with constructor arguments and report `path:section.key`.
- [ ] Assert active OBB configs contain no `angle_rep` values other than `0` or `3` after cleanup.
- [ ] Assert removed keys are absent: `offset_scale_source`, `use_gate_fusion`, `angle_step`, `use_angle_first`, `decoder_angle_encoding`.
- [ ] Update legacy-parity expectations so fixed behavior is not required to remain as YAML keys.
- [ ] Run the new tests before production cleanup and record the expected stale-key failures.

## Task 3: Remove offset-post behavior

**Files:**
- User modify: `engine/deim/dfine_utils.py`, `engine/deim/deim_criterion.py`, `engine/deim/deim_decoder.py`
- User modify/delete: baseline/preset offset keys, `abl_offset_post.yml`, synthetic offset-post configs
- Sisyphus modify: `test/test_obb_adr_geometry.py`

- [ ] Sisyphus adds signature assertions showing offset source is no longer configurable, then records RED.
- [ ] User removes `offset_scale_source` parameters and validation; retain only pre-adjustment geometry.
- [ ] User removes corresponding YAML keys and rejected post configs in the same batch.
- [ ] Sisyphus removes only post/mismatch tests; retain pre, inverse, clamp, degenerate, and stable-atan2 tests.
- [ ] Run `pytest -q test/test_obb_adr_geometry.py test/test_obb_config_contract.py` and review the diff.

## Task 4: Remove multi-angle anchors

**Files:**
- User modify: `engine/deim/deim_decoder.py`
- User delete/modify: `abl_mangle.yml`, synthetic multi-angle config, baseline/preset `angle_step`
- Sisyphus modify: `test/test_deimv2_obb_smoke.py`

- [ ] Sisyphus adds a constructor-signature assertion for removed `angle_step`, then records RED.
- [ ] User removes candidate expansion, memory repetition, constructor state, and YAML key.
- [ ] Sisyphus removes the two tests whose sole contract is multi-angle generation.
- [ ] Run focused anchor, retained-representation, and config-contract tests.

## Task 5: Remove angle-first and gate fusion

**Files:**
- User modify: `engine/deim/deim_decoder.py`
- User delete: `engine/deim/gated_fusion.py`
- User delete/modify: AFP/fused configs and baseline/preset keys
- Sisyphus modify: `test/test_deimv2_obb_smoke.py`, `test/test_obb_retained_representations.py`

- [ ] Sisyphus changes rep3 matrices to `use_angle_first=False` and records the constructor/state-dict RED tests.
- [ ] Sisyphus adds `test_rep3_has_no_gate_fusion_state_after_cleanup`; run it alone and record RED because current rep3 state dicts contain `gate_fusions` keys.
- [ ] User removes angle-first query construction, special first-layer flow, fusion construction/calls, and both constructor parameters.
- [ ] Preserve `decouple_angle_layers`; rep3 depends on it.
- [ ] Delete AFP/fused configs and `gated_fusion.py` only after `rg -n 'GatedSoftmaxFusion|gate_fusions' engine` shows no remaining caller.
- [ ] Run retained-representation tests; state dict must contain zero `gate_fusions` keys.

## Task 6: Remove rep1

**Files:**
- User modify: `engine/deim/deim_decoder.py`
- User delete: `abl_rep1.yml`, rep1 synthetic bases and all include dependents
- Sisyphus modify: `test/test_deimv2_obb_smoke.py`, `test/test_exp_020_obb_compare.py`, affected analysis tools under `test/`

- [ ] Sisyphus adds `angle_rep=1` construction rejection and changes retained matrices to `{0,3}`.
- [ ] Repoint the misleading `angle_rep=True` characterization to explicit rep3 where its real purpose is retained ADR/reference behavior.
- [ ] User removes rep1 head-dimension and representation branches.
- [ ] Remove/migrate rep1 include dependents before deleting their base config.
- [ ] Run smoke, comparison, and config-contract tests.

## Task 7: Remove rep2 and its diagnostics

**Files:**
- User modify: `engine/deim/deim_decoder.py`, `engine/deim/dfine_decoder.py`
- User delete: rep2 configs and synthetic dependents
- Sisyphus delete: `test/test_deimv2_obb_rep2_eval.py`, `test/test_rep2_nan_diagnostic.py`, `test/test_rep2_nan_failure_replay.py`, `test/tool_diagnose_rep2_nan.py`, `test/tool_replay_rep2_nan_failure.py`
- Sisyphus modify: rep2 entries in inference/debug/comparison tools under `test/`

- [ ] Sisyphus adds explicit rep2 construction rejection and records RED.
- [ ] User removes rep2 denoising conversion, 6D reference/head/anchor/auxiliary branches, and 6D attention conversion.
- [ ] Sisyphus reviews rep2-only files before deleting them; retain shared geometry and stable-atan2 coverage.
- [ ] Remove/migrate rep2 include dependents before their base configs.
- [ ] Run retained representation, ADR geometry, smoke, and config-contract suites.

## Task 8: Make shifted encoding unconditional

**Files:**
- User modify: `engine/deim/deim_decoder.py`, `engine/deim/denoising.py`, `engine/deim/dfine_decoder.py`
- User modify/delete: baseline/preset shifted keys and now-redundant `abl_shifted.yml`
- Sisyphus modify: `test/test_obb_angle_contract.py`, `test/test_deimv2_obb_smoke.py`, retained-representation tests

- [ ] Sisyphus adds a signature test proving `decoder_angle_encoding` is no longer accepted and records RED.
- [ ] User keeps only shifted conversion at decoder refinement, anchors, denoising, attention, encoder auxiliary output, and public conversion sites.
- [ ] User removes encoding parameters, validation, propagation, and proportional branches.
- [ ] Do not delete criterion-side `physical_rad_to_norm`/`norm_to_physical_rad` helpers.
- [ ] Delete `abl_shifted.yml` because shifted is now baseline behavior rather than an ablation.
- [ ] Run angle-contract, denoising, retained-representation, smoke, criterion-loss, and config-contract tests.

## Task 9: Simplify representation guards

**Files:**
- User modify: `engine/deim/deim_decoder.py`, related comments/docs

- [ ] Replace surviving multi-representation predicates with explicit rep0/rep3 branches.
- [ ] Add a constructor `ValueError` listing the accepted set `{0, 3}`.
- [ ] Remove comments describing rep1/rep2 or removed switches from active source/config docs.
- [ ] Sisyphus reviews every changed branch against rep0/rep3 contracts and reruns focused tests.

## Task 10: Add OBB checkpoint compatibility contract

**Files:**
- User modify: `engine/solver/_solver.py`
- User modify: `tools/inference/torch_inf.py`, `tools/inference/torch_inf_vis.py`, `tools/visualization/fiftyone_vis.py`, `tools/deployment/export_onnx.py`
- Sisyphus create: `test/test_obb_checkpoint_contract.py`

**Interfaces:**
- Produce `OBB_ANGLE_CONTRACT = "shifted_v1"`.
- Produce a typed compatibility error and a helper that classifies marked OBB, legacy OBB, and HBB-pretraining state dictionaries.

- [ ] Sisyphus writes RED tests: marked shifted OBB accepted; wrong marker rejected; unmarked 5D/6D OBB rejected; identifiable 4D HBB allowed only for tuning.
- [ ] User adds `meta.obb_angle_contract` when saving OBB solver state.
- [ ] User checks the marker on OBB resume and inference before `load_state_dict`.
- [ ] User checks tuning inputs: permit a marked shifted OBB checkpoint or an unmarked 4D HBB checkpoint; reject ambiguous old OBB checkpoints before the non-strict `_matched_state` load.
- [ ] Ensure HBB resume/inference behavior is unchanged.
- [ ] Run checkpoint tests plus one in-memory state round trip.

## Task 11: Repair stale tests and complete configuration cleanup

**Files:**
- Sisyphus modify: `test/test_kendall.py`, `test/test_model_correctness.py`, `test/test_model_output.py`, `test/test_obb_loss_integration.py`, `test/test_obb_transforms.py`, `test/test_early_stopping_configs.py`
- User modify/delete: remaining active OBB configs found by Task 2

- [ ] Replace references to deleted `configs/custom_obb/deimv2_obb_sp.yml` with the retained authoritative config appropriate to each test.
- [ ] Resolve the missing `sp_fz_rep0_nloss_amp.yml` expectation without inventing a compatibility alias.
- [ ] Delete or migrate `synthetic_exp_020_dec.yml` because `angle_rep: True` is rep1.
- [ ] Run `pytest -q test/test_obb_config_contract.py` until every retained config parses and constructs without stale keys.
- [ ] Run adjacent matcher, criterion, postprocessor, transforms, application, and CLI tests.

## Task 12: Review and full verification

**Files:** all changed files.

- [ ] Sisyphus reads every production/config diff and checks it against the approved design.
- [ ] Run diagnostics on every changed Python file.
- [ ] Run focused suites first, then `pytest -q test` (or the repository's documented complete test command).
- [ ] Run a CPU driver for S1/S2 and capture shapes, theta min/max, finiteness, and fusion-key count.
- [ ] Load and construct every retained OBB configuration; capture each `OK <path>` line.
- [ ] Run the checkpoint probe for marked shifted OBB, legacy OBB rejection, and HBB tuning acceptance.
- [ ] Run `rg -n 'use_gate_fusion|use_angle_first|angle_step|offset_scale_source|decoder_angle_encoding|GatedSoftmaxFusion' engine configs/custom_obb` and classify every residual match; active implementation/config matches are failures.
- [ ] Run `git diff --check` without staging or committing.
- [ ] Invoke the post-implementation review workflow before handoff because this cleanup changes more than three files and removes checkpoint-compatible state.

## Execution Order

Tasks execute strictly in order: `1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12`. Test work always precedes the production change it specifies. Constructor parameter deletion and corresponding active YAML cleanup stay in the same verified increment.

If commits are later requested, use one atomic commit per Tasks 3-10, with tests and production changes together. Study repository history before composing messages; do not create commits during planning.

---

# Appendix A: Production-Code Reference Operations (Tasks 3-10, user side)

> Line numbers are from the 2026-08-13 snapshot. Because edits shift line numbers, **locate each hunk by content first (`rg -n` or search for the anchor lines), then apply the before/after pairs below**. Each target snippet already accounts for later tasks.

## A3 Task 3: Remove offset-post

### A3.1 `engine/deim/dfine_utils.py` — `distance2bbox_obb` (lines 194-250)

Before (line 195):

```python
def distance2bbox_obb(
    points, distance, reg_scale, offset_scale_source: str = "pre", layer_idx=0
):
```

After:

```python
def distance2bbox_obb(points, distance, reg_scale, layer_idx=0):
```

Delete the validation block (lines 214-218):

```python
    if offset_scale_source not in ("pre", "post"):
        raise ValueError(
            f"offset_scale_source must be 'pre' or 'post', "
            f"got {offset_scale_source!r}"
        )
```

Before, offset scale selection (lines 229-235):

```python
        offset_scale_wh = (
            ext_rect_cxcywh[..., 2:]
            if offset_scale_source == "pre"
            else ext_adj_cxcywh[..., 2:]
        )
```

After (fixed to pre):

```python
        offset_scale_wh = ext_rect_cxcywh[..., 2:]
```

### A3.2 `engine/deim/dfine_utils.py` — `bbox2distance_obb` (lines 253-316)

Remove `offset_scale_source: str = "pre",` from the signature (line 260); after:

```python
def bbox2distance_obb(
    points,
    bbox,
    reg_max,
    reg_scale,
    up,
    eps=0.1,
    obbox_rep_dim=6,
):
```

Delete the identical validation block (lines 284-288).

Before, offset scale selection (lines 300-304):

```python
        offset_scale_wh = (
            rect_cxcywh_pred[..., 2:]
            if offset_scale_source == "pre"
            else rect_cxcywh_gt[..., 2:]
        )
```

After (fixed to pre):

```python
        offset_scale_wh = rect_cxcywh_pred[..., 2:]
```

### A3.3 `engine/deim/deim_criterion.py`

- Line 67 signature: remove `offset_scale_source="pre",`.
- Line 105: remove `self.offset_scale_source = offset_scale_source`.
- Both `bbox2distance_obb(...)` calls (lines 426-434 and 445-453): remove the `offset_scale_source=self.offset_scale_source,` argument line.

### A3.4 `engine/deim/deim_decoder.py`

- `TransformerDecoder.__init__`: remove `offset_scale_source="pre",` (line 184) and `self.offset_scale_source = offset_scale_source` (line 198).
- `DEIMTransformer.__init__`: remove `offset_scale_source="pre",` (line 603) and `self.offset_scale_source = offset_scale_source` (line 631).
- Line 728: remove `offset_scale_source=self.offset_scale_source,`.
- Both `distance2bbox_obb(...)` calls in the decode loop (509-514, 529-534): remove the trailing `offset_scale_source=self.offset_scale_source,` line. Before (lines 509-514):

```python
                    inter_ref_bbox = distance2bbox_obb(
                        ref_phys,
                        distance,
                        reg_scale,
                        offset_scale_source=self.offset_scale_source,
                    )
```

After:

```python
                    inter_ref_bbox = distance2bbox_obb(
                        ref_phys,
                        distance,
                        reg_scale,
                    )
```

Apply the same removal to lines 529-534 (the proportional branch, removed wholesale in Task 8).

### A3.5 YAML

Remove the `offset_scale_source` key from retained configs:

- `configs/custom_obb/dlzdt/sp_fz_common.yml:193` (DEIMTransformer section) and `:272` (DEIMCriterion section).
- `configs/app/presets/deimv2_dinov3_sp_obb.yml:64` and `:104`.
- `configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep0_offset_per.yml:259,305`, `synthetic_exp_020_anrep3_offset_per.yml:258,303`, `synthetic_exp_020_undec_offset_per.yml:257,302`.

Delete configs: `dlzdt/ablation/abl_offset_post.yml`, `synthetic_exp_020_undec_offset_post.yml`, `synthetic_exp_020_anrep2_offset_post.yml` (the anrep2 config nominally belongs to Task 7; delete it here to avoid a residual post key).

**Check:** `rg -n 'offset_scale_source' engine configs` leaves only comments/history; the retained `test_obb_adr_geometry.py` pre/inverse/clamp/degenerate/stable-atan2 tests pass.

## A4 Task 4: Remove multi-angle anchors

### A4.1 `engine/deim/deim_decoder.py` — constructor

- Line 605: remove `angle_step=0.0,`.
- Line 633: remove `self.angle_step = angle_step`.

### A4.2 `_generate_anchors` OBB branch (lines 1039-1105)

Before:

```python
        elif self.box_mode == "obb":
            if self.angle_rep == 2:
                ...  # rep2 6D anchors (deleted in Task 7; untouched here)
            else:
                for lvl, (h, w) in enumerate(spatial_shapes):
                    ...
                    if self.angle_step > 0:
                        n_angles = int(1.0 / self.angle_step)
                        ...  # 1075-1091 multi-angle candidate expansion
                    else:
                        default_r = (
                            0.5 if self.decoder_angle_encoding == "shifted" else 0.25
                        )
                        r = default_r * torch.ones(...)
                        lvl_anchors = torch.concat([grid_xy, wh, r], dim=-1).reshape(
                            -1, h * w, self._num_box_dof
                        )
```

After (delete the `if self.angle_step > 0:` branch and promote the else body; the `default_r` ternary becomes `0.5` in Task 8):

```python
        elif self.box_mode == "obb":
            if self.angle_rep == 2:
                ...  # unchanged
            else:
                for lvl, (h, w) in enumerate(spatial_shapes):
                    ...
                    default_r = (
                        0.5 if self.decoder_angle_encoding == "shifted" else 0.25
                    )
                    r = default_r * torch.ones(
                        *grid_xy.shape[:-1],
                        1,
                        dtype=grid_xy.dtype,
                        device=grid_xy.device,
                    )
                    lvl_anchors = torch.concat([grid_xy, wh, r], dim=-1).reshape(
                        -1, h * w, self._num_box_dof
                    )
```

### A4.3 `_get_decoder_input` (lines 1173-1180)

Delete the whole multi-angle memory replication block:

```python
        # Multi-angle anchors (angle_step > 0) expand the candidate pool to
        # (num_spatial_positions * n_angles). Replicate each spatial memory
        # token n_angles times so it aligns with the anchor layout
        # (position-major, angle-minor within each level). This only affects
        # query selection; the decoder still receives the original memory.
        if self.box_mode == "obb" and self.angle_rep != 2 and self.angle_step > 0:
            n_angles = int(1.0 / self.angle_step)
            memory = memory.repeat_interleave(n_angles, dim=1)
```

### A4.4 YAML

Remove the `angle_step` key: `sp_fz_common.yml:203`, `app/presets/deimv2_dinov3_sp_obb.yml:66`. Delete configs: `dlzdt/ablation/abl_mangle.yml`, `synthetic_configs/ablation/syn_ablation_mangle.yml`.

## A5 Task 5: Remove angle-first and gate fusion

### A5.1 `TransformerDecoder.__init__` (lines 168-239)

- Line 185: remove `use_gate_fusion=False,`; line 186: remove `use_angle_first=False,`.
- Lines 199-200: remove the two attribute stores.
- Lines 221-239, before:

```python
        if self.angle_rep != 0 and self.angle_rep != 1:
            from .gated_fusion import GatedSoftmaxFusion

            decouple_layer_template = self.layers[-1]
            self.decouple_angle_layers = nn.ModuleList(
                [
                    copy.deepcopy(decouple_layer_template)
                    for _ in range(self.num_decouple_layers)
                ]
            )

            self.gate_fusions = nn.ModuleList(
                [
                    GatedSoftmaxFusion(
                        d_model=hidden_dim, n_sources=2, hidden_dim=hidden_dim
                    )
                    for _ in range(self.num_decouple_layers - 1)
                ]
            )
```

After (keep `decouple_angle_layers`, delete the lazy import and `gate_fusions`; the guard becomes `== 3` in Task 9):

```python
        if self.angle_rep != 0 and self.angle_rep != 1:
            decouple_layer_template = self.layers[-1]
            self.decouple_angle_layers = nn.ModuleList(
                [
                    copy.deepcopy(decouple_layer_template)
                    for _ in range(self.num_decouple_layers)
                ]
            )
```

### A5.2 `TransformerDecoder.forward` query setup (lines 304-331)

Before, lines 305-328 contain a `use_angle_first` three-way split. After (remove the conditional at 319-328 and keep the else content):

```python
        if self.box_mode == "obb":
            if self.angle_rep == 0 or self.angle_rep == 1:
                ref_points_detach = F.sigmoid(ref_points_unact)
                query_pos_embed = query_pos_head(ref_points_detach).clamp(
                    min=-10, max=10
                )
            elif self.angle_rep == 2 or self.angle_rep == 3:
                dec_angle_output = target
                dec_angle_output_detach = 0
                dec_angle_pred_corners_undetach = 0
                ref_points_detach = F.sigmoid(ref_points_unact[..., :4])
                ref_dec_angle_detach = F.sigmoid(ref_points_unact)
                query_dec_angle_embed = query_angle_head(
                    F.sigmoid(ref_points_unact[..., 4:])
                )
                query_pos_embed = query_pos_head(ref_points_detach).clamp(
                    min=-10, max=10
                )
```

### A5.3 Angle-first block inside the layer loop (lines 335-377)

Delete the whole `if (self.use_angle_first and ...)` block (335-375) and keep the else body unconditionally:

```python
            ref_points_input = ref_points_detach.unsqueeze(2)
```

Note: the deleted block only contained the angle-first *call* of `decouple_angle_layers`; the standard path keeps its call in A5.4, and the module itself (A5.1) is unaffected.

### A5.4 OBB refinement inside the layer loop (lines 420-492)

Before: `if self.angle_rep == 2 or self.angle_rep == 3:` wrapping `if self.use_angle_first:` (422-442) and `else:` (443-492). After: delete the `if self.use_angle_first:` branch (including the rep2 sub-branch 424-434 and rep3 sub-branch 435-438), promote the else body, and delete the fusion call at 484-487:

```python
            if self.box_mode == "obb":
                if self.angle_rep == 2 or self.angle_rep == 3:
                    ref_dec_angle_input = ref_dec_angle_detach.unsqueeze(2)
                    dec_angle_output = self.decouple_angle_layers[layer_idx](
                        dec_angle_output,
                        ref_dec_angle_input,
                        value,
                        spatial_shapes,
                        attn_mask,
                        query_dec_angle_embed,
                    )
                    if layer_idx == 0:
                        dec_angle_initial = torch.sigmoid(
                            pre_angle_head(dec_angle_output)
                            + inverse_sigmoid(ref_dec_angle_detach)[..., 4:]
                        )
                        if self.angle_rep == 2:
                            ...  # rep2 sub-branch (deleted in Task 7; keep for now)
                        elif self.angle_rep == 3:
                            pre_bboxes = torch.concat(
                                [pre_bboxes, dec_angle_initial], dim=-1
                            )
                        ref_points_initial = pre_bboxes.detach()
                    dec_angle_pred_corners = (
                        dec_angle_head[layer_idx](
                            dec_angle_output + dec_angle_output_detach
                        )
                        + dec_angle_pred_corners_undetach
                    )
                    dec_angle_output_detach = dec_angle_output.detach()
                    dec_angle_pred_corners_undetach = dec_angle_pred_corners
                    pred_corners = torch.concat(
                        [pred_corners, dec_angle_pred_corners], dim=-1
                    )
```

Deleted fusion call (lines 484-487):

```python
                        if self.use_gate_fusion and layer_idx < len(self.gate_fusions):
                            dec_angle_output = self.gate_fusions[layer_idx](
                                [output, dec_angle_output], query=dec_angle_output
                            )
```

### A5.5 Constructor guard (lines 645-650)

Delete the whole `if use_angle_first and angle_rep == 2: raise ValueError(...)` block.

### A5.6 Head dims (lines 784-793)

Before (rep3 branch, lines 789-793):

```python
            elif self.angle_rep == 3:
                pre_bbox_head_out_dim = 4
                num_query_pos_in = 5 if self.use_angle_first else 4
                num_reg_dist_xywh = 4
                num_angle_describer = 1
```

After: `num_query_pos_in = 4` (everything else unchanged).

### A5.7 Delete `engine/deim/gated_fusion.py`

Precondition: `rg -n 'GatedSoftmaxFusion|gate_fusions' engine` returns nothing.

### A5.8 YAML

Remove `use_gate_fusion` (`sp_fz_common.yml:198`, `app/presets:65`) and `use_angle_first` (`sp_fz_common.yml:208`, `app/presets:67`). Delete configs: `dlzdt/ablation/abl_rep2_fused.yml`, `abl_rep3_fused.yml`, `abl_rep3_afp.yml`, `synthetic_configs/ablation/syn_ablation_fused.yml`, `syn_ablation_afp.yml`.

**Check:** `test_rep3_has_no_gate_fusion_state_after_cleanup` goes green; rep3 state dict has zero `gate_fusions` keys.

## A6 Task 6: Remove rep1

### A6.1 dof/reg_dist branches (lines 654-666)

Before:

```python
        if self.box_mode == "obb":
            if self.angle_rep == 0:
                self._num_box_dof = 5  # (cx,cy,w,h,θ)
                self.num_reg_dist = 6  # (α,β,γ,δ,ε,η)
            elif self.angle_rep == 1:
                self._num_box_dof = 5  # (cx,cy,w,h,θ)
                self.num_reg_dist = 5  # (α,β,γ,δ,deta_θ)
            elif self.angle_rep == 2:
                self._num_box_dof = 6  # (cx,cy,w,h,ε,η)
                self.num_reg_dist = 6  # (α,β,γ,δ,ε,η)
            elif self.angle_rep == 3:
                self._num_box_dof = 5  # (cx,cy,w,h,θ)
                self.num_reg_dist = 5  # (α,β,γ,δ,deta_θ)
```

After (delete the rep1 branch at 658-660; the rep2 branch goes in Task 7):

```python
        if self.box_mode == "obb":
            if self.angle_rep == 0:
                self._num_box_dof = 5  # (cx,cy,w,h,θ)
                self.num_reg_dist = 6  # (α,β,γ,δ,ε,η)
            elif self.angle_rep == 2:
                self._num_box_dof = 6  # (cx,cy,w,h,ε,η)
                self.num_reg_dist = 6  # (α,β,γ,δ,ε,η)
            elif self.angle_rep == 3:
                self._num_box_dof = 5  # (cx,cy,w,h,θ)
                self.num_reg_dist = 5  # (α,β,γ,δ,deta_θ)
```

### A6.2 Head dims (lines 775-793)

Delete the rep1 branch (lines 780-783):

```python
            elif self.angle_rep == 1:
                pre_bbox_head_out_dim = 5  # (cx,cy,w,h,θ)
                num_query_pos_in = 5
                num_reg_dist_xywh = 5  # (α,β,γ,δ,deta_theta)
```

### A6.3 YAML

Delete `dlzdt/ablation/abl_rep1.yml`, `synthetic_exp_020_anrep1_offset_per.yml`, and its include dependents (dependents first):

- `synthetic_exp_020_loss_kld.yml` (includes anrep1_offset_per)
- `synthetic_exp_020_loss_prob_kld.yml`
- `synthetic_exp_020_loss_prob_angle_kld.yml`
- `synthetic_configs/ablation/syn_ablation_loss_prob_kld.yml`
- `synthetic_configs/provenance/synthetic_exp_020_loss_kld.completed.yml`
- `synthetic_configs/provenance/synthetic_exp_020_loss_prob_kld.completed.yml`

## A7 Task 7: Remove rep2 and its diagnostics

### A7.1 Delete helper (lines 47-62)

Delete the whole `_obb_denoising_unact_to_rep2_unact` function.

### A7.2 Branch convergence

- Lines 654-666: delete the rep2 branch (661-663).
- Lines 775-793: delete the rep2 branch (784-788, including `num_angle_describer = 2`).
- Line 310: `elif self.angle_rep == 2 or self.angle_rep == 3:` → `elif self.angle_rep == 3:`.
- Lines 390, 421: `== 2 or == 3` → `== 3`.
- In the A5.4 target, delete the rep2 sub-branch, keeping only the rep3 case:

```python
                    if layer_idx == 0:
                        dec_angle_initial = torch.sigmoid(
                            pre_angle_head(dec_angle_output)
                            + inverse_sigmoid(ref_dec_angle_detach)[..., 4:]
                        )
                        if self.angle_rep == 3:
                            pre_bboxes = torch.concat(
                                [pre_bboxes, dec_angle_initial], dim=-1
                            )
                        ref_points_initial = pre_bboxes.detach()
```

### A7.3 Anchor generation (lines 1039-1063)

Delete the `if self.angle_rep == 2:` block inside the OBB branch (1040-1063); the else body becomes the direct OBB branch content (i.e., the A4.2 target without the outer if).

### A7.4 Encoder auxiliary angle conversion (lines 1198-1220)

Delete the outer `if self.angle_rep != 2:` guard and the `else: enc_topk_bboxes = enc_topk_bboxes` (1218-1220); promote the inner encoding conditional (Task 8 then removes its proportional branch).

### A7.5 Denoising concat (lines 1232-1241)

Before:

```python
        if denoising_bbox_unact is not None:
            if self.angle_rep != 2:
                enc_topk_bbox_unact = torch.concat(
                    [denoising_bbox_unact, enc_topk_bbox_unact], dim=1
                )
            else:
                dn_bbox_unact = _obb_denoising_unact_to_rep2_unact(denoising_bbox_unact)
                enc_topk_bbox_unact = torch.concat(
                    [dn_bbox_unact, enc_topk_bbox_unact], dim=1
                )
            content = torch.concat([denoising_logits, content], dim=1)
```

After:

```python
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat(
                [denoising_bbox_unact, enc_topk_bbox_unact], dim=1
            )
            content = torch.concat([denoising_logits, content], dim=1)
```

### A7.6 Encoder auxiliary output conversion (lines 1415-1422)

Delete the whole block:

```python
            if self.angle_rep == 2:
                enc_topk_bboxes_list = [
                    external_xywh_rect_to_oriented_box(
                        enc_topk_bboxes[..., :4],
                        enc_topk_bboxes[..., 4:],
                    )
                    for enc_topk_bboxes in enc_topk_bboxes_list
                ]
```

### A7.7 `engine/deim/dfine_decoder.py` attention 6D branch (lines 173-187)

Before:

```python
        elif reference_points.shape[-1] == 5 or reference_points.shape[-1] == 6:
            if reference_points.shape[-1] == 6:
                reference_points = external_xywh_rect_to_oriented_box(
                    reference_points[..., :4], reference_points[..., 4:]
                )
                angle = reference_points[..., 4:5]
            else:
                if self.angle_encoding == "shifted":
                    angle = shifted_norm_to_physical_rad(reference_points[..., 4:5])
                else:
                    angle = reference_points[..., 4:5] * torch.pi
```

After:

```python
        elif reference_points.shape[-1] == 5:
            if self.angle_encoding == "shifted":
                angle = shifted_norm_to_physical_rad(reference_points[..., 4:5])
            else:
                angle = reference_points[..., 4:5] * torch.pi
```

If line 26 `from .obb_geometry import external_xywh_rect_to_oriented_box` has no other caller in this file, remove the import too.

### A7.8 YAML

Delete `dlzdt/ablation/abl_rep2.yml`, `synthetic_exp_020_anrep2_offset_per.yml`, `anrep2_bn0.yml`, `anrep2_dn0.yml`, `synthetic_configs/ablation/syn_ablation_noangle.yml` (`anrep2_offset_post.yml` was already deleted in A3.5).

## A8 Task 8: Make shifted encoding unconditional

### A8.1 Decode loop (lines 498-538)

Before: `if self.decoder_angle_encoding == "shifted":` / `else:` split. After (keep only the shifted body, delete the else at 523-538):

```python
            elif self.box_mode == "obb":
                # [0,1)→[-pi/4,3*pi/4)
                ref_phys = torch.cat(
                    [
                        ref_points_initial[..., :4],
                        shifted_norm_to_physical_rad(ref_points_initial[..., 4:5]),
                    ],
                    dim=-1,
                )
                distance = integral(pred_corners, project)
                inter_ref_bbox = distance2bbox_obb(
                    ref_phys,
                    distance,
                    reg_scale,
                )
                # [-pi/4,3*pi/4)→[0,1)
                inter_ref_bbox = torch.cat(
                    [
                        inter_ref_bbox[..., :4],
                        physical_rad_to_shifted_norm(inter_ref_bbox[..., 4:]),
                    ],
                    dim=-1,
                )
```

### A8.2 Anchor default_r (lines 1093-1095)

`default_r = (0.5 if self.decoder_angle_encoding == "shifted" else 0.25)` → `default_r = 0.5`.

### A8.3 Encoder auxiliary conversion (lines 1199-1217, inner conditional after Task 7)

Before:

```python
                if self.decoder_angle_encoding == "shifted":
                    # 内部 θ_shift 还原为物理角 [0, π)
                    enc_topk_bboxes = torch.cat(
                        [
                            enc_topk_bboxes[..., :4],
                            shifted_norm_to_physical_rad(enc_topk_bboxes[..., 4:]),
                        ],
                        dim=-1,
                    )
                else:
                    # 角度量纲 [0,1]->[0, pi)
                    enc_topk_bboxes = torch.cat(
                        [
                            enc_topk_bboxes[..., :4],
                            norm_to_physical_rad(enc_topk_bboxes[..., 4:]),
                        ],
                        dim=-1,
                    )
```

After (delete the else branch):

```python
                # 内部 θ_shift 还原为物理角 [0, π)
                enc_topk_bboxes = torch.cat(
                    [
                        enc_topk_bboxes[..., :4],
                        shifted_norm_to_physical_rad(enc_topk_bboxes[..., 4:]),
                    ],
                    dim=-1,
                )
```

### A8.4 Public output conversion (lines 1346-1350)

Before:

```python
        if self.box_mode == "obb":
            if self.decoder_angle_encoding == "shifted":
                theta_decode = shifted_norm_to_physical_rad
            else:
                theta_decode = norm_to_physical_rad
```

After:

```python
        if self.box_mode == "obb":
            theta_decode = shifted_norm_to_physical_rad
```

(The `theta_decode(out_bboxes[..., 4:])` applications at 1351-1371 stay unchanged.)

### A8.5 Parameter and switch removal (`engine/deim/deim_decoder.py`)

- Line 44: delete `_VALID_DECODER_ANGLE_ENCODINGS = ("proportional", "shifted")`.
- `TransformerDecoderLayer.__init__`: remove `angle_encoding="proportional",` (line 79); lines 96-103 become:

```python
        self.cross_attn = MSDeformableAttention(
            d_model,
            n_head,
            n_levels,
            n_points,
            method=cross_attn_method,
        )
```

- `TransformerDecoder.__init__`: remove `angle_encoding="proportional",` (line 187) and `self.decoder_angle_encoding = angle_encoding` (line 202).
- `DEIMTransformer.__init__`: remove `decoder_angle_encoding="proportional",` (line 607); delete lines 635-642 wholesale (validation 635-639, rep2-forcing ternary 640-642).
- Lines 696, 710: remove `angle_encoding=self.decoder_angle_encoding,`.
- Line 731: remove `angle_encoding=self.decoder_angle_encoding,`.
- Line 1301: remove `angle_encoding=self.decoder_angle_encoding,`.

### A8.6 `engine/deim/denoising.py` (lines 21, 113-121)

- Line 21: remove `angle_encoding="proportional",`.
- Before, lines 113-121:

```python
    elif box_mode == "obb":
        # [0,pi) → decoder 私有编码 [0,1)
        if angle_encoding == "shifted":
            input_query_bbox[..., 4] = physical_rad_to_shifted_norm(
                input_query_bbox[..., 4]
            )
        else:
            input_query_bbox[..., 4] = physical_rad_to_norm(input_query_bbox[..., 4])
        input_query_bbox = torch.cat([noise_spatial, input_query_bbox[..., 4:]], dim=-1)
```

After:

```python
    elif box_mode == "obb":
        # [0,pi) → decoder 私有 shifted 编码 [0,1)
        input_query_bbox[..., 4] = physical_rad_to_shifted_norm(
            input_query_bbox[..., 4]
        )
        input_query_bbox = torch.cat([noise_spatial, input_query_bbox[..., 4:]], dim=-1)
```

- Line 9 import: if `physical_rad_to_norm` is no longer used in this file, drop it from the import (keep `physical_rad_to_shifted_norm`).

### A8.7 `engine/deim/dfine_decoder.py`

- Line 58: remove `angle_encoding="proportional",`.
- Line 100: remove `self.angle_encoding = angle_encoding`.
- Lines 184-187 (after Task 7):

```python
                if self.angle_encoding == "shifted":
                    angle = shifted_norm_to_physical_rad(reference_points[..., 4:5])
                else:
                    angle = reference_points[..., 4:5] * torch.pi
```

After:

```python
                angle = shifted_norm_to_physical_rad(reference_points[..., 4:5])
```

### A8.8 Keep untouched

`engine/deim/obb_angle_contract.py` stays intact: `physical_rad_to_norm`/`norm_to_physical_rad` are still used by the criterion's non-periodic L1 loss (`deim_criterion.py:382,389`), which is loss-side normalization unrelated to decoder-private encoding.

### A8.9 YAML

Remove the `decoder_angle_encoding` key: `sp_fz_common.yml:215`, `app/presets/deimv2_dinov3_sp_obb.yml:68`, `synthetic_exp_020_anrep0_offset_per.yml:260`. Delete `dlzdt/ablation/abl_shifted.yml` (shifted is now baseline behavior).

## A9 Task 9: Simplify guards and reject invalid values

### A9.1 Guard simplification list (only rep0/rep3 remain)

- `deim_decoder.py:305`: `if self.angle_rep == 0 or self.angle_rep == 1:` → `if self.angle_rep == 0:`.
- `deim_decoder.py:310`: `elif self.angle_rep == 2 or self.angle_rep == 3:` → `elif self.angle_rep == 3:`.
- `deim_decoder.py:390,421`: `== 2 or == 3` → `== 3`.
- `deim_decoder.py:551`: `if self.angle_rep != 0 and self.angle_rep != 1:` → `if self.angle_rep == 3:`.
- `deim_decoder.py:832,925`: `!= 0 and != 1` → `== 3`.
- `deim_decoder.py:221` (TransformerDecoder): `!= 0 and != 1` → `== 3`.

### A9.2 Constructor rejection of invalid values (`DEIMTransformer.__init__`, before the dof branches)

Insert before the `if self.box_mode == "obb":` if/elif chain:

```python
        if self.box_mode == "obb" and self.angle_rep not in (0, 3):
            raise ValueError(
                f"angle_rep must be 0 or 3 for box_mode='obb', got {self.angle_rep!r}"
            )
```

### A9.3 Comment cleanup

Remove comments in active source/config docs that describe rep1/rep2 or removed switches (historical plans/specs stay unchanged).

## A10 Task 10: OBB checkpoint compatibility contract

### A10.1 `engine/solver/_solver.py`

Add the constant, exception, and classify/assert helpers after `remove_module_prefix`:

```python
OBB_ANGLE_CONTRACT = "shifted_v1"


class CheckpointIncompatibleError(RuntimeError):
    """Raised when a checkpoint cannot be loaded under the current OBB contract."""


def classify_checkpoint_kind(state: Dict) -> str:
    """Classify 'hbb' / 'obb' / 'unknown' by encoder box-head output size.

    ``enc_bbox_head.layers.2.bias`` length equals ``_num_box_dof``:
    HBB=4, rep0/rep3=5, rep2=6. At execution time print the actual
    state_dict keys once; if wrapped (DataParallel/EMA), use the same key
    under the matching prefix.
    """
    try:
        model_state = state.get("model", {})
        if "module" in model_state:
            model_state = model_state["module"]
        dof = model_state["enc_bbox_head.layers.2.bias"].shape[0]
    except (KeyError, AttributeError):
        return "unknown"
    return "hbb" if dof == 4 else ("obb" if dof in (5, 6) else "unknown")


def assert_checkpoint_compat(state: Dict, expected: str = OBB_ANGLE_CONTRACT) -> None:
    """OBB checkpoints must carry a matching meta.obb_angle_contract marker."""
    marker = (state.get("meta") or {}).get("obb_angle_contract")
    if marker != expected:
        raise CheckpointIncompatibleError(
            "OBB checkpoint is incompatible with the current decoder: "
            f"expected meta.obb_angle_contract={expected!r}, got {marker!r}. "
            "Pre-cleanup OBB checkpoints (proportional encoding or gate-fusion "
            "state) must be retrained under the shifted-only contract."
        )
```

`state_dict` (lines 202-215): before `return state`, add:

```python
        if getattr(self.model, "box_mode", None) == "obb":
            state["meta"] = {"obb_angle_contract": OBB_ANGLE_CONTRACT}
```

`load_resume_state` (lines 240-248): before `self.load_state_dict(state)`, add:

```python
        if getattr(self.model, "box_mode", None) == "obb":
            assert_checkpoint_compat(state)
```

`load_tuning_state` (lines 250-275): before the head-adjustment logic, add:

```python
        if getattr(self.model, "box_mode", None) == "obb":
            kind = classify_checkpoint_kind(state)
            if kind != "hbb":
                assert_checkpoint_compat(state)
            else:
                print("Load unmarked 4D HBB checkpoint as OBB pretraining")
```

(Rules: `shifted_v1`-marked OBB checkpoints are accepted; unmarked checkpoints identifiable as 4D HBB are accepted as pretraining; all other unmarked/mismatched 5D/6D OBB checkpoints are rejected before the non-strict `_matched_state` load.)

### A10.2 Inference/export tools (four sites, identical logic)

`tools/inference/torch_inf.py:120-130`, `tools/inference/torch_inf_vis.py:117-127`, `tools/visualization/fiftyone_vis.py:238-248`, `tools/deployment/export_onnx.py:31-39`. Before each `cfg.model.load_state_dict(...)`, add (adjust variable names to each file's actual loaded checkpoint):

```python
        from engine.solver._solver import OBB_ANGLE_CONTRACT, CheckpointIncompatibleError

        if getattr(cfg.model, "box_mode", None) == "obb":
            marker = (checkpoint.get("meta") or {}).get("obb_angle_contract")
            if marker != OBB_ANGLE_CONTRACT:
                raise CheckpointIncompatibleError(
                    "OBB checkpoint is incompatible with the current decoder: "
                    f"expected meta.obb_angle_contract={OBB_ANGLE_CONTRACT!r}, "
                    f"got {marker!r}"
                )
```

HBB inference/export paths are unaffected. If the `deim_app` application layer has its own `strict=False` adapter path, insert the same check before it (verify the actual entry point at execution time).

**Check:** `test/test_obb_checkpoint_contract.py` fully green; one in-memory state round trip (save → classify → assert → load) succeeds.
