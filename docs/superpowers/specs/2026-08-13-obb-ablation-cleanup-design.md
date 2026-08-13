# OBB Ablation Cleanup Design

## Goal

Reduce the active OBB algorithm surface to the settings that improved results:

- keep `angle_rep=0` and `angle_rep=3`;
- make shifted angle encoding the only decoder encoding;
- remove rep1, rep2, gate fusion, angle-first prediction, post-adjustment offset scaling, and multi-angle anchors.

The public OBB contract remains `(cx, cy, w, h, theta)` with `theta` expressed in radians in `[0, pi)`.

## Ownership

The user owns production, configuration, and documentation changes. Sisyphus owns all changes under `test/`, writes tests before the corresponding production changes, reviews each production batch, and runs the final verification suite.

## Retained Architecture

`angle_rep=0` keeps the 5D reference plus 6D ADR residual path. Its external-rectangle and vertex-offset helpers are retained even though rep2 also used them.

`angle_rep=3` keeps the 5D direct-angle residual path and `decouple_angle_layers`. Its 5D geometry helpers are retained even though rep1 also used them.

Shifted encoding becomes unconditional at every decoder-private boundary: anchor initialization, denoising input, deformable-attention reference decoding, decoder refinement, encoder auxiliary output conversion, and final public-angle conversion.

`physical_rad_to_norm` and `norm_to_physical_rad` are not deleted globally. The criterion's retained non-periodic angle loss uses proportional normalization only to scale physical-angle L1 loss; that is separate from decoder-private encoding.

## Removed Architecture

The following constructor parameters and behavior are removed rather than fixed to configurable defaults:

- `use_gate_fusion` and `engine/deim/gated_fusion.py`;
- `use_angle_first`;
- `angle_step` and candidate expansion;
- `offset_scale_source`, with the retained geometry fixed to pre-adjustment scaling;
- `decoder_angle_encoding`, with shifted behavior fixed internally;
- all rep1- and rep2-specific heads, anchors, denoising conversions, auxiliary conversions, diagnostics, and configuration files.

After removal, accepted OBB representation values are exactly `0` and `3`. Invalid values fail during construction.

## Configuration Surface

The cleanup covers more than `configs/custom_obb/dlzdt/ablation/`. It includes direct and inherited references under `configs/custom_obb/synthetic_configs/`, application presets, test tools, and any other active YAML that supplies removed constructor keys.

Include dependents are removed or migrated before their rejected base files. `synthetic_exp_020_dec.yml` uses `angle_rep: True`, which is numerically rep1, and is treated as stale legacy configuration.

Stale constructor keys fail loudly at `engine/core/workspace.py:182`: the optional signature filter at lines 176-178 is commented out, so Python raises `TypeError` when the registered constructor is called. A repository configuration-contract test moves this failure to CI and reports the exact YAML section and key.

## Checkpoint Policy

Old proportional OBB checkpoints and pre-cleanup rep3 checkpoints are intentionally incompatible. They must not pass through `strict=False` tuning or application adapters silently.

New OBB checkpoints carry `meta.obb_angle_contract = "shifted_v1"`. OBB resume and inference require this marker. OBB tuning applies these rules:

- a marked `shifted_v1` OBB checkpoint is accepted;
- an unmarked checkpoint whose regression head identifies it as a 4D HBB model is accepted as HBB pretraining;
- an unmarked or differently marked 5D/6D OBB checkpoint is rejected with a dedicated compatibility error.

This distinguishes valid HBB-to-OBB initialization from semantically ambiguous old OBB weights.

## Test Contract

Sisyphus will lock these scenarios before production changes:

1. Rep0 + shifted constructs and runs a CPU forward pass, emits finite 5D public OBBs, and keeps `theta` in `[0, pi)`.
2. Rep3 + shifted constructs without angle-first or gate fusion, runs a CPU forward pass, and preserves finite decoupled-angle references and 5D public OBBs.
3. Active OBB YAML contains only accepted constructor keys and only rep0/rep3.
4. Removed constructor parameters and rep1/rep2 are rejected.
5. Old OBB checkpoints fail explicitly, while marked shifted OBB and identifiable HBB-pretraining checkpoints are accepted in their intended load paths.
6. Matcher, criterion, postprocessor, application prediction, and physical-angle boundaries remain unchanged.

Rep2-only tests and diagnostic tools are deleted only after their production path is removed. Shared geometry, stable-atan2, matcher, loss, and angle-contract tests remain.

## Verification

Each removal slice follows red, green, review, and regression. Final verification includes:

- focused CPU tests for rep0 and rep3;
- configuration parsing and model construction for every retained OBB config;
- explicit checkpoint-policy probes;
- the complete OBB and application test suites;
- diagnostics on changed source files;
- a repository search proving removed symbols no longer occur in active production/configuration code.

Historical specs and completed plans remain unchanged as records of prior experiments.
