# OBB Cleanup Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Resolve all merge-blocking findings from the post-implementation OBB cleanup review without restoring deleted representations, switches, compatibility aliases, or rep1-derived experiments.

**Architecture:** Detect OBB mode through the actual model wrapper boundary, enforce strict representation types at decoder construction, remove stale configuration state at its source, and delete or repoint every consumer of removed experiment artifacts. Tests exercise real wrapper/config integration rather than helper functions alone.

**Tech Stack:** Python 3.11, PyTorch, pytest, YAML configuration registry.

## Global Constraints

- Retain only integer `angle_rep` values `0` and `3`; reject booleans and floats.
- Public OBB output remains `(cx, cy, w, h, theta)` with `theta` in `[0, pi)`.
- New OBB checkpoints use `meta.obb_angle_contract = "shifted_v1"`.
- Do not restore deleted configs, compatibility aliases, rep1/rep2 paths, or proportional decoder selection.
- Remove rep1-derived loss experiment declarations rather than migrating them to rep0.
- Unit gates must run on CPU without external datasets or checkpoints.

### Task 1: Restore checkpoint contract integration

- [ ] Add failing tests for wrapper-aware OBB detection, marker save, resume rejection, and tuning classification.
- [ ] Add one shared model-mode helper and use it in solver plus four inference/export tools.
- [ ] Run checkpoint contract tests and diagnostics.

### Task 2: Enforce decoder/config contracts

- [ ] Add failing tests for bool/float `angle_rep` rejection.
- [ ] Enforce strict non-bool integer `0`/`3` in OBB decoder constructors.
- [ ] Add a failing cross-config construction test for `decouple_angle` registry pollution.
- [ ] Remove the stale YAML key and temporary test tolerance.

### Task 3: Remove deleted experiment consumers

- [ ] Remove the rep1-derived loss experiment test, provenance manifest, and comparison README.
- [ ] Repoint Kendall and early-stopping config tests to retained authoritative configs.
- [ ] Remove stale deleted-config entries from runnable diagnostic tools.
- [ ] Repair the synthetic config generator and live engineering documentation.

### Task 4: Verification

- [ ] Run diagnostics on every changed Python file.
- [ ] Run focused CPU-only contract suites.
- [ ] Run the full test suite and separate external-resource failures from regressions.
- [ ] Run rep0/rep3 forward, retained-config construction, checkpoint round-trip, residue scan, and `git diff --check`.
- [ ] Run the post-implementation `review-work` gate.
