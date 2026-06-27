# workflow-docs-cleanup - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** A clear document-ownership map that separates formal OpenSpec artifacts from Superpowers/agent workflow artifacts, plus indexes and fixed stale references so future docs land in the right place.

**Why this approach:** Keep both systems because they serve different purposes; reduce confusion by adding explicit rules and indexes rather than deleting or merging historical documents.

**What it will NOT do:** It will not change model code, training configs, tests, or delete historical documents. It will not commit unless explicitly requested.

**Effort:** Short
**Risk:** Low - documentation-only cleanup, with path-reference updates as the main risk.
**Decisions to sanity-check:** `openspec/` remains formal spec/change lifecycle; `docs/superpowers/` remains agent-generated design/plan/review/experiment workspace.

Your next move: approve this cleanup plan, or tell me if you want a different ownership rule. Full execution detail follows below.

---

> TL;DR (machine): Short, low-risk documentation cleanup: add ownership READMEs/indexes for `openspec/` and `docs/superpowers/`, update stale references, optionally move clearly misplaced workflow files.

## Scope
### Must have
- Add `openspec/README.md` defining OpenSpec-only ownership: proposals, formal accepted specs, change lifecycle, and OpenSpec-managed analysis.
- Add `docs/superpowers/README.md` defining Superpowers workflow ownership: brainstorming designs, implementation plans, reviews, experiment summaries, and agent workflow outputs.
- Add `openspec/INDEX.md` listing current formal specs/changes and how they relate to the DEIMv2-OBB work.
- Add `docs/superpowers/INDEX.md` listing current design/plan/review artifacts and their canonical paths.
- Update stale references in existing docs that point to removed old paths like `openspec/2026-06-25-decoder-decoupling-design.md` or `openspec/plans/2026-06-25-decoder-decoupling-plan.md`.
- Record the final ownership contract in a short cleanup note under `docs/superpowers/review/`.
### Must NOT have (guardrails, anti-slop, scope boundaries)
- Must not change model/source/config/test behavior.
- Must not delete historical review/design/plan content.
- Must not move OpenSpec lifecycle files out of `openspec/changes/**` or `openspec/specs/**` unless a file is plainly an agent workflow artifact.
- Must not commit changes unless explicitly requested.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after; documentation checks only.
- Evidence: `.omo/evidence/task-<N>-workflow-docs-cleanup.txt` for each todo, plus final `git diff -- docs/superpowers openspec .omo` review.

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.
- Wave 1: inventory and ownership docs can be drafted in parallel after plan approval.
- Wave 2: stale reference fixes and optional move/index updates depend on Wave 1's ownership rules.
- Wave 3: final consistency verification.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | approval | 3,5,6 | 2,4 |
| 2 | approval | 3,5,6 | 1,4 |
| 3 | 1,2 | 5,6 | 4 |
| 4 | approval | 5,6 | 1,2,3 |
| 5 | 1,2,3,4 | 6 | none |
| 6 | 5 | final verification | none |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [x] 1. `openspec/README.md`: Add OpenSpec ownership contract to clarify formal spec/change lifecycle - expect new contributors know what belongs in `openspec/`.
  What to do / Must NOT do: Create a concise README explaining that `openspec/` owns formal proposals, accepted specs, change/task lifecycle, and OpenSpec-managed analysis. Explicitly say agent workflow outputs, reviews, brainstorming design drafts, and implementation execution plans belong under `docs/superpowers/`. Must not rewrite existing OpenSpec spec content.
  Parallelization: Wave 1 | Blocked by: user approval | Blocks: 3,5,6
  References (executor has NO interview context - be exhaustive): `openspec/` currently contains `changes/` and `specs/`; `openspec/changes/deimv2-obb/proposal.md`; `openspec/changes/deimv2-obb/design.md`; `openspec/changes/deimv2-obb/tasks.md`; `openspec/specs/2026-06-24-hungarian-matching-diagnosis-design.md`.
  Acceptance criteria (agent-executable): `test -f openspec/README.md` and the file contains the strings `Belongs here`, `Does not belong here`, and `docs/superpowers`.
  QA scenarios (name the exact tool + invocation): happy: `python - <<'PY'` reads `openspec/README.md` and asserts required headings; failure: same script asserts no forbidden phrase claiming `openspec` owns Superpowers reviews. Evidence `.omo/evidence/task-1-workflow-docs-cleanup.txt`.
  Commit: N | docs(workflow): clarify openspec ownership

- [x] 2. `docs/superpowers/README.md`: Add Superpowers workflow ownership contract to clarify design/plan/review artifact location - expect agent outputs stop being placed in `openspec/`.
  What to do / Must NOT do: Create a concise README explaining that `docs/superpowers/` owns agent-generated designs, implementation plans, code/plan reviews, experiment summaries, and workflow notes. Explicitly say formal OpenSpec proposals/specs/change tasks belong under `openspec/`. Must not claim Superpowers replaces OpenSpec.
  Parallelization: Wave 1 | Blocked by: user approval | Blocks: 3,5,6
  References (executor has NO interview context - be exhaustive): `docs/superpowers/design/2026-06-25-decoder-decoupling-design.md:1-8`; `docs/superpowers/plans/2026-06-25-decoder-decoupling-plan.md:1-18`; `docs/superpowers/review/2026-06-25-decoder-decoupling_review_deepseekv4_pro_max.md:1-16`.
  Acceptance criteria (agent-executable): `test -f docs/superpowers/README.md` and the file contains `design/`, `plans/`, `review/`, and `openspec`.
  QA scenarios (name the exact tool + invocation): happy: Python script asserts required sections; failure: script asserts README does not instruct putting formal OpenSpec proposals in `docs/superpowers/`. Evidence `.omo/evidence/task-2-workflow-docs-cleanup.txt`.
  Commit: N | docs(workflow): clarify superpowers ownership

- [x] 3. `openspec/INDEX.md`: Add formal OpenSpec artifact index to make accepted specs and changes discoverable - expect users find OpenSpec documents without scanning folders.
  What to do / Must NOT do: Create an index grouped by `changes/`, `specs/`, and analysis documents currently under OpenSpec. Include short one-line descriptions. Must not include `docs/superpowers` design/plans as OpenSpec-owned docs; link to `docs/superpowers/INDEX.md` for workflow artifacts instead.
  Parallelization: Wave 2 | Blocked by: 1,2 | Blocks: 5,6
  References (executor has NO interview context - be exhaustive): glob output found `openspec/specs/2026-06-24-hungarian-matching-diagnosis-design.md`, `openspec/specs/2026-06-23-synthetic-ellipse-obb-implementation-plan.md`, `openspec/specs/2026-06-23-synthetic-ellipse-obb-dataset-design.md`, `openspec/changes/deimv2-obb/**`, `openspec/changes/deimv2-obb-eval-opt/proposal.md`.
  Acceptance criteria (agent-executable): `test -f openspec/INDEX.md` and `grep -q 'deimv2-obb' openspec/INDEX.md` and `grep -q 'docs/superpowers/INDEX.md' openspec/INDEX.md`.
  QA scenarios (name the exact tool + invocation): happy: Python script verifies every `.md` under `openspec/changes` and `openspec/specs` is mentioned or intentionally excluded; failure: script fails if old removed top-level paths appear. Evidence `.omo/evidence/task-3-workflow-docs-cleanup.txt`.
  Commit: N | docs(workflow): index openspec artifacts

- [x] 4. `docs/superpowers/INDEX.md`: Add workflow artifact index to make designs/plans/reviews discoverable - expect decoder-decoupling canonical documents are clear.
  What to do / Must NOT do: Create an index grouped by `design/`, `plans/`, and `review/`. Mark `docs/superpowers/design/2026-06-25-decoder-decoupling-design.md` and `docs/superpowers/plans/2026-06-25-decoder-decoupling-plan.md` as canonical current decoder-decoupling docs. Must not duplicate long document contents.
  Parallelization: Wave 2 | Blocked by: user approval | Blocks: 5,6
  References (executor has NO interview context - be exhaustive): glob output found 13 files under `docs/superpowers/**`, including decoder-decoupling design/plan and multiple reviews.
  Acceptance criteria (agent-executable): `test -f docs/superpowers/INDEX.md` and it contains `2026-06-25-decoder-decoupling-design.md`, `2026-06-25-decoder-decoupling-plan.md`, and `OBB_CODE_REVIEW.md`.
  QA scenarios (name the exact tool + invocation): happy: Python script verifies every Markdown file under `docs/superpowers` is listed; failure: script fails if a listed path does not exist. Evidence `.omo/evidence/task-4-workflow-docs-cleanup.txt`.
  Commit: N | docs(workflow): index superpowers artifacts

- [x] 5. Existing docs: Update stale path references to the canonical document locations - expect no references to removed decoder-decoupling `openspec/` top-level paths remain.
  What to do / Must NOT do: Replace references to removed paths `openspec/2026-06-25-decoder-decoupling-design.md` and `openspec/plans/2026-06-25-decoder-decoupling-plan.md` with current canonical paths under `docs/superpowers/design/` and `docs/superpowers/plans/`. Must not alter substantive review conclusions.
  Parallelization: Wave 2 | Blocked by: 1,2,3,4 | Blocks: 6
  References (executor has NO interview context - be exhaustive): `docs/superpowers/review/2026-06-25-decoder-decoupling_review_deepseekv4_pro_max.md:5-8` contains stale old paths; read/grep all `*.md` under `docs/` and `openspec/` for the two removed path strings.
  Acceptance criteria (agent-executable): `! grep -R "openspec/2026-06-25-decoder-decoupling-design.md\|openspec/plans/2026-06-25-decoder-decoupling-plan.md" docs openspec` returns success when inverted.
  QA scenarios (name the exact tool + invocation): happy: grep command finds zero stale references; failure: grep command reports any stale path and fails the todo. Evidence `.omo/evidence/task-5-workflow-docs-cleanup.txt`.
  Commit: N | docs(workflow): fix stale workflow paths

- [x] 6. `docs/superpowers/review/2026-06-27-workflow-docs-cleanup-note.md`: Add cleanup note documenting the final ownership rules and changed files - expect future contributors understand why the split exists.
  What to do / Must NOT do: Create a short review/cleanup note summarizing the root cause (partial doc migration around `doc: 调整文档位置`), final ownership contract, and list of changed files. Must not include implementation speculation.
  Parallelization: Wave 3 | Blocked by: 5 | Blocks: final verification
  References (executor has NO interview context - be exhaustive): git history includes `b2db979 doc: 调整文档位置`; `docs/superpowers/README.md`; `openspec/README.md`; both INDEX files.
  Acceptance criteria (agent-executable): file exists and contains `openspec`, `docs/superpowers`, and `canonical`.
  QA scenarios (name the exact tool + invocation): happy: Python script asserts note includes changed-file list; failure: script fails if note references non-existent paths. Evidence `.omo/evidence/task-6-workflow-docs-cleanup.txt`.
  Commit: N | docs(workflow): record workflow docs cleanup

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [x] F1. Plan compliance audit
- [x] F2. Code quality review
- [x] F3. Real manual QA
- [x] F4. Scope fidelity

## Commit strategy
- Do not commit automatically. If user later requests a commit, make one documentation-only commit after final verification: `docs(workflow): clarify openspec and superpowers document ownership`.

## Success criteria
- A new contributor can answer: "Should this document go under OpenSpec or Superpowers?" by reading one README.
- Every current Markdown file under `docs/superpowers/` and `openspec/` is discoverable via an index.
- No stale references remain to removed decoder-decoupling OpenSpec top-level paths.
- `git diff -- docs/superpowers openspec .omo` shows documentation-only changes.
