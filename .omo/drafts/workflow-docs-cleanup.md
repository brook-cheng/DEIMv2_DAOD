---
slug: workflow-docs-cleanup
status: awaiting-approval
intent: clear
pending-action: write .omo/plans/workflow-docs-cleanup.md
approach: Define a strict ownership contract between openspec and docs/superpowers, add README/INDEX files for both systems, update stale cross-references, and move only documents that are clearly in the wrong tree.
---

# Draft: workflow-docs-cleanup

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
| C1 | `openspec/` owns formal product/change specs only; no agent workflow/review artifacts | active | direct directory read + glob results |
| C2 | `docs/superpowers/` owns agent workflow artifacts: designs, plans, reviews, experiment writeups | active | direct directory read + glob results |
| C3 | Cross-references and indexes tell humans where to put/find each document type | active | planned README/INDEX changes |

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->
| Keep both systems | Do not merge `openspec` and `docs/superpowers`; clarify responsibilities instead | Both already contain meaningful artifacts and serve different workflows | Yes |
| Preserve history | Prefer adding README/INDEX and updating references over mass moving files | Avoid breaking existing links and git history unless file is clearly misplaced | Yes |
| No code changes | This cleanup touches documentation only | User asked about workflow docs, not runtime behavior | Yes |

## Findings (cited - path:lines)
- `openspec/` currently contains only `changes/` and `specs/`; previous top-level decoder design/plan paths no longer exist (`openspec/2026-06-25...`, `openspec/plans/...`). Evidence: directory read of `openspec/` returned only `changes/`, `specs/`.
- `docs/superpowers/` currently contains `design/`, `plans/`, and `review/`. Evidence: directory read of `docs/superpowers/` returned exactly those three directories.
- `docs/superpowers/design/2026-06-25-decoder-decoupling-design.md:1-8` is now the canonical decoder-decoupling design; it includes revision metadata and original-design source.
- `docs/superpowers/plans/2026-06-25-decoder-decoupling-plan.md:1-18` is now the canonical decoder-decoupling implementation plan; it includes 2026-06-26 revisions.
- `docs/superpowers/review/2026-06-25-decoder-decoupling_review_deepseekv4_pro_max.md:5-8` still references the old `openspec/2026-06-25...` and `openspec/plans/...` paths, which are stale in the current tree.
- Recent git history includes `b2db979 doc: 调整文档位置`, indicating the current confusion likely came from a partial doc-location migration.

## Decisions (with rationale)
- Canonical ownership: `openspec/` = formal OpenSpec change lifecycle and accepted specs; `docs/superpowers/` = agent workflow artifacts and reviews.
- Add a short README in each root (`openspec/README.md`, `docs/superpowers/README.md`) with "put this here / do not put this here" rules.
- Add index files (`openspec/INDEX.md`, `docs/superpowers/INDEX.md`) so current artifacts are discoverable without searching.
- Update stale references in review/design/plan documents after the canonical path contract is written.
- Do not delete or rewrite historical technical content; only move/update path references where necessary.

## Scope IN
- Documentation-only cleanup of `openspec/` and `docs/superpowers/` workflow ownership.
- Add README/INDEX files for both trees.
- Update stale references from old `openspec/...decoder-decoupling...` paths to current `docs/superpowers/...` paths.
- Optional small moves only for clearly misplaced workflow review files.

## Scope OUT (Must NOT have)
- No model/code/config/test behavior changes.
- No deletion of historical review/design/plan content.
- No git commit unless explicitly requested by the user.
- No broad rewrite of OpenSpec change content under `openspec/changes/deimv2-obb/`.

## Open questions
- None blocking. Default adopted: preserve both systems and clarify ownership, instead of collapsing one into the other.

## Approval gate
status: awaiting-approval
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
