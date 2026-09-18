# P1-D deterministic fixture test suite

This directory contains the formal regression suite for the redesigned lore-entry workflow.

## Scope

- JSON Schema valid / invalid cases
- cross-artifact reference missing / duplicate cases
- taxonomy D01-D21 exact set, unknown code, wrong parent, wrong dimension, invalid status/code combination
- canonical Review read-side filename/payload, duplicate key, Schema-invalid, missing-target defensive handling
- Review write-side shared Seq, incomplete-cycle rejection, Verdict aggregation, managed-field protection, append-only writes
- control-plane reconciliation transitions, stale Review handling, divergence BLOCK, no-op/idempotency
- Notion control-plane duplicate/missing row, optimistic concurrency, post-update verification
- canonical JSON and Review JSON renderer non-mutation

## Run locally

From `workspace/articles/f8fe59b49c2e1e/`:

```bash
python -m pytest -q tests
```

The GitHub Actions workflow `.github/workflows/article-f8fe59b49c2e1e-tests.yml` runs the same suite when relevant source, Schema, taxonomy, or test files change.

External Notion and GitHub services are not called by the fixture suite. I/O boundaries are replaced with deterministic fixtures / monkeypatches; end-to-end integration remains P1-E.


## Summary meaning-preservation contract

`10_contents.summary` is tested through explicit salient-content coverage rather than a hand-picked keyword list.

- Workflow 10 selects semantically salient `content_units` first and records them in `summary.coverage_refs`.
- Schema requires at least one unique `coverage_ref`; `reference_validator.py` rejects dangling Content refs.
- Review 10 performs Blind Decode from `summary.narrative + summary.structure` before seeing the IDs, then audits every `coverage_ref` with `NONE / LOSS / CONFLICT`.
- `review_writer.py` requires the audit set to match `summary.coverage_refs` exactly for new-contract targets. Any LOSS/CONFLICT requires a Finding.
- Regression tests verify the selected-unit contract and retain targeted checks for previously observed failures, including 0179's brother motif-echo / identity terminal state and the narrator's final forbidden sighting.

This separates deterministic traceability from semantic judgment: Python ensures complete coverage bookkeeping; Review decides whether the summary actually preserves each salient meaning.
