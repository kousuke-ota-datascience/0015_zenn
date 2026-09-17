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
