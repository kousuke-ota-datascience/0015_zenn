# 追加修正指示

## P0. Remove duplicated discovery artifact package

`causal_inference/discovery/` がまだ残っている。  
`causal_inference/discovery_artifacts/` へ移行済みであれば、旧 `causal_inference/discovery/` を削除すること。

Acceptance Criteria:

- `experiment/causal_inference/discovery/` が存在しない
- `experiment/causal_inference/discovery_artifacts/` が存在する
- `grep -R "causal_inference.discovery" experiment/` が空
- `uv run pytest workspace/articles/1ceee528ed7ee8/experiment/tests/test_runtime.py` が成功する

## P0. Fix Sphinx api.rst

`docs/sphinx/api.rst` を valid reStructuredText に修正すること。

Acceptance Criteria:

- Markdown heading `#` / `##` を使わない
- `.. automodule::` directive を不要にインデントしない
- 存在しない module を automodule しない
- `uv run sphinx-build -W -b html workspace/articles/1ceee528ed7ee8/experiment/docs/sphinx workspace/articles/1ceee528ed7ee8/experiment/docs/sphinx/_build/html` が成功する

## P1. Isolate full-run smoke artifacts

full-run smoke test は既存 artifact に依存しないよう、`tmp_path` 配下に output を出すこと。

Acceptance Criteria:

- smoke test が fresh tmp dir で通る
- generated discovery manifest / inference manifest / report がその test run で作られたことを確認する
- run_id / created_at / artifact_manifest_path を検証する

## P1. Add estimator truth regression tests

AIPW / IPW / g-computation について fixed-seed synthetic DGP を使った regression test を追加すること。

Acceptance Criteria:

- ATE true value に対して tolerance 内
- ATT true value に対して tolerance 内
- extreme propensity の warning が出る