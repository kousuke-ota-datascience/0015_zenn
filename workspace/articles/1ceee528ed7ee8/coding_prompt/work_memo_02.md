# 追加改善指示

## P0. Full-run smoke test

`--validate-only` と `--dry-run` だけでなく、通常実行の full-run smoke test を追加すること。

Command:

```bash
uv run python workspace/articles/1ceee528ed7ee8/experiment/05_causal_discovery_inference_completejourney.py
```

検証対象:

- discovery manifest
- inference manifest
- inference report
- causal design section
- estimand summary
- artifact manifest path
- no old package imports

## P1. Adjustment set validation

`FeatureSemanticSpec` を使って adjustment set validation を厳格化すること。

各 adjustment variable について以下を error 条件にする。

- feature semantics に存在しない
- role が covariate ではない
- allowed_for_adjustment が false
- post_treatment が true
- treatment / outcome / mediator / collider が含まれる

## P2. Feature semantics validation の mode-aware 化

- edge_weight mode: discovery graph node と inference feature semantics の厳密一致を要求
- treatment_effect mode: treatment / outcome / covariates の subset validation を要求

## P3. Sphinx API docs 修正

`docs/sphinx/api.rst` を valid RST に修正すること。Markdown heading と indented automodule directive を使わない。

## P4. Test robustness

CLI override test で `resolved_args[-4:-2]` のような order-dependent assertion を使わない。`arg_value(args, option)` helper を使う。

## P5. Naming cleanup

`causal_inference/discovery/` を以下のいずれかに rename すること。

- `causal_inference/discovery_artifacts/`
- `causal_inference/inputs/`
- `causal_inference/graph_inputs/`