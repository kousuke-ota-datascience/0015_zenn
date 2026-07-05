Configuration
=============

The integrated pipeline is configured by
``articles/1ceee528ed7ee8/conf/causal_inference/pipeline.yaml``.

Stage-specific settings live in:

* ``articles/1ceee528ed7ee8/conf/causal_discovery/analysis.yaml``
* ``articles/1ceee528ed7ee8/conf/causal_discovery/features.yaml``
* ``articles/1ceee528ed7ee8/conf/causal_inference/causal_inference_default.yaml``
* ``articles/1ceee528ed7ee8/conf/causal_inference/completejourney_household.yaml``
* ``articles/1ceee528ed7ee8/conf/causal_inference/feature_semantics.yaml``
* ``articles/1ceee528ed7ee8/conf/causal_inference/causal_design.yaml``

The integrated CLI exposes explicit prefix overrides such as
``--discovery-alpha`` and ``--inference-outcome``. These are resolved into an
``ExecutionPlan`` and then forwarded to stage runners.

Discovery-to-inference artifact transfer uses
``articles/1ceee528ed7ee8/artifacts/causal_discovery/manifest.yaml``. The
inference CLI accepts ``--discovery-manifest`` rather than a raw discovery
directory.
