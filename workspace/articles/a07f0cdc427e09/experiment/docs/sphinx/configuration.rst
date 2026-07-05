Configuration
=============

Pipeline-level settings live in
``articles/a07f0cdc427e09/conf/causal_discovery/analysis.yaml`` and
``articles/a07f0cdc427e09/conf/causal_inference/causal_inference_default.yaml``.

Command-line arguments explicitly supplied by the user override YAML values.
If the default YAML is unavailable, built-in defaults are used.

By default, generated causal inference outputs are written under
``articles/a07f0cdc427e09/artifacts/causal_inference``.
The default causal discovery input directory is
``articles/a07f0cdc427e09/artifacts/causal_discovery``.

Sphinx HTML is generated outside ``experiment``:
``articles/a07f0cdc427e09/artifacts/docs/_build/html``.
