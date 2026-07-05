Usage
=====

Run causal discovery and then causal inference:

.. code-block:: bash

   uv run python articles/a07f0cdc427e09/experiment/05_causal_discovery_inference_completejourney.py

Run only causal discovery:

.. code-block:: bash

   uv run python articles/a07f0cdc427e09/experiment/03_causal_discovery_completejourney.py \
     --analysis-config articles/a07f0cdc427e09/conf/causal_discovery/analysis.yaml

Run only causal inference:

.. code-block:: bash

   uv run python articles/a07f0cdc427e09/experiment/04_causal_inference_completejourney.py \
     --config articles/a07f0cdc427e09/conf/causal_inference/causal_inference_default.yaml

Run treatment-effect mode through the integrated entrypoint:

.. code-block:: bash

   uv run python articles/a07f0cdc427e09/experiment/05_causal_discovery_inference_completejourney.py \
     --mode treatment_effect \
     --treatment treated \
     --outcome outcome_sales_value
