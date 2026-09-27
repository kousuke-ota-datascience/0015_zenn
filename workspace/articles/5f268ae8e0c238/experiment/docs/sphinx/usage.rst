Usage
=====

Validate the integrated pipeline without running stages:

.. code-block:: bash

   uv run python articles/1ceee528ed7ee8/experiment/05_causal_discovery_inference_completejourney.py \
     --validate-only

Inspect the resolved execution plan:

.. code-block:: bash

   uv run python articles/1ceee528ed7ee8/experiment/05_causal_discovery_inference_completejourney.py \
     --dry-run

Run discovery and inference:

.. code-block:: bash

   uv run python articles/1ceee528ed7ee8/experiment/05_causal_discovery_inference_completejourney.py

Run treatment-effect mode through the integrated entrypoint:

.. code-block:: bash

   uv run python articles/1ceee528ed7ee8/experiment/05_causal_discovery_inference_completejourney.py \
     --inference-mode treatment_effect \
     --inference-treatment treated \
     --inference-outcome outcome_sales_value
