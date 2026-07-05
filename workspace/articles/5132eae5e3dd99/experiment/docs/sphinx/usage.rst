Usage
=====

Run the default edge-weight mode:

.. code-block:: bash

   uv run python articles/5132eae5e3dd99/experiment/04_causal_inference_completejourney.py \
     --config articles/5132eae5e3dd99/conf/causal_inference/causal_inference_default.yaml

Run treatment-effect mode:

.. code-block:: bash

   uv run python articles/5132eae5e3dd99/experiment/04_causal_inference_completejourney.py \
     --mode treatment_effect \
     --treatment treated \
     --outcome outcome_sales_value
