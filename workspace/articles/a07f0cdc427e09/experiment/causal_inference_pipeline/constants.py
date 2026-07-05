"""因果推論パイプライン固有の定数。"""

from __future__ import annotations

from pathlib import Path

from common_in_causal_inference.constants import (
    DEFAULT_CAUSAL_DISCOVERY_DIR,
    DEFAULT_CAUSAL_INFERENCE_DIR,
    DEFAULT_DATASET_YAML,
    SUPPORTED_DISCOVERY_ALGORITHMS,
)


SUPPORTED_MODES = ("edge_weight", "treatment_effect")
SUPPORTED_ALGORITHMS = SUPPORTED_DISCOVERY_ALGORITHMS
SUPPORTED_ESTIMANDS = ("ATE", "ATT")
SUPPORTED_ROBUST_SE = ("none", "HC0", "HC1", "HC2", "HC3")
SUPPORTED_EFFECT_METHODS = ("diff_in_means", "ols", "ipw", "aipw")
SUPPORTED_ADJUSTMENT_STRATEGIES = (
    "pre_treatment_covariates",
    "manual",
    "graph_parents",
)

DEFAULT_CONFIG_PATH = Path(
    "articles/a07f0cdc427e09/conf/causal_inference/causal_inference_default.yaml"
)
DEFAULT_FEATURE_CONFIG_PATH = Path(
    "articles/a07f0cdc427e09/conf/causal_inference/completejourney_household.yaml"
)
DEFAULT_DISCOVERY_DIR = DEFAULT_CAUSAL_DISCOVERY_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_CAUSAL_INFERENCE_DIR
