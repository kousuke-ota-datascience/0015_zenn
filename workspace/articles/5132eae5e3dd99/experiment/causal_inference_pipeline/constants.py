"""Shared constants for the causal inference pipeline."""

from __future__ import annotations

from pathlib import Path


SUPPORTED_MODES = ("edge_weight", "treatment_effect")
SUPPORTED_ALGORITHMS = ("pc", "ges", "lingam", "notears")
SUPPORTED_ESTIMANDS = ("ATE", "ATT")
SUPPORTED_ROBUST_SE = ("none", "HC0", "HC1", "HC2", "HC3")
SUPPORTED_EFFECT_METHODS = ("diff_in_means", "ols", "ipw", "aipw")
SUPPORTED_ADJUSTMENT_STRATEGIES = (
    "pre_treatment_covariates",
    "manual",
    "graph_parents",
)

DEFAULT_DATASET_YAML = Path("shared/py/myproj/conf/dataset/completejourney/10_interim.yaml")
DEFAULT_CONFIG_PATH = Path(
    "articles/5132eae5e3dd99/conf/causal_inference/causal_inference_default.yaml"
)
DEFAULT_FEATURE_CONFIG_PATH = Path(
    "articles/5132eae5e3dd99/conf/causal_inference/completejourney_household.yaml"
)
DEFAULT_DISCOVERY_DIR = Path("articles/5132eae5e3dd99/artifacts/causal_discovery")
DEFAULT_OUTPUT_DIR = Path("articles/5132eae5e3dd99/artifacts/causal_inference")
