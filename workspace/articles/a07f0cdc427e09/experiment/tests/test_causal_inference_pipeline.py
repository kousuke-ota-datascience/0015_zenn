from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[3]
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "py" / "myproj" / "src"))

from causal_inference_pipeline.cli import parse_args
from causal_inference_pipeline.config import load_pipeline_config, merge_cli_overrides
from causal_inference_pipeline.features.aggregations import aggregate_metrics
from causal_inference_pipeline.features.config import MetricSpec, load_feature_config
from causal_inference_pipeline.features.encoders import encode_binary_with_unknown
from causal_inference_pipeline.features.selectors import select_adjustment_set
from causal_inference_pipeline.estimation.linear_model import fit_linear_regression
from causal_inference_pipeline.estimation.treatment_effect import TreatmentEffectEstimator


ARTICLE_DIR = EXPERIMENT_DIR.parent
CONFIG_PATH = ARTICLE_DIR / "conf" / "causal_inference" / "causal_inference_default.yaml"
FEATURE_CONFIG_PATH = ARTICLE_DIR / "conf" / "causal_inference" / "completejourney_household.yaml"


def test_load_pipeline_config_from_yaml() -> None:
    config = load_pipeline_config(CONFIG_PATH)

    assert config.mode == "edge_weight"
    assert config.data.campaign_id == "18"
    assert str(config.data.discovery_dir) == "articles/a07f0cdc427e09/artifacts/causal_discovery"
    assert str(config.data.output_dir) == "articles/a07f0cdc427e09/artifacts/causal_inference"
    assert config.treatment_effect.outcome == "outcome_sales_value"


def test_cli_overrides_yaml_config() -> None:
    args = parse_args(
        [
            "--mode",
            "treatment_effect",
            "--outcome",
            "outcome_quantity",
            "--effect-methods",
            "diff_in_means",
        ]
    )
    config = merge_cli_overrides(load_pipeline_config(CONFIG_PATH), args)

    assert config.mode == "treatment_effect"
    assert config.treatment_effect.outcome == "outcome_quantity"
    assert config.treatment_effect.effect_methods == ("diff_in_means",)


def test_load_feature_config() -> None:
    feature_config = load_feature_config(FEATURE_CONFIG_PATH)

    assert feature_config.dataset.unit_key == "household_id"
    assert "pre_treatment_covariates" in feature_config.adjustment_sets
    assert feature_config.encodings["age_midpoint"].input_column == "age"


def test_binary_with_unknown_encoder() -> None:
    encoded = encode_binary_with_unknown(
        pd.Series(["Homeowner", "Renter", "Unknown", None]),
        positive_values=["Homeowner"],
    )

    assert encoded["positive"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert encoded["unknown"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_aggregate_sum_and_count_rows_metrics() -> None:
    frame = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "sales_value": [10.0, 2.5, 3.0],
        }
    )
    result = aggregate_metrics(
        frame,
        group_by=["household_id"],
        prefix="pre",
        metrics={
            "sales_value": MetricSpec(column="sales_value", agg="sum"),
            "rows": MetricSpec(column=None, agg="count_rows"),
        },
    ).sort_values("household_id")

    assert result["pre_sales_value"].tolist() == [12.5, 3.0]
    assert result["pre_rows"].tolist() == [2, 1]


def test_select_pre_treatment_covariates_from_config() -> None:
    feature_config = load_feature_config(FEATURE_CONFIG_PATH)
    frame = pd.DataFrame(
        {
            "treated": [0.0, 1.0, 0.0],
            "outcome_sales_value": [1.0, 2.0, 3.0],
            "age_midpoint": [21.5, 29.5, 39.5],
            "age_unknown": [0.0, 0.0, 0.0],
            "income_midpoint_k": [7.5, 19.5, 29.5],
            "household_size": [1.0, 2.0, 3.0],
            "outcome_quantity": [1.0, 1.0, 1.0],
        }
    )
    result = select_adjustment_set(
        frame,
        feature_config=feature_config,
        strategy="pre_treatment_covariates",
        treatment="treated",
        outcome="outcome_sales_value",
    )

    assert "age_midpoint" in result.selected
    assert "outcome_quantity" not in result.selected


def test_manual_covariates_raise_on_post_treatment_columns() -> None:
    feature_config = load_feature_config(FEATURE_CONFIG_PATH)
    frame = pd.DataFrame(
        {
            "treated": [0.0, 1.0],
            "outcome_sales_value": [1.0, 2.0],
            "outcome_quantity": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="manual covariate is not allowed"):
        select_adjustment_set(
            frame,
            feature_config=feature_config,
            strategy="manual",
            treatment="treated",
            outcome="outcome_sales_value",
            manual_covariates=["outcome_quantity"],
        )


def test_fit_linear_regression_known_result() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    fit = fit_linear_regression(y, x)

    assert np.allclose(fit.coefficients, [1.0, 2.0])
    assert fit.rank == 2


def test_diff_in_means_known_result() -> None:
    frame = pd.DataFrame(
        {
            "treated": [0.0, 0.0, 1.0, 1.0],
            "outcome": [1.0, 3.0, 5.0, 9.0],
        }
    )
    estimator = TreatmentEffectEstimator(
        frame,
        treatment="treated",
        outcome="outcome",
        covariates=[],
    )

    result = estimator.diff_in_means()

    assert result.effect == 5.0
    assert result.n_treated == 2
    assert result.n_control == 2
