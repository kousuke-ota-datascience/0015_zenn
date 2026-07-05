from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[3]
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "py" / "myproj" / "src"))

from causal_discovery_pipeline.config import load_analysis_config
from causal_discovery_pipeline.features.aggregations import aggregate_series
from causal_discovery_pipeline.features.config import load_feature_config
from causal_discovery_pipeline.features.encoders import unknown_mask
from common_in_causal_inference.features import (
    drop_collinear_columns,
    select_campaign_window,
)
from common_in_causal_inference.orchestration import (
    _build_discovery_args,
    _build_inference_args,
    parse_args,
)


ARTICLE_DIR = EXPERIMENT_DIR.parent
DISCOVERY_ANALYSIS_CONFIG = ARTICLE_DIR / "conf" / "causal_discovery" / "analysis.yaml"
DISCOVERY_FEATURE_CONFIG = ARTICLE_DIR / "conf" / "causal_discovery" / "features.yaml"


def test_discovery_config_paths_point_to_current_article_artifacts() -> None:
    analysis_config = load_analysis_config(DISCOVERY_ANALYSIS_CONFIG)
    feature_config = load_feature_config(DISCOVERY_FEATURE_CONFIG)

    assert str(analysis_config.run.output_dir) == (
        "articles/a07f0cdc427e09/artifacts/causal_discovery"
    )
    assert set(feature_config.tables) == {
        "campaign_descriptions",
        "campaigns",
        "demographics",
        "transactions",
    }


def test_common_campaign_window_clamps_to_transaction_weeks() -> None:
    campaign_descriptions = pd.DataFrame(
        {
            "campaign_id": ["18"],
            "start_date": [21],
            "end_date": [35],
        }
    )
    transactions = pd.DataFrame(
        {
            "week": [1, 2, 3, 4, 5],
            "transaction_timestamp": [0, 7 * 24 * 60 * 60, 14 * 24 * 60 * 60, 21 * 24 * 60 * 60, 28 * 24 * 60 * 60],
        }
    )

    window = select_campaign_window(
        campaign_descriptions=campaign_descriptions,
        transactions=transactions,
        campaign_id="18",
        campaign_id_column="campaign_id",
        start_day_column="start_date",
        end_day_column="end_date",
        week_column="week",
        transaction_timestamp_column="transaction_timestamp",
        pre_weeks=2,
        day_to_week_divisor=7,
    )

    assert window.start_week == 4
    assert window.end_week == 5
    assert window.pre_start_week == 2
    assert window.pre_end_week == 3


def test_common_drop_collinear_columns_keeps_first_column() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "x_copy": [1.0, 2.0, 3.0, 4.0],
            "z": [1.0, 1.0, 2.0, 3.0],
        }
    )

    retained, dropped = drop_collinear_columns(frame, collinearity_threshold=0.999)

    assert list(retained.columns) == ["x", "z"]
    assert dropped.to_dict(orient="records") == [
        {"column": "x_copy", "reason": "collinear_with:x"}
    ]


def test_discovery_aggregate_series_sums_by_household() -> None:
    frame = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "sales_value": [10.0, 2.5, 3.0],
        }
    )

    result = aggregate_series(
        frame,
        household_column="household_id",
        source_column="sales_value",
        aggregation="sum",
    )

    assert result.to_dict() == {1: 12.5, 2: 3.0}


def test_discovery_unknown_mask_matches_missing_and_configured_values() -> None:
    result = unknown_mask(pd.Series(["Known", "Unknown", None]), ("Unknown", None))

    assert result.tolist() == [False, True, True]


def test_integrated_entrypoint_forwards_shared_overrides() -> None:
    args = parse_args(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--campaign-id",
            "18",
            "--pre-weeks",
            "4",
            "--discovery-output-dir",
            "articles/a07f0cdc427e09/artifacts/causal_discovery_alt",
            "--mode",
            "treatment_effect",
            "--outcome",
            "outcome_quantity",
        ]
    )

    discovery_args = _build_discovery_args(args)
    inference_args = _build_inference_args(args)

    assert discovery_args[:2] == ["--project-root", str(PROJECT_ROOT)]
    assert ["--campaign-id", "18"] == discovery_args[2:4]
    assert "--output-dir" in discovery_args
    assert ["--mode", "treatment_effect"] == inference_args[-4:-2]
    assert "--discovery-dir" in inference_args
