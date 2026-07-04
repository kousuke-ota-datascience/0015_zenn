from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config_schema import FeatureConfig, FeatureSpec
from .schemas import CampaignWindow


LOG_TRANSFORMS = {"log1p", "signed_log1p"}


def table_frame(
    tables: dict[str, pd.DataFrame],
    feature_config: FeatureConfig,
    logical_name: str,
) -> pd.DataFrame:
    """Return a source table by its logical feature-config name.

    Args:
        tables: Loaded source tables keyed by dataset registry name.
        feature_config: Feature-generation configuration.
        logical_name: Logical table key from ``features.yaml``.

    Returns:
        Data frame for the requested logical table.
    """
    table = feature_config.tables[logical_name]
    return tables[table.name]


def build_household_index(
    transactions: pd.DataFrame,
    feature_config: FeatureConfig,
) -> pd.Index:
    """Build the household index used as the modeling unit.

    Args:
        transactions: Transaction-level data.
        feature_config: Feature-generation configuration containing key columns.

    Returns:
        Unique household identifier index in transaction order.
    """
    transaction_table = feature_config.tables["transactions"]
    household_column = transaction_table.household_key or feature_config.metadata["entity_id"]
    return pd.Index(
        transactions[household_column].dropna().unique(),
        name=household_column,
    )


def build_campaign_window(
    *,
    campaign_descriptions: pd.DataFrame,
    transactions: pd.DataFrame,
    campaign_id: str,
    feature_config: FeatureConfig,
    pre_weeks: int,
) -> CampaignWindow:
    """Derive pre-treatment and outcome week ranges for a campaign.

    Args:
        campaign_descriptions: Campaign date table.
        transactions: Transaction table used to infer week bounds.
        campaign_id: Campaign identifier to analyze.
        feature_config: Feature-generation configuration with table columns.
        pre_weeks: Number of weeks before treatment to include.

    Returns:
        Campaign window with clamped pre-treatment and outcome ranges.

    Raises:
        ValueError: If ``campaign_id`` is unknown or the selected campaign has
            no usable pre-treatment or outcome weeks.
    """
    campaign_table = feature_config.tables["campaign_descriptions"]
    transaction_table = feature_config.tables["transactions"]
    campaign_id_column = campaign_table.campaign_id or "campaign_id"
    start_column = campaign_table.start_day or "start_date"
    end_column = campaign_table.end_day or "end_date"
    week_column = transaction_table.week or feature_config.metadata["time_column"]
    timestamp_column = transaction_table.transaction_timestamp or "transaction_timestamp"
    divisor = int(
        feature_config.campaign_window.get("day_to_week", {}).get("divisor", 7)
    )

    first_transaction_date = pd.to_datetime(
        transactions[timestamp_column],
        unit="s",
    ).min()
    first_transaction_date = first_transaction_date.normalize()

    campaigns = campaign_descriptions.copy()
    campaigns["start_dt"] = pd.to_datetime(
        campaigns[start_column],
        unit="D",
        origin="unix",
    )
    campaigns["end_dt"] = pd.to_datetime(
        campaigns[end_column],
        unit="D",
        origin="unix",
    )
    campaigns["start_week"] = (
        (campaigns["start_dt"] - first_transaction_date).dt.days // divisor + 1
    ).astype(int)
    campaigns["end_week"] = (
        (campaigns["end_dt"] - first_transaction_date).dt.days // divisor + 1
    ).astype(int)

    matched = campaigns.loc[campaigns[campaign_id_column].astype(str).eq(campaign_id)]
    if matched.empty:
        raise ValueError(f"unknown campaign_id: {campaign_id}")

    campaign = matched.iloc[0]
    min_week = int(transactions[week_column].min())
    max_week = int(transactions[week_column].max())
    start_week = max(int(campaign["start_week"]), min_week)
    end_week = min(int(campaign["end_week"]), max_week)
    pre_end_week = start_week - 1
    pre_start_week = max(min_week, start_week - pre_weeks)

    if pre_start_week > pre_end_week:
        raise ValueError(
            f"campaign {campaign_id} has no pre-treatment weeks in transactions."
        )
    if start_week > end_week:
        raise ValueError(f"campaign {campaign_id} has no outcome weeks in transactions.")

    return CampaignWindow(
        campaign_id=campaign_id,
        start_week=start_week,
        end_week=end_week,
        pre_start_week=pre_start_week,
        pre_end_week=pre_end_week,
    )


def unknown_mask(series: pd.Series, unknown_values: tuple[Any, ...]) -> pd.Series:
    """Create a boolean mask for missing or configured unknown values.

    Args:
        series: Source values.
        unknown_values: Values that should be treated as unknown. ``None``
            matches missing values.

    Returns:
        Boolean series where missing or unknown values are ``True``.
    """
    mask = series.isna()
    non_null_unknowns = [value for value in unknown_values if value is not None]
    if non_null_unknowns:
        mask = mask | series.astype("string").isin(non_null_unknowns)
    return mask


def apply_feature_transform(
    series: pd.Series,
    spec: FeatureSpec,
    feature_config: FeatureConfig,
) -> pd.Series:
    """Apply an allowlisted feature transform.

    Args:
        series: Source or aggregated values.
        spec: Feature specification selecting the transform.
        feature_config: Full feature configuration for mapping lookups.

    Returns:
        Numeric series ready for discovery input.

    Raises:
        ValueError: If a transform is unsupported or receives invalid values.
    """
    transform = spec.transform

    if transform == "identity":
        return pd.to_numeric(series, errors="coerce").astype("float64")

    if transform == "ordered_category_midpoint":
        if spec.mapping is None:
            raise ValueError(f"ordered_category_midpoint requires mapping: {spec.name}")
        mapping = feature_config.categorical_mappings[spec.mapping]
        mapped = series.astype("string").map(mapping.midpoint_map()).astype("float64")
        return mapped.fillna(mapping.median_midpoint())

    if transform == "numeric_category":
        return pd.to_numeric(series.astype("string"), errors="coerce").fillna(0.0).astype("float64")

    if transform == "equals":
        return series.astype("string").fillna("").eq(str(spec.value)).astype("float64")

    if transform == "is_missing_or_unknown":
        values = spec.unknown_values
        if not values and spec.mapping is not None:
            values = feature_config.categorical_mappings[spec.mapping].unknown_values
        return unknown_mask(series, values).astype("float64")

    if transform == "campaign_membership":
        return series.astype("float64")

    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype("float64")
    if transform == "log1p":
        if (values < 0).any():
            raise ValueError(f"column must be non-negative before log1p: {spec.name}")
        return np.log1p(values)
    if transform == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))

    raise ValueError(f"unsupported transform: {transform}")


def build_baseline_features(
    demographics: pd.DataFrame,
    households: pd.Index,
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    """Build household-level baseline features from demographics.

    Args:
        demographics: Demographic source table.
        households: Modeling-unit index.
        feature_config: Feature-generation configuration.

    Returns:
        Baseline feature frame indexed by household.
    """
    demographic_table = feature_config.tables["demographics"]
    household_column = demographic_table.household_key or feature_config.metadata["entity_id"]
    aligned = demographics.set_index(household_column).reindex(households)

    frame = pd.DataFrame(index=households)
    for spec in feature_config.features_for_source_table("demographics"):
        frame[spec.name] = apply_feature_transform(
            aligned[spec.source_column],
            spec,
            feature_config,
        )
    return frame


def build_treatment_frame(
    campaigns: pd.DataFrame,
    households: pd.Index,
    campaign_id: str,
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    """Build the treatment indicator frame for campaign membership.

    Args:
        campaigns: Campaign household membership table.
        households: Modeling-unit index.
        campaign_id: Campaign identifier to mark as treated.
        feature_config: Feature-generation configuration.

    Returns:
        Treatment feature frame indexed by household.
    """
    campaign_table = feature_config.tables["campaigns"]
    household_column = campaign_table.household_key or feature_config.metadata["entity_id"]
    campaign_id_column = campaign_table.campaign_id or "campaign_id"

    treated_households = campaigns.loc[
        campaigns[campaign_id_column].astype(str).eq(campaign_id),
        household_column,
    ]
    membership = pd.Series(
        households.isin(treated_households).astype(float),
        index=households,
    )

    frame = pd.DataFrame(index=households)
    for spec in feature_config.features_for_source_table("campaigns"):
        frame[spec.name] = apply_feature_transform(membership, spec, feature_config)
    return frame


def window_bounds(window: CampaignWindow, name: str) -> tuple[int, int]:
    """Return week bounds for a named transaction window.

    Args:
        window: Campaign window with pre-treatment and outcome ranges.
        name: Window name, either ``"pre"`` or ``"outcome"``.

    Returns:
        Inclusive ``(start_week, end_week)`` pair.

    Raises:
        ValueError: If ``name`` is not a supported window.
    """
    if name == "pre":
        return window.pre_start_week, window.pre_end_week
    if name == "outcome":
        return window.start_week, window.end_week
    raise ValueError(f"unsupported transaction window: {name}")


def aggregate_series(
    transactions: pd.DataFrame,
    *,
    household_column: str,
    source_column: str,
    aggregation: str,
) -> pd.Series:
    """Aggregate a transaction column to household level.

    Args:
        transactions: Window-filtered transaction data.
        household_column: Household identifier column.
        source_column: Source column to aggregate.
        aggregation: Supported aggregation name.

    Returns:
        Household-indexed aggregated series.

    Raises:
        ValueError: If ``aggregation`` is unsupported.
    """
    grouped = transactions.groupby(household_column, observed=True)[source_column]
    if aggregation == "sum":
        return grouped.sum()
    if aggregation == "nunique":
        return grouped.nunique()
    if aggregation == "mean":
        return grouped.mean()
    if aggregation == "count":
        return grouped.count()
    raise ValueError(f"unsupported aggregation: {aggregation}")


def build_transaction_features(
    transactions: pd.DataFrame,
    households: pd.Index,
    window: CampaignWindow,
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    """Build configured pre-period and outcome transaction features.

    Args:
        transactions: Transaction source table.
        households: Modeling-unit index.
        window: Campaign window defining pre and outcome periods.
        feature_config: Feature-generation configuration.

    Returns:
        Transaction feature frame indexed by household.

    Raises:
        ValueError: If a transaction feature lacks window or aggregation.
    """
    transaction_table = feature_config.tables["transactions"]
    household_column = transaction_table.household_key or feature_config.metadata["entity_id"]
    week_column = transaction_table.week or feature_config.metadata["time_column"]

    frame = pd.DataFrame(index=households)
    for spec in feature_config.features_for_source_table("transactions"):
        if spec.window is None or spec.aggregation is None:
            raise ValueError(f"transaction feature requires window and aggregation: {spec.name}")

        start_week, end_week = window_bounds(window, spec.window)
        window_frame = transactions.loc[
            transactions[week_column].between(start_week, end_week),
            [household_column, spec.source_column],
        ]
        values = aggregate_series(
            window_frame,
            household_column=household_column,
            source_column=spec.source_column,
            aggregation=spec.aggregation,
        )
        fill_value = 0.0 if spec.fill_value is None else spec.fill_value
        frame[spec.name] = values.reindex(households).fillna(fill_value).astype("float64")

    return frame


def apply_configured_transforms(
    raw_frame: pd.DataFrame,
    feature_config: FeatureConfig,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Apply post-aggregation transforms configured for discovery features.

    Args:
        raw_frame: Raw household-level discovery feature frame.
        feature_config: Feature-generation configuration.

    Returns:
        Pair of transformed frame and mapping from transformed column name to
        transform name.
    """
    transformed = raw_frame.copy()
    transform_by_column: dict[str, str] = {}
    spec_by_name = feature_config.feature_by_name()

    for column in transformed.columns:
        spec = spec_by_name[column]
        if spec.transform not in LOG_TRANSFORMS:
            continue
        transformed[column] = apply_feature_transform(
            transformed[column],
            spec,
            feature_config,
        )
        transform_by_column[column] = spec.transform

    return transformed, transform_by_column


def build_variable_metadata(
    feature_config: FeatureConfig,
    columns: list[str],
    retained_columns: list[str],
    transform_by_column: dict[str, str],
) -> pd.DataFrame:
    """Build variable metadata from feature configuration.

    Args:
        feature_config: Feature-generation configuration.
        columns: Discovery-frame columns before standardization filtering.
        retained_columns: Columns retained after standardization and
            collinearity filtering.
        transform_by_column: Actual transforms applied during preprocessing.

    Returns:
        Variable metadata data frame for reporting and diagnostics.
    """
    retained = set(retained_columns)
    spec_by_name = feature_config.feature_by_name()
    records = []

    for column in columns:
        spec = spec_by_name[column]
        records.append(
            {
                "variable": column,
                "role": spec.role,
                "data_type": spec.data_type,
                "transform": transform_by_column.get(column, spec.transform),
                "background_tier": spec.background_tier,
                "used_in_discovery": spec.used_in_discovery and column in retained,
                "fisherz_caution": spec.fisherz_caution,
                "source_table": spec.source_table,
                "source_column": spec.source_column,
                "window": spec.window,
                "aggregation": spec.aggregation,
            }
        )

    return pd.DataFrame(records)
