"""Feature-config-driven construction of household-level analysis frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .aggregations import aggregate_metrics
from .config_schema import EncodingSpec, FeatureConfig
from .encoders import (
    encode_binary_with_unknown,
    encode_kids_count,
    encode_numeric_extract,
    encode_one_hot,
    encode_ordinal_map,
)


@dataclass(frozen=True)
class CampaignWindow:
    """Campaign-relative analysis windows.

    Attributes:
        campaign_id: Campaign identifier.
        start_week: First campaign week retained in the data.
        end_week: Last campaign week retained in the data.
        pre_start_week: First pre-treatment week.
        pre_end_week: Last pre-treatment week.
    """

    campaign_id: str
    start_week: int
    end_week: int
    pre_start_week: int
    pre_end_week: int


@dataclass(frozen=True)
class FeatureBuildResult:
    """Feature construction result consumed by analysis modes.

    Attributes:
        model_frame: Household-level frame including raw source columns.
        inference_frame: Numeric unstandardized frame used for treatment
            effects and original-scale edge coefficients.
        standardized: Z-scored frame used for standardized edge coefficients.
        dropped_columns: Columns removed before standardization with reasons.
    """

    model_frame: pd.DataFrame
    inference_frame: pd.DataFrame
    standardized: pd.DataFrame
    dropped_columns: pd.DataFrame


class FeatureBuilder:
    """Builds an analysis frame from raw input tables and feature config.

    The builder converts raw Complete Journey tables into a household-level
    analysis frame. It is driven by ``FeatureConfig`` so aggregation
    definitions, encodings, and adjustment candidates can be changed without
    editing Python code. The default feature config preserves the legacy
    ``04_causal_inference_completejourney.py`` feature names.

    Args:
        feature_config: Feature construction specification loaded from YAML.

    Attributes:
        feature_config: Validated feature construction specification.
    """

    def __init__(self, feature_config: FeatureConfig) -> None:
        """Initialize the builder.

        Args:
            feature_config: Feature construction specification.
        """
        self.feature_config = feature_config

    def build(
        self,
        tables: dict[str, pd.DataFrame],
        campaign_id: str,
        pre_weeks: int,
        collinearity_threshold: float,
    ) -> FeatureBuildResult:
        """Builds the household-level analysis frame.

        Args:
            tables: Mapping from logical table names to raw data frames.
            campaign_id: Campaign identifier used to define treatment and
                analysis windows.
            pre_weeks: Number of weeks before campaign start used as the
                pre-treatment window.
            collinearity_threshold: Absolute correlation threshold used to drop
                redundant standardized columns.

        Returns:
            A feature build result containing original and standardized frames.

        Raises:
            ValueError: If required tables or configured columns are missing.
        """
        model_frame = self.build_model_frame(tables, campaign_id, pre_weeks)
        inference_frame = self.build_inference_frame(model_frame)
        standardized, dropped_columns = self.standardize(
            inference_frame,
            collinearity_threshold=collinearity_threshold,
        )
        configured_notes = pd.DataFrame(
            list(self.feature_config.validation.dropped_column_notes),
            columns=["column", "reason"],
        )
        if not configured_notes.empty:
            dropped_columns = pd.concat([dropped_columns, configured_notes], ignore_index=True)
        return FeatureBuildResult(
            model_frame=model_frame.reset_index(),
            inference_frame=inference_frame.reset_index(drop=True),
            standardized=standardized.reset_index(drop=True),
            dropped_columns=dropped_columns,
        )

    def build_model_frame(
        self,
        tables: dict[str, pd.DataFrame],
        campaign_id: str,
        pre_weeks: int,
    ) -> pd.DataFrame:
        """Build a household-indexed frame before numeric encoding.

        Args:
            tables: Source tables keyed by logical feature-config name.
            campaign_id: Campaign identifier to analyze.
            pre_weeks: Number of pre-treatment weeks.

        Returns:
            Household-indexed model frame containing raw demographics,
            configured aggregations, and treatment.
        """
        transactions = self._table(tables, "transactions")
        campaigns = self._table(tables, "campaigns")
        demographics = self._table(tables, "demographics")
        campaign_descriptions = self._table(tables, "campaign_descriptions")
        unit_key = self.feature_config.dataset.unit_key

        households = pd.Index(transactions[unit_key].dropna().unique(), name=unit_key)
        frame = pd.DataFrame(index=households)

        window = self.select_campaign_window(
            campaign_descriptions=campaign_descriptions,
            transactions=transactions,
            campaign_id=campaign_id,
            pre_weeks=pre_weeks,
        )

        for block in self.feature_config.aggregations.values():
            source = self._table(tables, block.source_table)
            aggregated = self.aggregate_block(source, block_name=block.prefix, window=window)
            frame = frame.join(aggregated.set_index(list(block.group_by)))

        aggregated_columns = [
            column
            for column in frame.columns
            if any(column.startswith(f"{block.prefix}_") for block in self.feature_config.aggregations.values())
        ]
        frame[aggregated_columns] = frame[aggregated_columns].fillna(0.0)

        treatment_name = str(self.feature_config.treatment.get("name", "treated"))
        treatment_campaign_col = str(
            self.feature_config.treatment.get("campaign_id_column", "campaign_id")
        )
        treatment_household_col = str(
            self.feature_config.treatment.get("household_key_column", unit_key)
        )
        treated_households = campaigns.loc[
            campaigns[treatment_campaign_col].astype(str).eq(str(campaign_id)),
            treatment_household_col,
        ]
        frame[treatment_name] = frame.index.isin(treated_households).astype(int)

        demographics_key = self.feature_config.tables["demographics"].household_key or unit_key
        demographic_columns = list(
            dict.fromkeys(spec.input_column for spec in self.feature_config.encodings.values())
        )
        aligned_demographics = demographics.set_index(demographics_key).reindex(households)
        for column in demographic_columns:
            frame[column] = aligned_demographics[column].astype("string").fillna("Unknown")

        return frame

    def aggregate_block(
        self,
        frame: pd.DataFrame,
        *,
        block_name: str,
        window: CampaignWindow,
    ) -> pd.DataFrame:
        """Aggregate one configured transaction block.

        Args:
            frame: Source table.
            block_name: Aggregation block prefix.
            window: Campaign-relative window bounds.

        Returns:
            Aggregated feature frame including group key columns.

        Raises:
            ValueError: If the block references an unsupported window.
        """
        block = next(
            candidate
            for candidate in self.feature_config.aggregations.values()
            if candidate.prefix == block_name
        )
        start_week, end_week = self.window_bounds(window, block.window)
        time_key = self.feature_config.dataset.time_key
        needed_columns = set(block.group_by)
        for metric in block.metrics.values():
            if metric.column is not None:
                needed_columns.add(metric.column)
        filtered = frame.loc[
            frame[time_key].between(start_week, end_week),
            list(needed_columns),
        ]
        return aggregate_metrics(
            filtered,
            group_by=list(block.group_by),
            metrics=block.metrics,
            prefix=block.prefix,
        )

    def build_inference_frame(self, model_frame: pd.DataFrame) -> pd.DataFrame:
        """Encode raw model-frame columns into numeric inference columns.

        Args:
            model_frame: Household-indexed frame before encoding.

        Returns:
            Numeric inference frame on the original scale.
        """
        output = pd.DataFrame(index=model_frame.index)
        for spec in self.feature_config.encodings.values():
            self._append_encoded_columns(output, model_frame[spec.input_column], spec)

        for block in self.feature_config.aggregations.values():
            for metric_name in block.metrics:
                column = f"{block.prefix}_{metric_name}"
                output[column] = model_frame[column].astype(float)

        treatment_name = str(self.feature_config.treatment.get("name", "treated"))
        output[treatment_name] = model_frame[treatment_name].astype(float)

        for block in self.feature_config.aggregations.values():
            if block.window != "outcome":
                continue
            for metric_name in block.metrics:
                column = f"{block.prefix}_{metric_name}"
                if column not in output:
                    output[column] = model_frame[column].astype(float)

        ordered = self._legacy_column_order(output.columns)
        return output.loc[:, ordered]

    def _append_encoded_columns(
        self,
        output: pd.DataFrame,
        source: pd.Series,
        spec: EncodingSpec,
    ) -> None:
        """Append encoded columns to an output frame.

        Args:
            output: Mutable output frame.
            source: Source series to encode.
            spec: Encoding specification.

        Raises:
            ValueError: If the encoding type is unsupported or incomplete.
        """
        if spec.type == "ordinal_map":
            if spec.output is None or spec.map is None:
                raise ValueError(f"ordinal_map requires output and map: {spec.input_column}")
            output[spec.output] = encode_ordinal_map(
                source,
                spec.map,
                unknown_value=spec.unknown_value,
            )
            return
        if spec.type == "unknown_indicator":
            if spec.output is None:
                raise ValueError(f"unknown_indicator requires output: {spec.input_column}")
            unknown_values = {"Unknown", *(str(item) for item in spec.unknown_values if item is not None)}
            output[spec.output] = (
                source.isna() | source.astype("string").isin(unknown_values)
            ).astype(float)
            return
        if spec.type == "numeric_extract":
            if spec.output is None:
                raise ValueError(f"numeric_extract requires output: {spec.input_column}")
            output[spec.output] = encode_numeric_extract(
                source,
                unknown_value=spec.unknown_value,
            )
            return
        if spec.type == "kids_count":
            if spec.output is None:
                raise ValueError(f"kids_count requires output: {spec.input_column}")
            output[spec.output] = encode_kids_count(
                source,
                unknown_value=spec.unknown_value,
            )
            return
        if spec.type == "binary_with_unknown":
            if spec.output_positive is None or spec.output_unknown is None:
                raise ValueError(
                    f"binary_with_unknown requires output columns: {spec.input_column}"
                )
            encoded = encode_binary_with_unknown(source, spec.positive_values)
            aliases = spec.aliases or {}
            for alias, target in aliases.items():
                if target == spec.output_positive:
                    output[alias] = encoded["positive"]
            output[spec.output_positive] = encoded["positive"]
            output[spec.output_unknown] = encoded["unknown"]
            return
        if spec.type == "one_hot":
            if spec.output is None:
                raise ValueError(f"one_hot requires output prefix: {spec.input_column}")
            encoded = encode_one_hot(source, spec.output)
            for column in encoded.columns:
                output[column] = encoded[column]
            return
        raise ValueError(f"unsupported encoding type: {spec.type}")

    def standardize(
        self,
        frame: pd.DataFrame,
        *,
        collinearity_threshold: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Drop non-estimable columns and z-score remaining features.

        Args:
            frame: Numeric inference frame.
            collinearity_threshold: Absolute correlation threshold used to drop
                redundant columns.

        Returns:
            Pair of standardized frame and dropped-column table.
        """
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        records: list[dict[str, Any]] = []
        for column in frame.columns:
            if numeric[column].isna().all():
                records.append({"column": column, "reason": "all_missing"})
            elif not pd.api.types.is_numeric_dtype(frame[column]):
                records.append({"column": column, "reason": "non_numeric"})

        kept = numeric.drop(columns=[record["column"] for record in records])
        std = kept.std(axis=0)
        constant_columns = std[std <= 0].index.tolist()
        records.extend(
            {"column": column, "reason": "constant"}
            for column in constant_columns
        )
        kept = kept.drop(columns=constant_columns)
        kept, collinear = self.drop_collinear_columns(
            kept,
            collinearity_threshold=collinearity_threshold,
        )
        if not collinear.empty:
            records.extend(collinear.to_dict(orient="records"))
        standardized = (kept - kept.mean(axis=0)) / kept.std(axis=0)
        return standardized, pd.DataFrame(records, columns=["column", "reason"])

    def drop_collinear_columns(
        self,
        frame: pd.DataFrame,
        *,
        collinearity_threshold: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Drop columns with pairwise absolute correlation above a threshold.

        Args:
            frame: Numeric feature frame.
            collinearity_threshold: Absolute correlation threshold.

        Returns:
            Pair of retained frame and dropped-column table.
        """
        if frame.empty:
            return frame, pd.DataFrame(columns=["column", "reason"])
        corr = frame.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        records = []
        for column in upper.columns:
            correlated_with = upper.index[upper[column] >= collinearity_threshold].tolist()
            if correlated_with:
                records.append(
                    {
                        "column": column,
                        "reason": f"collinear_with:{correlated_with[0]}",
                    }
                )
        columns_to_drop = [record["column"] for record in records]
        return frame.drop(columns=columns_to_drop), pd.DataFrame(records)

    def select_campaign_window(
        self,
        *,
        campaign_descriptions: pd.DataFrame,
        transactions: pd.DataFrame,
        campaign_id: str,
        pre_weeks: int,
    ) -> CampaignWindow:
        """Select pre-treatment and outcome weeks for a campaign.

        Args:
            campaign_descriptions: Campaign date table.
            transactions: Transaction table used to infer global week bounds.
            campaign_id: Campaign identifier.
            pre_weeks: Number of pre-treatment weeks.

        Returns:
            Campaign window.

        Raises:
            ValueError: If the campaign is unknown or has no usable weeks.
        """
        campaign_table = self.feature_config.tables["campaign_descriptions"]
        transaction_table = self.feature_config.tables["transactions"]
        campaign_id_column = campaign_table.campaign_id or "campaign_id"
        start_column = campaign_table.start_day or "start_date"
        end_column = campaign_table.end_day or "end_date"
        week_column = transaction_table.week or self.feature_config.dataset.time_key
        timestamp_column = transaction_table.transaction_timestamp or "transaction_timestamp"
        divisor = int(
            _nested_get(
                self.feature_config.windows,
                ["campaign_dates", "day_to_week", "divisor"],
                default=7,
            )
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

        matched = campaigns.loc[campaigns[campaign_id_column].astype(str).eq(str(campaign_id))]
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
            campaign_id=str(campaign_id),
            start_week=start_week,
            end_week=end_week,
            pre_start_week=pre_start_week,
            pre_end_week=pre_end_week,
        )

    def window_bounds(self, window: CampaignWindow, name: str) -> tuple[int, int]:
        """Return inclusive week bounds for a named configured window.

        Args:
            window: Campaign-relative window values.
            name: Window name such as ``pre`` or ``outcome``.

        Returns:
            Inclusive ``(start_week, end_week)`` pair.

        Raises:
            ValueError: If the window name is unsupported.
        """
        if name == "pre":
            return window.pre_start_week, window.pre_end_week
        if name == "outcome":
            return window.start_week, window.end_week
        raise ValueError(f"unsupported transaction window: {name}")

    def _table(self, tables: dict[str, pd.DataFrame], logical_name: str) -> pd.DataFrame:
        """Return a configured table by logical name.

        Args:
            tables: Tables keyed by logical name.
            logical_name: Requested logical name.

        Returns:
            Source data frame.

        Raises:
            ValueError: If the table is missing.
        """
        if logical_name not in tables:
            raise ValueError(f"missing required logical table: {logical_name}")
        return tables[logical_name]

    def _legacy_column_order(self, columns: pd.Index) -> list[str]:
        """Return legacy inference-frame column order when columns exist.

        Args:
            columns: Available output columns.

        Returns:
            Ordered column names.
        """
        preferred = [
            "age_midpoint",
            "age_unknown",
            "income_midpoint_k",
            "income_unknown",
            "homeowner",
            "homeowner_yes",
            "homeowner_unknown",
            "married",
            "married_yes",
            "married_unknown",
            "household_size",
            "kids_count",
            "pre_baskets",
            "pre_quantity",
            "pre_sales_value",
            "pre_retail_disc",
            "pre_coupon_disc",
            "pre_coupon_match_disc",
            "treated",
            "outcome_baskets",
            "outcome_quantity",
            "outcome_sales_value",
            "outcome_retail_disc",
            "outcome_coupon_disc",
            "outcome_coupon_match_disc",
        ]
        available = list(columns)
        ordered = [column for column in preferred if column in available]
        ordered.extend(column for column in available if column not in ordered)
        return ordered


def _nested_get(mapping: dict[str, Any], keys: list[str], default: Any) -> Any:
    """Read a nested mapping value with a default.

    Args:
        mapping: Source mapping.
        keys: Nested key path.
        default: Value returned when any key is missing.

    Returns:
        Nested value or ``default``.
    """
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
