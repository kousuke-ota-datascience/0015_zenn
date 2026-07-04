from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .config_schema import FeatureConfig
from .feature_engineering import (
    apply_configured_transforms,
    build_baseline_features,
    build_campaign_window,
    build_household_index,
    build_transaction_features,
    build_treatment_frame,
    build_variable_metadata,
    table_frame,
)
from .schemas import PreprocessingResult


class AbstractPreprocessor(ABC):
    """Interface for objects that produce discovery-ready model data."""

    @abstractmethod
    def preprocess(self) -> PreprocessingResult:
        """Run preprocessing and return all intermediate data products.

        Returns:
            Complete preprocessing result.
        """
        raise NotImplementedError


class CompleteJourneyPreprocessor(AbstractPreprocessor):
    """Build and preprocess Complete Journey discovery features.

    Args:
        tables: Loaded source tables keyed by registry entry name.
        campaign_id: Campaign identifier to analyze.
        pre_weeks: Number of pre-treatment weeks.
        collinearity_threshold: Absolute correlation threshold for dropping
            redundant columns.
        feature_config: Feature-generation configuration.
    """

    def __init__(
        self,
        *,
        tables: dict[str, pd.DataFrame],
        campaign_id: str,
        pre_weeks: int,
        collinearity_threshold: float,
        feature_config: FeatureConfig,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            tables: Loaded source tables keyed by registry entry name.
            campaign_id: Campaign identifier to analyze.
            pre_weeks: Number of pre-treatment weeks.
            collinearity_threshold: Absolute correlation threshold for dropping
                redundant columns.
            feature_config: Feature-generation configuration.
        """
        self.tables = tables
        self.campaign_id = campaign_id
        self.pre_weeks = pre_weeks
        self.collinearity_threshold = collinearity_threshold
        self.feature_config = feature_config

    def preprocess(self) -> PreprocessingResult:
        """Create features, transform skewed variables, and standardize input.

        Returns:
            Model frame, raw discovery input, transformed input, standardized
            discovery input, and variable metadata.
        """
        model_frame = self.build_model_frame(self.tables)
        feature_names = [spec.name for spec in self.feature_config.used_feature_specs()]
        raw_discovery_frame = model_frame.loc[:, feature_names]
        discovery_frame, transform_by_column = apply_configured_transforms(
            raw_discovery_frame,
            self.feature_config,
        )
        standardized = self.standardize(discovery_frame)
        variable_metadata = build_variable_metadata(
            self.feature_config,
            columns=list(discovery_frame.columns),
            retained_columns=list(standardized.columns),
            transform_by_column=transform_by_column,
        )
        return PreprocessingResult(
            model_frame=model_frame.reset_index(),
            raw_discovery_frame=raw_discovery_frame.reset_index(drop=True),
            discovery_frame=discovery_frame.reset_index(drop=True),
            standardized=standardized.reset_index(drop=True),
            variable_metadata=variable_metadata,
        )

    def build_model_frame(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build the household-level feature frame before standardization.

        Args:
            tables: Loaded source tables keyed by registry entry name.

        Returns:
            Household-indexed model frame containing configured features.
        """
        transactions = table_frame(tables, self.feature_config, "transactions")
        households = build_household_index(transactions, self.feature_config)
        window = build_campaign_window(
            campaign_descriptions=table_frame(
                tables,
                self.feature_config,
                "campaign_descriptions",
            ),
            transactions=transactions,
            campaign_id=self.campaign_id,
            feature_config=self.feature_config,
            pre_weeks=self.pre_weeks,
        )

        frames = [
            build_baseline_features(
                table_frame(tables, self.feature_config, "demographics"),
                households,
                self.feature_config,
            ),
            build_transaction_features(
                transactions,
                households,
                window,
                self.feature_config,
            ),
            build_treatment_frame(
                table_frame(tables, self.feature_config, "campaigns"),
                households,
                self.campaign_id,
                self.feature_config,
            ),
        ]
        return pd.concat(frames, axis=1)

    def drop_collinear_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Drop columns whose pairwise absolute correlation exceeds the threshold.

        Args:
            frame: Numeric feature frame.

        Returns:
            Frame with redundant columns removed.
        """
        if frame.empty:
            return frame
        corr = frame.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        columns_to_drop = [
            column
            for column in upper.columns
            if any(upper[column] >= self.collinearity_threshold)
        ]
        return frame.drop(columns=columns_to_drop)

    def standardize(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove constant/collinear columns and z-score remaining features.

        Args:
            frame: Numeric discovery feature frame.

        Returns:
            Standardized frame suitable for discovery algorithms.
        """
        standardized = frame.copy()
        std = standardized.std(axis=0)
        non_constant = std[std > 0].index
        standardized = standardized.loc[:, non_constant]
        standardized = self.drop_collinear_columns(standardized)
        return (standardized - standardized.mean(axis=0)) / standardized.std(axis=0)
