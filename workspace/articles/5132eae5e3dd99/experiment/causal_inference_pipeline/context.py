"""Run-time context shared by pipeline modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .features.builder import FeatureBuildResult
from .features.config_schema import FeatureConfig


@dataclass(frozen=True)
class RunContext:
    """Resolved objects needed by an executable analysis mode.

    Attributes:
        config: Resolved pipeline configuration.
        feature_config: Resolved feature construction configuration.
        project_root: Repository root.
        dataset_yaml: Resolved dataset registry YAML path.
        discovery_dir: Resolved causal discovery output directory.
        output_dir: Resolved causal inference output directory.
        preprocessing_result: Household-level features and standardized frame.
    """

    config: PipelineConfig
    feature_config: FeatureConfig
    project_root: Path
    dataset_yaml: Path
    discovery_dir: Path
    output_dir: Path
    preprocessing_result: FeatureBuildResult

    @property
    def inference_frame(self) -> pd.DataFrame:
        """Return the unstandardized numeric analysis frame.

        Returns:
            Data frame on the original scale.
        """
        return self.preprocessing_result.inference_frame

    @property
    def standardized_frame(self) -> pd.DataFrame:
        """Return the standardized frame for edge-weight estimation.

        Returns:
            Z-scored data frame after dropping non-estimable columns.
        """
        return self.preprocessing_result.standardized

