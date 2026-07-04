from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CampaignWindow:
    """Week ranges used for one campaign analysis.

    Attributes:
        campaign_id: Campaign identifier.
        start_week: First campaign week used for outcomes.
        end_week: Last campaign week used for outcomes.
        pre_start_week: First pre-treatment week.
        pre_end_week: Last pre-treatment week.
    """

    campaign_id: str
    start_week: int
    end_week: int
    pre_start_week: int
    pre_end_week: int


@dataclass(frozen=True)
class PreprocessingResult:
    """Container for preprocessing outputs.

    Attributes:
        model_frame: Household-level feature frame before discovery selection.
        raw_discovery_frame: Discovery features before configured transforms.
        discovery_frame: Discovery features after configured transforms.
        standardized: Standardized frame used by discovery algorithms.
        variable_metadata: Metadata for discovery variables.
    """

    model_frame: pd.DataFrame
    raw_discovery_frame: pd.DataFrame
    discovery_frame: pd.DataFrame
    standardized: pd.DataFrame
    variable_metadata: pd.DataFrame


@dataclass(frozen=True)
class DiscoveryResult:
    """Container for one discovery algorithm result.

    Attributes:
        algorithm: Algorithm name.
        causal_graph: Native graph or model object returned by the algorithm.
        edges: Normalized edge records.
        status: ``ok``, ``skipped``, or ``failed``.
        message: Empty string or diagnostic message.
    """

    algorithm: str
    causal_graph: object | None
    edges: pd.DataFrame
    status: str
    message: str
