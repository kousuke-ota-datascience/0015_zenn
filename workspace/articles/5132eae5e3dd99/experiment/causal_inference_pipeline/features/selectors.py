"""Adjustment set selection utilities."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import pandas as pd

from .config_schema import FeatureConfig


@dataclass(frozen=True)
class AdjustmentSetResult:
    """Selected and excluded adjustment candidates.

    Attributes:
        selected: Covariates retained for adjustment.
        excluded: Candidate columns excluded with reasons.
    """

    selected: list[str]
    excluded: pd.DataFrame


def is_excluded_adjustment_column(
    column: str,
    treatment: str,
    outcome: str,
    feature_config: FeatureConfig,
) -> bool:
    """Return whether a column is forbidden for adjustment.

    Args:
        column: Candidate covariate.
        treatment: Treatment column.
        outcome: Outcome column.
        feature_config: Feature configuration with exclusion patterns.

    Returns:
        ``True`` when the candidate is treatment, outcome, or matches a
        configured post-treatment exclusion pattern.
    """
    if column in {treatment, outcome}:
        return True
    return any(re.search(pattern, column) for pattern in feature_config.exclude_patterns)


def select_adjustment_set(
    frame: pd.DataFrame,
    feature_config: FeatureConfig,
    strategy: str,
    treatment: str,
    outcome: str,
    manual_covariates: list[str] | tuple[str, ...] | None = None,
    graph_edges: pd.DataFrame | None = None,
) -> AdjustmentSetResult:
    """Selects adjustment covariates for treatment effect estimation.

    Args:
        frame: Analysis frame.
        feature_config: Feature configuration containing adjustment rules.
        strategy: Adjustment strategy name.
        treatment: Treatment column name.
        outcome: Outcome column name.
        manual_covariates: Manually specified covariates. Required when
            ``strategy`` is ``manual``.
        graph_edges: Directed edges used when ``strategy`` is
            ``graph_parents``.

    Returns:
        Selected covariates and excluded candidates with reasons.

    Raises:
        ValueError: If manual covariates are missing, invalid, or excluded by
            post-treatment safety rules.
    """
    excluded_records: list[dict[str, str]] = []
    if strategy == "manual":
        if not manual_covariates:
            raise ValueError("--covariates must be specified when --adjustment-strategy manual")
        missing = [column for column in manual_covariates if column not in frame.columns]
        if missing:
            raise ValueError(f"manual covariates are missing from frame: {missing}")
        candidates = list(manual_covariates)
    elif strategy in feature_config.adjustment_sets:
        candidates = _configured_candidates(frame, feature_config, strategy)
    elif strategy == "graph_parents":
        candidates = _graph_parent_candidates(frame, graph_edges, outcome)
    else:
        raise ValueError(f"unknown adjustment strategy: {strategy}")

    selected: list[str] = []
    seen: set[str] = set()
    for column in candidates:
        if is_excluded_adjustment_column(column, treatment, outcome, feature_config):
            excluded_records.append({"column": column, "reason": "excluded_by_safety_rule"})
            if strategy == "manual":
                raise ValueError(f"manual covariate is not allowed for adjustment: {column}")
            continue
        if column in seen:
            excluded_records.append({"column": column, "reason": "duplicate"})
            continue
        if column not in frame.columns:
            excluded_records.append({"column": column, "reason": "missing_from_frame"})
            continue
        selected.append(column)
        seen.add(column)

    if strategy != "manual":
        selected, pruned = prune_auto_adjustment_candidates(frame, selected)
        excluded_records.extend(pruned)

    return AdjustmentSetResult(
        selected=selected,
        excluded=pd.DataFrame(excluded_records, columns=["column", "reason"]),
    )


def prune_auto_adjustment_candidates(
    frame: pd.DataFrame,
    candidates: list[str],
    *,
    collinearity_threshold: float = 0.995,
) -> tuple[list[str], list[dict[str, str]]]:
    """Prune constant and collinear automatic adjustment candidates.

    Args:
        frame: Analysis frame.
        candidates: Candidate covariate names.
        collinearity_threshold: Absolute correlation threshold.

    Returns:
        Pair of retained covariates and excluded records.
    """
    selected: list[str] = []
    excluded: list[dict[str, str]] = []
    for column in candidates:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.dropna().nunique() <= 1:
            excluded.append({"column": column, "reason": "constant_or_all_missing"})
            continue
        if not selected:
            selected.append(column)
            continue
        correlations = frame.loc[:, [*selected, column]].corr(numeric_only=True).abs()
        correlated = correlations.loc[selected, column]
        if (correlated >= collinearity_threshold).any():
            first = str(correlated[correlated >= collinearity_threshold].index[0])
            excluded.append({"column": column, "reason": f"collinear_with:{first}"})
            continue
        selected.append(column)
    return selected, excluded


def _configured_candidates(
    frame: pd.DataFrame,
    feature_config: FeatureConfig,
    strategy: str,
) -> list[str]:
    """Return configured candidates for a named strategy.

    Args:
        frame: Analysis frame.
        feature_config: Feature configuration.
        strategy: Adjustment-set strategy name.

    Returns:
        Candidate covariate names.
    """
    spec = feature_config.adjustment_sets[strategy]
    candidates = list(spec.include)
    for pattern in spec.include_patterns:
        candidates.extend(column for column in frame.columns if re.search(pattern, column))
    return candidates


def _graph_parent_candidates(
    frame: pd.DataFrame,
    graph_edges: pd.DataFrame | None,
    outcome: str,
) -> list[str]:
    """Return graph-parent adjustment candidates.

    Args:
        frame: Analysis frame.
        graph_edges: Directed edge table.
        outcome: Outcome column.

    Returns:
        Parent columns that exist in ``frame``.
    """
    if graph_edges is None or graph_edges.empty:
        warnings.warn(
            "graph_parents adjustment requested but no graph edges were available.",
            stacklevel=2,
        )
        return []
    directed = graph_edges.loc[graph_edges["edge"].eq("-->")]
    return [
        str(row["source"])
        for _, row in directed.iterrows()
        if str(row["target"]) == outcome and str(row["source"]) in frame.columns
    ]

