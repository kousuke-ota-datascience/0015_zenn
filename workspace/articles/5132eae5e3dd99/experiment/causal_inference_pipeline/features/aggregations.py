"""Configured aggregation helpers for unit-level feature construction."""

from __future__ import annotations

import pandas as pd

from .config_schema import MetricSpec


def aggregate_metrics(
    frame: pd.DataFrame,
    group_by: list[str],
    metrics: dict[str, MetricSpec],
    prefix: str,
) -> pd.DataFrame:
    """Aggregates raw rows into unit-level feature columns.

    Args:
        frame: Source data frame after filtering.
        group_by: Columns used as aggregation keys.
        metrics: Metric specifications keyed by output metric names.
        prefix: Prefix added to output feature columns.

    Returns:
        Aggregated data frame keyed by ``group_by``.

    Raises:
        ValueError: If an aggregation method is unknown.
        KeyError: If a configured input column does not exist.
    """
    grouped = frame.groupby(group_by, observed=True)
    output = pd.DataFrame(index=grouped.size().index)
    for name, metric in metrics.items():
        column_name = f"{prefix}_{name}"
        if metric.agg == "count_rows":
            output[column_name] = grouped.size()
            continue
        if metric.column is None:
            raise ValueError(f"{metric.agg} requires a source column for {column_name}")
        if metric.column not in frame.columns:
            raise KeyError(f"missing aggregation column: {metric.column}")
        series_grouped = grouped[metric.column]
        if metric.agg == "sum":
            output[column_name] = series_grouped.sum()
        elif metric.agg == "mean":
            output[column_name] = series_grouped.mean()
        elif metric.agg == "count":
            output[column_name] = series_grouped.count()
        elif metric.agg == "nunique":
            output[column_name] = series_grouped.nunique()
        else:
            raise ValueError(f"unsupported aggregation: {metric.agg}")
    return output.reset_index()

