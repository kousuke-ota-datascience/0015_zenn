"""Shared numerical helpers for treatment-effect inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


def records_to_frame(records: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    """Convert a list of records to a data frame with stable columns.

    Args:
        records: Row dictionaries.
        columns: Desired column order.

    Returns:
        Data frame containing ``records`` or an empty frame with ``columns``.
    """
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records, columns=columns)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute a weighted mean.

    Args:
        values: Numeric values.
        weights: Non-negative weights.

    Returns:
        Weighted mean, or ``nan`` when the weight sum is non-positive.
    """
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return np.nan
    return float(np.sum(weights * values) / weight_sum)


def weighted_variance(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    """Compute a population-style weighted variance.

    Args:
        values: Numeric values.
        weights: Non-negative weights.
        mean: Weighted mean.

    Returns:
        Weighted variance, or ``nan`` when the weight sum is non-positive.
    """
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return np.nan
    return float(np.sum(weights * (values - mean) ** 2) / weight_sum)


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute the Kish effective sample size for weights.

    Args:
        weights: Non-negative weights.

    Returns:
        Effective sample size, or ``nan`` when the squared-weight sum is
        non-positive.
    """
    denominator = float(np.sum(weights**2))
    if denominator <= 0:
        return np.nan
    return float(np.sum(weights) ** 2 / denominator)

