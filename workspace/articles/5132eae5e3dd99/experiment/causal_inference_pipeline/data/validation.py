"""Validation utilities for source data tables."""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd


def validate_required_columns(
    frame: pd.DataFrame,
    required_columns: Collection[str],
    table_name: str,
) -> None:
    """Validate that a data frame contains required columns.

    Args:
        frame: Data frame to validate.
        required_columns: Required column names.
        table_name: Logical table name used in error messages.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")

