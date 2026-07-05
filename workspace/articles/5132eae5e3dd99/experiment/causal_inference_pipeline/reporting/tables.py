"""Report table formatting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def format_report_value(value: object) -> str:
    """Format a scalar value for Markdown reports.

    Args:
        value: Value to format.

    Returns:
        Human-readable string.
    """
    if isinstance(value, tuple | list):
        return ", ".join(str(item) for item in value)
    if value is None:
        return ""
    if isinstance(value, float | np.floating):
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{float(value):.6g}"
    return str(value)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a data frame as a simple Markdown table.

    Args:
        frame: Table to render.

    Returns:
        Markdown table string, or ``_No rows._`` for empty frames.
    """
    if frame.empty:
        return "_No rows._"
    printable = frame.copy()
    for column in printable.columns:
        printable[column] = printable[column].map(format_report_value)
    columns = list(printable.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in printable.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)

