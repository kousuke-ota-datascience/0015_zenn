from datetime import date, datetime
from decimal import Decimal
from math import isnan
from typing import Any

import duckdb
import pandas as pd


class QueryExecutionError(Exception):
    """Raised when DuckDB cannot execute a generated query."""


class QueryEngine:
    def execute(self, sql: str, parameters: list[Any] | None = None) -> list[dict[str, Any]]:
        try:
            with duckdb.connect(database=":memory:") as connection:
                dataframe = connection.execute(sql, parameters or []).fetchdf()
        except duckdb.Error as exc:
            raise QueryExecutionError(str(exc)) from exc

        return _dataframe_to_records(dataframe)


def _dataframe_to_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = dataframe.astype(object).where(pd.notnull(dataframe), None)
    records = normalized.to_dict(orient="records")
    return [
        {key: _to_jsonable(value) for key, value in row.items()}
        for row in records
    ]


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and isnan(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "item"):
        return value.item()
    return value
