from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.catalog import CatalogValidationError, DatasetCatalog, DatasetDefinition
from app.config import DATA_ROOT
from app.models import ColumnSpec, FilterSpec, QueryRequest


@dataclass(frozen=True)
class QueryPlan:
    sql: str
    parameters: list[Any]
    columns: list[ColumnSpec]


def build_query(
    request: QueryRequest,
    catalog: DatasetCatalog,
    data_root: Path = DATA_ROOT,
) -> QueryPlan:
    dataset = catalog.get_dataset(request.dataset)
    dimensions = catalog.validate_dimensions(request.dataset, request.dimensions)
    metrics = catalog.validate_metrics(request.dataset, request.metrics)
    _validate_sort_fields(request, dataset)
    _validate_filter_fields(request, dataset)

    select_terms = [
        f"{_quote_identifier(definition.column)} AS {_quote_identifier(dimension_id)}"
        for dimension_id, definition in dimensions.items()
    ]
    select_terms.extend(
        f"({definition.expression}) AS {_quote_identifier(metric_id)}"
        for metric_id, definition in metrics.items()
    )

    parameters: list[Any] = []
    sql_parts = [
        "SELECT",
        "  " + ",\n  ".join(select_terms),
        f"FROM {_source_relation(dataset, data_root)}",
    ]

    where_clause = _build_where_clause(request.filters, dataset, parameters)
    if where_clause:
        sql_parts.append(where_clause)

    if request.dimensions:
        sql_parts.append("GROUP BY " + ", ".join(str(index) for index in range(1, len(request.dimensions) + 1)))

    if request.sort:
        sql_parts.append(_build_order_by_clause(request))

    sql_parts.append(f"LIMIT {request.limit}")

    return QueryPlan(
        sql="\n".join(sql_parts),
        parameters=parameters,
        columns=catalog.columns_for_request(
            request.dataset,
            request.dimensions,
            request.metrics,
        ),
    )


def _validate_sort_fields(request: QueryRequest, dataset: DatasetDefinition) -> None:
    allowed_fields = set(dataset.dimensions) | set(dataset.metrics)
    unknown = [item.field for item in request.sort if item.field not in allowed_fields]
    if unknown:
        raise CatalogValidationError(
            f"Unknown sort field(s) for dataset '{request.dataset}': {', '.join(unknown)}"
        )


def _validate_filter_fields(request: QueryRequest, dataset: DatasetDefinition) -> None:
    unknown = [item.field for item in request.filters if item.field not in dataset.dimensions]
    if unknown:
        raise CatalogValidationError(
            f"Unknown filter field(s) for dataset '{request.dataset}': {', '.join(unknown)}"
        )


def _build_where_clause(
    filters: list[FilterSpec],
    dataset: DatasetDefinition,
    parameters: list[Any],
) -> str:
    if not filters:
        return ""

    conditions: list[str] = []
    for item in filters:
        dimension = dataset.dimensions[item.field]
        column = _quote_identifier(dimension.column)
        if item.operator == "in":
            placeholders = ", ".join("?" for _ in item.value)
            conditions.append(f"{column} IN ({placeholders})")
            parameters.extend(item.value)
        else:
            conditions.append(f"{column} {item.operator} ?")
            parameters.append(item.value)

    return "WHERE " + " AND ".join(conditions)


def _build_order_by_clause(request: QueryRequest) -> str:
    terms = [
        f"{_quote_identifier(item.field)} {item.direction.upper()}"
        for item in request.sort
    ]
    return "ORDER BY " + ", ".join(terms)


def _source_relation(dataset: DatasetDefinition, data_root: Path) -> str:
    source_path = Path(dataset.source)
    if not source_path.is_absolute():
        source_path = data_root / source_path

    source_literal = _quote_string_literal(str(source_path))
    if dataset.source_type == "csv":
        return f"read_csv_auto({source_literal})"
    if dataset.source_type == "parquet":
        return f"read_parquet({source_literal})"

    raise CatalogValidationError(f"Unsupported source_type: {dataset.source_type}")


def _quote_identifier(identifier: str) -> str:
    if identifier == "":
        raise CatalogValidationError("Empty SQL identifier is not allowed")
    return '"' + identifier.replace('"', '""') + '"'


def _quote_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
