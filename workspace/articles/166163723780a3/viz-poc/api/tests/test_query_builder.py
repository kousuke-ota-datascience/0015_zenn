import pytest

from app.catalog import CatalogValidationError, DatasetCatalog, DatasetNotFoundError
from app.config import CATALOG_PATH, DATA_ROOT
from app.models import QueryRequest
from app.query_builder import build_query


def test_build_query_from_valid_request() -> None:
    catalog = DatasetCatalog.load(CATALOG_PATH)
    request = QueryRequest(
        dataset="sales",
        dimensions=["region", "product"],
        metrics=["revenue", "order_count"],
        sort=[{"field": "revenue", "direction": "desc"}],
        limit=100,
    )

    plan = build_query(request, catalog, DATA_ROOT)

    assert "FROM read_csv_auto(" in plan.sql
    assert '"region" AS "region"' in plan.sql
    assert '"product" AS "product"' in plan.sql
    assert "(SUM(revenue)) AS \"revenue\"" in plan.sql
    assert "(COUNT(*)) AS \"order_count\"" in plan.sql
    assert "GROUP BY 1, 2" in plan.sql
    assert 'ORDER BY "revenue" DESC' in plan.sql
    assert "LIMIT 100" in plan.sql
    assert plan.parameters == []


def test_build_query_rejects_unknown_dataset() -> None:
    catalog = DatasetCatalog.load(CATALOG_PATH)
    request = QueryRequest(dataset="unknown", dimensions=[], metrics=["revenue"])

    with pytest.raises(DatasetNotFoundError):
        build_query(request, catalog, DATA_ROOT)


def test_build_query_rejects_unknown_dimension() -> None:
    catalog = DatasetCatalog.load(CATALOG_PATH)
    request = QueryRequest(dataset="sales", dimensions=["unknown"], metrics=["revenue"])

    with pytest.raises(CatalogValidationError):
        build_query(request, catalog, DATA_ROOT)


def test_build_query_rejects_unknown_metric() -> None:
    catalog = DatasetCatalog.load(CATALOG_PATH)
    request = QueryRequest(dataset="sales", dimensions=["region"], metrics=["unknown"])

    with pytest.raises(CatalogValidationError):
        build_query(request, catalog, DATA_ROOT)
