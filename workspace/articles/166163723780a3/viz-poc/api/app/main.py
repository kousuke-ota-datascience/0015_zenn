from fastapi import FastAPI, HTTPException

from app.catalog import (
    CatalogValidationError,
    DatasetCatalog,
    DatasetNotFoundError,
)
from app.config import CATALOG_PATH, DATA_ROOT
from app.models import DatasetSchema, QueryRequest, QueryResponse
from app.query_builder import build_query
from app.query_engine import QueryEngine, QueryExecutionError


app = FastAPI(title="Visualization PoC API")
catalog = DatasetCatalog.load(CATALOG_PATH)
query_engine = QueryEngine()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/datasets", response_model=list[DatasetSchema])
def list_datasets() -> list[DatasetSchema]:
    return catalog.list_datasets()


@app.get("/datasets/{dataset_id}/schema", response_model=DatasetSchema)
def get_dataset_schema(dataset_id: str) -> DatasetSchema:
    try:
        return catalog.get_schema(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        plan = build_query(request, catalog, DATA_ROOT)
        rows = query_engine.execute(plan.sql, plan.parameters)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CatalogValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except QueryExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return QueryResponse(columns=plan.columns, rows=rows)
