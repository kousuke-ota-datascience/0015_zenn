# Data Visualization PoC

## Overview

This is a lightweight data visualization system for data science PoC work.
It separates the dashboard UI from the backend API so that the same API and
semantic catalog can later be reused by a production application.

## Architecture

```text
Browser
  -> Dash dashboard
  -> HTTP
  -> FastAPI
  -> DuckDB
  -> CSV / Parquet
```

- `dashboard/`: Dash UI. It never reads data files directly.
- `api/`: FastAPI backend, semantic catalog, SQL builder, and DuckDB executor.
- `api/app/datasets.yaml`: dataset, dimension, and metric definitions.
- `data/`: local CSV or Parquet files mounted into the API container.

## Run

```bash
docker compose up --build
```

FastAPI docs:

```text
http://localhost:8000/docs
```

Dashboard:

```text
http://localhost:8050
```

Health check:

```bash
curl http://localhost:8000/health
```

## API Endpoints

- `GET /health`: returns `{"status": "ok"}`.
- `GET /datasets`: returns registered datasets with public dimensions and metrics.
- `GET /datasets/{dataset_id}/schema`: returns one dataset schema. Unknown datasets return `404`.
- `POST /query`: returns aggregated rows for registered dimensions and metrics.

Example query:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset": "sales",
    "dimensions": ["region", "product"],
    "metrics": ["revenue", "order_count"],
    "filters": [],
    "sort": [{"field": "revenue", "direction": "desc"}],
    "limit": 100
  }'
```

## Add a Dataset

1. Add the CSV or Parquet file under `data/`.
2. Add an entry to `api/app/datasets.yaml`.
3. Set `source_type` to `csv` or `parquet`.
4. Set `source` to a path relative to the project root, for example `data/sales.csv`.

Example:

```yaml
datasets:
  sales:
    label: "Sales"
    source_type: "csv"
    source: "data/sales.csv"
```

## Add Dimensions and Metrics

Dimensions expose allowed grouping and filter fields:

```yaml
dimensions:
  region:
    label: "Region"
    column: "region"
    type: "category"
```

Metrics expose allowed aggregate expressions:

```yaml
metrics:
  revenue:
    label: "Revenue"
    expression: "SUM(revenue)"
    type: "number"
```

The UI and API use semantic IDs such as `region` and `revenue`. Physical column
names and metric SQL expressions stay in the catalog.

## Tests

Run backend tests from the API directory:

```bash
cd api
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e .
pytest
```

## Notes

- User-supplied arbitrary SQL is not accepted.
- Query input is limited to registered dataset IDs, dimension IDs, metric IDs,
  filter operators, sort fields, sort direction, and limit.
- Metric definitions are centralized in `datasets.yaml`.
- This is a PoC. Production use needs authentication, authorization, audit logs,
  caching, stronger catalog governance, and operational monitoring.
