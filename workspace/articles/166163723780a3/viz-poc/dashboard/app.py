import os
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 10


def api_get(path: str) -> Any:
    response = requests.get(
        f"{API_BASE_URL}{path}",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    response = requests.post(
        f"{API_BASE_URL}{path}",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def load_datasets() -> tuple[list[dict[str, Any]], str | None]:
    try:
        datasets = api_get("/datasets")
    except requests.RequestException as exc:
        return [], f"API request failed: {exc}"
    return datasets, None


def dropdown_options(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"label": item["label"], "value": item["id"]} for item in items]


def empty_figure(message: str = "No data") -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
            }
        ],
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    return figure


def build_figure(
    rows: list[dict[str, Any]],
    dimensions: list[str],
    metrics: list[str],
    chart_type: str,
) -> go.Figure:
    if not rows:
        return empty_figure("No rows returned")
    if not metrics:
        return empty_figure("Select at least one metric")

    dataframe = pd.DataFrame(rows)
    metric = metrics[0]
    x_field = dimensions[0] if dimensions else None
    color_field = dimensions[1] if len(dimensions) > 1 else None

    if x_field is None:
        dataframe = dataframe.reset_index(names="row")
        x_field = "row"

    if chart_type == "line":
        figure = px.line(dataframe, x=x_field, y=metric, color=color_field, markers=True)
    elif chart_type == "scatter":
        figure = px.scatter(dataframe, x=x_field, y=metric, color=color_field)
    else:
        figure = px.bar(dataframe, x=x_field, y=metric, color=color_field)

    figure.update_layout(
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        legend_title_text=color_field,
    )
    return figure


datasets, startup_error = load_datasets()
initial_dataset = datasets[0]["id"] if datasets else None

app = Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        dcc.Interval(id="dataset-refresh", interval=3000, n_intervals=0),
        html.Div(
            [
                html.H1("Data Visualization PoC"),
                html.Div(
                    [
                        html.Label("Dataset", htmlFor="dataset-dropdown"),
                        dcc.Dropdown(
                            id="dataset-dropdown",
                            options=dropdown_options(datasets),
                            value=initial_dataset,
                            clearable=False,
                        ),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label("Dimensions", htmlFor="dimension-dropdown"),
                        dcc.Dropdown(
                            id="dimension-dropdown",
                            multi=True,
                            placeholder="Select dimensions",
                        ),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label("Metrics", htmlFor="metric-dropdown"),
                        dcc.Dropdown(
                            id="metric-dropdown",
                            multi=True,
                            placeholder="Select metrics",
                        ),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label("Chart", htmlFor="chart-type-dropdown"),
                        dcc.Dropdown(
                            id="chart-type-dropdown",
                            options=[
                                {"label": "Bar", "value": "bar"},
                                {"label": "Line", "value": "line"},
                                {"label": "Scatter", "value": "scatter"},
                            ],
                            value="bar",
                            clearable=False,
                        ),
                    ],
                    className="control",
                ),
                html.Button("Run", id="run-button", n_clicks=0),
                html.Div(startup_error or "", id="dataset-status-message"),
                html.Div("", id="schema-status-message"),
                html.Div("", id="query-status-message"),
            ],
            className="sidebar",
        ),
        html.Div(
            [
                dcc.Graph(id="result-graph", figure=empty_figure("Run a query")),
                dash_table.DataTable(
                    id="result-table",
                    columns=[],
                    data=[],
                    page_size=20,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "fontFamily": "system-ui, sans-serif",
                        "fontSize": "14px",
                        "padding": "8px",
                        "textAlign": "left",
                    },
                    style_header={"fontWeight": "600", "backgroundColor": "#f4f6f8"},
                ),
            ],
            className="content",
        ),
    ],
    className="shell",
)

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>Data Visualization PoC</title>
    {%favicon%}
    {%css%}
    <style>
      * { box-sizing: border-box; }
      body {
        margin: 0;
        background: #f7f8fa;
        color: #1f2933;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .shell {
        display: grid;
        grid-template-columns: minmax(260px, 320px) 1fr;
        min-height: 100vh;
      }
      .sidebar {
        background: #ffffff;
        border-right: 1px solid #d9dee7;
        padding: 24px;
      }
      .content {
        display: grid;
        grid-template-rows: minmax(320px, 45vh) 1fr;
        gap: 16px;
        padding: 24px;
        min-width: 0;
      }
      h1 {
        margin: 0 0 24px;
        font-size: 22px;
        line-height: 1.25;
      }
      label {
        display: block;
        margin-bottom: 6px;
        font-size: 13px;
        font-weight: 600;
      }
      .control { margin-bottom: 16px; }
      button {
        width: 100%;
        height: 40px;
        border: 0;
        border-radius: 6px;
        background: #2563eb;
        color: #ffffff;
        font-weight: 700;
        cursor: pointer;
      }
      button:hover { background: #1d4ed8; }
      #dataset-status-message,
      #schema-status-message,
      #query-status-message {
        margin-top: 14px;
        color: #b42318;
        font-size: 13px;
        line-height: 1.4;
        overflow-wrap: anywhere;
      }
      @media (max-width: 840px) {
        .shell { grid-template-columns: 1fr; }
        .sidebar { border-right: 0; border-bottom: 1px solid #d9dee7; }
        .content {
          grid-template-rows: 360px auto;
          padding: 16px;
        }
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>
"""


@app.callback(
    Output("dataset-dropdown", "options"),
    Output("dataset-dropdown", "value"),
    Output("dataset-status-message", "children"),
    Input("dataset-refresh", "n_intervals"),
    State("dataset-dropdown", "value"),
    prevent_initial_call=False,
)
def refresh_datasets(
    n_intervals: int,
    current_dataset_id: str | None,
) -> tuple[list[dict[str, str]] | Any, str | None | Any, str]:
    del n_intervals
    available_datasets, error = load_datasets()
    if error:
        return no_update, no_update, error

    options = dropdown_options(available_datasets)
    dataset_ids = {item["id"] for item in available_datasets}
    selected_dataset_id = (
        current_dataset_id
        if current_dataset_id in dataset_ids
        else available_datasets[0]["id"] if available_datasets else None
    )
    return options, selected_dataset_id, ""


@app.callback(
    Output("dimension-dropdown", "options"),
    Output("dimension-dropdown", "value"),
    Output("metric-dropdown", "options"),
    Output("metric-dropdown", "value"),
    Output("schema-status-message", "children"),
    Input("dataset-dropdown", "value"),
    prevent_initial_call=False,
)
def update_schema(dataset_id: str | None) -> tuple[
    list[dict[str, str]],
    list[str],
    list[dict[str, str]],
    list[str],
    str,
]:
    if dataset_id is None:
        return [], [], [], [], "No dataset is available"

    try:
        schema = api_get(f"/datasets/{dataset_id}/schema")
    except requests.RequestException as exc:
        return [], [], [], [], f"API request failed: {exc}"

    dimension_options = dropdown_options(schema["dimensions"])
    metric_options = dropdown_options(schema["metrics"])
    default_dimensions = [dimension_options[0]["value"]] if dimension_options else []
    default_metrics = [metric_options[0]["value"]] if metric_options else []
    return dimension_options, default_dimensions, metric_options, default_metrics, ""


@app.callback(
    Output("result-table", "columns"),
    Output("result-table", "data"),
    Output("result-graph", "figure"),
    Output("query-status-message", "children"),
    Input("run-button", "n_clicks"),
    State("dataset-dropdown", "value"),
    State("dimension-dropdown", "value"),
    State("metric-dropdown", "value"),
    State("chart-type-dropdown", "value"),
    prevent_initial_call=True,
)
def run_query(
    n_clicks: int,
    dataset_id: str | None,
    dimensions: list[str] | None,
    metrics: list[str] | None,
    chart_type: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], go.Figure, str]:
    if n_clicks == 0:
        return [], [], empty_figure("Run a query"), ""
    if dataset_id is None:
        return [], [], empty_figure("No dataset selected"), "Select a dataset"

    selected_dimensions = dimensions or []
    selected_metrics = metrics or []
    payload = {
        "dataset": dataset_id,
        "dimensions": selected_dimensions,
        "metrics": selected_metrics,
        "filters": [],
        "sort": [{"field": selected_metrics[0], "direction": "desc"}] if selected_metrics else [],
        "limit": 100,
    }

    try:
        result = api_post("/query", payload)
    except requests.RequestException as exc:
        return [], [], empty_figure("Query failed"), f"API request failed: {exc}"

    columns = [{"name": item["label"], "id": item["id"]} for item in result["columns"]]
    rows = result["rows"]
    figure = build_figure(rows, selected_dimensions, selected_metrics, chart_type)
    return columns, rows, figure, ""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
