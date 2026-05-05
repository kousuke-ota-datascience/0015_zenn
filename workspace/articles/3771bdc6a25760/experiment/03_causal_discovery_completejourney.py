from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from myproj.io.config_resolver import find_project_root, load_dataset_definition
from myproj.io.file_io import FileConfigRegistry, FileIOUtils
from myproj.logger.custom_logger import CustomLogger


DATASET_YAML = Path("shared/py/myproj/conf/dataset/completejourney/10_interim.yaml")
DEFAULT_CAMPAIGN_ID = "18"
DEFAULT_OUTPUT_DIR = Path("articles/3771bdc6a25760/experiment/causal_discovery")
LOGGER_NAME = "causal_discovery_completejourney"


AGE_ORDER = {
    "19-24": 21.5,
    "25-34": 29.5,
    "35-44": 39.5,
    "45-54": 49.5,
    "55-64": 59.5,
    "65+": 70.0,
}
INCOME_ORDER = {
    "Under 15K": 7.5,
    "15-24K": 19.5,
    "25-34K": 29.5,
    "35-49K": 42.0,
    "50-74K": 62.0,
    "75-99K": 87.0,
    "100-124K": 112.0,
    "125-149K": 137.0,
    "150-174K": 162.0,
    "175-199K": 187.0,
    "200-249K": 225.0,
    "250K+": 275.0,
}


@dataclass(frozen=True)
class CampaignWindow:
    campaign_id: str
    start_week: int
    end_week: int
    pre_start_week: int
    pre_end_week: int


def load_completejourney_tables(
    *,
    project_root: Path,
    dataset_yaml: Path,
) -> dict[str, pd.DataFrame]:
    logger = CustomLogger(
        LOGGER_NAME,
        Path(__file__).with_suffix(".log"),
    ).get_logger()
    dataset_definition = load_dataset_definition(dataset_yaml, project_root)
    registry = FileConfigRegistry.from_mapping(dataset_definition)
    file_io = FileIOUtils(logger)

    return {
        entry: file_io.read_file(registry.read_config(entry), use_dask=False)
        for entry in (
            "campaign_descriptions",
            "campaigns",
            "demographics",
            "transactions",
        )
    }


def convert_campaign_dates_to_weeks(
    campaign_descriptions: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    first_transaction_date = pd.to_datetime(
        transactions["transaction_timestamp"], unit="s"
    ).min()
    first_transaction_date = first_transaction_date.normalize()

    campaigns = campaign_descriptions.copy()
    campaigns["start_dt"] = pd.to_datetime(
        campaigns["start_date"], unit="D", origin="unix"
    )
    campaigns["end_dt"] = pd.to_datetime(campaigns["end_date"], unit="D", origin="unix")
    campaigns["start_week"] = (
        (campaigns["start_dt"] - first_transaction_date).dt.days // 7 + 1
    ).astype(int)
    campaigns["end_week"] = (
        (campaigns["end_dt"] - first_transaction_date).dt.days // 7 + 1
    ).astype(int)
    return campaigns


def select_campaign_window(
    *,
    campaign_descriptions: pd.DataFrame,
    transactions: pd.DataFrame,
    campaign_id: str,
    pre_weeks: int,
) -> CampaignWindow:
    campaigns = convert_campaign_dates_to_weeks(campaign_descriptions, transactions)
    matched = campaigns.loc[campaigns["campaign_id"].astype(str).eq(campaign_id)]
    if matched.empty:
        raise ValueError(f"unknown campaign_id: {campaign_id}")

    campaign = matched.iloc[0]
    min_week = int(transactions["week"].min())
    max_week = int(transactions["week"].max())
    start_week = max(int(campaign["start_week"]), min_week)
    end_week = min(int(campaign["end_week"]), max_week)
    pre_end_week = start_week - 1
    pre_start_week = max(min_week, start_week - pre_weeks)

    if pre_start_week > pre_end_week:
        raise ValueError(
            f"campaign {campaign_id} has no pre-treatment weeks in transactions."
        )
    if start_week > end_week:
        raise ValueError(f"campaign {campaign_id} has no outcome weeks in transactions.")

    return CampaignWindow(
        campaign_id=campaign_id,
        start_week=start_week,
        end_week=end_week,
        pre_start_week=pre_start_week,
        pre_end_week=pre_end_week,
    )


def aggregate_transactions(
    transactions: pd.DataFrame,
    *,
    start_week: int,
    end_week: int,
    prefix: str,
) -> pd.DataFrame:
    window = transactions.loc[
        transactions["week"].between(start_week, end_week),
        [
            "household_id",
            "basket_id",
            "quantity",
            "sales_value",
            "retail_disc",
            "coupon_disc",
            "coupon_match_disc",
        ],
    ]
    aggregated = window.groupby("household_id", observed=True).agg(
        baskets=("basket_id", "nunique"),
        quantity=("quantity", "sum"),
        sales_value=("sales_value", "sum"),
        retail_disc=("retail_disc", "sum"),
        coupon_disc=("coupon_disc", "sum"),
        coupon_match_disc=("coupon_match_disc", "sum"),
    )
    aggregated.columns = [f"{prefix}_{column}" for column in aggregated.columns]
    return aggregated


def build_model_frame(
    tables: dict[str, pd.DataFrame],
    *,
    campaign_id: str,
    pre_weeks: int,
) -> pd.DataFrame:
    transactions = tables["transactions"]
    campaigns = tables["campaigns"]
    demographics = tables["demographics"]
    window = select_campaign_window(
        campaign_descriptions=tables["campaign_descriptions"],
        transactions=transactions,
        campaign_id=campaign_id,
        pre_weeks=pre_weeks,
    )

    households = pd.Index(transactions["household_id"].dropna().unique(), name="household_id")
    frame = pd.DataFrame(index=households)
    treated_households = campaigns.loc[
        campaigns["campaign_id"].astype(str).eq(campaign_id), "household_id"
    ]
    frame["treated"] = frame.index.isin(treated_households).astype(int)

    pre = aggregate_transactions(
        transactions,
        start_week=window.pre_start_week,
        end_week=window.pre_end_week,
        prefix="pre",
    )
    outcome = aggregate_transactions(
        transactions,
        start_week=window.start_week,
        end_week=window.end_week,
        prefix="outcome",
    )
    frame = frame.join(pre).join(outcome)
    transaction_columns = [column for column in frame.columns if column != "treated"]
    frame[transaction_columns] = frame[transaction_columns].fillna(0.0)

    demographics = demographics.set_index("household_id").copy()
    demographic_columns = [
        "age",
        "income",
        "home_ownership",
        "marital_status",
        "household_size",
        "household_comp",
        "kids_count",
    ]
    frame = frame.join(demographics[demographic_columns])
    frame[demographic_columns] = frame[demographic_columns].astype("string").fillna("Unknown")
    return frame.reset_index()


def categorical_midpoint(
    series: pd.Series,
    mapping: dict[str, float],
    *,
    unknown_value: float,
) -> pd.Series:
    mapped = series.astype("string").map(mapping).astype("float64")
    return mapped.fillna(unknown_value)


def numeric_category(series: pd.Series, *, unknown_value: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(series.astype("string"), errors="coerce")
    return numeric.fillna(unknown_value).astype("float64")


def build_discovery_frame(model_frame: pd.DataFrame) -> pd.DataFrame:
    age_unknown = model_frame["age"].astype("string").eq("Unknown").astype(float)
    income_unknown = model_frame["income"].astype("string").eq("Unknown").astype(float)

    discovery = pd.DataFrame(
        {
            "age_midpoint": categorical_midpoint(
                model_frame["age"],
                AGE_ORDER,
                unknown_value=np.median(list(AGE_ORDER.values())),
            ),
            "age_unknown": age_unknown,
            "income_midpoint_k": categorical_midpoint(
                model_frame["income"],
                INCOME_ORDER,
                unknown_value=np.median(list(INCOME_ORDER.values())),
            ),
            "income_unknown": income_unknown,
            "homeowner": model_frame["home_ownership"]
            .astype("string")
            .eq("Homeowner")
            .astype(float),
            "married": model_frame["marital_status"]
            .astype("string")
            .eq("Married")
            .astype(float),
            "household_size": numeric_category(model_frame["household_size"]),
            "kids_count": numeric_category(model_frame["kids_count"]),
            "pre_baskets": model_frame["pre_baskets"].astype(float),
            "pre_quantity": model_frame["pre_quantity"].astype(float),
            "pre_sales_value": model_frame["pre_sales_value"].astype(float),
            "pre_retail_disc": model_frame["pre_retail_disc"].astype(float),
            "pre_coupon_disc": model_frame["pre_coupon_disc"].astype(float),
            "treated": model_frame["treated"].astype(float),
            "outcome_baskets": model_frame["outcome_baskets"].astype(float),
            "outcome_quantity": model_frame["outcome_quantity"].astype(float),
            "outcome_sales_value": model_frame["outcome_sales_value"].astype(float),
            "outcome_retail_disc": model_frame["outcome_retail_disc"].astype(float),
            "outcome_coupon_disc": model_frame["outcome_coupon_disc"].astype(float),
        }
    )
    return discovery


def drop_collinear_columns(
    frame: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    corr = frame.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    columns_to_drop = [
        column for column in upper.columns if any(upper[column] >= threshold)
    ]
    return frame.drop(columns=columns_to_drop)


def standardize(
    frame: pd.DataFrame,
    *,
    collinearity_threshold: float,
) -> pd.DataFrame:
    standardized = frame.copy()
    std = standardized.std(axis=0)
    non_constant = std[std > 0].index
    standardized = standardized.loc[:, non_constant]
    standardized = drop_collinear_columns(
        standardized,
        threshold=collinearity_threshold,
    )
    return (standardized - standardized.mean(axis=0)) / standardized.std(axis=0)


def build_background_knowledge(node_names: list[str]):
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    background_knowledge = BackgroundKnowledge()
    nodes = {name: GraphNode(name) for name in node_names}

    baseline = {
        "age_midpoint",
        "age_unknown",
        "income_midpoint_k",
        "income_unknown",
        "homeowner",
        "married",
        "household_size",
        "kids_count",
    }
    pre = {
        "pre_baskets",
        "pre_quantity",
        "pre_sales_value",
        "pre_retail_disc",
        "pre_coupon_disc",
    }
    treatment = {"treated"}
    outcome = {
        "outcome_baskets",
        "outcome_quantity",
        "outcome_sales_value",
        "outcome_retail_disc",
        "outcome_coupon_disc",
    }
    tiers = [baseline, pre, treatment, outcome]
    for tier_index, tier_names in enumerate(tiers):
        for name in sorted(tier_names.intersection(node_names)):
            background_knowledge.add_node_to_tier(nodes[name], tier_index)

    return background_knowledge


def run_pc(
    frame: pd.DataFrame,
    *,
    alpha: float,
    use_background_knowledge: bool,
):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    node_names = list(frame.columns)
    background_knowledge = (
        build_background_knowledge(node_names) if use_background_knowledge else None
    )
    return pc(
        frame.to_numpy(),
        alpha=alpha,
        indep_test=fisherz,
        stable=True,
        background_knowledge=background_knowledge,
        node_names=node_names,
        show_progress=False,
    )


def endpoint_name(endpoint) -> str:
    return endpoint.name.lower()


def edge_symbol(edge) -> str:
    endpoint1 = edge.get_endpoint1().name
    endpoint2 = edge.get_endpoint2().name
    if endpoint1 == "TAIL" and endpoint2 == "ARROW":
        return "-->"
    if endpoint1 == "TAIL" and endpoint2 == "TAIL":
        return "---"
    if endpoint1 == "ARROW" and endpoint2 == "ARROW":
        return "<->"
    if endpoint1 == "CIRCLE" and endpoint2 == "ARROW":
        return "o->"
    if endpoint1 == "TAIL" and endpoint2 == "CIRCLE":
        return "--o"
    if endpoint1 == "CIRCLE" and endpoint2 == "CIRCLE":
        return "o-o"
    return f"{endpoint1}-{endpoint2}"


def edge_records(causal_graph) -> pd.DataFrame:
    records = []
    for edge in causal_graph.G.get_graph_edges():
        records.append(
            {
                "source": edge.get_node1().get_name(),
                "target": edge.get_node2().get_name(),
                "endpoint_source": endpoint_name(edge.get_endpoint1()),
                "endpoint_target": endpoint_name(edge.get_endpoint2()),
                "edge": edge_symbol(edge),
            }
        )
    return pd.DataFrame(records).sort_values(["source", "target"]).reset_index(drop=True)


def mermaid_edge(row: pd.Series) -> str:
    source = row["source"]
    target = row["target"]
    edge = row["edge"]
    if edge == "-->":
        return f"    {source}[{source}] --> {target}[{target}]"
    if edge == "---":
        return f"    {source}[{source}] --- {target}[{target}]"
    if edge == "<->":
        return f"    {source}[{source}] <--> {target}[{target}]"
    if edge == "o->":
        return f"    {source}[{source}] o--> {target}[{target}]"
    if edge == "--o":
        return f"    {source}[{source}] --o {target}[{target}]"
    return f"    {source}[{source}] -. {edge} .- {target}[{target}]"


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No edges discovered._"

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_outputs(
    *,
    causal_graph,
    discovery_frame: pd.DataFrame,
    edges: pd.DataFrame,
    output_dir: Path,
    alpha: float,
    collinearity_threshold: float,
    campaign_id: str,
    pre_weeks: int,
    use_background_knowledge: bool,
) -> None:
    from causallearn.utils.GraphUtils import GraphUtils

    output_dir.mkdir(parents=True, exist_ok=True)
    discovery_frame.to_csv(output_dir / "causal_discovery_input.csv", index=False)
    edges.to_csv(output_dir / "pc_edges.csv", index=False)

    dot = GraphUtils.to_pydot(causal_graph.G, labels=list(discovery_frame.columns))
    (output_dir / "pc_graph.dot").write_text(dot.to_string(), encoding="utf-8")

    mermaid_lines = ["```mermaid", "flowchart LR"]
    mermaid_lines.extend(mermaid_edge(row) for _, row in edges.iterrows())
    mermaid_lines.append("```")

    report = [
        "# causal-learn PC Causal Discovery",
        "",
        f"- campaign_id: `{campaign_id}`",
        f"- pre_weeks: `{pre_weeks}`",
        f"- alpha: `{alpha}`",
        f"- collinearity_threshold: `{collinearity_threshold}`",
        f"- background_knowledge: `{use_background_knowledge}`",
        f"- samples: `{len(discovery_frame)}`",
        f"- variables: `{len(discovery_frame.columns)}`",
        f"- edges: `{len(edges)}`",
        "",
        "## Graph",
        "",
        *mermaid_lines,
        "",
        "## Edges",
        "",
        dataframe_to_markdown(edges),
        "",
        "## Interpretation",
        "",
        "- PC は条件付き独立性から CPDAG を推定する。矢印がない辺は、データと仮定だけでは向きが決まっていない。",
        "- `fisherz` は連続・線形 Gaussian 近似の検定なので、二値変数や順序化したカテゴリを含む今回の結果は探索的に読む。",
        "- `background_knowledge=True` では、世帯属性 -> 事前購買 -> キャンペーン対象 -> 結果期間購買、という時間順に反する向きを禁止している。",
    ]
    (output_dir / "pc_graph.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover causal graph structure in completejourney with causal-learn PC.",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dataset-yaml", type=Path, default=None)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--pre-weeks", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--collinearity-threshold", type=float, default=0.995)
    parser.add_argument(
        "--no-background-knowledge",
        action="store_true",
        help="Run PC without temporal tier constraints.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else find_project_root(Path(__file__))
    )
    dataset_yaml = (
        args.dataset_yaml.resolve()
        if args.dataset_yaml is not None
        else project_root / DATASET_YAML
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / DEFAULT_OUTPUT_DIR
    )

    tables = load_completejourney_tables(
        project_root=project_root,
        dataset_yaml=dataset_yaml,
    )
    model_frame = build_model_frame(
        tables,
        campaign_id=str(args.campaign_id),
        pre_weeks=args.pre_weeks,
    )
    discovery_frame = build_discovery_frame(model_frame)
    standardized = standardize(
        discovery_frame,
        collinearity_threshold=args.collinearity_threshold,
    )
    causal_graph = run_pc(
        standardized,
        alpha=args.alpha,
        use_background_knowledge=not args.no_background_knowledge,
    )
    edges = edge_records(causal_graph)
    write_outputs(
        causal_graph=causal_graph,
        discovery_frame=discovery_frame.loc[:, standardized.columns],
        edges=edges,
        output_dir=output_dir,
        alpha=args.alpha,
        collinearity_threshold=args.collinearity_threshold,
        campaign_id=str(args.campaign_id),
        pre_weeks=args.pre_weeks,
        use_background_knowledge=not args.no_background_knowledge,
    )

    print(f"samples: {len(discovery_frame):,}")
    print(f"variables: {len(standardized.columns):,}")
    print(f"edges: {len(edges):,}")
    print(f"output_dir: {output_dir}")
    print(edges.to_string(index=False))


if __name__ == "__main__":
    main()
