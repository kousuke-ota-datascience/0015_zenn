#!/usr/bin/env python3
# coding: utf-8
# Converted from articles/5132eae5e3dd99/notebooks/03_causal_discovery_completejourney.ipynb

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
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


@dataclass(frozen=True)
class PreprocessingResult:
    model_frame: pd.DataFrame
    discovery_frame: pd.DataFrame
    standardized: pd.DataFrame




@dataclass(frozen=True)
class DiscoveryResult:
    algorithm: str
    causal_graph: object | None
    edges: pd.DataFrame
    status: str
    message: str


class CompleteJourneyDataLoader:
    def __init__(self, *, project_root: Path, dataset_yaml: Path) -> None:
        self.project_root = project_root
        self.dataset_yaml = dataset_yaml

    def load_tables(self) -> dict[str, pd.DataFrame]:
        logger = CustomLogger(
            LOGGER_NAME,
            Path.cwd() / f"{LOGGER_NAME}.log",
        ).get_logger()
        dataset_definition = load_dataset_definition(self.dataset_yaml, self.project_root)
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


class AbstractPreprocessor(ABC):
    @abstractmethod
    def preprocess(self) -> PreprocessingResult:
        raise NotImplementedError


class CompleteJourneyPreprocessor(AbstractPreprocessor):
    def __init__(
        self,
        *,
        tables: dict[str, pd.DataFrame],
        campaign_id: str,
        pre_weeks: int,
        collinearity_threshold: float,
    ) -> None:
        self.tables = tables
        self.campaign_id = campaign_id
        self.pre_weeks = pre_weeks
        self.collinearity_threshold = collinearity_threshold

    def preprocess(self) -> PreprocessingResult:
        model_frame = self.build_model_frame(self.tables)
        discovery_frame = self.build_discovery_frame(model_frame)
        standardized = self.standardize(discovery_frame)
        return PreprocessingResult(
            model_frame=model_frame,
            discovery_frame=discovery_frame,
            standardized=standardized,
        )

    def convert_campaign_dates_to_weeks(
        self,
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
        campaigns["end_dt"] = pd.to_datetime(
            campaigns["end_date"], unit="D", origin="unix"
        )
        campaigns["start_week"] = (
            (campaigns["start_dt"] - first_transaction_date).dt.days // 7 + 1
        ).astype(int)
        campaigns["end_week"] = (
            (campaigns["end_dt"] - first_transaction_date).dt.days // 7 + 1
        ).astype(int)
        return campaigns

    def select_campaign_window(
        self,
        *,
        campaign_descriptions: pd.DataFrame,
        transactions: pd.DataFrame,
    ) -> CampaignWindow:
        campaigns = self.convert_campaign_dates_to_weeks(campaign_descriptions, transactions)
        matched = campaigns.loc[campaigns["campaign_id"].astype(str).eq(self.campaign_id)]
        if matched.empty:
            raise ValueError(f"unknown campaign_id: {self.campaign_id}")

        campaign = matched.iloc[0]
        min_week = int(transactions["week"].min())
        max_week = int(transactions["week"].max())
        start_week = max(int(campaign["start_week"]), min_week)
        end_week = min(int(campaign["end_week"]), max_week)
        pre_end_week = start_week - 1
        pre_start_week = max(min_week, start_week - self.pre_weeks)

        if pre_start_week > pre_end_week:
            raise ValueError(
                f"campaign {self.campaign_id} has no pre-treatment weeks in transactions."
            )
        if start_week > end_week:
            raise ValueError(
                f"campaign {self.campaign_id} has no outcome weeks in transactions."
            )

        return CampaignWindow(
            campaign_id=self.campaign_id,
            start_week=start_week,
            end_week=end_week,
            pre_start_week=pre_start_week,
            pre_end_week=pre_end_week,
        )

    def aggregate_transactions(
        self,
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

    def build_model_frame(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        transactions = tables["transactions"]
        campaigns = tables["campaigns"]
        demographics = tables["demographics"]
        window = self.select_campaign_window(
            campaign_descriptions=tables["campaign_descriptions"],
            transactions=transactions,
        )

        households = pd.Index(
            transactions["household_id"].dropna().unique(), name="household_id"
        )
        frame = pd.DataFrame(index=households)
        treated_households = campaigns.loc[
            campaigns["campaign_id"].astype(str).eq(self.campaign_id), "household_id"
        ]
        frame["treated"] = frame.index.isin(treated_households).astype(int)

        pre = self.aggregate_transactions(
            transactions,
            start_week=window.pre_start_week,
            end_week=window.pre_end_week,
            prefix="pre",
        )
        outcome = self.aggregate_transactions(
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
        frame[demographic_columns] = (
            frame[demographic_columns].astype("string").fillna("Unknown")
        )
        return frame.reset_index()

    def categorical_midpoint(
        self,
        series: pd.Series,
        mapping: dict[str, float],
        *,
        unknown_value: float,
    ) -> pd.Series:
        mapped = series.astype("string").map(mapping).astype("float64")
        return mapped.fillna(unknown_value)

    def numeric_category(
        self,
        series: pd.Series,
        *,
        unknown_value: float = 0.0,
    ) -> pd.Series:
        numeric = pd.to_numeric(series.astype("string"), errors="coerce")
        return numeric.fillna(unknown_value).astype("float64")

    def build_discovery_frame(self, model_frame: pd.DataFrame) -> pd.DataFrame:
        age_unknown = model_frame["age"].astype("string").eq("Unknown").astype(float)
        income_unknown = model_frame["income"].astype("string").eq("Unknown").astype(float)

        return pd.DataFrame(
            {
                "age_midpoint": self.categorical_midpoint(
                    model_frame["age"],
                    AGE_ORDER,
                    unknown_value=np.median(list(AGE_ORDER.values())),
                ),
                "age_unknown": age_unknown,
                "income_midpoint_k": self.categorical_midpoint(
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
                "household_size": self.numeric_category(model_frame["household_size"]),
                "kids_count": self.numeric_category(model_frame["kids_count"]),
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

    def drop_collinear_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        corr = frame.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        columns_to_drop = [
            column for column in upper.columns if any(upper[column] >= self.collinearity_threshold)
        ]
        return frame.drop(columns=columns_to_drop)

    def standardize(self, frame: pd.DataFrame) -> pd.DataFrame:
        standardized = frame.copy()
        std = standardized.std(axis=0)
        non_constant = std[std > 0].index
        standardized = standardized.loc[:, non_constant]
        standardized = self.drop_collinear_columns(standardized)
        return (standardized - standardized.mean(axis=0)) / standardized.std(axis=0)


class CausalDiscovery:
    def __init__(
        self,
        *,
        alpha: float,
        use_background_knowledge: bool,
        algorithms: tuple[str, ...] = ("pc", "ges", "lingam", "notears"),
        notears_threshold: float = 0.3,
    ) -> None:
        self.alpha = alpha
        self.use_background_knowledge = use_background_knowledge
        self.algorithms = algorithms
        self.notears_threshold = notears_threshold

    def run_all(self, frame: pd.DataFrame) -> dict[str, DiscoveryResult]:
        runners = {
            "pc": self.run_pc,
            "ges": self.run_ges,
            "lingam": self.run_lingam,
            "notears": self.run_notears,
        }
        results = {}
        for algorithm in self.algorithms:
            runner = runners[algorithm]
            try:
                causal_graph, edges = runner(frame)
                results[algorithm] = DiscoveryResult(
                    algorithm=algorithm,
                    causal_graph=causal_graph,
                    edges=edges,
                    status="ok",
                    message="",
                )
            except ImportError as exc:
                results[algorithm] = DiscoveryResult(
                    algorithm=algorithm,
                    causal_graph=None,
                    edges=pd.DataFrame(
                        columns=["source", "target", "endpoint_source", "endpoint_target", "edge"]
                    ),
                    status="skipped",
                    message=str(exc),
                )
        return results

    def run_pc(self, frame: pd.DataFrame):
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        node_names = list(frame.columns)
        background_knowledge = (
            self.build_background_knowledge(node_names)
            if self.use_background_knowledge
            else None
        )
        causal_graph = pc(
            frame.to_numpy(),
            alpha=self.alpha,
            indep_test=fisherz,
            stable=True,
            background_knowledge=background_knowledge,
            node_names=node_names,
            show_progress=False,
        )
        return causal_graph, self.graph_edge_records(causal_graph.G)

    def run_ges(self, frame: pd.DataFrame):
        from causallearn.search.ScoreBased.GES import ges

        result = ges(
            frame.to_numpy(),
            score_func="local_score_marginal_general",
            node_names=list(frame.columns),
        )
        causal_graph = result["G"]
        return causal_graph, self.graph_edge_records(causal_graph)

    def run_lingam(self, frame: pd.DataFrame):
        try:
            from lingam import DirectLiNGAM
        except ImportError as exc:
            raise ImportError(
                "LiNGAM requires the optional `lingam` package. Install it before running this algorithm."
            ) from exc

        model = DirectLiNGAM()
        model.fit(frame.to_numpy())
        edges = self.weight_matrix_edge_records(
            model.adjacency_matrix_,
            node_names=list(frame.columns),
            threshold=0.0,
        )
        return model, edges

    def run_notears(self, frame: pd.DataFrame):
        try:
            from notears.linear import notears_linear
        except ImportError as exc:
            raise ImportError(
                "NOTEARS requires an optional NOTEARS implementation such as the `notears` package. "
                "Install it before running this algorithm."
            ) from exc

        weight_matrix = notears_linear(frame.to_numpy(), lambda1=0.1, loss_type="l2")
        edges = self.weight_matrix_edge_records(
            weight_matrix,
            node_names=list(frame.columns),
            threshold=self.notears_threshold,
        )
        return weight_matrix, edges

    def build_background_knowledge(self, node_names: list[str]):
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

    def endpoint_name(self, endpoint) -> str:
        return endpoint.name.lower()

    def edge_symbol(self, edge) -> str:
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

    def graph_edge_records(self, graph) -> pd.DataFrame:
        records = []
        for edge in graph.get_graph_edges():
            records.append(
                {
                    "source": edge.get_node1().get_name(),
                    "target": edge.get_node2().get_name(),
                    "endpoint_source": self.endpoint_name(edge.get_endpoint1()),
                    "endpoint_target": self.endpoint_name(edge.get_endpoint2()),
                    "edge": self.edge_symbol(edge),
                }
            )
        if not records:
            return pd.DataFrame(
                columns=["source", "target", "endpoint_source", "endpoint_target", "edge"]
            )
        return pd.DataFrame(records).sort_values(["source", "target"]).reset_index(drop=True)

    def weight_matrix_edge_records(
        self,
        weight_matrix: np.ndarray,
        *,
        node_names: list[str],
        threshold: float,
    ) -> pd.DataFrame:
        records = []
        for target_index, source_index in zip(*np.where(np.abs(weight_matrix) > threshold)):
            if source_index == target_index:
                continue
            records.append(
                {
                    "source": node_names[source_index],
                    "target": node_names[target_index],
                    "endpoint_source": "tail",
                    "endpoint_target": "arrow",
                    "edge": "-->",
                    "weight": float(weight_matrix[target_index, source_index]),
                }
            )
        columns = ["source", "target", "endpoint_source", "endpoint_target", "edge", "weight"]
        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(records).sort_values(["source", "target"]).reset_index(drop=True)

    def mermaid_edge(self, row: pd.Series) -> str:
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

    def dataframe_to_markdown(self, frame: pd.DataFrame) -> str:
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
        self,
        *,
        results: dict[str, DiscoveryResult],
        discovery_frame: pd.DataFrame,
        output_dir: Path,
        collinearity_threshold: float,
        campaign_id: str,
        pre_weeks: int,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        discovery_frame.to_csv(output_dir / "causal_discovery_input.csv", index=False)

        summary_records = []
        for algorithm, result in results.items():
            algorithm_dir = output_dir / algorithm
            algorithm_dir.mkdir(parents=True, exist_ok=True)
            result.edges.to_csv(algorithm_dir / "edges.csv", index=False)
            summary_records.append(
                {
                    "algorithm": algorithm,
                    "status": result.status,
                    "edges": len(result.edges),
                    "message": result.message,
                }
            )
            if result.status == "ok":
                self.write_algorithm_report(
                    result=result,
                    discovery_frame=discovery_frame,
                    output_dir=algorithm_dir,
                    collinearity_threshold=collinearity_threshold,
                    campaign_id=campaign_id,
                    pre_weeks=pre_weeks,
                )

        summary = pd.DataFrame(summary_records)
        summary.to_csv(output_dir / "algorithm_summary.csv", index=False)

    def write_algorithm_report(
        self,
        *,
        result: DiscoveryResult,
        discovery_frame: pd.DataFrame,
        output_dir: Path,
        collinearity_threshold: float,
        campaign_id: str,
        pre_weeks: int,
    ) -> None:
        mermaid_lines = ["```mermaid", "flowchart LR"]
        mermaid_lines.extend(self.mermaid_edge(row) for _, row in result.edges.iterrows())
        mermaid_lines.append("```")

        report = [
            f"# {result.algorithm.upper()} Causal Discovery",
            "",
            f"- campaign_id: `{campaign_id}`",
            f"- pre_weeks: `{pre_weeks}`",
            f"- alpha: `{self.alpha}`",
            f"- collinearity_threshold: `{collinearity_threshold}`",
            f"- background_knowledge: `{self.use_background_knowledge}`",
            f"- samples: `{len(discovery_frame)}`",
            f"- variables: `{len(discovery_frame.columns)}`",
            f"- edges: `{len(result.edges)}`",
            "",
            "## Graph",
            "",
            *mermaid_lines,
            "",
            "## Edges",
            "",
            self.dataframe_to_markdown(result.edges),
            "",
            "## Interpretation",
            "",
            "- PC は条件付き独立性から CPDAG を推定する制約ベース手法。",
            "- GES はスコアを改善する貪欲探索で等価クラスを推定するスコアベース手法。",
            "- LiNGAM は線形・非 Gaussian・非巡回などの仮定を使って向き付き構造を推定する。",
            "- NOTEARS は DAG 制約を連続最適化問題として扱う。",
        ]
        (output_dir / "graph.md").write_text("\n".join(report), encoding="utf-8")


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
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["pc", "ges", "lingam", "notears"],
        choices=["pc", "ges", "lingam", "notears"],
    )
    parser.add_argument("--notears-threshold", type=float, default=0.3)
    return parser.parse_args()


# Notebook stub for values normally supplied by parse_args().
# args = argparse.Namespace(
#     project_root=find_project_root(Path.cwd()),
#     dataset_yaml=None,
#     campaign_id=DEFAULT_CAMPAIGN_ID,
#     pre_weeks=8,
#     alpha=0.01,
#     collinearity_threshold=0.995,
#     no_background_knowledge=False,
#     algorithms=("pc", "ges", "lingam", "notears"),
#     notears_threshold=0.3,
#     output_dir=None,
# )

def main() -> None:
    args = parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else find_project_root(Path.cwd())
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


    data_loader = CompleteJourneyDataLoader(
        project_root=project_root,
        dataset_yaml=dataset_yaml,
    )
    tables = data_loader.load_tables()

    preprocessor = CompleteJourneyPreprocessor(
        tables=tables,
        campaign_id=str(args.campaign_id),
        pre_weeks=args.pre_weeks,
        collinearity_threshold=args.collinearity_threshold,
    )
    preprocessing_result = preprocessor.preprocess()
    model_frame = preprocessing_result.model_frame
    discovery_frame = preprocessing_result.discovery_frame
    standardized = preprocessing_result.standardized

    causal_discovery = CausalDiscovery(
        alpha=args.alpha,
        use_background_knowledge=not args.no_background_knowledge,
        algorithms=args.algorithms,
        notears_threshold=args.notears_threshold,
    )
    ## run_all するパターン。これは時間がかかるので、PC のみを実行するパターンに変更。
    ## discovery_results = causal_discovery.run_all(standardized)

    ## run_pc returns a tuple, so convert it to the DiscoveryResult shape expected by write_outputs.
    causal_graph, edges = causal_discovery.run_pc(standardized)
    discovery_results = {
        "pc": DiscoveryResult(
            algorithm="pc",
            causal_graph=causal_graph,
            edges=edges,
            status="ok",
            message="",
        )
    }

    causal_discovery.write_outputs(
        results=discovery_results,
        discovery_frame=discovery_frame.loc[:, standardized.columns],
        output_dir=output_dir,
        collinearity_threshold=args.collinearity_threshold,
        campaign_id=str(args.campaign_id),
        pre_weeks=args.pre_weeks
    )
    summary = pd.DataFrame(
        {
            "algorithm": result.algorithm,
            "status": result.status,
            "edges": len(result.edges),
            "message": result.message,
        }
        for result in discovery_results.values()
    )

    print(f"samples: {len(discovery_frame):,}")
    print(f"variables: {len(standardized.columns):,}")
    print(f"output_dir: {output_dir}")
    print(summary.to_string(index=False))



if __name__ == "__main__":
    main()

