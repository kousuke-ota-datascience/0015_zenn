#!/usr/bin/env python3
# coding: utf-8
# Converted from articles/5132eae5e3dd99/notebooks/03_causal_discovery_completejourney.ipynb

from __future__ import annotations

import argparse
from collections import Counter
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
DEFAULT_OUTPUT_DIR = Path("articles/5132eae5e3dd99/experiment/causal_discovery")
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

BINARY_COLUMNS = {
    "age_unknown",
    "income_unknown",
    "homeowner",
    "married",
    "treated",
}
ORDINAL_MIDPOINT_COLUMNS = {
    "age_midpoint",
    "income_midpoint_k",
}
NUMERIC_CATEGORY_COLUMNS = {
    "household_size",
    "kids_count",
}
COUNT_COLUMNS = {
    "pre_baskets",
    "pre_quantity",
    "outcome_baskets",
    "outcome_quantity",
}
MONETARY_COLUMNS = {
    "pre_sales_value",
    "pre_retail_disc",
    "pre_coupon_disc",
    "outcome_sales_value",
    "outcome_retail_disc",
    "outcome_coupon_disc",
}
PC_INDEP_TESTS = ("fisherz", "kci", "chisq", "gsq")
DISCRETE_PC_INDEP_TESTS = {"chisq", "gsq"}
DEFAULT_ALPHA_GRID = (0.001, 0.005, 0.01, 0.05)


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
    raw_discovery_frame: pd.DataFrame
    discovery_frame: pd.DataFrame
    standardized: pd.DataFrame
    variable_metadata: pd.DataFrame


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
        raw_discovery_frame = self.build_discovery_frame(model_frame)
        discovery_frame, transform_by_column = self.transform_skewed_variables(
            raw_discovery_frame
        )
        standardized = self.standardize(discovery_frame)
        variable_metadata = self.build_variable_metadata(
            columns=list(discovery_frame.columns),
            transform_by_column=transform_by_column,
            retained_columns=list(standardized.columns),
        )
        return PreprocessingResult(
            model_frame=model_frame,
            raw_discovery_frame=raw_discovery_frame,
            discovery_frame=discovery_frame,
            standardized=standardized,
            variable_metadata=variable_metadata,
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

    def transform_skewed_variables(
        self,
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        transformed = frame.copy()
        transform_by_column: dict[str, str] = {}

        for column in sorted(COUNT_COLUMNS.intersection(transformed.columns)):
            values = transformed[column].astype(float)
            if (values < 0).any():
                raise ValueError(f"count column must be non-negative before log1p: {column}")
            transformed[column] = np.log1p(values)
            transform_by_column[column] = "log1p"

        for column in sorted(MONETARY_COLUMNS.intersection(transformed.columns)):
            values = transformed[column].astype(float)
            if (values < 0).any():
                transformed[column] = np.sign(values) * np.log1p(np.abs(values))
                transform_by_column[column] = "signed_log1p"
            else:
                transformed[column] = np.log1p(values)
                transform_by_column[column] = "log1p"

        return transformed, transform_by_column

    def variable_data_type(self, column: str) -> str:
        if column in BINARY_COLUMNS:
            return "binary"
        if column in ORDINAL_MIDPOINT_COLUMNS:
            return "ordinal_midpoint"
        if column in NUMERIC_CATEGORY_COLUMNS:
            return "numeric_category"
        if column in COUNT_COLUMNS:
            return "count"
        if column in MONETARY_COLUMNS:
            return "monetary_or_discount"
        return "numeric"

    def variable_role(self, column: str) -> str:
        if column == "treated":
            return "treatment"
        if column.startswith("pre_"):
            return "pre_treatment_behavior"
        if column.startswith("outcome_"):
            return "outcome"
        return "baseline_covariate"

    def build_variable_metadata(
        self,
        *,
        columns: list[str],
        transform_by_column: dict[str, str],
        retained_columns: list[str],
    ) -> pd.DataFrame:
        retained = set(retained_columns)
        records = []
        for column in columns:
            records.append(
                {
                    "variable": column,
                    "role": self.variable_role(column),
                    "data_type": self.variable_data_type(column),
                    "transform": transform_by_column.get(column, "identity"),
                    "used_in_discovery": column in retained,
                    "fisherz_caution": self.variable_data_type(column)
                    in {
                        "binary",
                        "ordinal_midpoint",
                        "numeric_category",
                        "count",
                        "monetary_or_discount",
                    },
                }
            )
        return pd.DataFrame(records)

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
        pc_indep_test: str = "fisherz",
        alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
        bootstrap_samples: int = 100,
        bootstrap_sample_fraction: float = 1.0,
        random_seed: int = 20260630,
        pc_discrete_bins: int = 4,
    ) -> None:
        if pc_indep_test not in PC_INDEP_TESTS:
            raise ValueError(f"pc_indep_test must be one of {PC_INDEP_TESTS}: {pc_indep_test}")
        if bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be non-negative")
        if not 0 < bootstrap_sample_fraction <= 1:
            raise ValueError("bootstrap_sample_fraction must be in (0, 1]")
        if pc_discrete_bins < 2:
            raise ValueError("pc_discrete_bins must be at least 2")

        self.alpha = alpha
        self.use_background_knowledge = use_background_knowledge
        self.algorithms = tuple(algorithms)
        self.notears_threshold = notears_threshold
        self.pc_indep_test = pc_indep_test
        self.alpha_grid = tuple(sorted({float(value) for value in (*alpha_grid, alpha)}))
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_sample_fraction = bootstrap_sample_fraction
        self.random_seed = random_seed
        self.pc_discrete_bins = pc_discrete_bins

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
            except Exception as exc:
                results[algorithm] = DiscoveryResult(
                    algorithm=algorithm,
                    causal_graph=None,
                    edges=pd.DataFrame(
                        columns=["source", "target", "endpoint_source", "endpoint_target", "edge"]
                    ),
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                )
        return results

    def run_pc(self, frame: pd.DataFrame, *, alpha: float | None = None):
        from causallearn.search.ConstraintBased.PC import pc

        pc_frame = self.prepare_pc_frame(frame)
        node_names = list(pc_frame.columns)
        background_knowledge = (
            self.build_background_knowledge(node_names)
            if self.use_background_knowledge
            else None
        )
        causal_graph = pc(
            pc_frame.to_numpy(),
            alpha=self.alpha if alpha is None else alpha,
            indep_test=self.pc_indep_test,
            stable=True,
            background_knowledge=background_knowledge,
            node_names=node_names,
            show_progress=False,
        )
        return causal_graph, self.graph_edge_records(causal_graph.G)

    def prepare_pc_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.pc_indep_test not in DISCRETE_PC_INDEP_TESTS:
            return frame

        return pd.DataFrame(
            {column: self.discretize_series(frame[column]) for column in frame.columns},
            index=frame.index,
        )

    def discretize_series(self, series: pd.Series) -> pd.Series:
        non_null = series.dropna()
        unique_values = non_null.nunique()
        if unique_values <= 1:
            return pd.Series(np.zeros(len(series), dtype=int), index=series.index)
        if unique_values <= 2:
            codes = pd.Categorical(series).codes
            return pd.Series(codes, index=series.index).astype(int)

        bins = min(self.pc_discrete_bins, unique_values)
        ranked = series.rank(method="average")
        binned = pd.qcut(ranked, q=bins, labels=False, duplicates="drop")
        return pd.Series(binned, index=series.index).fillna(0).astype(int)

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

    def adjacency_key(self, row: pd.Series) -> tuple[str, str]:
        source = str(row["source"])
        target = str(row["target"])
        return tuple(sorted((source, target)))

    def run_pc_alpha_sensitivity(
        self,
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary_columns = ["alpha", "status", "edges", "message"]
        edge_columns = [
            "alpha",
            "source",
            "target",
            "endpoint_source",
            "endpoint_target",
            "edge",
        ]
        summary_records = []
        edge_records = []

        for alpha in self.alpha_grid:
            try:
                _, edges = self.run_pc(frame, alpha=alpha)
            except Exception as exc:
                summary_records.append(
                    {
                        "alpha": alpha,
                        "status": "failed",
                        "edges": 0,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            summary_records.append(
                {
                    "alpha": alpha,
                    "status": "ok",
                    "edges": len(edges),
                    "message": "",
                }
            )
            for _, edge in edges.iterrows():
                record = {"alpha": alpha}
                record.update(edge.to_dict())
                edge_records.append(record)

        return (
            pd.DataFrame(summary_records, columns=summary_columns),
            pd.DataFrame(edge_records, columns=edge_columns),
        )

    def bootstrap_pc_edge_stability(
        self,
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        stability_columns = [
            "source",
            "target",
            "selection_count",
            "successful_bootstrap_samples",
            "selection_probability",
            "example_edge",
        ]
        failure_columns = ["bootstrap_iteration", "message"]
        if self.bootstrap_samples <= 0:
            return (
                pd.DataFrame(columns=stability_columns),
                pd.DataFrame(columns=failure_columns),
            )

        sample_size = max(2, int(round(len(frame) * self.bootstrap_sample_fraction)))
        rng = np.random.default_rng(self.random_seed)
        edge_counts: Counter[tuple[str, str]] = Counter()
        example_edges: dict[tuple[str, str], str] = {}
        failure_records = []

        for bootstrap_iteration in range(self.bootstrap_samples):
            sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            sample = frame.sample(
                n=sample_size,
                replace=True,
                random_state=sample_seed,
            ).reset_index(drop=True)
            try:
                _, edges = self.run_pc(sample, alpha=self.alpha)
            except Exception as exc:
                failure_records.append(
                    {
                        "bootstrap_iteration": bootstrap_iteration,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            for _, edge in edges.iterrows():
                key = self.adjacency_key(edge)
                edge_counts[key] += 1
                example_edges.setdefault(key, str(edge["edge"]))

        successful_samples = self.bootstrap_samples - len(failure_records)
        stability_records = []
        for (source, target), count in sorted(edge_counts.items()):
            stability_records.append(
                {
                    "source": source,
                    "target": target,
                    "selection_count": count,
                    "successful_bootstrap_samples": successful_samples,
                    "selection_probability": (
                        count / successful_samples if successful_samples else np.nan
                    ),
                    "example_edge": example_edges[(source, target)],
                }
            )

        return (
            pd.DataFrame(stability_records, columns=stability_columns),
            pd.DataFrame(failure_records, columns=failure_columns),
        )

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

    def variable_diagnostics(
        self,
        frame: pd.DataFrame,
        variable_metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        records = []
        for column in frame.columns:
            values = frame[column]
            records.append(
                {
                    "variable": column,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "zero_share": float(values.eq(0).mean()),
                    "unique_values": int(values.nunique(dropna=True)),
                }
            )
        diagnostics = pd.DataFrame(records)
        return variable_metadata.merge(diagnostics, on="variable", how="left")

    def write_outputs(
        self,
        *,
        results: dict[str, DiscoveryResult],
        raw_discovery_frame: pd.DataFrame,
        discovery_frame: pd.DataFrame,
        standardized_frame: pd.DataFrame,
        variable_metadata: pd.DataFrame,
        output_dir: Path,
        collinearity_threshold: float,
        campaign_id: str,
        pre_weeks: int,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_discovery_frame.to_csv(output_dir / "causal_discovery_input_raw.csv", index=False)
        discovery_frame.to_csv(output_dir / "causal_discovery_input.csv", index=False)
        standardized_frame.to_csv(
            output_dir / "causal_discovery_input_standardized.csv",
            index=False,
        )
        variable_metadata.to_csv(output_dir / "variable_metadata.csv", index=False)
        self.variable_diagnostics(discovery_frame, variable_metadata).to_csv(
            output_dir / "variable_diagnostics.csv",
            index=False,
        )

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
                    variable_metadata=variable_metadata,
                    output_dir=algorithm_dir,
                    collinearity_threshold=collinearity_threshold,
                    campaign_id=campaign_id,
                    pre_weeks=pre_weeks,
                )

        if "pc" in results and results["pc"].status == "ok":
            pc_dir = output_dir / "pc"
            alpha_summary, alpha_edges = self.run_pc_alpha_sensitivity(standardized_frame)
            alpha_summary.to_csv(pc_dir / "alpha_sensitivity.csv", index=False)
            alpha_edges.to_csv(pc_dir / "alpha_sensitivity_edges.csv", index=False)

            stability, failures = self.bootstrap_pc_edge_stability(standardized_frame)
            stability.to_csv(pc_dir / "edge_stability.csv", index=False)
            failures.to_csv(pc_dir / "bootstrap_failures.csv", index=False)

        summary = pd.DataFrame(summary_records)
        summary.to_csv(output_dir / "algorithm_summary.csv", index=False)

    def write_algorithm_report(
        self,
        *,
        result: DiscoveryResult,
        discovery_frame: pd.DataFrame,
        variable_metadata: pd.DataFrame,
        output_dir: Path,
        collinearity_threshold: float,
        campaign_id: str,
        pre_weeks: int,
    ) -> None:
        mermaid_lines = ["```mermaid", "flowchart LR"]
        mermaid_lines.extend(self.mermaid_edge(row) for _, row in result.edges.iterrows())
        mermaid_lines.append("```")

        used_metadata = variable_metadata.loc[variable_metadata["used_in_discovery"]]
        binary_variables = used_metadata.loc[
            used_metadata["data_type"].eq("binary"),
            "variable",
        ].tolist()
        transformed_variables = [
            f"{row.variable} ({row.transform})"
            for row in used_metadata.itertuples(index=False)
            if row.transform != "identity"
        ]

        report = [
            f"# {result.algorithm.upper()} Causal Discovery",
            "",
            f"- campaign_id: `{campaign_id}`",
            f"- pre_weeks: `{pre_weeks}`",
            f"- alpha: `{self.alpha}`",
            f"- collinearity_threshold: `{collinearity_threshold}`",
            f"- background_knowledge: `{self.use_background_knowledge}`",
            f"- pc_indep_test: `{self.pc_indep_test}`",
            f"- pc_discrete_bins: `{self.pc_discrete_bins}`",
            f"- bootstrap_samples: `{self.bootstrap_samples}`",
            f"- bootstrap_sample_fraction: `{self.bootstrap_sample_fraction}`",
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
            "## Data and Test Assumptions",
            "",
            f"- binary_variables: `{', '.join(binary_variables) or 'none'}`",
            f"- transformed_variables: `{', '.join(transformed_variables) or 'none'}`",
            "- count and monetary/discount variables are transformed before standardization; non-negative variables use log1p and signed discount-like variables use signed_log1p.",
            "- Fisher-z PC should be interpreted as a partial-correlation conditional-independence search when binary, ordinal-midpoint, numeric-category, count, or zero-inflated monetary variables are included.",
            "- `--pc-indep-test kci` is available as a nonlinear CI alternative, but it is not a complete mixed-type solution for binary plus continuous data.",
            "- `--pc-indep-test gsq` or `chisq` is available for a discrete-data sensitivity run; continuous variables are quantile-discretized with `--pc-discrete-bins`.",
            "- A proper mixed-type CI test is not implemented in this dependency set; treat that as a remaining methodological limitation rather than solved by standardization.",
            "- PC alpha sensitivity is written to `pc/alpha_sensitivity.csv` and `pc/alpha_sensitivity_edges.csv` when PC succeeds.",
            "- PC bootstrap edge stability is written to `pc/edge_stability.csv` when PC succeeds and `--bootstrap-samples` is positive.",
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
    parser.add_argument(
        "--pc-indep-test",
        choices=PC_INDEP_TESTS,
        default="fisherz",
        help="Conditional-independence test for PC. fisherz is a linear-Gaussian approximation; gsq/chisq discretize continuous variables first.",
    )
    parser.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_ALPHA_GRID),
        help="Alpha values for PC sensitivity analysis. The main --alpha is added if absent.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-sample-fraction", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=20260630)
    parser.add_argument("--pc-discrete-bins", type=int, default=4)
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
#     pc_indep_test="fisherz",
#     alpha_grid=list(DEFAULT_ALPHA_GRID),
#     bootstrap_samples=100,
#     bootstrap_sample_fraction=1.0,
#     random_seed=20260630,
#     pc_discrete_bins=4,
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
    raw_discovery_frame = preprocessing_result.raw_discovery_frame
    discovery_frame = preprocessing_result.discovery_frame
    standardized = preprocessing_result.standardized
    variable_metadata = preprocessing_result.variable_metadata

    causal_discovery = CausalDiscovery(
        alpha=args.alpha,
        use_background_knowledge=not args.no_background_knowledge,
        algorithms=args.algorithms,
        notears_threshold=args.notears_threshold,
        pc_indep_test=args.pc_indep_test,
        alpha_grid=tuple(args.alpha_grid),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_sample_fraction=args.bootstrap_sample_fraction,
        random_seed=args.random_seed,
        pc_discrete_bins=args.pc_discrete_bins,
    )

    ## run_all するパターン。--algorithms 引数が有効に使われる。
    ## ex: uv run python articles/5132eae5e3dd99/experiment/03_causal_discovery_completejourney.py  --algorithms pc ges
    discovery_results = causal_discovery.run_all(standardized)

    ## run_pc returns a tuple, so convert it to the DiscoveryResult shape expected by write_outputs.
    # causal_graph, edges = causal_discovery.run_pc(standardized)
    # discovery_results = {
    #     "pc": DiscoveryResult(
    #         algorithm="pc",
    #         causal_graph=causal_graph,
    #         edges=edges,
    #         status="ok",
    #         message="",
    #     )
    # }

    causal_discovery.write_outputs(
        results=discovery_results,
        raw_discovery_frame=raw_discovery_frame.loc[:, standardized.columns],
        discovery_frame=discovery_frame.loc[:, standardized.columns],
        standardized_frame=standardized,
        variable_metadata=variable_metadata,
        output_dir=output_dir,
        collinearity_threshold=args.collinearity_threshold,
        campaign_id=str(args.campaign_id),
        pre_weeks=args.pre_weeks,
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
    print(f"pc_indep_test: {args.pc_indep_test}")
    print(f"bootstrap_samples: {args.bootstrap_samples:,}")
    print(f"output_dir: {output_dir}")
    print(summary.to_string(index=False))



if __name__ == "__main__":
    main()

