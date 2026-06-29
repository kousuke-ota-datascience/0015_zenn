#!/usr/bin/env python3
# coding: utf-8
# Converted from articles/5132eae5e3dd99/notebooks/04_causal_inference_completejourney.ipynb

# %% [markdown] cell 0
# # Complete Journey Causal Inference
# 
# このノートブックは `03_causal_discovery_completejourney.ipynb` の出力したエッジ集合を読み込み、各 directed edge の重みを推定する。
# 
# 重要: ここで推定する重みは、発見済みグラフを前提にした線形回帰係数である。因果効果として解釈するには、グラフ構造、調整集合、未観測交絡なし、線形性などの仮定が必要である。


# %% [code] cell 1
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


# %% [code] cell 2
DATASET_YAML = Path("shared/py/myproj/conf/dataset/completejourney/10_interim.yaml")
DEFAULT_CAMPAIGN_ID = "18"
DEFAULT_DISCOVERY_DIR = Path("articles/5132eae5e3dd99/experiment/causal_discovery")
DEFAULT_OUTPUT_DIR = Path("articles/5132eae5e3dd99/experiment/causal_inference")
LOGGER_NAME = "causal_inference_completejourney"

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


# %% [code] cell 3
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
    inference_frame: pd.DataFrame
    standardized: pd.DataFrame


@dataclass(frozen=True)
class EdgeEffectResult:
    algorithm: str
    source: str
    target: str
    adjustment_set: tuple[str, ...]
    coefficient: float
    standard_error: float
    t_value: float
    n_samples: int
    r_squared: float


# %% [code] cell 4
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


# %% [code] cell 5
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
        inference_frame = self.build_inference_frame(model_frame)
        standardized = self.standardize(inference_frame)
        return PreprocessingResult(
            model_frame=model_frame,
            inference_frame=inference_frame,
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
        campaigns["end_dt"] = pd.to_datetime(campaigns["end_date"], unit="D", origin="unix")
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
            raise ValueError(f"campaign {self.campaign_id} has no outcome weeks in transactions.")

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

        households = pd.Index(transactions["household_id"].dropna().unique(), name="household_id")
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
        frame[demographic_columns] = frame[demographic_columns].astype("string").fillna("Unknown")
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

    def numeric_category(self, series: pd.Series, *, unknown_value: float = 0.0) -> pd.Series:
        numeric = pd.to_numeric(series.astype("string"), errors="coerce")
        return numeric.fillna(unknown_value).astype("float64")

    def build_inference_frame(self, model_frame: pd.DataFrame) -> pd.DataFrame:
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
                "homeowner": model_frame["home_ownership"].astype("string").eq("Homeowner").astype(float),
                "married": model_frame["marital_status"].astype("string").eq("Married").astype(float),
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


# %% [code] cell 6
class CausalInference:
    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        discovery_dir: Path,
        output_dir: Path,
        algorithms: tuple[str, ...],
    ) -> None:
        self.frame = frame
        self.discovery_dir = discovery_dir
        self.output_dir = output_dir
        self.algorithms = algorithms

    def load_edges(self, algorithm: str) -> pd.DataFrame:
        edge_path = self.discovery_dir / algorithm / "edges.csv"
        if not edge_path.exists():
            return pd.DataFrame(
                columns=["source", "target", "endpoint_source", "endpoint_target", "edge"]
            )
        edges = pd.read_csv(edge_path)
        required = {"source", "target", "edge"}
        missing = required.difference(edges.columns)
        if missing:
            raise ValueError(f"{edge_path} is missing required columns: {sorted(missing)}")
        return edges

    def directed_edges(self, edges: pd.DataFrame) -> pd.DataFrame:
        if edges.empty:
            return edges.copy()
        return edges.loc[edges["edge"].eq("-->")].copy()

    def parents_by_target(self, directed_edges: pd.DataFrame) -> dict[str, set[str]]:
        parents: dict[str, set[str]] = {}
        for _, row in directed_edges.iterrows():
            source = str(row["source"])
            target = str(row["target"])
            parents.setdefault(target, set()).add(source)
        return parents

    def estimate_all(self) -> pd.DataFrame:
        records = []
        for algorithm in self.algorithms:
            edges = self.directed_edges(self.load_edges(algorithm))
            parents = self.parents_by_target(edges)
            for _, edge in edges.iterrows():
                source = str(edge["source"])
                target = str(edge["target"])
                if source not in self.frame.columns or target not in self.frame.columns:
                    continue
                adjustment_set = tuple(sorted(parents.get(target, set()).difference({source})))
                result = self.estimate_edge_effect(
                    algorithm=algorithm,
                    source=source,
                    target=target,
                    adjustment_set=adjustment_set,
                )
                records.append(result.__dict__)
        columns = [
            "algorithm",
            "source",
            "target",
            "adjustment_set",
            "coefficient",
            "standard_error",
            "t_value",
            "n_samples",
            "r_squared",
        ]
        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(records).sort_values(["algorithm", "target", "source"]).reset_index(drop=True)

    def estimate_edge_effect(
        self,
        *,
        algorithm: str,
        source: str,
        target: str,
        adjustment_set: tuple[str, ...],
    ) -> EdgeEffectResult:
        regressors = [source, *adjustment_set]
        used_columns = [target, *regressors]
        data = self.frame.loc[:, used_columns].dropna()
        y = data[target].to_numpy(dtype=float)
        x = data.loc[:, regressors].to_numpy(dtype=float)
        x_design = np.column_stack([np.ones(len(data)), x])

        beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
        fitted = x_design @ beta
        residual = y - fitted
        dof = max(len(y) - x_design.shape[1], 1)
        sigma2 = float((residual @ residual) / dof)
        xtx_inv = np.linalg.pinv(x_design.T @ x_design)
        standard_errors = np.sqrt(np.diag(sigma2 * xtx_inv))
        source_index = 1
        coefficient = float(beta[source_index])
        standard_error = float(standard_errors[source_index])
        t_value = coefficient / standard_error if standard_error > 0 else np.nan
        centered = y - y.mean()
        total_sum_squares = float(centered @ centered)
        residual_sum_squares = float(residual @ residual)
        r_squared = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else np.nan

        return EdgeEffectResult(
            algorithm=algorithm,
            source=source,
            target=target,
            adjustment_set=adjustment_set,
            coefficient=coefficient,
            standard_error=standard_error,
            t_value=float(t_value),
            n_samples=len(data),
            r_squared=float(r_squared),
        )

    def write_outputs(self, effects: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        effects.to_csv(self.output_dir / "edge_effects.csv", index=False)
        report = [
            "# Causal Edge Weight Estimation",
            "",
            f"- discovery_dir: `{self.discovery_dir}`",
            f"- samples: `{len(self.frame)}`",
            f"- variables: `{len(self.frame.columns)}`",
            f"- estimated_edges: `{len(effects)}`",
            "",
            "## Assumptions",
            "",
            "- 発見済みグラフの directed edge を正しい候補構造として扱う。",
            "- 各 edge の target を、source と同じ target を持つ他の parents で線形回帰する。",
            "- 未観測交絡、非線形性、測定誤差が強い場合、係数は因果効果ではなく条件付き関連になる。",
            "",
            "## Effects",
            "",
            self.dataframe_to_markdown(effects),
        ]
        (self.output_dir / "edge_effects.md").write_text("\n".join(report), encoding="utf-8")

    def dataframe_to_markdown(self, frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No edge effects estimated._"
        printable = frame.copy()
        printable["adjustment_set"] = printable["adjustment_set"].map(lambda v: ", ".join(v))
        columns = list(printable.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for _, row in printable.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
        return "\n".join(lines)


# %% [code] cell 7
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate edge weights from causal discovery outputs for completejourney.",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dataset-yaml", type=Path, default=None)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--pre-weeks", type=int, default=8)
    parser.add_argument("--collinearity-threshold", type=float, default=0.995)
    parser.add_argument("--discovery-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["pc", "ges", "lingam", "notears"],
        choices=["pc", "ges", "lingam", "notears"],
    )
    return parser.parse_args()


# %% [code] cell 8
# Notebook stub for values normally supplied by parse_args().
args = argparse.Namespace(
    project_root=find_project_root(Path.cwd()),
    dataset_yaml=None,
    campaign_id=DEFAULT_CAMPAIGN_ID,
    pre_weeks=8,
    collinearity_threshold=0.995,
    discovery_dir=None,
    output_dir=None,
    algorithms=("pc", "ges", "lingam", "notears"),
)


# %% [code] cell 9
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
discovery_dir = (
    args.discovery_dir.resolve()
    if args.discovery_dir is not None
    else project_root / DEFAULT_DISCOVERY_DIR
)
output_dir = (
    args.output_dir.resolve()
    if args.output_dir is not None
    else project_root / DEFAULT_OUTPUT_DIR
)


# %% [code] cell 10
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
inference_frame = preprocessing_result.inference_frame
standardized = preprocessing_result.standardized

causal_inference = CausalInference(
    frame=standardized,
    discovery_dir=discovery_dir,
    output_dir=output_dir,
    algorithms=tuple(args.algorithms),
)
effects = causal_inference.estimate_all()
causal_inference.write_outputs(effects)

print(f"samples: {len(standardized):,}")
print(f"variables: {len(standardized.columns):,}")
print(f"discovery_dir: {discovery_dir}")
print(f"output_dir: {output_dir}")
print(effects.to_string(index=False))


# %% [code] cell 11
def main() -> None:
    pass


if __name__ == "__main__":
    main()


