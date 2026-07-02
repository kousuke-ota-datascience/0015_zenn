#!/usr/bin/env python3
# coding: utf-8
# Converted from articles/5132eae5e3dd99/notebooks/04_causal_inference_completejourney.ipynb

# %% [markdown] cell 0
# # Complete Journey Causal Inference
# 
# このノートブックは `03_causal_discovery_completejourney.ipynb` の出力したエッジ集合を読み込み、各 directed edge の重みを推定する。
# 
# 重要: ここで推定する重みは、発見済みグラフを前提にした線形回帰係数である。因果効果として解釈するには、グラフ構造、調整集合、未観測交絡なし、線形性などの仮定が必要である。


from __future__ import annotations

import argparse
import json
import math
import warnings
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
PRE_TREATMENT_COVARIATES = (
    "age_midpoint",
    "age_unknown",
    "income_midpoint_k",
    "income_unknown",
    "household_size",
    "kids_count",
    "pre_baskets",
    "pre_quantity",
    "pre_sales_value",
    "pre_coupon_disc",
    "pre_coupon_match_disc",
    "pre_retail_disc",
    "homeowner_yes",
    "homeowner_unknown",
    "married_yes",
    "married_unknown",
)
POST_TREATMENT_PREFIXES = ("outcome_", "post_", "campaign_")
EDGE_COLUMNS = ["source", "target", "endpoint_source", "endpoint_target", "edge"]
ROBUST_SE_CHOICES = ("none", "HC0", "HC1", "HC2", "HC3")


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
    dropped_columns: pd.DataFrame


@dataclass(frozen=True)
class EdgeEffectResult:
    algorithm: str
    source: str
    target: str
    adjustment_set: tuple[str, ...]
    coefficient_standardized: float
    coefficient_original_scale: float
    standard_error_standardized: float
    standard_error_original_scale: float
    t_value: float
    ci_low_standardized: float
    ci_high_standardized: float
    ci_low_original_scale: float
    ci_high_original_scale: float
    p_value: float
    n_samples: int
    r_squared: float
    condition_number: float


@dataclass(frozen=True)
class LinearRegressionFit:
    coefficients: np.ndarray
    standard_errors: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    rank: int
    condition_number: float
    r_squared: float
    dof: int


@dataclass(frozen=True)
class TreatmentEffectResult:
    method: str
    estimand: str
    treatment: str
    outcome: str
    adjustment_set: list[str]
    n: int
    n_treated: int
    n_control: int
    effect: float
    std_error: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    notes: str = ""


def normal_p_value(z_value: float) -> float:
    if not np.isfinite(z_value):
        return np.nan
    return float(math.erfc(abs(z_value) / math.sqrt(2.0)))


def confidence_interval(
    estimate: float,
    standard_error: float | None,
    *,
    z_value: float = 1.96,
) -> tuple[float | None, float | None]:
    if standard_error is None or not np.isfinite(standard_error):
        return None, None
    return float(estimate - z_value * standard_error), float(estimate + z_value * standard_error)


def none_to_nan(value: float | None) -> float:
    return np.nan if value is None else float(value)


def numeric_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(frame), 0), dtype=float)
    return frame.loc[:, columns].to_numpy(dtype=float)


def fit_linear_regression(
    y: np.ndarray,
    x: np.ndarray,
    *,
    robust_se: str = "none",
) -> LinearRegressionFit:
    if robust_se not in ROBUST_SE_CHOICES:
        raise ValueError(f"robust_se must be one of {ROBUST_SE_CHOICES}: {robust_se}")

    x_design = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    fitted = x_design @ beta
    residuals = y - fitted
    rank = int(np.linalg.matrix_rank(x_design))
    dof = int(max(len(y) - x_design.shape[1], 1))
    xtx_inv = np.linalg.pinv(x_design.T @ x_design)

    if robust_se == "none":
        sigma2 = float((residuals @ residuals) / dof)
        covariance = sigma2 * xtx_inv
    else:
        leverage = np.sum((x_design @ xtx_inv) * x_design, axis=1)
        if robust_se == "HC0":
            scaled_residuals = residuals**2
        elif robust_se == "HC1":
            scaled_residuals = residuals**2 * len(y) / dof
        elif robust_se == "HC2":
            scaled_residuals = residuals**2 / np.clip(1.0 - leverage, 1e-12, None)
        else:
            scaled_residuals = residuals**2 / np.clip(1.0 - leverage, 1e-12, None) ** 2
        meat = x_design.T @ (scaled_residuals[:, None] * x_design)
        covariance = xtx_inv @ meat @ xtx_inv

    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    centered = y - y.mean()
    total_sum_squares = float(centered @ centered)
    residual_sum_squares = float(residuals @ residuals)
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else np.nan
    )
    condition_number = (
        float(np.linalg.cond(x_design))
        if x_design.shape[1] > 0 and x_design.shape[0] >= x_design.shape[1]
        else np.inf
    )

    return LinearRegressionFit(
        coefficients=beta,
        standard_errors=standard_errors,
        fitted=fitted,
        residuals=residuals,
        rank=rank,
        condition_number=condition_number,
        r_squared=float(r_squared),
        dof=dof,
    )


def logistic(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic_propensity(
    treatment: np.ndarray,
    covariates: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, str]:
    x_design = np.column_stack([np.ones(len(treatment)), covariates])
    beta = np.zeros(x_design.shape[1], dtype=float)
    converged = False

    for _ in range(max_iter):
        probability = logistic(x_design @ beta)
        weights = np.clip(probability * (1.0 - probability), 1e-8, None)
        hessian = x_design.T @ (weights[:, None] * x_design)
        penalty = np.eye(x_design.shape[1]) * ridge
        penalty[0, 0] = 0.0
        score = x_design.T @ (treatment - probability) - penalty @ beta
        try:
            delta = np.linalg.solve(hessian + penalty, score)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(hessian + penalty) @ score
        beta = beta + delta
        if float(np.max(np.abs(delta))) < tol:
            converged = True
            break

    notes = "" if converged else "propensity_logit_not_converged"
    return logistic(x_design @ beta), notes


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return np.nan
    return float(np.sum(weights * values) / weight_sum)


def weighted_variance(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return np.nan
    return float(np.sum(weights * (values - mean) ** 2) / weight_sum)


def effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.sum(weights**2))
    if denominator <= 0:
        return np.nan
    return float(np.sum(weights) ** 2 / denominator)


def records_to_frame(records: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records, columns=columns)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    printable = frame.copy()
    for column in printable.columns:
        printable[column] = printable[column].map(format_report_value)
    columns = list(printable.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in printable.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def format_report_value(value: object) -> str:
    if isinstance(value, tuple | list):
        return ", ".join(str(item) for item in value)
    if value is None:
        return ""
    if isinstance(value, float | np.floating):
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{float(value):.6g}"
    return str(value)


def write_config(output_dir: Path, config: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )


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
        inference_frame = self.build_inference_frame(model_frame)
        standardized, dropped_columns = self.standardize(inference_frame)
        dropped_columns = pd.concat(
            [
                dropped_columns,
                pd.DataFrame(
                    [
                        {
                            "column": "household_comp",
                            "reason": "non_numeric_not_encoded",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        return PreprocessingResult(
            model_frame=model_frame,
            inference_frame=inference_frame,
            standardized=standardized,
            dropped_columns=dropped_columns,
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
        home_ownership = model_frame["home_ownership"].astype("string")
        marital_status = model_frame["marital_status"].astype("string")
        homeowner_yes = home_ownership.eq("Homeowner").astype(float)
        married_yes = marital_status.eq("Married").astype(float)

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
                "homeowner": homeowner_yes,
                "homeowner_yes": homeowner_yes,
                "homeowner_unknown": home_ownership.eq("Unknown").astype(float),
                "married": married_yes,
                "married_yes": married_yes,
                "married_unknown": marital_status.eq("Unknown").astype(float),
                "household_size": self.numeric_category(model_frame["household_size"]),
                "kids_count": self.numeric_category(model_frame["kids_count"]),
                "pre_baskets": model_frame["pre_baskets"].astype(float),
                "pre_quantity": model_frame["pre_quantity"].astype(float),
                "pre_sales_value": model_frame["pre_sales_value"].astype(float),
                "pre_retail_disc": model_frame["pre_retail_disc"].astype(float),
                "pre_coupon_disc": model_frame["pre_coupon_disc"].astype(float),
                "pre_coupon_match_disc": model_frame["pre_coupon_match_disc"].astype(float),
                "treated": model_frame["treated"].astype(float),
                "outcome_baskets": model_frame["outcome_baskets"].astype(float),
                "outcome_quantity": model_frame["outcome_quantity"].astype(float),
                "outcome_sales_value": model_frame["outcome_sales_value"].astype(float),
                "outcome_retail_disc": model_frame["outcome_retail_disc"].astype(float),
                "outcome_coupon_disc": model_frame["outcome_coupon_disc"].astype(float),
                "outcome_coupon_match_disc": model_frame[
                    "outcome_coupon_match_disc"
                ].astype(float),
            }
        )

    def drop_collinear_columns(
        self,
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        corr = frame.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        records = []
        for column in upper.columns:
            correlated_with = upper.index[upper[column] >= self.collinearity_threshold].tolist()
            if correlated_with:
                records.append(
                    {
                        "column": column,
                        "reason": f"collinear_with:{correlated_with[0]}",
                    }
                )
        columns_to_drop = [record["column"] for record in records]
        return frame.drop(columns=columns_to_drop), pd.DataFrame(records)

    def standardize(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        records = []
        for column in frame.columns:
            if numeric[column].isna().all():
                records.append({"column": column, "reason": "all_missing"})
            elif not pd.api.types.is_numeric_dtype(frame[column]):
                records.append({"column": column, "reason": "non_numeric"})

        kept = numeric.drop(columns=[record["column"] for record in records])
        std = kept.std(axis=0)
        constant_columns = std[std <= 0].index.tolist()
        records.extend(
            {"column": column, "reason": "constant"} for column in constant_columns
        )
        kept = kept.drop(columns=constant_columns)
        kept, collinear = self.drop_collinear_columns(kept)
        if not collinear.empty:
            records.extend(collinear.to_dict(orient="records"))
        standardized = (kept - kept.mean(axis=0)) / kept.std(axis=0)
        return standardized, pd.DataFrame(records, columns=["column", "reason"])


class EdgeWeightEstimator:
    def __init__(
        self,
        *,
        standardized_frame: pd.DataFrame,
        original_frame: pd.DataFrame,
        discovery_dir: Path,
        output_dir: Path,
        algorithms: tuple[str, ...],
        dropped_columns: pd.DataFrame,
    ) -> None:
        self.standardized_frame = standardized_frame
        self.original_frame = original_frame
        self.discovery_dir = discovery_dir
        self.output_dir = output_dir
        self.algorithms = algorithms
        self.dropped_columns = dropped_columns

    @property
    def edge_output_dir(self) -> Path:
        return self.output_dir / "edge_weight"

    def load_edges(self, algorithm: str) -> pd.DataFrame | None:
        edge_path = self.discovery_dir / algorithm / "edges.csv"
        if not edge_path.exists():
            return None
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

    def estimate_all_edge_coefficients(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        records = []
        skipped_records = []
        found_edge_files = 0
        for algorithm in self.algorithms:
            loaded_edges = self.load_edges(algorithm)
            if loaded_edges is None:
                skipped_records.append(
                    {
                        "algorithm": algorithm,
                        "source": "",
                        "target": "",
                        "reason": "missing_edges_file",
                    }
                )
                continue
            found_edge_files += 1
            edges = self.directed_edges(loaded_edges)
            parents = self.parents_by_target(edges)
            for _, edge in edges.iterrows():
                source = str(edge["source"])
                target = str(edge["target"])
                missing_source = source not in self.standardized_frame.columns
                missing_target = target not in self.standardized_frame.columns
                if missing_source or missing_target:
                    if missing_source and missing_target:
                        reason = "missing_source_and_target"
                    elif missing_source:
                        reason = "missing_source"
                    else:
                        reason = "missing_target"
                    skipped_records.append(
                        {
                            "algorithm": algorithm,
                            "source": source,
                            "target": target,
                            "reason": reason,
                        }
                    )
                    continue

                adjustment_set = tuple(
                    sorted(
                        parent
                        for parent in parents.get(target, set()).difference({source})
                        if parent in self.standardized_frame.columns
                        and parent in self.original_frame.columns
                    )
                )
                result, skipped_reason = self.estimate_edge_coefficient(
                    algorithm=algorithm,
                    source=source,
                    target=target,
                    adjustment_set=adjustment_set,
                )
                if result is None:
                    skipped_records.append(
                        {
                            "algorithm": algorithm,
                            "source": source,
                            "target": target,
                            "reason": skipped_reason,
                        }
                    )
                else:
                    records.append(result.__dict__)

        if found_edge_files == 0:
            raise FileNotFoundError(
                f"No edges.csv files found under {self.discovery_dir} for algorithms: "
                f"{', '.join(self.algorithms)}"
            )

        columns = [
            "algorithm",
            "source",
            "target",
            "adjustment_set",
            "coefficient_standardized",
            "coefficient_original_scale",
            "standard_error_standardized",
            "standard_error_original_scale",
            "t_value",
            "ci_low_standardized",
            "ci_high_standardized",
            "ci_low_original_scale",
            "ci_high_original_scale",
            "p_value",
            "n_samples",
            "r_squared",
            "condition_number",
        ]
        effects = records_to_frame(records, columns)
        if not effects.empty:
            effects = effects.sort_values(["algorithm", "target", "source"]).reset_index(drop=True)
        skipped = records_to_frame(
            skipped_records,
            ["algorithm", "source", "target", "reason"],
        )
        return effects, skipped

    def estimate_all(self) -> pd.DataFrame:
        effects, skipped_edges = self.estimate_all_edge_coefficients()
        self._last_skipped_edges = skipped_edges
        return effects

    def estimate_edge_coefficient(
        self,
        *,
        algorithm: str,
        source: str,
        target: str,
        adjustment_set: tuple[str, ...],
    ) -> tuple[EdgeEffectResult | None, str]:
        regressors = [source, *adjustment_set]
        used_columns = [target, *regressors]
        standardized = self.standardized_frame.loc[:, used_columns].dropna()
        original = self.original_frame.loc[:, used_columns].dropna()
        common_index = standardized.index.intersection(original.index)
        standardized = standardized.loc[common_index]
        original = original.loc[common_index]
        parameter_count = len(regressors) + 1
        if len(common_index) <= parameter_count:
            return None, "insufficient_sample_size"

        y_standardized = standardized[target].to_numpy(dtype=float)
        x_standardized = numeric_matrix(standardized, regressors)
        y_original = original[target].to_numpy(dtype=float)
        x_original = numeric_matrix(original, regressors)
        standardized_fit = fit_linear_regression(y_standardized, x_standardized)
        original_fit = fit_linear_regression(y_original, x_original)
        if standardized_fit.rank < parameter_count or original_fit.rank < parameter_count:
            return None, "rank_deficient"

        source_index = 1
        coefficient_standardized = float(standardized_fit.coefficients[source_index])
        standard_error_standardized = float(standardized_fit.standard_errors[source_index])
        coefficient_original = float(original_fit.coefficients[source_index])
        standard_error_original = float(original_fit.standard_errors[source_index])
        t_value = (
            coefficient_original / standard_error_original
            if standard_error_original > 0
            else np.nan
        )
        p_value = normal_p_value(float(t_value))
        ci_low_standardized, ci_high_standardized = confidence_interval(
            coefficient_standardized,
            standard_error_standardized,
        )
        ci_low_original, ci_high_original = confidence_interval(
            coefficient_original,
            standard_error_original,
        )

        return (
            EdgeEffectResult(
                algorithm=algorithm,
                source=source,
                target=target,
                adjustment_set=adjustment_set,
                coefficient_standardized=coefficient_standardized,
                coefficient_original_scale=coefficient_original,
                standard_error_standardized=standard_error_standardized,
                standard_error_original_scale=standard_error_original,
                t_value=float(t_value),
                ci_low_standardized=none_to_nan(ci_low_standardized),
                ci_high_standardized=none_to_nan(ci_high_standardized),
                ci_low_original_scale=none_to_nan(ci_low_original),
                ci_high_original_scale=none_to_nan(ci_high_original),
                p_value=p_value,
                n_samples=len(common_index),
                r_squared=original_fit.r_squared,
                condition_number=original_fit.condition_number,
            ),
            "",
        )

    def write_edge_weight_outputs(
        self,
        effects: pd.DataFrame,
        skipped_edges: pd.DataFrame,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.edge_output_dir.mkdir(parents=True, exist_ok=True)
        effects.to_csv(self.edge_output_dir / "edge_effects.csv", index=False)
        skipped_edges.to_csv(self.edge_output_dir / "skipped_edges.csv", index=False)
        self.dropped_columns.to_csv(self.edge_output_dir / "dropped_columns.csv", index=False)

        # Backward-compatible copies for existing notebooks/articles.
        effects.to_csv(self.output_dir / "edge_effects.csv", index=False)
        report = [
            "# Edge Weight Estimation Report",
            "",
            "## Interpretation",
            "",
            "This report estimates conditional linear coefficients for directed edges "
            "discovered in the causal discovery step.",
            "",
            "For each edge:",
            "",
            "`target ~ source + other_parents_of_target`",
            "",
            "The coefficient should not automatically be interpreted as an ATE, ATT, "
            "or total causal effect.",
            "",
            "It may be interpreted as a direct structural coefficient only under strong "
            "assumptions:",
            "",
            "- the discovered graph is correct",
            "- no unobserved confounding",
            "- no bad-control adjustment",
            "- linear additive structural form",
            "- no relevant measurement error",
            "- graph-selection uncertainty is ignored",
            "",
            "## Data Summary",
            "",
            f"- discovery_dir: `{self.discovery_dir}`",
            f"- samples: `{len(self.standardized_frame)}`",
            f"- standardized_variables: `{len(self.standardized_frame.columns)}`",
            f"- original_scale_variables: `{len(self.original_frame.columns)}`",
            f"- estimated_edges: `{len(effects)}`",
            f"- skipped_edges: `{len(skipped_edges)}`",
            f"- dropped_columns: `{len(self.dropped_columns)}`",
            "",
            "## Results",
            "",
            dataframe_to_markdown(effects),
            "",
            "## Skipped Edges",
            "",
            dataframe_to_markdown(skipped_edges),
            "",
            "## Dropped Columns",
            "",
            dataframe_to_markdown(self.dropped_columns),
            "",
            "## Encoding Notes",
            "",
            "- age and income are encoded as ordinal/midpoint numeric variables.",
            "- Sensitivity to alternative encoding, such as one-hot encoding, is not yet evaluated.",
            "- `homeowner` and `married` are retained for backward compatibility with existing discovery outputs; explicit `_yes` and `_unknown` columns are also available in the unstandardized frame.",
        ]
        report_text = "\n".join(report)
        (self.edge_output_dir / "edge_effects.md").write_text(report_text, encoding="utf-8")
        (self.output_dir / "edge_effects.md").write_text(report_text, encoding="utf-8")

    def write_outputs(self, effects: pd.DataFrame) -> None:
        skipped_edges = getattr(
            self,
            "_last_skipped_edges",
            pd.DataFrame(columns=["algorithm", "source", "target", "reason"]),
        )
        self.write_edge_weight_outputs(effects, skipped_edges)

class CausalInference(EdgeWeightEstimator):
    def __init__(
        self,
        *,
        frame: pd.DataFrame | None = None,
        standardized_frame: pd.DataFrame | None = None,
        original_frame: pd.DataFrame | None = None,
        discovery_dir: Path,
        output_dir: Path,
        algorithms: tuple[str, ...],
        dropped_columns: pd.DataFrame | None = None,
    ) -> None:
        if standardized_frame is None:
            if frame is None:
                raise ValueError("Either frame or standardized_frame must be provided.")
            standardized_frame = frame
        if original_frame is None:
            original_frame = standardized_frame
        if dropped_columns is None:
            dropped_columns = pd.DataFrame(columns=["column", "reason"])
        super().__init__(
            standardized_frame=standardized_frame,
            original_frame=original_frame,
            discovery_dir=discovery_dir,
            output_dir=output_dir,
            algorithms=algorithms,
            dropped_columns=dropped_columns,
        )



def is_excluded_adjustment_column(column: str, treatment: str, outcome: str) -> bool:
    return (
        column in {treatment, outcome}
        or column.startswith(POST_TREATMENT_PREFIXES)
    )


def select_adjustment_set(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
    strategy: str,
    manual_covariates: list[str] | None = None,
    graph_edges: pd.DataFrame | None = None,
) -> list[str]:
    if strategy == "manual":
        if not manual_covariates:
            raise ValueError("--covariates must be specified when --adjustment-strategy manual")
        missing = [column for column in manual_covariates if column not in frame.columns]
        if missing:
            raise ValueError(f"manual covariates are missing from frame: {missing}")
        candidates = list(manual_covariates)
    elif strategy == "pre_treatment_covariates":
        candidates = [column for column in PRE_TREATMENT_COVARIATES if column in frame.columns]
    elif strategy == "graph_parents":
        if graph_edges is None or graph_edges.empty:
            warnings.warn(
                "graph_parents adjustment requested but no graph edges were available.",
                stacklevel=2,
            )
            candidates = []
        else:
            directed = graph_edges.loc[graph_edges["edge"].eq("-->")]
            candidates = [
                str(row["source"])
                for _, row in directed.iterrows()
                if str(row["target"]) == outcome and str(row["source"]) in frame.columns
            ]
    else:
        raise ValueError(f"unknown adjustment strategy: {strategy}")

    selected = []
    seen = set()
    for column in candidates:
        if is_excluded_adjustment_column(column, treatment, outcome):
            continue
        if column in seen:
            continue
        selected.append(column)
        seen.add(column)
    if strategy in {"pre_treatment_covariates", "graph_parents"}:
        selected = prune_auto_adjustment_candidates(frame, selected)
    return selected


def prune_auto_adjustment_candidates(
    frame: pd.DataFrame,
    candidates: list[str],
    *,
    collinearity_threshold: float = 0.995,
) -> list[str]:
    selected: list[str] = []
    for column in candidates:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.dropna().nunique() <= 1:
            continue
        if not selected:
            selected.append(column)
            continue
        correlations = frame.loc[:, [*selected, column]].corr(numeric_only=True).abs()
        if (correlations.loc[selected, column] >= collinearity_threshold).any():
            continue
        selected.append(column)
    return selected


def validate_treatment_effect_inputs(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
) -> None:
    if treatment not in frame.columns:
        raise ValueError(f"treatment column is missing from frame: {treatment}")
    if outcome not in frame.columns:
        raise ValueError(f"outcome column is missing from frame: {outcome}")
    values = set(pd.to_numeric(frame[treatment].dropna(), errors="coerce").unique())
    if values.difference({0.0, 1.0}):
        raise ValueError(
            "Treatment must be binary for current estimators. "
            f"Observed values: {sorted(values)}"
        )


def load_graph_edges(discovery_dir: Path, algorithms: tuple[str, ...]) -> pd.DataFrame:
    records = []
    for algorithm in algorithms:
        edge_path = discovery_dir / algorithm / "edges.csv"
        if not edge_path.exists():
            warnings.warn(f"missing graph edge file: {edge_path}", stacklevel=2)
            continue
        edges = pd.read_csv(edge_path)
        missing = {"source", "target", "edge"}.difference(edges.columns)
        if missing:
            raise ValueError(f"{edge_path} is missing required columns: {sorted(missing)}")
        copied = edges.copy()
        copied["algorithm"] = algorithm
        records.append(copied)
    if not records:
        return pd.DataFrame(columns=[*EDGE_COLUMNS, "algorithm"])
    return pd.concat(records, ignore_index=True)


class DesignDiagnostics:
    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: list[str],
    ) -> None:
        self.frame = frame
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = covariates

    def treatment_counts(self) -> pd.DataFrame:
        treatment = self.frame[self.treatment].astype(float)
        n = int(treatment.notna().sum())
        n_treated = int((treatment == 1.0).sum())
        n_control = int((treatment == 0.0).sum())
        treated_rate = n_treated / n if n > 0 else np.nan
        return pd.DataFrame(
            [
                {
                    "n": n,
                    "n_treated": n_treated,
                    "n_control": n_control,
                    "treated_rate": treated_rate,
                }
            ]
        )

    def balance_table(self) -> pd.DataFrame:
        treatment = self.frame[self.treatment].astype(float)
        records = []
        for column in self.covariates:
            values = pd.to_numeric(self.frame[column], errors="coerce")
            treated_values = values.loc[treatment == 1.0].dropna()
            control_values = values.loc[treatment == 0.0].dropna()
            mean_treated = float(treated_values.mean()) if len(treated_values) else np.nan
            mean_control = float(control_values.mean()) if len(control_values) else np.nan
            std_treated = float(treated_values.std(ddof=1)) if len(treated_values) > 1 else np.nan
            std_control = float(control_values.std(ddof=1)) if len(control_values) > 1 else np.nan
            pooled_std = math.sqrt((std_treated**2 + std_control**2) / 2.0) if (
                np.isfinite(std_treated) and np.isfinite(std_control)
            ) else np.nan
            smd = (
                (mean_treated - mean_control) / pooled_std
                if np.isfinite(pooled_std) and pooled_std > 0
                else np.nan
            )
            records.append(
                {
                    "covariate": column,
                    "mean_treated": mean_treated,
                    "mean_control": mean_control,
                    "std_treated": std_treated,
                    "std_control": std_control,
                    "standardized_mean_difference": float(smd)
                    if np.isfinite(smd)
                    else np.nan,
                    "missing_rate": float(values.isna().mean()),
                }
            )
        return records_to_frame(
            records,
            [
                "covariate",
                "mean_treated",
                "mean_control",
                "std_treated",
                "std_control",
                "standardized_mean_difference",
                "missing_rate",
            ],
        )

    def outcome_distribution(self) -> pd.DataFrame:
        values = pd.to_numeric(self.frame[self.outcome], errors="coerce").dropna()
        if values.empty:
            return pd.DataFrame(
                [
                    {
                        "mean": np.nan,
                        "std": np.nan,
                        "min": np.nan,
                        "p25": np.nan,
                        "median": np.nan,
                        "p75": np.nan,
                        "max": np.nan,
                        "zero_rate": np.nan,
                        "skewness": np.nan,
                    }
                ]
            )
        return pd.DataFrame(
            [
                {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "p25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "p75": float(values.quantile(0.75)),
                    "max": float(values.max()),
                    "zero_rate": float((values == 0.0).mean()),
                    "skewness": float(values.skew()),
                }
            ]
        )

    def propensity_overlap(self, propensity_score: np.ndarray) -> pd.DataFrame:
        if len(propensity_score) == 0:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "ps_min": float(np.min(propensity_score)),
                    "ps_p01": float(np.quantile(propensity_score, 0.01)),
                    "ps_p05": float(np.quantile(propensity_score, 0.05)),
                    "ps_median": float(np.median(propensity_score)),
                    "ps_p95": float(np.quantile(propensity_score, 0.95)),
                    "ps_p99": float(np.quantile(propensity_score, 0.99)),
                    "ps_max": float(np.max(propensity_score)),
                    "n_ps_below_0_01": int((propensity_score < 0.01).sum()),
                    "n_ps_above_0_99": int((propensity_score > 0.99).sum()),
                }
            ]
        )


class TreatmentEffectEstimator:
    def __init__(
        self,
        frame: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: list[str],
        estimand: str = "ATE",
        robust_se: str = "HC3",
    ) -> None:
        if estimand not in {"ATE", "ATT"}:
            raise ValueError(f"estimand must be ATE or ATT: {estimand}")
        if robust_se not in ROBUST_SE_CHOICES:
            raise ValueError(f"robust_se must be one of {ROBUST_SE_CHOICES}: {robust_se}")
        validate_treatment_effect_inputs(frame, treatment, outcome)
        self.frame = frame
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = covariates
        self.estimand = estimand
        self.robust_se = robust_se
        self.last_propensity_score: np.ndarray | None = None
        self.last_propensity_notes = ""

    def complete_case_data(self, include_covariates: bool) -> pd.DataFrame:
        columns = [self.treatment, self.outcome]
        if include_covariates:
            columns.extend(self.covariates)
        return self.frame.loc[:, columns].dropna()

    def result(
        self,
        *,
        method: str,
        data: pd.DataFrame,
        effect: float,
        std_error: float | None,
        notes: str,
    ) -> TreatmentEffectResult:
        t = data[self.treatment].astype(float)
        ci_low, ci_high = confidence_interval(effect, std_error)
        p_value = (
            normal_p_value(effect / std_error)
            if std_error is not None and std_error > 0
            else None
        )
        return TreatmentEffectResult(
            method=method,
            estimand=self.estimand,
            treatment=self.treatment,
            outcome=self.outcome,
            adjustment_set=list(self.covariates),
            n=int(len(data)),
            n_treated=int((t == 1.0).sum()),
            n_control=int((t == 0.0).sum()),
            effect=float(effect),
            std_error=float(std_error) if std_error is not None else None,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            notes=notes,
        )

    def estimate(self, methods: list[str]) -> pd.DataFrame:
        records = []
        runners = {
            "diff_in_means": self.diff_in_means,
            "ols": self.ols,
            "ipw": self.ipw,
            "aipw": self.aipw,
        }
        for method in methods:
            records.append(runners[method]().__dict__)
        columns = list(TreatmentEffectResult.__dataclass_fields__.keys())
        return records_to_frame(records, columns)

    def diff_in_means(self) -> TreatmentEffectResult:
        data = self.complete_case_data(include_covariates=False)
        y = data[self.outcome].to_numpy(dtype=float)
        t = data[self.treatment].to_numpy(dtype=float)
        y_treated = y[t == 1.0]
        y_control = y[t == 0.0]
        if len(y_treated) == 0 or len(y_control) == 0:
            raise ValueError("diff_in_means requires both treated and control observations")
        effect = float(y_treated.mean() - y_control.mean())
        variance_treated = float(np.var(y_treated, ddof=1)) if len(y_treated) > 1 else 0.0
        variance_control = float(np.var(y_control, ddof=1)) if len(y_control) > 1 else 0.0
        standard_error = math.sqrt(
            variance_treated / len(y_treated) + variance_control / len(y_control)
        )
        return self.result(
            method="diff_in_means",
            data=data,
            effect=effect,
            std_error=standard_error,
            notes="unadjusted_difference; numerical contrast is identical for ATE and ATT",
        )

    def ols(self) -> TreatmentEffectResult:
        data = self.complete_case_data(include_covariates=True)
        y = data[self.outcome].to_numpy(dtype=float)
        regressors = [self.treatment, *self.covariates]
        x = numeric_matrix(data, regressors)
        fit = fit_linear_regression(y, x, robust_se=self.robust_se)
        effect = float(fit.coefficients[1])
        standard_error = float(fit.standard_errors[1])
        notes = [
            f"robust_se={self.robust_se}",
            f"rank={fit.rank}/{x.shape[1] + 1}",
            f"condition_number={fit.condition_number:.6g}",
            "normal_approximation_for_ci_and_p_value",
        ]
        if fit.rank < x.shape[1] + 1:
            notes.append("rank_deficient_design_pinv_used")
        return self.result(
            method="ols",
            data=data,
            effect=effect,
            std_error=standard_error,
            notes="; ".join(notes),
        )

    def propensity_data(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        data = self.complete_case_data(include_covariates=True)
        y = data[self.outcome].to_numpy(dtype=float)
        t = data[self.treatment].to_numpy(dtype=float)
        x = numeric_matrix(data, self.covariates)
        propensity_raw, notes = fit_logistic_propensity(t, x)
        propensity_score = np.clip(propensity_raw, 0.01, 0.99)
        self.last_propensity_score = propensity_raw
        self.last_propensity_notes = notes
        return data, y, t, propensity_score

    def ipw(self) -> TreatmentEffectResult:
        data, y, t, propensity_score = self.propensity_data()
        if self.estimand == "ATE":
            weights_treated = t / propensity_score
            weights_control = (1.0 - t) / (1.0 - propensity_score)
        else:
            weights_treated = t
            weights_control = (1.0 - t) * propensity_score / (1.0 - propensity_score)

        mean_treated = weighted_mean(y, weights_treated)
        mean_control = weighted_mean(y, weights_control)
        effect = float(mean_treated - mean_control)
        ess_treated = effective_sample_size(weights_treated)
        ess_control = effective_sample_size(weights_control)
        variance_treated = weighted_variance(y, weights_treated, mean_treated)
        variance_control = weighted_variance(y, weights_control, mean_control)
        standard_error = math.sqrt(
            variance_treated / ess_treated + variance_control / ess_control
        )
        raw = self.last_propensity_score
        overlap_warning = ""
        if raw is not None and ((raw < 0.01).any() or (raw > 0.99).any()):
            overlap_warning = "; overlap_warning=propensity_outside_[0.01,0.99]"
        notes = (
            f"propensity_logit; clip=[0.01,0.99]; "
            f"ess_treated={ess_treated:.6g}; ess_control={ess_control:.6g}"
            f"{overlap_warning}"
        )
        if self.last_propensity_notes:
            notes = f"{notes}; {self.last_propensity_notes}"
        return self.result(
            method="ipw",
            data=data,
            effect=effect,
            std_error=standard_error,
            notes=notes,
        )

    def aipw(self) -> TreatmentEffectResult:
        data, y, t, propensity_score = self.propensity_data()
        x = numeric_matrix(data, self.covariates)
        treated_mask = t == 1.0
        control_mask = t == 0.0
        if treated_mask.sum() <= x.shape[1] + 1 or control_mask.sum() <= x.shape[1] + 1:
            return self.result(
                method="aipw",
                data=data,
                effect=np.nan,
                std_error=None,
                notes="skipped_insufficient_sample_size_for_outcome_models",
            )

        treated_fit = fit_linear_regression(y[treated_mask], x[treated_mask])
        control_fit = fit_linear_regression(y[control_mask], x[control_mask])
        design = np.column_stack([np.ones(len(data)), x])
        mu_treated = design @ treated_fit.coefficients
        mu_control = design @ control_fit.coefficients

        if self.estimand == "ATE":
            score = (
                mu_treated
                - mu_control
                + t / propensity_score * (y - mu_treated)
                - (1.0 - t) / (1.0 - propensity_score) * (y - mu_control)
            )
        else:
            treated_rate = float(t.mean())
            score = (
                t / treated_rate * (y - mu_control)
                - (1.0 - t)
                * propensity_score
                / (1.0 - propensity_score)
                / treated_rate
                * (y - mu_control)
            )
        effect = float(score.mean())
        standard_error = float(np.std(score, ddof=1) / math.sqrt(len(score)))
        notes = (
            "aipw_linear_outcome_models; propensity_logit; clip=[0.01,0.99]; "
            "no_sample_splitting"
        )
        if self.last_propensity_notes:
            notes = f"{notes}; {self.last_propensity_notes}"
        return self.result(
            method="aipw",
            data=data,
            effect=effect,
            std_error=standard_error,
            notes=notes,
        )


def write_treatment_effect_outputs(
    *,
    output_dir: Path,
    effects: pd.DataFrame,
    design_diagnostics: pd.DataFrame,
    balance_table: pd.DataFrame,
    propensity_overlap: pd.DataFrame,
    outcome_distribution: pd.DataFrame,
    treatment: str,
    outcome: str,
    estimand: str,
    adjustment_strategy: str,
    adjustment_set: list[str],
) -> None:
    treatment_dir = output_dir / "treatment_effect"
    treatment_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(treatment_dir / "treatment_effects.csv", index=False)
    design_diagnostics.to_csv(treatment_dir / "design_diagnostics.csv", index=False)
    balance_table.to_csv(treatment_dir / "balance_table.csv", index=False)
    propensity_overlap.to_csv(treatment_dir / "propensity_overlap.csv", index=False)
    outcome_distribution.to_csv(treatment_dir / "outcome_distribution.csv", index=False)

    graph_warning = (
        "- `graph_parents` was used. This is experimental and may adjust for "
        "mediators or bad controls depending on graph errors and temporal ordering."
        if adjustment_strategy == "graph_parents"
        else ""
    )
    report = [
        "# Treatment Effect Estimation Report",
        "",
        "## Treatment Definition",
        "",
        f"- treatment: `{treatment}`",
        "",
        "## Outcome Definition",
        "",
        f"- outcome: `{outcome}`",
        "",
        "## Estimand",
        "",
        f"- estimand: `{estimand}`",
        "",
        "Under the specified adjustment assumptions, the estimated effect is reported below.",
        "",
        "## Adjustment Set",
        "",
        f"- strategy: `{adjustment_strategy}`",
        f"- covariates: `{', '.join(adjustment_set) if adjustment_set else '(none)'}`",
        graph_warning,
        "",
        "## Design Diagnostics",
        "",
        dataframe_to_markdown(design_diagnostics),
        "",
        "## Covariate Balance",
        "",
        dataframe_to_markdown(balance_table),
        "",
        "## Propensity Overlap",
        "",
        dataframe_to_markdown(propensity_overlap),
        "",
        "## Outcome Distribution",
        "",
        dataframe_to_markdown(outcome_distribution),
        "",
        "## Effect Estimates",
        "",
        dataframe_to_markdown(effects),
        "",
        "## Robustness Notes",
        "",
        "- Difference in means is unadjusted and is a baseline association.",
        "- OLS uses the requested heteroskedasticity-consistent SE when specified.",
        "- IPW and AIPW use a logistic propensity model fitted on the selected covariates and clip propensity scores to [0.01, 0.99].",
        "- Confidence intervals and p-values use a standard normal approximation.",
        "- AIPW is implemented without sample splitting; nuisance-model overfit uncertainty is not separately handled.",
        "",
        "## Limitations",
        "",
        "- unobserved confounding is not removed by this script",
        "- causal discovery results may be wrong",
        "- graph selection uncertainty is not reflected in standard errors",
        "- adjusting for post-treatment variables changes the estimand away from a total effect",
        "- poor overlap can make IPW and AIPW unstable",
        "- age and income are encoded as ordinal/midpoint numeric variables; one-hot encoding sensitivity is not yet evaluated",
    ]
    (treatment_dir / "treatment_effects.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class RunContext:
    args: argparse.Namespace
    project_root: Path
    dataset_yaml: Path
    discovery_dir: Path
    output_dir: Path
    preprocessing_result: PreprocessingResult

    @property
    def inference_frame(self) -> pd.DataFrame:
        return self.preprocessing_result.inference_frame

    @property
    def standardized_frame(self) -> pd.DataFrame:
        return self.preprocessing_result.standardized


def build_run_config(
    context: RunContext,
    *,
    selected_adjustment_set: list[str] | None = None,
) -> dict[str, object]:
    args = context.args
    config: dict[str, object] = {
        "mode": args.mode,
        "campaign_id": str(args.campaign_id),
        "pre_weeks": args.pre_weeks,
        "collinearity_threshold": args.collinearity_threshold,
        "discovery_dir": str(context.discovery_dir),
        "output_dir": str(context.output_dir),
        "algorithms": list(args.algorithms),
        "treatment": args.treatment,
        "outcome": args.outcome,
        "estimand": args.estimand,
        "adjustment_strategy": args.adjustment_strategy,
        "covariates": args.covariates or [],
        "effect_methods": list(args.effect_methods),
        "robust_se": args.robust_se,
    }
    if selected_adjustment_set is not None:
        config["selected_adjustment_set"] = selected_adjustment_set
    return config


class AnalysisModeStrategy(ABC):
    mode: str

    @abstractmethod
    def run(self, context: RunContext) -> None:
        raise NotImplementedError


class EdgeWeightModeStrategy(AnalysisModeStrategy):
    mode = "edge_weight"

    def run(self, context: RunContext) -> None:
        args = context.args
        write_config(context.output_dir, build_run_config(context))
        edge_weight = EdgeWeightEstimator(
            standardized_frame=context.standardized_frame,
            original_frame=context.inference_frame,
            discovery_dir=context.discovery_dir,
            output_dir=context.output_dir,
            algorithms=tuple(args.algorithms),
            dropped_columns=context.preprocessing_result.dropped_columns,
        )
        effects, skipped_edges = edge_weight.estimate_all_edge_coefficients()
        edge_weight.write_edge_weight_outputs(effects, skipped_edges)

        print(f"mode: {args.mode}")
        print(f"samples: {len(context.standardized_frame):,}")
        print(f"standardized_variables: {len(context.standardized_frame.columns):,}")
        print(f"discovery_dir: {context.discovery_dir}")
        print(f"output_dir: {context.output_dir}")
        print(effects.to_string(index=False))


class TreatmentEffectModeStrategy(AnalysisModeStrategy):
    mode = "treatment_effect"

    def run(self, context: RunContext) -> None:
        args = context.args
        inference_frame = context.inference_frame
        validate_treatment_effect_inputs(inference_frame, args.treatment, args.outcome)
        graph_edges = self.load_graph_edges_for_adjustment(context)
        adjustment_set = select_adjustment_set(
            inference_frame,
            treatment=args.treatment,
            outcome=args.outcome,
            strategy=args.adjustment_strategy,
            manual_covariates=args.covariates,
            graph_edges=graph_edges,
        )
        write_config(
            context.output_dir,
            build_run_config(context, selected_adjustment_set=adjustment_set),
        )
        estimator = TreatmentEffectEstimator(
            inference_frame,
            treatment=args.treatment,
            outcome=args.outcome,
            covariates=adjustment_set,
            estimand=args.estimand,
            robust_se=args.robust_se,
        )
        effects = estimator.estimate(list(args.effect_methods))
        diagnostics = DesignDiagnostics(
            frame=inference_frame,
            treatment=args.treatment,
            outcome=args.outcome,
            covariates=adjustment_set,
        )
        design_diagnostics = diagnostics.treatment_counts()
        balance_table = diagnostics.balance_table()
        outcome_distribution = diagnostics.outcome_distribution()
        propensity_overlap = self.build_propensity_overlap(diagnostics, estimator)
        write_treatment_effect_outputs(
            output_dir=context.output_dir,
            effects=effects,
            design_diagnostics=design_diagnostics,
            balance_table=balance_table,
            propensity_overlap=propensity_overlap,
            outcome_distribution=outcome_distribution,
            treatment=args.treatment,
            outcome=args.outcome,
            estimand=args.estimand,
            adjustment_strategy=args.adjustment_strategy,
            adjustment_set=adjustment_set,
        )

        print(f"mode: {args.mode}")
        print(f"samples: {len(inference_frame):,}")
        print(f"variables: {len(inference_frame.columns):,}")
        print(f"adjustment_set: {', '.join(adjustment_set) if adjustment_set else '(none)'}")
        print(f"output_dir: {context.output_dir}")
        print(effects.to_string(index=False))

    def load_graph_edges_for_adjustment(self, context: RunContext) -> pd.DataFrame | None:
        if context.args.adjustment_strategy != "graph_parents":
            return None
        return load_graph_edges(context.discovery_dir, tuple(context.args.algorithms))

    def build_propensity_overlap(
        self,
        diagnostics: DesignDiagnostics,
        estimator: TreatmentEffectEstimator,
    ) -> pd.DataFrame:
        if estimator.last_propensity_score is None:
            return pd.DataFrame(
                columns=[
                    "ps_min",
                    "ps_p01",
                    "ps_p05",
                    "ps_median",
                    "ps_p95",
                    "ps_p99",
                    "ps_max",
                    "n_ps_below_0_01",
                    "n_ps_above_0_99",
                ]
            )
        return diagnostics.propensity_overlap(estimator.last_propensity_score)


MODE_STRATEGIES: tuple[type[AnalysisModeStrategy], ...] = (
    EdgeWeightModeStrategy,
    TreatmentEffectModeStrategy,
)
MODE_STRATEGY_BY_NAME = {
    strategy.mode: strategy
    for strategy in MODE_STRATEGIES
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate causal-discovery edge weights or explicit treatment effects "
            "for completejourney."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_STRATEGY_BY_NAME),
        default="edge_weight",
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
    parser.add_argument("--treatment", default="treated")
    parser.add_argument("--outcome", default="outcome_sales_value")
    parser.add_argument(
        "--estimand",
        choices=["ATE", "ATT"],
        default="ATE",
    )
    parser.add_argument(
        "--adjustment-strategy",
        choices=["pre_treatment_covariates", "manual", "graph_parents"],
        default="pre_treatment_covariates",
    )
    parser.add_argument("--covariates", nargs="*", default=None)
    parser.add_argument(
        "--effect-methods",
        nargs="+",
        choices=["diff_in_means", "ols", "ipw", "aipw"],
        default=["diff_in_means", "ols", "ipw"],
    )
    parser.add_argument(
        "--robust-se",
        choices=list(ROBUST_SE_CHOICES),
        default="HC3",
    )
    return parser.parse_args()


# Notebook stub for values normally supplied by parse_args().
# args = argparse.Namespace(
#     project_root=find_project_root(Path.cwd()),
#     dataset_yaml=None,
#     campaign_id=DEFAULT_CAMPAIGN_ID,
#     pre_weeks=8,
#     collinearity_threshold=0.995,
#     discovery_dir=None,
#     output_dir=None,
#     algorithms=("pc", "ges", "lingam", "notears"),
#     mode="edge_weight",
#     treatment="treated",
#     outcome="outcome_sales_value",
#     estimand="ATE",
#     adjustment_strategy="pre_treatment_covariates",
#     covariates=None,
#     effect_methods=("diff_in_means", "ols", "ipw"),
#     robust_se="HC3",
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
    context = RunContext(
        args=args,
        project_root=project_root,
        dataset_yaml=dataset_yaml,
        discovery_dir=discovery_dir,
        output_dir=output_dir,
        preprocessing_result=preprocessing_result,
    )
    strategy = MODE_STRATEGY_BY_NAME[args.mode]()
    strategy.run(context)




if __name__ == "__main__":
    main()
