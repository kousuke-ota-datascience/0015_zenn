from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from myproj.io.config_resolver import find_project_root, load_dataset_definition
from myproj.io.file_io import FileConfigRegistry, FileIOUtils
from myproj.logger.custom_logger import CustomLogger


DATASET_YAML = Path("shared/py/myproj/conf/dataset/completejourney/10_interim.yaml")
DEFAULT_CAMPAIGN_ID = "18"
LOGGER_NAME = "econml_completejourney"


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
) -> tuple[pd.DataFrame, CampaignWindow]:
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
    return frame.reset_index(), window


def encode_features(
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    outcome = frame["outcome_sales_value"].astype(float)
    treatment = frame["treated"].astype(int)
    control_frame = frame.drop(
        columns=[
            "household_id",
            "treated",
            "outcome_sales_value",
            "outcome_baskets",
            "outcome_quantity",
            "outcome_retail_disc",
            "outcome_coupon_disc",
            "outcome_coupon_match_disc",
        ]
    )
    effect_frame = frame[
        [
            "pre_baskets",
            "pre_sales_value",
            "age",
        ]
    ]
    effect_features = pd.get_dummies(effect_frame, drop_first=True, dtype=float)
    controls = pd.get_dummies(control_frame, drop_first=True, dtype=float)
    return outcome, treatment, effect_features, controls


def fit_linear_dml(
    outcome: pd.Series,
    treatment: pd.Series,
    effect_features: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    random_state: int,
) -> Any:
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    estimator = LinearDML(
        model_y=RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=random_state,
            n_jobs=-1,
        ),
        model_t=RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=random_state,
            n_jobs=-1,
        ),
        discrete_treatment=True,
        cv=3,
        random_state=random_state,
    )
    estimator.fit(
        outcome.to_numpy(),
        treatment.to_numpy(),
        X=effect_features,
        W=controls,
        inference="statsmodels",
    )
    return estimator


def summarize_effects(
    estimator: Any,
    effect_features: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    effects = np.asarray(estimator.effect(effect_features)).reshape(-1)
    summary = frame[["household_id", "treated", "age", "income"]].copy()
    summary["cate"] = effects
    return summary


def print_results(
    *,
    estimator: Any,
    effect_features: pd.DataFrame,
    frame: pd.DataFrame,
    window: CampaignWindow,
) -> None:
    ate = float(np.asarray(estimator.ate(X=effect_features)).reshape(-1)[0])
    ci_low, ci_high = estimator.ate_interval(X=effect_features)
    ci_low = float(np.asarray(ci_low).reshape(-1)[0])
    ci_high = float(np.asarray(ci_high).reshape(-1)[0])

    effect_summary = summarize_effects(estimator, effect_features, frame)
    treated_count = int(frame["treated"].sum())
    control_count = int((1 - frame["treated"]).sum())

    print(f"campaign_id: {window.campaign_id}")
    print(f"pre_weeks: {window.pre_start_week}-{window.pre_end_week}")
    print(f"outcome_weeks: {window.start_week}-{window.end_week}")
    print(f"households: {len(frame):,} (treated={treated_count:,}, control={control_count:,})")
    print(f"ATE on campaign-period sales_value: {ate:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print()
    print("CATE by age")
    print(
        effect_summary.groupby("age", dropna=False)["cate"]
        .agg(["count", "mean", "std"])
        .sort_index()
        .to_string()
    )
    print()
    print("Top positive CATE households")
    print(effect_summary.nlargest(10, "cate").to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate campaign treatment effects on completejourney data with EconML.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to the nearest parent containing pyproject.toml.",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=None,
        help=f"Dataset YAML. Defaults to {DATASET_YAML}.",
    )
    parser.add_argument(
        "--campaign-id",
        default=DEFAULT_CAMPAIGN_ID,
        help="Campaign whose household assignment is used as the binary treatment.",
    )
    parser.add_argument(
        "--pre-weeks",
        type=int,
        default=8,
        help="Number of weeks before campaign start used for pre-treatment features.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for nuisance models and cross-fitting.",
    )
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

    tables = load_completejourney_tables(
        project_root=project_root,
        dataset_yaml=dataset_yaml,
    )
    frame, window = build_model_frame(
        tables,
        campaign_id=str(args.campaign_id),
        pre_weeks=args.pre_weeks,
    )
    outcome, treatment, effect_features, controls = encode_features(frame)
    estimator = fit_linear_dml(
        outcome,
        treatment,
        effect_features,
        controls,
        random_state=args.random_state,
    )
    print_results(
        estimator=estimator,
        effect_features=effect_features,
        frame=frame,
        window=window,
    )


if __name__ == "__main__":
    main()
