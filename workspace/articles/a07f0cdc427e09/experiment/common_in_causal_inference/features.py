"""特徴量作成で共有する期間計算と標準化補助関数。

このモジュールは、探索用特徴量と推論用特徴量の意味を統合しない。共有するのは
キャンペーン開始週から pre/outcome window を決める処理、window 名から週範囲を
返す処理、相関がほぼ完全に重複する列を落とす処理のような、データ形状に対して
中立な処理だけである。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CampaignWindow:
    """1 キャンペーンに対する分析週範囲。

    Attributes:
        campaign_id: 分析対象キャンペーン ID。
        start_week: outcome window の開始週。
        end_week: outcome window の終了週。
        pre_start_week: pre-treatment window の開始週。
        pre_end_week: pre-treatment window の終了週。
    """

    campaign_id: str
    start_week: int
    end_week: int
    pre_start_week: int
    pre_end_week: int


def select_campaign_window(
    *,
    campaign_descriptions: pd.DataFrame,
    transactions: pd.DataFrame,
    campaign_id: str,
    campaign_id_column: str,
    start_day_column: str,
    end_day_column: str,
    week_column: str,
    transaction_timestamp_column: str,
    pre_weeks: int,
    day_to_week_divisor: int = 7,
) -> CampaignWindow:
    """キャンペーン日付と購買履歴から pre/outcome の週範囲を求める。

    Args:
        campaign_descriptions: キャンペーン開始日・終了日を含むテーブル。
        transactions: 週番号と transaction timestamp を含む購買テーブル。
        campaign_id: 分析対象キャンペーン ID。
        campaign_id_column: campaign ID 列名。
        start_day_column: キャンペーン開始日列名。Complete Journey の既定では
            Unix epoch 起点の日数。
        end_day_column: キャンペーン終了日列名。
        week_column: transaction 側の週番号列名。
        transaction_timestamp_column: transaction timestamp 列名。
        pre_weeks: キャンペーン開始前に何週分を pre window に含めるか。
        day_to_week_divisor: 日数を週番号に変換する除数。

    Returns:
        transaction データの実在範囲に丸め込んだ campaign window。

    Raises:
        ValueError: 対象キャンペーンが存在しない場合、または pre/outcome window が
            transaction データ上で空になる場合。
    """
    first_transaction_date = pd.to_datetime(
        transactions[transaction_timestamp_column],
        unit="s",
    ).min()
    first_transaction_date = first_transaction_date.normalize()

    campaigns = campaign_descriptions.copy()
    campaigns["start_dt"] = pd.to_datetime(
        campaigns[start_day_column],
        unit="D",
        origin="unix",
    )
    campaigns["end_dt"] = pd.to_datetime(
        campaigns[end_day_column],
        unit="D",
        origin="unix",
    )
    campaigns["start_week"] = (
        (campaigns["start_dt"] - first_transaction_date).dt.days // day_to_week_divisor
        + 1
    ).astype(int)
    campaigns["end_week"] = (
        (campaigns["end_dt"] - first_transaction_date).dt.days // day_to_week_divisor
        + 1
    ).astype(int)

    matched = campaigns.loc[campaigns[campaign_id_column].astype(str).eq(str(campaign_id))]
    if matched.empty:
        raise ValueError(f"unknown campaign_id: {campaign_id}")

    campaign = matched.iloc[0]
    min_week = int(transactions[week_column].min())
    max_week = int(transactions[week_column].max())
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
        campaign_id=str(campaign_id),
        start_week=start_week,
        end_week=end_week,
        pre_start_week=pre_start_week,
        pre_end_week=pre_end_week,
    )


def window_bounds(window: CampaignWindow, name: str) -> tuple[int, int]:
    """window 名から週番号の閉区間を返す。

    Args:
        window: ``select_campaign_window`` で求めた campaign window。
        name: ``"pre"`` または ``"outcome"``。

    Returns:
        ``(start_week, end_week)``。

    Raises:
        ValueError: 未対応の window 名が渡された場合。
    """
    if name == "pre":
        return window.pre_start_week, window.pre_end_week
    if name == "outcome":
        return window.start_week, window.end_week
    raise ValueError(f"unsupported transaction window: {name}")


def drop_collinear_columns(
    frame: pd.DataFrame,
    *,
    collinearity_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """相関が閾値以上の冗長列を落とす。

    Args:
        frame: 数値特徴量フレーム。
        collinearity_threshold: 絶対相関の閾値。``1.0`` に近い値は完全重複に
            近い列だけを落とす。

    Returns:
        列削除後のフレームと、削除列・理由を持つデータフレーム。

    Notes:
        上三角行列を走査し、先に出現した列を残して後続列を落とす。これは
        既存実装の挙動を維持するための決定規則であり、因果的な優先順位を
        表しているわけではない。
    """
    if frame.empty:
        return frame, pd.DataFrame(columns=["column", "reason"])

    corr = frame.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    records = []
    for column in upper.columns:
        correlated_with = upper.index[upper[column] >= collinearity_threshold].tolist()
        if correlated_with:
            records.append(
                {
                    "column": column,
                    "reason": f"collinear_with:{correlated_with[0]}",
                }
            )

    columns_to_drop = [record["column"] for record in records]
    return frame.drop(columns=columns_to_drop), pd.DataFrame(records)
