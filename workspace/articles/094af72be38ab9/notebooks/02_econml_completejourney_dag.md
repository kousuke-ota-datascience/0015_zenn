# Complete Journey EconML Model DAG

```mermaid
flowchart LR
    U["U: 未観測の購買嗜好・店舗接触・季節要因"]

    D["世帯属性<br/>age / income / home_ownership / marital_status<br/>household_size / household_comp / kids_count"]
    P["事前購買行動<br/>pre_baskets / pre_quantity / pre_sales_value<br/>pre_retail_disc / pre_coupon_disc / pre_coupon_match_disc"]

    T["T: キャンペーン対象<br/>campaign_id = 18 の household assignment"]
    Y["Y: キャンペーン期間中の売上<br/>outcome_sales_value"]

    X["X: 効果異質性特徴量<br/>pre_baskets / pre_sales_value / age"]
    W["W: 交絡調整特徴量<br/>事前購買行動 + 世帯属性"]

    D --> T
    D --> Y
    P --> T
    P --> Y
    U -.-> T
    U -.-> Y

    T --> Y

    D --> W
    P --> W
    D --> X
    P --> X

    X -. CATE tau(X) .-> Y
    W -. nuisance adjustment .-> T
    W -. nuisance adjustment .-> Y
```

## Model Mapping

- `T`: `campaigns` に `campaign_id == "18"` で出現する世帯を 1、それ以外を 0。
- `Y`: campaign 18 の実施週 `44-52` における世帯別 `sales_value` 合計。
- `W`: DML の交絡調整に使う特徴量。アウトカム期間の変数は除き、事前購買行動と demographics を使う。
- `X`: CATE の異質性を見る特徴量。現在の実装では `pre_baskets`, `pre_sales_value`, `age`。
- `U`: データに入っていない要因。DAG に入れることで「観測変数で十分に調整できる」という仮定の強さを明示している。

## Identification Assumption

この推定はランダム化実験ではない。したがって、

```text
Y(1), Y(0) independent of T | W
```

つまり「事前購買行動と世帯属性で調整すれば、キャンペーン対象割当は条件付きで無作為に近い」という仮定に依存する。
