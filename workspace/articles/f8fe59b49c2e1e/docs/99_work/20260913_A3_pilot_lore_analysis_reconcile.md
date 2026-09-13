# 0. INTRODUCTION

本書はA3の**パイロット49件reconcileのコントロールプレーン**である。1伝承エントリ完了ごとに更新する。

- コード正本: `docs/20_analysis_summary/urban_legend_parent_child_full_application_v1.xlsx`
- tutorial 00: commit `e6bfa448f00af17f82c1e210507d3dbe5aa8c1b5` / blob `0ed8c22777eb2c213ca0b009016ccc66a01054ce`
- tutorial 10: commit `81c567a8012a58beceb239b22e430e603354a68d` / blob `dccc5e4473bc51c484f16ba916d653ea7383f2f5`
- coding: `10_urban_legend_analysis_axes_theoretical_design.md`, `20_urban_legend_parent_child_code_system.md`, `30_urban_legend_analysis_coding_rules.md`
- status: `未 / 作業中 / 完了 / −`

# 1. タスク整理

- **task 0 基準固定:** 49件・tutorial・coding規則・正本を固定。`完了`。
- **task 1 高密度化:** 旧薄様式18件を高密度化。コード値変更はしない。
- **task 2 tutorial準拠:** 既存24件の00/10をtutorialの見出し・項目・順序へ正規化。コード値・Statusは変更しない。
- **task 3 再コーディング:** 「同じものさしでもう一度測り直す」。49件のD01〜D21をEvidenceから再判定する。
- **task 4 未作成24件:** 新規00/10作成とtask3をEntry単位で実施。
- **task 5 Excel同期:** task3後に49件を正本Excelへ同期。
- **task 6 派生評価再整合:** pilot/full-application evaluationを正本変更後に監査。
- **task 7 横断QA:** 49件の文書・コード・Excel・評価版・commit履歴を最終確認。

Task2の10標準項目は `Primary Child / Value` / `Primary Parent` / `Secondary` / `Status` / `判定根拠` / `この伝承における現れ方`。`Primary:`、`Parent:`、`Secondary Parent:`、`現れ方:`、独自末尾メモは使用しない。留保は該当D内へ置く。

## 1.1. 実行原則

2026-09-13以降、横断的にtask単位で処理する方式をやめ、**Entry単位で必要なtaskを最後まで完結してから次Entryへ進む。**

```text
最新main取得
→ 当該Entryについて必要ならtask1
→ task2
→ task3
→ push後検証
→ 本control plane更新
→ 次Entry
```

すでにtask1/task2が完了しているEntryでは未完了taskだけを実施する。既完了作業を形式的にやり直さない。

# 2. task対応状況

## 2.1. 集計

| task | 完了 | 作業中 | 未 | − |
|---|---:|---:|---:|---:|
| task 1 | 11 | 0 | 7 | 31 |
| task 2 | 18 | 0 | 6 | 25 |
| task 3 | 17 | 0 | 32 | 0 |
| task 4 | 0 | 0 | 24 | 25 |
| task 5 | 0 | 0 | 49 | 0 |

Global: task0=`完了`, task6=`未`, task7=`未`。

## 2.2. Entry別一覧

| Entry_ID | 伝承 | task1 | task2 | task3 | task4 | task5 | remarks |
|---|---|---|---|---|---|---|---|
|0001|口裂け女|完了|完了|完了|−|未|`4e18c1a9`|
|0003|赤い紙・青い紙／赤マント系|完了|完了|完了|−|未|`5c577eb3`|
|0005|紫の鏡|完了|完了|完了|−|未|`c2a37792`|
|0006|メリーさんの電話|完了|完了|完了|−|未|`89b28650`|
|0011|こっくりさん|完了|完了|完了|−|未|`1b24d197`|
|0019|小さいおじさん|完了|完了|完了|−|未|`e7ada070`|
|0024|幸福の手紙|完了|完了|完了|−|未|task2 `c6de296c`; task3 `fc120f41`|
|0025|不幸の手紙|完了|完了|完了|−|未|task2 `566dd4b7`; task3 `aee0942a`|
|0059|深泥池の幽霊タクシー|−|完了|完了|−|未|task2 `b63ade25`; task3 `4c91bfea`|
|0060|タクシー幽霊|−|完了|完了|−|未|task2 `fee2ae0b`; task3 `9fe2bb53`|
|0081|ピアスの白い糸|−|完了|完了|−|未|task2 `3e3697c2`; task3 `ae8e8685`|
|0089|日本だるま／だるま女|−|完了|完了|−|未|task2 `f992e68b`; task3 `de2363b1`|
|0091|ベッドの下の男|−|完了|完了|−|未|task2 `0d795f4e`; task3 `830446ec`|
|0101|海外旅行で臓器を抜かれる|−|完了|完了|−|未|task2 `11f354c4`; task3 `5a1f02b7`|
|0112|事故物件は一度別人を住ませれば告知義務が消える|完了|完了|完了|−|未|task2 `be43f167`; task3 `ddb17303`|
|0113|井の頭公園のボートに乗ると別れる|完了|完了|完了|−|未|task2 `c6cc97ea`; task3 `bf507d60`|
|0118|函館山の切れない木|完了|完了|未|−|未|task2 `aa689d1c`|
|0132|八幡の藪知らず|未|完了|未|−|未|task2 `6d1ca134`; D16旧メモ矛盾保持; task1未|
|0133|将門塚の祟り|未|未|未|−|未|task1→task2→task3|
|0137|犬鳴村|未|未|未|−|未|task1→task2→task3|
|0152|青木ヶ原樹海で方位磁針が狂う|未|未|未|−|未|task1→task2→task3|
|0157|新郷村キリストの墓|未|未|未|−|未|task1→task2→task3|
|0158|虚舟|未|未|未|−|未|task1→task2→task3|
|0169|ノストラダムスの大予言|未|未|未|−|未|task1→task2→task3|
|0178|（未文書化）|−|−|未|未|未|task4＋task3|
|0179|（未文書化）|−|−|未|未|未|task4＋task3|
|0180|きさらぎ駅|−|−|完了|−|未|基準例 `4cd88890`|
|0181|（未文書化）|−|−|未|未|未|task4＋task3|
|0188|（未文書化）|−|−|未|未|未|task4＋task3|
|0198|（未文書化）|−|−|未|未|未|task4＋task3|
|0225|（未文書化）|−|−|未|未|未|task4＋task3|
|0250|（未文書化）|−|−|未|未|未|task4＋task3|
|0275|（未文書化）|−|−|未|未|未|task4＋task3|
|0309|（未文書化）|−|−|未|未|未|task4＋task3|
|0319|（未文書化）|−|−|未|未|未|task4＋task3|
|0349|（未文書化）|−|−|未|未|未|task4＋task3|
|0356|（未文書化）|−|−|未|未|未|task4＋task3|
|0362|（未文書化）|−|−|未|未|未|task4＋task3|
|0363|（未文書化）|−|−|未|未|未|task4＋task3|
|0365|（未文書化）|−|−|未|未|未|task4＋task3|
|0366|（未文書化）|−|−|未|未|未|task4＋task3|
|0384|（未文書化）|−|−|未|未|未|task4＋task3|
|0385|（未文書化）|−|−|未|未|未|task4＋task3|
|0394|（未文書化）|−|−|未|未|未|task4＋task3|
|0403|（未文書化）|−|−|未|未|未|task4＋task3|
|0410|（未文書化）|−|−|未|未|未|task4＋task3|
|0411|（未文書化）|−|−|未|未|未|task4＋task3|
|0412|（未文書化）|−|−|未|未|未|task4＋task3|
|0413|（未文書化）|−|−|未|未|未|task4＋task3|

# 3. 実行順序

既存文書Entryは、現在位置から次の順で完結させる。

```text
0118 → 0132 → 0133 → 0137 → 0152 → 0157 → 0158 → 0169
```

- task1/task2完了済みEntry: task3のみ実施。
- task1未・task2完了Entry: task1で内容密度を補強し、task2準拠を再確認した上でtask3。
- task1/task2とも未のEntry: task1→task2→task3。
- 未文書化24件: task4で高密度・tutorial準拠の00/10を作成し、そのままtask3まで完了。

その後 `task5 → task6 → task7`。

# 4. 完了条件

task0完了、task1対象18件高密度化、既存文書tutorial準拠、49件全個別文書、49件D01〜D21再測定、Markdown/Excel一致、評価資料版整合、横断QA完了、Entry単位commit/push後検証追跡可能。