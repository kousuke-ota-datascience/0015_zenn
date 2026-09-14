# 0. INTRODUCTION

本書は、A3パイロット49件を**現行workflow・現行coding rulesで全面再コーディングするためのコントロールプレーン**である。

旧 `20260913_A3_pilot_lore_analysis_reconcile.md` のtask3完了状態は履歴として保持するが、本再コーディングではそれを完了判定として引き継がない。49件すべてを同一baselineで再測定する。

目的は、workflow・coding rulesの精緻化後に、Entry間で同一の測定条件を確保することである。

- 集約Excel: `docs/20_analysis_summary/urban_legend_parent_child_full_application_v1.xlsx`
- 標準workflow: `docs/10_each_lore/0000_tutorial/0000_workflow.md`
- tutorial 00: `docs/10_each_lore/0000_tutorial/0000_00_contents.md`
- tutorial 10: `docs/10_each_lore/0000_tutorial/0000_10_analysis.md`
- 理論設計: `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- コード体系: `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- コーディング規則: `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

# 1. 再コーディングbaseline

本作業開始時点のbaselineを以下で固定する。

- baseline main commit: `9570faea998905f074e59df1691748b9cc54d03d`
- `10_urban_legend_analysis_axes_theoretical_design.md` blob: `64ed9801ecb17c44f39e04c364c23ce5cb682014`
- `20_urban_legend_parent_child_code_system.md` blob: `192c2f1593e29eb32292a31d564d21ad4aec2427`
- `30_urban_legend_analysis_coding_rules.md` blob: `4fe19cc922840a46367bc6d0162271d2422996a9`
- `0000_workflow.md` blob: `4d26549329e3e0d5fb20604217ec5e20080f19c0`
- `0000_00_contents.md` blob: `0ed8c22777eb2c213ca0b009016ccc66a01054ce`
- `0000_10_analysis.md` blob: `dccc5e4473bc51c484f16ba916d653ea7383f2f5`

このbaselineで49件を完走することを原則とする。

作業中にworkflow / `10 / 20 / 30` の変更が必要になった場合は、単一Entryの都合で即時変更しない。構造的な欠陥かを確認し、変更する場合は本書の「baseline変更履歴」に記録し、**変更前に完了したEntryへ再適用が必要かを明示する**。

# 2. 全面再コーディングの原則

## 2.1. 再利用するもの／しないもの

再利用してよいもの:
- 既存 `*_00_contents.md` に整理済みの典拠候補・Evidence候補
- 既存の一次資料・同時代資料・研究資料への到達経路
- tutorial準拠済みの文書構造

ただし、既存 `00_contents` の記載を無検証でEvidenceとみなさず、現行workflowのEvidence優先順位・終了条件に従って再確認する。

独立判定完了前に入力として使わないもの:
- 既存 `*_10_analysis.md` のD01〜D21
- 集約Excelの既存D01〜D21・Primary / Secondary・Status
- 旧task3の判定結果

原則は、**Evidenceは再利用可能、判定は再利用しない**である。

## 2.2. Entry単位で完結する

```text
最新main確認
→ baseline文書blob確認
→ Entry_ID確認
→ 再コーディング前 commit SHA確定
→ R1 Evidence / 00監査
→ R1内容commit / push
→ control plane更新commit / push
→ R2 Version Scope再固定
→ R2内容commit / push
→ control plane更新commit / push
→ R3 Independent Recode
→ R3内容commit / push
→ R3 後 commit SHA記録
→ control plane更新commit / push
→ R4 Entry QA
   - D12 → D13 → D15 causal QA
   - D18 L3 evidence QA
   - U / NA / C QA
   - taxonomy gap確認
   - *_10_analysis.md 更新
   - 再コーディング前snapshotの旧 *_10_analysis.md との差分比較・分類
→ R4内容commit / push
→ R4 後 commit SHA記録
→ control plane更新commit / push
→ 次Entry
```

R1〜R4は**各task完了ごとに内容commit / push、その後control plane更新commit / push**を行う。旧Excelとの比較・同期はEntry完了条件に含めずR5以降で一括実施する。

## 2.3. 正本関係

- **Evidence正本:** 各Entryの `*_00_contents.md`
- **Coding正本:** 各Entryの `*_10_analysis.md`
- **集約Excel:** Coding正本から同期される集約・派生成果物
- **本control plane:** 作業順序、task status、commit checkpoint、baseline変更履歴の管理正本

## 2.4. commit SHA checkpointの定義

- **再コーディング前 commit SHA:** R1着手以前に当該Entryの `*_00_contents.md` または `*_10_analysis.md` のいずれかを最後に変更したcommit。2ファイルで異なる場合はmain履歴上で後のcommit。
- **R3 後 commit SHA:** 旧 `*_10_analysis.md` 参照前に独立判定をfreezeした内容commit。
- **R4 後 commit SHA:** R4 Entry QA、Coding正本更新、旧10との差分比較・分類まで完了した内容commit。
- control plane更新commitはcheckpoint SHAに含めない。

# 3. タスク定義

## 3.1. Entry単位タスク

- **R0 baseline固定:** `完了`。
- **R1 Evidence / 00監査:** 外部典拠を再確認し、Evidence密度・traceability・終了条件を満たすか監査。不足時は補強。
- **R2 Version Scope再固定:** R1 Evidenceだけから現行規則でScopeを固定。既存Scopeへ合わせない。
- **R3 Independent Recode:** D01〜D21を既存コードを見ずに独立判定し、旧10参照前にcommit / pushしてfreeze。
- **R4 Entry QA:** D12→D13→D15、D18 L3、U/NA/C、taxonomy gapをQAし、Coding正本を更新後、再コーディング前snapshotの旧10との差分を比較・分類。旧Excel比較は含めない。

## 3.2. A3パイロット49件完了後

- **R5 Global Reconciliation:** 旧Excelとの比較、旧10↔旧Excel不整合、Entry間consistency、taxonomy gap、差分分類を横断監査。
- **R6 Excel Sync:** 新 `*_10_analysis.md` を正として集約Excelへ一括同期。
- **R7 Global QA:** 00 / 10 / Excel、baseline、差分分類、checkpoint、R5/R6を最終確認。

statusは `未 / 作業中 / 完了 / 保留 / −`。

# 4. 不一致の扱い

R4ではR3 freeze後に旧10を比較し、R5で旧Excelを含む横断reconciliationを行う。不一致は少なくとも `Scope mismatch / Evidence mismatch / Code-selection mismatch / Status mismatch / Taxonomy gap / Prior coding error` に分類する。旧値との一致を目的にしない。

# 5. 進捗集計

## 5.1. Entry単位

| task | 完了 | 作業中 | 未 | 保留 | − |
|---|---:|---:|---:|---:|---:|
| R0 baseline固定 | 1 | 0 | 0 | 0 | 0 |
| R1 Evidence / 00監査 | 7 | 0 | 42 | 0 | 0 |
| R2 Version Scope再固定 | 6 | 0 | 43 | 0 | 0 |
| R3 Independent Recode | 6 | 0 | 43 | 0 | 0 |
| R4 Entry QA | 6 | 0 | 43 | 0 | 0 |

## 5.2. A3パイロット49件完了後

- R5 Global Reconciliation: `未`
- R6 Excel Sync: `未`
- R7 Global QA: `未`

# 6. Entry別進捗

| Entry_ID | 伝承 | R1 00/Evidence | R2 Scope | R3 Recode | R4 QA | 再コーディング前 commit SHA | R3 後 commit SHA | R4 後 commit SHA | remarks |
|---|---|---|---|---|---|---|---|---|---|
|0001|口裂け女|完了|完了|完了|完了|4e18c1a977c8c223a26865fac0feeafef5d54b37|9a565a4879688a9a07e6c60a617813e026d46959|134a9ce8394ed0e2e5a7f491c6a52b19cb236c0c|R2 Scope=1979年初頭〜春の最小安定共有核。旧Excel比較はR5へ移管。|
|0003|赤い紙・青い紙／赤マント系|完了|完了|完了|完了|5c577eb3d3acfa3f69fd4ca841875888b27305cf|7e08f01ca3119e71268292513504ff79f6987d60|c0d5fac7edad8edd31f4d768277ce4ae6a11bda3|R2 Scope=1986年東京都の色選択型共通核。|
|0005|紫の鏡|完了|完了|完了|完了|c2a37792f12b6c2aa5b8760536b13764d14c714d|74c8f1367a53020c2ff02b3ae5072202ad74e7e9|658b95e6fe5ea8f8b0a3be83314cc851a788e016|D17忘却制御をR5横断確認へ。|
|0006|メリーさんの電話|完了|完了|完了|完了|89b2865013ecd7fca4b39fa914973a0f55eb3b3b|92bbf26600e52525c2c11398c55d32082f0b3d0e|6a36aebbc047908679f4884d4202da5d7a2efc5b|R4でD04をUへQA修正。|
|0011|こっくりさん|完了|完了|完了|完了|1b24d197f9cb94241e26a368a841bc1071671d23|4254c58c1dc3fab6315f62e2846e31401c89f3fb|0b0f861577ceae5f546d0d3796387e27aa65dfe2|D04文化的／物質的適応、D20超自然的情報保持者をR5候補。|
|0019|小さいおじさん|完了|完了|完了|完了|e7ada0703530358d46b387b4feb149fc9e1b8ad9|b3f91feaba652b5b3b8eeca202bf2bddb4449c34|1f180c1d0fde5100b0cf71d6bc184ad29a562f0d|R2 Scope=2007–2009年初期共有核。taxonomy gapなし。|
|0024|幸福の手紙|完了|未|未|未|fc120f41ab98bc1f150f624dcafaaa016e2c6d39|未|未|R1 commit=`e0df2aab4cc025832de0ee55410e3e60d799edff`。1922年研究資料＋1951年国会会議録でC/H/X/Aを補強。|
|0025|不幸の手紙|未|未|未|未|未|未|未||
|0059|深泥池の幽霊タクシー|未|未|未|未|未|未|未||
|0060|タクシー幽霊|未|未|未|未|未|未|未||
|0081|ピアスの白い糸|未|未|未|未|未|未|未||
|0089|日本だるま／だるま女|未|未|未|未|未|未|未||
|0091|ベッドの下の男|未|未|未|未|未|未|未||
|0101|海外旅行で臓器を抜かれる|未|未|未|未|未|未|未||
|0112|事故物件は一度別人を住ませれば告知義務が消える|未|未|未|未|未|未|未||
|0113|井の頭公園のボートに乗ると別れる|未|未|未|未|未|未|未||
|0118|函館山の切れない木|未|未|未|未|未|未|未||
|0132|八幡の藪知らず|未|未|未|未|未|未|未||
|0133|将門塚の祟り|未|未|未|未|未|未|未||
|0137|犬鳴村|未|未|未|未|未|未|未||
|0152|青木ヶ原樹海で方位磁針が狂う|未|未|未|未|未|未|未||
|0157|新郷村キリストの墓|未|未|未|未|未|未|未||
|0158|虚舟|未|未|未|未|未|未|未||
|0169|ノストラダムスの大予言|未|未|未|未|未|未|未||
|0178|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0179|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0180|きさらぎ駅|未|未|未|未|未|未|未|Golden Regression reference。|
|0181|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0188|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0198|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0225|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0250|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0275|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0309|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0319|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0349|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0356|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0362|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0363|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0365|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0366|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0384|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0385|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0394|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0403|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0410|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0411|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0412|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0413|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未|未文書化ならR1で新規00作成|

# 7. 実行順序

```text
0001 → 0003 → 0005 → 0006 → 0011 → 0019 → 0024 → 0025
→ 0059 → 0060 → 0081 → 0089 → 0091 → 0101
→ 0112 → 0113 → 0118 → 0132 → 0133 → 0137
→ 0152 → 0157 → 0158 → 0169 → 0178 → 0179 → 0180
→ 0181 → 0188 → 0198 → 0225 → 0250 → 0275 → 0309
→ 0319 → 0349 → 0356 → 0362 → 0363 → 0365 → 0366
→ 0384 → 0385 → 0394 → 0403 → 0410 → 0411 → 0412 → 0413
```

各EntryのR1→R4を完結してから次Entryへ進む。各taskの内容commit/push後にcontrol planeを更新する。49件R4完了後にR5→R6→R7へ進む。

# 8. Entry完了条件

R4完了には、Entry確認、pre-SHA確定、Evidence再確認、Scope再固定、D01〜D21独立判定、R3 freeze、D12→D13→D15 QA、D18 L3 QA、U/NA/C QA、taxonomy gap QA、Coding正本更新、旧10差分分類、R4 commit、R4 SHA記録、各task後control plane更新を要求する。旧Excel比較・同期はR4完了条件ではない。

# 9. baseline変更履歴

| 日付 | 変更対象 | 旧blob / commit | 新blob / commit | 理由 | 既完了Entryへの影響 |
|---|---|---|---|---|---|
|2026-09-14|初期baseline固定|−|main `9570faea998905f074e59df1691748b9cc54d03d`|全面再コーディング開始|全49件未から開始|
|2026-09-14|control plane task構造・正本関係|control plane blob `7740a96b1c6084c4bf22d1ad5a0be6324b0d9dd3`|commit `8ebcb326dc3ec1ba24019a75aa251a1f9b4eb0ad`|Entry単位R4と49件後のExcel reconciliation/syncを分離し、Coding正本を `*_10_analysis.md` と明示|0001は新R4完了条件を満たす。旧Excel比較はR5へ移管|

# 10. 最終完了条件

- 49件すべてR1〜R4完了
- 49件すべて現行baselineによるD01〜D21再測定完了
- 49件すべてpre / R3 / R4 checkpoint SHA確定
- 49件すべてCoding正本確定
- R5 Global Reconciliation完了
- R6 Excel Sync完了
- R6後、Coding正本と集約Excelが一致
- baseline変更時の必要再適用完了
- R7 Global QA完了