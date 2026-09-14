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

# 2. 全面再コーディングの原則

## 2.1. 再利用するもの／しないもの

再利用可: 既存00のEvidence候補、一次・同時代・研究資料への到達経路、tutorial準拠構造。独立判定前に入力として使わない: 旧10のD01〜D21、旧Excel coding値、旧task3判定。原則は **Evidenceは再利用可能、判定は再利用しない**。

## 2.2. Entry単位で完結する

```text
最新main確認 → baseline blob確認 → Entry確認 → pre-SHA確定
→ R1 Evidence / 00監査 → 内容commit → control plane commit
→ R2 Scope再固定 → 内容commit → control plane commit
→ R3 Independent Recode → 内容commit → R3 SHA記録 → control plane commit
→ R4 Entry QA（causal / L3 / U-NA-C / taxonomy gap / 10更新 / 旧10比較）
→ 内容commit → R4 SHA記録 → control plane commit → 次Entry
```

旧Excel比較・同期はR5以降。

## 2.3. 正本関係

- Evidence正本: `*_00_contents.md`
- Coding正本: `*_10_analysis.md`
- 集約Excel: Coding正本から同期される派生成果物
- control plane: task status / checkpoint / baseline管理正本

## 2.4. commit SHA checkpoint

- pre-SHA: R1前に00/10のいずれかを最後に変更したEntry固有commit
- R3 SHA: 旧10参照前の独立判定freeze内容commit
- R4 SHA: QA・10更新・旧10比較まで完了した内容commit
- control plane commitはcheckpointに含めない

# 3. タスク定義

- R0 baseline固定: 完了
- R1 Evidence / 00監査: 外部典拠再確認、traceability・終了条件確認、必要補強
- R2 Version Scope: R1 Evidenceのみで再固定
- R3 Independent Recode: 旧coding値を見ずD01〜D21再測定、旧10参照前freeze
- R4 Entry QA: D12→D13→D15、D18 L3、U/NA/C、taxonomy gap、Coding正本更新、旧10差分分類
- R5 Global Reconciliation: 49件後、旧Excel・Entry間consistency・taxonomy gap横断監査
- R6 Excel Sync: 新10を正としてExcel一括同期
- R7 Global QA: 最終横断確認

status: `未 / 作業中 / 完了 / 保留 / −`

# 4. 不一致の扱い

R4はR3 freeze後に旧10を比較、R5で旧Excelを比較する。不一致は `Scope mismatch / Evidence mismatch / Code-selection mismatch / Status mismatch / Taxonomy gap / Prior coding error` に分類し、旧値への一致を目的にしない。

# 5. 進捗集計

| task | 完了 | 作業中 | 未 | 保留 | − |
|---|---:|---:|---:|---:|---:|
| R0 baseline固定 | 1 | 0 | 0 | 0 | 0 |
| R1 Evidence / 00監査 | 8 | 0 | 41 | 0 | 0 |
| R2 Version Scope再固定 | 8 | 0 | 41 | 0 | 0 |
| R3 Independent Recode | 7 | 0 | 42 | 0 | 0 |
| R4 Entry QA | 7 | 0 | 42 | 0 | 0 |

- R5 Global Reconciliation: `未`
- R6 Excel Sync: `未`
- R7 Global QA: `未`

# 6. Entry別進捗

| Entry_ID | 伝承 | R1 | R2 | R3 | R4 | pre-SHA | R3 SHA | R4 SHA | remarks |
|---|---|---|---|---|---|---|---|---|---|
|0001|口裂け女|完了|完了|完了|完了|4e18c1a977c8c223a26865fac0feeafef5d54b37|9a565a4879688a9a07e6c60a617813e026d46959|134a9ce8394ed0e2e5a7f491c6a52b19cb236c0c|R2=1979年初頭〜春の最小安定核|
|0003|赤い紙・青い紙／赤マント系|完了|完了|完了|完了|5c577eb3d3acfa3f69fd4ca841875888b27305cf|7e08f01ca3119e71268292513504ff79f6987d60|c0d5fac7edad8edd31f4d768277ce4ae6a11bda3|R2=1986東京都色選択型|
|0005|紫の鏡|完了|完了|完了|完了|c2a37792f12b6c2aa5b8760536b13764d14c714d|74c8f1367a53020c2ff02b3ae5072202ad74e7e9|658b95e6fe5ea8f8b0a3be83314cc851a788e016|D17忘却制御をR5へ|
|0006|メリーさんの電話|完了|完了|完了|完了|89b2865013ecd7fca4b39fa914973a0f55eb3b3b|92bbf26600e52525c2c11398c55d32082f0b3d0e|6a36aebbc047908679f4884d4202da5d7a2efc5b|R4 D04=U|
|0011|こっくりさん|完了|完了|完了|完了|1b24d197f9cb94241e26a368a841bc1071671d23|4254c58c1dc3fab6315f62e2846e31401c89f3fb|0b0f861577ceae5f546d0d3796387e27aa65dfe2|D04/D20 gap候補をR5へ|
|0019|小さいおじさん|完了|完了|完了|完了|e7ada0703530358d46b387b4feb149fc9e1b8ad9|b3f91feaba652b5b3b8eeca202bf2bddb4449c34|1f180c1d0fde5100b0cf71d6bc184ad29a562f0d|taxonomy gapなし|
|0024|幸福の手紙|完了|完了|完了|完了|fc120f41ab98bc1f150f624dcafaaa016e2c6d39|41e27cc1c43d79fa231f32b9e0ffdf80fb296811|362878efa061cd073f48b2cf09d9e006c2af6ba9|R4でD01→U、D08→UNSUPPORTED_ASSERTION、D17→LUCK_EXPLOITATION。taxonomy gapなし。|
|0025|不幸の手紙|完了|完了|未|未|aee0942ab1c70a42008c786670c0c0d5b6e04a1e|未|未|R1=`c6f78ace8fcd1dd8255b09137330fdca5d2a0c16`。R2=`aa7bde9fad4d2271a3cb4f745e2feec7bef57ba3`。Scope=1970年前後の制裁前景型郵便チェーン。|
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
|0178|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0179|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0180|きさらぎ駅|未|未|未|未|未|未|未|Golden Regression reference|
|0181|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0188|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0198|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0225|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0250|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0275|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0309|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0319|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0349|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0356|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0362|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0363|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0365|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0366|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0384|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0385|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0394|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0403|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0410|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0411|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0412|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|
|0413|名称未確認|未|未|未|未|未|未|未|R1でinventory確認|

# 7. 実行順序

`0001 → 0003 → 0005 → 0006 → 0011 → 0019 → 0024 → 0025 → 0059 → 0060 → 0081 → 0089 → 0091 → 0101 → 0112 → 0113 → 0118 → 0132 → 0133 → 0137 → 0152 → 0157 → 0158 → 0169 → 0178 → 0179 → 0180 → 0181 → 0188 → 0198 → 0225 → 0250 → 0275 → 0309 → 0319 → 0349 → 0356 → 0362 → 0363 → 0365 → 0366 → 0384 → 0385 → 0394 → 0403 → 0410 → 0411 → 0412 → 0413`

各EntryのR1→R4完結後に次へ。49件R4完了後はR5→R6→R7。

# 8. Entry完了条件

Entry確認、pre-SHA、Evidence再確認、Scope、独立D01〜D21、R3 freeze、causal/L3/U-NA-C/taxonomy QA、Coding正本、旧10差分、R4 SHA、各task後control更新。旧Excel比較・同期はR4条件外。

# 9. baseline変更履歴

| 日付 | 変更対象 | 旧 | 新 | 理由 | 既完了Entryへの影響 |
|---|---|---|---|---|---|
|2026-09-14|初期baseline固定|−|main `9570faea998905f074e59df1691748b9cc54d03d`|全面再コーディング開始|全49件未から開始|
|2026-09-14|control plane task構造・正本関係|blob `7740a96b1c6084c4bf22d1ad5a0be6324b0d9dd3`|commit `8ebcb326dc3ec1ba24019a75aa251a1f9b4eb0ad`|R4とR5以降を分離、Coding正本明示|旧Excel比較をR5へ|

# 10. 最終完了条件

49件R1〜R4、全checkpoint、全Coding正本、R5、R6同期、一致確認、baseline変更再適用、R7を完了する。