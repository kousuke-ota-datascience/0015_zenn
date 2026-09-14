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

既存成果物をすべて廃棄するわけではない。

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

集約ExcelはEntry_ID・名称等のinventory metadata確認には使用してよいが、R3独立判定の入力として既存coding値を参照しない。

## 2.2. Entry単位で完結する

各EntryはR1〜R4を次の順で完結してから次へ進む。

```text
最新main確認
→ baseline文書のblobが固定値と一致することを確認
→ Entry_ID・対象Entryを確認
→ R1 Evidence / 00_contents監査・必要補強
→ R2 Version Scope再固定
→ R3 D01〜D21を独立再コーディング
→ R4 Entry QA
   - D12 → D13 → D15 causal QA
   - D18 L3 evidence QA
   - U / NA / C QA
   - taxonomy gap確認
   - *_10_analysis.md 更新
   - 独立判定後に旧 *_10_analysis.md と差分比較・分類
→ Entry単位commit・push後検証
→ 本control plane更新
→ 次Entry
```

**旧Excelとの比較およびExcelへの同期はEntry完了条件に含めない。** これらはA3パイロット49件のR4完了後、R5以降で一括実施する。

したがって、集約Excelが未比較・未同期であること自体は、個別Entryから次Entryへ進む際のblockerとしない。

## 2.3. 正本関係

本全面再コーディングにおける正本関係を以下のとおり固定する。

- **Evidence正本:** 各Entryの `*_00_contents.md`。典拠、Evidence role、traceability、調査終了根拠を保持する。
- **Coding正本:** 各Entryの `*_10_analysis.md`。Version Scope、D01〜D21、Primary / Secondary / Status、Entry QA結果を保持する。
- **集約Excel:** `urban_legend_parent_child_full_application_v1.xlsx` はCoding正本から同期される**集約・派生成果物**であり、coding judgmentの正本ではない。
- **本control plane:** 作業順序、task status、commit snapshot、baseline変更履歴の管理正本とする。

MarkdownとExcelが不一致の場合、R6 Excel Sync完了前は不一致それ自体を異常とみなさない。R5で原因を監査し、必要なcoding修正がある場合は先に該当 `*_10_analysis.md` を修正して正本を確定し、その後R6でExcelを同期する。

# 3. タスク定義

## 3.1. Entry単位タスク

- **R0 baseline固定:** 現行workflow・10/20/30・tutorial・集約Excelの対象ファイルを固定する。`完了`。
- **R1 Evidence / 00監査:** 外部典拠を再確認し、既存 `00_contents` が現行workflowのEvidence密度・traceability・終了条件を満たすか監査する。不足時は補強する。未文書化Entryは新規作成する。
- **R2 Version Scope再固定:** R1のEvidenceだけから現行規則でVersion Scopeを固定する。既存Scopeへ合わせない。
- **R3 Independent Recode:** D01〜D21を既存コードを見ずに独立判定する。Primary / Secondary / Statusをすべて再測定する。
- **R4 Entry QA:** workflow Step 6相当のEntry内QAを実施する。D12→D13→D15因果列QA、D18 L3 evidence QA、`U / NA / C` QA、taxonomy gap QAを行う。`*_10_analysis.md` をCoding正本として更新した後、再コーディング前snapshotの旧 `*_10_analysis.md` と比較し、不一致を分類する。旧Excelとの比較はR4に含めない。

## 3.2. A3パイロット49件完了後の集約タスク

- **R5 Global Reconciliation:** 49件すべてのR4完了後に開始する。baseline main commit時点の旧Excelと新Coding正本を比較し、旧 `10_analysis` ↔ 旧Excelの不整合、Entry間coding consistency、taxonomy gapの横断傾向、差分分類の一貫性を監査する。新判定を旧値へ合わせる工程ではない。
- **R6 Excel Sync:** R5で確定した新 `*_10_analysis.md` を正として、集約Excelへ49件を一括同期する。
- **R7 Global QA:** 49件の `00_contents` / `10_analysis` / Excel、baseline適用、差分分類、commit履歴、R5 reconciliation、R6同期結果を最終確認する。

statusは `未 / 作業中 / 完了 / 保留 / −` を使用する。

# 4. 不一致の扱い

不一致監査は、独立判定へのアンカリングを防ぐため二段階に分ける。

- **R4:** R3独立判定およびCoding正本更新後に、再コーディング前snapshotの旧 `*_10_analysis.md` と比較する。
- **R5:** 49件すべてのR4完了後に、baseline main commit時点の旧Excelを含む横断reconciliationを行う。

不一致を少なくとも以下へ分類する。

- `Scope mismatch`
- `Evidence mismatch`
- `Code-selection mismatch`
- `Status mismatch`
- `Taxonomy gap`
- `Prior coding error`

旧値との一致を目的にしない。現行baselineとEvidenceに照らして新判定の方が妥当なら、新判定を正とする。

単一Entryの不一致だけを理由にworkflow / taxonomyを変更しない。同型の不整合が複数Entryで再発する、または現行規則で一意に決められない構造的欠陥が確認された場合に限り、baseline変更候補とする。

R5でCoding正本側の修正が必要と判明した場合は、Excelへ直接補正を入れず、該当Entryの `*_10_analysis.md` を先に修正し、必要に応じてR4 statusを再オープンする。

# 5. 進捗集計

## 5.1. Entry単位

| task | 完了 | 作業中 | 未 | 保留 | − |
|---|---:|---:|---:|---:|---:|
| R0 baseline固定 | 1 | 0 | 0 | 0 | 0 |
| R1 Evidence / 00監査 | 1 | 0 | 48 | 0 | 0 |
| R2 Version Scope再固定 | 1 | 0 | 48 | 0 | 0 |
| R3 Independent Recode | 1 | 0 | 48 | 0 | 0 |
| R4 Entry QA | 1 | 0 | 48 | 0 | 0 |

## 5.2. A3パイロット49件完了後

- R5 Global Reconciliation: `未`
- R6 Excel Sync: `未`
- R7 Global QA: `未`

# 6. Entry別進捗

各Entryについて、再コーディング前後のrepository snapshotをfull commit SHAで記録する。

- `再コーディング前 commit SHA`: 当該Entryの再コーディングによる最初の変更を加える直前のcommit。原則としてEntry単位commitのparent commitを記録する。
- `再コーディング後 commit SHA`: 当該EntryのR1〜R4による `00_contents` / `10_analysis` 更新を確定した最後のEntry内容commitを記録する。control plane更新commitは含めない。
- SHAは省略せず40文字のfull SHAを記載する。
- 1 Entryで複数commitを使用した場合は、再コーディング前には最初のEntry変更直前、再コーディング後には最後のR1〜R4内容変更commitを記載する。

これにより、各Entryについて `再コーディング前 commit SHA` → `再コーディング後 commit SHA` の範囲で差分を直接確認できる状態を維持する。

R5〜R7は49件横断taskであるため、Entry別進捗表には列を設けない。

| Entry_ID | 伝承 | R1 00/Evidence | R2 Scope | R3 Recode | R4 QA | 再コーディング前 commit SHA | 再コーディング後 commit SHA | remarks |
|---|---|---|---|---|---|---|---|---|
|0001|口裂け女|完了|完了|完了|完了|3ee9ff1cc9150cec209020f104e9be3e8f190477|134a9ce8394ed0e2e5a7f491c6a52b19cb236c0c|過去regressionで差分検出済み。R1 commit=`c15b7103aeb0ce5d9fd25a7e2f95c54e86032091`。R2 commit=`e376a94f2240461be938583471ee4259b4bcff90`。R3 commit=`9a565a4879688a9a07e6c60a617813e026d46959`。R4内容commit=`134a9ce8394ed0e2e5a7f491c6a52b19cb236c0c`。R2 Scope=1979年初頭〜春の最小安定共有核。旧Excel比較はR5へ移管。|
|0003|赤い紙・青い紙／赤マント系|未|未|未|未|未|未||
|0005|紫の鏡|未|未|未|未|未|未||
|0006|メリーさんの電話|未|未|未|未|未|未||
|0011|こっくりさん|未|未|未|未|未|未|過去reproducibility testで差分検出済み。独立再判定後に比較|
|0019|小さいおじさん|未|未|未|未|未|未||
|0024|幸福の手紙|未|未|未|未|未|未||
|0025|不幸の手紙|未|未|未|未|未|未||
|0059|深泥池の幽霊タクシー|未|未|未|未|未|未||
|0060|タクシー幽霊|未|未|未|未|未|未||
|0081|ピアスの白い糸|未|未|未|未|未|未||
|0089|日本だるま／だるま女|未|未|未|未|未|未||
|0091|ベッドの下の男|未|未|未|未|未|未||
|0101|海外旅行で臓器を抜かれる|未|未|未|未|未|未||
|0112|事故物件は一度別人を住ませれば告知義務が消える|未|未|未|未|未|未||
|0113|井の頭公園のボートに乗ると別れる|未|未|未|未|未|未||
|0118|函館山の切れない木|未|未|未|未|未|未||
|0132|八幡の藪知らず|未|未|未|未|未|未||
|0133|将門塚の祟り|未|未|未|未|未|未||
|0137|犬鳴村|未|未|未|未|未|未||
|0152|青木ヶ原樹海で方位磁針が狂う|未|未|未|未|未|未||
|0157|新郷村キリストの墓|未|未|未|未|未|未||
|0158|虚舟|未|未|未|未|未|未||
|0169|ノストラダムスの大予言|未|未|未|未|未|未||
|0178|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0179|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0180|きさらぎ駅|未|未|未|未|未|未|Golden Regression reference。全件再コーディング工程上は未から開始|
|0181|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0188|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0198|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0225|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0250|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0275|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0309|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0319|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0349|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0356|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0362|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0363|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0365|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0366|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0384|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0385|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0394|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0403|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0410|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0411|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0412|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|
|0413|（集約Excel等のinventory sourceで名称確認）|未|未|未|未|未|未|未文書化ならR1で新規00作成|

# 7. 実行順序

原則としてEntry_ID昇順で実施する。

```text
0001 → 0003 → 0005 → 0006 → 0011 → 0019 → 0024 → 0025
→ 0059 → 0060 → 0081 → 0089 → 0091 → 0101
→ 0112 → 0113 → 0118 → 0132 → 0133 → 0137
→ 0152 → 0157 → 0158 → 0169 → 0178 → 0179 → 0180
→ 0181 → 0188 → 0198 → 0225 → 0250 → 0275 → 0309
→ 0319 → 0349 → 0356 → 0362 → 0363 → 0365 → 0366
→ 0384 → 0385 → 0394 → 0403 → 0410 → 0411 → 0412 → 0413
```

各EntryのR1→R4を完結してから次Entryへ進む。R4完了後に必要なのはCoding正本 `*_10_analysis.md` の確定と旧 `*_10_analysis.md` との差分比較までであり、旧Excel比較・Excel同期は要求しない。

49件すべてのR4完了後、次の順で集約工程へ進む。

```text
R5 Global Reconciliation
→ R6 Excel Sync
→ R7 Global QA
```

R5開始前まで、旧Excelのcoding値を後続EntryのR1〜R4の判定入力として使わない。これにより49件すべての独立再判定を旧Excelから切り離す。

# 8. Entry完了条件

R4完了には最低限、以下を要求する。

- Entry_ID・対象Entryを確認済み
- Evidenceを現行workflowに従って再確認済み
- `00_contents` が現行tutorial・Evidence traceabilityを満たす
- Evidence調査終了条件を満たす
- Version Scopeを現行規則で再固定済み
- D01〜D21を既存値を見ずに独立再判定済み
- Primary / Secondary / Statusを全21次元で確認済み
- D12→D13→D15因果列QA済み
- taxonomy gap QA済み
- D18 L3はX Evidenceの有無を確認済み
- `U / NA / C` を推測で埋めていない
- `*_10_analysis.md` をCoding正本として更新済み
- R3独立判定完了後に、再コーディング前snapshotの旧 `*_10_analysis.md` との差分を確認・分類済み
- `00_contents` と `10_analysis` が整合
- Entry単位commit・push後差分確認済み
- `再コーディング前 commit SHA` / `再コーディング後 commit SHA` をfull SHAで本control planeへ記録済み
- 本control planeを更新済み

**旧Excelとの比較およびExcel同期はR4完了条件ではない。** Excelが未比較・未同期でも上記を満たせば次Entryへ進んでよい。

# 9. baseline変更履歴

| 日付 | 変更対象 | 旧blob / commit | 新blob / commit | 理由 | 既完了Entryへの影響 |
|---|---|---|---|---|---|
|2026-09-14|初期baseline固定|−|main `9570faea998905f074e59df1691748b9cc54d03d`|全面再コーディング開始|全49件未から開始|
|2026-09-14|control plane task構造・正本関係|control plane blob `7740a96b1c6084c4bf22d1ad5a0be6324b0d9dd3`|本commit|Entry単位R4と49件後のExcel reconciliation/syncを分離し、Coding正本を `*_10_analysis.md` と明示|0001は既実施内容で新R4完了条件を満たすためR4完了へ読み替え。旧Excel比較はR5へ移管|

# 10. 最終完了条件

以下をすべて満たした時点で本全面再コーディングを完了とする。

- 49件すべてR1〜R4完了
- 49件すべて現行baselineによるD01〜D21再測定完了
- 49件すべてCoding正本 `*_10_analysis.md` 確定
- R5 Global Reconciliation完了
- R6 Excel Sync完了
- R6完了後、Coding正本と集約Excelの同期対象値が一致
- baseline変更があった場合、必要な再適用を完了
- R7 Global QA完了

本作業の終了時点で、49件は「過去task3実施済み」ではなく、**同一の現行baselineで独立再測定され、Entry QA・横断reconciliation・Excel同期・Global QAまで完了した状態**として扱えるようにする。