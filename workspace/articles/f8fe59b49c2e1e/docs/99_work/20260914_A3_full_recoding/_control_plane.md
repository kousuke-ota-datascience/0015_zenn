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
- Review成果物: `docs/99_work/review_10_each_lore/<Entry_ID>/`

# 1. 再コーディングbaseline

- baseline main commit: `9570faea998905f074e59df1691748b9cc54d03d`
- theoretical design blob: `64ed9801ecb17c44f39e04c364c23ce5cb682014`
- code system blob: `192c2f1593e29eb32292a31d564d21ad4aec2427`
- coding rules blob: `4fe19cc922840a46367bc6d0162271d2422996a9`
- workflow blob: `4d26549329e3e0d5fb20604217ec5e20080f19c0`
- tutorial 00 blob: `0ed8c22777eb2c213ca0b009016ccc66a01054ce`
- tutorial 10 blob: `dccc5e4473bc51c484f16ba916d653ea7383f2f5`

# 2. 全面再コーディングの原則

## 2.1. 再利用するもの／しないもの

再利用可: 既存00のEvidence候補、一次・同時代・研究資料への到達経路、tutorial準拠構造。独立判定前に入力として使わない: 旧10のD01〜D21、旧Excel coding値、旧task3判定。原則は **Evidenceは再利用可能、判定は再利用しない**。

## 2.2. Entry単位で完結する

```text
最新main確認 → baseline確認 → Entry確認 → pre-SHA
→ R1 Evidence / 00監査
→ R2 Version Scope
→ R3 Independent Recode → R3 freeze
→ R4 Entry QA → R4 SHA
→ レビュー待（00/10両方）
→ 指摘なし: 完了
→ 指摘あり: 要修正 → 再作業中 → 00修正 → 10再裁定・修正 → control plane更新 → 再レビュー待
→ 再レビュー指摘あり: 要修正へ戻る
→ 00/10とも承認: 完了
```

旧Excel比較・同期はR5以降。Reviewerは正本を直接修正せず、Review成果物を蓄積し、Coder側が裁定して正本を修正する。

`*_00_contents.md` と `*_10_analysis.md` は**両方とも必須レビュー対象**である。一方のみのレビュー／修正でEntryを完了扱いにしてはならない。

**複数Entryを同時並行で再作業しない。1 Entryについて00→10→control plane更新→再レビュー待まで閉じてから次Entryへ進む。**

## 2.3. 正本関係

- Evidence正本: `*_00_contents.md`
- Coding正本: `*_10_analysis.md`
- Review成果物: 外部Reviewerの指摘記録
- Coder裁定記録: `docs/99_work/20260914_A3_full_recoding/<Entry_ID>_review_<seq>_adjudication.md`
- 集約Excel: Coding正本から同期される派生成果物
- control plane: status / review lifecycle / checkpoint / baseline管理正本

## 2.4. checkpoint

- pre-SHA: R1前のEntry固有checkpoint
- R3 SHA: 旧10参照前の独立判定freeze
- R4 SHA: Coder側R4完了checkpoint
- Review後に正本を修正した場合、旧R4 SHAは履歴として保持する
- 再作業後の正本blob / commitはremarksまたはCoder裁定記録へ残す

# 3. タスク定義

- R0 baseline固定
- R1 Evidence / 00監査
- R2 Version Scope
- R3 Independent Recode
- R4 Entry QA
- Review Cycle: 00/10両方の外部Review→Coder修正→再Reviewを承認まで反復
- R5 Global Reconciliation
- R6 Excel Sync
- R7 Global QA

## 3.1. Entry Status

| Status | 定義 |
|---|---|
| `未` | Coder作業未着手 |
| `レビュー待` | Coder初回作業完了、初回レビュー待ち |
| `要修正` | Reviewで修正指摘あり、Coder再作業未着手 |
| `再作業中` | Coder修正中 |
| `再レビュー待` | Coder修正完了、再レビュー待ち |
| `完了` | 00/10ともReviewer承認済み |
| `－（対象外）` | 適用対象外 |

```text
未 → レビュー待 → 完了
未 → レビュー待 → 要修正 → 再作業中 → 再レビュー待 → 完了
再レビュー待 → 要修正 → 再作業中 → 再レビュー待
```

00/10のどちらか一方でも未解決ならEntry全体を完了にしない。

# 4. 不一致の扱い

不一致は `Scope mismatch / Evidence mismatch / Code-selection mismatch / Status mismatch / Taxonomy gap / Prior coding error / Evidence-Analysis responsibility mismatch / Format mismatch` に分類する。旧値またはReviewer値への機械的一致を目的にしない。

# 5. 進捗集計

## 5.1. Entry lifecycle

| Status | 件数 |
|---|---:|
| 未 | 34 |
| レビュー待 | 6 |
| 要修正 | 0 |
| 再作業中 | 0 |
| 再レビュー待 | 0 |
| 完了 | 9 |
| －（対象外） | 0 |

2026-09-15現在、先行9件はReview Cycle完了。0060・0081・0089・0091・0101・0112はCoder R1〜R4完了、00/10外部Review待ち。

## 5.2. Coder workflow checkpoint

| task | R1〜R4到達 | 未到達 |
|---|---:|---:|
| R1 Evidence / 00監査 | 15 | 34 |
| R2 Version Scope再固定 | 15 | 34 |
| R3 Independent Recode | 15 | 34 |
| R4 Entry QA | 15 | 34 |

- R0: `完了`
- Review Cycle: `0001/0003/0005/0006/0011/0019/0024/0025/0059 完了 / 0060・0081・0089・0091・0101・0112 レビュー待 / 34件 未`
- R5: `未`
- R6: `未`
- R7: `未`

# 6. Entry別進捗

| Entry_ID | 伝承 | R1 | R2 | R3 | R4 | Status | pre-SHA | R3 SHA | R4 SHA | remarks |
|---|---|---|---|---|---|---|---|---|---|---|
|0001|口裂け女|完了|完了|完了|完了|完了|4e18c1a977c8c223a26865fac0feeafef5d54b37|9a565a4879688a9a07e6c60a617813e026d46959|134a9ce8394ed0e2e5a7f491c6a52b19cb236c0c|Review_003: 00/10ともPass。Review Cycle完了。10 blob `29fa0f28bd3f3a6b51680c990d5572f2b7edfd51`。|
|0003|赤い紙・青い紙／赤マント系|完了|完了|完了|完了|完了|5c577eb3d3acfa3f69fd4ca841875888b27305cf|7e08f01ca3119e71268292513504ff79f6987d60|c0d5fac7edad8edd31f4d768277ce4ae6a11bda3|Review_004: 00/10ともPass。Review Cycle完了。10 blob `5267babadc521d4c5e33c395ea8471cbd396cf35`。D01=U、D04=Uを維持。|
|0005|紫の鏡|完了|完了|完了|完了|完了|c2a37792f12b6c2aa5b8760536b13764d14c714d|74c8f1367a53020c2ff02b3ae5072202ad74e7e9|658b95e6fe5ea8f8b0a3be83314cc851a788e016|Review_002: 00/10ともPass。D17忘却制御taxonomy gapを保持。|
|0006|メリーさんの電話|完了|完了|完了|完了|完了|89b2865013ecd7fca4b39fa914973a0f55eb3b3b|92bbf26600e52525c2c11398c55d32082f0b3d0e|6a36aebbc047908679f4884d4202da5d7a2efc5b|Review_004: 00/10ともPass。Review Cycle完了。10 blob `0e8829b9d6cc5f1bcde7875f7b1429ec1f3ae178`。D06=Uを維持。|
|0011|こっくりさん|完了|完了|完了|完了|完了|1b24d197f9cb94241e26a368a841bc1071671d23|4254c58c1dc3fab6315f62e2846e31401c89f3fb|0b0f861577ceae5f546d0d3796387e27aa65dfe2|Review_003: 00/10ともPass。Review Cycle完了。10 blob `20642572d572c6dafb0aeca1c9426e50d02bfff9`。|
|0019|小さいおじさん|完了|完了|完了|完了|完了|e7ada0703530358d46b387b4feb149fc9e1b8ad9|b3f91feaba652b5b3b8eeca202bf2bddb4449c34|1f180c1d0fde5100b0cf71d6bc184ad29a562f0d|Review_002: 00/10ともPass。|
|0024|幸福の手紙|完了|完了|完了|完了|完了|fc120f41ab98bc1f150f624dcafaaa016e2c6d39|41e27cc1c43d79fa231f32b9e0ffdf80fb296811|362878efa061cd073f48b2cf09d9e006c2af6ba9|Review_002: 00/10ともPass。|
|0025|不幸の手紙|完了|完了|完了|完了|完了|aee0942ab1c70a42008c786670c0c0d5b6e04a1e|7ea7647a2b29cf93687c71109adce9e5637f3e7f|443b8f6a87afd69f2a0276b8677b8f093b7266e2|Review_002: 00/10ともPass。D04パラメータ変異taxonomy gapを保持。|
|0059|深泥池の幽霊タクシー|完了|完了|完了|完了|完了|4c91bfea47f31b35a5a38799ee145c93f29580f7|67db4d07c94341c71b219b7b6469a4208ea3d39f|60ff1c2a50d75f95f74170a5a9085ee1fb2b5aa3|Review_003: 00/10ともPass。Review Cycle完了。10 blob `acebf290857c155ab5851922169695dae170426b`。|
|0060|タクシー幽霊|完了|完了|完了|完了|レビュー待|82da84f361e6860eea45e865aeea89241f83d400|71f43db9729071d94b456edd5ecca6984f66ea04|a4358ec9dd87c37534702739b919cbbc5587236a|00 Evidence補強 blob `14fb2ce588346d58e8a6969b44b604c0b28135ee`。10 blob `400810d067efea81d171e415ef81a5a0debb6b7e`。D20=U。車内消失異伝ではD13「対象の消失／不在化」taxonomy gapが再出するためGlobal Reconciliation候補。|
|0081|ピアスの白い糸|完了|完了|完了|完了|レビュー待|37beb438acd1f776f6dc7ef506229107172fd0b8|df9307f6b8b6cdd121e9b3f651053126f5aafe09|d71a28ed0a611f895d7d9965bd3bde4d061594bd|10 blob `7fea3f1b5c8ce01f538c182e1daa8a76a832ee13`。D01/D19/D20=U。1994年採録以前の成立・最古口承を推測で補完せず。|
|0089|日本だるま／だるま女|完了|完了|完了|完了|レビュー待|8bd4fa70eb8b87ebea465b5bccb0da2bce956e68|1c7f5568137fb78158556d54171a294fc5d5ca8c|dfe31012e4226d450c3df90be12e5c8ade0fd5b4|10 blob `1737b5fed31072d9d24ad250e7b495d145128713`。D01/D19/D20=U。特定国への犯罪帰属を避け、人間加害主体と身体被害へScopeを限定。|
|0091|ベッドの下の男|完了|完了|完了|完了|レビュー待|0430dbe8e7d2c5ef966fccbaf470739e92cd1f69|eeeb1446a0ce9d3cd78bd06644dc7a7a43595e4b|64078ce3d5db443f89559f00686ac8f84593c917|10 blob `2b0a0ad43d325d567797515af2c88c2195351f20`。D19/D20=U。D13「物理的潜伏・待ち伏せ」をGlobal Reconciliation taxonomy gap候補として保持。|
|0101|海外旅行で臓器を抜かれる|完了|完了|完了|完了|レビュー待|63876d6a290fc9788ae99815081ea7597667c58e|c006ffbf1b8c95188fbf60db3f5b118fa5dabfcf|64d376bd9a254770b1b837f5c257cb3daa2020fd|10 blob `723b3b75301f7ef8a19f52f96adb10e31b9e9d93`。1991年新聞を早期確認点に追加。D18 L3=0。|
|0112|事故物件は一度別人を住ませれば告知義務が消える|完了|完了|完了|完了|レビュー待|86de02b386427c6f99c856867520e191396950b6|267ed26645bec1532614e6e28e50cbc8f4066843|c90b7080769a9cd2ed44cd01e5736dc1b02135b2|10 blob `3a38dc4e0cb1f2fcf3e6009f7355744b584f2c7b`。2008-11公開Q&Aを早期確認点とし、国交省ガイドラインで俗説と制度事実を分離。D18 L3=0。|
|0113|井の頭公園のボートに乗ると別れる|未|未|未|未|未|未|未|未||
|0118|函館山の切れない木|未|未|未|未|未|未|未|未||
|0132|八幡の藪知らず|未|未|未|未|未|未|未|未||
|0133|将門塚の祟り|未|未|未|未|未|未|未|未||
|0137|犬鳴村|未|未|未|未|未|未|未|未||
|0152|青木ヶ原樹海で方位磁針が狂う|未|未|未|未|未|未|未|未||
|0157|新郷村キリストの墓|未|未|未|未|未|未|未|未||
|0158|虚舟|未|未|未|未|未|未|未|未||
|0169|ノストラダムスの大予言|未|未|未|未|未|未|未|未||
|0178|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0179|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0180|きさらぎ駅|未|未|未|未|未|未|未|未|Golden Regression reference|
|0181|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0188|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0198|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0225|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0250|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0275|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0309|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0319|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0349|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0356|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0362|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0363|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0365|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0366|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0384|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0385|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0394|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0403|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0410|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0411|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0412|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|
|0413|名称未確認|未|未|未|未|未|未|未|未|R1でinventory確認|

## 6.1. Review状態

| Entry_ID | 00 | 10 | 総合Status | 次Review |
|---|---|---|---|---|
|0001|Review_003 Pass|Review_003 Pass|完了|－|
|0003|Review_004 Pass|Review_004 Pass|完了|－|
|0005|Review_002 Pass|Review_002 Pass|完了|－|
|0006|Review_004 Pass|Review_004 Pass|完了|－|
|0011|Review_003 Pass|Review_003 Pass|完了|－|
|0019|Review_002 Pass|Review_002 Pass|完了|－|
|0024|Review_002 Pass|Review_002 Pass|完了|－|
|0025|Review_002 Pass|Review_002 Pass|完了|－|
|0059|Review_003 Pass|Review_003 Pass|完了|－|
|0060|未Review|未Review|レビュー待|Review_001|
|0081|未Review|未Review|レビュー待|Review_001|
|0089|未Review|未Review|レビュー待|Review_001|
|0091|未Review|未Review|レビュー待|Review_001|
|0101|未Review|未Review|レビュー待|Review_001|
|0112|未Review|未Review|レビュー待|Review_001|

# 7. 実行順序

`0001 → 0003 → 0005 → 0006 → 0011 → 0019 → 0024 → 0025 → 0059 → 0060 → 0081 → 0089 → 0091 → 0101 → 0112 → 0113 → 0118 → 0132 → 0133 → 0137 → 0152 → 0157 → 0158 → 0169 → 0178 → 0179 → 0180 → 0181 → 0188 → 0198 → 0225 → 0250 → 0275 → 0309 → 0319 → 0349 → 0356 → 0362 → 0363 → 0365 → 0366 → 0384 → 0385 → 0394 → 0403 → 0410 → 0411 → 0412 → 0413`

先行9件はすべてReview Cycle完了。0060・0081・0089・0091・0101・0112はCoder R1〜R4完了・Review_001待ち。次の新規Entryは`0113`。

# 8. Entry完了条件

Entry確認、pre-SHA、Evidence正本、Version Scope、独立D01〜D21、R3 freeze、causal/L3/U-NA-C/taxonomy QA、Coding正本、旧10差分、R4 SHA、外部Reviewerによる00/10双方のReview、指摘反映、修正版再Review、未解決指摘なし、各状態遷移後のcontrol plane更新をすべて満たして `完了` とする。旧Excel同期はR5以降。

# 9. baseline / lifecycle変更履歴

| 日付 | 変更 | 結果 |
|---|---|---|
|2026-09-14|初期baseline固定|main `9570faea998905f074e59df1691748b9cc54d03d`|
|2026-09-14|Review Cycle・7 Status導入|先行9件を旧完了からReview状態へ戻した|
|2026-09-14|0001 Review_001再作業|`要修正 → 再レビュー待`|
|2026-09-14|0003 Review_001再作業|`要修正 → 再レビュー待`|
|2026-09-14|0005 Review_001再作業|`要修正 → 再作業中 → 再レビュー待`|
|2026-09-14|0006 Review_001再作業|`要修正 → 再作業中 → 再レビュー待`|
|2026-09-14|0011 Review_001再作業|`要修正 → 再作業中 → 再レビュー待`|
|2026-09-14|0019 Review_001再作業|`要修正 → 再作業中 → 再レビュー待`|
|2026-09-14|0024 Review_001再作業|`要修正 → 再作業中 → 再レビュー待`|
|2026-09-14|0025 Review_001再作業|`要修正 → 再作業中 → 再レビュー待`|
|2026-09-14|0059 Review_001再作業|`要修正 → 再作業中 → 再レビュー待`|
|2026-09-15|先行9件 Review_002|`4件 完了 / 5件 要修正`|
|2026-09-15|0001 Review_002再作業|`要修正 → 再作業中 → 再レビュー待`|
|2026-09-15|0003 Review_002再作業|`要修正 → 再作業中 → 再レビュー待`|
|2026-09-15|0006 Review_002再作業|`要修正 → 再作業中 → 再レビュー待`|
|2026-09-15|0011 Review_002再作業|`要修正 → 再作業中 → 再レビュー待`|
|2026-09-15|0059 Review_002再作業|`要修正 → 再作業中 → 再レビュー待`|
|2026-09-15|先行5件 Review_003|`0001・0011・0059 完了 / 0003・0006 要修正`|
|2026-09-15|0003 Review_003再作業|`要修正 → 再作業中 → 再レビュー待`。D01/D04をUへ保守化。|
|2026-09-15|0006 Review_003再作業|`要修正 → 再作業中 → 再レビュー待`。D06をUへ保守化。|
|2026-09-15|0003・0006 Review_004|00/10ともPass。先行9件Review Cycle完了。|
|2026-09-15|0060 Coder R1〜R4|`未 → レビュー待`。00 Evidence補強後に独立再コード・R4 QA完了。|
|2026-09-15|0081 Coder R1〜R4|`未 → レビュー待`。D01/D19/D20をUとして保守化。|
|2026-09-15|0089 Coder R1〜R4|`未 → レビュー待`。特定国への犯罪帰属を避け、D01/D19/D20をU。|
|2026-09-15|0091 Coder R1〜R4|`未 → レビュー待`。D13潜伏・待ち伏せtaxonomy gapを保持。|
|2026-09-15|0101 Coder R1〜R4|`未 → レビュー待`。1991年同時代資料・1990年代デジタル再増幅Evidenceを反映。|
|2026-09-15|0112 Coder R1〜R4|`未 → レビュー待`。2008年公開Q&Aと国交省公式ガイドラインを分離し、L3は独立X Evidence不足のため0。|

# 10. 最終完了条件

49件についてR1〜R4 checkpointと00/10 Review Cycleを完了し、全Evidence正本・Coding正本を確定する。その後R5、R6、R7を完了する。