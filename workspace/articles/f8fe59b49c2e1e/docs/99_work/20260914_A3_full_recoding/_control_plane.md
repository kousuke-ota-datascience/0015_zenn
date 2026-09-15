# 0. INTRODUCTION

本書は、A3パイロット49件を**現行workflow・現行coding rulesで全面再コーディングするためのコントロールプレーン**である。旧task3完了状態は履歴として保持するが、本再コーディングでは完了判定として引き継がない。49件すべてを同一baselineで再測定する。

- 集約Excel: `docs/20_analysis_summary/urban_legend_parent_child_full_application_v1.xlsx`
- 標準workflow: `docs/10_each_lore/0000_tutorial/0000_workflow.md`
- tutorial 00: `docs/10_each_lore/0000_tutorial/0000_00_contents.md`
- tutorial 10: `docs/10_each_lore/0000_tutorial/0000_10_analysis.md`
- 理論設計: `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- コード体系: `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- コーディング規則: `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

# 1. 再コーディングbaseline

- baseline main commit: `9570faea998905f074e59df1691748b9cc54d03d`
- theoretical design blob: `64ed9801ecb17c44f39e04c364c23ce5cb682014`
- code system blob: `192c2f1593e29eb32292a31d564d21ad4aec2427`
- coding rules blob: `4fe19cc922840a46367bc6d0162271d2422996a9`
- workflow blob: `4d26549329e3e0d5fb20604217ec5e20080f19c0`
- tutorial 00 blob: `0ed8c22777eb2c213ca0b009016ccc66a01054ce`
- tutorial 10 blob: `dccc5e4473bc51c484f16ba916d653ea7383f2f5`

# 2. 全面再コーディングの原則

- Evidenceは再利用可能、判定は再利用しない。R3 freeze前に旧10・旧Excel coding値を参照しない。
- Entry単位で `最新main確認 → baseline確認 → Entry確認 → pre-SHA → R1 → R2 → R3 freeze → R4 → レビュー待` を完結する。
- 複数Entryを同時並行で再作業しない。00→10→control plane更新まで閉じてから次Entryへ進む。
- Evidence正本=`*_00_contents.md`、Coding正本=`*_10_analysis.md`、control plane=status/checkpoint正本。
- `U/NA/C`はDimension-level status。H3はPrimary exactly 1、Secondary 0–2、ParentはChildから一意導出。
- D18 L3は同一Version Scopeに対する独立した社会現実X Evidenceがある場合のみ1。
- Excel比較・同期はR5以降。00/10双方の外部Review承認前はEntryを`完了`にしない。

# 3. タスク定義

R0 baseline固定 / R1 Evidence監査 / R2 Version Scope / R3 Independent Recode / R4 Entry QA / Review Cycle / R5 Global Reconciliation / R6 Excel Sync / R7 Global QA。

## 3.1. Entry Status
`未 → レビュー待 → 完了` または `未 → レビュー待 → 要修正 → 再作業中 → 再レビュー待 → 完了`。

# 4. 不一致の扱い
`Scope mismatch / Evidence mismatch / Code-selection mismatch / Status mismatch / Taxonomy gap / Prior coding error / Evidence-Analysis responsibility mismatch / Format mismatch`。

# 5. 進捗集計

| Status | 件数 |
|---|---:|
| 未 | 20 |
| レビュー待 | 20 |
| 完了 | 9 |
| 要修正・再作業中・再レビュー待・対象外 | 0 |

| task | R1〜R4到達 | 未到達 |
|---|---:|---:|
| R1 Evidence / 00監査 | 29 | 20 |
| R2 Version Scope再固定 | 29 | 20 |
| R3 Independent Recode | 29 | 20 |
| R4 Entry QA | 29 | 20 |

- R0: `完了`
- Review Cycle: `0001/0003/0005/0006/0011/0019/0024/0025/0059 完了 / 0060・0081・0089・0091・0101・0112・0113・0118・0132・0133・0137・0152・0157・0158・0169・0178・0179・0180・0181・0188 レビュー待 / 20件 未`
- R5/R6/R7: `未`

# 6. Entry別進捗

| Entry_ID | 伝承 | R1 | R2 | R3 | R4 | Status | pre-SHA | R3 SHA | R4 SHA | remarks |
|---|---|---|---|---|---|---|---|---|---|---|
|0001|口裂け女|完了|完了|完了|完了|完了|4e18c1a977c8c223a26865fac0feeafef5d54b37|9a565a4879688a9a07e6c60a617813e026d46959|134a9ce8394ed0e2e5a7f491c6a52b19cb236c0c|Review Pass|
|0003|赤い紙・青い紙／赤マント系|完了|完了|完了|完了|完了|5c577eb3d3acfa3f69fd4ca841875888b27305cf|7e08f01ca3119e71268292513504ff79f6987d60|c0d5fac7edad8edd31f4d768277ce4ae6a11bda3|Review Pass|
|0005|紫の鏡|完了|完了|完了|完了|完了|c2a37792f12b6c2aa5b8760536b13764d14c714d|74c8f1367a53020c2ff02b3ae5072202ad74e7e9|658b95e6fe5ea8f8b0a3be83314cc851a788e016|Review Pass|
|0006|メリーさんの電話|完了|完了|完了|完了|完了|89b2865013ecd7fca4b39fa914973a0f55eb3b3b|92bbf26600e52525c2c11398c55d32082f0b3d0e|6a36aebbc047908679f4884d4202da5d7a2efc5b|Review Pass|
|0011|こっくりさん|完了|完了|完了|完了|完了|1b24d197f9cb94241e26a368a841bc1071671d23|4254c58c1dc3fab6315f62e2846e31401c89f3fb|0b0f861577ceae5f546d0d3796387e27aa65dfe2|Review Pass|
|0019|小さいおじさん|完了|完了|完了|完了|完了|e7ada0703530358d46b387b4feb149fc9e1b8ad9|b3f91feaba652b5b3b8eeca202bf2bddb4449c34|1f180c1d0fde5100b0cf71d6bc184ad29a562f0d|Review Pass|
|0024|幸福の手紙|完了|完了|完了|完了|完了|fc120f41ab98bc1f150f624dcafaaa016e2c6d39|41e27cc1c43d79fa231f32b9e0ffdf80fb296811|362878efa061cd073f48b2cf09d9e006c2af6ba9|Review Pass|
|0025|不幸の手紙|完了|完了|完了|完了|完了|aee0942ab1c70a42008c786670c0c0d5b6e04a1e|7ea7647a2b29cf93687c71109adce9e5637f3e7f|443b8f6a87afd69f2a0276b8677b8f093b7266e2|Review Pass|
|0059|深泥池の幽霊タクシー|完了|完了|完了|完了|完了|4c91bfea47f31b35a5a38799ee145c93f29580f7|67db4d07c94341c71b219b7b6469a4208ea3d39f|60ff1c2a50d75f95f74170a5a9085ee1fb2b5aa3|Review Pass|
|0060|タクシー幽霊|完了|完了|完了|完了|レビュー待|82da84f361e6860eea45e865aeea89241f83d400|71f43db9729071d94b456edd5ecca6984f66ea04|a4358ec9dd87c37534702739b919cbbc5587236a|D20=U、D13 taxonomy gap。|
|0081|ピアスの白い糸|完了|完了|完了|完了|レビュー待|37beb438acd1f776f6dc7ef506229107172fd0b8|df9307f6b8b6cdd121e9b3f651053126f5aafe09|d71a28ed0a611f895d7d9965bd3bde4d061594bd|D01/D19/D20=U。|
|0089|日本だるま／だるま女|完了|完了|完了|完了|レビュー待|8bd4fa70eb8b87ebea465b5bccb0da2bce956e68|1c7f5568137fb78158556d54171a294fc5d5ca8c|dfe31012e4226d450c3df90be12e5c8ade0fd5b4|D01/D19/D20=U。|
|0091|ベッドの下の男|完了|完了|完了|完了|レビュー待|0430dbe8e7d2c5ef966fccbaf470739e92cd1f69|eeeb1446a0ce9d3cd78bd06644dc7a7a43595e4b|64078ce3d5db443f89559f00686ac8f84593c917|D13潜伏taxonomy gap。|
|0101|海外旅行で臓器を抜かれる|完了|完了|完了|完了|レビュー待|63876d6a290fc9788ae99815081ea7597667c58e|c006ffbf1b8c95188fbf60db3f5b118fa5dabfcf|64d376bd9a254770b1b837f5c257cb3daa2020fd|L3=0。|
|0112|事故物件は一度別人を住ませれば告知義務が消える|完了|完了|完了|完了|レビュー待|86de02b386427c6f99c856867520e191396950b6|267ed26645bec1532614e6e28e50cbc8f4066843|c90b7080769a9cd2ed44cd01e5736dc1b02135b2|国交省ガイドライン分離、L3=0。|
|0113|井の頭公園のボートに乗ると別れる|完了|完了|完了|完了|レビュー待|b778af5f1b7bb38d10323dc563191f6d67fe1174|7acdb0e50cdfedfc984bd0e14cf45b739eeb90f7|5be227c1b03e2a1aeabcbb255822778c166a0fd0|D01=U、L3=0。|
|0118|函館山の切れない木|完了|完了|完了|完了|レビュー待|cfe370ce21791dcaaa815da1d636ff41572c11c6|c17adc9a37c3019e585346f3b8ebbdcdc0346f32|63c122601d2fe7bf975f047f5f8b7dd9b1fe2b3a|D10 taxonomy gap、L3=0。|
|0132|八幡の藪知らず|完了|完了|完了|完了|レビュー待|04a86cc9c6fbf4405bd6123aa5425a0e0b0bdd97|39ca6f738f62972ac58b1a4f5ba875143a5cabf9|9d662fe4e36baa4cfda390b9a4744260467aaba4|D01=G0/I、L3=1限定。|
|0133|将門塚の祟り|完了|完了|完了|完了|レビュー待|54f22c7712abcd2205231ed76a2fe1885a8ca911|058af26a784d134cd6c02e72fac278cb3447f037|ae74f13a5c03563f8c556987e17fae78343f3b28|D01=G0/I、L3=1。|
|0137|犬鳴村|完了|完了|完了|完了|レビュー待|b240224bcfdb7b215ef2ccf1bd45ad9e551cc8a9|5310aca86bb3719fa09703f6693804ad546ef62c|399880d335a6d9cc55b5d9004887e34dcc2a7c32|D01=U、L3=0。|
|0152|青木ヶ原樹海で方位磁針が狂う|完了|完了|完了|完了|レビュー待|b52c65cc5f1a7910784c571d1d9d9abf759318ea|684c16cb9c01a5fee5babcb8e415ca775fd65d6b|ed326e171be81b02f57026b429456142e48998c7|D17=U、大学検証L3=1。|
|0157|新郷村キリストの墓|完了|完了|完了|完了|レビュー待|4933870b3e3062c8ce16ee4eec7aa0ee76b0b693|68f901f86daa9a2f0e53d00a80ac791bc9b3fa3e|afcc03d239e668f141f10cd6c3d83b655358ace5|D01=G1/I、L3=1。|
|0158|虚舟|完了|完了|完了|完了|レビュー待|90f08e7968e188540022762a88926b7ea69586fc|aa7968f1cd5d12817409a534247050e5e8bf6cef|6f9e7370a626d64544e630302a06e93bcaa5c4e0|D01=G0/I、L3=0。|
|0169|ノストラダムスの大予言|完了|完了|完了|完了|レビュー待|f0632eee87636fe2f584511c3544cd443d9290d0|598f41fc79c2871b347c4981d4d9984057b87ae0|50f54b95bb478e739dd93aef58b248828433b285|D01=G3/D、D16=DEADLINE、L3=0。|
|0178|猿夢|完了|完了|完了|完了|レビュー待|1a85ab52869944c1f8c2ee3a6852fdba832884b6|e7f31217b47a73cf8c226e991b57cbe4c237e914|64429af4d65cfc5b941e6db878f0f6a81779ba01|2000年原型。D17覚醒離脱taxonomy gap、L3=0。|
|0179|くねくね|完了|完了|完了|完了|レビュー待|b6aac89cc94e864063d2e9cdff332bfa78a61153|c1e1c625fa13c2f1d164fc67703bcc7e09397888|55e44c086c579b9b4a97d21df109addafb828def|2001原型→2003増補。D11=UNDERSTAND_RECOGNIZE、D13認知災害、L3=0。|
|0180|きさらぎ駅|完了|完了|完了|完了|レビュー待|390a59182cffc7c6982c45af037e732c514c4fc0|24a7d6ec5a50972f27a2a39fff2a57045023aaf4|0ef39d4cb61802c34fe4e271b3cdc1aa867feeb9|D09=非解決維持、D11=ACCIDENT_INVOLVEMENT、D17=NO_KNOWN_ESCAPE、L3=0。|
|0181|コトリバコ|完了|完了|完了|完了|レビュー待|15e4277b3be93a189cc168fa20f62cfb5a63f048|2453520af5a0aa8d99e175f896e4df8f6a160db5|afc96e6404a9fb7ce0e4731052d5da5598bc6c57|2005-06-06→06-11早期増補、D21=A4、L3=0。|
|0188|八尺様|完了|完了|完了|完了|レビュー待|4d762489c220d89837df9edf603d84780b2195f9|9d59272c2d141ef05a8e4f9ec80633ceb3fb7a21|94d7be41f4d6e2e61a7f1546e11e77eb773e1d4b|00 blob `fde803b7ba94a24d9686d49ca1f592530f7208a3`、10 blob `0438034369f0e471f131a2bdaeddce8bc0d574eb`。D15=FUTURE_CONSTRAINT、L3=0。|
|0198|一人かくれんぼ|未|未|未|未|未|未|未|未|inventory確認済み: 2006〜、ネット発祥Yes。|
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
完了9件は既存の最終Review Passを維持。レビュー待20件は00/10とも未Review・次Review=`Review_001`。

# 7. 実行順序
`0001 → 0003 → 0005 → 0006 → 0011 → 0019 → 0024 → 0025 → 0059 → 0060 → 0081 → 0089 → 0091 → 0101 → 0112 → 0113 → 0118 → 0132 → 0133 → 0137 → 0152 → 0157 → 0158 → 0169 → 0178 → 0179 → 0180 → 0181 → 0188 → 0198 → 0225 → 0250 → 0275 → 0309 → 0319 → 0349 → 0356 → 0362 → 0363 → 0365 → 0366 → 0384 → 0385 → 0394 → 0403 → 0410 → 0411 → 0412 → 0413`

次の新規Entryは`0198`。

# 8. Entry完了条件
R1〜R4 checkpoint、00/10 Review Cycle、未解決指摘なしを満たして`完了`とする。旧Excel同期はR5以降。

# 9. lifecycle変更履歴
- 2026-09-14: baseline固定。
- 2026-09-15: 先行9件Review Cycle完了。
- 2026-09-15: 0060〜0169を順次Coder R1〜R4完了・レビュー待へ移行。
- 2026-09-15: 0178 猿夢をレビュー待へ移行。
- 2026-09-15: 0179 くねくねをレビュー待へ移行。
- 2026-09-15: 0180 きさらぎ駅をレビュー待へ移行。
- 2026-09-15: 0181 コトリバコをレビュー待へ移行。
- 2026-09-15: 0188 八尺様をレビュー待へ移行。2008原型を固定し、主人公の実現帰結を長期帰還制約として裁定。

# 10. 最終完了条件
49件についてR1〜R4 checkpointと00/10 Review Cycleを完了し、全Evidence正本・Coding正本を確定後、R5・R6・R7を完了する。
