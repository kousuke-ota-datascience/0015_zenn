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
- Entry単位で `最新main確認 → baseline確認 → Entry確認 → R1 → R2 → R3 freeze → R4 → 00/10レビュー待` を完結する。
- 複数Entryを同時並行で再作業しない。00→10→control plane更新まで閉じてから次Entryへ進む。
- 正本成果物は Evidence=`*_00_contents.md`、Coding=`*_10_analysis.md` の2本とする。
- R1/R2/R3/R4およびReview Cycleで作成する `docs/99_work/` 配下の文書は、監査・checkpoint・裁定記録であり、Entry別進捗の主単位にはしない。
- `U/NA/C`はDimension-level status。H3はPrimary exactly 1、Secondary 0–2、ParentはChildから一意導出。
- D18 L3は同一Version Scopeに対する独立した社会現実X Evidenceがある場合のみ1。
- Excel比較・同期はR5以降。00/10双方の外部Review承認前はEntryを`完了`にしない。

## 2.1. commit / push 単位

今後は**正本成果物1ファイルを1変更単位**としてcommit→pushする。

```text
成果物編集
→ 成果物のみcommit / push
→ post-SHA取得
→ control planeの該当行を更新
→ control planeを別commit / push
```

- `pre-SHA` / `post-SHA` は Git commit SHA を指す。
- `pre-SHA` は当該成果物を変更する直前のmain commit、`post-SHA` は当該成果物変更commitを指す。
- 00と10を同一commitへまとめない。
- Review修正で00/10双方を変更する場合も、成果物ごとにcommitを分離する。
- control plane自身の更新は成果物commitとは別commitにする。これによりpost-SHAの自己参照を避ける。
- 本フォーマット移行前のEntryは、旧 `pre-SHA / R3 SHA / R4 SHA` を失わないようlegacy checkpointとして移行する。00単独commitを示さないSHAはremarksで明示する。

# 3. タスク定義

## 3.1. Step & deliverables

| task ID | task name | 正本deliverables | audit / checkpoint artifacts |
|---|---|---|---|
| R0 | baseline固定 | － | control planeのbaseline SHA記録 |
| R1 | Evidence/00監査 | `*_00_contents.md` | 必要に応じ `*_R1_evidence_audit.md` |
| R2 | Version Scope | － | `*_R2_version_scope.md` |
| R3 | Independent Recode | － | `*_R3_independent_recode.md`（freeze対象） |
| R4 | Entry QA | `*_10_analysis.md` | `*_R4_entry_qa.md` |
| Review Cycle | 00/10両方の外部Review→Coder修正→再Reviewを承認まで反復 | 修正対象の `*_00_contents.md` / `*_10_analysis.md` | `review_10_each_lore/<Entry_ID>/Review_*`、Coder adjudication |
| R5 | Global Reconciliation | 必要に応じて各 `*_10_analysis.md` を更新 | global reconciliation記録 |
| R6 | Excel Sync | `urban_legend_parent_child_full_application_v1.xlsx` | sync / diff確認記録 |
| R7 | Global QA | 確定した00/10/Excel | global QA記録・control plane最終化 |

R2/R3/R4の補助文書は監査可能性のため保持するが、進捗管理の主単位は正本成果物 `00` / `10` とする。

## 3.2. Entry Status

| Status | 定義 |
|---|---|
| `未` | Coder作業未着手 |
| `レビュー待` | Coder初回作業完了、初回レビュー待ち |
| `要修正` | Reviewで修正指摘あり、Coder再作業未着手 |
| `再作業中` | Coder修正中 |
| `再レビュー待` | Coder修正完了、再レビュー待ち |
| `完了` | Reviewer承認済み |
| `－（対象外）` | 適用対象外 |

```text
未 → レビュー待 → 完了
未 → レビュー待 → 要修正 → 再作業中 → 再レビュー待 → 完了
再レビュー待 → 要修正 → 再作業中 → 再レビュー待
```

Section 6ではこのStatusを**成果物行単位**で適用する。Entry全体の`完了`は00行・10行がともに`完了`の場合のみ成立する。片方だけが承認済みの場合、Entry全体は未承認側の状態に従う。

## 3.3. 最新レビュー版

`最新レビュー版` は、その成果物に対して**最後に完了したReviewの3桁連番**を記録する。

- `－`: 未レビュー。`Review_001` がまだ実施されていない。
- `001`: `Review_<Entry_ID>_<成果物>_001.md` が最新レビュー。
- `002`: `Review_<Entry_ID>_<成果物>_002.md` が最新レビュー。以下同様。
- `完了 / 002`: `Review_002` でPassとなり、その成果物が承認完了したことを表す。
- `要修正 / 001`: `Review_001` で修正指摘が出て、Coder再作業前であることを表す。
- `再作業中 / 001`: 最新の完了Reviewは`001`で、その指摘に対するCoder修正中であることを表す。
- `再レビュー待 / 001`: `Review_001` の指摘に対する修正が完了し、次の `Review_002` を待っていることを表す。

したがって、`最新レビュー版` は「次に作るReview番号」ではなく、「最後に完了したReview番号」である。

# 4. 不一致の扱い

`Scope mismatch / Evidence mismatch / Code-selection mismatch / Status mismatch / Taxonomy gap / Prior coding error / Evidence-Analysis responsibility mismatch / Format mismatch`。

# 5. 進捗集計

## 5.1. Entry単位

| Status | 件数 |
|---|---:|
| 未 | 16 |
| レビュー待 | 0 |
| 要修正 | 21 |
| 再作業中 | 0 |
| 再レビュー待 | 3 |
| 完了 | 9 |
| 対象外 | 0 |

## 5.2. 正本成果物単位

| Status | 00/10成果物件数 |
|---|---:|
| 未 | 32 |
| レビュー待 | 0 |
| 要修正 | 42 |
| 再作業中 | 0 |
| 再レビュー待 | 6 |
| 完了 | 18 |
| 対象外 | 0 |
| 合計 | 98 |

- 完了Entry: `0001/0003/0005/0006/0011/0019/0024/0025/0059`
- 再レビュー待Entry: `0091/0101/0112`
- 要修正Entry: `0060/0081/0089/0113/0118/0132/0133/0137/0152/0157/0158/0169/0178/0179/0180/0181/0188/0198/0225/0250/0275`
- 未着手Entry: 16件
- R5/R6/R7は49件の00/10 Review Cycle確定後に実施する。

# 6. Entry別進捗

| Entry_ID | 伝承 | 成果物 | Status | 最新レビュー版 | pre-SHA | post-SHA | remarks |
|---|---|---|---|---|---|---|---|
| 0001 | 口裂け女 | 00 | 完了 | `003` | `4e18c1a977c8c223a26865fac0feeafef5d54b37` | `9a565a4879688a9a07e6c60a617813e026d46959` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0001 | 口裂け女 | 10 | 完了 | `003` | `9a565a4879688a9a07e6c60a617813e026d46959` | `134a9ce8394ed0e2e5a7f491c6a52b19cb236c0c` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0003 | 赤い紙・青い紙／赤マント系 | 00 | 完了 | `004` | `5c577eb3d3acfa3f69fd4ca841875888b27305cf` | `7e08f01ca3119e71268292513504ff79f6987d60` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0003 | 赤い紙・青い紙／赤マント系 | 10 | 完了 | `004` | `7e08f01ca3119e71268292513504ff79f6987d60` | `c0d5fac7edad8edd31f4d768277ce4ae6a11bda3` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0005 | 紫の鏡 | 00 | 完了 | `002` | `c2a37792f12b6c2aa5b8760536b13764d14c714d` | `74c8f1367a53020c2ff02b3ae5072202ad74e7e9` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0005 | 紫の鏡 | 10 | 完了 | `002` | `74c8f1367a53020c2ff02b3ae5072202ad74e7e9` | `658b95e6fe5ea8f8b0a3be83314cc851a788e016` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0006 | メリーさんの電話 | 00 | 完了 | `004` | `89b2865013ecd7fca4b39fa914973a0f55eb3b3b` | `92bbf26600e52525c2c11398c55d32082f0b3d0e` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0006 | メリーさんの電話 | 10 | 完了 | `004` | `92bbf26600e52525c2c11398c55d32082f0b3d0e` | `6a36aebbc047908679f4884d4202da5d7a2efc5b` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0011 | こっくりさん | 00 | 完了 | `003` | `1b24d197f9cb94241e26a368a841bc1071671d23` | `4254c58c1dc3fab6315f62e2846e31401c89f3fb` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0011 | こっくりさん | 10 | 完了 | `003` | `4254c58c1dc3fab6315f62e2846e31401c89f3fb` | `0b0f861577ceae5f546d0d3796387e27aa65dfe2` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0019 | 小さいおじさん | 00 | 完了 | `002` | `e7ada0703530358d46b387b4feb149fc9e1b8ad9` | `b3f91feaba652b5b3b8eeca202bf2bddb4449c34` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0019 | 小さいおじさん | 10 | 完了 | `002` | `b3f91feaba652b5b3b8eeca202bf2bddb4449c34` | `1f180c1d0fde5100b0cf71d6bc184ad29a562f0d` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0024 | 幸福の手紙 | 00 | 完了 | `002` | `fc120f41ab98bc1f150f624dcafaaa016e2c6d39` | `41e27cc1c43d79fa231f32b9e0ffdf80fb296811` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0024 | 幸福の手紙 | 10 | 完了 | `002` | `41e27cc1c43d79fa231f32b9e0ffdf80fb296811` | `362878efa061cd073f48b2cf09d9e006c2af6ba9` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0025 | 不幸の手紙 | 00 | 完了 | `002` | `aee0942ab1c70a42008c786670c0c0d5b6e04a1e` | `7ea7647a2b29cf93687c71109adce9e5637f3e7f` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0025 | 不幸の手紙 | 10 | 完了 | `002` | `7ea7647a2b29cf93687c71109adce9e5637f3e7f` | `443b8f6a87afd69f2a0276b8677b8f093b7266e2` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0059 | 深泥池の幽霊タクシー | 00 | 完了 | `003` | `4c91bfea47f31b35a5a38799ee145c93f29580f7` | `67db4d07c94341c71b219b7b6469a4208ea3d39f` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0059 | 深泥池の幽霊タクシー | 10 | 完了 | `003` | `67db4d07c94341c71b219b7b6469a4208ea3d39f` | `60ff1c2a50d75f95f74170a5a9085ee1fb2b5aa3` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0060 | タクシー幽霊 | 00 | 要修正 | `002` | `8c217bc57d73e0b35fa1790bb38c728c7cbabb28` | `61e7740b4268c3c7e8d66e91828e44de8cf291f4` | Review_002要修正。第3章をtutorial正規構造へ修正要。 |
| 0060 | タクシー幽霊 | 10 | 要修正 | `002` | `61e7740b4268c3c7e8d66e91828e44de8cf291f4` | `8c217bc57d73e0b35fa1790bb38c728c7cbabb28` | Review_002要修正。D04変容機構を保守化要。 |
| 0081 | ピアスの白い糸 | 00 | 要修正 | `002` | `8c217bc57d73e0b35fa1790bb38c728c7cbabb28` | `07548bfdd85214e36697ff4e567f73335b6236f7` | Review_002要修正。雪村2017の典拠階層と第3章構造を修正要。 |
| 0081 | ピアスの白い糸 | 10 | 要修正 | `002` | `07548bfdd85214e36697ff4e567f73335b6236f7` | `1e95b444dfe7c35c3b446fdc51353f3265a6c40e` | Review_002要修正。D04 Secondary、D16、D21を再判定要。 |
| 0089 | 日本だるま／だるま女 | 00 | 要修正 | `002` | `1e95b444dfe7c35c3b446fdc51353f3265a6c40e` | `a06b966743cc0208f8442e46633b3705fc292707` | Review_002要修正。Duke 2025の位置情報と第3章構造を補強要。 |
| 0089 | 日本だるま／だるま女 | 10 | 要修正 | `002` | `a06b966743cc0208f8442e46633b3705fc292707` | `faa70d145eeefda1965066ff29e395070c22ec01` | Review_002要修正。D10 Primary mismatch、D03 Scope媒体を再判定要。 |
| 0091 | ベッドの下の男 | 00 | 再レビュー待 | `001` | `84d381e004e33189e31ec590dbac53979e35a839` | `693ba24ff2f8e04bfc4b3957c8479a6df4607a92` | Review_001指摘対応。4Gamer 2025をContent補助に追加し、分析混入・共有核裁定・教訓推論を除去。 |
| 0091 | ベッドの下の男 | 10 | 再レビュー待 | `001` | `693ba24ff2f8e04bfc4b3957c8479a6df4607a92` | `af25ef37b491cfd16cea48956e4122f8ce4be21d` | Review_001指摘対応。tutorial形式復元、D01/D04等を保守化、D13潜伏・待ち伏せをtaxonomy gapとして正式化。 |
| 0101 | 海外旅行で臓器を抜かれる | 00 | 再レビュー待 | `001` | `af25ef37b491cfd16cea48956e4122f8ce4be21d` | `2aefa37f5d3343a15d3436823f9001db4a43ff40` | Review_001指摘対応。1991年新聞・Donovan 2002・Antonijević 2007をEvidence正本へ収録し、分析混入を除去。 |
| 0101 | 海外旅行で臓器を抜かれる | 10 | 再レビュー待 | `001` | `2aefa37f5d3343a15d3436823f9001db4a43ff40` | `6e4963fd1fa127d23ffc00e29608460fe80c862f` | Review_001指摘対応。複合Status解消、D01=U、D11=MEET_INTERACT、D17=医療介入、D18=I、D21=A1。 |
| 0112 | 事故物件は一度別人を住ませれば告知義務が消える | 00 | 再レビュー待 | `001` | `6e4963fd1fa127d23ffc00e29608460fe80c862f` | `146c6656f54ca7a97fec039ca06783375a5cbdc7` | Review_001指摘対応。summaryと0.3をEvidence記述へ戻し、意味形成・制度モデル分析を10へ分離。 |
| 0112 | 事故物件は一度別人を住ませれば告知義務が消える | 10 | 再レビュー待 | `001` | `146c6656f54ca7a97fec039ca06783375a5cbdc7` | `1aac7714b0bb128823fde7abbe6d11a6c854377e` | Review_001指摘対応。複合Status解消、D17を戦略利用へ限定、D18=I、tutorial形式復元。 |
| 0113 | 井の頭公園のボートに乗ると別れる | 00 | 要修正 | `001` | `b778af5f1b7bb38d10323dc563191f6d67fe1174` | `7acdb0e50cdfedfc984bd0e14cf45b739eeb90f7` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0113 | 井の頭公園のボートに乗ると別れる | 10 | 要修正 | `001` | `7acdb0e50cdfedfc984bd0e14cf45b739eeb90f7` | `5be227c1b03e2a1aeabcbb255822778c166a0fd0` | legacy移行: 旧R3→R4 checkpoint。D01=U、L3=0。 |
| 0118 | 函館山の切れない木 | 00 | 要修正 | `001` | `cfe370ce21791dcaaa815da1d636ff41572c11c6` | `c17adc9a37c3019e585346f3b8ebbdcdc0346f32` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0118 | 函館山の切れない木 | 10 | 要修正 | `001` | `c17adc9a37c3019e585346f3b8ebbdcdc0346f32` | `63c122601d2fe7bf975f047f5f8b7dd9b1fe2b3a` | legacy移行: 旧R3→R4 checkpoint。D10 taxonomy gap、L3=0。 |
| 0132 | 八幡の藪知らず | 00 | 要修正 | `001` | `04a86cc9c6fbf4405bd6123aa5425a0e0b0bdd97` | `39ca6f738f62972ac58b1a4f5ba875143a5cabf9` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0132 | 八幡の藪知らず | 10 | 要修正 | `001` | `39ca6f738f62972ac58b1a4f5ba875143a5cabf9` | `9d662fe4e36baa4cfda390b9a4744260467aaba4` | legacy移行: 旧R3→R4 checkpoint。D01=G0/I、L3=1限定。 |
| 0133 | 将門塚の祟り | 00 | 要修正 | `001` | `54f22c7712abcd2205231ed76a2fe1885a8ca911` | `058af26a784d134cd6c02e72fac278cb3447f037` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0133 | 将門塚の祟り | 10 | 要修正 | `001` | `058af26a784d134cd6c02e72fac278cb3447f037` | `ae74f13a5c03563f8c556987e17fae78343f3b28` | legacy移行: 旧R3→R4 checkpoint。D01=G0/I、L3=1。 |
| 0137 | 犬鳴村 | 00 | 要修正 | `001` | `b240224bcfdb7b215ef2ccf1bd45ad9e551cc8a9` | `5310aca86bb3719fa09703f6693804ad546ef62c` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0137 | 犬鳴村 | 10 | 要修正 | `001` | `5310aca86bb3719fa09703f6693804ad546ef62c` | `399880d335a6d9cc55b5d9004887e34dcc2a7c32` | legacy移行: 旧R3→R4 checkpoint。D01=U、L3=0。 |
| 0152 | 青木ヶ原樹海で方位磁針が狂う | 00 | 要修正 | `001` | `b52c65cc5f1a7910784c571d1d9d9abf759318ea` | `684c16cb9c01a5fee5babcb8e415ca775fd65d6b` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0152 | 青木ヶ原樹海で方位磁針が狂う | 10 | 要修正 | `001` | `684c16cb9c01a5fee5babcb8e415ca775fd65d6b` | `ed326e171be81b02f57026b429456142e48998c7` | legacy移行: 旧R3→R4 checkpoint。D17=U、大学検証L3=1。 |
| 0157 | 新郷村キリストの墓 | 00 | 要修正 | `001` | `4933870b3e3062c8ce16ee4eec7aa0ee76b0b693` | `68f901f86daa9a2f0e53d00a80ac791bc9b3fa3e` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0157 | 新郷村キリストの墓 | 10 | 要修正 | `001` | `68f901f86daa9a2f0e53d00a80ac791bc9b3fa3e` | `afcc03d239e668f141f10cd6c3d83b655358ace5` | legacy移行: 旧R3→R4 checkpoint。D01=G1/I、L3=1。 |
| 0158 | 虚舟 | 00 | 要修正 | `001` | `90f08e7968e188540022762a88926b7ea69586fc` | `aa7968f1cd5d12817409a534247050e5e8bf6cef` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0158 | 虚舟 | 10 | 要修正 | `001` | `aa7968f1cd5d12817409a534247050e5e8bf6cef` | `6f9e7370a626d64544e630302a06e93bcaa5c4e0` | legacy移行: 旧R3→R4 checkpoint。D01=G0/I、L3=0。 |
| 0169 | ノストラダムスの大予言 | 00 | 要修正 | `001` | `f0632eee87636fe2f584511c3544cd443d9290d0` | `598f41fc79c2871b347c4981d4d9984057b87ae0` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0169 | ノストラダムスの大予言 | 10 | 要修正 | `001` | `598f41fc79c2871b347c4981d4d9984057b87ae0` | `50f54b95bb478e739dd93aef58b248828433b285` | legacy移行: 旧R3→R4 checkpoint。D01=G3/D、D16=DEADLINE、L3=0。 |
| 0178 | 猿夢 | 00 | 要修正 | `001` | `1a85ab52869944c1f8c2ee3a6852fdba832884b6` | `e7f31217b47a73cf8c226e991b57cbe4c237e914` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0178 | 猿夢 | 10 | 要修正 | `001` | `e7f31217b47a73cf8c226e991b57cbe4c237e914` | `64429af4d65cfc5b941e6db878f0f6a81779ba01` | legacy移行: 旧R3→R4 checkpoint。2000年原型。D17覚醒離脱taxonomy gap、L3=0。 |
| 0179 | くねくね | 00 | 要修正 | `001` | `b6aac89cc94e864063d2e9cdff332bfa78a61153` | `c1e1c625fa13c2f1d164fc67703bcc7e09397888` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0179 | くねくね | 10 | 要修正 | `001` | `c1e1c625fa13c2f1d164fc67703bcc7e09397888` | `55e44c086c579b9b4a97d21df109addafb828def` | legacy移行: 旧R3→R4 checkpoint。2001原型→2003増補。D11=UNDERSTAND_RECOGNIZE、D13認知災害、L3=0。 |
| 0180 | きさらぎ駅 | 00 | 要修正 | `001` | `390a59182cffc7c6982c45af037e732c514c4fc0` | `24a7d6ec5a50972f27a2a39fff2a57045023aaf4` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0180 | きさらぎ駅 | 10 | 要修正 | `001` | `24a7d6ec5a50972f27a2a39fff2a57045023aaf4` | `0ef39d4cb61802c34fe4e271b3cdc1aa867feeb9` | legacy移行: 旧R3→R4 checkpoint。D09=非解決維持、D11=ACCIDENT_INVOLVEMENT、D17=NO_KNOWN_ESCAPE、L3=0。 |
| 0181 | コトリバコ | 00 | 要修正 | `001` | `15e4277b3be93a189cc168fa20f62cfb5a63f048` | `2453520af5a0aa8d99e175f896e4df8f6a160db5` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0181 | コトリバコ | 10 | 要修正 | `001` | `2453520af5a0aa8d99e175f896e4df8f6a160db5` | `afc96e6404a9fb7ce0e4731052d5da5598bc6c57` | legacy移行: 旧R3→R4 checkpoint。2005-06-06→06-11早期増補、D21=A4、L3=0。 |
| 0188 | 八尺様 | 00 | 要修正 | `001` | `4d762489c220d89837df9edf603d84780b2195f9` | `9d59272c2d141ef05a8e4f9ec80633ceb3fb7a21` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0188 | 八尺様 | 10 | 要修正 | `001` | `9d59272c2d141ef05a8e4f9ec80633ceb3fb7a21` | `94d7be41f4d6e2e61a7f1546e11e77eb773e1d4b` | legacy移行: 旧R3→R4 checkpoint。D15=FUTURE_CONSTRAINT、L3=0。 |
| 0198 | 一人かくれんぼ | 00 | 要修正 | `001` | `ed97ce65ee6856da024c91a0a14bbbc202353306` | `ebc2fc8400e97feaa85c5bfecb5be7b39ae821e3` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0198 | 一人かくれんぼ | 10 | 要修正 | `001` | `ebc2fc8400e97feaa85c5bfecb5be7b39ae821e3` | `37452dff345a613dc030eab5ce3aa296462efdb7` | legacy移行: 旧R3→R4 checkpoint。00 blob `2aecca1aa39a88e8edcf4c3e3aabe13aa74056d8`、10 blob `3d3ac5cf65adc327af87fe66f364fc6b7debae52`。2007年4月初期形成、D18=1/0/1。 |
| 0225 | 蛇口からポンジュース | 00 | 要修正 | `001` | `73568a3f4cbab96cb708e4b7ce5cae1cfb7300c0` | `3c3e87323619418878c3ad7a239bfc5658054d5f` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0225 | 蛇口からポンジュース | 10 | 要修正 | `001` | `3c3e87323619418878c3ad7a239bfc5658054d5f` | `36f1860f8490eeaec62d0f81f1ea1d73139c6037` | legacy移行: 旧R3→R4 checkpoint。昭和50年代頃/I、2007年企業再現、D18=1/0/1。 |
| 0250 | 七人ミサキ | 00 | 要修正 | `001` | `93f71046e0cdb61ed94b97ea525f57a9a4693110` | `b79c45fe6f256e134967725be6281b874a0dab3e` | 新運用: Evidence正本単独commit。Review_001要修正。 |
| 0250 | 七人ミサキ | 10 | 要修正 | `001` | `eb27237da162253685c49800d686e251590b3c39` | `29013e64175e96fd2d58dd30688b428487b7cbf4` | 新運用: Coding正本単独commit。R4 Final decisions反映。Review_001要修正。 |
| 0275 | 磐梯山の手長足長 | 00 | 要修正 | `001` | `c3afafad641a2aa2a12897b951f0e1027208540d` | `76b7c0755d8ffd2a981cbbc8d66375249b831625` | 新運用: Evidence正本単独commit。1992年直接採録＋2025年自治体文化施設X Evidence。Review_001要修正。 |
| 0275 | 磐梯山の手長足長 | 10 | 要修正 | `001` | `47340c42411e2c05b6fc1143f634172ba7ceca41` | `d7f2d5d7513d4b9cac9c3c4f5d04ea4816e44656` | 新運用: Coding正本単独commit。D01=G5、D18=1/0/1、D21=A4。Review_001要修正。 |
| 0309 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0309 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0319 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0319 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0349 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0349 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0356 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0356 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0362 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0362 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0363 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0363 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0365 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0365 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0366 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0366 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0384 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0384 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0385 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0385 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0394 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0394 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0403 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0403 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0410 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0410 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0411 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0411 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0412 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0412 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0413 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0413 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |

## 6.1. Review状態

- 完了9件は00/10双方の既存最終Review Passを維持する。Section 6の`最新レビュー版`は実在するReviewファイルの最大連番を記録した。
- `0060/0081/0089` はReview_002で修正指摘が返却済みで、00/10の6成果物を `要修正 / 002` とする。
- `0091/0101/0112` はReview_001指摘に対する修正を完了し、6成果物を `再レビュー待 / 001` とする。
- 残る `0113/0118/0132/0133/0137/0152/0157/0158/0169/0178/0179/0180/0181/0188/0198/0225/0250/0275` は `要修正 / 001`。
- Reviewで片方のみPassした場合は、その成果物行だけ`完了`へ更新する。もう片方のStatusは独立に管理する。
- Review完了時は、対象成果物行の`最新レビュー版`を実施済みReview番号へ更新する。修正後の`再レビュー待`では番号を先送りしない。

# 7. 実行順序

`0001 → 0003 → 0005 → 0006 → 0011 → 0019 → 0024 → 0025 → 0059 → 0060 → 0081 → 0089 → 0091 → 0101 → 0112 → 0113 → 0118 → 0132 → 0133 → 0137 → 0152 → 0157 → 0158 → 0169 → 0178 → 0179 → 0180 → 0181 → 0188 → 0198 → 0225 → 0250 → 0275 → 0309 → 0319 → 0349 → 0356 → 0362 → 0363 → 0365 → 0366 → 0384 → 0385 → 0394 → 0403 → 0410 → 0411 → 0412 → 0413`

次の新規Entryは`0309`。

# 8. Entry完了条件

以下をすべて満たしてEntryを`完了`とする。

- R1〜R4の工程要件を満たしている。
- `*_00_contents.md` の成果物行が`完了`。
- `*_10_analysis.md` の成果物行が`完了`。
- 00/10双方のReview Cycleで未解決指摘がない。
- 必要な修正commit / push とcontrol plane更新が完了している。

旧Excel同期はR5以降。

# 9. lifecycle変更履歴

- 2026-09-14: baseline固定。
- 2026-09-15: 先行9件Review Cycle完了。
- 2026-09-15: 0060〜0169を順次Coder R1〜R4完了・レビュー待へ移行。
- 2026-09-15: 0178 猿夢、0179 くねくね、0180 きさらぎ駅、0181 コトリバコ、0188 八尺様を順次レビュー待へ移行。
- 2026-09-15: 0198 一人かくれんぼをレビュー待へ移行。2006説を採らず2007年4月を直接定点とし、別参加者の実践行動を独立X EvidenceとしてD18 L3=1。
- 2026-09-15: 0225 蛇口からポンジュースをレビュー待へ移行。地域ジョークから2007年の企業による実物再現までをScope化し、D18 L3=1。
- 2026-09-15: 0250 七人ミサキをレビュー待へ移行。00/10を新運用の成果物単独commitで確定。
- 2026-09-15: 0275 磐梯山の手長足長をレビュー待へ移行。1992年直接採録を年代定点とし、2025年町立資料館の展示・参加型催事を独立X EvidenceとしてD18 L3=1。
- 2026-09-15: control planeをtask別進捗から正本成果物（00/10）単位の進捗管理へ変更。成果物ごとのcommit→push、pre/post-SHA、Review Statusを主キー化。
- 2026-09-15: `最新レビュー版` 列を追加。`－`=未レビュー、3桁連番=最後に完了したReview版として固定。
- 2026-09-15: 0060〜0275の24件について00/10のReview_001返却を反映し、48成果物を`要修正 / 001`へ同期。
- 2026-09-15: 0081・0089のReview_001指摘対応を完了し、00/10の4成果物を`再レビュー待 / 001`へ移行。
- 2026-09-15: 0091・0101・0112のReview_001指摘対応を完了し、00/10の6成果物を`再レビュー待 / 001`へ移行。
- 2026-09-15: 0060・0081・0089のReview_002返却を反映し、00/10の6成果物を`要修正 / 002`へ同期。

# 10. 最終完了条件

49件について00/10 Review Cycleを完了し、全Evidence正本・Coding正本を確定後、R5・R6・R7を完了する。