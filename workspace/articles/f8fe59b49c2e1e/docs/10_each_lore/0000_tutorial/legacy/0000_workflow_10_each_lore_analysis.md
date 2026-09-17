# 0. INTRODUCTION

本書は、`docs/10_each_lore/` 配下で1伝承エントリの

- `<Entry_ID>_00_contents.md`
- `<Entry_ID>_10_analysis.md`

を作成・更新・検証し、Review指摘への修正までを実行するための**個別伝承分析の標準作業手順**である。

本workflowにおける個別伝承分析は、次の2工程からなる。

```text
伝承内容 / Evidence整理
→ 分析コード付与
```

本書は理論・コード体系・分析コード付与規則・Review判定基準を再定義しない。各責務の正本は以下とする。

- 理論設計: `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- コード体系: `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- 分析コード付与規則: `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`
- 文書管理・命名規則: `docs/00_research_overview/90_ducumentation_metadata.md`
- 母集団・`Entry_ID`・全件コード集約の正本: `docs/20_analysis_summary/urban_legend_parent_child_full_application_v1.xlsx`
- `00_contents` 記載様式: `docs/10_each_lore/0000_tutorial/0000_00_contents.md`
- `10_analysis` 記載様式: `docs/10_each_lore/0000_tutorial/0000_10_analysis.md`
- Review workflow: `docs/10_each_lore/0000_tutorial/0000_workflow_20_review.md`

本書と上記正本が競合する場合は、各責務の正本を優先し、本書側を修正する。

## 0.1. 実行時入力

実行プロンプトでは、必要に応じて少なくとも次を指定する。

```text
repository
対象 Entry_ID
使用する control plane
```

control planeを使用するプロジェクトでは、その**実体パスを実行プロンプトから与える**。本workflow内に特定のcontrol planeファイル名・ディレクトリをハードコードしてはならない。

本workflowはcontrol planeについて、実体ではなく**フォーマット・列意味・状態遷移・更新規則**のみを定義する。

control planeが指定されていない場合、control planeの存在を推測して探索・更新してはならない。

# 1. 標準ワークフロー

1伝承エントリは原則として次の順序で完結させる。

```text
Step 0. 対象Entryと分析単位を確定
→ Step 1. 方針ドキュメント・tutorial・必要ならcontrol planeを確認
→ Step 2. 典拠調査とEvidence整理
→ Step 3. *_00_contents.md を作成・更新
→ Step 4. Version Scopeと分析条件を固定
→ Step 5. D01〜D21の分析コードを付与し *_10_analysis.md を作成・更新
→ Step 6. 00 / 10 横断QA
→ Step 7. 成果物ごとにcommit / pushし、必要ならcontrol planeを更新
→ Step 8. Reviewまたは再Reviewへ渡す
→ Step 9. Review返却後は本書のReview修正cycleに従う
→ Step 10. 必要な時点で全件正本Excelへ同期
```

**Evidence収集と分析コード付与を混ぜない。** まず主要Evidenceをまとめて確保し、`00_contents` に整理してから分析する。

既存Excelコード、過去の `10_analysis`、過去Review、作業メモは、独立判定完了前の「正解」として使用しない。

## 1.1. Step と成果物

現行Stepと、各Stepで作成・更新する正本成果物およびaudit / checkpoint artifactの対応は次のとおりとする。
この表だけで、各Stepの出力先と成果物の位置づけを判断できることを目的とする。

| Step | 作業内容 | 正本成果物 | audit / checkpoint artifact |
|---|---|---|---|
| Step 0 | 対象Entry・分析単位の確定 | － | control planeを使用する場合は対象・baselineを確認。新規artifactは原則作らない |
| Step 1 | 正本・tutorial・control plane確認 | － | － |
| Step 2 | 典拠調査・Evidence整理 | － | 必要に応じ `<Entry_ID>_02_evidence_audit.md` |
| Step 3 | `00_contents` 作成・更新 | `<Entry_ID>_00_contents.md` | Step 2のEvidence auditを使用する場合は同artifactへ記録 |
| Step 4 | Version Scope固定 | － | `<Entry_ID>_04_version_scope.md` |
| Step 5 | 独立判定を固定し、`10_analysis` を作成・更新 | `<Entry_ID>_10_analysis.md` | `<Entry_ID>_05_independent_recode.md` |
| Step 6 | 00 / 10 横断QA | 必要に応じ `00` / `10` を修正 | `<Entry_ID>_06_entry_qa.md` |
| Step 7 | 成果物commit / push・control plane更新 | commit済みの `00` / `10` | control planeのStatus / pre-SHA / post-SHA更新。artifact fileではない |
| Step 8 | Review / 再Reviewへ引渡し | － | Reviewer側成果物 `docs/99_work/review_10_each_lore/<Entry_ID>/Review_*.md`。本workflowのanalysis artifactではない |
| Step 9 | Review返却後のCoder判定・修正cycle | 修正対象の `00` / `10` | 必要に応じ `<Entry_ID>_09_review_<Review_Seq>_adjudication.md` |
| Step 10 | 全件正本Excelへの同期 | `docs/20_analysis_summary/urban_legend_parent_child_full_application_v1.xlsx` | sync / diff確認記録はプロジェクト側の運用に従う |

複数Entryを横断する照合・全体QAが必要な場合、それらは1 Entryを処理する本workflowのStepではなく、プロジェクト横断工程として、実行時に指定されたcontrol planeまたは別の横断workflowで定義・管理する。

## 1.2. audit / checkpoint artifact 出力規則

本workflowで作成するEvidence audit、Version Scope、Independent Recode、Entry QA、Coder adjudicationは**正本成果物ではなく、分析側のaudit / checkpoint artifact**である。

analysis artifact rootは次の優先順位で決定する。

1. 実行プロンプトでartifact rootが明示されている場合は、そのパスを使用する。
2. artifact rootが明示されておらず、control planeの実体パスが明示されている場合は、**control planeと同じディレクトリの `artifacts/`** を使用する。
3. control planeもartifact rootも指定されていない場合、artifact保存先を推測・探索してはならない。永続checkpointが必要なプロジェクトでは、実行側がartifact rootを明示する。

したがって、control planeが

```text
docs/99_work/<project>/_control_plane.md
```

である場合、既定のanalysis artifact rootは

```text
docs/99_work/<project>/artifacts/
```

となる。

現行workflowで新規作成するartifactの標準ファイル名は次のとおりとする。

```text
<Entry_ID>_02_evidence_audit.md
<Entry_ID>_04_version_scope.md
<Entry_ID>_05_independent_recode.md
<Entry_ID>_06_entry_qa.md
<Entry_ID>_09_review_<Review_Seq>_adjudication.md
```

`Entry_ID` 直後の2桁は、**そのartifactを作成・更新する本workflowのStep番号**を表す。1桁のStepもゼロ埋め2桁で記載し、同一Entryのartifactをファイル名順に並べたとき、workflow上の実行順を判別できるようにする。

本workflowのStep番号が将来変更された場合でも、作成済みartifactを新番号へ遡及renameしない。作成時点のファイル名は監査証跡の一部として保持する。

旧命名規則または旧運用で作成済みのanalysis artifactは、**過去運用artifact（legacy artifact）**として扱う。これらはファイル名・本文を変更せず、analysis artifact root配下の次のディレクトリへ格納する。

```text
<analysis artifact root>/legacy/
```

`legacy/` は過去運用artifactの保管専用とし、現行workflowで新規作成するartifactを保存してはならない。過去運用artifactのファイル名や内部識別子は、現行workflowの命名規則を定義するものではない。

`Review_*.md` はReview workflowの成果物であり、analysis artifact rootへ移動しない。Review workflowが定める保存先を使用する。

artifactは正本 `00` / `10` の代替ではない。artifactと正本が競合する場合は、現行の正本・workflow・各責務の規範文書を優先し、artifactは監査証跡として扱う。

# 2. Step 0 — 対象Entryと分析単位を確定

## 2.1. Entry_ID

作業開始前に全件正本Excelで対象伝承と `Entry_ID` を確認する。

```text
<4桁ゼロ埋めEntry_ID>_<伝承識別名>/
├── <4桁ゼロ埋めEntry_ID>_00_contents.md
└── <4桁ゼロ埋めEntry_ID>_10_analysis.md
```

`0000` はtutorial専用であり、実伝承へ使用しない。

名称未確認Entryでは、対象行の候補メタデータと出典URLから名称を同定する。D01〜D21等の既存分析値を使って名称・内容を逆算しない。

## 2.2. Entry境界

分析単位は、**独立して伝達可能な最小の意味形成モデル**である。

人物名・地名・被害人数・媒体・表現細部の差だけでは通常分割しない。以下が安定して変わり、世界のルール自体が変わる場合は別Entryを検討する。

- 意味形成対象
- 因果源
- 発動・接触条件
- 作用対象
- 作用機構
- 主帰結
- 回避・制御規則

# 3. Step 1 — 正本とcontrol planeの確認

作業開始時に `10 / 20 / 30`、`0000_00_contents.md`、`0000_10_analysis.md` を確認する。

control planeが実行時に指定されている場合は、対象Entry × 成果物（00 / 10）の行を確認する。

確認対象は原則として次である。

```text
Status
最新レビュー版
pre-SHA
post-SHA
remarks
```

control planeは**現在状態の正本**であり、Evidence内容、Version Scope、D01〜D21の定義・妥当性をcontrol planeから推定しない。

# 4. Step 2 — 伝承内容 / Evidence整理

## 4.1. Evidenceを先にまとめる

可能な範囲で以下を一括して調査する。

- 伝承内容
- 最古確認資料
- 流通媒体・流通範囲
- 異伝・派生・変容史
- 現実社会への外部効果
- 実在地・史実・制度・自然科学的背景等の現実アンカー

Evidence role、証拠強度、include / exclude等の詳細は `30_urban_legend_analysis_coding_rules.md` を正とする。

## 4.2. 調査終了条件

資料件数ではなく、Entry境界・Version Scope・D01〜D21の判定に必要なEvidenceが揃ったかで止める。

低品質資料を追加しても主要判断が変わらない場合、`U` を減らすことだけを目的に探索を延長しない。

## 4.3. Evidence audit checkpoint

persistent checkpointを使用するプロジェクトでは、Evidenceの欠落・重複・出典精度・リンク有効性・include / exclude上の論点を、analysis artifact rootの `<Entry_ID>_02_evidence_audit.md` に記録できる。

Evidence audit artifactはEvidence上の問題を列挙するだけの代替物ではない。正本へ反映すべき修正はStep 3で `00_contents` へ反映する。

# 5. Step 3 — `*_00_contents.md`

`0000_00_contents.md` の章構造・項目名・順序・要求粒度に従う。

`00_contents` は **伝承内容 / Evidence整理の正本**であり、分析コードを決める場所ではない。

最低限、第三者が次を再構成できる密度を確保する。

- 主要内容または中心命題
- 条件・規則・禁忌
- 主体・場所・物・媒体
- 帰結・終端
- 未解決点・不確実性
- 最古確認と流通・変容史
- 主要異伝
- 内容と典拠の対応

後代設定を初期形へ遡及せず、Evidenceがない内容を補完しない。

# 6. Step 4 — Version Scope

`00_contents` 完成後、`10_analysis` 作成前にVersion Scopeを固定する。

Scope選択の詳細な分析規則は `10 / 20 / 30` を正とし、本workflowで重複定義しない。

原則として、成立時または最古に確認できるcoherentな共有核を優先し、世界のルールを変える安定異伝・後代翻案を無理に統合しない。

persistent checkpointを使用するプロジェクトでは、固定したScopeと主要なinclude / exclude判断をanalysis artifact rootの `<Entry_ID>_04_version_scope.md` に記録する。

# 7. Step 5 — 分析コード付与 / `*_10_analysis.md`

## 7.1. Independent Recode checkpoint

分析コード付与は、`00_contents` と現行の規範文書から**独立にD01〜D21を判定する工程**から開始する。

独立判定を固定する前に、既存Excelコード、過去の `10_analysis`、過去Review、作業メモを正解として参照してはならない。既存正本 `10_analysis` を更新する場合も、まず独立判定を固定し、その後に既存値との差分を照合する。

persistent checkpointを使用するプロジェクトでは、この独立判定をanalysis artifact rootの `<Entry_ID>_05_independent_recode.md` にfreezeする。freeze後に既存分析値との差分を確認し、正本 `10_analysis` へ反映する。

## 7.2. `10_analysis` の作成・更新

`0000_10_analysis.md` の構造・項目名・順序に従う。

`10_analysis` は **分析コード付与の正本**であり、コード表のMarkdown転記ではない。

分析時は `30_urban_legend_analysis_coding_rules.md` に従い、Evidenceから独立にD01〜D21を判定する。

完成文書では、各次元について少なくとも次が追跡できること。

```text
00上の具体的Evidence
→ 分析者の解釈
→ Primary / Secondary / Value
→ Status
```

各次元の詳細な選択規則、Status、tie-break、taxonomy gap等は `20 / 30` を正とし、本workflowでは再定義しない。

# 8. Step 6 — 00 / 10 横断QA

少なくとも次を確認する。

- Entry_IDとファイル名が正しい。
- `00_contents` と `10_analysis` の責務が分離されている。
- Version Scopeが00 Evidenceから再構成できる。
- D01〜D21がすべて存在し、Statusがある。
- 10の判定根拠を00の具体的Evidenceへ戻せる。
- Scope外の異伝を分析コードへ混ぜていない。
- `U / NA / C` を無理に埋めていない。
- Parent / Child / Primary / Secondaryが現行体系と整合する。
- D18 L3等、独立Evidenceを要求する判定が規則を満たす。

Review基準の詳細は `0000_workflow_20_review.md` を参照する。

persistent checkpointを使用するプロジェクトでは、横断QAの結果と修正要否をanalysis artifact rootの `<Entry_ID>_06_entry_qa.md` に記録する。QAで正本の修正が必要と判明した場合は、artifactへの記録だけで完了とせず、該当する `00` / `10` を修正する。

# 9. Step 7 — commit / push単位

## 9.1. 原則

**正本成果物1ファイルを1変更単位**としてcommit / pushする。

```text
成果物編集
→ 成果物のみcommit / push
→ post-SHA取得
→ control planeがある場合は該当行を更新
→ control planeを別commit / push
```

- `00_contents` と `10_analysis` を同一commitへまとめない。
- Review修正で00/10双方を変更する場合も成果物ごとにcommitを分離する。
- 複数Entryを同一成果物commitへまとめない。
- control plane自身の更新は成果物commitとは別commitにする。
- テストファイル・一時ファイルをmainへ作らない。

analysis artifactは正本成果物ではないため、正本成果物commitへ混在させない。artifactを保存・更新する場合は、正本成果物commitとは分離して監査可能にする。

## 9.2. pre-SHA / post-SHA

`pre-SHA` / `post-SHA` はGit commit SHAを指す。

- `pre-SHA`: **今回のCoder編集の入力となる当該成果物版を固定したcommit checkpoint**。
- `post-SHA`: **今回のCoder編集を確定した当該成果物のcommit**。

`pre-SHA` は作業開始時点のrepository HEADを意味しない。

Review修正cycleでは、直前Reviewの対象となった旧 `post-SHA` を次cycleの `pre-SHA` とする。

# 10. control plane interface specification

## 10.1. 外部指定

control planeの実体パスは実行プロンプトから与える。本書から特定パスを参照しない。

control planeが指定された場合、その文書は少なくともEntry × 正本成果物単位の現在状態を保持する。

## 10.2. 必須表フォーマット

```markdown
| Entry_ID | 伝承 | 成果物 | Status | 最新レビュー版 | pre-SHA | post-SHA | remarks |
|---|---|---|---|---|---|---|---|
```

必須列の意味は次のとおり。

| 列 | 意味 |
|---|---|
| `Entry_ID` | 4桁ゼロ埋めEntry ID |
| `伝承` | 対象伝承名。未確認ならその状態を明示 |
| `成果物` | `00` または `10` |
| `Status` | 成果物単位の現在状態 |
| `最新レビュー版` | 最後に完了したReview Seq。未レビューは `－` |
| `pre-SHA` | 現在cycleの入力成果物checkpoint |
| `post-SHA` | 現在cycleで確定した成果物commit |
| `remarks` | 旧運用からの移行、例外、主要修正等の監査注記 |

00と10は独立した成果物行として管理し、片方のSHA・Statusを他方へ流用しない。

## 10.3. Status

許容Statusは次とする。

| Status | 定義 |
|---|---|
| `未` | Coder作業未着手 |
| `レビュー待` | 初回Coder作業完了、初回Review待ち |
| `要修正` | Reviewで修正指摘あり、Coder再作業未着手 |
| `再作業中` | Coder修正中 |
| `再レビュー待` | Coder修正完了、再Review待ち |
| `完了` | Reviewer承認済み |
| `－（対象外）` | 適用対象外 |

Entry全体を`完了`とみなせるのは、00行・10行がともに`完了`の場合だけである。

## 10.4. 最新レビュー版

`最新レビュー版` は**最後に完了したReview Seq**であり、次に作るReview番号ではない。

例:

```text
要修正 / 001      = Review_001で修正要求
再作業中 / 001    = Review_001指摘への対応中
再レビュー待 / 001 = Review_001指摘対応完了、Review_002待ち
完了 / 002        = Review_002でPass
```

## 10.5. 状態更新規則

| イベント | Status | 最新レビュー版 | pre-SHA | post-SHA |
|---|---|---|---|---|
| 初回Coder成果物commit完了 | `レビュー待` | `－` | 編集入力checkpoint | 新成果物commit |
| Reviewで修正要求確定 | `要修正` | 実施済みReview Seq | 変更しない | 変更しない |
| Review Pass確定 | `完了` | 実施済みReview Seq | 変更しない | 変更しない |
| 修正開始 | `再作業中` | 変更しない | `old post-SHA` | commit確定まで`old post-SHA`保持 |
| 修正成果物commit完了 | `再レビュー待` | 変更しない | 修正開始時の値を保持 | 新成果物commit |
| 再Reviewで修正要求 | `要修正` | 新Review Seq | 変更しない | 変更しない |
| 再Review Pass | `完了` | 新Review Seq | 変更しない | 変更しない |

Review返却だけでは正本成果物は変わらないため、Reviewファイルを保存したcommitを `post-SHA` に記録してはならない。

control planeに進捗集計がある場合、それはEntry表から導出し、同じcontrol plane更新commit内で同期する。

# 11. Review返却後の修正開始ゲート

## 11.1. review-SHA

本書で `review-SHA` は、最新Review mdに記録された**対象commit SHA**、すなわちReviewerが実際にレビューした成果物commitを指す。

Review md自体を保存したcommitは `review-file-commit-SHA` と呼び、成果物版照合には使用しない。

## 11.2. 必須照合

Review指摘への修正を開始する前に、次を確認する。

1. 最新mainを確認する。
2. control planeの対象成果物行から `Status / 最新レビュー版 / pre-SHA / post-SHA / remarks` を取得する。
3. 最新レビュー版に対応するReview mdから `対象commit SHA` を取得し、`review-SHA` とする。
4. `post-SHA` と `review-SHA` をcommit graph上で照合する。
5. Review指摘を根拠に修正を開始できるのは、原則として **`post-SHA == review-SHA` の場合だけ**とする。

SHA文字列そのものに大小関係はない。新旧関係はcommit graphで判定する。

| 判定 | graph上の関係 | 修正開始 | 対処 |
|---|---|---|---|
| A | `post-SHA == review-SHA` | 可 | Review結果を採用して修正へ進む |
| B | `post-SHA` が `review-SHA` の祖先 | 不可 | control plane更新漏れ等を調査し、checkpointを修復 |
| C | `review-SHA` が `post-SHA` の祖先 | 不可 | Reviewが古い。現行post-SHAのReview有無を確認 |
| D | 相互に祖先関係なし | 不可 | branch/divergenceを調査し正本系列を確定 |

概念的な判定例:

```bash
if [ "$POST_SHA" = "$REVIEW_SHA" ]; then
  echo "A: exact match"
elif git merge-base --is-ancestor "$POST_SHA" "$REVIEW_SHA"; then
  echo "B: control plane post-SHA is older"
elif git merge-base --is-ancestor "$REVIEW_SHA" "$POST_SHA"; then
  echo "C: review target is older"
else
  echo "D: diverged"
fi
```

B/C/DではReview内容を現在の成果物へ機械的に適用しない。

旧運用由来など、成果物単独commitではないSHAを使う場合は、Review workflowに従い対象blob SHAも照合する。通常運用へ戻す時点でcheckpointを正規化する。

# 12. Review修正cycle

SHA整合性を確認した後、Review指摘への対応は次の順序で実施する。

```text
Review返却
→ post-SHA == review-SHA を確認
→ Review結果を採用
→ control planeがあれば Status=再作業中、pre-SHA=old post-SHA
→ 正本成果物を修正
→ 成果物のみcommit / push
→ 新しいpost-SHA取得
→ control planeがあれば Status=再レビュー待、post-SHA更新
→ 再Review
→ Reviewerは新しいpost-SHAを対象commit SHAとしてReview mdへ記録
→ Review返却
→ Passなら完了、修正要求なら同cycleを反復
```

不変条件:

```text
Cycle n:
pre_n
  ↓ Coder編集
post_n
  ↓ Review
review-SHA_n = post_n
  ↓ 修正要求
pre_(n+1) = post_n = review-SHA_n
  ↓ Coder編集
post_(n+1)
```

Review結果の判定そのものは `0000_workflow_20_review.md` の責務である。本workflowは、返却されたReviewと現在成果物の対応確認、およびCoder側修正・状態更新を担う。

persistent checkpointを使用するプロジェクトでは、Review Findingごとの採否、修正方針、修正対象成果物をCoder側のanalysis artifactとして `<Entry_ID>_09_review_<Review_Seq>_adjudication.md` に記録できる。これはReviewerの `Review_*.md` を置換・改変するものではない。

# 13. Reviewとの責務境界

Coder側は次を行う。

- Review対象版と現在成果物版の整合確認。
- Review指摘の採否判断と修正。
- 成果物commit / push。
- control planeがある場合のStatus / pre-SHA / post-SHA更新。

Reviewer側は `0000_workflow_20_review.md` に従い、次を行う。

- Review対象commit / blobの固定。
- 00/10のレビュー。
- Review Seqの採番。
- Pass / 要修正判定。
- Review mdへの対象commit SHA / blob SHAの記録。

ReviewerはCoder成果物を修正せず、CoderはReview判定をReview文書へ遡及改変しない。

# 14. 全件正本Excelへの同期

個別Entryの分析結果が確定した後、D01〜D21・Status等を全件正本Excelへ同期する。

この同期は**分析結果の出力反映**であり、分析コード付与前の入力工程ではない。

プロジェクト側で同期時点を後段フェーズへまとめる場合は、その運用を実行プロンプトまたはcontrol plane側の現在状態として管理し、本workflowへ特定プロジェクトの同期タイミングをハードコードしない。

# 15. 完了条件

1 Entryの個別伝承分析は、少なくとも次を満たすまで完了としない。

- Entry境界が妥当でEntry_IDが正しい。
- `*_00_contents.md` と `*_10_analysis.md` が存在する。
- 伝承内容 / Evidence整理と分析コード付与の責務が分離されている。
- Version ScopeがEvidenceに基づき固定されている。
- D01〜D21が現行正本に従って判定されている。
- 00→10のtraceabilityが成立する。
- 成果物ごとのcommit差分を確認した。
- control planeがある場合、その状態が成果物commitと整合している。
- Review対象プロジェクトでは00/10双方がReviewer承認済みである。
- 必要な時点で全件正本Excelへ同期済み、またはプロジェクト上の同期待ち状態が明示されている。

# 16. Workflow変更時の回帰確認

本workflowまたは分析コード付与規則の変更が、Entry作成・Version Scope・分析コード判定へ影響し得る場合は、既存のgolden regression / blind validation方針に従い、意図しない判定変化がないことを確認する。

既存golden referenceを用いる場合も、golden一致をblind / inter-coder reproducibilityの証明として扱わない。

回帰試験の目的は、workflow変更が既知の期待結果へ意図しない影響を与えていないかを検出することであり、分析コード体系そのものを本workflow内で再定義することではない。