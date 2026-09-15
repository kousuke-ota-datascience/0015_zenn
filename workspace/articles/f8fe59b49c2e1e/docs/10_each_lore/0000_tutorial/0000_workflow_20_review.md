# 0. INTRODUCTION

本書は、`docs/10_each_lore/<Entry>/` 配下の

- `<Entry_ID>_00_contents.md`
- `<Entry_ID>_10_analysis.md`

をレビューし、その結果をReview mdとして保存するための**個別伝承Review標準作業手順**である。

Reviewerは対象成果物を修正しない。問題、根拠、分析上の影響、修正方向をReview mdへ記録し、Pass / 要修正を判定する。

本書はCoder側の修正手順、control planeの状態遷移、pre/post-SHA更新規則を定義しない。それらは `0000_workflow_10_each_lore_analysis.md` を正とする。

## 0.1. 実行時入力

Review実行時には、対象Entryに加え、必要に応じて次を外部から指定する。

```text
repository
対象 Entry_ID
使用する control plane
```

control planeの実体パスは実行時に外部から与える。本workflow内に特定のcontrol planeパスをハードコードしない。

control planeを使用しないReviewでは、レビュー対象commitを別途明示して固定する。

# 1. Review時に確認する正本

## 1.1. workflow / tutorial

- `docs/10_each_lore/0000_tutorial/0000_workflow_10_each_lore_analysis.md`
- `docs/10_each_lore/0000_tutorial/0000_00_contents.md`
- `docs/10_each_lore/0000_tutorial/0000_10_analysis.md`

## 1.2. 理論・コード・分析コード付与規則

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`
- 必要に応じ `docs/00_research_overview/80_appendix/20_what_is_sense_making.md`

Reviewでは、既存コード値、旧Excel、過去Review、作業メモを正しいものとして前提化しない。

# 2. Reviewの基本単位

## 2.1. 1 Entryずつ完結させる

複数Entryを依頼された場合でも、原則として次の単位で閉じる。

```text
Entry A の 00 をReview
→ Entry A の 10 をReview
→ Review_A_00 / Review_A_10 を保存
→ 保存内容を再確認
→ Entry A 完了
→ 次Entryへ進む
```

## 2.2. Review_Seq

`Entry_ID` は4桁ゼロ埋め、`Review_Seq` は3桁ゼロ埋めとする。

保存形式:

```text
docs/99_work/review_10_each_lore/<Entry_ID>/Review_<Entry_ID>_00_<Review_Seq>.md
docs/99_work/review_10_each_lore/<Entry_ID>/Review_<Entry_ID>_10_<Review_Seq>.md
```

同一Review cycleの00/10は同じSeqを使う。既存最大Seq+1を用い、既存Reviewを上書きしない。

# 3. Review対象版の固定

## 3.1. commit SHA / blob SHA

Review開始時に、00/10それぞれについて次を固定する。

```text
対象commit SHA
対象blob SHA
```

commit SHAは履歴上の対象時点、blob SHAは内容同一性を固定する。

## 3.2. control planeが指定されている場合

対象Entry × 成果物行から次を確認する。

```text
Status
最新レビュー版
pre-SHA
post-SHA
remarks
```

原則として、レビュー対象commit SHAはcontrol planeの `post-SHA` と一致させる。

次を照合する。

```text
control plane の post-SHA
→ そのcommit時点の対象ファイルblob
→ 実際にReviewする対象ファイルblob
```

一致する場合、`post-SHA` を対象commit SHAとしてReview mdへ記録する。

一致しない場合、Reviewerは最新mainを黙ってReviewしない。対象版を確定できるまでReviewを開始せず、不一致を作業記録へ残す。

control planeのcheckpoint修復、Status変更、pre/post-SHA更新はReviewerの責務ではない。必要な整合性回復は `0000_workflow_10_each_lore_analysis.md` に従ってCoder側で行う。

legacy checkpointで成果物単独commitではない場合は、そのcommit時点の対象blobとの同一性を確認する。

# 4. Reviewの順序

必ず次の順序で確認する。

```text
00:
1. tutorial構造
2. tutorial必須記載内容
3. tutorial記載粒度
4. Evidence / Content内容

10:
1. tutorial構造
2. tutorial必須記載内容
3. tutorial記載粒度
4. Version Scope
5. D01〜D21
6. sense-making / 因果構造
7. 00→10 traceability
```

構造が合っているだけではtutorial準拠としない。必須内容と、第三者が再構成できる記載粒度まで確認する。

# 5. `00_contents` Review

`00_contents` は**伝承内容 / Evidence整理の正本**としてReviewする。

最低限、次を確認する。

- tutorialの必須見出し・順序・フィールドを満たす。
- summaryから主要な伝承形・命題を再構成できる。
- 主体、場所、物、条件、規則、禁忌、帰結、不確実性が必要な粒度で記録されている。
- 最古確認と実際の起源を混同していない。
- 初期形と後代異伝を区別している。
- 典拠がURL列挙だけでなく、どの内容を何のEvidenceとして支えるか追跡できる。
- 本文未実見・資料不足・競合を隠していない。
- D01〜D21、Primary / Secondary、Version Scope裁定等をEvidence layerへ混入していない。
- 後代設定を初期Versionへ遡及していない。

Evidence roleや証拠強度の規則は `30_urban_legend_analysis_coding_rules.md` を正とする。

# 6. `10_analysis` Review

`10_analysis` は**分析コード付与の正本**としてReviewする。

## 6.1. tutorial準拠

最低限、次を確認する。

- 基本情報とVersion Scopeが存在する。
- D01〜D21がすべて存在する。
- 各Dimensionに必要なPrimary / Value、Parent、Secondary、Statusがある。
- 判定根拠が00の具体的Evidenceに接続する。
- 「この伝承における現れ方」がコード定義の言い換えではなくEntry固有の説明になっている。
- `U / NA / C` の理由が具体的に記載されている。

## 6.2. 分析コード妥当性

コードID、Parent / Child、Primary / Secondary、Status、tie-break、taxonomy gap等の詳細は `20 / 30` を正とする。

Reviewでは少なくとも次を確認する。

- 存在しないcode IDを使用していない。
- ParentがChildから正しく導出される。
- Primaryが中心質問に最も直接的である。
- Secondaryが独立した追加構造であり、単なる関連要素ではない。
- Evidence不足を最もありそうな値で埋めていない。
- taxonomy gapを `U` や近似Childで隠していない。

## 6.3. sense-making / 因果構造QA

特に次を確認する。

```text
D07: 何が説明対象か
D08: 何を手掛かりに問題化されるか
D09: どう理解可能にするか
D10: 原因を何として世界に置くか
D11: 因果系への入口
D12: 誰／何に作用するか
D13: 何をするか
D14: 帰結極性
D15: 最終的に何の領域が変化するか
D16: 因果系内部の時間構造
D17: 回避・制御・利用
```

D12→D13→D15は一本の因果列として読めることを要求する。

## 6.4. その他の重点QA

- D01: 最古確認資料を生成時期へ自動変換しない。
- D02: 起源媒体を推測しない。
- D03: Version Scope内で確認できる流通媒体だけを扱う。
- D04: 変容をHistory Evidenceなしに断定しない。
- D05: Evidence資料の形式と伝承の提示形式を混同しない。
- D06: 研究者の真偽評価を伝承自身の真実性提示へ移さない。
- D18 L3: Scope内の独立X Evidenceを要求する。
- D19: 公開可能性を全国流通と同一視しない。
- D20: 情報がないことを `NO_ONE_KNOWS` と同一視しない。
- D21: Evidence資料の実在性ではなくScoped Contentの現実アンカーを見る。

# 7. 00 / 10 横断QA

次を確認する。

1. Version Scopeが00 Evidenceから再構成できる。
2. 10の判定根拠が00の具体内容・典拠へ戻れる。
3. 00で未確認の事項を10が確定していない。
4. Scope外の異伝を分析コードへ混ぜていない。
5. D07〜D17が同じVersionの一つの意味形成モデルになっている。
6. D18 L3のX Evidenceが00に記録され、Scope内である。
7. tutorial構造・内容・粒度の判定とFindingsが整合する。

理想的なtraceabilityは次である。

```text
00の具体的Evidence
→ Version Scope
→ 分析者の解釈
→ D01〜D21
```

# 8. Finding重大度と最終判定

## 8.1. Major

- Version Scope、Entry境界、中核の意味形成モデルを変える可能性が高い。
- Primary / Statusへ直接影響する重大な誤り。
- Evidence / Analysis責務混同が大きい。
- taxonomy gapを誤コードで隠している。
- tutorial必須内容・粒度が広範に欠け、再現性を確保できない。

## 8.2. Moderate

- 一部Dimension・一部節のコード / Statusへ影響し得る。
- Evidence directness、Secondary、局所的tutorial不足等の再確認が必要。

## 8.3. Minor

- 見出し番号、表記、軽微なフォーマット等で、分析判断を変えない局所修正。

## 8.4. 最終判定

```text
Findingなし              → 問題なし（Pass）
Minorのみ                 → 要修正（Minor）
Moderateあり、Majorなし   → 要修正（Moderate）
Majorあり                 → 要修正（Major）
```

# 9. Review mdの共通記載事項

00/10どちらのReview mdにも最低限次を記録する。

```text
Entry_ID
Review_Seq
対象ファイル
対象commit SHA
対象blob SHA
control plane使用時は pre-SHA / post-SHA
レビュー日
最終判定
Findings
維持可能な点
QA
修正優先順位
```

各Findingは原則として次を含む。

```text
何が現在書かれているか
→ 何が規則・理論・tutorialと衝突するか
→ なぜ分析上 / 再現性上問題か
→ どの方向へ修正すべきか
```

Reviewerは修正後コードを勝手に正本へ書き込まず、必要なら「修正方向」または作業仮説として示す。

# 10. Reviewファイル標準フォーマット

## 10.1. 00

```markdown
# Review <Entry_ID>_00_<Review_Seq> — Entry <Entry_ID> <伝承名> / `<Entry_ID>_00_contents.md`

## 0. レビュー情報
- `Entry_ID`: `<Entry_ID>`
- `Review_Seq`: `<Review_Seq>`
- 対象commit SHA: `<...>`
- 対象blob SHA: `<...>`
- control plane pre-SHA: `<... / －>`
- control plane post-SHA: `<... / －>`
- レビュー日: `YYYY-MM-DD`
- 判定: **問題なし（Pass） / 要修正（Minor|Moderate|Major）**

## 1. 結論
## 2. Findings
## 3. 維持可能な点
## 4. QA
## 5. 修正優先順位
## 6. 最終判定
```

## 10.2. 10

```markdown
# Review <Entry_ID>_10_<Review_Seq> — Entry <Entry_ID> <伝承名> / `<Entry_ID>_10_analysis.md`

## 0. レビュー情報
- `Entry_ID`: `<Entry_ID>`
- `Review_Seq`: `<Review_Seq>`
- 対象commit SHA: `<...>`
- 対象blob SHA: `<...>`
- control plane pre-SHA: `<... / －>`
- control plane post-SHA: `<... / －>`
- レビュー日: `YYYY-MM-DD`
- 判定: **問題なし（Pass） / 要修正（Minor|Moderate|Major）**

## 1. 結論
## 2. Findings
## 3. 維持可能な点
## 4. Sense-making再構成案
## 5. QA
## 6. 修正優先順位
## 7. 最終判定
```

FindingがなくてもQAと最終判定を省略しない。

# 11. Review完了チェック

1 EntryのReviewは次をすべて満たした場合のみ完了とする。

- 対象00/10を取得した。
- 対象commit SHA / blob SHAを固定した。
- control plane使用時はpost-SHAと対象版を照合した。
- 00のtutorial構造・内容・粒度を確認した。
- 00をEvidence layerとしてReviewした。
- 10のtutorial構造・内容・粒度を確認した。
- Version Scopeを検証した。
- D01〜D21をReviewした。
- D07〜D17を一つの意味形成モデルとして確認した。
- 00→10 traceabilityを確認した。
- taxonomy gap / U / NA / Cを区別した。
- 00/10で同じReview Seqを使用した。
- Review mdへ対象commit SHA / blob SHAを記録した。
- 保存後にReview mdを再確認した。

Review完了後のCoder修正、control planeのStatus変更、pre/post-SHA更新は `0000_workflow_10_each_lore_analysis.md` に従う。
