# 1. Quick Reference

本章は、個別伝承Reviewを**決められた順序どおりに実行するための実行手順**である。

Reviewerは本章の順序を変更・省略・統合してはならない。各工程の確認を完了してから次工程へ進む。内容上もっともらしい、意味的に同等、過去ReviewでPassしている等の理由で、先行工程の確認要件を緩和してはならない。

詳細な判断基準・背景・Review mdフォーマットは第2章を参照する。第2章は本章の実行順序を変更しない。

## 1.1. Reviewerの基本原則

本書は、`docs/10_each_lore/<Entry>/` 配下の次の正本成果物をレビューし、その結果をReview mdとして保存するための**個別伝承Review標準作業手順**である。

- `<Entry_ID>_00_contents.md`
- `<Entry_ID>_10_analysis.md`

必ず次を守る。

- Reviewerは対象の正本成果物を修正しない。
- Reviewerはcontrol planeを更新しない。
- 既存コード値、旧Excel、過去Review、作業メモを正しいものとして前提化しない。
- 複数Entryでも、**1 Entryずつ 00 → 10 → 保存 → 再確認まで完結**させてから次Entryへ進む。
- 各Review工程は、前工程の確認完了後にのみ開始する。
- tutorial構造・必須記載内容・記載粒度は、後続の内容評価とは独立して判定する。
- 「同じ意味の情報がある」ことを、tutorial所定の見出し・順序・フィールド・ブロック形式を満たすことの代替として扱わない。
- format上の不適合を発見しても、Findingを確定したうえで残りのQAを継続し、同一Review cycleで他の問題も取り切る。
- 対象版を固定できない場合だけはReviewを開始せず、停止理由を記録する。
- Findingが1件でもあればPassにしない。**Passは全QAを完了し、Findingが0件の場合のみ**とする。

本書はCoder側の修正手順、control planeの状態遷移、pre/post-SHA更新規則を定義しない。それらは `0000_workflow_10_each_lore_analysis.md` を正とする。

## 1.2. 実行前に固定するもの

Review実行時には、対象Entryに加え、必要に応じて次を外部から指定する。

```text
repository
対象 Entry_ID
使用する control plane
```

control planeの実体パスは実行時に外部から与える。本workflow内から別のcontrol planeを推測・探索しない。

各Entryについて、Review開始前に次を確定する。

```text
00 対象commit SHA
00 対象blob SHA
10 対象commit SHA
10 対象blob SHA
Review_Seq
```

control planeを使用する場合は、対象Entry × 成果物行の `post-SHA` と、そのcommit時点の対象ファイルblobを照合する。対象版を確定できない場合はReviewを開始しない。

`Review_Seq` は既存最大Seq+1とし、同一Review cycleの00/10で同じSeqを使う。既存Reviewは上書きしない。

## 1.3. Entryごとの必須実行順序

各Entryを、**必ず次の順序**で処理する。

```text
1. 対象版固定
   - control plane確認
   - 00/10 commit SHA固定
   - 00/10 blob SHA固定
   - Review_Seq固定

2. 00 tutorial構造
   - 必須見出し
   - 見出し順序
   - 必須フィールド
   - 所定の表・ブロック構造

3. 00 tutorial必須記載内容

4. 00 tutorial記載粒度
   - 1.1 summaryだけを入力としてBlind Decodeを先に固定
   - その後、原Evidence / 1.2以降からReference Storyを確認
   - Blind DecodeとReference Storyへ同一Structural Probeを適用
   - 分析上の回答が保存されるかAnalysis Invarianceを確認
   - Probe Matrix / Reconstruction VerdictをReview_00へEvidenceとして保存

5. 00 Evidence / Content Review

6. 10 tutorial構造
   - 基本情報
   - D01〜D21
   - 各Dimensionの所定フィールド・ブロック
   - 見出し・順序

7. 10 tutorial必須記載内容

8. 10 tutorial記載粒度

9. Version Scope Review

10. D01〜D21 Review

11. sense-making / 因果構造QA

12. 00→10 traceability QA

13. Finding重大度・最終判定

14. Review_00 / Review_10 保存

15. 保存したReview mdを再取得して確認

16. Entry完了
```

**工程2〜12を並べ替えない。工程2〜4を完了する前に00のEvidence / Content判定へ進まず、工程6〜8を完了する前にVersion ScopeやD01〜D21の判定へ進まない。**

工程4では、Blind Decodeを固定した後に限り、記載粒度比較のためReference Storyを原Evidence / 1.2以降から確認してよい。この確認はsummaryの情報保存性を検証するための比較材料の取得であり、工程5のEvidence品質・directness・資料分類の妥当性判定を先取りするものではない。

ある工程でFAILを確認した場合は、そのFindingを記録したうえで次工程へ進み、後続QAを省略しない。ただし対象版固定に失敗した場合はReview自体を開始しない。

## 1.4. Review時の最小チェック

00では最低限、次を確認する。

- tutorialの必須見出し・順序・フィールドを満たす。
- summaryだけからBlind Decodeした伝承と、原Evidenceから確認できるReference Storyが、主要な構造・分析判断を実質的に保存している。
- Blind DecodeはReference Story確認前に固定し、比較後に書き換えない。
- Structural Probe MatrixとAnalysis Invarianceの差分がReview_00にEvidenceとして残っている。
- 主体、場所、物、条件、規則、禁忌、帰結、不確実性が必要な粒度で記録されている。
- 最古確認と実際の起源を混同していない。
- 初期形と後代異伝を区別している。
- 各内容がどの資料の何をEvidenceとしているか追跡できる。
- 本文未実見・資料不足・競合を隠していない。
- Analysis layerをEvidence layerへ混入していない。
- 後代設定を初期Versionへ遡及していない。

10では最低限、次を確認する。

- 基本情報とVersion Scopeが存在する。
- D01〜D21がすべて存在する。
- 各DimensionのPrimary / Value、Parent、Secondary、Statusが所定形式で存在する。
- 判定根拠が00の具体的Evidenceに接続する。
- 「この伝承における現れ方」がEntry固有である。
- `U / NA / C` の理由が具体的である。
- code ID、Parent / Child、Primary / Secondary、Statusが規則に合う。
- Evidence不足を推測で埋めていない。
- taxonomy gapを `U` や近似Childで隠していない。
- D07〜D17が一つの意味形成・因果モデルとして読める。
- D12→D13→D15が一本の因果列として読める。
- D18 L3にScope内の独立X Evidenceがある。
- D19で公開可能性を全国流通と同一視していない。
- 00→10の根拠追跡が可能である。

## 1.5. 判定・保存・完了条件

最終判定はFindingの最大重大度で決める。

```text
Findingなし              → 問題なし（Pass）
Minorのみ                 → 要修正（Minor）
Moderateあり、Majorなし   → 要修正（Moderate）
Majorあり                 → 要修正（Major）
```

保存形式は次とする。

```text
docs/99_work/review_10_each_lore/<Entry_ID>/Review_<Entry_ID>_00_<Review_Seq>.md
docs/99_work/review_10_each_lore/<Entry_ID>/Review_<Entry_ID>_10_<Review_Seq>.md
```

Entry完了前に、保存したReview mdを再取得し、最低限次を確認する。

- 00/10が同じReview Seqで保存されている。
- 対象commit SHA / blob SHAが正しい。
- control plane使用時のpre-SHA / post-SHAが記録されている。
- 最終判定がFindingsと整合する。
- 00 ReviewにSummary Reconstruction Evidenceが保存されている。
- Findings、QA、修正優先順位、最終判定が欠落していない。

ここまで完了して初めて次Entryへ進む。

# 2. Detailed Guidance

本章は、第1章の各工程をどのように判断するかを定義する。**実行順序は第1章を正とし、本章の記載を理由に工程を並べ替えない。**

## 2.1. Review時に確認する正本と責務

Reviewでは次を正本として参照する。

workflow / tutorial:

- `docs/10_each_lore/0000_tutorial/0000_workflow_10_each_lore_analysis.md`
- `docs/10_each_lore/0000_tutorial/0000_00_contents.md`
- `docs/10_each_lore/0000_tutorial/0000_10_analysis.md`

理論・コード・分析コード付与規則:

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`
- 必要に応じ `docs/00_research_overview/80_appendix/20_what_is_sense_making.md`

文書構造・見出し番号・命名規則は次を正とする。

- `docs/00_research_overview/90_ducumentation_metadata.md`

Reviewでは、既存コード値、旧Excel、過去Review、作業メモを正しいものとして前提化しない。過去ReviewはReview_Seqの決定や履歴確認には使用できるが、現行成果物の適否判定を省略する根拠にはしない。

Reviewerの責務は、対象成果物を修正することではなく、問題、根拠、分析上の影響、修正方向をReview mdへ記録し、Pass / 要修正を判定することである。

Coder側の修正手順、control planeの状態遷移、pre/post-SHA更新規則は `0000_workflow_10_each_lore_analysis.md` を正とする。Reviewerはcontrol planeのcheckpoint修復、Status変更、pre/post-SHA更新を行わない。

## 2.2. Reviewの基本単位とReview_Seq

複数Entryを依頼された場合でも、原則として次の単位で閉じる。

```text
Entry A の 00 をReview
→ Entry A の 10 をReview
→ Review_A_00 / Review_A_10 を保存
→ 保存内容を再確認
→ Entry A 完了
→ 次Entryへ進む
```

`Entry_ID` は4桁ゼロ埋め、`Review_Seq` は3桁ゼロ埋めとする。

同一Review cycleの00/10は同じSeqを使う。既存最大Seq+1を用い、既存Reviewを上書きしない。

control planeの「最新レビュー版」は参照情報であり、Reviewer自身が作成した未反映Review等が存在し得るため、Review_Seq決定時は実際のReview保存先にある既存Seqも確認する。

## 2.3. Review対象版の固定

Review開始時に、00/10それぞれについて次を固定する。

```text
対象commit SHA
対象blob SHA
```

commit SHAは履歴上の対象時点、blob SHAは内容同一性を固定する。

control planeが指定されている場合、対象Entry × 成果物行から次を確認する。

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

control planeを使用しないReviewでは、レビュー対象commitを別途明示して固定する。

legacy checkpointで成果物単独commitではない場合は、そのcommit時点の対象blobとの同一性を確認する。

対象版を固定できないことは唯一のfail-stop条件である。format・内容・コード上の不適合はReviewを止める理由にはせず、Findingとして記録し、残りのQAも完了させる。

## 2.4. `00_contents` Reviewの詳細

`00_contents` は**伝承内容 / Evidence整理の正本**としてReviewする。

第1章の順序どおり、まずtutorial構造、次にtutorial必須記載内容、次にtutorial記載粒度を確認し、その後にEvidence / Content内容へ進む。

**tutorial構造**では、`0000_00_contents.md` を正として、必須見出し、見出し順序、フィールド名、表列、固定区分、ブロック構造を照合する。意味的に同等の情報が別形式で存在するだけでは構造準拠としない。

**tutorial必須記載内容**では、各必須節・フィールドに、その節が要求する内容が実際に記載されているかを見る。空欄、別節への暗黙委譲、項目名だけ存在して内容がない状態を準拠としない。

**tutorial記載粒度**では、第三者が00だけから主要な伝承形、Evidenceの役割、初期形と異伝、不確実性を再構成できるかを見る。構造が合っているだけではtutorial準拠としない。

**Summary Reconstruction Equivalence Test**

`1.1. summary` の記載粒度は、summaryの長さ、情報項目数、表面的な文章類似度ではなく、**summaryのみから復元した伝承が原Evidenceから確認できる伝承構造を保存しているか**で検証する。

このテストではEncoderを実施しない。EncoderはCoderがEvidence / 詳細記述からsummaryを作成した時点ですでに行われている。ReviewerはDecoderと比較だけを行う。

実行順序は次とする。

```text
A. Blind Decode
   1. `1.1. summary` だけを読む。
   2. 原Evidence、1.2以降、10_analysis、過去Reviewの知識で補完しない。
   3. summaryから復元できる伝承ストーリー／論理構造を文章として固定する。
   4. このBlind DecodeはReference Story確認後に書き換えない。

B. Reference Story確認
   1. Blind Decode固定後に、原Evidenceおよび00の1.2以降を確認する。
   2. Evidenceから確認できる伝承ストーリー／論理構造をReference Storyとして整理する。
   3. この段階ではEvidence資料自体の品質・directness・分類の最終判定は行わない。それは工程5で行う。

C. Structural Probe
   Blind DecodeとReference Storyへ同一の問いを適用し、回答を並列比較する。

D. Analysis Invariance
   両ストーリーに同一のDimension中心質問を適用し、主要な分析回答が保存されるか確認する。

E. Reconstruction Verdict
   Structural / Analysis上の差分をもとにsummary粒度のPASS / FAILを確定する。
```

Structural Probeは最低限、次を用いる。

| Probe | 問い |
|---|---|
| P01 | 主な主体・役割は誰／何か |
| P02 | 主な作用対象は誰／何か |
| P03 | 何を契機に異常・中心命題の系列へ入るか |
| P04 | 原因／作用主体は何として描かれるか |
| P05 | 対象に具体的に何が起きる／何が作用するか |
| P06 | 対象は前後でどのような状態遷移をするか |
| P07 | 出来事はどの順序・反復・遅延・継続関係にあるか |
| P08 | 条件・規則・禁忌は何か |
| P09 | 回避・制御・利用方法と、その成否は何か |
| P10 | 明示された終端状態は何か |
| P11 | 何が未解決・不明のまま残るか |
| P12 | 初期形・後代形・主要異伝の境界は何か |

各Probeの差分は次で記録する。

```text
NONE     = 実質的に同じ構造を復元できる
LOSS     = ReferenceにはあるがBlind Decodeからは復元できない、または分析上意味のある具体性が失われる
CONFLICT = Blind DecodeからReferenceとは異なる構造が復元される
```

`LOSS` があるだけで自動的にFAILとはしない。固有名詞、描写上の装飾、分析に影響しない細部等が省略されてもよい。

次にAnalysis Invarianceを確認する。文章同士の類似度ではなく、Blind DecodeとReference Storyへ**同じDimension中心質問**を適用した場合に、主要な回答が変化するかを見る。この段階では現行10_analysisのコードを正しいものとして参照せず、00のsummary粒度検証用の独立プローブとして実施する。

少なくとも、差分により次が変わり得るかを確認する。

- D07〜D10の意味形成構造
- D11の発動・接触条件
- D12の作用対象
- D13の作用機構またはtaxonomy gapの有無
- D14の帰結極性
- D15の帰結領域・状態変化
- D16の時間構造
- D17の回避・制御・利用
- D18の作用レイヤー
- D19〜D20の流通・情報保持構造
- D21の現実アンカー
- Version Scopeまたは主要異伝境界

Blind DecodeとReference Storyの表現が異なっていても、これらの主要分析回答が保存されるならsummaryの圧縮は許容できる。

逆に、Reference Storyを読めば成立する合理的な分析候補・状態遷移・因果構造・Version境界が、Blind Decodeでは成立しない、別の回答になる、または判断不能へ縮退する場合、そのsummaryはanalysis-preservingではない。

判定の中心原則は次である。

> **原Evidenceを読んだ第三者と、summaryだけを読んだ第三者が、実質的に同じ伝承構造と主要な分析可能性を再構成できるか。**

「大筋が分かる」「主要イベント名が存在する」「詳細は1.2にある」「10_analysisを読めば補える」「意味的には似ている」はPassの根拠としない。一方で、原文の全ディテールをsummaryへ移すことも要求しない。

とくに、00へ分析結果そのものを書き込むことと、分析を可能にするContentを保持することを区別する。ある解釈を断定してはならない場合でも、その解釈の成立可否を左右する伝承内の具体的行動・状態変化・規則・終端を削ってはならない。

Summary Reconstruction Equivalence Testの結果は、00 Review mdへ**レビュー判断のEvidence**として保存する。最低限、次を残す。

```text
Blind Decode
Structural Probe Matrix
Analysis Invariance
Reconstruction Verdict
```

Structural Probe Matrixは原則として次の形式を用いる。

```markdown
| Probe | Blind Decode | Reference Story | Difference |
|---|---|---|---|
| P01 | ... | ... | NONE / LOSS / CONFLICT |
| ... | ... | ... | ... |
| P12 | ... | ... | NONE / LOSS / CONFLICT |
```

Analysis Invarianceは全Dimensionを長文化する必要はない。差分がない範囲はまとめて記録してよいが、差分があるDimensionは、Blind Decode側とReference Story側で何が変わるかを具体的に残す。

Evidence / Content内容では最低限、次を確認する。

- summaryから主要な伝承形・命題を再構成できる。
- 主体、場所、物、条件、規則、禁忌、帰結、不確実性が必要な粒度で記録されている。
- 最古確認と実際の起源を混同していない。
- 初期形と後代異伝を区別している。
- 典拠がURL列挙だけでなく、どの内容を何のEvidenceとして支えるか追跡できる。
- 本文未実見・資料不足・競合を隠していない。
- D01〜D21、Primary / Secondary、Version Scope裁定等をEvidence layerへ混入していない。
- 後代設定を初期Versionへ遡及していない。

Evidence roleや証拠強度の規則は `30_urban_legend_analysis_coding_rules.md` を正とする。

資料分類では、一次資料・同時代資料、原文ミラー・転載・復刻資料、二次資料・研究資料等の区別を、実際の到達資料に即して行う。転載保存を原ページそのものとして扱わず、後代資料から初期時点へ設定を遡及しない。

00でEvidenceとして採用した重要Contentは、第三者が資料名・URL・対象箇所等から追跡できる粒度を要求する。とくに後続10のPrimary、Status、Version Scopeを左右するContentが曖昧な「初期レス群」「後代資料」等に留まる場合は、traceability不足として扱う。

## 2.5. `10_analysis` Reviewの詳細

`10_analysis` は**分析コード付与の正本**としてReviewする。

第1章の順序どおり、まずtutorial構造、次にtutorial必須記載内容、次にtutorial記載粒度を確認し、その後にVersion Scope、D01〜D21、sense-making / 因果構造へ進む。

**tutorial構造**では、`0000_10_analysis.md` を正として、基本情報、Dimension見出し、中心質問、各DimensionのPrimary / Value、Parent、Secondary、Status、`判定根拠`、`この伝承における現れ方` 等の所定構造を照合する。所定の独立ブロックをインラインへ圧縮する等、tutorial所定形式を独自形式へ置換している場合は、内容が存在していても構造上の差分として判定する。

**tutorial必須記載内容**では最低限、次を確認する。

- 基本情報とVersion Scopeが存在する。
- D01〜D21がすべて存在する。
- 各Dimensionに必要なPrimary / Value、Parent、Secondary、Statusがある。
- 判定根拠が00の具体的Evidenceに接続する。
- 「この伝承における現れ方」がコード定義の言い換えではなくEntry固有の説明になっている。
- `U / NA / C` の理由が具体的に記載されている。

**tutorial記載粒度**では、第三者が各Dimensionについて「どのEvidenceから、どの解釈を経て、そのコード・Statusへ到達したか」を再構成できるかを見る。

分析コード妥当性では、コードID、Parent / Child、Primary / Secondary、Status、tie-break、taxonomy gap等の詳細について `20 / 30` を正とし、少なくとも次を確認する。

- 存在しないcode IDを使用していない。
- ParentがChildから正しく導出される。
- Primaryが中心質問に最も直接的である。
- Secondaryが独立した追加構造であり、単なる関連要素ではない。
- Evidence不足を最もありそうな値で埋めていない。
- taxonomy gapを `U` や近似Childで隠していない。
- EvidenceのdirectnessとDimension Statusが整合する。
- 下位資料しか固定できていない場合に、根拠なく `D` へ引き上げていない。

Version Scopeは00 Evidenceから再構成可能でなければならない。Scopeの外にある後代異伝、派生、翻案、別時点のContentを、理由なく分析コードへ混入しない。

sense-making / 因果構造QAでは特に次を確認する。

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

その他の重点QAは次とする。

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

## 2.6. 00 / 10 横断QA

00と10を別々に確認した後、次を確認する。

1. Version Scopeが00 Evidenceから再構成できる。
2. 10の判定根拠が00の具体内容・典拠へ戻れる。
3. 00で未確認の事項を10が確定していない。
4. Scope外の異伝を分析コードへ混ぜていない。
5. D07〜D17が同じVersionの一つの意味形成モデルになっている。
6. D18 L3のX Evidenceが00に記録され、Scope内である。
7. tutorial構造・内容・粒度の判定とFindingsが整合する。
8. Summary Reconstruction EvidenceでLOSS / CONFLICTとなった構造が、10の判定根拠で暗黙補完されていない。

理想的なtraceabilityは次である。

```text
00の具体的Evidence
→ Version Scope
→ 分析者の解釈
→ D01〜D21
```

00のEvidence強度・不確実性より10の確信度が高くなっていないかも確認する。00で未固定・転載経由・二次資料依存としている内容を、10でDirect扱いする場合は、独立した根拠がなければ不整合とする。

## 2.7. Finding重大度と最終判定

Majorは次を目安とする。

- Version Scope、Entry境界、中核の意味形成モデルを変える可能性が高い。
- Primary / Statusへ直接影響する重大な誤り。
- Evidence / Analysis責務混同が大きい。
- taxonomy gapを誤コードで隠している。
- tutorial必須内容・粒度が広範に欠け、再現性を確保できない。

Moderateは次を目安とする。

- 一部Dimension・一部節のコード / Statusへ影響し得る。
- Evidence directness、Secondary、局所的tutorial不足等の再確認が必要。
- Summary Blind DecodeとReference Storyの差分が、一部の主要Dimension回答または合理的分析候補を変える。

Minorは次を目安とする。

- 見出し番号、表記、軽微なフォーマット等で、分析判断を変えない局所修正。
- Summary上のLOSSが分析不変性を壊さず、伝承構造の理解にも実質影響しない局所的欠落。

最終判定は次とする。

```text
Findingなし              → 問題なし（Pass）
Minorのみ                 → 要修正（Minor）
Moderateあり、Majorなし   → 要修正（Moderate）
Majorあり                 → 要修正（Major）
```

複数Findingがある場合、最終判定は最大重大度に従う。内容上維持可能な点が多くてもFindingを相殺しない。format、Evidence、code、traceabilityは独立したQA軸として扱う。

## 2.8. Review mdの記載事項と標準フォーマット

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

加えて、00 Review mdには次を必須記録とする。

```text
Summary Reconstruction Evidence
- Blind Decode
- Structural Probe Matrix
- Analysis Invariance
- Reconstruction Verdict
```

各Findingは原則として次を含む。

```text
何が現在書かれているか
→ 何が規則・理論・tutorialと衝突するか
→ なぜ分析上 / 再現性上問題か
→ どの方向へ修正すべきか
```

Reviewerは修正後コードを勝手に正本へ書き込まず、必要なら「修正方向」または作業仮説として示す。

00 Review mdの標準フォーマットは次とする。

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
## 4. Summary Reconstruction Evidence
### 4.1. Blind Decode
### 4.2. Structural Probe Matrix
### 4.3. Analysis Invariance
### 4.4. Reconstruction Verdict
## 5. QA
## 6. 修正優先順位
## 7. 最終判定
```

10 Review mdの標準フォーマットは次とする。

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

## 2.9. 保存後確認とReview完了条件

1 EntryのReviewは次をすべて満たした場合のみ完了とする。

- 対象00/10を取得した。
- 対象commit SHA / blob SHAを固定した。
- control plane使用時はpost-SHAと対象版を照合した。
- 00のtutorial構造・内容・粒度を、第1章の順序どおり確認した。
- 00 Summary Blind DecodeをReference確認前に固定した。
- 00 Structural Probe Matrix / Analysis Invariance / Reconstruction VerdictをReview mdへ保存した。
- 00をEvidence layerとしてReviewした。
- 10のtutorial構造・内容・粒度を、第1章の順序どおり確認した。
- Version Scopeを検証した。
- D01〜D21をReviewした。
- D07〜D17を一つの意味形成モデルとして確認した。
- 00→10 traceabilityを確認した。
- taxonomy gap / U / NA / Cを区別した。
- 00/10で同じReview Seqを使用した。
- Review mdへ対象commit SHA / blob SHAを記録した。
- Review mdにFindings、維持可能な点、QA、修正優先順位、最終判定を記録した。
- 保存後にReview mdを再取得し、保存内容を確認した。

保存後再確認では、Reviewファイルが存在するだけでは完了としない。対象commit SHA / blob SHA、Review Seq、最終判定、Findingsが意図した内容で保存されていることを確認する。00 ReviewではSummary Reconstruction Evidenceも再取得内容に含まれていることを確認する。

Review完了後のCoder修正、control planeのStatus変更、pre/post-SHA更新は `0000_workflow_10_each_lore_analysis.md` に従う。Reviewerはこれらを代行しない。