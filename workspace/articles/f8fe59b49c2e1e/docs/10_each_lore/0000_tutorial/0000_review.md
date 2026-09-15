# 個別伝承レビュー手順書

## 0. この文書の目的

本文書は、`docs/10_each_lore/<Entry>/` 配下の個別伝承について、

- `<Entry_ID>_00_contents.md`
- `<Entry_ID>_10_analysis.md`

をレビューし、その結果を `docs/99_work/review_10_each_lore/` 配下へ保存するための**独立実行可能なレビュー手順書**である。

レビュー担当者は、既存コード値や過去レビューを正しいものとして前提化しない。レビュー開始時には本文書と、本文書が指定する正本を確認する。

このレビューの目的は、単にコードの妥当性を点検することではない。以下を確認することである。

1. **最初に** `00_contents` / `10_analysis` が、それぞれのtutorialについて、
   - 見出し・節・フィールド等の**構造**
   - 各節・各Dimensionに要求される**記載内容**
   - 第三者が内容・Evidence・分析判断を再構成できる**記載粒度**
   のすべてに準拠しているか。
2. Evidence / Content layer と Analysis / Coding layer が責務分離されているか。
3. Version Scope が証拠に対して保守的かつ再現可能に固定されているか。
4. D01〜D21 が理論設計、コード体系、コーディング規則に適合しているか。
5. 特に D07〜D17 が、一つの coherent な **sense-making model / 意味形成モデル**として接続しているか。
6. `原資料上の具体的展開 → 分析者の解釈 → コード` を第三者が追跡できるか。
7. 証拠不足、競合、非該当、taxonomy不足が混同されていないか。

**重要:** tutorial準拠とは「見出しが存在する」ことだけを意味しない。tutorialがその節で要求する情報が実際に書かれ、要求される密度で説明されていることまで含む。

---

# 1. レビュー時に必ず読む正本

## 1.1. tutorial / workflow

- `docs/10_each_lore/0000_tutorial/0000_workflow.md`
- `docs/10_each_lore/0000_tutorial/0000_00_contents.md`
- `docs/10_each_lore/0000_tutorial/0000_10_analysis.md`

## 1.2. 理論・コード・運用規則

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`
- `docs/00_research_overview/80_appendix/20_what_is_sense_making.md`

## 1.3. 文書間の責務

| 論点 | 正本 |
|---|---|
| 分析単位、D01〜D21の概念定義・境界 | `10_urban_legend_analysis_axes_theoretical_design.md` |
| 使用可能なコードID、Parent / Child / Value | `20_urban_legend_parent_child_code_system.md` |
| Evidence、Status、Primary / Secondary | `30_urban_legend_analysis_coding_rules.md` |
| 個別Entry作業順序・責務分離・QA | `0000_workflow.md` |
| `00_contents` の構造・記載内容・記載粒度 | `0000_00_contents.md` |
| `10_analysis` の構造・記載内容・記載粒度 | `0000_10_analysis.md` |
| sense-makingの理解補助 | `20_what_is_sense_making.md` |

`sense-making` 補助文書と理論設計が競合する場合は理論設計を正とする。

過去の個別Entry、過去レビュー、作業メモ、旧Excel値は正本ではない。

---

# 2. レビューの基本単位と保存規則

## 2.1. 1 Entryずつ完結させる

複数Entryを依頼された場合でも、必ず次の単位で処理する。

```text
Entry A の 00 をレビュー
→ Entry A の 10 をレビュー
→ Review_A_00 と Review_A_10 を保存
→ 保存した2ファイルを再取得して確認
→ Entry A 完了
→ 次の Entry B へ進む
```

Entry A のレビュー結果を保存・確認する前に Entry B へ進まない。

## 2.2. Entry_ID

`Entry_ID` は4桁ゼロ埋め。

```text
1 → 0001
11 → 0011
59 → 0059
```

## 2.3. Review_Seq

`Review_Seq` は3桁ゼロ埋め。

保存先:

```text
docs/99_work/review_10_each_lore/<Entry_ID>/Review_<Entry_ID>_00_<Review_Seq>.md
docs/99_work/review_10_each_lore/<Entry_ID>/Review_<Entry_ID>_10_<Review_Seq>.md
```

同一レビューサイクルの00/10は同じSeqを使う。既存最大Seq+1を使い、既存ファイルを上書きしない。

最大Seqで片側だけが存在する場合、同一作業の継続かつ対象版も同一と確認できる場合のみ同Seqで補完する。それ以外は次Seqを使う。

---

# 3. レビューと修正作業を分離する

明示的な修正依頼がない限り、レビュー中に次を行わない。

- 対象00を編集しない。
- 対象10を編集しない。
- taxonomyを変更しない。
- 新しいChild / Statusを勝手に作らない。
- Evidence不足を推測で補わない。

レビューでは、**問題、根拠、分析上の影響、修正方向**をレビュー結果へ記録する。

---

# 4. レビュー開始時の入力確認と必須順序

対象Entryについて00/10を取得し、可能なら対象blob SHAを記録する。

レビュー順序は必ず次とする。

```text
00:
1. tutorial構造準拠
2. tutorial記載内容準拠
3. tutorial記載粒度準拠
4. Evidence / Content内容レビュー

10:
1. tutorial構造準拠
2. tutorial記載内容準拠
3. tutorial記載粒度準拠
4. Version Scope / D01〜D21 / sense-making内容レビュー
```

**構造が合っていても、tutorialが要求する内容が欠ける、または記載が薄く第三者が再構成できない場合はtutorial非準拠である。**

非準拠はQAだけに書かず、修正が必要なら独立Findingとする。

### tutorial非準拠の重大度の目安

- `Minor`: 見出し番号、中心質問、軽微な表記等。必要な内容・粒度は保たれる。
- `Moderate`: 必須フィールド、典拠位置、説明項目等が欠け、比較・再現性が明確に低下する。
- `Major`: 広範な内容欠落・粒度不足により、Evidence traceabilityや分析判断の再構成が困難。

---

# 5. `00_contents` レビュー手順

`00_contents` は **Evidence / Content layer** である。

## 5.1. 最初のゲート — tutorial準拠確認

### 5.1.1. 構造準拠

現行 `0000_00_contents.md` と対象00を比較し、少なくとも次の構造を確認する。

```text
# 0. INTRODUCTION
## 0.1. 文書の役割
## 0.2. 典拠と証拠上の扱い
### 0.2.1. 一次資料・同時代資料
### 0.2.2. 原文ミラー・転載・復刻資料
### 0.2.3. 二次資料・研究資料

# 1. 伝承内容
## 1.1. summary
## 1.2. 詳細展開
## 1.3. 登場人物・主体
## 1.4. 場所・空間
## 1.5. 物・記号・媒体
## 1.6. 条件・規則・禁忌
## 1.7. 異常・不可解な現象／中心命題
## 1.8. 行動・対応
## 1.9. 帰結・終端状態
## 1.10. 未解決点・不確実性

# 2. 伝承史・異伝
## 2.1. 最古確認
## 2.2. 流通・拡散
## 2.3. 主要な異伝・派生
## 2.4. 変容上の重要点

# 3. 証拠対応
## 3.1. 内容と典拠の対応表
## 3.2. 短い原文引用
## 3.3. 証拠上の保留事項
```

確認事項:

- 必須節が存在するか。
- 見出し階層・番号・順序が正規形に沿うか。
- 独自構造へ置換していないか。
- 必須情報が別節へ散在して比較困難になっていないか。

### 5.1.2. 記載内容準拠

**各節が存在するだけでなく、その節にtutorialが要求する情報が実際に記載されているかを逐節確認する。**

最低限、次を確認する。

#### 0.2 典拠

- 一次資料・同時代資料: 資料名、URL / 書誌 / 所蔵、対象箇所、確認日、Evidence上の位置付け。
- ミラー・転載・復刻: 一次資料との関係、転載種別、証拠上の注意。
- 二次・研究資料: 書誌、確認日、用途。
- 本文未実見や原資料未固定の場合はその限界。

#### 1.1 summary

物語型なら、Evidenceが存在する範囲で次が追えるか。

```text
発端
→ 異常の顕在化
→ 展開
→ 対応
→ エスカレーション／転換
→ 終端
→ 未解決部分
```

命題・俗信・規則型なら、次が追えるか。

```text
対象状況
→ 条件
→ 主張／規則
→ 想定作用
→ 帰結
→ 回避・制御・利用
→ 証拠上の限界
```

#### 1.2 詳細展開

- summaryの主要展開が分解されているか。
- 物語なら時系列、命題型なら論理展開が追えるか。
- 可能な範囲で頁、段落、レス番号、記事日、採録番号等へ接続しているか。

#### 1.3〜1.5 主体・場所・物

- tutorial指定の列・情報が埋められているか。
- 主体: 種別、役割、典拠上確認できる情報、典拠位置。
- 場所: 実在性、伝承内の役割、典拠位置。
- 物・記号・媒体: 種別、伝承内の役割、典拠位置。
- 該当なしの場合は「特記事項なし」等で明示されているか。

#### 1.6 条件・規則・禁忌

- 条件、規則、禁忌、対処・回避、利用法を区別しているか。
- 資料にない項目を推測で補っていないか。

#### 1.7〜1.10

- 1.7: 主要な異常現象・中心命題が具体的に列挙されているか。
- 1.8: 実際に語られる行動と後代の推奨・解釈を分けているか。
- 1.9: 明示された帰結、暗示された帰結、確認されない帰結を区別しているか。
- 1.10: 未解決点・資料間競合・推測禁止事項を残しているか。

#### 2.1〜2.4 伝承史・異伝

- 最古確認と実際の起源を分けているか。
- 流通・拡散をEvidenceのある範囲で時系列化しているか。
- 異伝ごとに時期、地域・媒体、差分、典拠が追えるか。
- 変容点を記述しても、Version ScopeやDコードの裁定まで00で行っていないか。

#### 3.1〜3.3 証拠対応

- 主要内容について `内容要素 → 典拠 → 位置 → 証拠種別` が追えるか。
- 短い原文引用が出典と対応しているか。
- 保留事項が明示されているか。

### 5.1.3. 記載粒度準拠

記載粒度は文章量ではなく、**第三者が再構成できる情報密度**で判定する。

合格基準:

- 原典を毎回開き直さなくても、主要な伝承形・条件・作用・帰結・異伝差を再構成できる。
- 重要な主張が「怖い話である」「危険な場所である」等の抽象要約だけで終わらず、人物・場所・物・行為・順序・因果主張が残っている。
- 典拠欄がURLの羅列ではなく、何のEvidenceとして使う資料か分かる。
- 表や箇条書きが空欄埋めではなく、後続分析で実際に使える具体性を持つ。
- Evidenceがない場合は省略せず、「確認できない」「特記事項なし」等で限界を明示する。

不合格例:

- 1.1 summaryが数行の辞書的紹介で、展開・条件・終端が再構成できない。
- 1.3〜1.6が見出しだけ、または一語・一文だけでtutorial要求情報を満たさない。
- 3.1に資料名だけあり、どの内容をどの位置が支えるか不明。
- 「詳細は原典参照」で済ませ、00自身に分析可能なEvidence情報が残っていない。

**見出し構造が完全でも、上記粒度を満たさなければFindingとする。**

## 5.2. Evidenceの強度と役割

- 一次・同時代資料 > ミラー・転載・復刻 > 研究 > 後代まとめ、の順を意識する。
- 書誌ページ・販売ページ・目録だけから本文内容を推測しない。
- 本文未実見は明示する。
- 実在事件・制度・場所を、伝承内部の超常因果の証明にしない。
- fact-checkによる否定を、伝承不存在の証拠にしない。

Evidence roleを使う場合:

- `C`: Content
- `H`: History
- `X`: External effect
- `A`: Anchor

## 5.3. Evidence layerへの分析混入

00に次が入っていないか確認する。

- `Version_Scope` の採用／除外判断。
- D01〜D21コード。
- Primary / Secondary裁定。
- 「意味形成対象」「意味形成核」等、D07〜D09の分析結論。
- 心理・社会過程の断定。
- 後代代表形を初期形へ遡及する分析。

## 5.4. 最古確認と起源

```text
現在確認できる最古資料 ≠ 実際の起源
公刊年 ≠ 成立年
新聞報道日 ≠ 噂の発生日
古い伝承らしい ≠ 口承起源が確認済み
```

## 5.5. 異伝の保持

- 初期形と後代定型を分ける。
- 地域差・媒体差・時代差を一つの固定シナリオへ混ぜない。
- 後代の有名設定を古いVersionへ遡及しない。
- Conflict、不明点、未確認事項を消さない。

## 5.6. 証拠対応

理想:

```text
内容要素
→ 典拠
→ 頁／レス／段落／記事日／採録番号
→ 直接確認できる文言・出来事
```

---

# 6. `10_analysis` レビュー手順

`10_analysis` は **Analysis / Coding layer** である。

推奨思考順序:

```text
D07 → D08 → D09 → D10 → D11 → D12 → D13 → D14 → D15 → D16 → D17 → D18
→ D01 → D02 → D03 → D04 → D05 → D06 → D19 → D20 → D21
```

レビュー結果の記載順はD01→D21でよい。

## 6.1. 最初のゲート — tutorial準拠確認

### 6.1.1. 構造準拠

現行 `0000_10_analysis.md` と対象10を比較し、最低限次を確認する。

```text
# 1. 伝承エントリ基本情報
- Entry_ID
- 伝承エントリ名称
- Macro_Category
- Entry_Type
- Version_Scope

# 2. 分析概念次元
## 2.1. 来歴・流通・提示の次元
### 2.1.1. D01: 生成年代
...
### 2.1.6. D06: 真実性提示

## 2.2. 意味形成の次元
### 2.2.1. D07: 意味形成対象
...
### 2.2.4. D10: 因果源存在論

## 2.3. 因果・行動モデルの次元
### 2.3.1. D11: 発動・接触条件
...
### 2.3.8. D18: 作用レイヤー

## 2.4. 社会的埋め込みの次元
### 2.4.1. D19: 流通範囲
### 2.4.2. D20: 特権情報保持者
### 2.4.3. D21: 現実アンカー
```

各Dimensionは原則としてtutorialの次の構造を持つ。

```text
### <階層番号>. Dxx: <次元名>

**<中心質問>**

- Primary Child / Value:
- Primary Parent:
- Secondary:
- Status:

**判定根拠**
...

**この伝承における現れ方**
...

必要に応じて境界・留保
```

D18等、固有構造がある次元はtutorial正本に従う。

確認事項:

- Entry_IDは4桁ゼロ埋めか。
- トップレベル章、次元グループ、D01〜D21見出しが揃うか。
- 階層番号・中心質問を独自に省略していないか。
- tutorialにないR3/R4 checkpoint、旧Excel比較、作業commit等をCoding正本へ混在させていないか。

### 6.1.2. 記載内容準拠

**各Dimensionに必要な項目が形式上あるだけでなく、tutorialが要求する意味内容が書かれているかを確認する。**

最低限、各Dimensionで次を確認する。

- `Primary Child / Value`、`Primary Parent`、`Secondary`、`Status` が、その次元で必要な形で記載されている。
- `判定根拠` が単なるコード定義の言い換えではなく、00の具体的内容・Evidence・典拠に基づく。
- `この伝承における現れ方` が、人物・場所・物・出来事・主張・条件・帰結を使って、当該Entryで実際に何が起きるためそのコードになるかを説明する。
- 異伝差、後代設定、Evidence不足、別コードとの境界が重要な場合、境界・留保が明示される。

Status別の必須内容:

- `D`: どの直接Evidenceが値を支えるか分かる。
- `I`: Evidenceからどの推論を経て値へ到達したか分かる。
- `U`: **何が確認できず、どこまで言えないためUなのか**を書く。
- `NA`: **なぜ伝承構造上その次元が成立しないか**を書く。
- `C`: **何と何が同一Scope内で競合し、なぜ解消できないか**を書く。

Version Scopeについては、単なる一文の要約ではなく、少なくとも次が再構成できる必要がある。

- 今回コードする伝承形。
- 時期・媒体・地域等、Scopeを限定する条件。
- 主要な採用要素。
- 重要な後代異伝・別系統を除外する場合、その境界。

### 6.1.3. 記載粒度準拠

粒度の合格基準は、**後からコード表や作業メモへ戻らなくても、00 Evidence → 解釈 → コードの対応を第三者が再構成できること**である。

各D01〜D21について、最低限次を満たすか確認する。

- なぜそのPrimaryが選ばれたかが分かる。
- Secondaryがある場合、Primaryだけでは失われる独立構造が分かる。
- Statusの理由が分かる。
- そのコードが当該伝承でどう現れるかが具体的に分かる。
- 判断上重要な代替解釈・境界がある場合、それを無視していない。

不合格例:

```text
判定根拠: 伝承内容から判断。
この伝承における現れ方: 危険な場所として現れる。
```

のように、具体的Evidence・解釈過程・Entry固有展開を再構成できない記載。

また、次も粒度不足とする。

- 21次元の多くが一文のコード定義言い換えだけ。
- `U` なのに「Evidence不足」の一語だけで、何が不足するか不明。
- `C` なのに競合内容を記載しない。
- `この伝承における現れ方` が毎Dimensionほぼ同じ抽象文で、Entry固有情報がない。
- 判定根拠がR3/R4や旧Excelを参照するだけで、10自身から判断を再構成できない。

**コード値自体が正しくても、tutorial要求の記載内容・粒度を満たさなければFindingとする。**

## 6.2. Version Scope

- 00 Evidenceから再構成可能か。
- 最古のcoherentな伝承形を優先しているか。
- 後代設定を「有名だから」で遡及していないか。
- 競合Versionを無理に統合していないか。
- D07、D10〜D17が変わるならEntry境界への影響を確認する。

## 6.3. Status

Statusは `D / I / U / NA / C` のいずれか一つ。

- `D`: Direct
- `I`: Inferred
- `U`: relevantだがEvidence不足
- `NA`: 伝承構造上非該当
- `C`: 同一Scope内部で解消不能の競合

`D/I` 等の独自複合Statusを作らない。

## 6.4. Primary / Secondary

D / I のH3等では:

- Primaryは1つ。
- Secondaryは0〜2個。
- Primaryは中心質問への直接性が最も高いもの。
- Secondaryは独立した追加構造がある場合のみ。

Tie-break:

```text
中心質問への直接性
→ Evidence強度
→ sense-making構造上の不可欠性
→ unsupported assumptionを増やさない保守性
```

## 6.5. コードの存在とParent整合

- code systemに実在するか。
- Parentが正しいか。
- 名前ではなく定義本文に適合するか。
- 近いコードへ無理に押し込んでいないか。

## 6.6. sense-making / 意味形成レビュー

### 6.6.1. 基本原則

**1伝承エントリ = 1意味形成モデル**。

直接コードするのは心理・社会過程そのものではなく、伝承に符号化されたsense-making artifactである。

基本変換:

```text
異常・不可解・不確実な入力
→ 問題の切り出し
→ cue選択
→ 命名・分類・原因帰属・主体帰属・パターン化・規則化
→ 予測・禁忌・回避・利用・検証
→ 共有可能な世界モデル
```

### 6.6.2. D07 / D08 / D09

- D07: この伝承がなければ何が説明されないまま残るか。
- D08: 何を手掛かりに問題が立ち上がるか。
- D09: 不可解なものをどう理解可能にするか。

D13の作用結果をD08へ逆流させない。

Evidenceがないことを `UNSUPPORTED_ASSERTION` の正Evidenceへ自動変換しない。

### 6.6.3. D09 / D10

- D09 = 理解可能にする操作。
- D10 = 原因を何として世界に置くか。

超自然性が証明されないことを、通常人間の正Evidenceへ変換しない。

### 6.6.4. D08 / D11

- D08 = 問題化のcue。
- D11 = 因果系への入口。

---

## 6.7. D10〜D17 因果モデルQA

### 6.7.1. D10

強い存在論は、Scope Evidenceが明示・安定して支持する場合に限る。

### 6.7.2. D11

因果系への入口をコードする。安定した発動条件が既存Childにない場合、近似コードで隠さずtaxonomy gapを検討する。

### 6.7.3. D12 → D13 → D15

一本の因果列として確認する。

```text
D12 誰／何に
→ D13 何をして
→ D15 最終的に何の領域が変化するか
```

`MANIFEST_ONLY` は「現れる／認識されるが追加作用を必須としない」場合のみ。作用コード不明時のfallbackにしない。

### 6.7.4. D14 / D15

- 不気味だからNegativeにしない。
- 攻撃があるから死亡にしない。
- 投稿停止を死亡・失踪にしない。

### 6.7.5. D16

伝承の流通年数ではなく、**発動条件から主作用・帰結までの内部時間構造**を見る。

### 6.7.6. D17

- 対処法の記載がない ≠ 回避不能。
- `XするとY` から `Xしなければ安全` を分析者が逆算しただけでは回避規則Evidenceにならない。
- 強い不可避コードは明示Evidenceを要求する。
- 忘却等、安定構造を表すChildがなければtaxonomy gapを検討する。

---

## 6.8. D01〜D06 / D18〜D21 重点QA

### 6.8.1. D01

**最古確認資料を生成時期へ変換しない。**

### 6.8.2. D02

最古確認流通媒体を扱う。起源媒体を推測しない。古いから口承としない。

### 6.8.3. D03

Version Scope内で確認できる流通媒体だけを扱う。

### 6.8.4. D04

時間的変容を示すHistory Evidenceが必要。

**単に複数異伝が並存するだけで、どのように変化したかを断定しない。**

### 6.8.5. D05

伝承の提示形式を見る。Evidence資料の形式と混同しない。

### 6.8.6. D06

伝承自身が要求する「本当らしさ」を見る。研究者・報道側の真偽評価をそのまま移さない。

### 6.8.7. D18

- L1: 伝承内部の因果。
- L2: 現実受容者が伝承内容上の因果対象に入る自己適用。
- L3: 伝承流通によって現実社会に結果が生じたX Evidence。

L3 EvidenceはVersion Scope内でなければならない。

### 6.8.8. D19

Scope Evidenceが支持する流通範囲を超えて全国等へ拡張しない。公開Webで閲覧可能 ≠ 全国流通。

### 6.8.9. D20

- `NO_ONE_KNOWS` ≠ 資料に答えがない。
- `NO_HIDDEN_TRUTH` ≠ 特権保持者を確認できない。
- experiencerが詳細を知る ≠ 真相保持者。

### 6.8.10. D21

Evidence資料の保存場所ではなく、Scoped Content自体の現実アンカーを見る。

---

## 6.9. taxonomy gap

```text
A. 何が起きるかEvidenceが足りない → U候補
B. 何が起きるか明確だが表現できるChildがない → taxonomy gap
```

対象10がtaxonomy gapを認識しながら、

- Uで隠す
- 近いChildへ押し込む
- 定義を拡張する
- R4等ではgapと認識しながらCoding正本では通常コードにする

場合はFindingとする。

レビュー担当者は新コードを作らず、taxonomy / 保存規則側へのエスカレーションを記録する。

---

# 7. 00 / 10 横断QA

1. Version Scopeが00 Evidenceから再構成できるか。
2. 10の判定根拠が00の具体内容・典拠へ戻れるか。
3. 00で未確認の事項を10が確定していないか。
4. 後代異伝をScopeへ無断混入していないか。
5. D07〜D17が同じ一つのVersion世界モデルになっているか。
6. D18 L3のX Evidenceが00に記録され、Scope内か。
7. D01〜D06とD07〜D17で異なるVersionを混ぜていないか。
8. 00/10の`tutorial構造`判定とFindingが一致するか。
9. 00/10の`tutorial記載内容・粒度`判定とFindingが一致するか。

理想:

```text
00の具体的Evidence
→ Version Scope
→ 分析者の解釈
→ D01〜D21コード
```

---

# 8. Findingの重大度

## 8.1. Major

- Version Scopeを変える可能性が高い。
- D07〜D17の中核モデルを変える。
- Primary、Status、Entry境界へ直接影響する。
- Evidence / Analysis責務混同が大きい。
- taxonomy gapを誤コードで隠す。
- tutorialの必須内容・粒度が広範に欠け、Evidence traceabilityまたは分析再現性を確保できない。

## 8.2. Moderate

- 一部次元・一部節のコード／Statusへ影響し得る。
- Evidence directness、Secondary等の再確認が必要。
- tutorial必須内容・記載粒度が局所的に不足し、比較・再現性が明確に低下する。

## 8.3. Minor

- 見出し番号、4桁ゼロ埋め、中心質問の欠落等。
- **必要な記載内容・記載粒度は満たしており**、分析結果・Evidence解釈を変えない局所修正。

## 8.4. 最終判定

```text
Findingなし              → 問題なし（Pass）
Minorのみ                 → 要修正（Minor）
Moderateあり、Majorなし   → 要修正（Moderate）
Majorあり                 → 要修正（Major）
```

---

# 9. Reviewファイルの書き方

各Findingは原則として、

```text
何が現在書かれているか
→ 何が規則・理論・tutorialと衝突するか
→ なぜ分析上／再現性上問題か
→ どの方向へ修正すべきか
```

を含む。

事実、レビュー判断、作業仮説を区別する。

---

# 10. `Review_<Entry_ID>_00_<Review_Seq>.md` 標準フォーマット

```markdown
# Review <Entry_ID>_00_<Review_Seq> — Entry <Entry_ID> <伝承名> / `<Entry_ID>_00_contents.md`

## 0. レビュー情報
- `Entry_ID`: `<Entry_ID>`
- `Review_Seq`: `<Review_Seq>`
- 対象: `...`
- 対象blob SHA: `<...>`
- レビュー日: `YYYY-MM-DD`
- 判定: **問題なし（Pass） / 要修正（Minor|Moderate|Major）**

## 1. 結論

## 2. Findings

### F00-001 — <Findingタイトル>
- 重大度: **Major / Moderate / Minor**
- 対象: `<節>`

### 問題
### 根拠
### 修正方向

## 3. 維持可能な点

## 4. QA

| 項目 | 結果 | コメント |
|---|---|---|
| tutorial構造 | PASS / FAIL / REVIEW | **最初に確認** |
| tutorial記載内容・粒度 | PASS / FAIL / REVIEW | **構造とは別に確認** |
| 一次・二次資料の分離 | PASS / FAIL / REVIEW | |
| summary再構成可能性 | PASS / FAIL / REVIEW | |
| 最古確認と起源の分離 | PASS / FAIL / REVIEW | |
| 後代異伝の遡及防止 | PASS / FAIL / REVIEW | |
| Evidence / Analysis責務分離 | PASS / FAIL / REVIEW | |
| 典拠traceability | PASS / FAIL / REVIEW | |
| 不確実性の保持 | PASS / FAIL / REVIEW | |

## 5. 修正優先順位

## 6. 最終判定
```

FindingなしでもFindings、QA、最終判定を省略しない。

---

# 11. `Review_<Entry_ID>_10_<Review_Seq>.md` 標準フォーマット

```markdown
# Review <Entry_ID>_10_<Review_Seq> — Entry <Entry_ID> <伝承名> / `<Entry_ID>_10_analysis.md`

## 0. レビュー情報
- `Entry_ID`: `<Entry_ID>`
- `Review_Seq`: `<Review_Seq>`
- 対象: `...`
- 対象blob SHA: `<...>`
- レビュー日: `YYYY-MM-DD`
- 判定: **問題なし（Pass） / 要修正（Minor|Moderate|Major）**

## 1. 結論

## 2. Findings

### F10-001 — <Findingタイトル>
- 重大度: **Major / Moderate / Minor**
- 対象: `Dxx` または該当節
- 現在値: `<必要な場合>`

### 問題
### 根拠
### 修正方向

## 3. 維持可能な点

## 4. Sense-making再構成案

必要な場合のみ。確定コードではなく作業仮説と明示する。

## 5. QA

| 項目 | 結果 | コメント |
|---|---|---|
| tutorial構造 | PASS / FAIL / REVIEW | **最初に確認** |
| tutorial記載内容・粒度 | PASS / FAIL / REVIEW | **構造とは別に確認** |
| Version Scope | PASS / FAIL / REVIEW | |
| 後代異伝の遡及排除 | PASS / FAIL / REVIEW | |
| D01〜D21存在 | PASS / FAIL / REVIEW | |
| code ID / Parent整合 | PASS / FAIL / REVIEW | |
| Primary / Secondary規則 | PASS / FAIL / REVIEW | |
| StatusとEvidence強度 | PASS / FAIL / REVIEW | |
| D07/D08/D09 sense-making連鎖 | PASS / FAIL / REVIEW | |
| D08 / D11分離 | PASS / FAIL / REVIEW | |
| D09 / D10分離 | PASS / FAIL / REVIEW | |
| D12→D13→D15因果列 | PASS / FAIL / REVIEW | |
| D13 fallback誤用なし | PASS / FAIL / REVIEW | |
| D17 Evidence整合 | PASS / FAIL / REVIEW | |
| D18 L3 X Evidence | PASS / FAIL / REVIEW | |
| taxonomy gap分離 | PASS / FAIL / REVIEW | |
| 00→10 traceability | PASS / FAIL / REVIEW | |

## 6. 修正優先順位

## 7. 最終判定
```

---

# 12. レビュー実行チェックリスト

## 12.1. Entry開始前

- [ ] Entry_IDを4桁ゼロ埋めで確定した。
- [ ] 対象00/10を取得した。
- [ ] 対象blob SHAを記録した。
- [ ] 正本を確認した。
- [ ] 次Review_Seqを確定した。

## 12.2. 00レビュー

- [ ] **最初に**tutorial構造を比較した。
- [ ] **最初に**tutorial各節の必須記載内容を比較した。
- [ ] **最初に**tutorial要求の記載粒度を比較した。
- [ ] 構造が合っているだけでPassにしていない。
- [ ] 1.1〜3.3について、各節がtutorialの役割を果たす具体性を持つか確認した。
- [ ] 非準拠は必要に応じFinding化した。
- [ ] Evidence強度・責務分離・異伝・traceabilityを確認した。
- [ ] レビューmdを作成した。

## 12.3. 10レビュー

- [ ] **最初に**tutorial構造を比較した。
- [ ] **最初に**各Dimensionの必須記載内容を比較した。
- [ ] **最初に**各Dimensionの記載粒度を比較した。
- [ ] コード値が正しいだけでPassにしていない。
- [ ] 判定根拠から00 Evidenceへ戻れるか確認した。
- [ ] `この伝承における現れ方` がEntry固有の具体性を持つか確認した。
- [ ] U / NA / Cの理由がtutorial要求粒度で書かれているか確認した。
- [ ] Version Scope、D01〜D21、sense-making、taxonomy gapを確認した。
- [ ] レビューmdを作成した。

## 12.4. Entry完了前

- [ ] 00/10で同じSeqを使用した。
- [ ] 2レビューを保存した。
- [ ] 保存後に両方を再取得した。
- [ ] 最終判定と最大Finding重大度が一致する。
- [ ] このEntryを完了してから次へ進む。

---

# 13. レビューで禁止するショートカット

- tutorialの**見出しだけ**確認して準拠とみなす。
- 必須節があるだけで、節内の必須情報や粒度を確認しない。
- 10でコード値が妥当だから、判定根拠・現れ方が薄くてもPassにする。
- 「詳細は原典参照」「Evidence不足」等の一言記載を、tutorial要求粒度を満たすとみなす。
- フォーマット／内容／粒度の非準拠をQA欄だけに書き、必要なFindingを作らない。
- 過去コードや旧ExcelをEvidenceにする。
- 有名な後代設定を初期Versionへ入れる。
- 古いから口承、怪談だから幽霊、等のジャンル推定をする。
- Evidence不足を最もありそうなコードで埋める。
- taxonomy gapをUや近似Childで隠す。
- `MANIFEST_ONLY` をfallbackにする。
- triggerを反転して回避法を作る。
- 回避法の記載なしを回避不能とする。
- Scope外X EvidenceをD18 L3に使う。
- 00と10で別Versionを混ぜる。
- 保存・再取得確認前に次Entryへ進む。

---

# 14. 完了条件

1 Entryのレビューは次をすべて満たした場合のみ完了とする。

```text
1. 対象00/10と正本を確認した
2. 00のtutorial構造を確認した
3. 00のtutorial必須記載内容を確認した
4. 00のtutorial記載粒度を確認した
5. 00をEvidence / Content layerとしてレビューした
6. 10のtutorial構造を確認した
7. 10のtutorial必須記載内容を確認した
8. 10のtutorial記載粒度を確認した
9. Version Scopeを検証した
10. D01〜D21をレビューした
11. D07〜D17を一つのsense-making modelとして確認した
12. 00→10 traceabilityを確認した
13. taxonomy gap / U / NA / Cを区別した
14. Review_00を保存した
15. Review_10を保存した
16. 保存した2ファイルを再取得して確認した
```

上記を満たす前に、次のEntryへ進んではならない。
