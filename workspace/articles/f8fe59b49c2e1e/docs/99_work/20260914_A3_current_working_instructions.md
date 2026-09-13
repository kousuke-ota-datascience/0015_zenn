# 0. INTRODUCTION

本書は、A3パイロット49件reconcile作業で**現在実際に適用している作業インストラクションのスナップショット**である。

本書は理論・コード体系・コーディング規則の正本ではない。正本の内容を独自に変更・再定義せず、「どの文書を正本として、どの順序・粒度・コミット単位で作業するか」を再現可能にするための運用文書である。

内部のシステム／プラットフォーム指示は対象外とし、本プロジェクトにおいて明示されたユーザー指示、repository上のtutorial、Research Overview、reconcile運用のみを対象とする。

# 1. 正本と優先順位

## 1.1. コントロールプレーン

進捗・Entry別task状態・実行順序の正本は以下とする。

- `docs/99_work/20260913_A3_pilot_lore_analysis_reconcile.md`

1伝承エントリを完了するたびに更新する。

## 1.2. 理論・コード・コーディング規則

責務分担は以下とする。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
    - なぜ・何を測るか
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
    - どのParent / Child / Value / bitで測るか
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`
    - 証拠からコードへどう到達するか
- `docs/00_research_overview/90_ducumentation_metadata.md`
    - 文書構造・命名・管理規則

競合時は各責務の正本を優先し、本書側を修正する。

## 1.3. 個別伝承tutorial

個別伝承文書の機械可読な正規形は以下を正本とする。

- `docs/10_each_lore/0000_tutorial/0000_00_contents.md`
- `docs/10_each_lore/0000_tutorial/0000_10_analysis.md`

現行mainで対応するtutorial更新commitは以下。

- `0000_00_contents.md`: commit `e6bfa448f00af17f82c1e210507d3dbe5aa8c1b5`
- `0000_10_analysis.md`: commit `81c567a8012a58beceb239b22e430e603354a68d`

semanticに同じであっても、独自の項目名・独自の節・独自の省略表現を作らない。

## 1.4. Entry_IDとExcel

Entry_IDおよび最終的な49件コード同期先の正本は以下。

- `docs/20_analysis_summary/urban_legend_parent_child_full_application_v1.xlsx`

Entry_IDは4桁ゼロ埋めで個別伝承ディレクトリ・ファイル名へ使用する。

# 2. 分析単位と個別ファイルの責務

## 2.1. 分析単位

分析単位は「伝承エントリ」とする。

1伝承エントリ = 独立して伝達可能な最小のsense-making modelとし、最終的にExcelの1行と対応させる。

## 2.2. `*_00_contents.md`

Evidence / Content専用とする。

主に記録するもの:

- 伝承内容
- 典拠
- 伝承史
- 異伝
- source間の差異
- 証拠上の不確実性

原則として記録しないもの:

- Entry_ID
- Macro_Category
- Entry_Type
- Version_Scope
- D01〜D21のコード値

`00`だけを読んで、原典を毎回開き直さなくても伝承の内容・論理構造・異伝・証拠限界を再構成できる密度にする。

## 2.3. `*_10_analysis.md`

分析定義・コーディング専用とする。

記録するもの:

- Entry_ID
- 伝承エントリ名称
- Macro_Category
- Entry_Type
- Version_Scope
- D01〜D21
- 各次元の判定根拠
- 当該伝承における具体的な現れ方
- 必要な留保

単なるコード一覧にせず、21次元分析として再読可能な自然言語情報を保持する。

# 3. tutorial厳守事項

## 3.1. `00_contents` の章構造

以下の章・節名を正規形として使用する。

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

独自の `0.3` や、「詳細構造」「論理構造」等への見出し変更を行わない。

## 3.2. `10_analysis` の基本情報

基本情報は以下5項目だけを使用する。

```text
# 1. 伝承エントリ基本情報

- `Entry_ID`:
- `伝承エントリ名称`:
- `Macro_Category`:
- `Entry_Type`:
- `Version_Scope`:
```

`Evidence_Rank`、`主な成立・流通期`、`ネット発祥`等を追加しない。

## 3.3. D01〜D21の項目

全次元で以下を使用する。

```text
- Primary Child / Value:
- Primary Parent:
- Secondary:
- Status:

**判定根拠**

...

**この伝承における現れ方**

...
```

使用しない独自表現:

- `Primary:`
- `Value:`
- `Primary Child:`
- `Secondary 1 Child:`
- `Secondary Parent:`
- `現れ方:`
- 独自の末尾「パイロットメモ」章

留保は該当するDの本文内へ置く。

# 4. taskの意味

## 4.1. task 1 — 高密度化

旧薄様式の個別文書を、後からEvidenceと分析を再検討できる密度へ引き上げる。

このtask自体を理由にコード値を変更しない。

## 4.2. task 2 — tutorial準拠

既存文書をtutorialの見出し・項目・順序・文言へ正規化する。

このtask自体を理由にコード値・Statusを変更しない。

## 4.3. task 3 — 再コーディング

**コード体系・コーディング規則は変更せず、同じ物差しで各伝承エントリを測り直す。**

Evidenceが改善した結果、個別EntryのD01〜D21は変更してよい。

コード体系そのものに構造的問題を発見しても、その場で `10` / `20` / `30` を変更しない。体系変更が必要なら、個別reconcileとは別taskとして扱う。

## 4.4. task 4 — 未文書化Entry

未作成24件について、最初から高密度・tutorial準拠の00/10を作成し、そのEntry内でtask3まで完了する。

## 4.5. task 5〜7

- task 5: 49件の再コーディング結果をExcel正本へ同期
- task 6: pilot/full-application等の派生評価資料を再整合
- task 7: 文書・コード・Excel・評価資料・commit履歴の横断QA

# 5. 現在の実行単位

## 5.1. Entry単位で縦に完結する

横断的にtask 1だけ、task 2だけを先に全件実施しない。

1 Entryについて必要なtaskを最後まで終えてから次へ進む。

```text
最新main取得
→ 当該EntryのEvidence確認
→ 必要ならtask1
→ task2
→ task3
→ Entry commit
→ push後差分検証
→ control planeを別commitで更新
→ 停止または次Entry
```

すでに完了しているtaskは形式的にやり直さない。

## 5.2. 1 Entryのcommit境界

既存Entryでは原則として以下2ファイルだけを1つのEntry commitに含める。

```text
<ENTRY>_00_contents.md
<ENTRY>_10_analysis.md
```

片方だけ変更が必要な場合は、その1ファイルだけでもよい。

reconcile/control planeはEntry commitへ混ぜず、Entry commit検証後に別commitで更新する。

無関係なEntry、正本コード体系、評価資料、一時ファイルをEntry commitへ混入させない。

# 6. Evidenceと再コーディングの原則

## 6.1. Evidenceの優先順位

原則として以下の順に重視する。

1. 一次資料・同時代資料
2. 原文に近いミラー・転載・復刻
3. 信頼できる研究・二次資料
4. 後代のまとめ・解説

後代資料から最初期形を遡及推定しない。

## 6.2. `U` / `C` / `D` / `I`

- `U`: 証拠不足。不明を「最もありそうな値」で埋めない。
- `C`: 主要な異伝・資料・解釈が競合し、一意に閉じない。
- `D`: 証拠に直接明示される。
- `I`: 定義と証拠から合理的に推論できるが直接明示ではない。

## 6.3. D02

最古確認流通媒体をコードする。

- 古い伝承だから口承だったはず、は禁止。
- 後代研究の存在は、そのまま最初期流通媒体ではない。
- 実際に公開・配布された新聞・書籍・Web投稿等は、その時点で流通媒体として扱える。
- 起源と最古確認媒体を分離する。

## 6.4. D04

「定型化・カノン化」は時間的安定の証拠がある場合だけ使用する。

異伝増補、地域化、媒体移行、事件への付着等が直接確認できる場合は、それらを優先して検討する。

## 6.5. D15

語りが途絶えたことを死亡・失踪へ自動変換しない。

明示された最終状態だけをコードする。

## 6.6. D18

- L1: 伝承内因果
- L2: 現実の受容者自身が伝承内容上の因果対象になる
- L3: 伝承流通により、観察可能な社会行動・制度・市場・共同体変化が起きる

単に「受容者が怖がる」「メディアが影響する」だけでL2/L3を付けない。

## 6.7. Parent / Child

ParentはChildから一意に導出可能でなければならない。

ChildとParentを独立に発明・変更しない。

# 7. 調査と作業速度の運用

## 7.1. 調査はEntry単位

対象Entry以外の伝承調査へ横展開しない。

まず当該Entryについて、内容・最古確認・流通・主要異伝・反証／現実アンカーをまとめてEvidenceセットとして取得する。

## 7.2. 21次元を1件ずつ検索しない

コード体系は固定済みなので、毎回D01からD21まで個別検索を繰り返さない。

推奨順序:

1. 当該EntryのEvidenceをまとめて固定
2. 現行21次元を一括で仮判定
3. 旧コードとの差分と不確実な次元だけを抽出
4. その次元だけコード体系・coding rulesを再照合
5. 00/10をまとめて作成

既に定義が分かっているChildを、確認目的だけで何度も検索しない。

## 7.3. 過剰検証を避ける

Evidenceが不足している場合は `U` を使用し、追加検索を無制限に続けない。

「成立年代を確定したい」「最古媒体を必ず埋めたい」等を理由に、証拠以上の精度を追求しない。

# 8. GitHub運用

## 8.1. 書き込み前

- 最新main HEADを確認する。
- 対象ファイルの最新blob SHAを確認する。
- 並行変更があれば再取得する。

## 8.2. 書き込み

Entry commitは対象Entryのファイルだけを含める。

可能ならGit tree/commitを使い、00/10を同一commitへまとめる。

## 8.3. 書き込み後

必ずbase/head比較またはcommit差分を確認し、意図したファイルだけが変更されていることを検証する。

その後control planeを別commitで更新する。

## 8.4. 禁止事項

- repositoryへテスト用・一時ファイルを作らない。
- 存在しないbranchへの試験writeを行わない。
- Entry作業中に無関係なファイルを更新しない。
- 未commit blobを完了扱いしない。
- push後検証前にcontrol planeを完了へ進めない。

# 9. control plane更新規則

1 Entry完了ごとに以下を更新する。

- 当該Entryのtask1〜task5 status
- remarksのcommit SHA
- task別集計
- 必要なら次の実行対象

status値は以下だけを使用する。

```text
未 / 作業中 / 完了 / −
```

reconcile表と実ファイル／commit履歴が食い違う場合、実ファイルと検証済みcommitを確認してreconcile表を修正する。

# 10. 現在位置に関する扱い

進捗の最新状態は本書へ固定しない。必ずcontrol planeを参照する。

本書作成時点では `0137 犬鳴村` までEntry reconcileが完了しており、次対象として `0152 青木ヶ原樹海で方位磁針が狂う` の調査・再分析に着手している。ただし、未commitの作業は完了として扱わない。

# 11. 完了判定

個別Entryは、以下を満たして初めて完了とする。

1. 必要なtask1が完了
2. tutorial準拠が確認済み
3. D01〜D21を固定済みの同一コード体系で再測定
4. 00と10が整合
5. Entry commitがmainへ反映
6. commit差分が対象Entryだけであることを検証
7. control planeを別commitで更新

49件全体のreconcile完了は、さらにtask5〜task7が完了した時点とする。
