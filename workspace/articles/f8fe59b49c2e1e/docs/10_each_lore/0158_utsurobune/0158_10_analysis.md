記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は `0158_00_contents.md`、R1〜R4監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0158`
- `伝承エントリ名称`: 虚舟
- `Macro_Category`: 古伝承・怪異遭遇
- `Entry_Type`: 物語・伝説
- `Version_Scope`: 1803年の出来事として常陸国海岸へ異形舟が漂着し、異国風女性・箱・不明文字が観察され、正体不明のまま海へ戻されたとする江戸期奇談を共有核とする。UFO・宇宙人説は後代再文脈化として含めるが初期説明へ遡及しない。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Primary Child / Value: `D01.G0` — 前近代（〜1867）
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 江戸期資料が1803年の出来事として記録した史料系譜を確認できる。原本本文未固定のためI。

### D02 最古確認流通媒体
- Primary Child / Value: `D02.PRT.BOOK` — 書籍
- Primary Parent: `D02.PRT`
- Secondary: なし
- Status: `I`
- 判定根拠: 現在安全に固定できる最古流通層は『兎園小説』等の江戸期随筆・奇談書。原本未固定のためI。

### D03 確認流通媒体ポートフォリオ
- Primary Child / Value: `D03.PRT.BOOK` — 書籍
- Primary Parent: `D03.PRT`
- Secondary: なし
- Status: `I`
- 判定根拠: 近世書籍を基盤に後代出版・研究で再流通する。

### D04 生成・変容パターン
- Primary Child / Value: `D04.REC.CONTEXT_UPDATE` — 時代適応
- Primary Parent: `D04.REC`
- Secondary: `D04.VAR.ACCRETION` — 増補
- Status: `I`
- 判定根拠: 異国・漂着奇談が近現代にUFO・宇宙人という新しい説明枠へ更新される。

### D05 提示形式
- Primary Child / Value: `D05.HYB.NARRATIVE_EXPLANATION` — 物語＋解説
- Primary Parent: `D05.HYB`
- Secondary: なし
- Status: `I`
- 判定根拠: 漂着・観察・詮議・返送という事件物語と正体推測が結合する。

### D06 真実性提示
- Primary Child / Value: `D06.T6` — 真偽未確定
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 奇談・記録として語られるが、出来事の史実性と女性・舟の正体は開かれたまま残る。

## 2.2. 意味形成

### D07 意味形成対象
- Primary Child / Value: `D07.ANO.UNKNOWN_EXISTENCE` — 未知存在
- Primary Parent: `D07.ANO`
- Secondary: `D07.ANO.UNEXPLAINED_EVENT` — 説明不能事象
- Status: `D/I`
- 判定根拠: 未知の舟・女性・文字の正体と漂着事件そのものの説明不能性が中心。

### D08 意味形成契機
- Primary Child / Value: `D08.DEX.DIRECT_EVENT` — 出来事への直接遭遇
- Primary Parent: `D08.DEX`
- Secondary: `D08.TRC.TEXT_DOCUMENT` — 文書・記載
- Status: `D/I`
- 判定根拠: 物語内部では村人の直接遭遇、現代受容者には江戸期記録が手掛かりとなる。

### D09 意味付与操作
- Primary Child / Value: `D09.CAT.TYPE_ASSIGNMENT` — 類型化
- Primary Parent: `D09.CAT`
- Secondary: `D09.UNK.DELIBERATE_NONRESOLUTION` — 非解決維持
- Status: `I`
- 判定根拠: 異国人・未知舟等の既知カテゴリへ仮分類しつつ、最終的な正体は閉じない。

### D10 因果源存在論
- Primary Child / Value: `D10.PHN.ANOMALOUS_EXPERIENCE` — 異常体験
- Primary Parent: `D10.PHN`
- Secondary: `D10.OBJ.ARTIFACT_DEVICE` — 人工物・機器
- Status: `C`
- 判定根拠: 共有核は異常な遭遇現象であり、舟の人工物性や後代UFO解釈をSecondaryで保持する。未知女性を特定存在カテゴリへ固定しない。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary Child / Value: `D11.CON.LOCATION_STATE` — 位置・状態条件
- Primary Parent: `D11.CON`
- Secondary: なし
- Status: `I`
- 判定根拠: 異形舟が海岸へ漂着して観察可能となる状態が接触条件。

### D12 作用対象
- Primary Child / Value: `D12.FOC.PROTAGONIST_EXPERIENCER` — 主人公・体験者
- Primary Parent: `D12.FOC`
- Secondary: なし
- Status: `I`
- 判定根拠: 漂着物を観察・詮議する漁民・村人が焦点人物。

### D13 作用機構
- Primary Child / Value: `D13.MAN.MANIFEST_ONLY` — 顕現のみ
- Primary Parent: `D13.MAN`
- Secondary: なし
- Status: `I`
- 判定根拠: 未知存在が出現・観測されること自体が主作用で、安定した攻撃・呪詛はない。

### D14 帰結極性
- Primary Child / Value: `D14.NEU` — 中立
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 共有核は遭遇と不明性であり、明示的な利益・危害を必須としない。

### D15 帰結領域
- Primary Child / Value: `D15.KNW.UNCERTAINTY_PRESERVED` — 不確実性維持
- Primary Parent: `D15.KNW`
- Secondary: なし
- Status: `I`
- 判定根拠: 女性・舟・文字の正体は不明のまま去る。

### D16 因果時間構造
- Primary Child / Value: `D16.EVT.SINGLE_OBSERVATION` — 単発観測
- Primary Parent: `D16.EVT`
- Secondary: なし
- Status: `I`
- 判定根拠: 主たる怪異は一回の漂着・遭遇イベントとして成立する。

### D17 回避・制御方式
- Primary Child / Value: `D17.NON.OBSERVATIONAL_ONLY` — 観測のみ
- Primary Parent: `D17.NON`
- Secondary: なし
- Status: `I`
- 判定根拠: 共有核に危害解除の固有規則はなく、観察・詮議が中心。海へ返す行為を一般的回避法へ昇格しない。

### D18 作用レイヤー
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `I`
- 判定根拠: 伝承内部の漂着・遭遇因果はあるが、受容自体の作用や独立した社会現実効果は固定できない。

## 2.4. 社会的分布・現実接続

### D19 流通範囲
- Primary Child / Value: `D19.LOC.LOCAL_TRADITION` — 地域伝承圏
- Primary Parent: `D19.LOC`
- Secondary: `D19.MAS.NATIONAL_PUBLIC` — 全国的大衆
- Status: `I`
- 判定根拠: 常陸国の漂着伝承を基盤にしつつ、印刷・研究・オカルト再話で地域外へ流通する。

### D20 特権情報保持者
- Primary Child / Value: `D20.UNK.NO_ONE_KNOWS` — 誰も知らない
- Primary Parent: `D20.UNK`
- Secondary: なし
- Status: `I`
- 判定根拠: 舟・女性・文字の決定的正体を保持する主体は固定されず、謎として継承される。

### D21 現実アンカー
- Primary Child / Value: `D21.A4` — 史実・記録・既存伝承を因果統合
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 1803年という日時、常陸国の地理、江戸期記録、後代解釈が一つの遭遇伝承へ統合される。
