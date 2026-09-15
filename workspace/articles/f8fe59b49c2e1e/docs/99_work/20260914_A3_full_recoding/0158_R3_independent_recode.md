# 0158 R3 Independent Recode

- Evidence input: `0158_00_contents.md`
- Version Scope: `0158_R2_version_scope.md`
- Old 10: 未参照。R3 freeze後に比較する。

## D01–D21

- D01: `D01.G0` — 前近代（〜1867）; Status `I`。1820年代の江戸期資料が1803年の出来事として記録する史料系譜を二次資料で確認。
- D02: Primary `D02.PRT.BOOK`, Parent `D02.PRT`; Status `I`。現在安全に固定できる最古流通媒体は『兎園小説』等の江戸期書籍・随筆記録。原本本文未固定のためI。
- D03: Primary `D03.PRT.BOOK`, Parent `D03.PRT`; Secondary `D03.WEB.WEBSITE`; Status `I`。近世印刷資料を基盤とし、後代研究・Web等で再流通。
- D04: Primary `D04.REC.CONTEXT_UPDATE`, Parent `D04.REC`; Secondary `D04.VAR.ACCRETION`; Status `I`。異国・漂着奇談が近現代のUFO・宇宙人概念へ時代適応して再解釈される。
- D05: Primary `D05.HYB.NARRATIVE_EXPLANATION`, Parent `D05.HYB`; Status `I`。漂着→観察→詮議→返送という事件物語と、女性・舟の正体推測が結合する。
- D06: `D06.T6` — 真偽未確定; Status `I`。奇談・記録として提示されるが、史実性・正体は未確定のまま保持される。
- D07: Primary `D07.ANO.UNKNOWN_EXISTENCE`, Parent `D07.ANO`; Secondary `D07.ANO.UNEXPLAINED_EVENT`; Status `D/I`。未知の舟・女性・文字の正体と漂着事件の説明不能性が中心。
- D08: Primary `D08.DEX.DIRECT_EVENT`, Parent `D08.DEX`; Secondary `D08.TRC.TEXT_DOCUMENT`; Status `D/I`。物語内部では村人の直接遭遇、現代受容者には江戸期記録が手掛かり。
- D09: Primary `D09.CAT.TYPE_ASSIGNMENT`, Parent `D09.CAT`; Secondary `D09.UNK.DELIBERATE_NONRESOLUTION`; Status `I`。異国人・未知舟等の既知カテゴリへ仮分類する一方、正体を閉じない。
- D10: Primary `D10.HUM.INDIVIDUAL_HUMAN`, Parent `D10.HUM`; Secondary `D10.OBJ.ARTIFACT_DEVICE`; Status `C`。江戸期核では未知女性と異形舟、後代UFO型では人工物／非人間解釈が競合する。
- D11: Primary `D11.CON.LOCATION_STATE`, Parent `D11.CON`; Status `I`。異形舟が海岸へ漂着して観察可能になる状態が接触条件。
- D12: Primary `D12.FOC.PROTAGONIST_EXPERIENCER`, Parent `D12.FOC`; Status `I`。漂着物を観察・詮議する漁民・村人が焦点作用対象。
- D13: Primary `D13.MAN.MANIFEST_ONLY`, Parent `D13.MAN`; Status `I`。共有核では未知存在が出現・観測されること自体が作用で、安定した攻撃・呪詛はない。
- D14: `D14.NEU`; Status `I`。共有核の中心は遭遇と不明性で、明確な利益・害を必須としない。
- D15: Primary `D15.KNW.UNCERTAINTY_PRESERVED`, Parent `D15.KNW`; Status `D/I`。女性・舟・文字の正体が不明のまま去る。
- D16: Primary `D16.EVT.SEQUENTIAL_EPISODE`, Parent `D16.EVT`; Status `D`。漂着→観察→詮議→返送という時間順の物語。
- D17: Primary `D17.AVO.DISTANCE_ROUTE_CHANGE`, Parent `D17.AVO`; Status `I`。海へ返す型では、未知対象との接触を終わらせ距離を取る対応が結末となる。ただし安定した『回避法』ではないためR4でNAとの境界を監査する。
- D18: `L1=1; L2=0; L3=0`; Status `I`。伝承内の漂着・遭遇因果はある。話を知ること自体は作用条件ではなく、現実社会効果の独立X Evidenceは固定しない。
- D19: Primary `D19.MAS.NATIONAL_PUBLIC`, Parent `D19.MAS`; Secondary `D19.NET.OPEN_FORUM_WEB`; Status `I`。近世書籍・後代出版/オカルト再流通で地域外へ広く流通。
- D20: Primary `D20.UNK.NO_ONE_KNOWS`, Parent `D20.UNK`; Status `I`。舟・女性・文字の決定的真相保持者は伝承内外で固定されない。
- D21: `D21.A4`; Status `I`。1803年という日時、常陸国の地理、江戸期記録、後代解釈を一つの遭遇伝承へ統合する。

## QA before freeze

- 江戸期核とUFO解釈を分離。
- 1803年事件の史実性は確定しない。
- 原資料未固定のD01/D02はI。
- H3 Primary exactly 1 / Secondary <=2。
- L3=0。
