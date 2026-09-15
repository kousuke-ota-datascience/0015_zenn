# 0178 R3 Independent Recode — freeze

旧10・旧Excel coding値は参照せず、0178_00と固定baseline codebook/rulesから独立判定した。

| D | Primary | Parent | Secondary | Status | 根拠 |
|---|---|---|---|---|---|
| D01 | `D01.G6` 2000年代 | — | — | `I` | 保存転載が2000-08-02投稿を示す。原ページ未固定のためI。 |
| D02 | `D02.WEB.FORUM` | `D02.WEB` | — | `I` | 最古級記録は匿名掲示板投稿。保存転載経由。 |
| D03 | `D03.WEB.FORUM` | `D03.WEB` | `D03.WEB.WEBSITE` | `D/I` | 掲示板起点、後代Web保存・再話を確認。 |
| D04 | `D04.MIG.DIGITAL_REAMPLIFICATION` | `D04.MIG` | `D04.VAR.ACCRETION` | `I` | ネット上で再増幅され、2003年以降に読者感染型が増補。 |
| D05 | `D05.EXP.RETROSPECTIVE_1P` | `D05.EXP` | — | `D` | 「私は、夢をみていました」という一人称回想。 |
| D06 | `D06.T1` | — | — | `D` | 語り手自身の経験として提示。真実性の外部検証とは別。 |
| D07 | `D07.ANO.DREAM_SLEEP_ANOMALY` | `D07.ANO` | `D07.ANO.UNEXPLAINED_EVENT` | `D/I` | 反復し継続する異常夢が中心。 |
| D08 | `D08.DEX.DREAM_SLEEP_EVENT` | `D08.DEX` | — | `D` | 意味形成の契機は語り手が直接経験する夢。 |
| D09 | `D09.UNK.DELIBERATE_NONRESOLUTION` | `D09.UNK` | — | `I` | 原因・主体を最終確定せず、夢の異常性を未解決のまま保持。 |
| D10 | `D10.PHN.ANOMALOUS_EXPERIENCE` | `D10.PHN` | — | `I` | 因果源を外部の特定主体より異常夢現象に置く。 |
| D11 | `D11.CON.LOCATION_STATE` | `D11.CON` | — | `I` | 睡眠・夢状態に入っていることが因果系の前提。 |
| D12 | `D12.FOC.PROTAGONIST_EXPERIENCER` | `D12.FOC` | — | `D` | 主作用対象は夢を見ている投稿者。 |
| D13 | `D13.PHY.PHYSICAL_ATTACK` | `D13.PHY` | `D13.COG.MENTAL_INFLUENCE` | `I` | 夢内で順次身体危害が示され、投稿者には恐怖・脅威が作用。 |
| D14 | `D14.NEG` | — | — | `D` | 残酷な危害・追跡的脅威。 |
| D15 | `D15.MND.FEAR_TRAUMA` | `D15.MND` | `D15.LIF.FUTURE_CONSTRAINT` | `I` | 生還する中心人物の共有終端は恐怖と「次回は逃がさない」という将来制約。 |
| D16 | `D16.REC.TRIGGERED_RECURRENCE` | `D16.REC` | — | `D` | 年月を置いて同じ夢が続きから再発する。 |
| D17 | `D17.AVO.FLEE_ESCAPE` | `D17.AVO` | — | `I` | 夢だと認識し、覚醒することで危害直前に脱出する。物理逃走とは異なるためtaxonomy適合は要Global QA。 |
| D18 | `L1=1; L2=0; L3=0` | — | — | `I` | 原型内部の夢因果のみ。読者感染型はScope外、独立X Evidenceなし。 |
| D19 | `D19.NET.OPEN_FORUM_WEB` | `D19.NET` | — | `D` | 公開匿名掲示板を起点とする。 |
| D20 | `D20.UNK.NO_ONE_KNOWS` | `D20.UNK` | — | `I` | 原型内で夢の真相を確定できる特権保持者はいない。 |
| D21 | `D21.A1` | — | — | `D` | 無人駅・列車等は一般的現実背景で、特定実在地点へ固定されない。 |

## freeze

本ファイルのcommitをR3 SHAとする。R3 freeze後にのみ旧10の有無を確認する。
