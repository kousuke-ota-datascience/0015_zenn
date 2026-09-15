# 0179 R3 Independent Recode — freeze

旧10・旧Excel coding値は参照せず、0179_00と固定baseline codebook/rulesから独立判定した。

| D | Primary | Parent | Secondary | Status | 根拠 |
|---|---|---|---|---|---|
| D01 | `D01.G6` | — | — | `I` | 2001原型〜2003定型を保存転載で確認。 |
| D02 | `D02.WEB.FORUM` | `D02.WEB` | — | `I` | 2001掲示板投稿の保存転載が最古級。 |
| D03 | `D03.WEB.FORUM` | `D03.WEB` | `D03.WEB.WEBSITE` | `D/I` | 掲示板起点・後代Web再話。 |
| D04 | `D04.VAR.ACCRETION` | `D04.VAR` | `D04.VAR.RETELLING` | `D` | 2003投稿者自身が先行話との混成・詳述を明示。 |
| D05 | `D05.EXP.RETROSPECTIVE_1P` | `D05.EXP` | — | `D` | 2003型は幼少期体験の回想として提示。 |
| D06 | `D06.T1` | — | — | `D` | 自身の体験として提示。 |
| D07 | `D07.ANO.UNKNOWN_EXISTENCE` | `D07.ANO` | `D07.ANO.ANOMALOUS_PERCEPTION` | `D/I` | 白い存在の正体と視認経験が中心。 |
| D08 | `D08.DEX.DIRECT_EVENT` | `D08.DEX` | `D08.DEX.PERCEPTUAL_ANOMALY` | `D` | 遠方の異常存在への直接遭遇・視認。 |
| D09 | `D09.UNK.FORBIDDEN_INQUIRY` | `D09.UNK` | `D09.UNK.DELIBERATE_NONRESOLUTION` | `D/I` | 正体を知ること自体を禁じ、最終的に正体を閉じない。 |
| D10 | `D10.PHN.ANOMALOUS_EXPERIENCE` | `D10.PHN` | — | `C` | 白い存在の存在論は人物・妖怪・自然物のいずれにも固定されない。 |
| D11 | `D11.INF.UNDERSTAND_RECOGNIZE` | `D11.INF` | `D11.SEN.VISUAL_EXPOSURE` | `D` | 詳細視認し正体を理解することが危害条件。 |
| D12 | `D12.OTH.SPECIFIC_OTHER` | `D12.OTH` | `D12.FOC.PROTAGONIST_EXPERIENCER` | `D` | 主要被害は兄、語り手も最後に危険へ近づく。 |
| D13 | `D13.COG.COGNITION_TRIGGERED_HARM` | `D13.COG` | `D13.COG.MENTAL_INFLUENCE` | `D/I` | 理解を契機に精神・人格異変が発生。 |
| D14 | `D14.NEG` | — | — | `D` | 回復不能級の異変。 |
| D15 | `D15.MND.SELF_IDENTITY_DISRUPTION` | `D15.MND` | `D15.MND.COMPULSION_BEHAVIOR` | `I` | 兄は以前の人格状態を失い、くねくね動く。 |
| D16 | `D16.EVT.SEQUENTIAL_EPISODE` | `D16.EVT` | — | `D` | 発見→詳細視認→理解→異変が一続きに進む。 |
| D17 | `D17.AVO.DO_NOT_ENGAGE` | `D17.AVO` | — | `D` | 見るな・理解するな、という接触回避が明示。 |
| D18 | `L1=1; L2=0; L3=0` | — | — | `I` | 物語内認知危害のみ。読者や社会への独立作用証拠なし。 |
| D19 | `D19.NET.OPEN_FORUM_WEB` | `D19.NET` | — | `D` | 公開掲示板・Webで流通。 |
| D20 | `D20.INS.LOCAL_RESIDENT` | `D20.INS` | — | `D` | 祖父が「見てはならない」という追加知識を保持。 |
| D21 | `D21.A1` | — | — | `I` | 秋田・田園は現実背景だが特定実在地点への固定は弱い。 |

## freeze

本ファイルcommitをR3 SHAとする。ここから旧10の有無・内容を確認する。
