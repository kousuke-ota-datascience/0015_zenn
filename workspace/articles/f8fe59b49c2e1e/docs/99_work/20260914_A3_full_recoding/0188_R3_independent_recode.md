# 0188 R3 Independent Recode

## 前提
- Entry_ID: `0188`
- 伝承: 八尺様
- pre-SHA: `4d762489c220d89837df9edf603d84780b2195f9`
- Version Scope: 2008-08-26 レス908〜916
- 旧10はR3 freeze前に参照しない。

## 独立コーディング

| D | Primary | Parent | Secondary | Status | 根拠 |
|---|---|---|---|---|---|
| D01 | `D01.G6` | — | — | D | 2008年投稿。 |
| D02 | `D02.WEB.FORUM` | `D02.WEB` | — | D | 2ちゃんねる掲示板。 |
| D03 | `D03.WEB.FORUM` | `D03.WEB` | — | D | 原型Scopeは掲示板投稿。 |
| D04 | `D04.STB.SINGLE_FIXED` | `D04.STB` | — | D | 9レスに分割された一続きの完成回想譚。 |
| D05 | `D05.EXP.RETROSPECTIVE_1P` | `D05.EXP` | — | D | 十年以上前の自身の高校時代体験を一人称回想。 |
| D06 | `D06.T1` | — | — | D | 自身の体験として提示。 |
| D07 | `D07.ANO.UNKNOWN_EXISTENCE` | `D07.ANO` | `D07.DCF.UNEXPLAINED_DEATH_LOSS` | D/I | 八尺様という未知存在の正体と、魅入られた者の死が意味形成対象。 |
| D08 | `D08.DEX.DIRECT_EVENT` | `D08.DEX` | `D08.DEX.PERCEPTUAL_ANOMALY` | D | 異常に背の高い女性姿・奇妙な声への直接遭遇と知覚。 |
| D09 | `D09.CAT.TYPE_ASSIGNMENT` | `D09.CAT` | `D09.AGN.AGENCY_ATTRIBUTION` | D/I | 祖母が遭遇を「八尺様」と同定し、過去の死・現在の追跡を同存在の作用へ帰属。 |
| D10 | `D10.SUP.YOKAI_ENTITY` | `D10.SUP` | — | I | 女性姿、意思、声模倣、追跡を行う人格的地域怪異として提示。幽霊・死者霊までは特定しない。 |
| D11 | `D11.SEN.VISUAL_EXPOSURE` | `D11.SEN` | `D11.CON.LOCATION_STATE` | D/I | 主人公は地域内で八尺様を視認後に「魅入られた」とされる。地域内滞在が成立条件。 |
| D12 | `D12.FOC.PROTAGONIST_EXPERIENCER` | `D12.FOC` | — | D | 主人公が追跡・誘引の直接対象。 |
| D13 | `D13.REL.TARGETING` | `D13.REL` | — | D/I | 八尺様が特定の若者を魅入り、声を模倣し、地域外脱出まで追跡する。 |
| D14 | `D14.NEG` | — | — | I | 死亡危険、恐怖、地域帰還不能。 |
| D15 | `D15.LIF.FUTURE_CONSTRAINT` | `D15.LIF` | `D15.BEH.AVOIDANCE_ROUTE_CHANGE` | D | 主人公は生存するが祖父母の地域へ戻れず、葬儀にも参加できない。 |
| D16 | `D16.PRG.STAGED_PROGRESSION` | `D16.PRG` | — | D/I | 視認→魅入られ判定→夜間接近→翌朝脱出→長期回避→封印破壊後日談と段階的に進む。 |
| D17 | `D17.RUL.OBEY_TABOO` | `D17.RUL` | `D17.RIT.RELIGIOUS_SPECIALIST` | D | 部屋を出ない・声に応じない・札を持つ・目を開けない等の明示手順を守り、Kの祈祷を受けて脱出。 |
| D18 | `L1=1 / L2=0 / L3=0` | — | — | D/I/I | 怪異→主人公の物語内作用。読者感染規則なし。独立社会現実X Evidenceなし。 |
| D19 | `D19.NET.OPEN_FORUM_WEB` | `D19.NET` | — | D | 公開Web掲示板。 |
| D20 | `D20.INS.LOCAL_RESIDENT` | `D20.INS` | `D20.EXP.RELIGIOUS_FOLKLORE` | D/I | 祖父母・地域血縁者が八尺様の規則を知り、Kが儀礼専門知を持つ。 |
| D21 | `D21.A1` | — | — | I | 農村、祖父母宅等の一般的現実背景はあるが、舞台市町村は匿名化され、特定実在対象へ固定できない。 |

## QA
- H3 Primary exactly 1、Secondary 0〜1、Parent derived。
- 視認自体を危害結果とせず、D11の入口とD13の標的化を分離。
- 主人公は死亡していないためD15を死亡にせず、確定した長期生活制約を採用。
- 物語内部の地蔵・村協定をD18 L3の外部Evidenceへ流用しない。
