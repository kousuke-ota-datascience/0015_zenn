# 0137 R3 Independent Recode

- Evidence input: `0137_00_contents.md`
- Version Scope: `0137_R2_version_scope.md`
- Old 10: comparison禁止。R3 freeze後に解禁。
- Procedural note: コードブック検索時に旧10のD07断片が検索UIに偶発表示されたため、D07のみ完全blindではない。下記D07は00本文と固定コードブック定義から再裁定し、R4で明示監査する。

## D01–D21

- D01: codeなし; Status `U`。1999年は早期確認点であり成立年ではない。
- D02: Primary `D02.WEB.FORUM`, Parent `D02.WEB`; Status `I`。研究論文・保存記録が1999年の公開Web掲示板層を示すが原ログ未固定。
- D03: Primary `D03.WEB.FORUM`, Parent `D03.WEB`; Status `I`。公開Webでの流通を固定。後代映画・書籍は再提示Evidenceだが、正確なD03媒体ChildはR4でコードブック照合する。
- D04: Primary `D04.VAR.ACCRETION`, Parent `D04.VAR`; Secondary `D04.MED.CROSS_MEDIA`; Status `I`。秘密村核へ実在地・事件・後代メディア要素が増補される。
- D05: Primary `D05.HRS.FOAF`, Parent `D05.HRS`; Status `D`。知人の体験として現実性を付与する。
- D06: `D06.T2` — 近接伝聞事実; Status `D`。
- D07: Primary `D07.PSE.HIDDEN_VANISHED_PLACE`, Parent `D07.PSE`; Secondary `D07.INO.SYSTEM_CAPACITY_FAILURE`; Status `I`。隠された村と通常制度が届かない不安が中心。
- D08: Primary `D08.STY.FOAF_REPORT`, Parent `D08.STY`; Status `D/I`。近接伝聞が存在主張の契機。
- D09: Primary `D09.SSI.GROUP_BOUNDARY`, Parent `D09.SSI`; Secondary `D09.SSI.HIDDEN_STRUCTURE`; Status `I`。内外集団境界と通常社会の外部構造として説明する。
- D10: Primary `D10.HUM.INFORMAL_GROUP`, Parent `D10.HUM`; Status `D`。因果源は排他的な村人共同体。
- D11: Primary `D11.MOV.ENTER_PASS_CROSS`, Parent `D11.MOV`; Status `D`。細道・境界の先へ進入することが危険系への入口。
- D12: Primary `D12.FOC.PROTAGONIST_EXPERIENCER`, Parent `D12.FOC`; Status `I`。外来者・探索者が作用対象。
- D13: Primary `D13.PHY.PHYSICAL_ATTACK`, Parent `D13.PHY`; Status `D`。追跡・威嚇・暴行という通常物理的作用。
- D14: `D14.NEG`; Status `D`。
- D15: Primary `D15.BOD.SEVERE_INJURY`, Parent `D15.BOD`; Status `I`。共有核では外来者への暴力・生命身体危険が終端。死亡まで固定しない。
- D16: Primary `D16.EVT.SEQUENTIAL_EPISODE`, Parent `D16.EVT`; Status `I`。境界進入→制度不全→遭遇→追跡・攻撃という段階進行。
- D17: Primary `D17.AVO.DO_NOT_ENGAGE`, Parent `D17.AVO`; Secondary `D17.AVO.FLEE_ESCAPE`; Status `I`。最善は入らないこと、侵入後は逃走。
- D18: `L1=1; L2=0; L3=0`; Status `I`。伝承内因果はあるが、噂が現実制度・集団行動を変えた独立X Evidenceは固定しない。
- D19: Primary `D19.NET.OPEN_FORUM_WEB`, Parent `D19.NET`; Secondary `D19.MAS.NATIONAL_PUBLIC`; Status `I`。公開Web起点の広域再流通。
- D20: codeなし; Status `U`。村の「真相」を持つ特権主体をEvidenceから固定できない。
- D21: `D21.A4`; Status `I`。実在地名・旧集落・トンネル・ダム等の現実断片を秘密村因果へ統合する。

## QA before freeze

- U/NA/Cをコード値へ混入していない。
- H3 Primary exactly 1、Secondary 0–2。
- 実在旧集落と秘密村を同一視していない。
- L3は0。映画の社会影響を伝承効力のX Evidenceへ転用しない。
- D07の手続汚染はR4で再点検対象。
