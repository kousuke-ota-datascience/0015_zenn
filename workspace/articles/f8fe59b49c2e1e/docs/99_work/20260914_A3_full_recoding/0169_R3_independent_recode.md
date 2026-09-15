# 0169 R3 Independent Recode

- Evidence input: `0169_00_contents.md`
- Version Scope: `0169_R2_version_scope.md`
- Old 10: 未参照。R3 freeze後に比較する。

## D01–D21

- D01: `D01.G3` — 1970年代; Status `D`。日本の代表的大衆形は1973年刊行書に直接固定できる。
- D02: Primary `D02.PRT.BOOK`, Parent `D02.PRT`; Status `D`。1973年版をNDL書誌で直接確認。
- D03: Primary `D03.PRT.BOOK`, Parent `D03.PRT`; Secondary `D03.BRD.TELEVISION`; Status `D/I`。書籍を直接固定、テレビ増幅は00記述からI。
- D04: Primary `D04.REC.CONTEXT_UPDATE`, Parent `D04.REC`; Secondary `D04.MED.MASS_ADAPTATION`; Status `I`。曖昧な予言詩を時代ごとの危機へ再対応し、出版・テレビ等へ大規模展開。
- D05: Primary `D05.PRP.PREDICTIVE_CLAIM`, Parent `D05.PRP`; Secondary `D05.PRP.EXPLANATORY_CLAIM`; Status `D`。1999年7月の未来破局を予測し、古い詩をその根拠として説明する。
- D06: `D06.T4` — 一般・制度・科学事実; Status `I`。受容時には「実際に予言されている未来事実」であるかのように提示される。真偽が正しいという判定ではない。
- D07: Primary `D07.ISU.FUTURE_SOCIAL_CHANGE`, Parent `D07.ISU`; Secondary `D07.ISU.CRISIS_SCARCITY`; Status `D/I`。未来の人類規模破局と危機不確実性を扱う。
- D08: Primary `D08.TRC.TEXT_DOCUMENT`, Parent `D08.TRC`; Status `D`。予言詩という文書が意味形成の主要手掛かり。
- D09: Primary `D09.PPR.CORRELATION_RULE`, Parent `D09.PPR`; Secondary `D09.HST.HISTORICAL_ANCHOR`; Status `I`。曖昧な詩句と未来の特定年月・危機を対応付ける。
- D10: Primary `D10.OBJ.INFORMATION_CONTENT`, Parent `D10.OBJ`; Secondary `D10.HUM.INDIVIDUAL_HUMAN`; Status `I`。因果源・権威源は予言内容とノストラダムスという歴史人物。
- D11: Primary `D11.INF.READ_VIEW_MEDIA`, Parent `D11.INF`; Status `I`。予言を読む・見ることが受容者の因果モデルへの入口。
- D12: Primary `D12.AUD.GENERAL_PUBLIC`, Parent `D12.AUD`; Status `D/I`。予言の受容対象は人類一般・日本の読者視聴者。
- D13: Primary `D13.COG.MENTAL_INFLUENCE`, Parent `D13.COG`; Secondary `D13.COG.INFORMATION_INDUCED_ACTION`; Status `I`。予言情報が恐怖・不安を誘発し、備え・解釈行動を促す。
- D14: `D14.NEG`; Status `D`。中心帰結は人類規模の破局・滅亡。
- D15: Primary `D15.MND.FEAR_TRAUMA`, Parent `D15.MND`; Secondary `D15.COL.COMMUNITY_CHANGE`; Status `I`。受容者の終末不安と広域文化現象への波及。
- D16: Primary `D16.DLY.DEADLINE`, Parent `D16.DLY`; Status `D`。1999年7月という明示期限が構造の中心。
- D17: codeなし; Status `U`。共有核として安定した回避・解除法を固定できない。
- D18: `L1=1; L2=0; L3=0`; Status `I`。伝承内では未来破局因果がある。聞くこと自体が超常危害条件ではなく、社会行動効果の独立X Evidenceを今回固定していない。
- D19: Primary `D19.MAS.NATIONAL_PUBLIC`, Parent `D19.MAS`; Status `D/I`。1973年出版と後代大衆メディアで全国規模に流通。
- D20: Primary `D20.EXP.RELIGIOUS_FOLKLORE`, Parent `D20.EXP`; Secondary `D20.ORG.INSTITUTION_AUTHORITY`; Status `I`。予言解釈の追加情報は解釈者・出版主体へ偏る。
- D21: `D21.A4`; Status `I`。歴史人物・原詩・1973年出版・1999年という現実時点を予言因果へ統合する。

## QA before freeze

- ノストラダムス原詩と五島解釈を分離。
- 1999年不成就を伝承不存在と混同しない。
- L3=0: 書籍流通自体と独立した社会効果を区別。
- D17=U: 固有の回避法を推測しない。
- H3 Primary exactly 1 / Secondary <=2。
