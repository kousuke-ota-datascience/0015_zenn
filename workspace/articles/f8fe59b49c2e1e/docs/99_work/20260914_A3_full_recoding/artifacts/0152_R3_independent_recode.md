# 0152 R3 Independent Recode

- Evidence input: `0152_00_contents.md`
- Version Scope: `0152_R2_version_scope.md`
- Old 10: 未参照。R3 freeze後に比較する。

## D01–D21

- D01: codeなし; Status `U`。2007以前にテレビ検証対象となっていたことは分かるが成立年代を固定できない。
- D02: Primary `D02.BRD.TELEVISION`, Parent `D02.BRD`; Status `I`。2002–2006年の番組で俗説が検証対象になったことを複数後代記録が示すが正確な放送日・原映像未固定。
- D03: Primary `D03.BRD.TELEVISION`, Parent `D03.BRD`; Secondary `D03.WEB.WEBSITE`, `D03.WEB.FORUM`; Status `I`。テレビ検証、2007 Web再録、2009公開Q&Aで流通を確認。
- D04: Primary `D04.CON.DEBUNK_RECIRCULATION`, Parent `D04.CON`; Secondary `D04.VAR.ACCRETION`; Status `D/I`。反証とともに再流通し、GPS等へ増補される。
- D05: Primary `D05.PRP.EXPLANATORY_CLAIM`, Parent `D05.PRP`; Status `D`。磁性溶岩→磁針異常→迷うという説明命題。
- D06: `D06.T4` — 一般・科学事実; Status `D`。自然科学的事実であるかのように提示される。
- D07: Primary `D07.PSE.ENVIRONMENTAL_ANOMALY`, Parent `D07.PSE`; Secondary `D07.PSE.SPATIAL_ROUTE_ANOMALY`; Status `I`。不可視の磁気異常と方向判断不能を説明対象化。
- D08: Primary `D08.CLM.UNSUPPORTED_ASSERTION`, Parent `D08.CLM`; Secondary なし; Status `I`。全域で磁針が使えないという一般化が先行命題として流通する。
- D09: Primary `D09.CAU.DIRECT_CAUSE`, Parent `D09.CAU`; Status `I`。道迷いを磁気異常による方位磁針不全へ直接因果化する。
- D10: Primary `D10.NAT.PHYSICAL_ENV_PROCESS`, Parent `D10.NAT`; Secondary `D10.SPC.SPECIFIC_PLACE`; Status `D/I`。磁性を持つ溶岩・樹海環境を因果源とする。
- D11: Primary `D11.MOV.ENTER_PASS_CROSS`, Parent `D11.MOV`; Status `I`。樹海内部で方向判断しようとする状況が発動条件。
- D12: Primary `D12.FOC.PROTAGONIST_EXPERIENCER`, Parent `D12.FOC`; Status `I`。樹海へ入った人が作用対象。
- D13: Primary `D13.PHY.ENV_OBJECT_MANIPULATION`, Parent `D13.PHY`; Status `I`。環境磁気が方位磁針の指示を変える／機能不全化するとされる。
- D14: `D14.NEG`; Status `D`。
- D15: Primary `D15.MND.MEMORY_COGNITION`, Parent `D15.MND`; Status `I`。共有核の終端は方向・帰還経路を判断できない認知的混乱。死亡・永久失踪は固定しない。
- D16: Primary `D16.STA.ENDURING_CONDITION`, Parent `D16.STA`; Status `I`。樹海全域が恒常的に磁針不全を生むという場所属性として提示される。
- D17: Primary `D17.INF.VERIFY_DEBUNK`, Parent `D17.INF`; Status `I`。現地測定・科学検証によって俗説を制御・反証できる。
- D18: `L1=1; L2=0; L3=1`; Status `I`。L1は伝承内因果。L3は大学の現地実測・テレビ検証が当該俗説を明示的検証課題として実行されたX Evidenceに限定する。
- D19: Primary `D19.MAS.NATIONAL_PUBLIC`, Parent `D19.MAS`; Secondary `D19.NET.OPEN_FORUM_WEB`; Status `I`。全国テレビ・公開Web・事典等で広域流通。
- D20: Primary `D20.EXP.MEDICAL_SCIENTIFIC`, Parent `D20.EXP`; Status `D/I`。科学的真偽・局所磁気影響の追加情報は大学研究者等が提示する。
- D21: `D21.A4`; Status `I`。実在樹海・溶岩地質・実測記録を俗説の因果説明と検証へ統合する。

## QA before freeze

- 科学的反証と伝承の存在を混同しない。
- 局所磁気影響を全域異常へ一般化しない。
- GPS等の増補を共有核へ遡及しない。
- H3 Primary exactly 1 / Secondary <=2。
- L3=1は大学・番組による明示的な検証行動に限定し、迷子統計等へ拡張しない。
