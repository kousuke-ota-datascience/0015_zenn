# 0101 海外旅行で臓器を抜かれる — R3 Independent Recode

- Entry_ID: `0101`
- Version Scope正本: `docs/99_work/20260914_A3_full_recoding/0101_R2_version_scope.md`
- Evidence正本: `docs/10_each_lore/0101_organ_theft_travel/0101_00_contents.md`
- R1監査補助: `docs/99_work/20260914_A3_full_recoding/0101_R1_evidence_audit.md`
- baseline: control plane記載の固定blob
- 旧 `0101_10_analysis.md`: **未参照**
- 旧Excel coding値: **未参照**

# 1. 独立判定

- D01: `D01.G5` — 1990年代, Status `D`。成人旅行者kidney-heist型を1991年11月の同時代新聞で直接確認。
- D02: `D02.PRT.NEWSPAPER`, Parent `D02.PRT`, Status `D`。現在固定できる最古の実流通媒体は1991年新聞。
- D03: Primary `D03.PRT.NEWSPAPER`, Parent `D03.PRT`; Secondary `D03.EDG.BBS`, `D03.EDG.EMAIL`; Status `D/I`。1991年新聞を直接確認し、1990年代後半のnewsgroup・email再流通を研究・同時代報道で確認。
- D04: Primary `D04.MIG.DIGITAL_REAMPLIFICATION`, Parent `D04.MIG`; Secondary `D04.VAR.LOCALIZATION`, `D04.VAR.RETELLING`; Status `D/I`。既存のkidney-heist型が1990年代後半に初期ネットで急増し、都市・被害者属性を差し替えて再話される。
- D05: Primary `D05.HRS.FOAF`, Parent `D05.HRS`; Secondary `D05.RUL.WARNING`; Status `D`。友人・同僚等の実話らしい近接伝聞と旅行者への警告。
- D06: `D06.T2` — 近接伝聞事実, Status `D/I`。1991年読者投稿・1997年chain型とも身近な被害として提示。
- D07: Primary `D07.ICT.HIDDEN_CRIMINAL_PRACTICE`, Parent `D07.ICT`; Secondary `D07.ICT.ABDUCTION_ASSAULT`; Status `D`。表から見えない臓器摘出犯罪ネットワークと身体暴力が中心。
- D08: Primary `D08.STY.FOAF_REPORT`, Parent `D08.STY`; Secondary `D08.TRC.PHYSICAL_TRACE`; Status `D/I`。受容者にはFOAF報告、物語内被害者には手術痕・チューブ等の身体痕跡が問題化契機。
- D09: Primary `D09.AGN.AGENCY_ATTRIBUTION`, Parent `D09.AGN`; Secondary `D09.CAU.HIDDEN_CAUSE`, `D09.NOR.TABOOIZATION`; Status `I`。意識空白と身体痕跡を秘密の犯罪者主体へ帰属し、旅行先での不審な社交を回避すべき行為へ変換。
- D10: Primary `D10.HUM.INFORMAL_GROUP`, Parent `D10.HUM`; Secondaryなし; Status `I`。組織像は曖昧だが、複数の人間犯罪者・闇ネットワークが想定される。正式組織とは断定しない。
- D11: Primary `D11.PAS.SPONTANEOUS_SELECTION`, Parent `D11.PAS`; Secondary `D11.SOC.MEET_INTERACT`; Status `I`。旅行者が加害者に標的化されることが入口で、会話・飲食は接触手段。
- D12: Primary `D12.OTH.VICTIM_TARGET`, Parent `D12.OTH`; Secondaryなし; Status `D`。直接作用対象は旅行者本人。
- D13: Primary `D13.PHY.BODY_TRANSFORMATION`, Parent `D13.PHY`; Secondary `D13.SOC.COERCE_CONFINE`, `D13.PHY.PHYSIOLOGICAL_CHANGE`; Status `D/I`。身体内部から臓器を除去する不可逆的改変が主作用。意識を奪い拘束する過程と機能損失をSecondary。
- D14: `D14.NEG`, Status `D`。重大身体被害・犯罪被害。
- D15: Primary `D15.BOD.SEVERE_INJURY`, Parent `D15.BOD`; Secondary `D15.LIF.FUTURE_CONSTRAINT`; Status `D`。臓器喪失による重大障害と長期健康制約。
- D16: Primary `D16.EVT.SEQUENTIAL_EPISODE`, Parent `D16.EVT`; Secondary `D16.DLY.DELAYED`; Status `D`。接触→意識喪失→摘出→覚醒→被害確認という一エピソード内の段階進行で、被害認識は時間差で生じる。
- D17: Primary `D17.AVO.DO_NOT_ENGAGE`, Parent `D17.AVO`; Secondary `D17.RIT.MEDICAL_PROFESSIONAL`; Status `I`。警告としては不審な接触・飲食を避ける。被害後は医療確認を要する。
- D18: `L1=1, L2=0, L3=0`, Status `D`。物語内部に犯罪因果はあるが、受容自体は危害条件でない。十分な独立X Evidenceを固定していない。
- D19: Primary `D19.MAS.TRANSNATIONAL`, Parent `D19.MAS`; Secondary `D19.MAS.NATIONAL_PUBLIC`; Status `D/I`。成人kidney-heist型は複数国・都市へローカライズされ初期ネットで広域流通。日本語圏でも後代事典へ収録。
- D20: Primary `D20.ORG.PERPETRATOR_CRIMINAL`, Parent `D20.ORG`; Secondary `D20.EXP.MEDICAL_SCIENTIFIC`; Status `I`。犯罪過程の全貌は加害側が保持し、被害者の臓器喪失は医療者が確認する型がある。
- D21: `D21.A3` — 実在制度・社会史が成立条件, Status `I`。臓器移植・臓器不足・闇市場という実在の医療制度／社会問題が、伝承のもっともらしさと因果モデルの成立に必要。ただし特定実事件へは依存しない。

# 2. causal precheck

```text
D11: 旅行先で犯罪者に選ばれ、接触する
→ D12: 旅行者本人が被害標的となる
→ D13: 意識喪失中に臓器を摘出され身体が改変される
→ D15: 重大身体障害と長期健康制約へ至る
```

接続は成立する。

# 3. D18 L3 precheck

同時代報道には個人の旅行不安反応や移植関係者の懸念があるが、Entry単位で安定した社会効果を示す独立X Evidenceとしては不足するため `L3=0`。

# 4. U / NA / C precheck

U/NA/Cなし。D03/D04/D06/D13/D19は資料横断再構成を含むためstatusにIを含める。

# 5. taxonomy gap precheck

現行Scopeは既存taxonomyで表現可能。新Child要求なし。

# 6. R3 freeze

本ファイルは旧 `0101_10_analysis.md` および旧Excel coding値を参照せず作成した独立判定である。以後のR4で初めて旧10を参照する。