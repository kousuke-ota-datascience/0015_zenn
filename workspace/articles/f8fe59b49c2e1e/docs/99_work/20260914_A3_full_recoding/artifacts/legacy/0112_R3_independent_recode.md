# 0112 事故物件は一度別人を住ませれば告知義務が消える — R3 Independent Recode

- Entry_ID: `0112`
- Version Scope正本: `docs/99_work/20260914_A3_full_recoding/0112_R2_version_scope.md`
- Evidence正本: `docs/10_each_lore/0112_accident_property_disclosure/0112_00_contents.md`
- R1監査補助: `docs/99_work/20260914_A3_full_recoding/0112_R1_evidence_audit.md`
- baseline: control plane記載の固定blob
- 旧 `0112_10_analysis.md`: **未参照**
- 旧Excel coding値: **未参照**

# 1. 独立判定

- D01: Valueなし, Status `U`。2008年の公開Web確認点はあるが、成立年代は固定できない。
- D02: `D02.WEB.FORUM`, Parent `D02.WEB`, Status `D`。現時点の最古確認媒体は2008年Yahoo!知恵袋の公開Q&A。
- D03: Primary `D03.WEB.FORUM`, Parent `D03.WEB`; Secondaryなし; Status `D`。公開掲示板・Q&Aで実流通を確認。後代の反証記事はEvidence媒体であって必ずしも俗説の主流通媒体とは数えない。
- D04: Primary `D04.CON.DEBUNK_RECIRCULATION`, Parent `D04.CON`; Secondary `D04.VAR.RETELLING`; Status `D/I`。反証・制度解説の文脈で俗説自体が繰り返し再可視化される。
- D05: Primary `D05.PRP.PREDICTIVE_CLAIM`, Parent `D05.PRP`; Secondary `D05.HYB.QA_EXPERT`; Status `D`。『一人挟めば次は告知不要になる』という条件命題が核で、Q&A形式でも流通。
- D06: `D06.T4` — 一般・制度・科学事実, Status `D`。俗説は法・業界の実ルールであるかのように提示される。真偽判定とは分離する。
- D07: Primary `D07.INO.HIDDEN_RULE_PROCEDURE`, Parent `D07.INO`; Secondary `D07.INO.OCCUPATIONAL_HIDDEN_PRACTICE`; Status `D`。不動産取引の見えないルール／抜け道を説明することが中心。
- D08: Primary `D08.CLM.UNSUPPORTED_ASSERTION`, Parent `D08.CLM`; Secondary `D08.OPA.OPAQUE_RULE`; Status `D/I`。根拠未提示の制度主張と、一般人から見えにくい告知判断の複雑さが意味形成契機。
- D09: Primary `D09.SSI.SYSTEM_LOGIC`, Parent `D09.SSI`; Secondary `D09.SSI.HIDDEN_STRUCTURE`; Status `I`。中間入居を制度状態をリセットする内部ロジックとして説明し、表面下の抜け道構造を仮定する。
- D10: Primary `D10.HUM.ORGANIZATION`, Parent `D10.HUM`; Secondaryなし; Status `I`。伝承内の因果源は、不動産取引・告知を運用する制度／業界主体として擬人化された組織的ルール。
- D11: Primary `D11.MAN.CREATE_ALTER_MANIPULATE`, Parent `D11.MAN`; Secondary `D11.SOC.CONTRACT_EXCHANGE`; Status `D`。中間入居者を入れ、契約・入居履歴を一段挟む操作が発動条件。
- D12: Primary `D12.ORG.INSTITUTION_SYSTEM`, Parent `D12.ORG`; Secondaryなし; Status `D`。D13の制度操作が直接作用すると主張される対象は告知義務・取引ルールという制度状態。
- D13: Primary `D13.SOC.INSTITUTIONAL_MANIPULATION`, Parent `D13.SOC`; Secondaryなし; Status `D`。中間入居によって制度上の告知要否を操作できるという主張。
- D14: `D14.MIX`, Status `I`。俗説上は貸主・売主側に取引上の利得、次の借主・買主側には情報不利益が併存する。
- D15: Primary `D15.BEH.COMPLIANCE_DECISION`, Parent `D15.BEH`; Secondary `D15.MAT.MONEY_GAIN_LOSS`; Status `I`。告知する／しないという遵守判断が変わり、それが取引条件・価格等の経済的利害へ接続し得る。
- D16: Primary `D16.STA.STATIC_RULE`, Parent `D16.STA`; Secondaryなし; Status `D`。『一人挟めば次は不要』という条件規則であり、経時進行自体は主題でない。
- D17: Primary `D17.INF.VERIFY_DEBUNK`, Parent `D17.INF`; Secondary `D17.RIT.LEGAL_OFFICIAL`; Status `D`。回避・制御は公式ガイドライン・専門情報で制度主張を検証し、必要なら公的・法的専門経路へ確認すること。
- D18: `L1=1, L2=0, L3=0`, Status `D`。俗説内容内では中間入居→制度状態変更という因果がある。聞くこと自体は作用条件でなく、実市場効果を示す独立X Evidenceは未固定。
- D19: Primary `D19.NET.OPEN_FORUM_WEB`, Parent `D19.NET`; Secondaryなし; Status `D`。公開Q&A・掲示板等で流通を直接確認。全国的大衆への浸透は推測で追加しない。
- D20: Primary `D20.ORG.INSTITUTION_AUTHORITY`, Parent `D20.ORG`; Secondary `D20.EXP.TECHNICAL_OCCUPATIONAL`; Status `I`。正確な告知判断については行政・業界制度主体と不動産実務専門家が特権情報保持者となる。
- D21: `D21.A3` — 実在制度・社会史が成立条件, Status `D`。宅建業者の告知実務・国交省ガイドライン等の実在制度が伝承の成立条件。

# 2. causal precheck

```text
D11: 中間入居者を契約・入居させる
→ D12: 告知義務という制度状態が対象になる
→ D13: 制度状態を『リセットできる』とされる
→ D15: 次回取引で告知不要という遵守判断・経済利害へつながる
```

伝承内因果として接続する。現行制度がこの因果を支持するわけではない。

# 3. D18 L3 precheck

俗説を実際に利用した市場慣行の規模を示す独立X Evidenceを固定していないため `L3=0`。

# 4. U / NA / C precheck

- D01=`U`: 成立年代未確定。
- NA/Cなし。

# 5. taxonomy gap precheck

現行Scopeは既存taxonomyで表現可能。新Child要求なし。

# 6. R3 freeze

本ファイルは旧 `0112_10_analysis.md` および旧Excel coding値を参照せず作成した独立判定である。以後のR4で初めて旧10を参照する。