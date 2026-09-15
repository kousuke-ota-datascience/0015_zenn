# 0091 ベッドの下の男 — R3 Independent Recode

- Entry_ID: `0091`
- Version Scope正本: `docs/99_work/20260914_A3_full_recoding/0091_R2_version_scope.md`
- Evidence正本: `docs/10_each_lore/0091_man_under_bed/0091_00_contents.md`
- R1監査補助: `docs/99_work/20260914_A3_full_recoding/0091_R1_evidence_audit.md`
- baseline: control plane記載の固定blob
- 旧 `0091_10_analysis.md`: **未参照**
- 旧Excel coding値: **未参照**

# 1. 独立判定

- D01: `D01.G5` (1990年代), Status `I`。後代資料が1990年代の日本流通を明示し1996年採録が確認できるが、成立年そのものは直接固定できないためI。
- D02: `D02.PRT.BOOK`, Parent `D02.PRT`, Status `D`。直接確認できる最古媒体は1996年書籍。
- D03: Primary `D03.PRT.BOOK`, Parent `D03.PRT`, Secondaryなし, Status `D`。噂・漫画媒体は後代整理のみなのでポートフォリオへ推測追加しない。
- D04: Primary `D04.VAR.RETELLING`, Parent `D04.VAR`, Secondary `D04.VAR.LOCALIZATION`, Status `I`。自宅／ホテル、発見者、刃物等を変えつつ核を保持。
- D05: Primary `D05.HRS.FOAF`, Parent `D05.HRS`, Secondary `D05.RUL.WARNING`, Status `I`。近接伝聞の犯罪談として提示され、防犯警告へ接続。
- D06: `D06.T2`, Status `I`。知人周辺の実話らしい近接伝聞。
- D07: Primary `D07.ICT.STRANGER_THREAT`, Parent `D07.ICT`, Secondary `D07.ICT.ABDUCTION_ASSAULT`, Status `D`。安全な私室へ見知らぬ加害者が潜伏することが中心不安。
- D08: Primary `D08.STY.FOAF_REPORT`, Parent `D08.STY`, Secondary `D08.TRC.PHYSICAL_TRACE`, Status `I`。受容者にはFOAF報告が契機、物語内では物の位置・鏡・気配等が察知手掛かりになり得る。
- D09: Primary `D09.AGN.AGENCY_ATTRIBUTION`, Parent `D09.AGN`, Secondary `D09.NOR.TABOOIZATION`, Status `I`。私室の違和感・危険を侵入者という意図主体へ帰属し、一人で確認しない規範へ変換。
- D10: Primary `D10.HUM.INDIVIDUAL_HUMAN`, Parent `D10.HUM`, Secondaryなし, Status `D`。因果源は人間の侵入者。
- D11: Primary `D11.PAS.SPONTANEOUS_SELECTION`, Parent `D11.PAS`, Secondary `D11.CON.LOCATION_STATE`, Status `I`。帰宅・入室自体が怪異の発動操作ではなく、侵入者が対象を選び私室へ先行潜伏している状態が入口。
- D12: Primary `D12.OTH.VICTIM_TARGET`, Parent `D12.OTH`, Secondaryなし, Status `D`。直接標的は居住者／宿泊者。
- D13: Primary `D13.REL.TARGETING`, Parent `D13.REL`, Secondary `D13.SOC.COERCE_CONFINE`, Status `I`。侵入者は特定の居住者を標的に潜伏し、襲撃機会を待つ。監禁・強制は異伝により潜在的であるためSecondary。
- D14: `D14.NEG`, Status `I`。実害回避型でも、中心因果は殺傷・暴行の切迫した負の危険として提示される。
- D15: Primary `D15.BEH.AVOIDANCE_ROUTE_CHANGE`, Parent `D15.BEH`, Secondary `D15.KNW.REVELATION_KNOWLEDGE`, Status `I`。実現した終端は退室・危険回避と侵入者存在の確認であり、未遂の殺害を実現帰結としてはコードしない。
- D16: Primary `D16.EVT.SEQUENTIAL_EPISODE`, Parent `D16.EVT`, Secondaryなし, Status `D`。入室→違和感／第三者察知→退避→確認という一エピソード内の段階進行。
- D17: Primary `D17.AVO.FLEE_ESCAPE`, Parent `D17.AVO`, Secondary `D17.RIT.LEGAL_OFFICIAL`, Status `I`。直接対決せず退避し、警察等へ確認を委ねる型が代表的。
- D18: `L1=1, L2=0, L3=0`, Status `D`。伝承内の人間危険のみ。受容自体は危害条件でなく、現実社会効果X Evidenceなし。
- D19: Primary / Parent / Secondaryなし, Status `U`。1990年代に日本で流行という二次整理はあるが、個別話の社会範囲をtaxonomy上の一範囲へ安全に固定する独立Evidenceが不足。
- D20: Primary / Parent / Secondaryなし, Status `U`。発見者・警察・侵入者等で情報保持構造が異伝により変わり、共有核で一意に固定できない。
- D21: `D21.A1`, Status `D`。自宅・ホテル・ベッド・警察という一般的現実背景で成立し、特定施設・事件へ依存しない。

# 2. causal precheck

```text
D11: 日常生活中に侵入者から標的化される
→ D12: 居住者／宿泊者が被害標的となる
→ D13: 侵入者が同一私室に潜伏し標的化を維持する
→ D15: 危険察知後に退避し、侵入者存在を確認する
```

実現帰結と未遂危害を分離して接続する。

# 3. D18 L3 precheck

防犯意識を高めた可能性はあるが、独立X Evidence未確認のため `L3=0`。

# 4. U / NA / C precheck

D19=`U`、D20=`U`。NA/Cなし。

# 5. taxonomy gap precheck

「物理的に隠れて潜伏する」行為を直接表すD13 Childがない。現行Primaryは標的化で近似するが、**D13 潜伏・隠匿（physical hiding / ambush）**はGlobal Reconciliationのtaxonomy gap候補として保持する。

# 6. R3 freeze

本ファイルは旧 `0091_10_analysis.md` および旧Excel coding値を参照せず作成した。以後のR4で初めて旧10を参照する。