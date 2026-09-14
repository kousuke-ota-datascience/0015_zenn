記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

伝承内容・典拠・異伝・証拠上の不確実性は、同一ディレクトリの `0005_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0005`
- `伝承エントリ名称`: 紫の鏡
- `Macro_Category`: 学校・大学
- `Entry_Type`: 物語・伝説
- `Version_Scope`: **1998–1999年の静岡県・栃木県採録群で直接確認できる「危険語の記憶期限→死亡」型の最小安定共有核**。すなわち、「紫の鏡」という語を知った者が、その語を将来の一定年齢・節目まで忘れずに記憶していると、期限到達時に死亡するとされる記憶期限型の俗信。
- `Scope_Status`: `D`

R2で以下は既定Coding Scopeから除外した。

- 期限を必ず20歳へ固定すること
- 「不幸になる」のみを帰結とする型
- 「紫地蔵」「赤い沼」等の複合危険語
- 「ピンクの鏡」等の対抗語・中和語
- 鏡から血だらけの女が出て殺す人格怪異型
- 実物の鏡を見る・所有する・唱える等の物理的儀式
- 後代の由来譚・起源物語

1995年『みんなの学校の怪談 緑本』は来歴Evidenceとして使用するが、本文を直接実見していないため、1998–1999年採録で確認した具体的死亡規則を1995年へ遡及しない。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示の次元

### 2.1.1. D01: 生成年代

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

1995年12月には公刊記録があるが、刊行年は成立年ではない。1995年以前の起源を安全に年代帯へ写像できる証拠がないため、1990年代へ推測で埋めない。

### 2.1.2. D02: 最古確認流通媒体

- Primary Child / Value: `D02.PRT.BOOK` — 書籍
- Primary Parent: `D02.PRT`
- Secondary: なし
- Status: `D`

**判定根拠**

現在の証拠で年代を固定して直接確認できる最古の実流通媒体は、1995-12-11刊の常光徹『みんなの学校の怪談 緑本』である。同書は「最近のウワサ」として「おぼえてはいけない『紫の鏡』」を収録しているため、それ以前に別の流通があったことは示唆されるが、その媒体を直接特定できない。したがって「学校口承だったはず」と遡及推定せず、最古直接確認媒体を書籍とする。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）

- Primary Child / Value: `D03.ORL.SCHOOL_ORAL` — 学校内伝承
- Primary Parent: `D03.ORL`
- Secondary: なし
- Status: `I`

**判定根拠**

1998–1999年Scopeは高校生・若者から採録された学校怪談・俗信圏の内容であり、学校・若者集団内の対人伝承を受容回路として推定する。研究DB自体は流通媒体に含めない。1995年書籍は具体的死亡規則を本文実見できていないため、Scoped Contentの確認媒体として追加しない。

### 2.1.4. D04: 生成・変容パターン

- Primary Child / Value: `D04.VAR.ORAL_VARIATION` — 口承変異
- Primary Parent: `D04.VAR`
- Secondary: `D04.VAR.ACCRETION` — 増補
- Status: `I`

**判定根拠**

1998–1999年採録群で期限年齢、帰結、複合語、対抗語、人格怪異化が併存することは直接確認できる。ただし、それらが具体的な口承伝達過程で変形・増補された過程自体は直接観察していないためIとする。

### 2.1.5. D05: 提示形式

- Primary Child / Value: `D05.PRP.PREDICTIVE_CLAIM` — 予測命題
- Primary Parent: `D05.PRP`
- Secondary: `D05.RUL.WARNING` — 警告
- Status: `I`

**判定根拠**

「期限まで『紫の鏡』を覚えているなら死亡する」という条件付き未来予測が形式核である。同時に危険語を覚え続けないよう警告する機能を持つが、原発話の完全形を固定できないためIとする。

### 2.1.6. D06: 真実性提示

- Primary Child / Value: `D06.T5` — 条件付き信念
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

「X歳まで覚えていると死ぬ」という条件付き俗信・規則として提示される。共同体で知られていることと、その命題が要求する認識論的スタンスを区別し、`T3`ではなく`T5`とする。

## 2.2. 意味形成の次元

### 2.2.1. D07: 意味形成対象

- Primary Child / Value: `D07.DCF.FATE_OMEN` — 運命・予兆
- Primary Parent: `D07.DCF`
- Secondary: なし
- Status: `I`

**判定根拠**

この伝承がなければ、ある語を記憶しているという現在状態が将来の死をどう予告・決定するのかという規則が残らない。未知存在の正体ではなく、未来の災厄予測が意味形成対象である。

### 2.2.2. D08: 意味形成契機

- Primary Child / Value: `D08.CLM.UNSUPPORTED_ASSERTION` — 根拠未提示の主張
- Primary Parent: `D08.CLM`
- Secondary: `D08.STY.COMMUNITY_REPETITION` — 共同体反復証言
- Status: `I`

**判定根拠**

物的痕跡や事件記録ではなく、「覚えていると死ぬ」という根拠未提示の主張が先に与えられる。学校・若者圏での反復伝達が、その命題を保持する追加契機となる。

### 2.2.3. D09: 意味付与操作

- Primary Child / Value: `D09.PPR.CORRELATION_RULE` — 相関規則化
- Primary Parent: `D09.PPR`
- Secondary: `D09.NOR.TABOOIZATION` — 禁忌化
- Status: `I`

**判定根拠**

「期限時の記憶保持」と「死亡」を条件付き規則として結び、その結果として危険語を覚え続ける状態を禁忌化する。禁忌化だけでなく、X→Yの予測規則形成をPrimaryとする。

### 2.2.4. D10: 因果源存在論

- Primary Child / Value: `D10.OBJ.INFORMATION_CONTENT` — 情報内容
- Primary Parent: `D10.OBJ`
- Secondary: なし
- Status: `I`

**判定根拠**

Scoped Contentでは人格怪異・実物の鏡を必須とせず、「紫の鏡」という情報内容の記憶保持自体が因果条件として働く。独立した呪詛主体を追加推定しない。

## 2.3. 因果・行動モデルの次元

### 2.3.1. D11: 発動・接触条件

- Primary Child / Value: `D11.INF.SAY_NAME_REMEMBER` — 言う・名前を唱える・記憶する
- Primary Parent: `D11.INF`
- Secondary: `D11.CON.AGE_LIFESTAGE` — 年齢・ライフステージ
- Status: `D`

**判定根拠**

危険条件は語を期限まで記憶していること。16・18・20歳等の年齢・ライフステージ条件が期限を形成する。一度聞くだけではなく、期限時までの記憶保持が発動条件である。

### 2.3.2. D12: 作用対象

- Primary Child / Value: `D12.AUD.READER_LISTENER` — 読者・聞き手
- Primary Parent: `D12.AUD`
- Secondary: なし
- Status: `D`

**判定根拠**

D13の認知災害が直接向くのは、伝承を受容して危険語を知り、その記憶を保持する者である。特定の物語主人公に限定されず、現実側の受容者が伝承内容上の作用対象へ入る。

### 2.3.3. D13: 作用機構

- Primary Child / Value: `D13.COG.COGNITION_TRIGGERED_HARM` — 認知災害
- Primary Parent: `D13.COG`
- Secondary: なし
- Status: `D`

**判定根拠**

「紫の鏡」という情報を記憶している認知状態が、期限到達時の致死的作用条件になる。死亡は作用機構ではなくD15の終端帰結として分離する。人格怪異による攻撃や独立した運命固定機構はScopeへ追加しない。

### 2.3.4. D14: 帰結極性

- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `D`

**判定根拠**

Scoped Contentの終端は死亡であり、明確に負である。

### 2.3.5. D15: 帰結領域

- Primary Child / Value: `D15.BOD.DEATH` — 死亡
- Primary Parent: `D15.BOD`
- Secondary: なし
- Status: `D`

**判定根拠**

R2で死亡型の安定共有核へScopeを固定した。複数の1998–1999年採録で死亡が直接示される。「不幸になる」型は別Variationとして00に保持し、同一Scope内のConflictにはしない。

### 2.3.6. D16: 因果時間構造

- Primary Child / Value: `D16.DLY.DEADLINE` — 期限付き
- Primary Parent: `D16.DLY`
- Secondary: なし
- Status: `D`

**判定根拠**

情報曝露直後ではなく、指定年齢・節目まで記憶が残っているかが帰結発生条件となる。期限そのものが構造核である。

### 2.3.7. D17: 回避・制御方式

- Primary Child / Value: `D17.RUL.OBEY_TABOO` — 禁忌遵守
- Primary Parent: `D17.RUL`
- Secondary: なし
- Status: `I`

**判定根拠**

期限前に危険語を忘れ、「覚え続けない」ことが基本的な回避となる。現行taxonomyに「忘れる／記憶を消す」に完全一致するChildがないため、危険な記憶保持を避ける規則として最も近い`OBEY_TABOO`へ暫定写像する。

ただし、これは完全適合ではない。意図的または結果的な**忘却そのものを制御方式とする構造**はtaxonomy gap候補として保持する。

### 2.3.8. D18: 作用レイヤー

- Primary Child / Value: `D18.L1=1; D18.L2=1; D18.L3=0`
- Primary Parent: なし
- Secondary: なし
- Status: `D`

**判定根拠**

- L1: 伝承内で記憶条件→死亡の因果が作動する。
- L2: 伝承を聞いて危険語を知る現実側受容者が、そのまま伝承内容上の将来の作用対象になる。
- L3: R1で噂流通による現実の行動・制度・市場等の外部効果を示すX Evidenceは確認していないため0。

受容行為と因果対象化が接続する自己適用型である。

## 2.4. 社会的埋め込みの次元

### 2.4.1. D19: 流通範囲

- Primary Child / Value: `D19.KIN.SCHOOL_YOUTH` — 学校・若者集団
- Primary Parent: `D19.KIN`
- Secondary: `D19.LOC.REGIONAL` — 地域・地方
- Status: `D`

**判定根拠**

1995年学校怪談集と1998年静岡県・1999年栃木県の若者採録により、学校・若者文化を中心に複数地域へまたがる存在を確認できる。全国的大衆までの広がりを、複数地域で採録されたことだけから推測しない。

### 2.4.2. D20: 特権情報保持者

- Primary Child / Value: `D20.HAZ.DANGEROUS_TO_KNOW` — 知ること自体が危険
- Primary Parent: `D20.HAZ`
- Secondary: なし
- Status: `I`

**判定根拠**

厳密には一度知るだけで即時危害ではなく、知った情報を期限まで保持することが危険である。しかし、情報へのアクセス・保持そのものが受容者を危険条件へ入れるという情報構造は`DANGEROUS_TO_KNOW`に最も近い。単に「皆が知っている」ことをD20の特権情報保持者判定へ流用しない。

### 2.4.3. D21: 現実アンカー

- Primary Child / Value: `D21.A0` — 匿名・抽象
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

学校・若者圏で流通するが、Scoped Contentの因果成立は特定校、地域、実在の鏡、事件、制度を必要としない。「紫の鏡」という危険語と記憶期限だけで別の場所へ移植できる。

# 3. R4 Entry QA

## 3.1. D12 → D13 → D15 causal QA

最終因果列は次のように接続する。

```text
D11: D11.INF.SAY_NAME_REMEMBER
  + Secondary: D11.CON.AGE_LIFESTAGE
→ D12: D12.AUD.READER_LISTENER
→ D13: D13.COG.COGNITION_TRIGGERED_HARM
→ D15: D15.BOD.DEATH
```

- D12は、危険語を受容・保持しD13の認知災害を直接受ける聞き手。
- D13は、記憶という認知状態を危害条件へ変換する作用機構。
- D15は、その終端で変化する生命・身体領域＝死亡。

**QA結果: Pass。**

## 3.2. D18 L3 evidence QA

R1 Evidenceには、紫の鏡の流通によって現実の学校運用、制度、市場、集団行動等が変化したことを確認する`X` Evidenceはない。

恐怖や「忘れようとする行動」が起こりそうだという心理的推測だけでL3を付与しない。

**QA結果: `L3=0` を維持。Pass。**

## 3.3. U / NA / C QA

- `U`: D01のみ。公刊確認年を生成年代へ読み替えないため意図的にU。
- `NA`: なし。
- `C`: なし。

旧10でD15が`C`だったのは死亡型と不幸型を同一Version Scopeへ含めていたためであり、R2で死亡型の安定共有核へScopeを再固定した結果、Conflictは解消した。

**QA結果: 推測によるU埋め、不要なNA/Cなし。Pass。**

## 3.4. taxonomy gap QA

D17について、現行taxonomyは「期限前に忘れる／記憶から失う」ことを直接表現するChildを持たない。

- `D17.AVO.DO_NOT_ENGAGE`: 最初から見ない・入らない・触らない等の接触回避であり、一度知った後の忘却とは異なる。
- `D17.RUL.OBEY_TABOO`: 「危険な状態を維持しない」という規則面では近いが、忘却という認知操作を直接表さない。

R3の`D17.RUL.OBEY_TABOO / I`を暫定値として維持し、**D17忘却制御**をtaxonomy gap候補としてR5横断確認へ送る。単一Entryのみを理由にbaseline taxonomyは変更しない。

旧10も同じ不適合を認識しながら`DO_NOT_ENGAGE`へ近似していたため、今回のgapは単なるR3の偶発的迷いではなく、現行Child集合との適合問題として再現した。

**QA結果: gap候補あり。baseline変更は行わない。**

## 3.5. R3 freezeからR4確定値への変更

R3 checkpoint `74c8f1367a53020c2ff02b3ae5072202ad74e7e9` を現行rulesとEntry QAで再検証した結果、**D01〜D21のコード値・Status変更なし**。

R4では旧10を参照したが、新値を旧値へ一致させる修正は行っていない。

# 4. 再コーディング前 `0005_10_analysis.md` との差分監査

比較対象は、再コーディング前 checkpoint `c2a37792f12b6c2aa5b8760536b13764d14c714d` の旧 `0005_10_analysis.md`。旧ExcelはR4では参照しない。

## 4.1. Version Scope差分

旧Scopeは「死亡／不幸」を同一Scopeへ含め、期限・帰結・併記語の異伝を広く扱っていた。

新Scopeは、1998–1999年の複数採録で安定して直接比較できる**危険語の記憶期限→死亡**型へ限定した。期限年齢差はパラメータ差として残す一方、不幸型、複合語、対抗語、人格怪異化をVariationへ退避した。

このScope変更により、特にD15のConflictが解消し、D13・D17等でも派生要素を混成しない判定になった。

## 4.2. D01〜D21差分分類

| D | 旧値 | R4確定値 | 差分分類 | 判定 |
|---|---|---|---|---|
| D01 | `U` | `U` | same | 1995年公刊資料を追加しても成立年代は安全に固定できない。 |
| D02 | `SCHOOL_ORAL / I` | `BOOK / D` | `Evidence mismatch`, `Code-selection mismatch`, `Status mismatch` | 1995年公刊記録を新たに確認。そこからさらに古い口承媒体を推測せず、最古直接確認媒体を採用。 |
| D03 | `SCHOOL_ORAL / I` | 同左 | same | Scoped Contentの受容回路は学校・若者口承と推定。 |
| D04 | `ORAL_VARIATION + ACCRETION / I` | 同左 | same | 変更なし。 |
| D05 | `TABOO / I` | `PREDICTIVE_CLAIM + WARNING / I` | `Code-selection mismatch` | 提示形式の中心を「XならYになる」という条件付き予測に置き、警告をSecondaryへ。 |
| D06 | `T3 / I` | `T5 / I` | `Code-selection mismatch` | 共同体で知られている事実と、命題が要求する条件付き信念を分離。 |
| D07 | `FATE_OMEN / I` | 同左 | same | 変更なし。 |
| D08 | `UNSUPPORTED_ASSERTION / I` | `UNSUPPORTED_ASSERTION + COMMUNITY_REPETITION / I` | `Code-selection mismatch` | 学校・若者圏での反復を独立Secondaryとして保持。 |
| D09 | `TABOOIZATION / I` | `CORRELATION_RULE + TABOOIZATION / I` | `Code-selection mismatch` | X=期限時記憶保持とY=死亡の規則形成をPrimaryへ。 |
| D10 | `INFORMATION_CONTENT / I` | 同左 | same | 変更なし。 |
| D11 | `SAY_NAME_REMEMBER / D` | `SAY_NAME_REMEMBER + AGE_LIFESTAGE / D` | `Code-selection mismatch` | 年齢・節目が独立した期限条件なのでSecondaryへ明示。 |
| D12 | `PROTAGONIST_EXPERIENCER / I` | `READER_LISTENER / D` | `Code-selection mismatch`, `Status mismatch` | 認知災害の直接対象は特定主人公ではなく、危険情報を受容する聞き手。 |
| D13 | `COGNITION_TRIGGERED_HARM + FATE_FIXING / I` | `COGNITION_TRIGGERED_HARM / D` | `Code-selection mismatch`, `Status mismatch`, `Scope mismatch` | 死亡型Scopeでは記憶条件による危害が直接。独立した運命固定機構を追加しない。 |
| D14 | `NEG / I` | `NEG / D` | `Status mismatch`, `Evidence mismatch` | Scoped Contentで死亡が直接明示される。 |
| D15 | `DEATH + LUCK_MISFORTUNE / C` | `DEATH / D` | `Scope mismatch`, `Status mismatch` | 不幸型をScope外へ分離したためConflict解消。 |
| D16 | `DEADLINE / D` | 同左 | same | 変更なし。 |
| D17 | `DO_NOT_ENGAGE / I` | `OBEY_TABOO / I` | `Taxonomy gap`, `Code-selection mismatch` | 一度知った後の「忘却」は接触回避ではない。OBEY_TABOOへ暫定写像するが専用Child不在。 |
| D18 | `L1=1,L2=1,L3=0 / D` | 同左 | same | 自己適用型のL2を維持。X EvidenceなしのためL3=0。 |
| D19 | `NATIONAL_PUBLIC / I` | `SCHOOL_YOUTH + REGIONAL / D` | `Code-selection mismatch`, `Status mismatch`, `Evidence mismatch` | 複数地域採録から全国大衆を推測せず、直接確認できる若者圏＋地域横断までに留める。 |
| D20 | `COMMON_KNOWLEDGE / I` | `DANGEROUS_TO_KNOW / I` | `Code-selection mismatch` | D20は単なる共有範囲でなく情報保持構造。危険情報の受容・保持自体が因果対象化を生む。 |
| D21 | `A1 / I` | `A0 / I` | `Code-selection mismatch` | 学校は流通文脈であってScoped Contentの成立条件ではない。特定実在背景なしで成立するためA0。 |

# 5. R4結論

- D12→D13→D15 causal QA: **Pass**
- D18 L3 evidence QA: **Pass**
- U / NA / C QA: **Pass**
- taxonomy gap QA: **D17「忘却による制御」を候補として継続 / baseline変更なし**
- Coding正本 `0005_10_analysis.md`: **更新済み**
- 再コーディング前旧10との差分比較: **完了**
- R3 freezeからR4確定値への変更: **なし**
- 旧Excel比較: **R5 Global Reconciliationへ移管**

R4確定後の0005は、1995年の公刊確認を来歴へ反映しつつ、Content Scopeを1998–1999年の死亡型共有核へ限定した状態で、情報内容→記憶期限→受容者への認知災害→死亡という自己適用的因果構造を現行baselineに基づいて確定した。
