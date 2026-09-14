記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

伝承内容・典拠・異伝・証拠上の不確実性は、同一ディレクトリの `0001_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0001`
- `伝承エントリ名称`: 口裂け女
- `Macro_Category`: 全国型怪異・儀式
- `Entry_Type`: 物語・伝説
- `Version_Scope`: **1979年初頭〜同年春に確認できる最小安定共有核**。日常社会の中に「口が裂けた女性（口裂け女）」が出現し、主として小中学生・子どもを脅かし／襲うとされる遭遇・危害型の噂を分析対象とする。

R2で、以下は最初期共有核へ安全に遡及できないためScope外とした。

- 「私、きれい？」という定型質問
- マスクを外して口を見せる二段階の正体提示
- 赤いコート
- 鎌・鋏等の特定武器
- 人間離れした高速走行
- ポマード
- べっこう飴
- 「普通」「まあまあ」等の特定回答による回避
- 整形失敗・交通事故等の由来説明
- 個別地域固有の禁忌・弱点・逃走規則

Scope Statusは `I`。1979年1月26日付『岐阜日日新聞』の記事存在は国立国会図書館の実見レファレンスで確認できる一方、本文全文を今回直接精査していない。1979年春の同時代雑誌見出しと専門家による変容史整理から、後代要素を遡及しない最小核を保守的に固定した。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示の次元

### 2.1.1. D01: 生成年代

**いつ成立したか**

- Primary Child / Value: `D01.G3` — 1970年代
- Primary Parent: なし
- Secondary: なし
- Status: `I` — Inferred

**判定根拠**

`0001_00_contents.md` では、1978年暮頃の岐阜で子どもの噂として流通していたという後代専門家整理と、1979年1月26日付『岐阜日日新聞』「編集余記」の同時代記録を確認した。成立年代帯は1970年代へ安全に写像できるが、1978年暮れという開始時点自体は同時代一次資料から直接固定していない。

**この伝承における現れ方**

口裂け女は1970年代末の子ども社会で急速に可視化された現代伝承として扱う。新聞掲載日を起源日とはしない。

### 2.1.2. D02: 最古確認流通媒体

**現在の証拠で、伝承が実際に人から人へ流通したことを最も古く確認できる媒体は何か**

- Primary Child / Value: `D02.ORL.PEER_ORAL` — 友人・仲間内伝承
- Primary Parent: `D02.ORL` — 口承・限定共同体
- Secondary: なし
- Status: `I` — Inferred

**判定根拠**

飯倉義之の専門家整理は、1978年暮れの岐阜で子ども同士の噂として流通し、反復の中で細部が付加されたことを明示する。現行coding rulesも、口裂け女についてこの種の後代専門家整理から `D02.ORL.PEER_ORAL` を `I/D` とし得る例を明示している。

**この伝承における現れ方**

現時点で最も古く確認可能な実流通回路は、新聞そのものではなく子ども・同世代間の対人口承として再構成される。ただし原採録ではないためStatusはIとする。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）

**今回コードするVersion Scopeで、伝承が受容者へ流通したことを確認できる媒体は何か**

- Primary Child / Value: `D03.ORL.PEER_ORAL` — 友人・仲間内伝承
- Primary Parent: `D03.ORL` — 口承・限定共同体
- Secondary: `D03.PRT.MAGAZINE` — 雑誌; `D03.PRT.NEWSPAPER` — 新聞
- Status: `I` — Inferred

**判定根拠**

Scopeの成立・変容に子ども同士の口頭伝達が構造的に重要である。一方、1979年1月には新聞、3〜4月には全国誌で記事化されており、印刷媒体での再提示もScope内に確認できる。

**この伝承における現れ方**

子ども同士の口承を中心に噂が反復され、新聞・雑誌がその存在と拡大を広域社会へ再提示する複合媒体構造を持つ。

### 2.1.4. D04: 生成・変容パターン

**時間とともにどう変形したか**

- Primary Child / Value: `D04.VAR.ACCRETION` — 増補
- Primary Parent: `D04.VAR` — 再話・変異
- Secondary: `D04.MIG.MEDIUM_SHIFT` — 媒体移行; `D04.VAR.ORAL_VARIATION` — 口承変異
- Status: `I` — Inferred

**判定根拠**

専門家整理では、子ども間の反復によりマスク、赤いコート、鎌、高速走行、ポマード、べっこう飴等の新属性・新規則が追加された。さらに口承から新聞・雑誌へ流通媒体が移行したことも同時代書誌で追跡できる。

**この伝承における現れ方**

単なる言い換えより、新しい能力・弱点・ルールを付け足す「増補」が特徴的であり、それに口承変異と媒体移行が重なる。

### 2.1.5. D05: 提示形式

**どんなコミュニケーション形式で提示されるか**

- Primary Child / Value: `D05.HRS.SCHOOL_WORK_HEARSAY` — 学校・職場伝承
- Primary Parent: `D05.HRS` — 伝聞叙述
- Secondary: `D05.PRP.FACT_CLAIM` — 事実主張
- Status: `I` — Inferred

**判定根拠**

Scope内では、子ども・学校・塾を含む同世代ネットワークで共有される噂としての提示が主要である。また1979年4月の雑誌見出しは「口のさけた女性が小・中学生を襲う」という事実主張型命題として再提示している。

**この伝承における現れ方**

特定の完成した一人称怪談ではなく、「子ども社会でそう言われている」という伝聞と、危険な女性が存在するという主張が結びつく。

### 2.1.6. D06: 真実性提示

**どんな「本当らしさ」を要求するか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U` — Unknown

**判定根拠**

同時代新聞・雑誌は噂を報道対象としているが、それは報道側のメタ的フレーミングである。受容者間の原発話・採録本文を固定できていないため、`共同体既知事実`、`条件付き信念`、`真偽未確定`のどの認識論的スタンスで伝えられたかを一意に決めない。

**この伝承における現れ方**

「噂だった」という研究者側の整理を、そのまま伝承自身の真実性提示へ変換しない。

## 2.2. 意味形成の次元

### 2.2.1. D07: 意味形成対象

**何が不可解・不確実なのか**

- Primary Child / Value: `D07.ICT.STRANGER_THREAT` — 見知らぬ他者の脅威
- Primary Parent: `D07.ICT` — 対人・犯罪脅威
- Secondary: `D07.ICT.ABDUCTION_ASSAULT` — 誘拐・暴行
- Status: `I` — Inferred

**判定根拠**

R2 Scopeで確実に保持できるのは、日常空間に現れる見知らぬ女性が子どもを脅かし／襲うという構造である。幽霊・妖怪・超常能力は最小核に固定しない。

**この伝承における現れ方**

日常空間で遭遇し得る見知らぬ他者への危険を、「口裂け女」という具体的な脅威モデルへ変換する。

### 2.2.2. D08: 意味形成契機

**何を手掛かりに問題化されるか**

- Primary Child / Value: `D08.STY.COMMUNITY_REPETITION` — 共同体反復証言
- Primary Parent: `D08.STY` — 社会的証言
- Secondary: `D08.CLM.UNSUPPORTED_ASSERTION` — 根拠未提示の主張
- Status: `I` — Inferred

**判定根拠**

子ども集団で反復して語られる噂が問題化の主要な手掛かりであり、初期Scopeでは物的痕跡や公式記録より、根拠未提示の共同体反復が先行する。

**この伝承における現れ方**

「誰かが見たらしい」「近くに出るらしい」と共同体内で繰り返されること自体が、通常の街路や見知らぬ女性を危険として再解釈させる。

### 2.2.3. D09: 意味付与操作

**不可解なものをどう理解可能にするか**

- Primary Child / Value: `D09.AGN.AGENCY_ATTRIBUTION` — 主体化
- Primary Parent: `D09.AGN` — 主体・意図帰属
- Secondary: `D09.CAT.NAMING` — 命名; `D09.CAU.DIRECT_CAUSE` — 直接原因化
- Status: `I` — Inferred

**判定根拠**

曖昧な対人危険を、名称を持つ一人の女性主体「口裂け女」へ集約し、その主体が子どもへの襲撃を起こすと説明する。

**この伝承における現れ方**

危険を無名の不安のまま残さず、識別可能な主体へ変換することで、共同体内で共有・警戒可能なモデルにする。

### 2.2.4. D10: 因果源存在論

**原因を何として世界に置くか**

- Primary Child / Value: `D10.HUM.INDIVIDUAL_HUMAN` — 個人
- Primary Parent: `D10.HUM` — 人間・社会主体
- Secondary: なし
- Status: `I` — Inferred

**判定根拠**

R2の最小Scopeは「口が裂けた女性」を因果主体とするが、その主体が幽霊・妖怪であることや超自然能力を持つことまでは同時代初期Evidenceから固定しない。強い存在論を追加せず、保守的に個人としてコードする。

**この伝承における現れ方**

危険は抽象的な呪い・場所ではなく、遭遇可能な女性個人に帰属される。後代の妖怪化はScope外。

## 2.3. 因果・行動モデルの次元

### 2.3.1. D11: 発動・接触条件

**何を契機に因果系へ入るか**

- Primary Child / Value: `D11.PAS.SPONTANEOUS_SELECTION` — 偶然選ばれる・遭遇する
- Primary Parent: `D11.PAS` — 受動発生
- Secondary: なし
- Status: `I` — Inferred

**判定根拠**

Scope内では、特定の質問への回答・儀式・場所進入などを行えば再現的に発動するという規則は固定できない。子どもが本人の選択によらず危険主体に遭遇する構造を採る。

**この伝承における現れ方**

日常生活の中で偶然に口裂け女へ遭遇することが因果系への入口となる。

### 2.3.2. D12: 作用対象

**誰／何に作用するか**

- Primary Child / Value: `D12.GRP.DEMOGRAPHIC_GROUP` — 属性集団
- Primary Parent: `D12.GRP` — 集団・共同体
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

1979年4月『週刊新潮』記事見出しが「口のさけた女性が小・中学生を襲う」と明示する。D13の主作用である「襲撃」の直接対象は小中学生・子どもという属性集団である。

**この伝承における現れ方**

個々の主人公という物語上の焦点ではなく、子どもという属性群が危険対象として一般化される。

### 2.3.3. D13: 作用機構

**因果源が対象へ何をするか**

- Primary Child / Value: `D13.PHY.PHYSICAL_ATTACK` — 物理攻撃
- Primary Parent: `D13.PHY` — 身体・物質作用
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

1979年4月『週刊新潮』見出しが「小・中学生を襲う」と直接示す。武器、切創方法、死亡等の詳細はScope内Evidenceから補わない。

**この伝承における現れ方**

口裂け女は子どもへ危害を加える攻撃主体として提示される。追跡は後代代表形では重要だが、最小Scopeでは主作用として固定しない。

### 2.3.4. D14: 帰結極性

**結果は正・負・中立・混合か**

- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

Scope内の主作用が子どもへの襲撃・脅威であり、正の利益を含まない。

**この伝承における現れ方**

遭遇は危険な出来事としてのみ意味付けられる。

### 2.3.5. D15: 帰結領域

**何の領域が最終的に変わるか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U` — Unknown

**判定根拠**

「襲う」という作用は同時代Evidenceで確認できるが、最終的に重傷、死亡、失踪、単なる逃走等のどの終端へ至るかはR2 Scopeで固定できない。D13の物理攻撃を、そのままD15の重傷・死亡へ読み替えない。

**この伝承における現れ方**

危険の存在は確定しているが、被害の終端領域は初期Evidence上未確定のまま保持する。

### 2.3.6. D16: 因果時間構造

**発動条件・原因成立から主作用／主帰結までの関係は、時間上どのように編成されているか**

- Primary Child / Value: `D16.EVT.SINGLE_EPISODE` — 単一エピソード
- Primary Parent: `D16.EVT` — 単一エピソード・事象内
- Secondary: なし
- Status: `I` — Inferred

**判定根拠**

偶然の遭遇から襲撃までが一つの連続した遭遇事象として編成される。即時性そのものが独立した意味上のルールであること、遅延・周期・長期進行があることはScope内で確認しない。

**この伝承における現れ方**

一度の遭遇エピソード内で危険が成立する。

### 2.3.7. D17: 回避・制御方式

**結果をどう回避・制御・利用できるか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U` — Unknown

**判定根拠**

ポマード、べっこう飴、特定回答等の回避規則は流行過程で確認されるが、R2で最小Scope外とした。初期共有核に有効な回避法が存在したとも、回避不能とも断定しない。

**この伝承における現れ方**

後代の豊富な攻略法を初期形へ遡及せず、制御可能性は不明として保持する。

### 2.3.8. D18: 作用レイヤー

**因果効力は伝承内／受容者／社会現実のどこに及ぶか**

- Primary Child / Value: `D18.L1=1; D18.L2=0; D18.L3=1`
- Primary Parent: なし
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

- L1: 伝承内部で口裂け女が子どもへ襲撃作用を持つ。
- L2: 噂を聞く・読むこと自体が伝承内の発動条件ではないため0。
- L3: 1979年6月15日『秋田魁新報』記事見出しに「子供の登校拒否騒ぎも」とあり、噂流通に伴う現実行動上の効果を示すX Evidenceを同時代資料として確認した。

**この伝承における現れ方**

物語内の襲撃因果に加え、現実社会でも子どもの行動変化が報道対象化するほどの効果を持った。

## 2.4. 社会的分布・情報構造・現実接続の次元

### 2.4.1. D19: 流通範囲

**誰の間に伝承が流通するか**

- Primary Child / Value: `D19.KIN.SCHOOL_YOUTH` — 学校・若者集団
- Primary Parent: `D19.KIN` — 家族・仲間
- Secondary: `D19.LOC.REGIONAL` — 地域・地方
- Status: `D` — Direct

**判定根拠**

小中学生・子どもが主要受容集団であることは同時代雑誌見出しで直接確認できる。1979年3月には京都への到達、4月には「各地」とする記事見出しがあり、春段階で地域を越えた広がりも確認できる。6月末の「全国」はR2 Scope後続の流通史として保持する。

**この伝承における現れ方**

全国一般大衆より先に、学校・若者集団を中核として複数地域へ広がる。

### 2.4.2. D20: 特権情報保持者

**誰が真相・追加情報を持つか**

- Primary Child / Value: `D20.NON.NO_HIDDEN_TRUTH` — 隠れた真相なし
- Primary Parent: `D20.NON` — 特権なし
- Secondary: なし
- Status: `I` — Inferred

**判定根拠**

最小Scopeは「口の裂けた女性が子どもを襲う」という共有主張で成立し、特定の専門家・内部者・権威だけが真相や有効な回避情報を保持する構造を要求しない。後代の弱点・攻略知識はScope外。

**この伝承における現れ方**

伝承の中核情報は共同体で共有される噂であり、隠れた情報保持者の存在を前提にしない。

### 2.4.3. D21: 現実アンカー

**実在世界へどの程度固定されるか**

- Primary Child / Value: `D21.A1` — 一般的現実背景
- Primary Parent: なし
- Secondary: なし
- Status: `I` — Inferred

**判定根拠**

流通史は岐阜、京都等の実在地域に接続するが、Scoped Content自体は特定地点・制度・史実を因果構造へ組み込まなくても成立する。学校・街路・子ども・女性という一般的現実背景への依存が中心である。

**この伝承における現れ方**

特定の一地点に固定されないため、別地域の日常空間へ移植できる。

# 3. R4 Entry QA

## 3.1. D12 → D13 → D15 因果列QA

```text
D12: D12.GRP.DEMOGRAPHIC_GROUP
（小中学生・子ども）
→ D13: D13.PHY.PHYSICAL_ATTACK
（その子どもを襲う）
→ D15: U
（攻撃後の最終帰結領域はScope内Evidence不足）
```

D12とD13は直接接続する。D15を重傷・死亡等で推測補完しないため、因果列として矛盾はない。

## 3.2. taxonomy gap QA

現行Scopeを表現できない安定構造は確認しない。

D19について、全国規模の広がりと「小中学生」という人口属性を同時表現する課題はあるが、本Scopeは1979年春までであり、`SCHOOL_YOUTH + REGIONAL` で表現可能である。本Entry単独を理由にtaxonomy変更候補とはしない。

## 3.3. D18 L3 QA

L3を1にするためのX Evidenceとして、1979年6月15日『秋田魁新報』記事見出し「子供の登校拒否騒ぎも」を確認済み。単なる「起こりそう」という推測ではない。

パトロール・集団下校は専門家による後代整理として確認しているが、個別同時代一次記録を固定していないため、L3の直接根拠には使わない。

## 3.4. U / NA / C QA

- D06=`U`: 原発話・採録本文不足。報道側の「噂」「風説」を伝承自身の真実性提示へ転用しない。
- D15=`U`: 襲撃の最終結果不足。
- D17=`U`: 回避法はR2 Scope外。
- `NA` は使用しない。各次元は構造上適用可能だが、上記3次元はEvidence不足である。
- `C` は使用しない。R2で競合異伝を最小Scopeから除外したため、同一Scope内部の競合として保持すべき次元はない。

## 3.5. 旧 `0001_10_analysis.md` との差分分類

独立R3完了後に旧10を参照した。旧10は「質問→裂けた口の提示→追跡・危害／回避」までを含む広いScopeであり、R2の最小安定共有核より後代要素を多く含んでいた。

| D | 旧値の要点 | 新値の要点 | 分類 | 判定 |
|---|---|---|---|---|
| D01 | G3 / D | G3 / I | `Status mismatch` | 年代帯は同じ。1978年暮れ開始を直接Evidence扱いしないためIへ変更。 |
| D02 | U | PEER_ORAL / I | `Evidence mismatch` / `Prior coding error` | 現行rulesが許容する専門家H EvidenceをR1で再確認。 |
| D03 | PEER_ORAL + NEWSPAPER + MAGAZINE | PEER_ORAL + MAGAZINE + NEWSPAPER | 実質一致 | Secondaryは順不同。保存時のみcode ID順にcanonicalize。 |
| D04 | ORAL_VARIATION + ACCRETION | ACCRETION + MEDIUM_SHIFT + ORAL_VARIATION | `Code-selection mismatch` | 新ルール追加が構造変化の中心で、媒体移行Evidenceも追加。 |
| D05 | WARNING | SCHOOL_WORK_HEARSAY + FACT_CLAIM | `Scope mismatch` / `Code-selection mismatch` | 旧値は後代の回避・警告構造を含む。 |
| D06 | T6 / I | U | `Status mismatch` / `Evidence mismatch` | 報道側の「噂」表現を受容者の認識論的スタンスへ転用しない。 |
| D07 | UNKNOWN_EXISTENCE | STRANGER_THREAT + ABDUCTION_ASSAULT | `Scope mismatch` | 超自然・未知存在性を最小核へ固定しない。 |
| D08 | UNSUPPORTED_ASSERTION | COMMUNITY_REPETITION + UNSUPPORTED_ASSERTION | `Code-selection mismatch` | R1で子ども集団の反復流通Evidenceを明確化。 |
| D09 | AVOIDANCE_RULE | AGENCY_ATTRIBUTION + NAMING + DIRECT_CAUSE | `Scope mismatch` | 回避規則はR2 Scope外。 |
| D10 | YOKAI_ENTITY | INDIVIDUAL_HUMAN | `Scope mismatch` | 超自然存在論を初期核へ遡及しない。 |
| D11 | MEET_INTERACT | SPONTANEOUS_SELECTION | `Scope mismatch` / `Code-selection mismatch` | 定型質問・会話規則はScope外。日常遭遇は受動発生として扱う。 |
| D12 | PROTAGONIST_EXPERIENCER | DEMOGRAPHIC_GROUP | `Code-selection mismatch` | 現行D12規則に従い、D13主作用の直接対象「小中学生」をコード。 |
| D13 | PURSUIT + PHYSICAL_ATTACK | PHYSICAL_ATTACK | `Scope mismatch` | 追跡は最小同時代Evidenceで固定せず、「襲う」を直接採用。 |
| D14 | NEG / I | NEG / D | `Status mismatch` | 「襲う」という同時代直接Evidenceで負を直接支持。 |
| D15 | SEVERE_INJURY + DEATH | U | `Scope mismatch` / `Status mismatch` | 傷害・死亡を最小核へ遡及しない。 |
| D16 | IMMEDIATE | SINGLE_EPISODE | `Scope mismatch` / `Code-selection mismatch` | 定型質問直後という即時性を外し、一遭遇内完結のみ保持。 |
| D17 | CORRECT_ANSWER | U | `Scope mismatch` / `Status mismatch` | 正答・ポマード等はScope外。 |
| D18 | L1=1,L2=0,L3=0 | L1=1,L2=0,L3=1 | `Evidence mismatch` / `Prior coding error` | R1で同時代X Evidenceを追加固定。 |
| D19 | NATIONAL_PUBLIC | SCHOOL_YOUTH + REGIONAL | `Scope mismatch` | 旧値は6月末の全国化まで含む。新Scopeは春まで。 |
| D20 | COMMON_KNOWLEDGE | NO_HIDDEN_TRUTH | `Scope mismatch` / `Code-selection mismatch` | 後代の弱点共同知を除き、最小核の情報構造を再評価。 |
| D21 | A1 | A1 | 一致 | 変更なし。 |

旧値との一致を目的にせず、R1 EvidenceとR2 Scopeに基づく新判定を正とする。

## 3.6. 正本Excel旧値との比較

**未検証。** GitHub接続上の正本Excelは `.xlsx` バイナリであり、現セッションのGitHubコネクタではbase64断片の取得までは可能だが、ワークブックとしてローカルへ取得・展開する手段がない。このため、旧Excel行を推測して旧10と同一とみなすことはしない。

この未検証事項はR4の完了条件に影響する。旧Excel値の比較が完了するまで、control plane上のR4は `保留` とする。

# 4. 現時点の検証結果

- R1 Evidence / 00監査: 完了
- R2 Version Scope: 完了
- R3 21次元独立再コード: 完了
- R4 D12→D13→D15因果列QA: 完了
- R4 taxonomy gap QA: 完了
- R4 D18 L3 X Evidence確認: 完了
- R4 旧10差分比較・分類: 完了
- R4 正本Excel旧値比較: **未検証**
- `00_contents` と本10のR1〜R3新判定整合: 確認済み

したがって、R4は現時点で `保留` とする。Excel旧値の取得・照合後に完了へ変更する。
