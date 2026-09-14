記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0011_00_contents.md`。本ファイルをCoding正本とする。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `11`
- `伝承エントリ名称`: こっくりさん
- `Macro_Category`: 全国型怪異・儀式
- `Entry_Type`: 儀式伝承
- `Version_Scope`: **1886–1887年に直接確認できる明治器具型の最小安定核。複数人がテーブルまたは飯櫃の蓋等の可動器具へ手を置き、問いを発し、参加者が意図的に動かしていないと感じる器具の動きを、外部的・霊的な主体からの意味ある回答として読む占い／神意判断の実践。**
- Scope外: 現代紙・硬貨型、学校型禁忌、憑依・祟り・終了規則、1970年代学校流行を明治期内容へ混入すること。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01: 生成年代

- Value: `D01.G1` — 1868–1944
- Status: `I`

1884年頃の伝来・1886年頃の各地流行という東洋大学公式整理、1886年新聞、1887年井上円了著作から明治期成立層へ安全に写像できる。ただし1884年を唯一の起源年とは断定しない。

### D02: 最古確認流通媒体

- Primary: `D02.PRT.NEWSPAPER`
- Parent: `D02.PRT`
- Status: `D`

現在直接固定できる最古の公共流通媒体は1886年7月16日『朝日新聞（大阪）』「神下しと狐狗狸の拘引」。実践自体が記事以前に存在したことと最古確認媒体は区別する。

### D03: 確認流通媒体ポートフォリオ

- Primary: `D03.PRT.NEWSPAPER`
- Secondary: `D03.PRT.BOOK`
- Status: `D`

Scope期間に1886年新聞と1887年『妖怪玄談 狐狗狸の事』を確認できる。後代の学校口承はScope媒体へ入れない。

### D04: 生成・変容パターン

- Primary: なし
- Status: `U`

西洋のテーブル・ターニングが日本で飯櫃の蓋・竹脚等へ適応された変化自体はEvidenceで確認できる。しかし現行 `D04.VAR.LOCALIZATION` は地名・施設・人物の地域置換を定義しており、**器具・実践形式の文化的／物質的適応**には直接一致しない。R3での `LOCALIZATION / I` は撤回し、taxonomy gap候補として保留する。

### D05: 提示形式

- Primary: `D05.RUL.RITUAL_PROCEDURE`
- Parent: `D05.RUL`
- Status: `I`

複数人が器具へ手を置き、問いを発し、動きを回答として読む再現可能な占い手順として実践される。

### D06: 真実性提示

- Value: `D06.T5` — 条件付き信念
- Status: `I`

「この実践を行えば器具の動きから回答・神意を得られる」という実践的信念を要求する。研究者による現象説明とは分離する。

## 2.2. 意味形成

### D07: 意味形成対象

- Primary: `D07.ISU.INFORMATION_VOID`
- Secondary: `D07.DCF.FATE_OMEN`
- Parent: `D07.ISU`
- Status: `I`

この実践がなければ、問いへの未知の回答、特に未来・吉凶に関する情報が空白のまま残る。器具運動そのものの説明不能性だけでなく、実践目的である未知情報取得を中心質問に置く。

### D08: 意味形成契機

- Primary: `D08.DEX.DIRECT_EVENT`
- Parent: `D08.DEX`
- Status: `I`

参加者が意図的に動かしていないと感じる器具が実際に動くという**異常出来事への直接遭遇**が意味形成を開始する。R3の `PERCEPTUAL_ANOMALY` は、知覚の異常より出来事自体が中心であるためQAで修正した。

### D09: 意味付与操作

- Primary: `D09.AGN.AGENCY_ATTRIBUTION`
- Secondary: `D09.PPR.OMEN_FORECAST`
- Parent: `D09.AGN`
- Status: `I`

器具運動を偶然・無主体現象ではなく、外部の意思ある主体からの応答とみなす。得られた回答を未来・吉凶の予測へ用いる構造をSecondaryとする。

### D10: 因果源存在論

- Primary: なし
- Candidate A: `D10.SUP.DEITY_DIVINE`
- Candidate B: `D10.SUP.YOKAI_ENTITY`
- Status: `C`

明治期Evidenceでは神意判断という枠組みと「狐狗狸」等の人格的外部主体の表現が併存する。一方、現代の死者霊解釈を明治期へ遡及できない。現Evidenceでは単一Childへ保守的に固定しない。

## 2.3. 因果・行動モデル

### D11: 発動・接触条件

- Primary: `D11.MAN.PERFORM_RITUAL`
- Parent: `D11.MAN`
- Status: `D`

複数人が器具へ接触し問いを発する実践そのものが再現可能な発動条件である。

### D12: 作用対象

- Primary: `D12.OTD.OBJECT_PRODUCT`
- Parent: `D12.OTD`
- Status: `I`

D13主作用が直接向けられる対象は参加者ではなく、テーブル・飯櫃の蓋等の可動器具である。

### D13: 作用機構

- Primary: `D13.PHY.ENV_OBJECT_MANIPULATION`
- Parent: `D13.PHY`
- Status: `I`

伝承内部では外部主体が器具を動かし、その運動が回答として解釈される。

### D14: 帰結極性

- Value: `D14.NEU`
- Status: `I`

明治期Scopeの必須帰結は回答取得であり、その内容は利益・危害のいずれにも固定されない。後代の霊障・祟りを混入しない。

### D15: 帰結領域

- Primary: `D15.KNW.REVELATION_KNOWLEDGE`
- Parent: `D15.KNW`
- Status: `I`

実践の中心的終端は、問いに対する回答・未知情報を得たと理解することである。

### D16: 因果時間構造

- Primary: `D16.EVT.SEQUENTIAL_EPISODE`
- Parent: `D16.EVT`
- Status: `I`

接触・質問→器具移動→回答解釈という一つの実践内の段階列で成立する。

### D17: 回避・制御方式

- Primary: `D17.USE.DELIBERATE_INVOCATION`
- Parent: `D17.USE`
- Status: `I`

明治期Scopeでは危害回避より、外部主体を意図的に呼び出して未知情報を得る利用構造が中心。後代の終了禁忌はScope外。

### D18: 作用レイヤー

- `D18.L1=1`
- `D18.L2=0`
- `D18.L3=1`
- Status: `I`

L1では外部主体→器具移動→回答という伝承内因果がある。L2は、伝承を読む／聞くだけで受容者が作用対象になる規則がないため0。L3は1886年同時代新聞見出し「神下しと狐狗狸の拘引」が実践に対する現実側の社会的対応を直接示すため1とする。ただし本文全体未実見なのでStatusはIとし、拘引の詳細を推測しない。

## 2.4. 社会的埋め込み

### D19: 流通範囲

- Primary: `D19.MAS.NATIONAL_PUBLIC`
- Parent: `D19.MAS`
- Status: `I`

東洋大学公式整理が1886年頃に「日本各地で大流行」とする。単一地域内部に限定されない広域流行として扱うが、当時の人口認知率を直接測定していないためI。

### D20: 特権情報保持者

- Primary: なし
- Status: `U` — **taxonomy gap暫定表現**

実践では、参加者が知らない回答・神意を「こっくりさん／外部主体」が保持し、器具運動を通じて提示する構造がある。ところが現行D20は、人間・家族・内部者・専門家・組織・到達不能・危険情報等のみで、**超自然的主体が特権情報を保持する構造**を表すChildを持たない。

ここでの`U`はEvidence不足を意味する通常のUとは異なり、現taxonomyではコード化不能なための暫定プレースホルダーである。R5で他Entryにも同型が出るか横断確認し、再発する場合はD20 Child追加を検討する。

### D21: 現実アンカー

- Value: `D21.A1` — 一般的現実背景
- Status: `I`

日常的な家庭・器具・複数人の実践を用いるが、伝承内容自体が特定の唯一地点・人物・制度・史実を成立条件としない。成立史の具体的資料と内容アンカーを分離する。

# 3. R4 Entry QA

## 3.1. D12 → D13 → D15 causal QA

```text
D12: テーブル・飯櫃の蓋等の可動器具
→ D13: 外部主体が器具を動かす
→ D15: 参加者がその動きから問いへの回答・知識を得る
```

**結果: Pass。** D15の受益者が参加者であっても、D12はD13主作用の直接対象である器具を取るため因果列は整合する。

## 3.2. D18 L3 evidence QA

1886年『朝日新聞（大阪）』の見出しに「拘引」が明示され、実践が現実社会側の対応対象となったことは確認できる。本文未実見なので対象・原因・人数等は推測しない。

**結果: `L3=1 / I`を維持。Pass。**

## 3.3. U / NA / C QA

- D04=`U`: 現行Childへ無理に写像せず、器具・実践形式の文化的／物質的適応をtaxonomy gap候補として保持。
- D10=`C`: Scope内の因果主体表現を一意に固定できないため妥当。
- D20=`U`: 通常のEvidence不足Uではなくtaxonomy gap暫定表現。明示注記を付す。
- NA: なし。

**結果: Pass。**

## 3.4. taxonomy gap QA

R5横断確認へ送る候補は2点。

1. **D04: cultural/material adaptation** — 外来実践を地域の日常器具・実践形式へ置換する変容。現行 `LOCALIZATION` は地名・施設・人物置換に限定される。
2. **D20: supernatural privileged information holder** — 神格・霊・怪異等が、人間参加者の知らない情報を保持し回答する構造。

単一Entryでbaseline taxonomyは変更しない。

**結果: Pass。baseline変更なし。**

## 3.5. R3 → R4 QA差分

| 次元 | R3 freeze | R4確定 | 理由 |
|---|---|---|---|
| D04 | `D04.VAR.LOCALIZATION / I` | `U` | 現行LOCALIZATION定義は地名・施設・人物の置換で、器具の文化的適応に一致しない |
| D08 | `D08.DEX.PERCEPTUAL_ANOMALY / I` | `D08.DEX.DIRECT_EVENT / I` | 知覚自体の異常ではなく、器具が動く異常出来事への直接遭遇が契機 |

その他のD01–D21はR3 freezeを維持した。

# 4. 再コーディング前旧10との差分比較

比較対象: `1b24d197f9cb94241e26a368a841bc1071671d23` の旧 `0011_10_analysis.md`。R3 freeze後にのみ参照した。

| 次元 | 旧判定 | R4確定 | 差分分類 | 要点 |
|---|---|---|---|---|
| D01 | `G1 / D` | `G1 / I` | `Status mismatch` | 1884頃伝来は公式研究整理を介するため成立層はI |
| D02 | `NEWSPAPER / D` | 同左 | 一致 | — |
| D03 | `SCHOOL_ORAL` + newspaper/book / I | newspaper + book / D | `Scope mismatch` / `Evidence mismatch` | 現代学校型をScope外へ分離 |
| D04 | `ACCRETION` + oral variation / I | `U` | `Scope mismatch` / `Taxonomy gap` | 後代危険規則を除外し、明治期の物質的適応は現Childで表現困難 |
| D05 | `RITUAL_PROCEDURE / D` | 同code / I | `Status mismatch` | 詳細手順は公式整理を介するためI |
| D06 | `T6 / I` | `T5 / I` | `Code-selection mismatch` | 明治Scopeは実践すれば回答が得られる条件付き信念 |
| D07 | `UNEXPLAINED_EVENT / I` | `INFORMATION_VOID` + `FATE_OMEN / I` | `Code-selection mismatch` | 実践目的に中心質問を合わせる |
| D08 | `DIRECT_EVENT / I` | 同左 | 一致 | R4 QAで旧判定とも一致したが独立に到達 |
| D09 | `RITUAL_RULE / I` | `AGENCY_ATTRIBUTION` + `OMEN_FORECAST / I` | `Code-selection mismatch` | 意味付与操作とD11手順を分離 |
| D10 | `YOKAI_ENTITY` + `GHOST_SPIRIT / C` | `DEITY_DIVINE` vs `YOKAI_ENTITY / C` | `Scope mismatch` / `Code-selection mismatch` | 現代死者霊を除外、明治資料内競合のみ保持 |
| D11 | `PERFORM_RITUAL / D` | 同左 | 一致 | — |
| D12 | `PROTAGONIST_EXPERIENCER / I` | `OBJECT_PRODUCT / I` | `Prior coding error` / `Code-selection mismatch` | D13主作用の直接対象は器具 |
| D13 | `ENV_OBJECT_MANIPULATION` + info action / I | `ENV_OBJECT_MANIPULATION / I` | `Scope mismatch` | 後続行動作用をScope核から外す |
| D14 | `MIX / I` | `NEU / I` | `Scope mismatch` | 後代の危険帰結を除外 |
| D15 | `REVELATION_KNOWLEDGE / I` | 同左 | 一致 | — |
| D16 | `SEQUENTIAL_EPISODE / I` | 同左 | 一致 | — |
| D17 | `RITUAL_CLOSURE / I` | `DELIBERATE_INVOCATION / I` | `Scope mismatch` / `Code-selection mismatch` | 後代終了禁忌を除外し利用構造を採る |
| D18 | `L1=1,L2=0,L3=1 / I` | 同左 | 一致 | — |
| D19 | `NATIONAL_PUBLIC / D` | 同code / I | `Status mismatch` | 「各地で大流行」は公式後代整理によるためI |
| D20 | `COMMON_KNOWLEDGE / I` | `U` taxonomy gap | `Taxonomy gap` / `Prior coding error` | 手順共有ではなく、未知回答を誰が保持するかが中心質問 |
| D21 | `A1 / I` | 同左 | 一致 | — |

# 5. R4結論

- D12→D13→D15 causal QA: Pass
- D18 L3 evidence QA: Pass
- U / NA / C QA: Pass
- taxonomy gap QA: Pass。D04・D20の2候補をR5へ送る
- R3→R4修正: D04、D08の2次元
- 旧10との差分比較・分類: 完了
- 旧Excel比較: R5 Global Reconciliationへ移管
