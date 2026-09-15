# 0011 こっくりさん — R3 Independent Recode

## 1. 独立性

- Version Scope正本: `0011_R2_version_scope.md`
- Evidence正本: `docs/10_each_lore/0011_kokkuri_san/0011_00_contents.md`
- baseline: control plane §1固定blob
- **旧 `0011_10_analysis.md` および旧ExcelのD01–D21は未参照。**

本ファイルをR3 freeze点とし、commit/push後にのみR4で旧10を参照する。

## 2. R3 Version Scope

1886–1887年に直接確認できる明治器具型。複数人がテーブルまたは飯櫃の蓋等の可動器具へ手を置き、問いを発し、参加者が意図的に動かしていないと感じる器具の動きを、外部的・霊的な主体からの意味ある回答として読む占い／神意判断の実践。

現代紙・硬貨型、学校型禁忌、憑依・祟り・終了規則はScope外。

# 3. D01–D21

## D01 生成年代

- Value: `D01.G1` — 1868–1944
- Status: `I`
- 根拠: 東洋大学公式整理は1884年頃の伝来・1886年頃の各地流行を、井上円了の明治資料に基づき示す。1886年新聞、1887年研究書も存在する。伝来以前の同一実践を仮定しない。

## D02 最古確認流通媒体

- Primary: `D02.PRT.NEWSPAPER`
- Parent: `D02.PRT`
- Status: `D`
- 根拠: 現在直接固定できる最古の公共流通媒体は1886年7月16日『朝日新聞（大阪）』の記事。実践自体が先行していたことと、最古確認媒体を区別する。

## D03 確認流通媒体ポートフォリオ

- Primary: `D03.PRT.NEWSPAPER`
- Secondary: `D03.PRT.BOOK`
- Status: `D`
- 根拠: Scope期間に1886年新聞と1887年井上円了著作を直接確認できる。後代の学校口承をScope媒体へ入れない。

## D04 生成・変容パターン

- Primary: `D04.VAR.LOCALIZATION`
- Parent: `D04.VAR`
- Status: `I`
- 根拠: 西洋のテーブル・ターニングを、日本の日常器具である飯櫃の蓋・竹脚等へ適応したことが公式整理で確認できる。通信媒体移行ではなく、実践器具・名称の地域適応として扱う。

## D05 提示形式

- Primary: `D05.RUL.RITUAL_PROCEDURE`
- Parent: `D05.RUL`
- Status: `I`
- 根拠: 複数人が器具へ手を置き、問いを発し、動きを回答として読む再現可能な占い手順として実践される。

## D06 真実性提示

- Value: `D06.T5` — 条件付き信念
- Status: `I`
- 根拠: 実践すれば器具の動きから回答・神意を得られるという条件付きの実践的信念を要求する。研究者側の無意識運動説明とは分離する。

## D07 意味形成対象

- Primary: `D07.ISU.INFORMATION_VOID` — 情報空白
- Secondary: `D07.DCF.FATE_OMEN` — 運命・予兆
- Status: `I`
- 根拠: 問いへの未知の回答を得ることが中心で、未来・吉凶の予測はその主要下位用途の一つ。

## D08 意味形成契機

- Primary: `D08.DEX.PERCEPTUAL_ANOMALY`
- Parent: `D08.DEX`
- Status: `I`
- 根拠: 参加者が意図的に動かしていないと感じる器具の移動が、外部主体による回答という意味形成を開始する知覚上の異常である。

## D09 意味付与操作

- Primary: `D09.AGN.AGENCY_ATTRIBUTION`
- Secondary: `D09.PPR.OMEN_FORECAST`
- Status: `I`
- 根拠: 器具運動を偶然・機械現象ではなく意思ある外部主体の応答とみなし、さらに回答を未来・吉凶の予測に用いる。

## D10 因果源存在論

- Candidate A: `D10.SUP.DEITY_DIVINE`
- Candidate B: `D10.SUP.YOKAI_ENTITY`
- Status: `C`
- 根拠: 明治期資料では神意判断としての表現と「狐狗狸」という人格的／動物霊的説明が併存し、現Evidenceでは単一Childへ保守的に固定できない。現代の死者霊解釈は持ち込まない。

## D11 発動・接触条件

- Primary: `D11.MAN.PERFORM_RITUAL`
- Parent: `D11.MAN`
- Status: `D`
- 根拠: 複数人が器具に触れ質問を行う実践そのものが再現的な発動条件である。

## D12 作用対象

- Primary: `D12.OTD.OBJECT_PRODUCT`
- Parent: `D12.OTD`
- Status: `I`
- 根拠: D13主作用は可動器具を動かすことなので、その直接対象は参加者ではなくテーブル・飯櫃の蓋等の物体である。

## D13 作用機構

- Primary: `D13.PHY.ENV_OBJECT_MANIPULATION`
- Parent: `D13.PHY`
- Status: `I`
- 根拠: 伝承内部では外部主体が器具を動かし、その運動が回答を表す。

## D14 帰結極性

- Value: `D14.NEU`
- Status: `I`
- 根拠: 明治期Scopeの必須帰結は質問への回答取得であり、回答内容は正負いずれにもなりうる。後代の危険・祟りを混入しない。

## D15 帰結領域

- Primary: `D15.KNW.REVELATION_KNOWLEDGE`
- Parent: `D15.KNW`
- Status: `I`
- 根拠: 実践の終端で参加者が得る中心的変化は、問いに対する回答・知識の獲得である。

## D16 因果時間構造

- Primary: `D16.EVT.SEQUENTIAL_EPISODE`
- Parent: `D16.EVT`
- Status: `I`
- 根拠: 準備・接触→質問→器具移動→回答解釈という一つの実践内の段階列で成立する。

## D17 回避・制御方式

- Primary: `D17.USE.DELIBERATE_INVOCATION`
- Parent: `D17.USE`
- Status: `I`
- 根拠: 明治期Scopeでは危害回避より、こっくりを意図的に呼び出して未知情報を得る利用構造が中心である。後代の終了禁忌はScope外。

## D18 作用レイヤー

- `D18.L1=1`
- `D18.L2=0`
- `D18.L3=1`
- Status: `I`
- 根拠:
  - L1: 伝承内部で外部主体→器具移動→回答という因果を持つ。
  - L2: 伝承を聞くだけで受容者が作用対象になる規則はない。
  - L3: 1886年同時代新聞記事の見出し自体に「拘引」が明示され、現実側の社会的対応が存在したことを示すX Evidenceがある。ただし本文未実見のため具体的因果・対象は推測しない。

## D19 流通範囲

- Primary: `D19.MAS.NATIONAL_PUBLIC`
- Parent: `D19.MAS`
- Status: `I`
- 根拠: 東洋大学公式整理が1886年頃「日本各地で大流行」とする。単一地域・単一学校に限定されない広域流行として扱う。

## D20 特権情報保持者

- Primary: なし
- Status: `U`
- 根拠: 実践構造では、通常の参加者が知らない回答を「こっくりさん／神意」側が保持し提示する。しかし現行D20 taxonomyには超自然主体・神格・怪異を情報保持者として表すChildがない。既存Childへ無理に写像しない。
- Taxonomy gap candidate: **超自然的情報保持者 / supernatural privileged information holder**。

## D21 現実アンカー

- Value: `D21.A1` — 一般的現実背景
- Status: `I`
- 根拠: 日常的な家庭・器具・複数人の実践を用いるが、伝承内容自体が特定の唯一地点・人物・制度・史実を必須としない。伝来史の実在性と内容アンカーを分離する。

# 4. R3 pre-QA

## 4.1. D12 → D13 → D15

```text
D12: テーブル・飯櫃の蓋等の可動器具
→ D13: 外部主体が器具を動かす
→ D15: その動きから参加者が問いへの回答・知識を得る
```

因果列は接続する。

## 4.2. U / C

- D10=`C`: 明治Scope内の因果主体表現が神意／狐狗狸で競合。
- D20=`U`: Evidence不足ではなく、現taxonomyに超自然的情報保持者Childがないため保守的にU。R4 taxonomy gap QA対象。
- NA: なし。

## 4.3. taxonomy gap

D20に「超自然的情報保持者」がない点をgap候補としてR4へ送る。単一Entryだけでbaselineは変更しない。

## 4.4. D18 L3

1886年新聞見出しの「拘引」をX Evidenceとして採る。ただし見出しを越える具体的社会因果はコードしない。
