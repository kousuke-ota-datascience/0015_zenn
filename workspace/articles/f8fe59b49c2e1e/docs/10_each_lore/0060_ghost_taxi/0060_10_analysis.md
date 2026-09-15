記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0060_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0060`
- `伝承エントリ名称`: タクシー幽霊
- `Macro_Category`: 交通・インフラ
- `Entry_Type`: 物語・伝説
- `Version_Scope`: 朝里2023で日本の「タクシー幽霊」として整理される再話群のうち、タクシー運転手が人物を客として乗せ、移動中・到着時にその客が不在となる、または目的地で既に死亡した人物と一致すると判明し、死者・幽霊の乗客として理解される型を対象とする。夜・雨・特定墓地・濡れた座席・特定死因は必須条件にしない。1950年代新聞本文は未実見であり、その記事固有の提示形式・真実性姿勢はScope Evidenceとして固定しない。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示の次元

### 2.1.1. D01: 生成年代

**いつ成立したか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

朝里2023は1950年代の新聞掲載を整理しているが、当該新聞本文・紙名・日付は未実見であり、成立時期を独立に限定するHistory Evidenceはない。

**この伝承における現れ方**

1950年代までの存在確認点はあるが、成立年代そのものは未確定である。

### 2.1.2. D02: 最古確認流通媒体

**現在の証拠で、伝承が実際に人から人へ流通したことを最も古く確認できる媒体は何か**

- Primary Child / Value: `D02.PRT.NEWSPAPER` — 新聞
- Primary Parent: `D02.PRT`
- Secondary: なし
- Status: `I`

**判定根拠**

朝里2023が1950年代の新聞掲載を明示する。原新聞未実見のため間接Evidenceである。

**この伝承における現れ方**

現時点で追跡できる早期媒体定点は新聞である。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）

**今回コードするVersion Scopeで、伝承が受容者へ流通したことを確認できる媒体は何か**

- Primary Child / Value: `D03.PRT.NEWSPAPER` — 新聞
- Primary Parent: `D03.PRT`
- Secondary: `D03.NET.WEB_ARTICLE` — Web記事
- Status: `I`

**判定根拠**

1950年代新聞掲載は二次資料による間接確認、2023年webムーでは複数再話を本文で確認できる。

**この伝承における現れ方**

印刷媒体の早期確認点と、後代Web上での再提示が確認できる。

### 2.1.4. D04: 生成・変容パターン

**時間とともにどう変形したか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

00では故人確認型、車内消失型、墓地結合型、災害後の被災地型等の併存を確認できるが、それらの連続した流通経路や先後関係は復元できていない。異伝の存在だけから `RETELLING`、`LOCALIZATION`、`EVENT_ATTACHMENT` という変容機構を確定しない。

**この伝承における現れ方**

複数型の差異は確認できるが、どの型がどの型から時間的に変容したかは未確定である。

### 2.1.5. D05: 提示形式

**どんなコミュニケーション形式で提示されるか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

1950年代新聞本文が未実見であり、Scoped Contentの提示形式を直接再構成できない。

**この伝承における現れ方**

体験談風・伝聞風・記事風のいずれをPrimaryとすべきか確定できない。

### 2.1.6. D06: 真実性提示

**どんな「本当らしさ」を要求するか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

後代資料が怪談・都市伝説と分類することは、原話自身の真実性提示を示さない。

**この伝承における現れ方**

実話断定・真偽留保等の姿勢を現在のEvidenceから一意に決められない。

## 2.2. 意味形成の次元

### 2.2.1. D07: 意味形成対象

**何が不可解・不確実なのか**

- Primary Child / Value: `D07.ANO.UNEXPLAINED_EVENT` — 説明不能事象
- Primary Parent: `D07.ANO`
- Secondary: `D07.ANO.UNKNOWN_EXISTENCE` — 未知存在
- Status: `I`

**判定根拠**

通常の乗客が不在となる、または死亡済み人物と一致するという不連続と存在論的不確実性が中心となる。

**この伝承における現れ方**

普通の客としていた人物が通常の生者として連続しない。

### 2.2.2. D08: 意味形成契機

**何を手掛かりに問題化されるか**

- Primary Child / Value: `D08.DEX.DIRECT_EVENT` — 出来事への直接遭遇
- Primary Parent: `D08.DEX`
- Secondary: `D08.STY.WITNESS_REPORT` — 目撃証言
- Status: `I`

**判定根拠**

運転手自身が客の不在化または故人確認に直面し、その経験が後に再話される。

**この伝承における現れ方**

通常営業中の直接経験が問題化の手掛かりとなる。

### 2.2.3. D09: 意味付与操作

**不可解なものをどう理解可能にするか**

- Primary Child / Value: `D09.CAT.TYPE_ASSIGNMENT` — 類型化
- Primary Parent: `D09.CAT`
- Secondary: `D09.AGN.AGENCY_ATTRIBUTION` — 主体化
- Status: `I`

**判定根拠**

消失や故人との一致を「死者・幽霊の乗客」というカテゴリーへ割り当てる。

**この伝承における現れ方**

普通の乗客ではなく幽霊客だったという理解へ接続される。

### 2.2.4. D10: 因果源存在論

**原因を何として世界に置くか**

- Primary Child / Value: `D10.SUP.GHOST_SPIRIT` — 幽霊・死者霊
- Primary Parent: `D10.SUP`
- Secondary: なし
- Status: `D`

**判定根拠**

Version Scopeを、乗客が死者・幽霊として理解される型に限定している。

**この伝承における現れ方**

死者が一時的に通常の乗客として現れたものとして出来事が理解される。

## 2.3. 因果・行動モデルの次元

### 2.3.1. D11: 発動・接触条件

**何を契機に因果系へ入るか**

- Primary Child / Value: `D11.PAS.SPONTANEOUS_SELECTION` — 偶然選ばれる・遭遇する
- Primary Parent: `D11.PAS`
- Secondary: `D11.SOC.MEET_INTERACT` — 会う・会話する
- Status: `I`

**判定根拠**

運転手が通常営業中に客を乗せることで出来事が始まり、召喚手順や禁忌違反は確認されない。

**この伝承における現れ方**

日常業務の延長で異常な乗客と接触する。

### 2.3.2. D12: 作用対象

**誰／何に作用するか**

- Primary Child / Value: `D12.FOC.PROTAGONIST_EXPERIENCER` — 主人公・体験者
- Primary Parent: `D12.FOC`
- Secondary: なし
- Status: `I`

**判定根拠**

幽霊客の顕現・消失・故人判明を直接経験する焦点人物はタクシー運転手である。

**この伝承における現れ方**

運転手が通常の乗客という初期認識を覆す出来事に直面する。

### 2.3.3. D13: 作用機構

**因果源が対象へ何をするか**

- Primary Child / Value: `D13.MAN.MANIFEST_ONLY` — 顕現のみ
- Primary Parent: `D13.MAN`
- Secondary: なし
- Status: `I`

**判定根拠**

Scope内で安定して確認できる主作用は、死者・幽霊と理解される人物が通常の乗客として現れることである。

**この伝承における現れ方**

乗客として顕現し、その後の消失または故人確認で異常性が明らかになる。

### 2.3.4. D14: 帰結極性

**結果は正・負・中立・混合か**

- Primary Child / Value: `D14.NEU` — 中立
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

運転手への身体危害・死亡・利益を主要帰結として固定できず、中心は異常経験と認識転換である。

**この伝承における現れ方**

不可解な出来事を経験するが、必ず害を受けるとはされない。

### 2.3.5. D15: 帰結領域

**何の領域が最終的に変わるか**

- Primary Child / Value: `D15.KNW.BELIEF_REVISION` — 信念変更
- Primary Parent: `D15.KNW`
- Secondary: `D15.KNW.REVELATION_KNOWLEDGE` — 真相・知識獲得
- Status: `I`

**判定根拠**

普通の乗客だという初期認識が、死者・幽霊だったという理解へ変更される。

**この伝承における現れ方**

「通常の客を運んだ」が「死者を乗せた」へ書き換えられる。

### 2.3.6. D16: 因果時間構造

**発動条件・原因成立から主作用／主帰結までの関係は、時間上どのように編成されているか**

- Primary Child / Value: `D16.EVT.SEQUENTIAL_EPISODE` — エピソード内段階進行
- Primary Parent: `D16.EVT`
- Secondary: なし
- Status: `D`

**判定根拠**

接触→乗車・移動→不在化または目的地到着→故人確認という順序が、一回の乗車エピソード内で進む。

**この伝承における現れ方**

通常営業として始まった出来事が途中または終端で怪異経験へ転換する。

### 2.3.7. D17: 回避・制御方式

**結果をどう回避・制御・利用できるか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

共通の回避手順が確認できないというabsenceから、制御不能等の伝承内規則を推定しない。

**この伝承における現れ方**

確立した回避・制御方法の有無は未確定である。

### 2.3.8. D18: 作用レイヤー

**因果効力は伝承内／受容者／社会現実のどこに及ぶか**

- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `I`

**判定根拠**

L1は伝承内部で幽霊客の顕現と認識変化が起きるため1。L2の自己適用規則とL3の独立X Evidenceは確認していない。不在値を含むvector全体はIとする。

**この伝承における現れ方**

因果作用は主として物語内部の運転手と幽霊客の接触にある。

## 2.4. 社会的埋め込みの次元

### 2.4.1. D19: 流通範囲

**誰の間に伝承が流通するか**

- Primary Child / Value: `D19.MAS.NATIONAL_PUBLIC` — 全国的大衆
- Primary Parent: `D19.MAS`
- Secondary: `D19.LOC.REGIONAL` — 地域・地方
- Status: `D`

**判定根拠**

朝里2023は日本各地に類話があり、青山墓地・深泥池等の地域版が存在することを整理している。

**この伝承における現れ方**

広域の話型と地域版の双方が確認される。

### 2.4.2. D20: 特権情報保持者

**誰が真相・追加情報を持つか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

故人確認型では遺族等が追加情報を持つが、車内消失型では同じ構造を必要としない。

**この伝承における現れ方**

Scope全体として一意の情報保持者を固定できない。

### 2.4.3. D21: 現実アンカー

**実在世界へどの程度固定されるか**

- Primary Child / Value: `D21.A1` — 一般的現実背景
- Primary Parent: なし
- Secondary: なし
- Status: `D`

**判定根拠**

タクシー、運転手、乗客、道路、自宅等の一般的現実要素に依存するが、単一の特定地点・企業・人物を必須としない。

**この伝承における現れ方**

怪異は日常的なタクシー営業という現実背景の中で起こる。