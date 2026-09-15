記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

伝承内容・典拠・異伝・証拠上の不確実性は、同一ディレクトリの `0179_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0179`
- `伝承エントリ名称`: くねくね
- `Macro_Category`: ネット怪談
- `Entry_Type`: 物語・伝説
- `Version_Scope`: 2001年後代再録で確認される「分からないほうがいい」型と、2003年obake.cc保存記録で投稿者自身が先行話と自身の体験を混ぜたと明示する「くねくね」型。秋田・田園・双眼鏡・祖父・名称は2003年増補として扱う。

# 2. 分析概念次元

各次元は、コード値、判定根拠、この伝承における現れ方の順で記述する。

## 2.1. 来歴・流通・提示の次元

### 2.1.1. D01: 生成年代
**いつ成立したか**
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**
2001年7月7日は後代再録で確認できる最古級定点であり、生成年を直接示さない。

**この伝承における現れ方**
少なくとも2001年には原型が投稿されたと後代再録から確認できるが、それ以前の形成は不明である。

### 2.1.2. D02: 最古確認流通媒体
**現在の証拠で、伝承が実際に人から人へ流通したことを最も古く確認できる媒体は何か**
- Primary Child / Value: `D02.WEB.FORUM`
- Primary Parent: `D02.WEB`
- Secondary: なし
- Status: `I`

**判定根拠**
2001年レス212は後代Web再録から確認するため、媒体写像は可能だがDirectとはしない。

**この伝承における現れ方**
初期確認形は匿名Web掲示板の怪談投稿である。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）
**今回コードするVersion Scopeで、伝承が受容者へ流通したことを確認できる媒体は何か**
- Primary Child / Value: `D03.WEB.FORUM`
- Primary Parent: `D03.WEB`
- Secondary: `D03.WEB.WEBSITE`
- Status: `I`

**判定根拠**
原型・2003年型はいずれも掲示板由来だが、現時点では後代再録・保存Webを介して確認する。

**この伝承における現れ方**
掲示板投稿が保存サイト・再録記事を通じて後代へ残る。

### 2.1.4. D04: 生成・変容パターン
**時間とともにどう変形したか**
- Primary Child / Value: `D04.VAR.ACCRETION`
- Primary Parent: `D04.VAR`
- Secondary: `D04.REC.RETELLING`
- Status: `I`

**判定根拠**
2003年投稿者が先行話と自身の体験を混ぜて詳しく書いたと保存記録上で明示し、秋田・双眼鏡・祖父等を追加する。原ログ未固定のため `I`。

**この伝承における現れ方**
先行する匿名怪談を再話しつつ、自身の体験要素を加えて現在型に近い形へ増補する。

### 2.1.5. D05: 提示形式
**どんなコミュニケーション形式で提示されるか**
- Primary Child / Value: `D05.EXP.RETROSPECTIVE_1P`
- Primary Parent: `D05.EXP`
- Secondary: なし
- Status: `I`

**判定根拠**
2003年型は投稿者自身の子供時代の体験を一人称で語るが、保存記録経由なので `I`。

**この伝承における現れ方**
自己体験を材料にした回想形式で提示される。

### 2.1.6. D06: 真実性提示
**どんな「本当らしさ」を要求するか**
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `C`

**判定根拠**
2003年投稿者は自身の体験を材料にする一方、先行する「分からないほうがいい」と混ぜて詳しく書いたと明示する。完成した2003年型全体を純粋な `T1` 直接体験事実へ単純化できず、直接体験提示と再構成・混成提示が同一artifact内で競合するため `C` とする。

**この伝承における現れ方**
「自分の体験」であることを示しながら、既存怪談を意識的に混ぜた再構成でもある。

## 2.2. 意味形成の次元

### 2.2.1. D07: 意味形成対象
**何が不可解・不確実なのか**
- Primary Child / Value: `D07.ANO.UNKNOWN_EXISTENCE`
- Primary Parent: `D07.ANO`
- Secondary: `D07.ANO.ANOMALOUS_PERCEPTION`
- Status: `I`

**判定根拠**
遠方の白い存在が何者か分からず、不自然な動き自体も異常知覚として語られる。

**この伝承における現れ方**
見えているのに正体を確定できない対象が怪異の中心になる。

### 2.2.2. D08: 意味形成契機
**何を手掛かりに問題化されるか**
- Primary Child / Value: `D08.DEX.DIRECT_EVENT`
- Primary Parent: `D08.DEX`
- Secondary: `D08.DEX.PERCEPTUAL_ANOMALY`
- Status: `I`

**判定根拠**
人物が白い存在を直接視認することが問題化の入口だが、Contentは保存記録・後代再録から確認するため `I`。

**この伝承における現れ方**
遠方の異常な対象を目撃するところから謎が始まる。

### 2.2.3. D09: 意味付与操作
**不可解なものをどう理解可能にするか**
- Primary Child / Value: `D09.UNK.FORBIDDEN_INQUIRY`
- Primary Parent: `D09.UNK`
- Secondary: `D09.UNK.DELIBERATE_NONRESOLUTION`
- Status: `I`

**判定根拠**
正体を理解すること自体が危険として扱われ、読者へ正体は開示されない。

**この伝承における現れ方**
「知るべきではない」という構造が謎を解かずに保持する。

### 2.2.4. D10: 因果源存在論
**原因を何として世界に置くか**
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**
白い存在の正体は確定されず、妖怪・霊等へ固定できない。

**この伝承における現れ方**
作用主体は「正体不明の白い何か」のまま残る。

## 2.3. 因果・行動モデルの次元

### 2.3.1. D11: 発動・接触条件
**何を契機に因果系へ入るか**
- Primary Child / Value: `D11.COG.UNDERSTAND_RECOGNIZE`
- Primary Parent: `D11.COG`
- Secondary: `D11.SNS.VISUAL_EXPOSURE`
- Status: `I`

**判定根拠**
詳しく視認し「何であるか」を理解することが人物異変に先行する。原ログ未固定のため `I`。

**この伝承における現れ方**
単に遠くから見るより、詳しく見て理解した人物に危害が集中する。

### 2.3.2. D12: 作用対象
**誰／何に作用するか**
- Primary Child / Value: `D12.FOC.SPECIFIC_OTHER`
- Primary Parent: `D12.FOC`
- Secondary: `D12.FOC.PROTAGONIST_EXPERIENCER`
- Status: `I`

**判定根拠**
異変を受けるのは詳しく見て理解した兄であり、語り手はその変化を観察する。

**この伝承における現れ方**
焦点は対象を理解してしまった特定人物へ置かれる。

### 2.3.3. D13: 作用機構
**因果源が対象へ何をするか**
- Primary Child / Value: `D13.COG.COGNITION_TRIGGERED_HARM`
- Primary Parent: `D13.COG`
- Secondary: `D13.MNT.MENTAL_INFLUENCE`
- Status: `I`

**判定根拠**
対象を理解したことと精神・行動上の異変が物語内で結び付けられる。現実医学的因果は主張しない。

**この伝承における現れ方**
「知ること」が危害契機になる認知危害型として描かれる。

### 2.3.4. D14: 帰結極性
**結果は正・負・中立・混合か**
- Primary Child / Value: `D14.NEG`
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**
人物の重大な異変として語られるため負である。保存資料由来なので `I`。

**この伝承における現れ方**
対象理解は利益ではなく明確な人物崩壊へ接続される。

### 2.3.5. D15: 帰結領域
**何の領域が最終的に変わるか**
- Primary Child / Value: `D15.PSY.SELF_IDENTITY_DISRUPTION`
- Primary Parent: `D15.PSY`
- Secondary: なし
- Status: `I`

**判定根拠**
兄が通常状態から大きく変化した人物として描かれるが、特定診断や強迫行動を生成しない。

**この伝承における現れ方**
人物の人格・自己状態が不可逆的に崩れたように語られる。

### 2.3.6. D16: 因果時間構造
**発動条件・原因成立から主作用／主帰結までの関係は、時間上どのように編成されているか**
- Primary Child / Value: `D16.EVT.SEQUENTIAL_EPISODE`
- Primary Parent: `D16.EVT`
- Secondary: なし
- Status: `I`

**判定根拠**
発見→詳細視認→理解→異変→警告という段階系列で進行する。保存資料由来なので `I`。

**この伝承における現れ方**
一回の遭遇内で因果段階が順に進む。

### 2.3.7. D17: 回避・制御方式
**結果をどう回避・制御・利用できるか**
- Primary Child / Value: `D17.RUL.DO_NOT_ENGAGE`
- Primary Parent: `D17.RUL`
- Secondary: なし
- Status: `I`

**判定根拠**
2003年型で祖父が「見てはならない」と独立に警告するため、D11の単純反転ではなく明示的回避情報として扱う。原ログ未固定のため `I`。

**この伝承における現れ方**
対象を詳しく見ず、理解へ進まないことが危害回避として示される。

### 2.3.8. D18: 作用レイヤー
**因果効力は伝承内／受容者／社会現実のどこに及ぶか**
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `I`

**判定根拠**
危害は物語内人物へ作用する。読者が記事を理解しただけで作用する型や独立した社会現実効果は今回Scopeで固定しない。

**この伝承における現れ方**
作用は白い存在を見た物語内人物に限定される。

## 2.4. 社会的埋め込みの次元

### 2.4.1. D19: 流通範囲
**誰の間に伝承が流通するか**
- Primary Child / Value: `D19.NET.OPEN_FORUM_WEB`
- Primary Parent: `D19.NET`
- Secondary: なし
- Status: `I`

**判定根拠**
掲示板投稿由来であることを保存・再録資料から確認するため、公開Web流通として `I`。

**この伝承における現れ方**
匿名掲示板を介して不特定受容者へ流通する。

### 2.4.2. D20: 特権情報保持者
**誰が真相・追加情報を持つか**
- Primary Child / Value: `D20.PER.FAMILY_BLOODLINE`
- Primary Parent: `D20.PER`
- Secondary: なし
- Status: `I`

**判定根拠**
2003年型の祖父は、他の人物より先に「見てはならない」という危険回避情報を保持し、視認の有無を確認する。対象の完全な正体知識までは固定できないため、追加情報保持に限定して家族・血縁Childへ写像する。

**この伝承における現れ方**
祖父が正体そのものではなく、危険な接触を避けるための追加情報を偏在的に持つ。

### 2.4.3. D21: 現実アンカー
**実在世界へどの程度固定されるか**
- Primary Child / Value: `D21.A1`
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**
秋田という地域名はあるが具体地点は比定不能で、主な現実接点は匿名掲示板投稿・保存記録である。

**この伝承における現れ方**
実在地域名を持つ一方、怪異地点そのものは特定されない。