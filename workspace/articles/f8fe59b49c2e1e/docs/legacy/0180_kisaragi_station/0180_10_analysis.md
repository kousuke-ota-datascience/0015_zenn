記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

伝承内容・典拠・異伝・証拠上の不確実性は、同一ディレクトリの `0180_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0180`
- `伝承エントリ名称`: きさらぎ駅
- `Macro_Category`: ネット怪談
- `Entry_Type`: 物語・伝説
- `Version_Scope`: 2004年1月8日深夜から9日未明にかけて匿名掲示板2ちゃんねる「身のまわりで変なことが起こったら実況するスレ26」で進行した初期実況。レス98から投稿者「はすみ」の最終投稿レス635までを中心とする。後年の類似体験談、異界駅一般化、翻案作品、遠州鉄道等による後代コンテンツ展開はScope外。

# 2. 分析概念次元

各次元は、コード値、判定根拠、この伝承における現れ方の順で記述する。

## 2.1. 来歴・流通・提示の次元

### 2.1.1. D01: 生成年代

**いつ成立したか**

- Primary Child / Value: `D01.G6` — 2000年代
- Primary Parent: なし
- Secondary: なし
- Status: `D`

**判定根拠**

Version Scope自体が2004年のリアルタイム実況artifactとして公式過去ログで直接確認できるため、単なる最古確認資料からの推定ではなく生成形を直接固定できる。

**この伝承における現れ方**

初期形は2004年の掲示板実況として成立する。

### 2.1.2. D02: 最古確認流通媒体

**現在の証拠で、伝承が実際に人から人へ流通したことを最も古く確認できる媒体は何か**

- Primary Child / Value: `D02.WEB.FORUM`
- Primary Parent: `D02.WEB`
- Secondary: なし
- Status: `D`

**判定根拠**

2ちゃんねる公式過去ログを直接確認できる。

**この伝承における現れ方**

公開Web掲示板で初期実況が進行する。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）

**今回コードするVersion Scopeで、伝承が受容者へ流通したことを確認できる媒体は何か**

- Primary Child / Value: `D03.WEB.FORUM`
- Primary Parent: `D03.WEB`
- Secondary: なし
- Status: `D`

**判定根拠**

今回のScopeは初期掲示板実況そのものに限定し、その媒体を公式過去ログで直接確認できる。

**この伝承における現れ方**

参加者が同一スレッドで閲覧・応答する。

### 2.1.4. D04: 生成・変容パターン

**時間とともにどう変形したか**

- Primary Child / Value: `D04.COL.LIVE_COCREATION`
- Primary Parent: `D04.COL`
- Secondary: なし
- Status: `D`

**判定根拠**

投稿者の逐次報告へ参加者が質問・検索・助言・解釈を返し、その往復の中で実況が進む。

**この伝承における現れ方**

一方向の完成済み怪談ではなく、リアルタイム共同生成型として展開する。

### 2.1.5. D05: 提示形式

**どんなコミュニケーション形式で提示されるか**

- Primary Child / Value: `D05.HYB.COLLAB_THREAD`
- Primary Parent: `D05.HYB`
- Secondary: `D05.EXP.LIVE_1P`
- Status: `D`

**判定根拠**

投稿者の一人称実況と複数参加者の応答が一体化する。

**この伝承における現れ方**

はすみの現在進行形報告へスレッド参加者が介入する。

### 2.1.6. D06: 真実性提示

**どんな「本当らしさ」を要求するか**

- Primary Child / Value: `D06.T1`
- Primary Parent: なし
- Secondary: なし
- Status: `D`

**判定根拠**

投稿者は自身が現在経験している事態として逐次報告する。これは投稿内容の客観的真実性認定とは別である。

**この伝承における現れ方**

提示形式上は直接体験事実として語られる。

## 2.2. 意味形成の次元

### 2.2.1. D07: 意味形成対象

**何が不可解・不確実なのか**

- Primary Child / Value: `D07.PSE.SPATIAL_ROUTE_ANOMALY`
- Primary Parent: `D07.PSE`
- Secondary: `D07.PSE.HIDDEN_VANISHED_PLACE`
- Status: `D`

**判定根拠**

通常路線から説明できない駅・経路へ接続し、通常地理へ位置づけられないことがログに直接現れる。

**この伝承における現れ方**

日常の鉄道移動から位置不明の駅・経路へ逸脱する。

### 2.2.2. D08: 意味形成契機

**何を手掛かりに問題化されるか**

- Primary Child / Value: `D08.DEX.DIRECT_EVENT`
- Primary Parent: `D08.DEX`
- Secondary: なし
- Status: `D`

**判定根拠**

普段なら数分ごとに停まる列車が20分以上停まらないという投稿者の直接経験から始まる。

**この伝承における現れ方**

通常運行との差が最初の異常cueになる。

### 2.2.3. D09: 意味付与操作

**不可解なものをどう理解可能にするか**

- Primary Child / Value: `D09.UNK.DELIBERATE_NONRESOLUTION`
- Primary Parent: `D09.UNK`
- Secondary: `D09.CAT.TYPE_ASSIGNMENT`
- Status: `I`

**判定根拠**

参加者は通常運行異常、異界、あの世等へ仮分類するが、いずれも確定しない。

**この伝承における現れ方**

複数説明を提示しつつ、真相を閉じないまま終わる。

### 2.2.4. D10: 因果源存在論

**原因を何として世界に置くか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

空間異常は現象として確認できるが、経路・場所・霊・異界等のどれが原因源かはログから固定できない。

**この伝承における現れ方**

何が異常を起こしているかは未確定のまま残る。

## 2.3. 因果・行動モデルの次元

### 2.3.1. D11: 発動・接触条件

**何を契機に因果系へ入るか**

- Primary Child / Value: `D11.PAS.ACCIDENT_INVOLVEMENT`
- Primary Parent: `D11.PAS`
- Secondary: `D11.MOV.RIDE_BOARD`
- Status: `I`

**判定根拠**

特別な儀式や探索ではなく、通常の帰宅乗車中に非意図的に異常へ巻き込まれる。`ACCIDENT_INVOLVEMENT` への写像は分析判断を含むため `I`。

**この伝承における現れ方**

日常的な乗車が異常エピソードへの入口になる。

### 2.3.2. D12: 作用対象

**誰／何に作用するか**

- Primary Child / Value: `D12.FOC.PROTAGONIST_EXPERIENCER`
- Primary Parent: `D12.FOC`
- Secondary: なし
- Status: `D`

**判定根拠**

異常経路・未知駅・位置特定不能等の影響を直接受ける中心ははすみである。

**この伝承における現れ方**

投稿者が一貫して経験主体となる。

### 2.3.3. D13: 作用機構

**因果源が対象へ何をするか**

- Primary Child / Value: `D13.RST.SPATIAL_DISTORTION`
- Primary Parent: `D13.RST`
- Secondary: なし
- Status: `I`

**判定根拠**

通常路線では説明できない接続、未知駅への到達、位置特定不成立が続く。原因主体はUでも、作用としての空間異常はログから推定できる。

**この伝承における現れ方**

通常地理との対応が崩れる。

### 2.3.4. D14: 帰結極性

**結果は正・負・中立・混合か**

- Primary Child / Value: `D14.NEG`
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

安全な帰宅が破綻し、未知の場所・異常人物・不審な運転者にさらされるため負の帰結と読む。

**この伝承における現れ方**

帰宅不能と危険が主な方向性になる。

### 2.3.5. D15: 帰結領域

**何の領域が最終的に変わるか**

- Primary Child / Value: `D15.KNW.UNCERTAINTY_PRESERVED`
- Primary Parent: `D15.KNW`
- Secondary: なし
- Status: `D`

**判定根拠**

初期ログでは帰還・所在・身体被害・真相のいずれも確定しない。投稿停止を失踪・死亡へ変換しない。

**この伝承における現れ方**

未解決状態が終端として残る。

### 2.3.6. D16: 因果時間構造

**発動条件・原因成立から主作用／主帰結までの関係は、時間上どのように編成されているか**

- Primary Child / Value: `D16.EVT.SEQUENTIAL_EPISODE`
- Primary Parent: `D16.EVT`
- Secondary: なし
- Status: `D`

**判定根拠**

長時間無停車→未知駅→探索→線路移動→老人→トンネル→車→投稿終了と、一晩で段階進行する。

**この伝承における現れ方**

実況時系列そのものが因果列を形成する。

### 2.3.7. D17: 回避・制御方式

**結果をどう回避・制御・利用できるか**

- Primary Child / Value: `D17.UNA.NO_KNOWN_ESCAPE`
- Primary Parent: `D17.UNA`
- Secondary: なし
- Status: `I`

**判定根拠**

家族・警察への連絡や参加者の複数助言を試すが、有効性が確認された帰還法へ収束しない。単なる情報欠如ではなく、複数対処が未成功・未確立のまま終わる。

**この伝承における現れ方**

対応行動は多数あるが、確実な脱出規則は得られない。

### 2.3.8. D18: 作用レイヤー

**因果効力は伝承内／受容者／社会現実のどこに及ぶか**

- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `I`

**判定根拠**

L1の伝承内部因果は確認できるが、L2/L3の不在判断を含むためvector全体をDirectにしない。後代の事業者展開等はScope外である。

**この伝承における現れ方**

初期実況内部の作用に限定してコードする。

## 2.4. 社会的埋め込みの次元

### 2.4.1. D19: 流通範囲

**誰の間に伝承が流通するか**

- Primary Child / Value: `D19.NET.OPEN_FORUM_WEB`
- Primary Parent: `D19.NET`
- Secondary: なし
- Status: `D`

**判定根拠**

初期版は公開掲示板上で不特定多数が閲覧・参加可能であり、実際の参加者応答もログで確認できる。

**この伝承における現れ方**

スレッド参加者が同時に受容・応答する公開Web流通である。

### 2.4.2. D20: 特権情報保持者

**誰が真相・追加情報を持つか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

投稿者・参加者・家族・警察等の誰も真相や確実な追加情報を特権的に保持する主体として固定できない。ただし「誰も知らない」と積極的に構造化されているわけでもないため強い否定コードは付与しない。

**この伝承における現れ方**

複数人物が探索・助言するが、真相保持者は確定しない。

### 2.4.3. D21: 現実アンカー

**実在世界へどの程度固定されるか**

- Primary Child / Value: `D21.A2`
- Primary Parent: なし
- Secondary: なし
- Status: `D`

**判定根拠**

新浜松、静岡県内の私鉄、比奈等の実在地名・交通背景へ具体的に接続する。ただし「きさらぎ駅」自体の実在を意味しない。

**この伝承における現れ方**

現実の鉄道・地域名へ接続しつつ、その中に確認不能な駅・経路が挿入される。