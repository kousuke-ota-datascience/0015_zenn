# 0. INTRODUCTION

記載内容については、以下3文書のインストラクションに従う。

- docs/00_research_overview/
    - 10_urban_legend_analysis_axes_theoretical_design.md
    - 20_urban_legend_parent_child_code_system.md
    - 30_urban_legend_analysis_coding_rules.md

**対象伝承:** きさらぎ駅

**Version Scope:** 2004年1月8日深夜から9日未明にかけての初期実況。後年の異界駅派生・追体験投稿・翻案は除外する。

**分析上の基本方針:** 後代に一般化した「異世界駅」という解釈を初期版へ逆輸入しない。初期実況が直接または合理的に支持する範囲だけをコードする。

# 1. D01. 生成年代

**いつ成立したか**

- Value: `D01.G6` — 2000年代
- Status: `D`

2004年1月8日深夜の掲示板実況を初期版として確認できるため、2000年代にコードする。

# 2. D02. 最古確認流通媒体

**現在の証拠で、伝承が実際に人から人へ流通したことを最も古く確認できる媒体は何か**

- Child: `D02.WEB.FORUM` — Web掲示板・フォーラム
- Parent: `D02.WEB` — オープンWeb・ソーシャル
- Status: `D`

2004年の2ちゃんねる実況スレッドが、現時点で直接確認できる最古の流通媒体である。

# 3. D03. 確認流通媒体ポートフォリオ（Version Scope）

**今回コードするVersion Scopeで、伝承が受容者へ流通したことを確認できる媒体は何か**

- Primary Child: `D03.WEB.FORUM` — Web掲示板・フォーラム
- Primary Parent: `D03.WEB` — オープンWeb・ソーシャル
- Secondary: なし
- Status: `D`

今回のScopeを初期実況に限定しているため、成立と受容の中心媒体はいずれも公開Web掲示板である。後年の書籍・映像・SNS流通はScope外とする。

# 4. D04. 生成・変容パターン

**時間とともにどう変形したか**

- Primary Child: `D04.COL.LIVE_COCREATION` — 実況共同生成
- Primary Parent: `D04.COL` — 共同・分散生成
- Secondary: なし
- Status: `D`

投稿者が状況を逐次報告し、掲示板参加者が助言・質問を返し、それを受けて投稿者が次の行動や報告を行う。初期版そのものがリアルタイム相互作用の中で成立しているため、実況共同生成をPrimaryとする。

# 5. D05. 提示形式

**どんなコミュニケーション形式で提示されるか**

- Primary Child: `D05.HYB.COLLAB_THREAD` — 参加型実況
- Primary Parent: `D05.HYB` — 集合・複合形式
- Secondary Child: `D05.EXP.LIVE_1P` — 一人称実況
- Secondary Parent: `D05.EXP` — 体験叙述
- Status: `D`

一人称のリアルタイム体験報告であると同時に、複数参加者の応答によって進行するスレッド形式である。構造上より特徴的な参加型実況をPrimaryとする。

# 6. D06. 真実性提示

**どんな「本当らしさ」を要求するか**

- Value: `D06.T1` — 直接体験事実
- Status: `D`

投稿者は進行中の出来事を自分自身の現在の体験として報告する。研究者が実話と認定するという意味ではなく、伝承が要求する真実性提示の形式をコードしている。

# 7. D07. 意味形成対象

**何が不可解・不確実なのか**

- Primary Child: `D07.PSE.SPATIAL_ROUTE_ANOMALY` — 空間・経路異常
- Primary Parent: `D07.PSE` — 場所・空間・環境
- Secondary Child: `D07.PSE.HIDDEN_VANISHED_PLACE` — 隠れた・消えた場所
- Secondary Parent: `D07.PSE` — 場所・空間・環境
- Status: `D`

通常利用している鉄道路線で、通常の運行・駅配置では説明しにくい経路へ入り、確認できない駅へ到達することが中心的な不可解さである。駅そのものの存在不明性をSecondaryとする。

# 8. D08. 意味形成契機

**何を手掛かりに問題化されるか**

- Primary Child: `D08.DEX.DIRECT_EVENT` — 出来事への直接遭遇
- Primary Parent: `D08.DEX` — 直接経験
- Secondary: なし
- Status: `D`

「普段利用する電車が異常に長時間停車しない」という直接経験が、最初に問題を立ち上げる契機になっている。

# 9. D09. 意味付与操作

**不可解なものをどう理解可能にするか**

- Primary Child: `D09.UNK.MYSTERY_LABEL` — 謎として命名
- Primary Parent: `D09.UNK` — 不可知性保持
- Secondary: なし
- Status: `I`

初期実況は、異常な場所・経路を最終的な原因説明へ閉じず、正体不明の「きさらぎ駅」という謎として保持する。後代の「異世界」解釈を採用せず、初期版における非解決性を重視した推定である。

# 10. D10. 因果源存在論

**原因を何として世界に置くか**

- Primary Child: `D10.SPC.ROUTE_CONNECTION` — 経路・接続
- Primary Parent: `D10.SPC` — 場所・時空
- Secondary: なし
- Status: `I`

初期実況では、明確な怪異主体や呪いは原因として確定されない。通常路線から説明不能な駅へ接続したという空間・経路そのものの異常を、最も保守的な因果源としてコードする。「異界・別世界」は後代解釈を含み得るためPrimaryにはしない。

# 11. D11. 発動・接触条件

**何を契機に因果系へ入るか**

- Primary Child: `D11.PAS.SPONTANEOUS_SELECTION` — 偶然選ばれる・遭遇する
- Primary Parent: `D11.PAS` — 受動発生
- Secondary Child: `D11.MOV.RIDE_BOARD` — 乗る
- Secondary Parent: `D11.MOV` — 移動・空間進入
- Status: `I`

投稿者は特別な儀式や意図的探索を行ったのではなく、普段の電車に乗っている最中に異常へ巻き込まれる。乗車は直接行為だが、異常経路への進入自体は本人の選択として提示されないため、受動発生をPrimaryとする。

# 12. D12. 作用対象

**誰／何に作用するか**

- Primary Child: `D12.FOC.PROTAGONIST_EXPERIENCER` — 主人公・体験者
- Primary Parent: `D12.FOC` — 焦点人物
- Secondary: なし
- Status: `D`

異常経路・駅・周辺環境によって直接影響を受ける中心対象は投稿者本人である。

# 13. D13. 作用機構

**因果源が対象へ何をするか**

- Primary Child: `D13.RST.SPATIAL_DISTORTION` — 空間異常
- Primary Parent: `D13.RST` — 現実・時空作用
- Secondary: なし
- Status: `I`

通常の鉄道路線では説明できない接続、見知らぬ駅への到達、帰還経路の不成立という構造から、距離・接続・地理を崩す空間異常としてコードする。初期版が「別世界への転移」を明言するわけではないため、現実置換までは付与しない。

# 14. D14. 帰結極性

**結果は正・負・中立・混合か**

- Value: `D14.NEG` — 負
- Status: `I`

投稿者は安全な帰宅に失敗し、未知の場所で不安と危険にさらされ、最終的に連絡が途絶える。最終状態は明示的な死亡ではないが、主帰結は負と判断する。

# 15. D15. 帰結領域

**何の領域が最終的に変わるか**

- Primary Child: `D15.LIF.DISAPPEARANCE` — 失踪・消失
- Primary Parent: `D15.LIF` — 人生・存在・同一性
- Secondary Child: `D15.KNW.UNCERTAINTY_PRESERVED` — 不確実性維持
- Secondary Parent: `D15.KNW` — 世界認識・知識
- Status: `I`

初期実況は投稿者のその後を確認できないまま終了し、伝承としては「消息が分からなくなる」ことが主要な終点になる。ただし物理的失踪そのものは外部確認されていないためStatusは`I`とし、真相が不明のまま残ることをSecondaryに保持する。

# 16. D16. 因果時間構造

**発動条件・原因成立から主作用／主帰結までの関係は、時間上どのように編成されているか**

- Primary Child: `D16.EVT.SEQUENTIAL_EPISODE` — エピソード内段階進行
- Primary Parent: `D16.EVT` — 単一エピソード・事象内
- Secondary: なし
- Status: `D`

一晩の実況の中で、長時間停車しない電車、未知の駅への到着、周辺探索、異常な遭遇、車への乗車、連絡途絶という複数段階が順に進む。長期遅延や周期ではなく、一つの連続エピソード内の段階進行である。

# 17. D17. 回避・制御方式

**結果をどう回避・制御・利用できるか**

- Code: なし
- Status: `U`

掲示板参加者は、戻る、線路を歩かない、警察や家族へ連絡する等の助言を行うが、初期実況の範囲では「この方法を使えば確実に帰還できる」という安定した制御規則は成立していない。助言が存在することだけから有効な回避法を推定しない。

# 18. D18. 作用レイヤー

**因果効力は伝承内／受容者／社会現実のどこに及ぶか**

- `D18.L1` — 伝承内因果層: `1`
- `D18.L2` — 受容者層: `0`
- `D18.L3` — 社会現実層: `0`
- Status: `D`

初期版の異常作用は投稿者と伝承内の場所・経路に生じる。読むだけで読者が同じ駅へ行くという因果規則は初期版にはなく、後年の観光・メディア反響も今回のVersion Scopeではコードしない。

# 19. D19. 流通範囲

**誰の間に伝承が流通するか**

- Primary Child: `D19.NET.OPEN_FORUM_WEB` — 公開Web・掲示板
- Primary Parent: `D19.NET` — ネットワーク公開圏
- Secondary: なし
- Status: `D`

初期版は公開掲示板上の不特定参加者間で流通・進行する。

# 20. D20. 特権情報保持者

**誰が真相・追加情報を持つか**

- Primary Child: `D20.UNK.NO_ONE_KNOWS` — 誰も知らない
- Primary Parent: `D20.UNK` — 到達不能・不明
- Secondary: なし
- Status: `I`

投稿者も掲示板参加者も、きさらぎ駅の正体や帰還方法を確定的には知らない。初期版内部には真相を説明する権威者・内部者が現れないため、真相保持者不在として推定する。

# 21. D21. 現実アンカー

**実在世界へどの程度固定されるか**

- Value: `D21.A2` — 具体的実在対象
- Status: `D`

投稿では新浜松発の私鉄など、具体的な実在交通圏を想起させる情報が提示される。一方、「きさらぎ駅」自体は実在確認されないため、史実や制度が因果構造の成立条件となるA3/A4までは上げない。
