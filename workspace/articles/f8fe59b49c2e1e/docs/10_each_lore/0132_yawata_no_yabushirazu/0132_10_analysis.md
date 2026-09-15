記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0132_00_contents.md`、R1/R2/R3/R4監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0132`
- `伝承エントリ名称`: 八幡の藪知らず
- `Macro_Category`: 場所・異界・祟り・禁忌
- `Entry_Type`: 禁足地伝承／場所伝説
- `Version_Scope`: 千葉県市川市八幡の八幡不知森について、中へ立ち入ってはならず、禁を破って入ると出られなくなる、または祟り・災いを受けるとする禁足地伝承。日本武尊・葛飾八幡宮・平将門／八門遁甲・行徳入会地等は競合する由来説として保持し、徳川光圀侵入譚は禁忌を物語化する主要派生として含む。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Primary Child / Value: `D01.G0` — 前近代（〜1867）
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 天保年間の『江戸名所図会』に禁足・祟りと複数由来説が記録されており、少なくとも前近代に共有核が存在した。刊行年を起源年とはしない。
- この伝承における現れ方: 現代に新規生成した怪談ではなく、江戸後期には地域伝承として成立済みだったことが確認できる。

### D02 最古確認流通媒体
- Primary Child / Value: `D02.PRT.BOOK` — 書籍
- Primary Parent: `D02.PRT`
- Secondary: なし
- Status: `I`
- 判定根拠: 公的書誌・復刻案内を通じ、天保年間の『江戸名所図会』に伝承が印刷・流通していたことを確認する。今回原本本文を直接閲覧していないためIとする。
- この伝承における現れ方: 地域の禁足知識が江戸後期には名所案内・地誌系印刷物へ入り、地域外にも流通可能になった。

### D03 確認流通媒体ポートフォリオ
- Primary Child / Value: `D03.PRT.BOOK` — 書籍
- Primary Parent: `D03.PRT`
- Secondary: `D03.WEB.WEBSITE` — Webサイト
- Status: `D`
- 判定根拠: 前近代地誌・後代出版と、現代の市川市公式Webで伝承の再提示を確認。錦絵は現taxonomyへ無理に割り当てない。
- この伝承における現れ方: 印刷文化から現代Webまで、媒体を変えながら禁足地としての核が継承される。

### D04 生成・変容パターン
- Primary Child / Value: `D04.CON.CONTESTED_VERSION` — 競合版併存
- Primary Parent: `D04.CON`
- Secondary: `D04.VAR.ACCRETION` — 増補; `D04.MED.CROSS_MEDIA` — クロスメディア化
- Status: `D/I`
- 判定根拠: 禁足理由について複数の由来説が現在も併存し、光圀譚などの物語が増補され、錦絵・文学・Webへ展開した。
- この伝承における現れ方: 「入るな」という核は維持される一方、なぜ危険なのかという説明が一つに収束しない。

### D05 提示形式
- Primary Child / Value: `D05.RUL.TABOO` — 禁忌
- Primary Parent: `D05.RUL`
- Secondary: `D05.HYB.NARRATIVE_EXPLANATION` — 物語＋解説
- Status: `D`
- 判定根拠: 最小核は「中へ入ってはいけない」という禁止規則。光圀譚や由来説が禁忌を物語・解説で補強する。
- この伝承における現れ方: 完全な由来を知らなくても「入るな」だけで伝承が機能する。

### D06 真実性提示
- Primary Child / Value: `D06.T3` — 共同体既知事実
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 江戸期資料では里人が立入を禁じる場所として記録され、現代でも地域で継承される既知の場所知識として提示される。
- この伝承における現れ方: 一人の体験談ではなく「ここは昔から入ってはいけない」という共同体的既知性が本当らしさを与える。

## 2.2. 意味形成

### D07 意味形成対象
- Primary Child / Value: `D07.PSE.DANGEROUS_PLACE` — 危険・怪異場所
- Primary Parent: `D07.PSE`
- Secondary: `D07.PSE.SPATIAL_ROUTE_ANOMALY` — 空間・経路異常
- Status: `D`
- 判定根拠: 中心問題は、市街地の小規模な竹藪がなぜ長期に危険な禁足地として区別されるのかという点。「入ると出られない」がその主要な空間異常型。
- この伝承における現れ方: 日常空間のすぐ隣が、境界を越えると通常とは異なる危険領域へ変わる。

### D08 意味形成契機
- Primary Child / Value: `D08.CLM.INHERITED_SAYING` — 既存の言い伝え
- Primary Parent: `D08.CLM`
- Secondary: `D08.HIS.PLACE_NAME_RUIN` — 地名・遺構・記念物
- Status: `D`
- 判定根拠: 「昔から入ってはいけない」という伝承が出発点で、名称・1857年石碑・鳥居・祠・竹藪が具体地点へ固定する手掛かりになる。
- この伝承における現れ方: 受容者は先に禁忌を知り、現地の境界物を見ることで通常の竹藪を禁足地として再認識する。

### D09 意味付与操作
- Primary Child / Value: `D09.NOR.TABOOIZATION` — 禁忌化
- Primary Parent: `D09.NOR`
- Secondary: `D09.HST.HISTORICAL_ANCHOR` — 歴史接続
- Status: `I`
- 判定根拠: 通常の小空間を越えてはいけない境界へ変換し、その理由を将門、日本武尊、八幡宮、光圀、入会地等へ歴史接続する。
- この伝承における現れ方: 土地の差異が「歴史的・宗教的理由があるため入るな」という規範へ変換される。

### D10 因果源存在論
- Primary Child / Value: `D10.SPC.SPECIFIC_PLACE` — 特定場所
- Primary Parent: `D10.SPC`
- Secondary: `D10.SUP.IMPERSONAL_CURSE` — 非人格的呪力
- Status: `I`
- 判定根拠: 八幡不知森という場所自体に異常効力が固定され、祟り型では人格霊を必須としない非人格的呪力が作用するとされる。
- この伝承における現れ方: 危険は特定人物に狙われることではなく、この境界を越えること自体に付着している。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary Child / Value: `D11.MOV.ENTER_PASS_CROSS` — 入る・通る・越える
- Primary Parent: `D11.MOV`
- Secondary: なし
- Status: `D`
- 判定根拠: 藪の内部へ立ち入ることが禁忌違反・作用条件。
- この伝承における現れ方: 外から見ることではなく境界を越える行為が因果系への入口となる。

### D12 作用対象
- Primary Child / Value: `D12.FOC.PROTAGONIST_EXPERIENCER` — 主人公・体験者
- Primary Parent: `D12.FOC`
- Secondary: なし
- Status: `D`
- 判定根拠: 禁を破って入った人物が迷失・祟り・警告の対象になる。
- この伝承における現れ方: 光圀譚では侵入者本人が具体的作用対象として物語化される。

### D13 作用機構
- Primary Child / Value: `D13.RST.SPATIAL_DISTORTION` — 空間異常
- Primary Parent: `D13.RST`
- Secondary: `D13.FAT.CURSE_MISFORTUNE` — 呪詛・不運付与
- Status: `D`
- 判定根拠: 「入ると出られない」という空間作用と「禁を破ると祟りがある」という吉凶作用の双方が資料上確認される。
- この伝承における現れ方: 小規模な藪で通常の出口関係が機能しない型と、禁忌違反に災いが返る型が併存する。

### D14 帰結極性
- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 脱出不能・祟り・災いはいずれも負の帰結。
- この伝承における現れ方: 禁を破る利益型は共有核にない。

### D15 帰結領域
- Primary Child / Value: `D15.OPP.LUCK_MISFORTUNE` — 幸運・不運
- Primary Parent: `D15.OPP`
- Secondary: `D15.LIF.DISAPPEARANCE` — 失踪・消失
- Status: `I`
- 判定根拠: 祟り型では結果種別が固定されず広い災いとして語られる一方、非帰還型は強い主要異伝。永久失踪・死亡を全型へ一般化しない。
- この伝承における現れ方: 禁忌違反には不特定の災いが予告され、強い型では共同体へ戻れない状態になる。

### D16 因果時間構造
- Primary Child / Value: `D16.STA.STATIC_RULE` — 静的規則
- Primary Parent: `D16.STA`
- Secondary: なし
- Status: `I`
- 判定根拠: 「この場所へ入れば、脱出不能・祟りが生じる」という恒常的条件規則で、特定の期限・潜伏期間・段階進行を必要としない。
- この伝承における現れ方: 時期に関係なく「入るな」という条件規則が成立する。

### D17 回避・制御方式
- Primary Child / Value: `D17.RUL.OBEY_TABOO` — 禁忌遵守
- Primary Parent: `D17.RUL`
- Secondary: なし
- Status: `D`
- 判定根拠: 最も一貫した回避法は立入禁止を守ること。解除儀礼・脱出手順は共有核にない。
- この伝承における現れ方: 危険を解決するより、境界を越えないことで因果系へ入らない。

### D18 作用レイヤー
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `1`
- Status: `I`
- 判定根拠: L1では侵入者への迷失・祟りがある。L2は知るだけで作用しない。L3は、江戸期の記録が里人が祟りを理由に現実に立入を禁じる行動規範を報告しており、伝承と場所利用行動の接続を示す歴史X Evidenceとして評価する。
- この伝承における現れ方: 怪異因果は物語内部だけでなく、「入らない」という現実の行動規範にも接続したと記録される。ただし現代の土地管理全体を伝承の効果とはしない。

## 2.4. 社会的分布・現実接続

### D19 流通範囲
- Primary Child / Value: `D19.LOC.LOCAL_TRADITION` — 地域伝承圏
- Primary Parent: `D19.LOC`
- Secondary: `D19.MAS.NATIONAL_PUBLIC` — 全国的大衆
- Status: `I`
- 判定根拠: 市川市八幡の地域史・場所に不可分な伝承である一方、江戸期名所図会・明治錦絵・現代Webを通じ地域外へ広域化した。
- この伝承における現れ方: 作用地点は局所的だが、話自体は広域の読者・観客へ流通する。

### D20 特権情報保持者
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
- 判定根拠: 禁足の真の起源について複数説が併存し、地元住民・神職・歴史研究者等の誰か一者へ真相保持を固定できない。「誰も知らない」と伝承内で明示されるわけでもない。
- この伝承における現れ方: 諸説があることと、特権的な真相保持者がいる／いないことを区別する。

### D21 現実アンカー
- Primary Child / Value: `D21.A2` — 具体的実在対象
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 八幡不知森は市川市八幡に実在し、竹藪・鳥居・祠・1857年石碑によって具体的に特定できる。由来異伝の歴史人物は共有核の成立条件ではないためA4にはしない。
- この伝承における現れ方: 「どこかの禁足地」ではなく、この実在地点であることがEntry識別条件。

# 3. QA・保留事項

- D01=G0/I。前近代存在は確認できるが起源年代は未確定。
- D02=BOOK/I。今回は原本本文を直接閲覧せず公的案内で固定。
- D04は競合由来説、増補、媒体横断を分離して保持。
- D18 L3=1は、江戸期記録が祟り信念と実際の立入回避を結び付けて記述する範囲に限定。
- D20=U。「複数説がある」ことを`NO_ONE_KNOWS`へ読み替えない。
- D21=A2。歴史由来説を史実的因果アンカーへ過剰昇格しない。
- 新規taxonomy gapなし。