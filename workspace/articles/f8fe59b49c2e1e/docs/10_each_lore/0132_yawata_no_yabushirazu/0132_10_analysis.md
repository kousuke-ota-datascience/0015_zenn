記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

伝承内容・典拠・異伝・証拠上の不確実性は `0132_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0132`
- `伝承エントリ名称`: 八幡の藪知らず
- `Macro_Category`: 場所・異界・祟り・禁忌
- `Entry_Type`: 禁足地伝承／場所伝説
- `Version_Scope`: 千葉県市川市八幡の八幡不知森について、中へ立ち入ってはならず、禁を破って入ると出られなくなる、または祟り・災いを受けるとする禁足地伝承。日本武尊・葛飾八幡宮・平将門／八門遁甲・行徳入会地等は異なる由来説明として保持し、徳川光圀侵入譚は主要派生として含む。各由来説を一つの史実へ統合しない。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示の次元

### 2.1.1. D01: 生成年代
**いつ成立したか**
- Primary Child / Value: `D01.G0` — 前近代（〜1867）
- Primary Parent: なし
- Secondary: なし
- Status: `I`
**判定根拠**: 市川市立図書館資料編が寛延2年（1749）『葛飾記』の八幡不知森記述を本文抜粋で提示し、禁足・非帰還・祟り・里諺という共有核を確認できる。これは少なくとも1749年までに共有核が存在したEarliest Attestationであり、生成年を1749年へ固定するEvidenceではないため `I`。
**この伝承における現れ方**: 少なくとも18世紀中葉には禁足地伝承として記録されていたが、実際の形成時期はそれ以前を含め不明。

### 2.1.2. D02: 最古確認流通媒体
**現在の証拠で最も古く確認できる流通媒体は何か**
- Primary Child / Value: `D02.PRT.BOOK` — 書籍
- Primary Parent: `D02.PRT`
- Secondary: なし
- Status: `I`
**判定根拠**: 1749年『葛飾記』に共有核が記録されていることを、市川市立図書館の公的な本文抜粋・翻刻紹介経由で固定する。原本を今回直接閲覧したわけではないため `I`。
**この伝承における現れ方**: 地域の里諺・禁足知識が18世紀中葉には地誌的印刷記録へ収録されていた。

### 2.1.3. D03: 確認流通媒体ポートフォリオ
**Version Scopeで確認できる媒体は何か**
- Primary Child / Value: `D03.PRT.BOOK`
- Primary Parent: `D03.PRT`
- Secondary: `D03.WEB.WEBSITE`
- Status: `I`
**判定根拠**: 『葛飾記』『江戸名所図会』等の前近代印刷資料と、現代の市川市・市川市立図書館Webで再提示を確認する。錦絵は現行D03 taxonomyへ無理に割り当てない。
**この伝承における現れ方**: 印刷資料と現代Webの双方で場所・禁足・由来説明が流通する。

### 2.1.4. D04: 生成・変容パターン
**時間とともにどう変形したか**
- Primary Child / Value: `D04.CON.CONTESTED_VERSION`
- Primary Parent: `D04.CON`
- Secondary: なし
- Status: `I`
**判定根拠**: 神聖地、将門、八門遁甲、日本武尊、入会地等の由来Versionが併存し、単一由来へ収束しない。1749年Evidenceを追加しても、どのVersionからどれが派生したかを示す連続History Evidenceはない。
**この伝承における現れ方**: 禁足規則を共有しつつ理由説明が競合する。

### 2.1.5. D05: 提示形式
**どんなコミュニケーション形式で提示されるか**
- Primary Child / Value: `D05.RUL.TABOO`
- Primary Parent: `D05.RUL`
- Secondary: `D05.HYB.NARRATIVE_EXPLANATION`
- Status: `D`
**判定根拠**: 1749年資料を含め、森へ入らないという共同体規範が共有核として記録される。光圀譚・由来説はその周囲の物語・説明。
**この伝承における現れ方**: 「中へ入ってはいけない」という禁足規則が中心。

### 2.1.6. D06: 真実性提示
**どんな本当らしさを要求するか**
- Primary Child / Value: `D06.T3`
- Primary Parent: なし
- Secondary: なし
- Status: `I`
**判定根拠**: 1749年『葛飾記』は昔からの里諺として、19世紀資料も里人が禁じる場所として記述する。個人の一回的体験ではなく共同体既知の場所として提示されるが、原発話の態度を直接採録したものではない。
**この伝承における現れ方**: 昔から知られた禁足地として提示される。

## 2.2. 意味形成の次元

### 2.2.1. D07: 意味形成対象
**何が不可解・不確実なのか**
- Primary Child / Value: `D07.PSE.DANGEROUS_PLACE`
- Primary Parent: `D07.PSE`
- Secondary: `D07.PSE.SPATIAL_ROUTE_ANOMALY`
- Status: `D`
**判定根拠**: 実在する藪が危険な禁足地として区別され、1749年資料でも入れば出ない／死して出ないとされる。
**この伝承における現れ方**: 市街地内の具体的小空間が非帰還・祟りの場所として扱われる。

### 2.2.2. D08: 意味形成契機
**何を手掛かりに問題化されるか**
- Primary Child / Value: `D08.CLM.INHERITED_SAYING`
- Primary Parent: `D08.CLM`
- Secondary: `D08.HIS.PLACE_NAME_RUIN`
- Status: `D`
**判定根拠**: 1749年『葛飾記』が「昔より里諺」として共有核を記録し、名称・石碑・現存地点が具体場所を固定する手掛かりになる。
**この伝承における現れ方**: 既存の言い伝えと実在地点・標識が対応する。

### 2.2.3. D09: 意味付与操作
**不可解なものをどう理解可能にするか**
- Primary Child / Value: `D09.NOR.TABOOIZATION`
- Primary Parent: `D09.NOR`
- Secondary: `D09.HST.HISTORICAL_ANCHOR`
- Status: `I`
**判定根拠**: 特定空間への侵入を禁止行為として規範化し、その理由を複数の歴史・宗教的要素へ接続する。
**この伝承における現れ方**: 立入制限が由来物語と結び付けられる。

### 2.2.4. D10: 因果源存在論
**原因を何として世界に置くか**
- Primary Child / Value: `D10.SPC.SPECIFIC_PLACE`
- Primary Parent: `D10.SPC`
- Secondary: `D10.SUP.IMPERSONAL_CURSE`
- Status: `I`
**判定根拠**: 八幡不知森という特定場所そのものへ異常効力が固定され、祟り型では非人格的災いとして語られる。
**この伝承における現れ方**: 任意の藪ではなく八幡不知森への侵入が危険。

## 2.3. 因果・行動モデルの次元

### 2.3.1. D11: 発動・接触条件
**何を契機に因果系へ入るか**
- Primary Child / Value: `D11.MOV.ENTER_PASS_CROSS`
- Primary Parent: `D11.MOV`
- Secondary: なし
- Status: `D`
**判定根拠**: 藪内部へ立ち入ることが非帰還・祟りの条件。
**この伝承における現れ方**: 境界を越えて内部へ入ることが入口。

### 2.3.2. D12: 作用対象
**誰／何に作用するか**
- Primary Child / Value: `D12.FOC.PROTAGONIST_EXPERIENCER`
- Primary Parent: `D12.FOC`
- Secondary: なし
- Status: `D`
**判定根拠**: 禁を破って内部へ入った人物自身が作用対象。
**この伝承における現れ方**: 侵入者本人が非帰還・祟り・警告を受ける。

### 2.3.3. D13: 作用機構
**どのように作用するか**
- Primary Child / Value: `D13.RST.SPATIAL_DISTORTION`
- Primary Parent: `D13.RST`
- Secondary: `D13.FAT.CURSE_MISFORTUNE`
- Status: `D`
**判定根拠**: 入ると出られないという空間作用と、祟り・災いがあるという作用が歴史資料で確認できる。
**この伝承における現れ方**: 非帰還型と祟り型を併存させる。

### 2.3.4. D14: 帰結極性
**結果はどの方向か**
- Primary Child / Value: `D14.NEG`
- Primary Parent: なし
- Secondary: なし
- Status: `D`
**判定根拠**: 非帰還・祟り・災いはいずれも負。
**この伝承における現れ方**: 禁忌違反は危害へ結び付く。

### 2.3.5. D15: 帰結領域
**最終的に何が起きるか**
- Primary Child / Value: `D15.OPP.LUCK_MISFORTUNE`
- Primary Parent: `D15.OPP`
- Secondary: `D15.LIF.DISAPPEARANCE`
- Status: `I`
**判定根拠**: 祟り型では広い災い、非帰還型では戻れない結果が記録される。1749年には死して出ないという強い表現もあるが、死亡を全Versionへ一般化しない。
**この伝承における現れ方**: 不運・祟りを主帰結とし、強い型で非帰還を伴う。

### 2.3.6. D16: 因果時間構造
**時間上どう編成されるか**
- Primary Child / Value: `D16.STA.STATIC_RULE`
- Primary Parent: `D16.STA`
- Secondary: なし
- Status: `I`
**判定根拠**: 「入れば出られない／祟りがある」という恒常的条件規則で、期限・段階を必須としない。
**この伝承における現れ方**: 境界侵入と負の帰結が静的規則として結び付く。

### 2.3.7. D17: 回避・制御方式
**どう回避・制御できるか**
- Primary Child / Value: `D17.RUL.OBEY_TABOO`
- Primary Parent: `D17.RUL`
- Secondary: なし
- Status: `D`
**判定根拠**: 1749年資料では昔から入る者がいないとされ、19世紀資料でも里人が祟りを理由に立入を禁じる。これはD11の単純反転ではなく独立した共同体行動Evidence。
**この伝承における現れ方**: 境界を越えないことが共同体規範として実行される。

### 2.3.8. D18: 作用レイヤー
**作用はどのレイヤーまで確認できるか**
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `1`
- Status: `I`
**判定根拠**: L1は侵入者への非帰還・祟り。L2は情報接触だけで怪異が発動する規則なし。L3は1749年・19世紀記録で、里人が伝承・祟り観念と結び付けて実際に立ち入らない／立入を禁じる共同体行動が記録されるため、独立した歴史X Evidenceとして限定的に採用する。
**この伝承における現れ方**: 怪異因果と現実の場所利用規範を分離しつつ、歴史記録上の接続を認める。

## 2.4. 社会的分布・現実接続の次元

### 2.4.1. D19: 流通範囲
**どの範囲で流通するか**
- Primary Child / Value: `D19.LOC.LOCAL_TRADITION`
- Primary Parent: `D19.LOC`
- Secondary: `D19.MAS.NATIONAL_PUBLIC`
- Status: `I`
**判定根拠**: 市川市八幡の具体地点に不可分な地域伝承だが、地誌・錦絵・現代Webを通じ地域外へ提示された。
**この伝承における現れ方**: 作用地点は局所的、物語流通は地域外にも及ぶ。

### 2.4.2. D20: 特権情報保持者
**誰が特権的情報を保持するか**
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
**判定根拠**: 禁足の真の起源には複数説があり、誰か一者へ真相保持を固定できない。「誰も知らない」と明示されるわけでもない。
**この伝承における現れ方**: 由来不明と特権保持者不明を区別する。

### 2.4.3. D21: 現実アンカー
**どの程度具体的現実対象へ固定されるか**
- Primary Child / Value: `D21.A2`
- Primary Parent: なし
- Secondary: なし
- Status: `D`
**判定根拠**: 八幡不知森は市川市八幡に実在し、竹藪・鳥居・祠・石碑等で具体的に特定できる。
**この伝承における現れ方**: 八幡不知森という実在地点がEntry識別条件。
