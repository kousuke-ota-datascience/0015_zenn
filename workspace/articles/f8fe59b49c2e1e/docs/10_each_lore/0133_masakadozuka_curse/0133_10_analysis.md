記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0133_00_contents.md`、R1/R2/R3/R4監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0133`
- `伝承エントリ名称`: 将門塚の祟り
- `Macro_Category`: 歴史人物・怨霊・場所・禁忌
- `Entry_Type`: 祟り伝承／場所伝説
- `Version_Scope`: 東京都千代田区大手町の将門塚を粗末に扱う、撤去・破壊する、供養を怠る等の不敬に対し、平将門の御霊・怨霊が事故・病気・災厄等の不幸を与えるとする祟り伝承。鎮魂・供養・尊重・保存を対抗規則として含む。高権威資料で固定できない個別事故逸話や「一切工事できない」という強い版は共有核に含めない。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Primary Child / Value: `D01.G0` — 前近代（〜1867）
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 神田明神社伝は1307年の将門公鎮魂と1309年の奉祀を伝え、現行板石塔婆も1307年板碑拓本に基づくと説明する。今回1307年原資料そのものは直接検証していないためI。
- この伝承における現れ方: 近代都市開発で新規生成しただけの噂ではなく、前近代の御霊・鎮魂伝承を基盤とする。

### D02 最古確認流通媒体
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
- 判定根拠: 中世以来の継承は確認できるが、最古の流通回路が口承、寺社縁起、板碑、記録文書等のいずれかを現在Evidenceから直接固定できない。
- この伝承における現れ方: 伝承の古さと媒体の確定を分離し、古いから口承と推定しない。

### D03 確認流通媒体ポートフォリオ
- Primary Child / Value: `D03.WEB.WEBSITE` — Webサイト
- Primary Parent: `D03.WEB`
- Secondary: なし
- Status: `D`
- 判定根拠: 神田明神公式、神田明神1300年事業、千代田区観光協会の一般向けWebで首伝説・祟り・鎮魂・保存史が直接再提示される。
- この伝承における現れ方: 古い宗教伝承が現代には公的・宗教組織のWebを通じて広域受容者へ流通する。

### D04 生成・変容パターン
- Primary Child / Value: `D04.REC.EVENT_ATTACHMENT` — 実事件への付着
- Primary Parent: `D04.REC`
- Secondary: `D04.REC.CONTEXT_UPDATE` — 時代適応; `D04.VAR.ACCRETION` — 増補
- Status: `I`
- 判定根拠: 中世の「荒廃→災厄→鎮魂」モデルへ関東大震災後の整地・官庁地再建等が接続され、都市開発文脈へ更新されながら新しい祟り逸話を吸収する。
- この伝承における現れ方: 時代ごとの現実の出来事が、古い将門怨霊モデルの新しい解釈材料になる。

### D05 提示形式
- Primary Child / Value: `D05.RUL.TABOO` — 禁忌
- Primary Parent: `D05.RUL`
- Secondary: `D05.HYB.NARRATIVE_EXPLANATION` — 物語＋解説
- Status: `I`
- 判定根拠: 行動的な核は「将門塚を粗末に扱うな／無礼に撤去・破壊するな」。首伝説、中世鎮魂、近現代の出来事が理由を説明する。
- この伝承における現れ方: 個々の事故逸話を知らなくても、塚への敬意という規則だけで伝承が機能する。

### D06 真実性提示
- Primary Child / Value: `D06.T5` — 条件付き信念
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 公的観光資料も超自然部分を伝説・言い伝えとして扱い、「粗末に扱えば祟る」という条件付き信念として維持する。
- この伝承における現れ方: 完全な事実認定を要求せず、「念のため敬意を払う」という行動に十分な信念強度で機能する。

## 2.2. 意味形成

### D07 意味形成対象
- Primary Child / Value: `D07.DCF.MISFORTUNE_STREAK` — 不運・成功失敗の偏り
- Primary Parent: `D07.DCF`
- Secondary: `D07.PSE.DANGEROUS_PLACE` — 危険・怪異場所
- Status: `I`
- 判定根拠: 塚の荒廃・改変と前後して語られる災厄・不幸のまとまりを、偶然ではなく祟りとして理解する。結果として塚は慎重に扱うべき場所となる。
- この伝承における現れ方: 異質な不幸が「将門塚に手を出した」という共通原因へ束ねられる。

### D08 意味形成契機
- Primary Child / Value: `D08.HIS.HISTORICAL_EVENT` — 歴史事件・戦争・災害
- Primary Parent: `D08.HIS`
- Secondary: `D08.HIS.PLACE_NAME_RUIN` — 地名・遺構・記念物
- Status: `I`
- 判定根拠: 平将門の死、関東大震災、塚の整地・再建等の歴史と、現在も残る将門塚・碑・祭祀が祟り解釈の手掛かりとなる。
- この伝承における現れ方: 抽象的怨霊話ではなく、具体的史跡と歴史出来事が意味形成を支える。

### D09 意味付与操作
- Primary Child / Value: `D09.CAU.DIRECT_CAUSE` — 直接原因化
- Primary Parent: `D09.CAU`
- Secondary: `D09.HST.HISTORICAL_ANCHOR` — 歴史接続; `D09.NOR.TABOOIZATION` — 禁忌化
- Status: `I`
- 判定根拠: 災害・事故・不幸を「将門の御霊を粗末にしたため」と因果接続し、平将門・首塚・都市史へ結び付け、「無礼に扱うな」という規範を形成する。
- この伝承における現れ方: 偶然の出来事が将門の怒りという歴史的人格因果へ変換される。

### D10 因果源存在論
- Primary Child / Value: `D10.SUP.GHOST_SPIRIT` — 幽霊・死者霊
- Primary Parent: `D10.SUP`
- Secondary: `D10.SPC.SPECIFIC_PLACE` — 特定場所
- Status: `I`
- 判定根拠: 祟りの中心主体は平将門の御霊・怨霊で、その作用が将門塚という具体地点へ固定される。
- この伝承における現れ方: 場所だけが無人格に作用するのではなく、将門の死者霊が怒り・鎮まる主体として置かれる。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary Child / Value: `D11.MAN.CREATE_ALTER_MANIPULATE` — 作る・加工・操作する
- Primary Parent: `D11.MAN`
- Secondary: なし
- Status: `I`
- 判定根拠: 塚を荒らす、撤去・破壊する、無礼に改変することが代表的発動条件。供養を怠るという不作為は現taxonomyに完全対応するChildがないため混成しない。
- この伝承における現れ方: 通常の都市整備でも、将門塚へ無礼に手を加える場合は禁忌違反として扱われる。

### D12 作用対象
- Primary Child / Value: `D12.OTH.VICTIM_TARGET` — 特定被害者
- Primary Parent: `D12.OTH`
- Secondary: なし
- Status: `I`
- 判定根拠: 祟りは塚の荒廃・撤去・改変に関与した人物へ向かうとされる。高権威Evidenceで固定できない具体的人名・被害種別は一般化しない。
- この伝承における現れ方: 「塚に手を出した者」が祟り対象として選ばれる。

### D13 作用機構
- Primary Child / Value: `D13.FAT.CURSE_MISFORTUNE` — 呪詛・不運付与
- Primary Parent: `D13.FAT`
- Secondary: なし
- Status: `D`
- 判定根拠: 将門の御霊が禁忌違反者へ事故・病気・災厄等の不幸を与えるという祟り作用が共有核。
- この伝承における現れ方: 具体的な物理機構ではなく、異なる不幸を同じ祟り作用へ回収する。

### D14 帰結極性
- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 祟り・災厄はいずれも負。
- この伝承における現れ方: 塚への無礼による利益型は共有核にない。

### D15 帰結領域
- Primary Child / Value: `D15.OPP.LUCK_MISFORTUNE` — 幸運・不運
- Primary Parent: `D15.OPP`
- Secondary: なし
- Status: `I`
- 判定根拠: 異伝ごとに事故・病気・死亡等が変わるため、具体的身体被害へ固定せず広い不運・災厄を主帰結とする。
- この伝承における現れ方: 「何が起きるか」は変動しても「良くないことが起きる」は安定する。

### D16 因果時間構造
- Primary Child / Value: `D16.REC.TRIGGERED_RECURRENCE` — 条件再発
- Primary Parent: `D16.REC`
- Secondary: なし
- Status: `I`
- 判定根拠: 塚の荒廃・無礼な改変という条件が時代ごとに再成立するたび、新しい災厄が同じ祟りモデルへ回収される反復可能な構造を持つ。
- この伝承における現れ方: 中世の荒廃、近代の整地、後代の工事などへ「また塚に手を加えた→また祟り」という因果を再利用できる。

### D17 回避・制御方式
- Primary Child / Value: `D17.RIT.RITUAL_CLOSURE` — 儀式終了・祓い
- Primary Parent: `D17.RIT`
- Secondary: `D17.CST.ONGOING_MANAGEMENT` — 継続管理
- Status: `D`
- 判定根拠: 1307年鎮魂伝承、1925年慰霊祭、現行例祭・保存活動から、供養・鎮魂と継続的尊重・保存が制御方法として制度化されている。
- この伝承における現れ方: 祟りは破壊して消す対象ではなく、祭り・保存しながら関係を維持する対象となる。

### D18 作用レイヤー
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `1`
- Status: `D`
- 判定根拠: L1では不敬に対する祟り。L2は話を知るだけでは発動しない。L3は神田明神公式が慰霊祭・将門塚例祭・史蹟将門塚保存会・1961年以降の複数回整備と2021年改修を、将門の鎮魂・祭祀・保存に関わる現実実践として記録する。
- この伝承における現れ方: 祟り・信仰の意味体系が、現実の祭祀・保存・改修の方法へ接続している。改修自体は禁止されていない。

## 2.4. 社会的分布・現実接続

### D19 流通範囲
- Primary Child / Value: `D19.LOC.LOCAL_TRADITION` — 地域伝承圏
- Primary Parent: `D19.LOC`
- Secondary: `D19.MAS.NATIONAL_PUBLIC` — 全国的大衆
- Status: `I`
- 判定根拠: 大手町・神田地域の祭祀・保存史に根差す一方、公的観光情報等を通じ全国的に認識可能な著名伝承へ拡大している。
- この伝承における現れ方: 作用地点は一つの塚に固定されるが、話は地域外へ広く流通する。

### D20 特権情報保持者
- Primary Child / Value: `D20.EXP.RELIGIOUS_FOLKLORE` — 宗教・伝承専門家
- Primary Parent: `D20.EXP`
- Secondary: `D20.ORG.INSTITUTION_AUTHORITY` — 組織・権限主体
- Status: `I`
- 判定根拠: 鎮魂祭祀・由緒の体系は神田明神等の宗教主体が保持し、保存・改修の具体的実務は史蹟将門塚保存会等の組織主体に偏在する。祟りの超自然的真実を独占するという意味ではない。
- この伝承における現れ方: 一般受容者は概要を知れるが、どのように祭祀・保存を行うかという実務知は専門主体に集中する。

### D21 現実アンカー
- Primary Child / Value: `D21.A4` — 史実・記録・既存伝承を因果統合
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 平将門という歴史人物、実在将門塚、関東大震災、官庁地再建、慰霊祭、保存改修等の実在史を祟り・鎮魂の因果物語へ組み込む。実在史が超自然因果を実証するわけではない。
- この伝承における現れ方: 史跡だけでなく確認可能な歴史・祭祀・保存記録群が伝承を現実世界へ強く固定する。

# 3. QA・保留事項

- D01=G0/I。1307原資料そのものは未検証。
- D02=U。伝承の古さから最古媒体を推測しない。
- D04は実事件への付着を主変容とし、時代適応・増補を分離。
- D11は撤去・破壊型を既存Childで記述できるが、「供養を怠る／維持を放棄する」という不作為条件を直接表すChild不足をGlobal Reconciliationのtaxonomy gap候補とする。
- D16は同じ禁忌条件の再成立ごとに祟りが再利用されるためTRIGGERED_RECURRENCE。
- D18 L3=1は公式記録された慰霊・祭祀・保存実践に基づく。超自然因果の実在性とは独立。
- 公式整備史により「一切工事できない」という版は採らない。