# 0133 将門塚の祟り — R3 Independent Recode

> **Freeze rule:** 本文はR1 Evidence正本とR2 Version Scopeだけを入力に作成した。旧 `0133_10_analysis.md` および旧Excel coding値は未参照。

## 1. Entry基本情報

- Entry_ID: `0133`
- 伝承名: 将門塚の祟り
- Macro_Category: `歴史人物・怨霊・場所・禁忌`
- Entry_Type: `祟り伝承／場所伝説`
- Version_Scope: 大手町の将門塚を粗末に扱う、撤去・破壊する、供養を怠る等の不敬に対し、平将門の御霊・怨霊が災厄を与えるとする祟り伝承。鎮魂・供養・尊重・保存を対抗規則として含む。

## 2. D01〜D21 独立判定

### D01 生成年代
- Value: `D01.G0` — 前近代（〜1867）
- Status: `I`
- 根拠: 神田明神社伝は1307年の祟り鎮魂と1309年の奉祀を伝え、現行板石塔婆も1307年板碑拓本に基づくと説明する。原資料自体を今回直接検証していないため、前近代存在はIで判定する。

### D02 最古確認流通媒体
- Primary: なし
- Parent: なし
- Status: `U`
- 根拠: 中世以来の祭祀・伝承継承は確認できるが、現在のEvidenceから「最古の人から人への流通媒体」を口承・石碑・書籍等のいずれかへ安全に固定できない。現代Webを最古媒体へ置換しない。

### D03 確認流通媒体ポートフォリオ
- Primary: `D03.WEB.WEBSITE` — Webサイト
- Parent: `D03.WEB`
- Secondary: なし
- Status: `D`
- 根拠: 神田明神公式、神田明神1300年事業、千代田区観光協会の一般向けWebで祟り・鎮魂・首伝説が直接再提示される。歴史的媒体は今回のコードEvidenceとして直接固定しない。

### D04 生成・変容パターン
- Primary: `D04.REC.EVENT_ATTACHMENT` — 実事件への付着
- Parent: `D04.REC`
- Secondary: `D04.REC.CONTEXT_UPDATE` — 時代適応; `D04.VAR.ACCRETION` — 増補
- Status: `I`
- 根拠: 中世の祟り・鎮魂モデルへ、関東大震災後の整地・官庁地再建・近現代保存史が接続され、都市開発文脈に適応しながら具体逸話が増補される。

### D05 提示形式
- Primary: `D05.RUL.TABOO` — 禁忌
- Parent: `D05.RUL`
- Secondary: `D05.HYB.NARRATIVE_EXPLANATION` — 物語＋解説
- Status: `I`
- 根拠: 実践核は「将門塚を粗末に扱う／無礼に破壊・撤去してはならない」という禁忌。首伝説・中世鎮魂・近現代事件がその由来説明として語られる。

### D06 真実性提示
- Value: `D06.T5` — 条件付き信念
- Status: `I`
- 根拠: 「塚を粗末に扱えば祟りがある」という条件付き信念として機能し、現実の事故・災害を超自然因果の実証とはしない。

### D07 意味形成対象
- Primary: `D07.DCF.MISFORTUNE_STREAK` — 不運・成功失敗の偏り
- Parent: `D07.DCF`
- Secondary: `D07.PSE.DANGEROUS_PLACE` — 危険・怪異場所
- Status: `I`
- 根拠: 塚の荒廃・改変に関連して語られる災厄・不幸のまとまりを、偶然ではなく祟りとして理解することが中心。結果として塚は慎重に扱うべき危険地点となる。

### D08 意味形成契機
- Primary: `D08.HIS.DEATH_CRIME_HISTORY` — 死亡・犯罪履歴
- Parent: `D08.HIS`
- Secondary: `D08.HIS.HISTORICAL_EVENT` — 歴史事件・戦争・災害
- Status: `I`
- 根拠: 平将門の討死・首晒し・首飛来伝説という死の履歴と、関東大震災・都市再建等の歴史事件が祟り解釈の手掛かりになる。

### D09 意味付与操作
- Primary: `D09.CAU.DIRECT_CAUSE` — 直接原因化
- Parent: `D09.CAU`
- Secondary: `D09.HST.HISTORICAL_ANCHOR` — 歴史接続; `D09.NOR.TABOOIZATION` — 禁忌化
- Status: `I`
- 根拠: 現実の不幸・災害を将門の御霊へ因果帰属し、平将門・首塚・都市史へ接続しながら「無礼に扱うな」という規範を形成する。

### D10 因果源存在論
- Primary: `D10.SUP.GHOST_SPIRIT` — 幽霊・死者霊
- Parent: `D10.SUP`
- Secondary: `D10.SPC.SPECIFIC_PLACE` — 特定場所
- Status: `D/I`
- 根拠: 神田明神側の伝承は将門の御霊を鎮めると明示し、作用主体は死者霊としての平将門。効力は将門塚という特定地点へ固定される。

### D11 発動・接触条件
- Primary: `D11.MAN.CREATE_ALTER_MANIPULATE` — 作る・加工・操作する
- Parent: `D11.MAN`
- Secondary: なし
- Status: `I`
- 根拠: 塚を破壊・撤去・粗雑に改変する行為が代表的な発動条件。供養放棄という不作為は現taxonomyで直接表現しにくいためPrimaryへ混成しない。

### D12 作用対象
- Primary: `D12.OTH.VICTIM_TARGET` — 特定被害者
- Parent: `D12.OTH`
- Secondary: なし
- Status: `I`
- 根拠: 祟りは塚へ無礼な行為をした関係者・当事者へ向かうとされる。具体的被害者逸話はEvidence不足のため一般化しない。

### D13 作用機構
- Primary: `D13.FAT.CURSE_MISFORTUNE` — 呪詛・不運付与
- Parent: `D13.FAT`
- Secondary: なし
- Status: `D/I`
- 根拠: 将門の御霊が事故・病気・災厄等の不幸を与えるという非物理的祟り作用が共有核。

### D14 帰結極性
- Value: `D14.NEG`
- Status: `D`
- 根拠: 祟りの帰結は災厄・事故・病気等の負。

### D15 帰結領域
- Primary: `D15.OPP.LUCK_MISFORTUNE` — 幸運・不運
- Parent: `D15.OPP`
- Secondary: なし
- Status: `I`
- 根拠: 共有核では具体的傷害・死亡へ固定せず、広い災い・不幸として保持する。

### D16 因果時間構造
- Primary: `D16.STA.STATIC_RULE` — 静的規則
- Parent: `D16.STA`
- Secondary: なし
- Status: `I`
- 根拠: 「塚を粗末に扱えば祟られる」という恒常的条件規則が中心で、一定の期限・潜伏期間を共有核としない。

### D17 回避・制御方式
- Primary: `D17.RIT.RITUAL_CLOSURE` — 儀式終了・祓い
- Parent: `D17.RIT`
- Secondary: `D17.CST.ONGOING_MANAGEMENT` — 継続管理
- Status: `D`
- 根拠: 1307年鎮魂伝承、1925年慰霊祭、現行例祭・保存改修から、供養・鎮魂と継続的な尊重・保存管理が制御方法として明示される。

### D18 作用レイヤー
- Value: `D18.L1=1; D18.L2=0; D18.L3=1`
- Status: `I`
- 根拠: L1では無礼な改変に対する祟り。L2は伝承を聞くだけで危害を受けない。L3は、慰霊祭・例祭・史蹟保存会・複数回の保存改修が、将門を祀り尊重する現実の社会的・宗教的実践として公式記録されるため1。

### D19 流通範囲
- Primary: `D19.MAS.CROSS_GENERATIONAL` — 世代横断的大衆
- Parent: `D19.MAS`
- Secondary: `D19.LOC.LOCAL_TRADITION` — 地域伝承圏
- Status: `I`
- 根拠: 中世鎮魂伝承から江戸期祭祀、近現代保存史・現代Webまで長期継承される一方、大手町・神田明神の地域宗教文化に固定される。

### D20 特権情報保持者
- Primary: `D20.EXP.RELIGIOUS_FOLKLORE` — 宗教・伝承専門家
- Parent: `D20.EXP`
- Secondary: なし
- Status: `I`
- 根拠: 真教上人による鎮魂譚、神田明神による祭祀・例祭、寺院・神職が担う鎮魂実践から、制御・祭祀知識は宗教専門家へ制度化される。ただし祟りの超自然的真実を独占するという意味ではない。

### D21 現実アンカー
- Value: `D21.A4` — 史実・記録・既存伝承を因果統合
- Status: `I`
- 根拠: 平将門という歴史人物、大手町の実在塚、関東大震災・官庁再建・保存改修という実在史を、祟り・鎮魂の因果説明へ組み込む。これは超自然因果の史実認定ではない。

## 3. QA前メモ

- D01=G0/I。1307原資料自体は未検証。
- D02=U。古さから口承・石碑・書籍を推定しない。
- D04は中世祟りモデルが近現代の災害・都市開発へ再文脈化された点を中心にする。
- D11では「供養を怠る」という不作為を既存Childへ無理に符号化しない。必要ならGlobal Reconciliation候補。
- D17とD18 L3は、現実に継続する鎮魂・例祭・保存実践という公式X Evidenceを根拠とする。
- D21=A4は、現実史を超自然因果の証明にするのではなく、伝承が現実史を因果物語へ取り込む構造を表す。

**R3 Independent Recode: FROZEN.**