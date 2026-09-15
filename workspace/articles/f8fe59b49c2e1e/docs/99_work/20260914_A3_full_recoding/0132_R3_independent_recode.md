# 0132 八幡の藪知らず — R3 Independent Recode

> **Freeze rule:** 本文はR1 Evidence正本とR2 Version Scopeだけを入力に作成した。旧 `0132_10_analysis.md` および旧Excel coding値は未参照。

## 1. Entry基本情報

- Entry_ID: `0132`
- 伝承名: 八幡の藪知らず
- Macro_Category: `場所・異界・祟り・禁忌`
- Entry_Type: `禁足地伝承／場所伝説`
- Version_Scope: 市川市八幡の八幡不知森について、中へ立ち入ってはならず、入ると出られなくなる、または祟り・災いを受けるとする禁足地伝承。複数由来説と徳川光圀侵入譚を主要派生として含む。

## 2. D01〜D21 独立判定

### D01 生成年代
- Value: `D01.G0` — 前近代（〜1867）
- Status: `D`
- 根拠: 天保年間の『江戸名所図会』に禁足・祟りと複数由来説が記録され、少なくとも江戸後期には共有核が存在したことを直接確認できる。刊行年を起源年とはしない。

### D02 最古確認流通媒体
- Primary: `D02.PRT.BOOK` — 書籍
- Parent: `D02.PRT`
- Status: `D`
- 根拠: 現在固定できる最古の実流通回路として『江戸名所図会』の印刷物を確認。

### D03 確認流通媒体ポートフォリオ
- Primary: `D03.PRT.BOOK`
- Parent: `D03.PRT`
- Secondary: `D03.WEB.WEBSITE`
- Status: `D`
- 根拠: 前近代の地誌・後代出版に加え、現在は市川市公式Webでも伝承が一般向けに再提示される。錦絵は現taxonomyの媒体Childへ無理に押し込まない。

### D04 生成・変容パターン
- Primary: `D04.CON.CONTESTED_VERSION` — 競合版併存
- Parent: `D04.CON`
- Secondary: `D04.VAR.ACCRETION` — 増補; `D04.MED.CROSS_MEDIA` — クロスメディア化
- Status: `D/I`
- 根拠: 同じ禁足規則に対し、日本武尊・八幡宮旧地・平将門／八門遁甲・入会地など競合する由来説が併存し、光圀譚等の物語が増補され、錦絵・文学・現代Webへ展開する。

### D05 提示形式
- Primary: `D05.RUL.TABOO` — 禁忌
- Parent: `D05.RUL`
- Secondary: `D05.HYB.NARRATIVE_EXPLANATION` — 物語＋解説
- Status: `D`
- 根拠: 最小核は「中へ入ってはいけない」という禁止規則。光圀譚や由来説は禁忌を説明する物語・解説として付加される。

### D06 真実性提示
- Value: `D06.T3` — 共同体既知事実
- Status: `I`
- 根拠: 『江戸名所図会』では里人が祟りを理由に入ることを禁じる地域既知の規則として記録され、現代でも地域名所伝承として継承される。

### D07 意味形成対象
- Primary: `D07.PSE.SPATIAL_ROUTE_ANOMALY` — 空間・経路異常
- Parent: `D07.PSE`
- Secondary: `D07.PSE.DANGEROUS_PLACE` — 危険・怪異場所
- Status: `C`
- 根拠: 「狭い藪なのに入ると出られない」という空間異常型と、「入れば祟りがある」という危険場所型が主要な共有異伝として併存する。

### D08 意味形成契機
- Primary: `D08.CLM.INHERITED_SAYING` — 既存の言い伝え
- Parent: `D08.CLM`
- Secondary: `D08.HIS.PLACE_NAME_RUIN` — 地名・遺構・記念物
- Status: `D/I`
- 根拠: 「入ってはいけない」という既存伝承が解釈の出発点で、現地の藪・鳥居・祠・1857年石碑がその継続を物的にアンカーする。

### D09 意味付与操作
- Primary: `D09.NOR.TABOOIZATION` — 禁忌化
- Parent: `D09.NOR`
- Secondary: `D09.HST.HISTORICAL_ANCHOR` — 歴史接続; `D09.CAU.HIDDEN_CAUSE` — 隠れた原因化
- Status: `D/I`
- 根拠: 実在小空間を越えてはいけない境界へ変換し、その理由を将門・八幡宮・日本武尊・入会地等の歴史／社会史へ接続して説明する。

### D10 因果源存在論
- Primary: `D10.SPC.SPECIFIC_PLACE` — 特定場所
- Parent: `D10.SPC`
- Secondary: `D10.SUP.IMPERSONAL_CURSE` — 非人格的呪力
- Status: `C`
- 根拠: 最小核では場所そのものが禁足効力を持つように語られる一方、祟り型では非人格的呪力が原因として置かれる。由来説ごとの人格主体は共有核に固定できない。

### D11 発動・接触条件
- Primary: `D11.MOV.ENTER_PASS_CROSS` — 入る・通る・越える
- Parent: `D11.MOV`
- Secondary: なし
- Status: `D`
- 根拠: 藪の内部へ立ち入ること自体が禁忌破り・作用条件。

### D12 作用対象
- Primary: `D12.FOC.PROTAGONIST_EXPERIENCER` — 主人公・体験者
- Parent: `D12.FOC`
- Secondary: なし
- Status: `D`
- 根拠: 禁を破って入った人物が迷失・祟り・警告の対象になる。光圀譚でも侵入者本人が対象。

### D13 作用機構
- Primary: `D13.RST.SPATIAL_DISTORTION` — 空間異常
- Parent: `D13.RST`
- Secondary: `D13.FAT.CURSE_MISFORTUNE` — 呪詛・不運付与
- Status: `C`
- 根拠: 「入ると出られない」型では空間・経路が正常に機能しないことが作用機構であり、祟り型では非物理的不運が作用する。主要異伝間で機構が競合する。

### D14 帰結極性
- Value: `D14.NEG`
- Status: `D`
- 根拠: 迷失・祟り・災いという負の帰結が中心。

### D15 帰結領域
- Primary: `D15.LIF.DISAPPEARANCE` — 失踪・消失
- Parent: `D15.LIF`
- Secondary: `D15.OPP.LUCK_MISFORTUNE` — 幸運・不運
- Status: `C`
- 根拠: 「出られない」型では帰還不能・迷失が中心、祟り型では一般的災いが中心で、Scope内で帰結領域が競合する。死亡までは共通しない。

### D16 因果時間構造
- Primary: `D16.STA.STATIC_RULE` — 静的規則
- Parent: `D16.STA`
- Secondary: なし
- Status: `D`
- 根拠: 「この場所へ入れば出られない／祟られる」という場所条件規則が中心で、期限・潜伏期間・段階進行を必須としない。

### D17 回避・制御方式
- Primary: `D17.AVO.DO_NOT_ENGAGE` — 接触回避
- Parent: `D17.AVO`
- Secondary: `D17.RUL.OBEY_TABOO` — 禁忌遵守
- Status: `D`
- 根拠: 最も安定した回避は藪へ入らないこと。解除儀礼や脱出手順は共有核にない。

### D18 作用レイヤー
- Value: `D18.L1=1; D18.L2=0; D18.L3=0`
- Status: `I`
- 根拠: 伝承内では侵入者へ迷失・祟りが作用する。伝承を知ること自体は危害条件ではなく、伝承流通が現実の土地管理・制度を変えた独立X Evidenceも今回固定していない。

### D19 流通範囲
- Primary: `D19.MAS.CROSS_GENERATIONAL` — 世代横断的大衆
- Parent: `D19.MAS`
- Secondary: `D19.LOC.LOCAL_TRADITION` — 地域伝承圏
- Status: `I`
- 根拠: 江戸後期印刷物、1857年石碑、1881年錦絵、近現代文学・自治体資料まで長期間継承される一方、場所固有の地域伝承であり続ける。

### D20 特権情報保持者
- Primary: なし
- Parent: なし
- Secondary: なし
- Status: `U`
- 根拠: 由来の「真相」は複数説が競合し、地元住民・神職・歴史研究者のいずれか一者へ特権的知識を固定できない。

### D21 現実アンカー
- Value: `D21.A4` — 史実・記録・既存伝承を因果統合
- Status: `I`
- 根拠: 実在地点に加え、日本武尊、平将門、平貞盛、徳川光圀、入会地等の歴史人物・社会史・既存伝承を由来説明として因果構造へ組み込む。ただし各由来説の史実性は確定しない。

## 3. QA前メモ

- D01=G0は「起源を江戸後期に固定」ではなく、「前近代に存在が直接確認できる」という粗年代判定。
- D07/D13/D15は「迷失型」と「祟り型」の競合をCで保持する。
- D10は由来説ごとの神格・歴史人物を共有因果源へ混成しない。
- D18 L3=0。現地の鳥居・祠・石碑が存在すること自体は、伝承が社会現実を変えたX Evidenceではない。
- D21=A4は由来説が歴史人物・土地制度を因果説明へ組み込む点を表す。史実認定ではない。

**R3 Independent Recode: FROZEN.**