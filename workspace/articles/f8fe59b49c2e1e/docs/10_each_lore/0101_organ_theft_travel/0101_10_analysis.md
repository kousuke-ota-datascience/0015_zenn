記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0101_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0101`
- `伝承エントリ名称`: 海外旅行で臓器を抜かれる
- `Macro_Category`: 犯罪・社会不安
- `Entry_Type`: FOAF
- `Version_Scope`: 1991年までに新聞上で確認できる成人旅行者kidney-heist型。旅行中の成人が見知らぬ人物と接触し、薬物等で意識を失い、その空白時間に腎臓等を摘出され、後から臓器喪失を知る型を対象とする。1990年代後半の初期ネット再増幅を流通史に含めるが、1980年代の子ども臓器盗難rumor一般、浴槽・メモ等の全異伝を必須条件にはしない。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示の次元

### 2.1.1. D01: 生成年代

**いつ成立したか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

1991年新聞は成人旅行者型の存在確認点であり、生成時期そのものを示さない。独立した成立History Evidenceは固定できていない。

**この伝承における現れ方**

1991年までには成人旅行者kidney-heist型が流通しているが、それ以前の成立時期は未確定である。

### 2.1.2. D02: 最古確認流通媒体

**現在の証拠で、伝承が実際に人から人へ流通したことを最も古く確認できる媒体は何か**

- Primary Child / Value: `D02.PRT.NEWSPAPER` — 新聞
- Primary Parent: `D02.PRT`
- Secondary: なし
- Status: `D`

**判定根拠**

1991-11-15のDeseret Newsで成人旅行者・薬物・腎臓摘出・闇市場という型を直接確認できる。

**この伝承における現れ方**

読者から寄せられた噂が新聞紙面上でurban legendとして可視化される。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）

**今回コードするVersion Scopeで、伝承が受容者へ流通したことを確認できる媒体は何か**

- Primary Child / Value: `D03.PRT.NEWSPAPER` — 新聞
- Primary Parent: `D03.PRT`
- Secondary: `D03.EDG.BBS` — BBS・電子掲示板
- Status: `I`

**判定根拠**

新聞はDirect Evidence、1990年代後半のnewsgroup再増幅はDonovan 2002による研究Evidenceで確認する。Eメール個別文面は今回00で直接固定していないためSecondaryには置かない。

**この伝承における現れ方**

印刷媒体で確認された犯罪伝説が初期インターネット掲示系媒体へ再流通する。

### 2.1.4. D04: 生成・変容パターン

**時間とともにどう変形したか**

- Primary Child / Value: `D04.MIG.DIGITAL_REAMPLIFICATION` — ネット再増幅
- Primary Parent: `D04.MIG`
- Secondary: なし
- Status: `I`

**判定根拠**

Donovan 2002は1996〜1998年のkidney/organ theft関連newsgroup投稿の増加を数量的に示す。地名・浴槽・メモ等の変化方向は固定できないため、地域化・再話をSecondaryに置かない。

**この伝承における現れ方**

既存の臓器盗難犯罪伝説が初期インターネット上で大量に再提示される。

### 2.1.5. D05: 提示形式

**どんなコミュニケーション形式で提示されるか**

- Primary Child / Value: `D05.HRS.FOAF` — FOAF
- Primary Parent: `D05.HRS`
- Secondary: なし
- Status: `I`

**判定根拠**

1991年記事は読者から寄せられた身近な被害談として紹介される。後代の全媒体を一律に警告形式とはしない。

**この伝承における現れ方**

具体的な成人旅行者の被害談として、近接伝聞的な事件話の形を取る。

### 2.1.6. D06: 真実性提示

**どんな「本当らしさ」を要求するか**

- Primary Child / Value: `D06.T2` — 近接伝聞事実
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

1991年の読者投稿では具体的被害事例として提示されるが、研究者側はurban legendとして位置づける。受容者向け語りの真実性姿勢は近接伝聞事実に最も近いが、Direct本文の全異伝を固定していないためIとする。

**この伝承における現れ方**

「旅行者に実際に起きた事件らしい」という形で提示される。

## 2.2. 意味形成の次元

### 2.2.1. D07: 意味形成対象

**何が不可解・不確実なのか**

- Primary Child / Value: `D07.ICT.HIDDEN_CRIMINAL_PRACTICE` — 隠れた犯罪慣行
- Primary Parent: `D07.ICT`
- Secondary: `D07.ICT.ABDUCTION_ASSAULT` — 誘拐・暴行
- Status: `I`

**判定根拠**

意識空白の間に臓器摘出・売買が秘密裏に行われたと説明される。

**この伝承における現れ方**

通常の旅行空間の背後に、表から見えない犯罪実践が存在すると理解される。

### 2.2.2. D08: 意味形成契機

**何を手掛かりに問題化されるか**

- Primary Child / Value: `D08.TRC.PHYSICAL_TRACE` — 物的痕跡
- Primary Parent: `D08.TRC`
- Secondary: `D08.STY.FOAF_REPORT` — 近接伝聞
- Status: `I`

**判定根拠**

物語内では身体痕跡・臓器喪失が問題化の直接手掛かりとなり、受容者側には被害報告が伝達契機となる。

**この伝承における現れ方**

被害者は覚醒後の身体異常から、受容者は被害談から隠れた犯罪を認識する。

### 2.2.3. D09: 意味付与操作

**不可解なものをどう理解可能にするか**

- Primary Child / Value: `D09.AGN.AGENCY_ATTRIBUTION` — 主体化
- Primary Parent: `D09.AGN`
- Secondary: `D09.CAU.HIDDEN_CAUSE` — 隠れた原因化
- Status: `I`

**判定根拠**

意識空白と臓器喪失を、秘密の人間犯罪者による摘出行為へ帰属する。明示Evidenceのない禁忌化はSecondaryから外す。

**この伝承における現れ方**

原因不明の身体損失が、見えない犯罪者の行為として説明される。

### 2.2.4. D10: 因果源存在論

**原因を何として世界に置くか**

- Primary Child / Value: `D10.HUM.INFORMAL_GROUP` — 非制度的集団・共同体
- Primary Parent: `D10.HUM`
- Secondary: なし
- Status: `I`

**判定根拠**

接触・薬物投与・外科処置・売買を担う複数の人間犯罪者が想定されるが、正式組織構造は固定できない。

**この伝承における現れ方**

超自然ではなく秘密の人間犯罪ネットワークが原因として置かれる。

## 2.3. 因果・行動モデルの次元

### 2.3.1. D11: 発動・接触条件

**何を契機に因果系へ入るか**

- Primary Child / Value: `D11.SOC.MEET_INTERACT` — 会う・会話する
- Primary Parent: `D11.SOC`
- Secondary: なし
- Status: `D`

**判定根拠**

1991年資料で直接確認できる入口は、旅行者が酒場で見知らぬ女性と出会うこと。加害者側の不可視な「選択」を補わない。

**この伝承における現れ方**

通常の社交的接触が被害エピソードの入口となる。

### 2.3.2. D12: 作用対象

**誰／何に作用するか**

- Primary Child / Value: `D12.OTH.VICTIM_TARGET` — 被害者・標的
- Primary Parent: `D12.OTH`
- Secondary: なし
- Status: `D`

**判定根拠**

薬物投与・臓器摘出を直接受ける成人旅行者が作用対象である。

**この伝承における現れ方**

一般旅行者が犯罪被害の対象になる。

### 2.3.3. D13: 作用機構

**因果源が対象へ何をするか**

- Primary Child / Value: `D13.PHY.BODY_TRANSFORMATION` — 身体変容
- Primary Parent: `D13.PHY`
- Secondary: `D13.SOC.COERCE_CONFINE` — 強制・監禁
- Status: `I`

**判定根拠**

中心作用は身体内部から腎臓等を摘出する不可逆的改変。強制・拘束はその過程として推定されるためSecondaryはI範囲で保持する。

**この伝承における現れ方**

身体内部そのものが盗難対象となり、身体状態が不可逆に変化する。

### 2.3.4. D14: 帰結極性

**結果は正・負・中立・混合か**

- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `D`

**判定根拠**

臓器喪失・重大身体被害が中心帰結である。

**この伝承における現れ方**

被害者に利益をもたらす型ではない。

### 2.3.5. D15: 帰結領域

**何の領域が最終的に変わるか**

- Primary Child / Value: `D15.BOD.SEVERE_INJURY` — 重傷・障害
- Primary Parent: `D15.BOD`
- Secondary: `D15.LIF.FUTURE_CONSTRAINT` — 将来制約
- Status: `I`

**判定根拠**

臓器喪失は重大な身体障害であり、その後の健康制約も含意される。

**この伝承における現れ方**

一時的盗難ではなく、長期に影響し得る身体損失として終わる。

### 2.3.6. D16: 因果時間構造

**発動条件・原因成立から主作用／主帰結までの関係は、時間上どのように編成されているか**

- Primary Child / Value: `D16.EVT.SEQUENTIAL_EPISODE` — エピソード内段階進行
- Primary Parent: `D16.EVT`
- Secondary: `D16.DLY.DELAYED` — 遅延
- Status: `I`

**判定根拠**

接触→無力化→摘出→覚醒→被害認識という段階進行があり、被害者の認識には時間差がある。

**この伝承における現れ方**

犯行の核心が意識空白へ置かれ、結果だけが後から認識される。

### 2.3.7. D17: 回避・制御方式

**結果をどう回避・制御・利用できるか**

- Primary Child / Value: `D17.RIT.MEDICAL_PROFESSIONAL` — 医療介入
- Primary Parent: `D17.RIT`
- Secondary: なし
- Status: `I`

**判定根拠**

後代異伝では覚醒後に医療確認・救命へ移る型がある。一方、「見知らぬ人物・飲食物を避ける」という事前回避は明示Evidenceから固定できないため採用しない。

**この伝承における現れ方**

被害後に専門医療へ移り、身体状態を確認する。

### 2.3.8. D18: 作用レイヤー

**因果効力は伝承内／受容者／社会現実のどこに及ぶか**

- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `I`

**判定根拠**

物語内部の犯罪者→旅行者作用は確認できる。受容行為自体が被害条件となるEvidence、流通が社会制度・市場等を変えた独立X Evidenceは固定していない。不在値を含むvector全体はIとする。

**この伝承における現れ方**

確認できる因果効力は物語内部に留まる。

## 2.4. 社会的埋め込みの次元

### 2.4.1. D19: 流通範囲

**誰の間に伝承が流通するか**

- Primary Child / Value: `D19.MAS.TRANSNATIONAL` — 国際的・越境的大衆
- Primary Parent: `D19.MAS`
- Secondary: `D19.NET.OPEN_FORUM_WEB` — 公開Web・掲示板
- Status: `I`

**判定根拠**

Antonijević 2007は臓器盗難legend/rumorを国際的現象として論じ、Donovan 2002は初期ネット上の大量流通を示す。ただし成人旅行者型だけの全地理分布を直接測定しているわけではないためI。

**この伝承における現れ方**

国境を越える犯罪伝説として流通し、初期インターネット上でも再増幅する。

### 2.4.2. D20: 特権情報保持者

**誰が真相・追加情報を持つか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

医療者・メモ・周囲の人物など追加情報源は異伝差があり、Scope全体で一意に固定できない。

**この伝承における現れ方**

被害者がどの経路で真相を知るかは再話により異なる。

### 2.4.3. D21: 現実アンカー

**実在世界へどの程度固定されるか**

- Primary Child / Value: `D21.A1` — 一般的現実背景
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

旅行、酒場、外科処置、臓器市場等の現実的要素を用いるが、特定制度・事件・組織が成立条件として固定されていない。現実の臓器取引一般をScopeの制度条件へ引き上げない。

**この伝承における現れ方**

一般的な旅行・医療・犯罪の現実背景に埋め込まれる。