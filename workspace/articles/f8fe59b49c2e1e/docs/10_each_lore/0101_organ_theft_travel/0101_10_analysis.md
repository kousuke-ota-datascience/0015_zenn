記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0101_00_contents.md`、R1/R2/R3監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0101`
- `伝承エントリ名称`: 海外旅行で臓器を抜かれる
- `Macro_Category`: 犯罪・社会不安
- `Entry_Type`: FOAF
- `Version_Scope`: 1991年までに成人旅行者型として確認でき、1990年代に印刷・初期デジタル媒体で再増幅した“kidney heist”型。海外・出張先等で旅行者が見知らぬ人物と接触し、薬物等で意識を失い、その空白時間に腎臓等を秘密裏に摘出され、覚醒後の傷・メモ・医療確認等から臓器喪失を知る共有核を対象とする。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Primary Child / Value: `D01.G5` — 1990年代
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 成人旅行者kidney-heist型を1991年11月の同時代新聞で直接確認できる。
- この伝承における現れ方: 1990年代初頭には旅行者・薬物・腎臓摘出・闇市場という核が成立していた。

### D02 最古確認流通媒体
- Primary Child / Value: `D02.PRT.NEWSPAPER` — 新聞
- Primary Parent: `D02.PRT`
- Secondary: なし
- Status: `D`
- 判定根拠: 現在固定できる最古の成人旅行者型の実流通媒体は1991年11月の新聞。
- この伝承における現れ方: 読者から寄せられた噂が民俗学者の回答付きで新聞上に可視化される。

### D03 確認流通媒体ポートフォリオ
- Primary Child / Value: `D03.PRT.NEWSPAPER` — 新聞
- Primary Parent: `D03.PRT`
- Secondary: `D03.EDG.BBS` — BBS・電子掲示板; `D03.EDG.EMAIL` — Eメール
- Status: `D/I`
- 判定根拠: 1991年新聞を直接確認し、1990年代後半のnewsgroup・e-mail再流通を研究・同時代報道で確認する。
- この伝承における現れ方: 印刷媒体で確認された型が初期デジタル通信へ移り、大量転送・再話される。

### D04 生成・変容パターン
- Primary Child / Value: `D04.MIG.DIGITAL_REAMPLIFICATION` — ネット再増幅
- Primary Parent: `D04.MIG`
- Secondary: `D04.VAR.LOCALIZATION` — 地域化; `D04.VAR.RETELLING` — 再話
- Status: `D/I`
- 判定根拠: 既存のkidney-heist型が1990年代後半に初期インターネットで急増し、都市・被害者属性を差し替えて再話される。
- この伝承における現れ方: New Orleans等の具体地名や被害者属性を変えながら同じ警告筋書きが反復される。

### D05 提示形式
- Primary Child / Value: `D05.HRS.FOAF` — FOAF
- Primary Parent: `D05.HRS`
- Secondary: `D05.RUL.WARNING` — 警告
- Status: `D`
- 判定根拠: 友人・同僚等の実話らしい近接伝聞として提示され、旅行者への警告文として再流通する。
- この伝承における現れ方: 被害者を身近な第三者へ置くことで検証困難性と現実感を両立する。

### D06 真実性提示
- Primary Child / Value: `D06.T2` — 近接伝聞事実
- Primary Parent: なし
- Secondary: なし
- Status: `D/I`
- 判定根拠: 1991年読者投稿や1997年頃のchain型は、身近な被害事例・公式警告らしい体裁で事実として提示される。
- この伝承における現れ方: 「友人の知人／出張者に本当に起きた」事件として信憑性を要求する。

## 2.2. 意味形成

### D07 意味形成対象
- Primary Child / Value: `D07.ICT.HIDDEN_CRIMINAL_PRACTICE` — 隠れた犯罪慣行
- Primary Parent: `D07.ICT`
- Secondary: `D07.ICT.ABDUCTION_ASSAULT` — 誘拐・暴行
- Status: `D`
- 判定根拠: 表から見えない臓器摘出・売買ネットワークと、その過程での身体暴力が中心不安。
- この伝承における現れ方: 通常の旅行空間の背後に人体を商品化する犯罪市場があると説明する。

### D08 意味形成契機
- Primary Child / Value: `D08.STY.FOAF_REPORT` — 近接伝聞
- Primary Parent: `D08.STY`
- Secondary: `D08.TRC.PHYSICAL_TRACE` — 物的痕跡
- Status: `D/I`
- 判定根拠: 受容者にはFOAF報告、物語内被害者には手術痕・チューブ等の身体痕跡が問題化契機になる。
- この伝承における現れ方: 聞き手は証言から、被害者は身体痕跡から「見えない犯罪」を知る。

### D09 意味付与操作
- Primary Child / Value: `D09.AGN.AGENCY_ATTRIBUTION` — 主体化
- Primary Parent: `D09.AGN`
- Secondary: `D09.CAU.HIDDEN_CAUSE` — 隠れた原因化; `D09.NOR.TABOOIZATION` — 禁忌化
- Status: `I`
- 判定根拠: 意識空白と臓器喪失を秘密の犯罪者主体へ帰属し、旅行先での不審な社交・飲食を回避すべき行為へ変換する。
- この伝承における現れ方: 原因不明の身体損失が「臓器泥棒」の意図と秘密活動で説明される。

### D10 因果源存在論
- Primary Child / Value: `D10.HUM.INFORMAL_GROUP` — 非制度的集団・共同体
- Primary Parent: `D10.HUM`
- Secondary: なし
- Status: `I`
- 判定根拠: 接触、薬物投与、外科処置、売買を担う複数の人間犯罪者が想定されるが、正式な組織構造はEvidence上固定できない。
- この伝承における現れ方: 超自然ではなく、秘密の人間ネットワークが原因となる。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary Child / Value: `D11.PAS.SPONTANEOUS_SELECTION` — 偶然選ばれる・遭遇する
- Primary Parent: `D11.PAS`
- Secondary: `D11.SOC.MEET_INTERACT` — 会う・会話する
- Status: `I`
- 判定根拠: 旅行者が加害者に標的化されることが因果系への入口で、会話・飲食は接触手段。旅行や飲酒自体を再現的発動条件とはしない。
- この伝承における現れ方: 通常の社交中に本人の意図と無関係に被害対象へ選ばれる。

### D12 作用対象
- Primary Child / Value: `D12.OTH.VICTIM_TARGET` — 被害者・標的
- Primary Parent: `D12.OTH`
- Secondary: なし
- Status: `D`
- 判定根拠: 直接の薬物投与・摘出を受ける旅行者本人が作用対象。
- この伝承における現れ方: 一般旅行者が人体資源として標的化される。

### D13 作用機構
- Primary Child / Value: `D13.PHY.BODY_TRANSFORMATION` — 身体変容
- Primary Parent: `D13.PHY`
- Secondary: `D13.PHY.PHYSIOLOGICAL_CHANGE` — 生理変化; `D13.SOC.COERCE_CONFINE` — 強制・監禁
- Status: `D/I`
- 判定根拠: 身体内部から臓器を除去する不可逆的改変が主作用。意識喪失・拘束過程と臓器喪失による生理機能変化をSecondaryに保持する。
- この伝承における現れ方: 財布ではなく身体内部そのものが盗難対象となる。

### D14 帰結極性
- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 臓器喪失・重大身体障害・犯罪被害が中心帰結。
- この伝承における現れ方: 利益・救済を中心とする型ではない。

### D15 帰結領域
- Primary Child / Value: `D15.BOD.SEVERE_INJURY` — 重傷・障害
- Primary Parent: `D15.BOD`
- Secondary: `D15.LIF.FUTURE_CONSTRAINT` — 将来制約
- Status: `D`
- 判定根拠: 臓器喪失による重大な身体障害と、その後の長期的健康制約が終端を形成する。
- この伝承における現れ方: 被害は一時的な盗難ではなく、生涯に影響し得る身体損失として描かれる。

### D16 因果時間構造
- Primary Child / Value: `D16.EVT.SEQUENTIAL_EPISODE` — エピソード内段階進行
- Primary Parent: `D16.EVT`
- Secondary: `D16.DLY.DELAYED` — 遅延
- Status: `D`
- 判定根拠: 接触→意識喪失→摘出→覚醒→被害確認が一つのエピソード内で順に進み、本人が被害を知る時点には時間差がある。
- この伝承における現れ方: 犯行の核心が認識不能な空白時間へ置かれる。

### D17 回避・制御方式
- Primary Child / Value: `D17.AVO.DO_NOT_ENGAGE` — 接触回避
- Primary Parent: `D17.AVO`
- Secondary: `D17.RIT.MEDICAL_PROFESSIONAL` — 医療介入
- Status: `I`
- 判定根拠: 警告としては不審な人物・飲食物との接触を避ける。被害後は医療機関で確認・救命処置を受ける型がある。
- この伝承における現れ方: 事前には対人回避、事後には専門医療へ移行する。

### D18 作用レイヤー
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `D`
- 判定根拠: 物語内部では犯罪者が旅行者へ作用する。話を聞くこと自体が危害条件ではない。独立した十分なX Evidenceを固定していないためL3を付けない。
- この伝承における現れ方: 伝承の因果作用は物語内部の犯罪被害に閉じる。

## 2.4. 社会的分布・現実接続

### D19 流通範囲
- Primary Child / Value: `D19.MAS.TRANSNATIONAL` — 国際的・越境的大衆
- Primary Parent: `D19.MAS`
- Secondary: `D19.MAS.NATIONAL_PUBLIC` — 全国的大衆
- Status: `D/I`
- 判定根拠: 成人kidney-heist型は複数国・都市へローカライズされ、1990年代の初期ネットで越境的に流通した。日本語圏でも後代事典へ収録される。
- この伝承における現れ方: 特定都市のローカル事件ではなく、旅行先を差し替えて国境を越える犯罪伝説として流通する。

### D20 特権情報保持者
- Primary Child / Value: `D20.ORG.PERPETRATOR_CRIMINAL` — 加害者・犯罪者
- Primary Parent: `D20.ORG`
- Secondary: `D20.EXP.MEDICAL_SCIENTIFIC` — 医療・科学専門家
- Status: `I`
- 判定根拠: 意識喪失中の犯罪過程の全貌は加害側だけが知り、被害者の臓器喪失は医療者が後から確認する型がある。
- この伝承における現れ方: 被害者本人は最重要局面を観察できず、痕跡と専門確認を通じて事後的に真相へ接近する。

### D21 現実アンカー
- Primary Child / Value: `D21.A3` — 実在制度・社会史が成立条件
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 臓器移植・臓器不足・臓器売買という実在の医療制度／社会問題が、この伝承のもっともらしさと因果モデル成立に必要。ただし特定実事件へは依存しない。
- この伝承における現れ方: 実在する移植医療と犯罪市場の語彙が、定型FOAFへ現実性を与える。

# 3. R4 QA

## 3.1. causal chain

```text
D11: 旅行先で犯罪者に選ばれ、接触する
→ D12: 旅行者本人が被害標的となる
→ D13: 意識喪失中に臓器を摘出され身体が改変される
→ D15: 重大身体障害と長期健康制約へ至る
```

D12/D13/D15は接続する。

## 3.2. L3

同時代報道には個人の旅行不安反応や移植関係者の懸念があるが、Entry単位の安定した社会効果を示す十分な独立X Evidenceとしては固定しない。`L3=0`。

## 3.3. U / NA / C

U/NA/Cなし。D03/D04/D06/D13/D19等の資料横断再構成はstatusにIを含める。

## 3.4. taxonomy gap

現行Scopeは既存taxonomyで表現可能。新Child要求なし。

## 3.5. 旧10との差分

R3 freeze後に旧10を確認した。主要差分は、R1で1991年同時代資料を追加したことによりD01 `U→G5`、D02 `U→NEWSPAPER`、D03 `BOOK→NEWSPAPER + BBS/EMAIL`、D04 `LOCALIZATION→DIGITAL_REAMPLIFICATION`へ変更した点である。意味形成ではD09を旧 `HIDDEN_STRUCTURE` から `AGENCY_ATTRIBUTION` Primaryへ、D11を通常行為ではなく被害者選択中心へ再裁定した。D19は越境流通Evidenceを反映しTRANSNATIONALへ、D21は臓器移植制度・臓器市場がモデル成立に構造的に必要とみてA3へ引き上げた。旧値への一致は目的としていない。