記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0112_00_contents.md`、R1/R2/R3監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0112`
- `伝承エントリ名称`: 事故物件は一度別人を住ませれば告知義務が消える
- `Macro_Category`: 制度・陰謀
- `Entry_Type`: 命題型噂
- `Version_Scope`: 少なくとも2008年までに公開Web上で確認でき、その後も反証・解説を伴って再流通する制度抜け道俗説。「事故物件でも、一度だけ別の人物を入居させて退去させれば、その次の契約では人の死に関する告知義務が自動的に消える」という人数ベースの条件規則を対象とする。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
- 判定根拠: 2008年11月の公開Web確認点はあるが、成立年代は固定できない。
- この伝承における現れ方: 2008年時点で既知の裏ルールとして語られるが、成立時期は不明。

### D02 最古確認流通媒体
- Primary Child / Value: `D02.WEB.FORUM` — Web掲示板・フォーラム
- Primary Parent: `D02.WEB`
- Secondary: なし
- Status: `D`
- 判定根拠: 現時点で最古に直接確認できる実流通は2008年Yahoo!知恵袋の公開Q&A。
- この伝承における現れ方: 制度質問への回答として俗説が公衆へ伝達される。

### D03 確認流通媒体ポートフォリオ
- Primary Child / Value: `D03.WEB.FORUM` — Web掲示板・フォーラム
- Primary Parent: `D03.WEB`
- Secondary: `D03.WEB.WEBSITE` — Webサイト
- Status: `D`
- 判定根拠: Q&A・掲示板で直接流通し、後代にはWeb記事の制度解説・反証の中でも元俗説が再提示される。
- この伝承における現れ方: 短い裏ルールとして掲示板を流れ、反証記事でも再可視化される。

### D04 生成・変容パターン
- Primary Child / Value: `D04.CON.DEBUNK_RECIRCULATION` — 反証を伴う再流通
- Primary Parent: `D04.CON`
- Secondary: `D04.VAR.ACCRETION` — 増補
- Status: `D/I`
- 判定根拠: 後代には「今は違う」という訂正とともに元命題が再流通し、外国人・短期入居者等の具体的手口が付加される。
- この伝承における現れ方: 反証によって消えるのではなく、誤解の代表例として繰り返し引用される。

### D05 提示形式
- Primary Child / Value: `D05.PRP.PREDICTIVE_CLAIM` — 予測命題
- Primary Parent: `D05.PRP`
- Secondary: `D05.HYB.QA_EXPERT` — Q&A・解説混合
- Status: `D`
- 判定根拠: 「一人挟めば、次は告知不要になる」という条件付き制度予測が核であり、Q&A形式でも流通する。
- この伝承における現れ方: 複雑な実務が単純なif-then規則へ圧縮される。

### D06 真実性提示
- Primary Child / Value: `D06.T4` — 一般・制度・科学事実
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 法・不動産実務上の一般ルールであるかのように提示される。これは提示様式のコードであり、制度的正しさを意味しない。
- この伝承における現れ方: 「業界ではそういう決まり」として受容される。

## 2.2. 意味形成

### D07 意味形成対象
- Primary Child / Value: `D07.INO.HIDDEN_RULE_PROCEDURE` — 隠れた制度・手続
- Primary Parent: `D07.INO`
- Secondary: `D07.INO.OCCUPATIONAL_HIDDEN_PRACTICE` — 職業上の隠れた慣行
- Status: `D`
- 判定根拠: 一般契約者から見えにくい告知ルールと不動産実務上の抜け道が中心問題。
- この伝承における現れ方: 不透明な個別判断を「一人挟む」という裏手続で説明する。

### D08 意味形成契機
- Primary Child / Value: `D08.OPA.OPAQUE_RULE` — 不透明な規則・判断
- Primary Parent: `D08.OPA`
- Secondary: `D08.CLM.UNSUPPORTED_ASSERTION` — 根拠薄弱な主張
- Status: `D/I`
- 判定根拠: 告知要否が一般人から直接観察しにくいことが裏ルールを求める契機となり、根拠未提示の断定が説明候補として流通する。
- この伝承における現れ方: 制度の見えにくさが単純な俗説を受け入れやすくする。

### D09 意味付与操作
- Primary Child / Value: `D09.SSI.SYSTEM_LOGIC` — 制度論理化
- Primary Parent: `D09.SSI`
- Secondary: `D09.SSI.HIDDEN_STRUCTURE` — 隠れた構造化
- Status: `I`
- 判定根拠: 中間入居を制度状態を切り替える内部ロジックとして置き、見えない抜け道構造を仮定する。
- この伝承における現れ方: 複雑な実務が人数一変数で予測できるモデルへ変換される。

### D10 因果源存在論
- Primary Child / Value: `D10.HUM.ORGANIZATION` — 組織・制度主体
- Primary Parent: `D10.HUM`
- Secondary: なし
- Status: `I`
- 判定根拠: 伝承内では不動産業者・貸主等が制度を運用し、告知状態を変えられる主体として想定される。
- この伝承における現れ方: 超自然ではなく組織的な制度運用が原因とされる。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary Child / Value: `D11.CON.LOCATION_STATE` — 位置・状態条件
- Primary Parent: `D11.CON`
- Secondary: `D11.MAN.CREATE_ALTER_MANIPULATE` — 作る・加工・操作する
- Status: `D`
- 判定根拠: 共有核では「事故後に別人が一度居住済み」という物件履歴状態が条件。派生ではこの状態を意図的に作る操作として語られる。
- この伝承における現れ方: 人数履歴が一段進むと告知状態が切り替わる、とされる。

### D12 作用対象
- Primary Child / Value: `D12.ORG.INSTITUTION_SYSTEM` — 制度・社会システム
- Primary Parent: `D12.ORG`
- Secondary: `D12.OTH.SPECIFIC_OTHER` — 特定他者
- Status: `I`
- 判定根拠: D13 Primaryの制度操作が直接対象とするのは告知義務という制度状態。Secondaryの情報隠蔽によって次の契約者も影響を受ける。
- この伝承における現れ方: 制度状態と後続契約者への情報提供が同時に切り替わると説明される。

### D13 作用機構
- Primary Child / Value: `D13.SOC.INSTITUTIONAL_MANIPULATION` — 制度操作
- Primary Parent: `D13.SOC`
- Secondary: `D13.SOC.CONCEAL_SUPPRESS` — 隠蔽・抑圧
- Status: `D`
- 判定根拠: 中間入居により告知義務を解除でき、事故歴を後続契約者へ告げずに済むというモデルが核。
- この伝承における現れ方: 入居履歴を操作して情報開示ルールを変えるとされる。

### D14 帰結極性
- Primary Child / Value: `D14.MIX` — 混合
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 貸主・業者側には取引上の利得、後続契約者側には重要情報を得られない不利益が併存する。
- この伝承における現れ方: 同じ抜け道が主体によって利益にも損失にもなる。

### D15 帰結領域
- Primary Child / Value: `D15.BEH.COMPLIANCE_DECISION` — 遵守・意思決定変化
- Primary Parent: `D15.BEH`
- Secondary: なし
- Status: `I`
- 判定根拠: 共有核で直接変わるとされるのは「次回契約で告知する／しない」という実務判断。金銭効果は必須ではない。
- この伝承における現れ方: 中間入居後は「もう告知しなくてよい」という判断へ移るとされる。

### D16 因果時間構造
- Primary Child / Value: `D16.STA.STATIC_RULE` — 静的規則
- Primary Parent: `D16.STA`
- Secondary: なし
- Status: `D`
- 判定根拠: 日数経過ではなく「一人入居済みなら次は不要」という条件対応そのものが主題。
- この伝承における現れ方: 時間ではなく入居履歴の人数で結果を決める規則として記憶される。

### D17 回避・制御方式
- Primary Child / Value: `D17.INF.VERIFY_DEBUNK` — 検証・反証
- Primary Parent: `D17.INF`
- Secondary: `D17.USE.STRATEGIC_RULE_USE` — 規則の戦略利用
- Status: `I`
- 判定根拠: 受容者側は公式ガイドライン等で俗説を検証できる。一方、伝承内部では業者側が「一人挟む」規則を戦略的に利用する派生がある。
- この伝承における現れ方: 同じルールが反証対象にも抜け道としての利用対象にもなる。

### D18 作用レイヤー
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `D`
- 判定根拠: L1では「一人挟む→告知状態が変わる」という伝承内因果がある。聞くこと自体は作用条件ではない。俗説により市場・制度・集団行動が実際に変化したことを示す十分な独立X Evidenceを固定できないためL3は0。
- この伝承における現れ方: 現実実務との接続可能性はあるが、伝承流通の社会効果としては未立証。

## 2.4. 社会的分布・現実接続

### D19 流通範囲
- Primary Child / Value: `D19.NET.OPEN_FORUM_WEB` — 公開Web・掲示板
- Primary Parent: `D19.NET`
- Secondary: なし
- Status: `D`
- 判定根拠: 公開Q&A・掲示板等での流通を直接確認できる。全国的大衆への浸透度は閲覧可能性だけから推測しない。
- この伝承における現れ方: 特定地域内部ではなく公開Web圏で流通する。

### D20 特権情報保持者
- Primary Child / Value: `D20.ORG.INSTITUTION_AUTHORITY` — 組織・権限主体
- Primary Parent: `D20.ORG`
- Secondary: `D20.EXP.TECHNICAL_OCCUPATIONAL` — 技術・職業専門家
- Status: `I`
- 判定根拠: 物件履歴・告知実務は貸主、管理会社、行政・業界制度主体が保持し、制度解釈は不動産実務者等が追加知識を持つ。
- この伝承における現れ方: 契約者と業界側の情報非対称が説得力を支える。

### D21 現実アンカー
- Primary Child / Value: `D21.A3` — 実在制度・社会史が成立条件
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 宅地建物取引、人の死の告知実務、国交省ガイドライン等の実在制度が伝承成立に不可欠。
- この伝承における現れ方: 実在制度の複雑さが誤った簡略ルールの土台になる。

# 3. R4 QA

## 3.1. causal chain

```text
D11: 「事故後に別人が一度居住済み」という物件状態になる
→ D12: 告知義務という制度状態が対象になる
→ D13: 告知状態を制度的に切り替え、事故歴を隠せるとされる
→ D15: 次回契約で「告知しない」という遵守判断へ至る
```

D12/D13/D15は接続する。これは伝承内因果であり、現行制度がこの因果を支持するという意味ではない。

## 3.2. 現行制度との照合

国土交通省の現行ガイドラインは、死因、発生場所、特殊清掃、賃貸借での概ね3年、質問の有無、特段の事情等で告知判断を整理しており、「一人挟めば自動的に告知不要」という人数ベースの解除規則を示していない。俗説の存在と制度事実を分離する。

## 3.3. L3

旧10は業界回顧記述からL3=1としていたが、control planeの厳格なL3要件に照らすと、伝承流通が現実の制度・市場行動を変えたことを示す独立X Evidenceとして不足する。`L3=0`へ保守化する。

## 3.4. U / NA / C

- D01=`U`: 成立年代未確定。
- NA: なし。
- C: なし。

## 3.5. taxonomy gap

現行Scopeは既存taxonomyで表現可能。新Child要求なし。

## 3.6. 旧10との差分

R3 freeze後に旧10を確認した。R4では旧分析のうち、D08は不透明な制度判断をPrimary、D11は「一人入居済み」という状態条件をPrimary、D12に後続契約者、D13に情報隠蔽、D17に戦略利用をSecondaryとして採用した。一方、D18 L3=1は独立X Evidence不足のため採用せず0へ保守化し、D19も「全国から閲覧可能」だけで全国的大衆へ一般化せず公開Web圏に限定した。D05は条件規則をより直接表す `PREDICTIVE_CLAIM` を維持する。