記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0089_00_contents.md`、R1/R2/R3監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0089`
- `伝承エントリ名称`: 日本だるま／だるま女
- `Macro_Category`: 犯罪・社会不安
- `Entry_Type`: FOAF
- `Version_Scope`: 1994年までに「日本だるま」として採録された、日本人女性の海外旅行をめぐるFOAF型。旅行中に同行者から短時間離れた女性が失踪し、後に別地点で四肢を切断された状態で再発見され、誘拐・監禁・搾取を受けていたと説明される共有核を対象とする。特定国・都市・搾取目的は必須化しない。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
- 判定根拠: 1994年採録は固定できるが、最初期口承・成立年代は直接確認できない。後代資料の1980年代整理を成立年へ機械的に写像しない。
- この伝承における現れ方: 近現代の海外旅行文化を前提とするが、共有核の成立年代は未確定。

### D02 最古確認流通媒体
- Primary Child / Value: `D02.PRT.BOOK` — 書籍
- Primary Parent: `D02.PRT` — 印刷・書簡
- Secondary: なし
- Status: `D`
- 判定根拠: 現在直接確認できる最古媒体は1994年刊の現代伝説集への「日本だるま」収録。
- この伝承における現れ方: 少なくとも1994年には採録・読者提示されている。

### D03 確認流通媒体ポートフォリオ
- Primary Child / Value: `D03.PRT.BOOK` — 書籍
- Primary Parent: `D03.PRT`
- Secondary: なし
- Status: `D`
- 判定根拠: Version Scopeで直接確認できる実流通媒体は書籍。口承等は推測で追加しない。
- この伝承における現れ方: FOAF型が現代伝説集で再提示される。

### D04 生成・変容パターン
- Primary Child / Value: `D04.VAR.LOCALIZATION` — 地域化
- Primary Parent: `D04.VAR`
- Secondary: `D04.VAR.RETELLING` — 再話
- Status: `I`
- 判定根拠: 旅行先・失踪場所・再発見場所が差し替えられながら、失踪→後日の四肢切断状態で再発見という核が維持される。
- この伝承における現れ方: 受容者にとって「怖い海外」へ地名が差し替えられる。

### D05 提示形式
- Primary Child / Value: `D05.HRS.FOAF` — FOAF
- Primary Parent: `D05.HRS`
- Secondary: `D05.RUL.WARNING` — 警告
- Status: `I`
- 判定根拠: 「友人の友人／知人の旅行者に起きた実話」として語る近接伝聞が中心で、単独行動回避の警告へ接続する。
- この伝承における現れ方: 検証困難だが身近に感じられる距離で犯罪被害を提示する。

### D06 真実性提示
- Primary Child / Value: `D06.T2` — 近接伝聞事実
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 語り手自身ではなく、知人関係上の具体的被害として真実性を主張する。
- この伝承における現れ方: 「実際に知人の知人へ起きた事件らしい」という受け取り方を要求する。

## 2.2. 意味形成

### D07 意味形成対象
- Primary Child / Value: `D07.ICT.ABDUCTION_ASSAULT` — 誘拐・暴行
- Primary Parent: `D07.ICT`
- Secondary: `D07.ICT.HIDDEN_CRIMINAL_PRACTICE` — 隠れた犯罪慣行; `D07.ICT.OUTGROUP_THREAT` — 外集団脅威
- Status: `I`
- 判定根拠: 中心は失踪・身体暴力であり、見えない犯罪的搾取と異文化圏への不安が補助的に意味形成される。
- この伝承における現れ方: 海外旅行の漠然とした不安が具体的な誘拐・搾取リスクへ変換される。

### D08 意味形成契機
- Primary Child / Value: `D08.STY.FOAF_REPORT` — 近接伝聞
- Primary Parent: `D08.STY`
- Secondary: なし
- Status: `I`
- 判定根拠: 受容者にとっての危険認識は、知人ネットワークを介した被害報告から始まる。
- この伝承における現れ方: 公式統計ではなく、身近な証言らしさが不安を具体化する。

### D09 意味付与操作
- Primary Child / Value: `D09.AGN.AGENCY_ATTRIBUTION` — 主体化
- Primary Parent: `D09.AGN`
- Secondary: `D09.CAU.DIRECT_CAUSE` — 直接原因化; `D09.NOR.TABOOIZATION` — 禁忌化
- Status: `I`
- 判定根拠: 失踪の空白を、意図をもつ犯罪者の行為として理解可能にし、単独行動を危険原因・回避対象へ変換する。
- この伝承における現れ方: 「なぜ消えたか」を秘密の犯罪者集団の意図で閉じる。

### D10 因果源存在論
- Primary Child / Value: `D10.HUM.INFORMAL_GROUP` — 非制度的集団・共同体
- Primary Parent: `D10.HUM`
- Secondary: なし
- Status: `I`
- 判定根拠: 誘拐・監禁・切断・搾取を行う複数の人間犯罪者が想定されるが、正式組織であるEvidenceはない。
- この伝承における現れ方: 超自然ではなく悪意ある人間集団が原因となる。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary Child / Value: `D11.PAS.SPONTANEOUS_SELECTION` — 偶然選ばれる・遭遇する
- Primary Parent: `D11.PAS`
- Secondary: `D11.CON.LOCATION_STATE` — 特定場所・状態にいる
- Status: `I`
- 判定根拠: 海外旅行や試着室利用それ自体が再現的な発動手順ではなく、犯罪者に選ばれる／遭遇することが直接の入口。同行者から離れた状態は脆弱性の文脈。
- この伝承における現れ方: 日常行動の最中に、本人の意図とは無関係に被害対象化される。

### D12 作用対象
- Primary Child / Value: `D12.OTH.VICTIM_TARGET` — 被害者・標的
- Primary Parent: `D12.OTH`
- Secondary: なし
- Status: `D`
- 判定根拠: 女性旅行者が誘拐・監禁・切断・搾取を直接受ける。
- この伝承における現れ方: 被害者は加害者に標的化された個人として描かれる。

### D13 作用機構
- Primary Child / Value: `D13.PHY.BODY_TRANSFORMATION` — 身体変容
- Primary Parent: `D13.PHY`
- Secondary: `D13.REL.TARGETING` — 標的化; `D13.SOC.COERCE_CONFINE` — 強制・監禁
- Status: `I`
- 判定根拠: 終端で物語を規定する主作用は四肢切断による身体構造の改変。監禁・標的化はその前提過程。
- この伝承における現れ方: 失踪者が後に極端に身体を損なわれた状態で発見される。

### D14 帰結極性
- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 失踪、監禁、重大身体被害、搾取が一貫した負の帰結。
- この伝承における現れ方: 利益・救済を中心とする異伝はScopeにない。

### D15 帰結領域
- Primary Child / Value: `D15.BOD.SEVERE_INJURY` — 重傷・障害
- Primary Parent: `D15.BOD`
- Secondary: `D15.LIF.DISAPPEARANCE` — 失踪・消失; `D15.LIF.FUTURE_CONSTRAINT` — 将来制約
- Status: `D`
- 判定根拠: 四肢切断による重大障害が主帰結。失踪期間と長期搾取による将来制約も独立して重要。
- この伝承における現れ方: 身体と人生の双方が不可逆的に奪われる。

### D16 因果時間構造
- Primary Child / Value: `D16.DLY.DELAYED` — 遅延
- Primary Parent: `D16.DLY`
- Secondary: `D16.PRG.STAGED_PROGRESSION` — 段階進行
- Status: `D`
- 判定根拠: 分離→失踪→捜索失敗→時間経過→再発見という遅延が物語上重要で、その間に被害が進展する。
- この伝承における現れ方: 「すぐ救えなかった時間」が極端な終端を可能にする。

### D17 回避・制御方式
- Primary Child / Value: `D17.AVO.DO_NOT_ENGAGE` — 接触回避
- Primary Parent: `D17.AVO`
- Secondary: `D17.AVO.DISTANCE_ROUTE_CHANGE` — 距離・経路変更
- Status: `I`
- 判定根拠: 見知らぬ場所で単独にならない、不審な状況へ入らないという回避が物語の実践的警告となる。
- この伝承における現れ方: 同行者から離れないことが暗黙の安全規則になる。

### D18 作用レイヤー
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `D`
- 判定根拠: 伝承内部では人間犯罪者による因果作用がある。受容行為自体は危害条件ではなく、流通による現実社会効果を示す独立X Evidenceも未確認。
- この伝承における現れ方: 因果作用は基本的に物語内部の旅行者と加害者の関係で閉じる。

## 2.4. 社会的分布・現実接続

### D19 流通範囲
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
- 判定根拠: 日本の現代伝説集への採録は確認できるが、この個別話の社会的流通範囲を直接固定できない。
- この伝承における現れ方: 「有名だった」という印象と測定可能な流通Evidenceを分離する。

### D20 特権情報保持者
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
- 判定根拠: 犯罪者側が内部事情を知ることは推測できるが、伝承内で真相・回避情報を持つ特権保持者として安定して提示されない。
- この伝承における現れ方: 被害の全容は語り手・同行者にも完全にはアクセスできない。

### D21 現実アンカー
- Primary Child / Value: `D21.A1` — 一般的現実背景
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 海外旅行、店舗、人間犯罪という一般的現実背景を用いるが、特定国・事件・組織へ安定して依存しない。
- この伝承における現れ方: 具体的地名は可変で、世界規則は一般的な「知らない海外」に置かれる。

# 3. R4 QA

## 3.1. causal chain

```text
D11: 旅行先で犯罪者に選ばれる／遭遇する
→ D12: 女性旅行者が被害標的になる
→ D13: 監禁を経て身体を切断・変容させられる
→ D15: 重大障害・失踪・長期制約へ至る
```

D12/D13/D15は接続する。

## 3.2. L3

海外旅行行動を抑制した可能性を推測でコードせず、独立X Evidenceがないため `L3=0`。

## 3.3. U / NA / C

- D01=`U`: 最初期成立年代未確定。
- D19=`U`: 個別話の流通範囲未固定。
- D20=`U`: 特権情報保持者の安定構造なし。
- NA: なし。
- C: なし。

## 3.4. taxonomy gap

現行Scopeは既存taxonomyで表現可能。新Child要求なし。

## 3.5. 旧10との差分

R3 freeze後に旧10を確認した。主要差分は、D02 `U→BOOK`（1994年書籍を現時点の最古確認媒体として扱う）、D04 Secondaryを固定化ではなく再話へ、D05に警告をSecondary追加、D09 `HIDDEN_CAUSE→AGENCY_ATTRIBUTION`、D11 `LOCATION_STATE→SPONTANEOUS_SELECTION`、D13で四肢切断の直接作用をPrimaryへ明示、D19をEvidence不足としてUとした点である。いずれもVersion Scopeと現行coding rulesを優先した再裁定である。