記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0091_00_contents.md`、R1/R2/R3監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0091`
- `伝承エントリ名称`: ベッドの下の男
- `Macro_Category`: 犯罪・社会不安
- `Entry_Type`: FOAF
- `Version_Scope`: 1990年代に日本で流通し1996年までに現代伝説集へ採録された、私的空間への人間侵入者潜伏型。自宅・ホテル等へ居住者／宿泊者より先に見知らぬ人物が侵入し、ベッド下等の死角へ隠れ、本人は危険を知らないまま同室する。第三者や違和感を介して危険が察知され、本人が退避した後に侵入者の存在が確認される共有核を対象とする。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Primary Child / Value: `D01.G5` — 1990年代
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 1990年代の日本で流通したという後代整理と1996年採録を確認できる。単一初出・成立年は直接固定できないためI。
- この伝承における現れ方: 少なくとも1990年代には日本の犯罪不安型現代伝説として流通していた。

### D02 最古確認流通媒体
- Primary Child / Value: `D02.PRT.BOOK` — 書籍
- Primary Parent: `D02.PRT` — 印刷・書簡
- Secondary: なし
- Status: `D`
- 判定根拠: 現在直接確認できる最古媒体は1996年刊『走るお婆さん―日本の現代伝説』。
- この伝承における現れ方: 少なくとも1996年には採録・読者提示されている。

### D03 確認流通媒体ポートフォリオ
- Primary Child / Value: `D03.PRT.BOOK` — 書籍
- Primary Parent: `D03.PRT`
- Secondary: なし
- Status: `D`
- 判定根拠: Version Scopeで直接確認できる実流通媒体は書籍。口承・漫画等は後代整理のみで、具体的媒体資料を固定していないため追加しない。
- この伝承における現れ方: 私室の潜伏者譚が現代伝説集として再提示される。

### D04 生成・変容パターン
- Primary Child / Value: `D04.VAR.RETELLING` — 再話
- Primary Parent: `D04.VAR`
- Secondary: `D04.VAR.LOCALIZATION` — 地域化
- Status: `I`
- 判定根拠: 自宅／ホテル、発見者、鏡・メモ・物の位置、刃物等を差し替えながら、先行潜伏→未認識→察知→退避→確認という核を保つ。
- この伝承における現れ方: 生活空間に合わせて細部を変えながら同型の危険譚として反復される。

### D05 提示形式
- Primary Child / Value: `D05.HRS.FOAF` — FOAF
- Primary Parent: `D05.HRS`
- Secondary: `D05.RUL.WARNING` — 警告
- Status: `I`
- 判定根拠: 知人周辺の実話らしい近接伝聞として提示され、防犯上の警告へ接続する。
- この伝承における現れ方: 「友人の知人の部屋で起きた」程度の検証しにくい近さが現実感を与える。

### D06 真実性提示
- Primary Child / Value: `D06.T2` — 近接伝聞事実
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 語り手自身ではなく、身近な人脈上の具体的被害として事実らしく提示される。
- この伝承における現れ方: 現実にあり得る住居侵入とFOAF距離が結びつき、実話らしさを作る。

## 2.2. 意味形成

### D07 意味形成対象
- Primary Child / Value: `D07.ICT.STRANGER_THREAT` — 見知らぬ他者の脅威
- Primary Parent: `D07.ICT`
- Secondary: `D07.ICT.ABDUCTION_ASSAULT` — 誘拐・暴行
- Status: `D`
- 判定根拠: 安全なはずの私室に見知らぬ人間がすでに入り、至近距離に潜伏していることが中心不安。殺傷・暴行可能性が補助的に強調される。
- この伝承における現れ方: 「危険は外にある」という境界が崩れる。

### D08 意味形成契機
- Primary Child / Value: `D08.STY.FOAF_REPORT` — 近接伝聞
- Primary Parent: `D08.STY`
- Secondary: `D08.TRC.PHYSICAL_TRACE` — 物的痕跡
- Status: `I`
- 判定根拠: 受容者にはFOAF報告が危険認識の契機となり、物語内では鏡・物の位置等の痕跡が潜伏者察知の手掛かりとなる異伝がある。
- この伝承における現れ方: 普段無視する死角や小さな違和感が犯罪危険の証拠として再解釈される。

### D09 意味付与操作
- Primary Child / Value: `D09.AGN.AGENCY_ATTRIBUTION` — 主体化
- Primary Parent: `D09.AGN`
- Secondary: `D09.NOR.TABOOIZATION` — 禁忌化
- Status: `I`
- 判定根拠: 私室の違和感・危険を、人間侵入者という意図主体へ帰属し、一人で確認・対峙しないという行動規範へ変換する。
- この伝承における現れ方: 曖昧な異変が「誰かがいる」という明確な脅威になる。

### D10 因果源存在論
- Primary Child / Value: `D10.HUM.INDIVIDUAL_HUMAN` — 個人
- Primary Parent: `D10.HUM`
- Secondary: なし
- Status: `D`
- 判定根拠: 因果源は超自然存在ではなく、私室へ侵入し潜伏する人間。
- この伝承における現れ方: 普通の人間の悪意と死角利用だけで恐怖が成立する。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary Child / Value: `D11.PAS.SPONTANEOUS_SELECTION` — 偶然選ばれる・遭遇する
- Primary Parent: `D11.PAS`
- Secondary: `D11.CON.LOCATION_STATE` — 特定場所・状態にいる
- Status: `I`
- 判定根拠: 帰宅・入室自体が再現的な発動操作ではない。居住者が侵入者に被害対象として選ばれ、侵入者が先に私室へ潜伏している状態が因果系への入口。
- この伝承における現れ方: 日常生活の延長で、本人の意図と無関係に危険へ巻き込まれる。

### D12 作用対象
- Primary Child / Value: `D12.OTH.VICTIM_TARGET` — 被害者・標的
- Primary Parent: `D12.OTH`
- Secondary: なし
- Status: `D`
- 判定根拠: 居住者／宿泊者が侵入者の潜在的襲撃対象。
- この伝承における現れ方: 主人公は知らないまま標的になっている。

### D13 作用機構
- Primary Child / Value: `D13.REL.TARGETING` — 標的化
- Primary Parent: `D13.REL`
- Secondary: `D13.SOC.COERCE_CONFINE` — 強制・監禁
- Status: `I`
- 判定根拠: 侵入者は特定居住者を襲撃可能な対象として待つ。現行taxonomyには「物理的に隠れて待ち伏せる／潜伏する」を直接表すChildがないため、標的化で近似する。監禁・強制は異伝で潜在的な作用として保持する。
- この伝承における現れ方: 「すでに自分が狙われ、至近距離で待たれていた」という関係が恐怖の中心になる。

### D14 帰結極性
- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 明示終端で退避に成功しても、中心因果は侵入・潜伏・殺傷可能性という負の脅威である。回避成功を正帰結として同格に数えず、Scopeの危害方向をコードする。
- この伝承における現れ方: 「助かった」結末でも、意味上の中心は襲われる寸前だったという危険にある。

### D15 帰結領域
- Primary Child / Value: `D15.BEH.AVOIDANCE_ROUTE_CHANGE` — 回避・経路変更
- Primary Parent: `D15.BEH`
- Secondary: `D15.KNW.REVELATION_KNOWLEDGE` — 真相・知識獲得
- Status: `I`
- 判定根拠: 代表形で実現する終端は、主人公が室外へ退避して危険との接触を避け、後から侵入者の存在を確認すること。未遂の殺害・傷害を実現帰結としてはコードしない。
- この伝承における現れ方: 行動選択が「部屋に留まる」から「退避する」へ変わり、その判断が侵入者確認によって正当化される。

### D16 因果時間構造
- Primary Child / Value: `D16.EVT.SEQUENTIAL_EPISODE` — エピソード内段階進行
- Primary Parent: `D16.EVT`
- Secondary: なし
- Status: `D`
- 判定根拠: 先行潜伏→入室→違和感／第三者察知→退避→確認が一つのエピソード内で順に進む。
- この伝承における現れ方: 情報差が段階的に解消されることで恐怖が形成される。

### D17 回避・制御方式
- Primary Child / Value: `D17.AVO.FLEE_ESCAPE` — 逃走
- Primary Parent: `D17.AVO`
- Secondary: `D17.RIT.LEGAL_OFFICIAL` — 公的・法的介入
- Status: `I`
- 判定根拠: 危険を察した後、本人が侵入者へ接近せず室外へ退避し、警察等へ確認を委ねる型が代表的。
- この伝承における現れ方: 自力対決ではなく、物理的距離を取り公的主体へ処理を移す。

### D18 作用レイヤー
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `D`
- 判定根拠: 伝承内部では人間侵入者による因果作用がある。聞くこと自体が危害条件ではなく、流通による現実社会効果を示す独立X Evidenceも確認していない。
- この伝承における現れ方: 因果作用は物語内部の侵入者と潜在的被害者の関係で閉じる。

## 2.4. 社会的分布・現実接続

### D19 流通範囲
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
- 判定根拠: 1990年代に日本で流行したという二次整理はあるが、この個別話の社会範囲をtaxonomy上の一範囲へ安全に固定する独立Evidenceが不足する。
- この伝承における現れ方: 流行の存在と、測定可能な流通範囲を分離する。

### D20 特権情報保持者
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
- 判定根拠: 発見者・警察・侵入者など情報保持構造が異伝ごとに変わり、共有核で一意に固定できない。
- この伝承における現れ方: 主人公より先に危険を知る人物がいる型はあるが、その役割は固定されない。

### D21 現実アンカー
- Primary Child / Value: `D21.A1` — 一般的現実背景
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 自宅、ホテル、ベッド、警察という一般的現実背景で成立し、特定施設・実在事件へ依存しない。
- この伝承における現れ方: どこにでもある私的空間と死角が恐怖装置になる。

# 3. R4 QA

## 3.1. causal chain

```text
D11: 日常生活中に侵入者から標的化される
→ D12: 居住者／宿泊者が被害標的となる
→ D13: 侵入者が同一私室に潜伏し標的化を維持する
→ D15: 危険察知後に退避し、侵入者存在を確認する
```

実現帰結と未遂危害を分離して接続する。

## 3.2. L3

防犯意識への影響は推測可能でも、独立X Evidenceを確認していないため `L3=0`。

## 3.3. U / NA / C

- D19=`U`: 個別話の流通範囲未固定。
- D20=`U`: 特権情報保持者の安定構造なし。
- NA: なし。
- C: なし。

## 3.4. taxonomy gap

現行D13には「物理的に隠れて潜伏する／待ち伏せする」を直接表すChildがない。`D13.REL.TARGETING` で近似しているが、**D13 潜伏・隠匿（physical hiding / ambush）**をGlobal Reconciliation候補として保持する。

## 3.5. 旧10との差分

R3 freeze後に旧10を確認した。主要差分は、D01 `U→G5(I)`、D02 `U→BOOK(D)`、D04 `STABILIZED_CANON→RETELLING`、D05にWARNINGを追加、D09 `HIDDEN_CAUSE→AGENCY_ATTRIBUTION`、D11 `ENTER_PASS_CROSS→SPONTANEOUS_SELECTION`、D14 `MIX→NEG`、D15の旧「災厄回避」から現行taxonomy上の `AVOIDANCE_ROUTE_CHANGE` へ再配置、D17 `DO_NOT_ENGAGE→FLEE_ESCAPE`、D19をUで維持した点である。差分はVersion Scope・Evidence責任・現行taxonomyの再適用による。