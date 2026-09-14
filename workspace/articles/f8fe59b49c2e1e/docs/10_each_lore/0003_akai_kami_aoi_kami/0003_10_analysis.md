記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

伝承内容・典拠・異伝・証拠上の不確実性は、同一ディレクトリの `0003_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0003`
- `伝承エントリ名称`: 赤い紙・青い紙／赤マント系
- `Macro_Category`: 学校・大学
- `Entry_Type`: 物語・伝説
- `Version_Scope`: **1986年東京都の複数採録で直接確認できる色選択型の最小安定共有核**。学校のトイレで、姿の見えない／正体不明の問いかけ手から紙または色の選択を求められ、遭遇者が答えた色に応じて、意味的に対応する身体的危害または安全な帰結が生じるという学校伝承を分析対象とする。
- `Scope_Status`: `D`

R2で、以下は既定Coding Scopeから除外した。

- 1972年兵庫県尼崎市の赤／白型に、1986年の色別帰結を遡及すること
- 1956年カイナデ／カイナビの対処句を0003の直接祖型とみなすこと
- 1986年赤いちゃんちゃんこ型
- 1990年赤いマント型
- 必ず赤／青二択であること
- 固定された安全色
- 固定された死因・危害機構
- 固定された個室番号・位置
- 怪異の具体的外見

1972年赤／白型はD01/D02等の来歴Evidence、および児童の集団トイレ利用というD18 L3のX Evidenceとして参照する。赤いちゃんちゃんこ／赤マントは伝承史上の近接型として `0003_00_contents.md` に保持するが、D11〜D17の因果列へ混成しない。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示の次元

### 2.1.1. D01: 生成年代

**いつ成立したか**

- Primary Child / Value: `D01.G3` — 1970年代
- Primary Parent: なし
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

1972年刊行の兵庫県尼崎市採録で、学校低学年に「赤い紙／白い紙」の便所伝説が流通していることを直接確認できる。1956年の同句は問い手・答え手と因果機能が異なる関連モチーフなので、0003の生成層へ繰り上げない。

**この伝承における現れ方**

現代的な赤／青型より前に、学校便所で色を問う伝承核が1970年代に確認できる。1972年は起源年ではなく、現時点で安全に年代帯へ写像できる早期直接確認層である。

### 2.1.2. D02: 最古確認流通媒体

**現在の証拠で、伝承が実際に人から人へ流通したことを最も古く確認できる媒体は何か**

- Primary Child / Value: `D02.ORL.SCHOOL_ORAL` — 学校内伝承
- Primary Parent: `D02.ORL` — 口承・限定共同体
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

1972年採録は、低学年、特に1年生の間に伝説が広がっていることを記録する。現行rulesでは採録情報が実流通媒体を直接示す場合はDとできるため、研究誌そのものではなく学校内口承を最古確認媒体とする。

**この伝承における現れ方**

便所の怪異と色の問いが児童間で共有され、学校共同体内部の対人口承として保持される。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）

**今回コードするVersion Scopeで、伝承が受容者へ流通したことを確認できる媒体は何か**

- Primary Child / Value: `D03.ORL.SCHOOL_ORAL` — 学校内伝承
- Primary Parent: `D03.ORL` — 口承・限定共同体
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

1986年Scopeは中学生から採録された「学校の世間話」であり、学校・生徒集団内の対人口承が確認流通回路である。研究論文・データベースへの収録は伝承の流通媒体ではなくEvidence記録なのでD03へ含めない。

**この伝承における現れ方**

色・安全回答・危害内容等のローカルなルールが児童・生徒集団内で共有される。

### 2.1.4. D04: 生成・変容パターン

**時間とともにどう変形したか**

- Primary Child / Value: `D04.VAR.ORAL_VARIATION` — 口承変異
- Primary Parent: `D04.VAR` — 再話・変異
- Secondary: `D04.VAR.ACCRETION` — 増補
- Status: `I` — Inferred

**判定根拠**

同一1986年資料群で赤／青／白、赤／青／黄、赤／紫が併存し、色・帰結が置換され、第三色・安全回答等が追加されていること自体は直接確認できる。ただし、それらが具体的にどの口承伝達過程で変形・増補されたかは直接観察していないため、生成・変容パターンとしてのStatusはIとする。

**この伝承における現れ方**

「トイレで問われる→色を選ぶ→色に対応した結果」という生成規則を保ちながら、選択肢と帰結が学校・語りごとに差し替えられる。

### 2.1.5. D05: 提示形式

**どんなコミュニケーション形式で提示されるか**

- Primary Child / Value: `D05.HRS.SCHOOL_WORK_HEARSAY` — 学校・職場伝承
- Primary Parent: `D05.HRS` — 伝聞叙述
- Secondary: `D05.PRP.PREDICTIVE_CLAIM` — 予測命題
- Status: `I` — Inferred

**判定根拠**

学校内で共有される世間話として受容され、「色Xを答えるとYになる」という条件付き予測命題を伴う。採録要約から原発話の完全な提示形式までは復元できないためIとする。

**この伝承における現れ方**

完成した一人称体験談というより、学校内で共有される「このトイレではこう問われ、こう答えるとこうなる」という条件付き伝聞として機能する。

### 2.1.6. D06: 真実性提示

**どんな「本当らしさ」を要求するか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U` — Unknown

**判定根拠**

日文研DB要約と採録書誌から、受容者へ「共同体既知事実」「条件付き信念」「真偽未確定」のどの認識論的スタンスで提示されたかを一意に固定できない。学校伝承であることだけからT3を推測しない。

**この伝承における現れ方**

「学校で共有される」ことと「事実として信じる」ことを分離し、原発話の真実性提示が不足する部分はUに保持する。

## 2.2. 意味形成の次元

### 2.2.1. D07: 意味形成対象

**何が不可解・不確実なのか**

- Primary Child / Value: `D07.PSE.DANGEROUS_PLACE` — 危険・怪異場所
- Primary Parent: `D07.PSE` — 場所・空間・環境
- Secondary: `D07.ANO.UNKNOWN_EXISTENCE` — 未知存在
- Status: `I` — Inferred

**判定根拠**

この伝承がなければ、通常の日常施設である学校トイレの一部がなぜ危険・怪異的なのか、そこで応答を迫る正体不明の存在が何なのかが説明されない。R2の共通核では場所危険性をPrimaryとし、不可視の問いかけ手をSecondaryとする。

**この伝承における現れ方**

普通の学校トイレが、特定の問いと選択規則を持つ危険空間として再解釈され、その背後に人格的な未知存在が置かれる。

### 2.2.2. D08: 意味形成契機

**何を手掛かりに問題化されるか**

- Primary Child / Value: `D08.STY.COMMUNITY_REPETITION` — 共同体反復証言
- Primary Parent: `D08.STY` — 社会的証言
- Secondary: `D08.CLM.UNSUPPORTED_ASSERTION` — 根拠未提示の主張
- Status: `I` — Inferred

**判定根拠**

学校集団で反復して共有される世間話そのものが、通常のトイレを問題化する主要な手掛かりである。怪異による現実の殺傷事件・物的痕跡を前提としないため、根拠未提示の主張もSecondaryとする。

**この伝承における現れ方**

「そこではこう聞かれる」という共同体内の反復が、場所と回答を危険の徴候へ変える。

### 2.2.3. D09: 意味付与操作

**不可解なものをどう理解可能にするか**

- Primary Child / Value: `D09.PPR.CORRELATION_RULE` — 相関規則化
- Primary Parent: `D09.PPR` — パターン化・予測
- Secondary: `D09.AGN.AGENCY_ATTRIBUTION` — 主体化
- Status: `D` — Direct

**判定根拠**

1986年採録群は「赤→血・危害」「青→青ざめる」等、回答語と帰結を意味対応させる規則を明示する。さらに、問いを出して回答後の結果を実行する不可視の主体が置かれる。

**この伝承における現れ方**

無秩序な怪異ではなく、色と身体結果の対応表を持つ予測可能な規則として整理される。

### 2.2.4. D10: 因果源存在論

**原因を何として世界に置くか**

- Primary Child / Value: `D10.SUP.YOKAI_ENTITY` — 妖怪・人格怪異
- Primary Parent: `D10.SUP` — 超自然主体・力
- Secondary: なし
- Status: `I` — Inferred

**判定根拠**

正体・姿は固定されないが、声で問い、回答に応じて通常では不可能な身体結果を起こす人格的怪異として因果源が置かれる。死者霊・幽霊という由来までは固定しない。

**この伝承における現れ方**

危害はトイレ設備の自然作用ではなく、応答を理解して結果を変える人格的な怪異主体へ帰属される。

## 2.3. 因果・行動モデルの次元

### 2.3.1. D11: 発動・接触条件

**何を契機に因果系へ入るか**

- Primary Child / Value: `D11.MAN.RESPOND_CHOOSE` — 答える・選択する
- Primary Parent: `D11.MAN` — 操作・儀式
- Secondary: `D11.CON.LOCATION_STATE` — 位置・状態条件
- Status: `D` — Direct

**判定根拠**

色を回答する／選ぶことが帰結分岐の直接条件として複数採録に明示される。学校トイレにいることも共通の成立条件だが、単にトイレへ入ることより回答選択が結果を確定するためPrimaryとする。

**この伝承における現れ方**

場所へ居合わせ、問いを受け、回答することで、危害または安全という異なる結果へ分岐する。

### 2.3.2. D12: 作用対象

**誰／何に作用するか**

- Primary Child / Value: `D12.FOC.PROTAGONIST_EXPERIENCER` — 主人公・体験者
- Primary Parent: `D12.FOC` — 焦点人物
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

D13の主作用である身体攻撃を直接受けるのは、問いへ回答した遭遇者本人である。

**この伝承における現れ方**

学校全体へ一律に作用するのではなく、個室で怪異と応答関係に入った人物へ作用が向く。

### 2.3.3. D13: 作用機構

**因果源が対象へ何をするか**

- Primary Child / Value: `D13.PHY.PHYSICAL_ATTACK` — 物理攻撃
- Primary Parent: `D13.PHY` — 身体・物質作用
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

1986年各型で、血だらけにする、首を絞める、血を抜く、便器へ引き込む等の身体的危害が回答に対応して発生する。

**この伝承における現れ方**

色は単なる予兆ではなく、回答者の身体へ結果を実現する攻撃規則として働く。

### 2.3.4. D14: 帰結極性

**結果は正・負・中立・混合か**

- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

有害分岐の主帰結は身体危害・死亡である。安全回答は「何もされない／怪異が消える」という危害の不発であり、独立した正の利益ではない。したがって `D14.MIX` の「重要な正負帰結が併存する」には当てず、負とする。安全分岐はD17で保持する。

**この伝承における現れ方**

伝承は危険を提示し、その危険を回答知識によって回避できる構造であり、成功時に追加利益を与える話ではない。

### 2.3.5. D15: 帰結領域

**何の領域が最終的に変わるか**

- Primary Child / Value: `D15.BOD.SEVERE_INJURY` — 重傷・障害
- Primary Parent: `D15.BOD` — 身体・健康
- Secondary: `D15.BOD.DEATH` — 死亡
- Status: `D` — Direct

**判定根拠**

1986年Scopeでは血だらけ、首絞め、失血等の重い身体被害が直接示され、別型では殺害も明示される。安全分岐は被害不発であって独立した帰結領域の変化として追加しない。

**この伝承における現れ方**

回答語が身体状態へ変換され、危険色では重傷・死亡に至り得る。

### 2.3.6. D16: 因果時間構造

**発動条件・原因成立から主作用／主帰結までの関係は、時間上どのように編成されているか**

- Primary Child / Value: `D16.EVT.SEQUENTIAL_EPISODE` — エピソード内段階進行
- Primary Parent: `D16.EVT` — 単一エピソード・事象内
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

トイレで問いを受ける→色を答える→回答に対応した帰結が起こる、という複数段階が一続きの遭遇内で順に編成される。遅延時間そのものではなく、この順序が伝承の構造核である。

**この伝承における現れ方**

問いと選択がなければ後続結果が確定しないため、単なる「即時」より段階進行として記述する。

### 2.3.7. D17: 回避・制御方式

**結果をどう回避・制御・利用できるか**

- Primary Child / Value: `D17.RUL.CORRECT_ANSWER` — 正答・選択
- Primary Parent: `D17.RUL` — 規則遵守
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

1986年の赤／青／白、赤／青／黄、赤／紫の各型には、白・黄・紫等、型ごとに異なる安全選択肢が直接確認される。特定色を普遍的正答とはしないが、「安全な回答がある」という制御構造は共通する。

**この伝承における現れ方**

怪異の危害は完全不可避ではなく、ローカルな回答知識によって回避できる。ただし何色が正しいかはVersionごとに変わる。

### 2.3.8. D18: 作用レイヤー

**因果効力は伝承内／受容者／社会現実のどこに及ぶか**

- Primary Child / Value: `D18.L1=1; D18.L2=0; D18.L3=1`
- Primary Parent: なし
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

L1では回答と身体被害の因果が伝承内で作動する。L2では伝承を聞く・知ること自体が怪異の発動条件ではない。L3は、1972年兵庫採録に、噂を恐れた低学年児童が数珠つなぎで便所へ行くというX Evidenceが直接記録されているため1とする。

**この伝承における現れ方**

怪談内部の身体危害と、噂を受容した現実の児童の行動変化を別レイヤーとして保持する。

## 2.4. 社会的埋め込みの次元

### 2.4.1. D19: 流通範囲

**誰の間に伝承が流通するか**

- Primary Child / Value: `D19.KIN.SCHOOL_YOUTH` — 学校・若者集団
- Primary Parent: `D19.KIN` — 家族・仲間
- Secondary: `D19.LOC.REGIONAL` — 地域・地方
- Status: `D` — Direct

**判定根拠**

1972年兵庫と1986年東京の異なる地域で学校児童・生徒圏の採録が確認できる。受容集団としては学校・若者文化が最も直接的であり、地理的には少なくとも単一学校・単一市町村に限定されないためREGIONALをSecondaryとする。全国的大衆までの拡張はR2 Scopeからは行わない。

**この伝承における現れ方**

学校ごとにルールを変えながら、地域をまたいで児童・生徒文化の中に現れる。

### 2.4.2. D20: 特権情報保持者

**誰が真相・追加情報を持つか**

- Primary Child / Value: `D20.NON.COMMON_KNOWLEDGE` — 一般共有
- Primary Parent: `D20.NON` — 特権なし
- Secondary: なし
- Status: `I` — Inferred

**判定根拠**

安全色を含む回答規則そのものが学校伝承として共有され、Scope内では特定の専門家・内部者だけが追加真相や有効な制御情報を独占する構造を必要としない。

**この伝承における現れ方**

攻略知識は個人の秘密ではなく、当該児童・生徒共同体の共有知として流通する。

### 2.4.3. D21: 現実アンカー

**実在世界へどの程度固定されるか**

- Primary Child / Value: `D21.A1` — 一般的現実背景
- Primary Parent: なし
- Secondary: なし
- Status: `D` — Direct

**判定根拠**

学校・トイレという実在の日常環境に依存する一方、特定の実在校・制度・事件史を成立条件としない。

**この伝承における現れ方**

どの学校にもあるトイレへ移植可能であり、固有地名を交換しても意味形成核が維持される。

# 3. R4 Entry QA

## 3.1. D12 → D13 → D15 causal QA

最終因果列は次のように接続する。

```text
D11: D11.MAN.RESPOND_CHOOSE
  + Secondary: D11.CON.LOCATION_STATE
→ D12: D12.FOC.PROTAGONIST_EXPERIENCER
→ D13: D13.PHY.PHYSICAL_ATTACK
→ D15: D15.BOD.SEVERE_INJURY
  + Secondary: D15.BOD.DEATH
```

- D12はD13の直接対象である回答者本人。
- D13は「何をするか」であり、身体攻撃。
- D15は攻撃の結果として変化する身体・健康領域。
- 安全回答の分岐は攻撃の不発なのでD17へ保持し、D15へ架空の正帰結を追加しない。

**QA結果: Pass。**

## 3.2. D18 L3 evidence QA

1972年兵庫採録には、伝説を恐れた低学年児童が数珠つなぎで便所へ行くという現実行動が記録される。これは噂の受容が現実の学校行動を変えた `X` Evidenceである。

**QA結果: `L3=1` を維持。Pass。**

## 3.3. U / NA / C QA

- `U`: D06のみ。原発話の認識論的スタンスを一意に決められないため意図的にU。
- `NA`: なし。
- `C`: なし。

R3ではD14を `MIX` としたが、現行taxonomyの `MIX` は重要な正負帰結の併存を指す。安全回答は正の利益ではなく「危害が起きない」分岐なので、R4 QAで `D14.NEG` へ修正した。

**QA結果: 推測によるU埋め、不要なNA/Cなし。Pass。**

## 3.4. taxonomy gap QA

既存taxonomyで安定共有核を表現できる。

R3で留意した「有害分岐＋無害分岐」の極性問題は、

- D14=`NEG`: 主帰結は危害
- D17=`CORRECT_ANSWER`: 危害を回避する安全選択

と分離することで表現できるため、本Entry単独ではtaxonomy gapとしない。

**QA結果: taxonomy変更候補なし。Pass。**

## 3.5. R3 freezeからR4確定値への変更

R3 checkpoint `7e08f01ca3119e71268292513504ff79f6987d60` から、Entry QAにより次の2点を修正した。

| D | R3 freeze | R4確定 | 理由 |
|---|---|---|---|
| D04 Status | `D` | `I` | 複数変種の存在は直接確認できるが、それが口承過程で変化・増補したという生成過程は推論。 |
| D14 | `D14.MIX / D` | `D14.NEG / D` | 安全回答は正の利益ではなく危害の不発。MIXの「正負併存」定義に該当しない。 |

### R4手順上の監査注記

本R4では、R3 freeze後に旧 `0003_10_analysis.md` を取得した後、正式なQA記述を確定した。そのため、上記D04/D14修正は旧値を既知の状態で行われており、完全ブラインドなQAではない。

ただし、D04は「変種の直接確認」と「変容過程の推論」を区別するDimension Status規則、D14は `NEG/MIX` のtaxonomy定義からそれぞれ再導出した。旧値へ一致させることを根拠にはしていない。この順序逸脱は監査制約として明示し、隠蔽しない。

## 3.6. R3ブラインド監査

R3時、taxonomy探索のGitHub code search結果が旧0003のD01=`D01.G3`、D02=`D02.ORL.SCHOOL_ORAL` の2値だけを偶発的に表示した。旧本文・根拠・D03〜D21はR3 freeze前に意図的に開いていない。

D01/D02はR1 Evidenceと現行rulesから再導出して同値に到達したが、厳密にはこの2次元のみ完全ブラインドではない。R3 work fileにも同じ監査注記を保持する。

# 4. 再コーディング前 `0003_10_analysis.md` との差分監査

比較対象は、再コーディング前 checkpoint `5c577eb3d3acfa3f69fd4ca841875888b27305cf` の旧 `0003_10_analysis.md`。旧ExcelはR4では参照しない。

## 4.1. Version Scope差分

旧Scopeは、1972年赤／白、1986年赤／紫・赤いちゃんちゃんこ、1990年赤いマント等を同一Coding Scopeへ広く含めていた。

新Scopeは、1986年東京都で直接比較できる**色選択型の共通核**へ限定し、赤いちゃんちゃんこ／赤マントは近接派生としてEvidence史へ残すが因果コードへ混成しない。

このScope差は、D05、D07、D11、D17、D19等の差分に波及する。

## 4.2. D01〜D21差分分類

| D | 旧値 | R4確定値 | 差分分類 | 判定 |
|---|---|---|---|---|
| D01 | `D01.G3 / D` | `D01.G3 / D` | same | 変更なし。なおR3時に旧値が偶発表示された2次元の1つ。 |
| D02 | `D02.ORL.SCHOOL_ORAL / I` | `D02.ORL.SCHOOL_ORAL / D` | `Status mismatch`, `Evidence mismatch` | 1972年原採録が学校内流通を直接記すためDへ強化。R3時の偶発表示対象。 |
| D03 | `D03.ORL.SCHOOL_ORAL / I` | `D03.ORL.SCHOOL_ORAL / D` | `Status mismatch`, `Evidence mismatch` | 1986年の学校世間話としての採録を直接流通Evidenceと評価。 |
| D04 | `ORAL_VARIATION + ACCRETION / I` | 同左 | same | R3ではDとしたがR4 QAでIへ戻した。生成過程自体は推論。 |
| D05 | `WARNING / I` | `SCHOOL_WORK_HEARSAY + PREDICTIVE_CLAIM / I` | `Code-selection mismatch`, `Scope mismatch` | 警告機能より、受容時の学校伝聞形式＋条件予測を優先。 |
| D06 | `D06.T3 / I` | `U` | `Evidence mismatch`, `Prior coding error` | 旧判定は1990年赤マント系の「生徒には周知」を広いScopeから真実性提示へ転用。新Scopeの原発話ではスタンスを固定できない。 |
| D07 | `UNKNOWN_EXISTENCE / I` | `DANGEROUS_PLACE + UNKNOWN_EXISTENCE / I` | `Code-selection mismatch`, `Scope mismatch` | 共通核ではトイレ空間の危険化をPrimary、未知存在をSecondaryへ。 |
| D08 | `UNSUPPORTED_ASSERTION / I` | `COMMUNITY_REPETITION + UNSUPPORTED_ASSERTION / I` | `Code-selection mismatch` | 学校内反復を意味形成契機のPrimaryへ。 |
| D09 | `AVOIDANCE_RULE / I` | `CORRELATION_RULE + AGENCY_ATTRIBUTION / D` | `Code-selection mismatch`, `Status mismatch` | 回避知識はD17へ分離し、色→帰結の対応規則を中心操作とした。 |
| D10 | `YOKAI_ENTITY / I` | `YOKAI_ENTITY / I` | same | 変更なし。 |
| D11 | `RESPOND_CHOOSE / D` | `RESPOND_CHOOSE + LOCATION_STATE / D` | `Code-selection mismatch` | トイレにいることを独立したSecondary条件として明示。 |
| D12 | `PROTAGONIST_EXPERIENCER / I` | `PROTAGONIST_EXPERIENCER / D` | `Status mismatch`, `Evidence mismatch` | 回答者本人が直接攻撃対象であることを採録内容から直接確認。 |
| D13 | `PHYSICAL_ATTACK / I` | `PHYSICAL_ATTACK / D` | `Status mismatch`, `Evidence mismatch` | 首絞め・失血等の作用が採録に直接記載される。 |
| D14 | `NEG / I` | `NEG / D` | `Status mismatch` | コードは同じ。R3 interimのMIXはR4 QAで定義に従いNEGへ修正。 |
| D15 | `SEVERE_INJURY + DEATH / I` | 同コード / `D` | `Status mismatch`, `Evidence mismatch` | 重傷・殺害が1986年採録に直接記載される。 |
| D16 | `EVT.IMMEDIATE / I` | `EVT.SEQUENTIAL_EPISODE / D` | `Code-selection mismatch`, `Status mismatch` | 問い→回答→帰結という段階順序が構造核。単なる即時性より段階進行を優先。 |
| D17 | `CORRECT_ANSWER + DO_NOT_ENGAGE / C` | `CORRECT_ANSWER / D` | `Scope mismatch`, `Code-selection mismatch`, `Status mismatch` | 1986年色選択型では各採録に安全回答があり、場所回避という外部行動をContent上のSecondary制御へ混ぜない。 |
| D18 | `L1=1,L2=0,L3=1 / D` | 同左 | same | 変更なし。 |
| D19 | `NATIONAL_PUBLIC / D` | `SCHOOL_YOUTH + REGIONAL / D` | `Scope mismatch`, `Code-selection mismatch` | 後代の全国整理を既定Scopeへ遡及せず、直接確認できる学校若者圏＋地域横断までに留める。 |
| D20 | `COMMON_KNOWLEDGE / I` | 同左 | same | 変更なし。 |
| D21 | `A1 / I` | `A1 / D` | `Status mismatch`, `Evidence mismatch` | 学校・トイレという一般的現実背景は採録内容に直接明示される。 |

# 5. R4結論

- D12→D13→D15 causal QA: **Pass**
- D18 L3 evidence QA: **Pass**
- U / NA / C QA: **Pass**
- taxonomy gap QA: **Pass / 変更候補なし**
- Coding正本 `0003_10_analysis.md`: **更新済み**
- 再コーディング前旧10との差分比較: **完了**
- 旧Excel比較: **R5 Global Reconciliationへ移管**

R4確定後の0003は、1972年早期史と1986年色選択型のContent Scopeを分離し、赤いちゃんちゃんこ／赤マントを近接派生として混成しない状態で、現行baselineに基づくD01〜D21を確定した。
