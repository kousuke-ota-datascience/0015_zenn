記載内容については、以下3文書のインストラクションに従う。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

伝承内容・典拠・異伝・証拠上の不確実性は、同一ディレクトリの `0089_00_contents.md` を参照する。

本ファイルは、パイロット分析で確定した `Pilot Coding` の値・Statusを文書形式へ移植する。今回の文書化だけを理由にコードを再判定しない。
# 1. 伝承エントリ基本情報

- `Entry_ID`: `89`
- `伝承エントリ名称`: 日本だるま／だるま女
- `Macro_Category`: 犯罪・社会不安
- `Entry_Type`: FOAF
- `Version_Scope`: Candidate Masterの代表形。異伝を混成せず、メモ・主典拠の範囲でコード
- `Evidence_Rank`: `B`
- `主な成立・流通期`: 1980-90年代
- `ネット発祥`: No

# 2. 分析概念次元

## 2.1. 来歴・流通・提示の次元

### 2.1.1. D01: 生成年代

**いつ成立したか**

- Value: `D01.G4` — 1980年代
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

### 2.1.2. D02: 最古確認流通媒体

**現在の証拠で、伝承が実際に人から人へ流通したことを最も古く確認できる媒体は何か**

- Primary: なし
- Secondary: なし
- Status: `U` — Unknown（証拠不足）

パイロット時点の典拠では、この次元を一意に判定するための証拠が不足しているため `U` を保持する。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）

**今回コードするVersion Scopeで、伝承が受容者へ流通したことを確認できる媒体は何か**

- Primary Child: `D03.ORL.PEER_ORAL` — 友人・仲間内伝承
- Primary Parent: `D03.ORL` — 口承・限定共同体
- Secondary: なし
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

### 2.1.4. D04: 生成・変容パターン

**時間とともにどう変形したか**

- Primary Child: `D04.STB.STABILIZED_CANON` — 定型化・カノン化
- Primary Parent: `D04.STB` — 安定・固定
- Secondary: なし
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

### 2.1.5. D05: 提示形式

**どんなコミュニケーション形式で提示されるか**

- Primary Child: `D05.HRS.FOAF` — FOAF
- Primary Parent: `D05.HRS` — 伝聞叙述
- Secondary: なし
- Status: `D` — Direct（典拠に直接明示）

`0180` と同様、`Pilot Coding` の既存判定を文書化した。直接証拠の確認先は `*_00_contents.md` の典拠・証拠対応を参照する。

### 2.1.6. D06: 真実性提示

**どんな「本当らしさ」を要求するか**

- Value: `D06.T2` — 近接伝聞事実
- Status: `D` — Direct（典拠に直接明示）

`0180` と同様、`Pilot Coding` の既存判定を文書化した。直接証拠の確認先は `*_00_contents.md` の典拠・証拠対応を参照する。

## 2.2. 意味形成の次元

### 2.2.1. D07: 意味形成対象

**何が不可解・不確実なのか**

- Primary Child: `D07.ICT.STRANGER_THREAT` — 見知らぬ他者の脅威
- Primary Parent: `D07.ICT` — 対人・犯罪脅威
- Secondary: なし
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

### 2.2.2. D08: 意味形成契機

**何を手掛かりに問題化されるか**

- Primary Child: `D08.STY.FOAF_REPORT` — 近接伝聞
- Primary Parent: `D08.STY` — 社会的証言
- Secondary: なし
- Status: `D` — Direct（典拠に直接明示）

`0180` と同様、`Pilot Coding` の既存判定を文書化した。直接証拠の確認先は `*_00_contents.md` の典拠・証拠対応を参照する。

### 2.2.3. D09: 意味付与操作

**不可解なものをどう理解可能にするか**

- Primary Child: `D09.CAT.NAMING` — 命名
- Primary Parent: `D09.CAT` — カテゴリー化
- Secondary: なし
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

### 2.2.4. D10: 因果源存在論

**原因を何として世界に置くか**

- Primary Child: `D10.HUM.INDIVIDUAL_HUMAN` — 個人
- Primary Parent: `D10.HUM` — 人間・社会主体
- Secondary: なし
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

## 2.3. 因果・行動モデルの次元

### 2.3.1. D11: 発動・接触条件

**何を契機に因果系へ入るか**

- Primary Child: `D11.SOC.MEET_INTERACT` — 会う・会話する
- Primary Parent: `D11.SOC` — 社会関係・取引
- Secondary: なし
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

### 2.3.2. D12: 作用対象

**誰／何に作用するか**

- Primary Child: `D12.FOC.PROTAGONIST_EXPERIENCER` — 主人公・体験者
- Primary Parent: `D12.FOC` — 焦点人物
- Secondary: なし
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

### 2.3.3. D13: 作用機構

**因果源が対象へ何をするか**

- Primary Child: `D13.PHY.PHYSICAL_ATTACK` — 物理攻撃
- Primary Parent: `D13.PHY` — 身体・物質作用
- Secondary: なし
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

### 2.3.4. D14: 帰結極性

**結果は正・負・中立・混合か**

- Value: `D14.NEG` — 負
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

### 2.3.5. D15: 帰結領域

**何の領域が最終的に変わるか**

- Primary Child: `D15.BOD.SEVERE_INJURY` — 重傷・障害
- Primary Parent: `D15.BOD` — 身体・健康
- Secondary 1 Child: `D15.BOD.DEATH` — 死亡
- Secondary 1 Parent: `D15.BOD` — 身体・健康
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

### 2.3.6. D16: 因果時間構造

**発動条件・原因成立から主作用／主帰結までの関係は、時間上どのように編成されているか**

- Primary Child: `D16.PRG.STAGED_PROGRESSION` — 段階進行
- Primary Parent: `D16.PRG` — 進行・長期
- Secondary: なし
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

### 2.3.7. D17: 回避・制御方式

**結果をどう回避・制御・利用できるか**

- Primary Child: `D17.AVO.DO_NOT_ENGAGE` — 接触回避
- Primary Parent: `D17.AVO` — 回避・逃走
- Secondary: なし
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

### 2.3.8. D18: 作用レイヤー

**因果効力は伝承内／受容者／社会現実のどこに及ぶか**

- `D18.L1` — 伝承内因果層: `1`
- `D18.L2` — 受容者層: `0`
- `D18.L3` — 社会現実層: `0`
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

## 2.4. 社会的埋め込みの次元

### 2.4.1. D19: 流通範囲

**誰の間に伝承が流通するか**

- Primary Child: `D19.MAS.NATIONAL_PUBLIC` — 全国的大衆
- Primary Parent: `D19.MAS` — 大衆・広域社会
- Secondary: なし
- Status: `D` — Direct（典拠に直接明示）

`0180` と同様、`Pilot Coding` の既存判定を文書化した。直接証拠の確認先は `*_00_contents.md` の典拠・証拠対応を参照する。

### 2.4.2. D20: 特権情報保持者

**誰が真相・追加情報を持つか**

- Primary Child: `D20.ORG.PERPETRATOR_CRIMINAL` — 加害者・犯罪者
- Primary Parent: `D20.ORG` — 組織・加害主体
- Secondary: なし
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

### 2.4.3. D21: 現実アンカー

**実在世界へどの程度固定されるか**

- Value: `D21.A1` — 一般的現実背景
- Status: `I` — Inferred（定義と証拠から合理的に推定）

`0180` と同様、典拠に明示された事実だけでなく、コード定義と記録内容から合理的に推定したパイロット判定を保持する。

# 3. パイロット時点のコーディングメモ

- `D03:U; D16:U`

このメモはパイロットExcelに記録されたレビュー用メモをそのまま保持する。
