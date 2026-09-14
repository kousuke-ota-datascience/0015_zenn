記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

伝承内容・典拠・異伝・証拠上の不確実性は、同一ディレクトリの `0006_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `6`
- `伝承エントリ名称`: メリーさんの電話
- `Macro_Category`: 全国型怪異・儀式
- `Entry_Type`: 物語・伝説
- `Version_Scope`: 1999年までに公刊確認できる「人形の怪異メリーさんから反復して電話がかかる」という核を基礎とし、同じ意味形成核を維持した後代の安定代表形から「電話で現在地を反復通知しながら受信者へ段階的に接近する」構造までを含む。襲撃・死亡、リカちゃん電話直接起源説、携帯電話化それ自体はScope外とする。
- `Version_Scope_Status`: `I`
- `再コーディング前 commit SHA`: `89b2865013ecd7fca4b39fa914973a0f55eb3b3b`
- `R3 後 commit SHA`: `92bbf26600e52525c2c11398c55d32082f0b3d0e`

# 2. 分析概念次元

各次元はコード値と、その判定が当該伝承で具体的に何を意味するかを記録する。

## 2.1. 来歴・流通・提示

### D01: 生成年代

- Primary Child / Value: なし
- Status: `U`

**判定根拠:** 1999年2月の公刊固定点は直接確認できるが、これは生成・成立年そのものではない。1980年代成立等を後代資料から遡及しない。

### D02: 最古確認流通媒体

- Primary Child: `D02.PRT.BOOK` — 書籍
- Primary Parent: `D02.PRT`
- Status: `D`

**判定根拠:** 現在のEvidenceで実際の流通媒体として直接固定できる最古点は、1999年2月刊の斉藤洋『メリーさんの電話』である。これは起源媒体の断定ではなく、「現在確認できる最古媒体」をコードするD02の規則に従う。

### D03: 確認流通媒体ポートフォリオ

- Primary Child: `D03.PRT.BOOK` — 書籍
- Primary Parent: `D03.PRT`
- Secondary: なし
- Status: `D`

**判定根拠:** Version Scopeに含める内容について、受容媒体として直接固定できるのは書籍である。後代研究が学校怪談文脈へ位置付けることだけを理由に、学校口承を確認媒体として追加しない。

### D04: 生成・変容パターン

- Primary Child: なし
- Status: `U`

**判定根拠:** R3では1999年に直接確認できる「人形＋反復電話」と後代の段階接近型との差を `D04.VAR.ACCRETION` とした。しかし1999年資料は本文全体ではなく短い書誌・紹介情報であり、段階接近が要約に出ていないことは「当時存在しなかった」証拠にならない。現行規則は変化を示すHistory EvidenceがなければD04を推測で埋めないため、R4 QAで `U` に修正した。

### D05: 提示形式

- Primary Child: `D05.HRS.SCHOOL_WORK_HEARSAY` — 学校・職場伝承
- Primary Parent: `D05.HRS`
- Status: `I`

**判定根拠:** 後代研究は1990年代の学校怪談・人形怪談の文脈へ位置付ける。受容者へは「こういう怪談がある」という伝聞として提示される構造が中心であり、儀式手順ではない。

### D06: 真実性提示

- Value: `D06.T6` — 真偽未確定
- Status: `I`

**判定根拠:** 特定事件・公的記録に固定せず、怪談・噂として真偽を開いたまま流通する。

## 2.2. 意味形成

### D07: 意味形成対象

- Primary Child: `D07.ANO.UNKNOWN_EXISTENCE` — 未知存在
- Primary Parent: `D07.ANO`
- Status: `I`

**判定根拠:** 通常の所有物である人形が自ら名乗り、通信・移動・追跡を行う主体へ変わることが中心的な不可解さである。

### D08: 意味形成契機

- Primary Child: `D08.DEX.DIRECT_EVENT` — 出来事への直接遭遇
- Primary Parent: `D08.DEX`
- Status: `I`

**判定根拠:** 物語内部の受信者にとって、問題化の契機は「怪談を聞いたこと」ではなく、人形を名乗る異常な着信と位置通知を直接受けることである。D11の発動条件とは分離する。

### D09: 意味付与操作

- Primary Child: `D09.AGN.AGENCY_ATTRIBUTION` — 主体化
- Primary Parent: `D09.AGN`
- Secondary: `D09.PPR.RECURRENCE_RULE` — 反復規則
- Status: `I`

**判定根拠:** 人形へ意思ある主体性を与え、さらに着信の反復ごとに距離が縮む規則として不可解な現象を構造化する。

### D10: 因果源存在論

- Primary Child: `D10.SUP.YOKAI_ENTITY` — 妖怪・人格怪異
- Primary Parent: `D10.SUP`
- Secondary: `D10.OBJ.CURSED_OBJECT` — 呪物
- Status: `I`

**判定根拠:** メリーさんは名乗り、位置を告げ、対象へ接近する意思ある人格怪異として機能する。同時に人形という物体基体に結び付くため呪物性をSecondaryとする。

## 2.3. 因果・行動モデル

### D11: 発動・接触条件

- Primary Child: `D11.PAS.SPONTANEOUS_SELECTION` — 偶然選ばれる・遭遇する
- Primary Parent: `D11.PAS`
- Status: `I`

**判定根拠:** Scopeでは「人形を捨てれば必ず発動する」という再現的規則をEvidenceから固定できない。最初の電話受信は既に怪異作用の一部であり、それ自体を発動条件へ重ねない。したがって、本人の明確な再現操作なしに追跡対象にされる受動型を採る。

### D12: 作用対象

- Primary Child: `D12.OTH.VICTIM_TARGET` — 特定被害者
- Primary Parent: `D12.OTH`
- Status: `I`

**判定根拠:** D13の主作用である追跡・接近が直接向けられるのは電話の受信者である。物語上の主人公であることではなく、主作用の直接対象であることを優先する。

### D13: 作用機構

- Primary Child: `D13.REL.PURSUIT` — 追跡
- Primary Parent: `D13.REL`
- Status: `I`

**判定根拠:** 現在地を反復通知しながら対象との距離を縮めることが主作用である。Scope外の身体攻撃を持ち込まない。

### D14: 帰結極性

- Value: `D14.NEG` — 負
- Status: `I`

**判定根拠:** 安全圏への侵入、追跡、恐怖が主帰結であり、利益を与える構造ではない。

### D15: 帰結領域

- Primary Child: `D15.MND.FEAR_TRAUMA` — 恐怖・トラウマ
- Primary Parent: `D15.MND`
- Secondary: `D15.KNW.UNCERTAINTY_PRESERVED` — 不確実性維持
- Status: `I`

**判定根拠:** Scopeは身体的殺害を必須とせず、至近距離到達による恐怖を安定終端とする。また到達後の生死・真相を確定しない型があるため、終端の未確定性をSecondaryとして保持する。

### D16: 因果時間構造

- Primary Child: `D16.EVT.SEQUENTIAL_EPISODE` — エピソード内段階進行
- Primary Parent: `D16.EVT`
- Status: `I`

**判定根拠:** 反復着信→位置更新→距離短縮→安全圏侵入が、一つの追跡エピソード内で順に進む。単に追跡が長く続くことより、段階列そのものが構造上重要である。

### D17: 回避・制御方式

- Primary Child: なし
- Status: `U`

**判定根拠:** 電話を切る、無視する、逃げる、戸締まりする等は試行されるが、Scope共通の有効な停止規則を固定できない。資料不足を `NO_KNOWN_ESCAPE` や `FIXED_OUTCOME` へ読み替えない。

### D18: 作用レイヤー

- `D18.L1=1`
- `D18.L2=0`
- `D18.L3=0`
- Status: `D`

**判定根拠:** 伝承内部では追跡・接近の因果が明示される。一方、怪談を聞いた現実側受容者自身がメリーさんの対象になる自己適用規則はなく、社会現実で確認された制度・市場・集団行動等のX Evidenceもない。

## 2.4. 社会的埋め込み

### D19: 流通範囲

- Primary Child: `D19.KIN.SCHOOL_YOUTH` — 学校・若者集団
- Primary Parent: `D19.KIN`
- Status: `I`

**判定根拠:** 後代研究が1990年代の学校怪談・若者文化の文脈に位置付ける。今回の固定Evidenceだけから全国一般大衆への認知範囲を推定しない。

### D20: 特権情報保持者

- Primary Child: `D20.NON.NO_HIDDEN_TRUTH` — 隠れた真相なし
- Primary Parent: `D20.NON`
- Status: `I`

**判定根拠:** 物語は「誰か専門家だけが真相や解除法を知る」という情報構造を必要としない。怪談そのものが共有されることと、特権情報保持者の有無を分ける。

### D21: 現実アンカー

- Value: `D21.A1` — 一般的現実背景
- Status: `I`

**判定根拠:** 人形、電話、自宅、駅・道路等の一般的な日常環境を使うが、特定地点・人物・制度・史実には固定されない。

# 3. R4 Entry QA

## 3.1. D12 → D13 → D15 causal QA

```text
D12: 電話の受信者・追跡対象
→ D13: メリーさんが反復着信と位置通知を伴って対象へ接近・追跡する
→ D15: 対象に恐怖が生じ、到達後の最終状態は一部型で未確定のまま残る
```

**結果:** Pass。

身体的襲撃・死亡はR2 Scope外であり、因果列へ追加しない。

## 3.2. D18 L3 evidence QA

R1で確認した資料に、噂の流通が現実社会の制度・市場・集団行動を変えたことを示す `X` Evidenceはない。

**結果:** `L3=0` を維持。Pass。

## 3.3. U / NA / C QA

- `D01=U`: 公刊固定点を成立年代へ変換しないため妥当。
- `D04=U`: 変容史を示すHistory Evidence不足。R3の推定をR4で撤回。
- `D17=U`: 共通して有効な回避法を確定できないため妥当。
- `NA`: なし。
- `C`: なし。

**結果:** Pass。

## 3.4. taxonomy gap QA

現行Version Scopeは既存taxonomyで表現可能であり、確定的な新Child要求はない。

ただし、「人形を捨てる／放棄する」ことが将来より強い一次Evidenceで再現的な発動条件として固定された場合、D11には「捨てる・放棄する」を直接表すChildがない。この点は**監視項目**として残すが、現Evidenceではtaxonomy gap候補へ昇格させない。

**結果:** Pass。baseline変更なし。

## 3.5. R3 → R4 QA差分

| 次元 | R3 freeze | R4確定 | 理由 |
|---|---|---|---|
| D04 | `D04.VAR.ACCRETION / I` | `U` | 1999年の短い書誌要約に段階接近がないことは、後代増補のHistory Evidenceにならないため |

その他のD01–D21はR3 freezeを維持した。

# 4. 再コーディング前旧10との差分比較

比較対象は `再コーディング前 commit SHA` = `89b2865013ecd7fca4b39fa914973a0f55eb3b3b` の `0006_10_analysis.md`。R3 freeze後にのみ参照した。

| 次元 | 旧判定 | R4確定判定 | 差分分類 | 要点 |
|---|---|---|---|---|
| D01 | `U` | `U` | 一致 | 成立年代は引き続き不明 |
| D02 | `U` | `D02.PRT.BOOK / D` | `Evidence mismatch` / `Status mismatch` | R1で1999年児童書を直接固定したため |
| D03 | `D03.ORL.SCHOOL_ORAL / I` | `D03.PRT.BOOK / D` | `Evidence mismatch` / `Code-selection mismatch` / `Status mismatch` | 学校怪談という後代位置付けより、直接確認できる流通媒体を優先 |
| D04 | `D04.STB.STABILIZED_CANON` + `D04.MIG.MEDIUM_SHIFT / I` | `U` | `Evidence mismatch` / `Status mismatch` / `Prior coding error` | 変容史の直接証拠が不足しており、安定化・媒体移行を推測で固定しない |
| D05 | `D05.HRS.SCHOOL_WORK_HEARSAY / I` | 同左 | 一致 | — |
| D06 | `D06.T6 / I` | 同左 | 一致 | — |
| D07 | `D07.ANO.UNKNOWN_EXISTENCE / I` | 同左 | 一致 | — |
| D08 | `D08.CLM.UNSUPPORTED_ASSERTION / I` | `D08.DEX.DIRECT_EVENT / I` | `Code-selection mismatch` | 受容者一般の怪談命題ではなく、物語内部の異常着信を意味形成契機として採る |
| D09 | `D09.AGN.AGENCY_ATTRIBUTION / I` | 同Primary + `D09.PPR.RECURRENCE_RULE` secondary | `Code-selection mismatch` | 段階接近の反復規則を独立追加構造として保持 |
| D10 | `YOKAI_ENTITY` + `CURSED_OBJECT / I` | 同左 | 一致 | — |
| D11 | `D11.INF.RECEIVE_MESSAGE / I` | `D11.PAS.SPONTANEOUS_SELECTION / I` | `Code-selection mismatch` | 電話受信を主作用の一部とみなし、再現的発動条件が固定できないため受動対象化を採る |
| D12 | `D12.FOC.PROTAGONIST_EXPERIENCER / I` | `D12.OTH.VICTIM_TARGET / I` | `Code-selection mismatch` / `Prior coding error` | 現行規則では主人公性でなくD13主作用の直接対象を採る |
| D13 | `D13.REL.PURSUIT / I` | 同左 | 一致 | — |
| D14 | `D14.NEG / I` | 同左 | 一致 | — |
| D15 | `D15.MND.FEAR_TRAUMA / I` | 同Primary + `D15.KNW.UNCERTAINTY_PRESERVED` secondary | `Code-selection mismatch` | 到達後を確定しない型を終端構造として追加保持 |
| D16 | `D16.CON.PURSUIT_DURATION / I` | `D16.EVT.SEQUENTIAL_EPISODE / I` | `Code-selection mismatch` | 継続時間より、反復着信による段階列が時間構造の中心 |
| D17 | `U` | `U` | 一致 | — |
| D18 | `L1=1,L2=0,L3=0 / I` | 同bit / `D` | `Status mismatch` | Scope内作用レイヤーを直接構造として確定 |
| D19 | `D19.MAS.NATIONAL_PUBLIC / I` | `D19.KIN.SCHOOL_YOUTH / I` | `Evidence mismatch` / `Code-selection mismatch` | 全国認知を推測せず、固定Evidenceが支える学校・若者範囲へ狭める |
| D20 | `D20.NON.COMMON_KNOWLEDGE / I` | `D20.NON.NO_HIDDEN_TRUTH / I` | `Code-selection mismatch` | 共有範囲ではなく「特権的な追加真相保持者が必要か」という中心質問へ合わせる |
| D21 | `D21.A1 / I` | 同左 | 一致 | — |

# 5. R4結論

- D12→D13→D15 causal QA: Pass
- D18 L3 evidence QA: Pass
- U / NA / C QA: Pass
- taxonomy gap QA: Pass。baseline変更候補なし
- R3→R4修正: D04のみ `ACCRETION / I` → `U`
- Coding正本 `0006_10_analysis.md`: 更新済み
- 旧 `0006_10_analysis.md` 差分比較・分類: 完了
- 旧Excel比較: R5 Global Reconciliationへ移管
