記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は `0025_00_contents.md`、R2/R3監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `25`
- `伝承エントリ名称`: 不幸の手紙
- `Macro_Category`: 情報・メタ伝説
- `Entry_Type`: 規則・チェーン伝承
- `Version_Scope`: 1970年秋〜1977年に確認できる制裁前景型の郵便チェーンレターの最小安定共有核。受信者が指定期限内に同一文面を指定人数へ転送しなければ本人に不幸・災難が訪れるとされ、転送がその災厄の回避手段として要求される型。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Value: なし
- Status: `U`
- 根拠: 1970年前後は制裁前景型の早期流行・確認層だが、1920年代の幸福型に負の制裁の前史があるため、独立型の生成時期を確定しない。

### D02 最古確認流通媒体
- Primary: `D02.PRT.CHAIN_LETTER`
- Parent: `D02.PRT`
- Status: `I`
- 根拠: 1970年初期層は研究資料を介して紙のチェーンレターとして確認される。原新聞・原手紙未実見のためI。

### D03 確認流通媒体ポートフォリオ
- Primary: `D03.PRT.CHAIN_LETTER`
- Parent: `D03.PRT`
- Status: `I`
- 根拠: 1970〜1977年Scopeの成立に、同文を複写・郵送するチェーンレター媒体が構造的に不可欠。1977年は直接確認できるが、Scope全体の早期層を含むためIとする。

### D04 生成・変容パターン
- Primary: `D04.VAR.RETELLING`
- Parent: `D04.VAR`
- Status: `I`
- 根拠: 1970年型の「50時間・29人」と1977年赤池型の「数日・20人」のように、制裁前景の因果核を保ったまま期限・人数等の文面細部が変化する。

### D05 提示形式
- Primary: `D05.RUL.CHAIN_INSTRUCTION`
- Parent: `D05.RUL`
- Secondary: `D05.RUL.JINX_RULE`
- Status: `I`
- 根拠: 同文転送命令が中心で、不転送と不幸を結ぶジンクス規則を伴う。

### D06 真実性提示
- Value: `D06.T5`
- Status: `I`
- 根拠: 「期限内に送らなければ不幸になる」という条件付き因果を信じるよう要求する。

## 2.2. 意味形成

### D07 意味形成対象
- Primary: `D07.DCF.FATE_OMEN`
- Parent: `D07.DCF`
- Status: `I`
- 根拠: 将来の不幸・災厄を予測し回避可能なものとして扱う。

### D08 意味形成契機
- Primary: `D08.CLM.UNSUPPORTED_ASSERTION`
- Parent: `D08.CLM`
- Status: `I`
- 根拠: 「送らなければ不幸になる」という根拠未提示の因果主張が文面上で先行する。

### D09 意味付与操作
- Primary: `D09.CTL.TRANSMISSION_RULE`
- Parent: `D09.CTL`
- Secondary: `D09.NOR.TABOOIZATION`
- Status: `I`
- 根拠: 災厄を転送規則へ変換すると同時に、「連鎖を止めること」を災厄を招く禁止行為として禁忌化する。

### D10 因果源存在論
- Primary: `D10.OBJ.INFORMATION_CONTENT`
- Parent: `D10.OBJ`
- Status: `I`
- 根拠: 特定の霊・神・人物を必須とせず、自己複製される文面とその規則内容が因果条件を保持する。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary: `D11.INF.RECEIVE_MESSAGE`
- Parent: `D11.INF`
- Status: `I`
- 根拠: 制裁規則を含む手紙を受け取った時点で、受信者が期限・転送条件の対象になる。

### D12 作用対象
- Primary: `D12.AUD.READER_LISTENER`
- Parent: `D12.AUD`
- Status: `I`
- 根拠: D13主作用である不運付与が直接向けられるのは、文面を受信し転送義務を課された現受信者。

### D13 作用機構
- Primary: `D13.FAT.CURSE_MISFORTUNE`
- Parent: `D13.FAT`
- Status: `I`
- 根拠: 不転送時に本人へ不幸・災難を与えることが制裁前景型の主作用。

### D14 帰結極性
- Value: `D14.NEG`
- Status: `I`
- 根拠: 主帰結は不転送時の不幸・災難である。

### D15 帰結領域
- Primary: `D15.OPP.LUCK_MISFORTUNE`
- Parent: `D15.OPP`
- Status: `I`
- 根拠: 具体的被害は版ごとに変わるが、共通終端は本人の一般的な不運・災厄。

### D16 因果時間構造
- Primary: `D16.DLY.DEADLINE`
- Parent: `D16.DLY`
- Secondary: `D16.TRN.CHAIN_SPREAD`
- Status: `I`
- 根拠: 明示期限が制裁回避条件として重要で、転送時にはA→B→Cと同じ規則が連鎖する。

### D17 回避・制御方式
- Primary: `D17.CST.TRANSFER_SUBSTITUTE`
- Parent: `D17.CST`
- Secondary: `D17.RUL.TIMING_ORDER`
- Status: `I`
- 根拠: 同じ脅威を次の受信者へ送ることで自分の災厄を回避し、期限遵守が独立した条件となる。

### D18 作用レイヤー
- `D18.L1=1`
- `D18.L2=1`
- `D18.L3=1`
- Status: `D`
- 根拠: L1は文面内部の不転送→不幸因果。L2は現実側受信者自身が受信により伝承内容上の因果対象になる自己適用構造。L3は1977年自治体広報が住民不安、破棄、派出所届出勧告という社会現実効果を直接記録する。

## 2.4. 社会的埋め込み

### D19 流通範囲
- Primary: `D19.MAS.NATIONAL_PUBLIC`
- Parent: `D19.MAS`
- Status: `I`
- 根拠: 研究史は1970年前後の日本で広く流行した型として扱い、1977年には自治体資料でも別途実流通を確認できる。認知率等は直接測定していないためI。

### D20 特権情報保持者
- Primary: `D20.NON.NO_HIDDEN_TRUTH`
- Parent: `D20.NON`
- Status: `I`
- 根拠: 文面自体が条件と制裁を開示し、秘密の解除法や真相を専門家・内部者だけが持つ構造を必要としない。

### D21 現実アンカー
- Value: `D21.A1`
- Status: `I`
- 根拠: 郵便・地域社会という一般的現実背景を用いるが、特定地点・人物・制度規則が伝承内因果の必須条件ではない。

# 3. R4 Entry QA

## 3.1. D12 → D13 → D15 causal QA

```text
D12: 制裁規則を受信した現受信者
→ D13: 不転送時に不幸・災難を付与
→ D15: 本人の運勢が不運側へ変化
```

**結果: Pass。**

## 3.2. D18 L3 evidence QA

1977年『広報あかいけ』は、地域で実際に「不幸の手紙」が流通していること、受信者が悩んでいること、自治体が破棄または派出所への届出を勧告したことを直接記録する。

**結果: `L3=1`。Pass。**

広報内の「自殺報道」言及は原報道未確認のためL3根拠には用いない。

## 3.3. U / NA / C QA

- `D01=U`: 1970年前後の早期流行層を絶対生成時期へ変換しないため妥当。
- `NA`: なし。
- `C`: なし。

**結果: Pass。**

## 3.4. taxonomy gap QA

現行Scopeは既存taxonomyで表現可能。確定的な新Child要求なし。

**結果: Pass。baseline変更なし。**

## 3.5. R3 → R4 QA差分

| 次元 | R3 freeze | R4確定 | 理由 |
|---|---|---|---|
| D09 Secondary | `D09.PPR.CORRELATION_RULE` | `D09.NOR.TABOOIZATION` | 制裁前景型では「連鎖を止めるな」という禁止規範が、単なる対応関係より独立した意味付与操作を直接表す |

その他のD01–D21はR3 freezeを維持した。

# 4. 再コーディング前旧10との差分比較

比較対象: `再コーディング前 commit SHA = aee0942ab1c70a42008c786670c0c0d5b6e04a1e` の旧 `0025_10_analysis.md`。R3 freeze後にのみ参照した。

| 次元 | 旧判定 | R4確定 | 差分分類 | 要点 |
|---|---|---|---|---|
| D01 | `U` | `U` | 一致 | 最古確認と生成時期を分離 |
| D02 | `CHAIN_LETTER / I` | 同左 | 一致 | — |
| D03 | `CHAIN_LETTER / D` | `CHAIN_LETTER / I` | `Status mismatch` | Scope早期層は研究資料経由 |
| D04 | `FORMAT_TRANSLATION + ACCRETION / I` | `RETELLING / I` | `Scope mismatch` / `Code-selection mismatch` | 後代電子化を除外し、1970–77年内の期限・人数差へ限定 |
| D05 | `CHAIN_INSTRUCTION / D` | `CHAIN_INSTRUCTION + JINX_RULE / I` | `Code-selection mismatch` / `Status mismatch` | 制裁規則を独立Secondaryとして保持 |
| D06 | `T6 / I` | `T5 / I` | `Code-selection mismatch` | 真偽の謎より条件付き信念 |
| D07 | `FATE_OMEN / I` | 同左 | 一致 | — |
| D08 | `UNSUPPORTED_ASSERTION / I` | 同左 | 一致 | — |
| D09 | `TRANSMISSION_RULE + TABOOIZATION / D` | 同code / `I` | `Status mismatch` | Scope全体としてIを維持 |
| D10 | `INFORMATION_CONTENT / I` | 同左 | 一致 | — |
| D11 | `READ_VIEW_MEDIA / D` | `RECEIVE_MESSAGE / I` | `Code-selection mismatch` / `Status mismatch` | 期限・義務は受信時点から開始 |
| D12 | `READER_LISTENER + NEXT_RECIPIENT / D` | `READER_LISTENER / I` | `Code-selection mismatch` / `Status mismatch` | D13主作用の現時点の直接対象だけを採る |
| D13 | `CURSE_MISFORTUNE + MEDIA_OBJECT_TRANSFER / D` | `CURSE_MISFORTUNE / I` | `Code-selection mismatch` / `Status mismatch` | 伝播はD16で表現し、主作用と分離 |
| D14 | `NEG / I` | 同左 | 一致 | — |
| D15 | `LUCK_MISFORTUNE / I` | 同左 | 一致 | — |
| D16 | `CHAIN_SPREAD / D` | `DEADLINE + CHAIN_SPREAD / I` | `Code-selection mismatch` / `Status mismatch` | 明示期限をPrimary、連鎖をSecondaryへ |
| D17 | `TRANSFER_SUBSTITUTE + PROCEDURAL_RULE / D` | `TRANSFER_SUBSTITUTE + TIMING_ORDER / I` | `Code-selection mismatch` / `Status mismatch` | 具体的制御条件は期限遵守 |
| D18 | `L1=1,L2=1,L3=1 / D` | 同左 | 一致 | — |
| D19 | `NATIONAL_PUBLIC / I` | 同左 | 一致 | — |
| D20 | `COMMON_KNOWLEDGE / I` | `NO_HIDDEN_TRUTH / I` | `Code-selection mismatch` | 共有範囲ではなく特権的追加情報の有無を問う |
| D21 | `A0 / I` | `A1 / I` | `Code-selection mismatch` | 一般的郵便・地域社会へ埋め込まれる |

# 5. R4結論

- D12→D13→D15 causal QA: Pass
- D18 L3 evidence QA: Pass
- U / NA / C QA: Pass
- taxonomy gap QA: Pass、baseline変更候補なし
- R3→R4修正: D09 Secondaryのみ
- 旧10との差分比較・分類: 完了
- Coding正本 `0025_10_analysis.md`: 更新済み
- 旧Excel比較: R5 Global Reconciliationへ移管