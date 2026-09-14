記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0024_00_contents.md`、R2/R3監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `24`
- `伝承エントリ名称`: 幸福の手紙
- `Macro_Category`: 情報・メタ伝説
- `Entry_Type`: 規則・チェーン伝承
- `Version_Scope`: 1922年の日本流行層で研究資料から具体的に確認できる、郵便葉書・手紙による自己複製型の「幸福の手紙」。受信者が一定期限内に同一文面を規定枚数・人数へ複写して送れば本人に幸運が訪れ、連鎖を止めれば悪運・災難が訪れ得る、とする報酬前景型のチェーンレター。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Value: なし
- Status: `U`
- 根拠: 1922年は日本で追跡できる古い流行層であり、すでに海外系統から移入された可能性がある。最古確認点を生成時期へ変換しない。

### D02 最古確認流通媒体
- Primary: `D02.PRT.CHAIN_LETTER`
- Parent: `D02.PRT`
- Status: `I`
- 根拠: 丸山2012が1922年の葉書チェーンを同時代新聞資料から固定するが、本作業では原葉書／新聞原紙を直接閲覧していない。

### D03 確認流通媒体ポートフォリオ
- Primary: `D03.PRT.CHAIN_LETTER`
- Parent: `D03.PRT`
- Status: `I`
- 根拠: Scoped Versionの成立に、複写・郵送されるチェーン葉書／手紙が構造的に不可欠。1951年の直接資料は同型の継続を補強するが、1922年Scopeの証拠強度は研究資料経由なのでI。

### D04 生成・変容パターン
- Status: `U`
- 根拠: 1922年の狭いScope内で安定した時間的変容を確定できない。後代の電子化や「不幸の手紙」化を混ぜない。

### D05 提示形式
- Primary: `D05.RUL.CHAIN_INSTRUCTION`
- Parent: `D05.RUL`
- Secondary: `D05.RUL.JINX_RULE`
- Status: `I`
- 根拠: 同文の規定数転送を直接要求し、その行為を幸運／悪運へ結ぶ。

### D06 真実性提示
- Value: `D06.T5`
- Status: `I`
- 根拠: 「この条件を満たせば幸運、破れば悪運」という条件付き信念として提示される。

## 2.2. 意味形成

### D07 意味形成対象
- Primary: `D07.DCF.FATE_OMEN`
- Parent: `D07.DCF`
- Status: `I`
- 根拠: 不確実な将来の幸運／悪運を予測・制御可能なものとして扱う。

### D08 意味形成契機
- Primary: `D08.CLM.UNSUPPORTED_ASSERTION`
- Parent: `D08.CLM`
- Status: `I`
- 根拠: 「転送すれば幸運、断てば悪運」という因果主張が文面上で先行提示される。文書という媒体そのものではなく、その根拠未提示の命題が意味形成を開始させる。

### D09 意味付与操作
- Primary: `D09.CTL.TRANSMISSION_RULE`
- Parent: `D09.CTL`
- Secondary: `D09.PPR.CORRELATION_RULE`
- Status: `I`
- 根拠: 将来の吉凶を転送規則へ変換し、転送／断絶と幸運／悪運を対応づける。

### D10 因果源存在論
- Primary: `D10.OBJ.INFORMATION_CONTENT`
- Parent: `D10.OBJ`
- Status: `I`
- 根拠: 人格主体を必要とせず、複製される文面とその規則内容が因果条件を保持して移動する。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary: `D11.INF.RECEIVE_MESSAGE`
- Parent: `D11.INF`
- Status: `I`
- 根拠: 手紙を受信することで期限・転送条件と吉凶規則の対象になる。1951年資料も「受取ってから」の期限を例示する。

### D12 作用対象
- Primary: `D12.FOC.PRACTITIONER`
- Parent: `D12.FOC`
- Status: `I`
- 根拠: D13の幸運／悪運付与が直接向けられるのは、転送規則を実行または不実行する受信者本人。

### D13 作用機構
- Primary: `D13.FAT.LUCK_BENEFIT`
- Parent: `D13.FAT`
- Secondary: `D13.FAT.CURSE_MISFORTUNE`
- Status: `I`
- 根拠: 報酬前景型として転送時の幸運付与がPrimary。同じ初期文面に連鎖断絶時の悪運付与が含まれるためSecondary。

### D14 帰結極性
- Value: `D14.MIX`
- Status: `I`
- 根拠: 同一Scoped規則に正の報酬と負の制裁が重要な分岐として併存する。

### D15 帰結領域
- Primary: `D15.OPP.LUCK_MISFORTUNE`
- Parent: `D15.OPP`
- Status: `I`
- 根拠: 最終的に変わるとされるのは本人の一般的な運勢・吉凶。

### D16 因果時間構造
- Primary: `D16.DLY.DEADLINE`
- Parent: `D16.DLY`
- Secondary: `D16.TRN.CHAIN_SPREAD`
- Status: `I`
- 根拠: 明示的な期限が結果条件として重要であり、同時にA→B→Cと受信者間を移る連鎖拡散が成立する。

### D17 回避・制御方式
- Primary: `D17.USE.LUCK_EXPLOITATION`
- Parent: `D17.USE`
- Secondary: `D17.RUL.TIMING_ORDER`
- Status: `I`
- 根拠: 幸運を得るために規則を意図的に利用し、指定期限を守ることが伝承内部の制御法となる。

### D18 作用レイヤー
- `D18.L1=1`
- `D18.L2=1`
- `D18.L3=1`
- Status: `D`
- 根拠: L1は文面内部の転送→吉凶因果、L2は現実側受信者自身が文面上の因果対象になる自己適用、L3は1951年国会会議録が実際の転送行動と制度的検討を直接記録するX Evidence。

## 2.4. 社会的埋め込み

### D19 流通範囲
- Primary: `D19.LOC.REGIONAL`
- Parent: `D19.LOC`
- Status: `I`
- 根拠: Scopeの最古層は1922年東京の新聞資料を中心に追跡される。後代の全国認知を初期Scopeへ遡及しない。

### D20 特権情報保持者
- Primary: `D20.NON.NO_HIDDEN_TRUTH`
- Parent: `D20.NON`
- Status: `I`
- 根拠: 実行条件・報酬・制裁は文面自体に開示され、専門家・内部者だけが持つ追加の真相や解除情報を必要としない。

### D21 現実アンカー
- Value: `D21.A1`
- Status: `I`
- 根拠: 葉書・郵便という一般的現実通信環境を用いるが、特定の制度規則・人物・事件を伝承内因果へ組み込まない。

# 3. R4 Entry QA

## 3.1. D12 → D13 → D15 causal QA

```text
D12: 転送規則を実行／不実行する受信者本人
→ D13: 転送時は幸運、断絶時は悪運を付与
→ D15: 本人の幸運／不運が変化
```

**結果: Pass。**

## 3.2. D18 L3 evidence QA

1951年3月26日の参議院郵政委員会会議録は、「幸運の手紙」を具体的転送行動を伴う郵便現象として扱い、郵便法・通信上の対応を審議している。

**結果: `L3=1`。Pass。**

## 3.3. U / NA / C QA

- `D01=U`: 1922年確認層を絶対生成時期へ変換しないため妥当。
- `D04=U`: Scoped Version内の変容史を示すH Evidence不足。
- `NA`: なし。
- `C`: なし。

**結果: Pass。**

## 3.4. taxonomy gap QA

現行Scopeは既存taxonomyで表現可能。確定的な新Child要求なし。

**結果: Pass。baseline変更なし。**

## 3.5. R3 → R4 QA差分

| 次元 | R3 freeze | R4確定 | 理由 |
|---|---|---|---|
| D01 | `D01.G1 / I` | `U` | 1922年は最古確認流行層であり生成時期そのものではない |
| D08 | `TEXT_DOCUMENT` + `UNSUPPORTED_ASSERTION / I` | `UNSUPPORTED_ASSERTION / I` | 媒体そのものではなく先行因果主張が意味形成契機 |
| D17 | `STRATEGIC_RULE_USE` + `TIMING_ORDER / I` | `LUCK_EXPLOITATION` + `TIMING_ORDER / I` | 幸運利用を直接表すより具体的Childを優先 |

その他はR3 freezeを維持した。

# 4. 再コーディング前旧10との差分比較

比較対象: `再コーディング前 commit SHA = fc120f41ab98bc1f150f624dcafaaa016e2c6d39` の旧 `0024_10_analysis.md`。R3 freeze後にのみ参照した。

| 次元 | 旧判定 | R4確定 | 差分分類 | 要点 |
|---|---|---|---|---|
| D01 | `U` | `U` | 一致 | 最古確認と生成時期を分離 |
| D02 | `CHAIN_LETTER / I` | 同左 | 一致 | — |
| D03 | `CHAIN_LETTER / D` | `CHAIN_LETTER / I` | `Status mismatch` | 1922年Scopeの直接原紙未実見 |
| D04 | `FORMAT_TRANSLATION + ACCRETION / I` | `U` | `Scope mismatch` / `Evidence mismatch` | 後代電子化・増補を1922年Scopeへ混ぜない |
| D05 | `CHAIN_INSTRUCTION / D` | `CHAIN_INSTRUCTION + JINX_RULE / I` | `Scope mismatch` / `Code-selection mismatch` / `Status mismatch` | 初期資料の報酬＋制裁をScope内へ固定 |
| D06 | `T6 / I` | `T5 / I` | `Code-selection mismatch` | 真偽の謎より条件付き信念として提示 |
| D07 | `FATE_OMEN / I` | 同左 | 一致 | — |
| D08 | `UNSUPPORTED_ASSERTION / I` | 同左 | 一致 | R4でR3を修正 |
| D09 | `TRANSMISSION_RULE + CORRELATION_RULE / D` | 同code / `I` | `Status mismatch` | 1922年Scopeは研究資料経由 |
| D10 | `INFORMATION_CONTENT / I` | 同左 | 一致 | — |
| D11 | `READ_VIEW_MEDIA / D` | `RECEIVE_MESSAGE / I` | `Code-selection mismatch` / `Status mismatch` | 期限は受信時点から開始する構造 |
| D12 | `READER_LISTENER + NEXT_RECIPIENT / D` | `PRACTITIONER / I` | `Code-selection mismatch` / `Prior coding error` | D12はD13主作用の直接対象を採る |
| D13 | `LUCK_BENEFIT + MEDIA_OBJECT_TRANSFER / D` | `LUCK_BENEFIT + CURSE_MISFORTUNE / I` | `Scope mismatch` / `Code-selection mismatch` / `Status mismatch` | 伝播は時間・流通構造へ、初期制裁作用をSecondaryへ |
| D14 | `POS / I` | `MIX / I` | `Scope mismatch` / `Code-selection mismatch` | 初期文面内に幸運と悪運が併存 |
| D15 | `LUCK_MISFORTUNE / I` | 同左 | 一致 | — |
| D16 | `CHAIN_SPREAD / D` | `DEADLINE + CHAIN_SPREAD / I` | `Code-selection mismatch` / `Status mismatch` | 明示期限をPrimary、連鎖をSecondaryに保持 |
| D17 | `LUCK_EXPLOITATION + PROCEDURAL_RULE / D` | `LUCK_EXPLOITATION + TIMING_ORDER / I` | `Code-selection mismatch` / `Status mismatch` | 制御の具体条件は期限遵守 |
| D18 | `L1=1,L2=1,L3=1 / I` | 同bit / `D` | `Status mismatch` | L3を国会会議録の直接X Evidenceで固定 |
| D19 | `NATIONAL_PUBLIC / D` | `REGIONAL / I` | `Scope mismatch` / `Evidence mismatch` / `Code-selection mismatch` / `Status mismatch` | 1922年初期Scopeに限定 |
| D20 | `COMMON_KNOWLEDGE / I` | `NO_HIDDEN_TRUTH / I` | `Code-selection mismatch` | 共有範囲ではなく特権的追加情報の有無を問う |
| D21 | `A0 / I` | `A1 / I` | `Code-selection mismatch` | 一般的な葉書・郵便環境へ埋め込まれる |

# 5. R4結論

- D12→D13→D15 causal QA: Pass
- D18 L3 evidence QA: Pass
- U / NA / C QA: Pass
- taxonomy gap QA: Pass、baseline変更候補なし
- R3→R4修正: D01 / D08 / D17
- 旧10との差分比較・分類: 完了
- Coding正本 `0024_10_analysis.md`: 更新済み
- 旧Excel比較: R5 Global Reconciliationへ移管