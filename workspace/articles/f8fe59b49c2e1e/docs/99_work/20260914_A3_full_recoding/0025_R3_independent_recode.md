# 0025 不幸の手紙 — R3 Independent Recode

- Entry_ID: `0025`
- Version Scope正本: `docs/99_work/20260914_A3_full_recoding/0025_R2_version_scope.md`
- Evidence正本: `docs/10_each_lore/0025_fuko_no_tegami/0025_00_contents.md`
- R1監査: `docs/99_work/20260914_A3_full_recoding/0025_R1_evidence_audit.md`
- 旧 `0025_10_analysis.md`: **未参照**
- 旧Excel coding値: **未参照**

## 1. 独立判定

### D01 生成年代
- Status: `U`
- 根拠: 1970年前後は制裁前景型の早期流行・確認層として研究上追跡できるが、絶対的な生成時期そのものとは確定できない。1920年代の幸福型との連続性もあるため、最古確認点を生成年代へ変換しない。

### D02 最古確認流通媒体
- Primary: `D02.PRT.CHAIN_LETTER`
- Parent: `D02.PRT`
- Status: `I`
- 根拠: 1970年初期層は研究資料を介して郵便チェーンレターとして確認される。原新聞・原手紙未実見のためI。

### D03 確認流通媒体ポートフォリオ
- Primary: `D03.PRT.CHAIN_LETTER`
- Parent: `D03.PRT`
- Status: `I`
- 根拠: 1970〜1977年Scopeの成立に、受信者が同文を複写して郵送するチェーンレター媒体が構造的に不可欠。

### D04 生成・変容パターン
- Primary: `D04.VAR.RETELLING`
- Parent: `D04.VAR`
- Status: `I`
- 根拠: 1970年型の「50時間・29人」と1977年赤池型の「数日・20人」のように、制裁前景の因果核を保ったまま期限・人数等の細部が変化している。口承変異とはせず、文面の再話・再複製による変異として扱う。

### D05 提示形式
- Primary: `D05.RUL.CHAIN_INSTRUCTION`
- Parent: `D05.RUL`
- Secondary: `D05.RUL.JINX_RULE`
- Status: `I`
- 根拠: 同文を規定人数へ期限内に転送する命令が中心であり、不転送と不幸を結ぶジンクス規則が独立して不可欠。

### D06 真実性提示
- Value: `D06.T5` — 条件付き信念
- Status: `I`
- 根拠: 「期限内に送らなければ不幸になる」という条件付き因果を受容者に信じさせる提示形式。

### D07 意味形成対象
- Primary: `D07.DCF.FATE_OMEN`
- Parent: `D07.DCF`
- Status: `I`
- 根拠: 将来の不幸・災厄を予測し回避可能にすることが中心問題である。

### D08 意味形成契機
- Primary: `D08.CLM.UNSUPPORTED_ASSERTION`
- Parent: `D08.CLM`
- Status: `I`
- 根拠: 「送らなければ不幸になる」という根拠未提示の因果主張が文面上で先行する。

### D09 意味付与操作
- Primary: `D09.CTL.TRANSMISSION_RULE`
- Parent: `D09.CTL`
- Secondary: `D09.PPR.CORRELATION_RULE`
- Status: `I`
- 根拠: 不確実な災厄を転送規則へ変換し、「不転送 ↔ 不幸」の対応を規則化する。

### D10 因果源存在論
- Primary: `D10.OBJ.INFORMATION_CONTENT`
- Parent: `D10.OBJ`
- Status: `I`
- 根拠: 特定の霊・神・人物を必須とせず、自己複製される文面とその規則内容が因果条件を保持する。

### D11 発動・接触条件
- Primary: `D11.INF.RECEIVE_MESSAGE`
- Parent: `D11.INF`
- Status: `I`
- 根拠: 制裁規則を含む手紙を受け取った時点で受信者が期限・転送条件の対象となる。

### D12 作用対象
- Primary: `D12.AUD.READER_LISTENER`
- Parent: `D12.AUD`
- Status: `I`
- 根拠: D13主作用である不運付与が直接向けられるのは、文面を受信し転送義務を課された現受信者。転送を実行しない場合も作用対象であるため、PRACTITIONERより受信者役割を優先する。

### D13 作用機構
- Primary: `D13.FAT.CURSE_MISFORTUNE`
- Parent: `D13.FAT`
- Status: `I`
- 根拠: 不転送時に本人へ不幸・災難を与えることが制裁前景型の主作用。

### D14 帰結極性
- Value: `D14.NEG`
- Status: `I`
- 根拠: 主帰結は不転送時の不幸・災難であり、正の報酬はScopeの中心ではない。

### D15 帰結領域
- Primary: `D15.OPP.LUCK_MISFORTUNE`
- Parent: `D15.OPP`
- Status: `I`
- 根拠: 最終的に変わるとされるのは受信者の一般的な不運・災厄。

### D16 因果時間構造
- Primary: `D16.DLY.DEADLINE`
- Parent: `D16.DLY`
- Secondary: `D16.TRN.CHAIN_SPREAD`
- Status: `I`
- 根拠: 「50時間以内」「数日以内」等の期限が制裁回避条件として重要であり、転送すればA→B→Cと次の受信者へ同じ規則が連鎖する。

### D17 回避・制御方式
- Primary: `D17.CST.TRANSFER_SUBSTITUTE`
- Parent: `D17.CST`
- Secondary: `D17.RUL.TIMING_ORDER`
- Status: `I`
- 根拠: 伝承内部では、同じ脅威を次の受信者へ転送することで自分の災厄を回避する構造が中心。期限遵守が独立した制御条件となる。

### D18 作用レイヤー
- `D18.L1=1`
- `D18.L2=1`
- `D18.L3=1`
- Status: `D`
- 根拠:
  - L1: 文面内部で不転送→不幸という因果が成立。
  - L2: 現実側の受信者自身が手紙を受信することで伝承内容上の因果対象になる。
  - L3: 1977年自治体広報が、実際の受信者不安と破棄・派出所届出という行政対応を直接記録するX Evidence。

### D19 流通範囲
- Primary: `D19.MAS.NATIONAL_PUBLIC`
- Parent: `D19.MAS`
- Status: `I`
- 根拠: 研究史は1970年前後の日本で独立流行した型として扱い、1977年には別地域の自治体資料でも具体的流通を確認できる。単一地域限定ではなく広域社会へ流通した型と推定できるが、認知率等を直接測定していないためI。

### D20 特権情報保持者
- Primary: `D20.NON.NO_HIDDEN_TRUTH`
- Parent: `D20.NON`
- Status: `I`
- 根拠: 文面自体が条件と制裁を開示し、秘密の解除法や真相を専門家・内部者だけが保持する構造を必要としない。

### D21 現実アンカー
- Value: `D21.A1`
- Status: `I`
- 根拠: 郵便・地域社会という一般的な現実背景を用いるが、特定地点・人物・制度規則が伝承内因果の成立条件ではない。

## 2. causal precheck

```text
D12: 制裁規則を受信した現受信者
→ D13: 不転送時に不幸・災難を付与
→ D15: 本人の運勢が不運側へ変化
```

接続は成立する。

## 3. taxonomy gap precheck

現行Scopeは既存taxonomyで表現可能。確定的な新Child要求なし。

## 4. R3 freeze

本ファイルは旧 `0025_10_analysis.md` および旧Excel coding値を参照せずに作成した独立判定である。以後のR4で初めて旧10を参照する。