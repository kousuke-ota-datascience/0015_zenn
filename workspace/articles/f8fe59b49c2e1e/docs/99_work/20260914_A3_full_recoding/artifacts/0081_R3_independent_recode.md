# 0081 ピアスの白い糸 — R3 Independent Recode

- Entry_ID: `0081`
- Version Scope正本: `docs/99_work/20260914_A3_full_recoding/0081_R2_version_scope.md`
- Evidence正本: `docs/10_each_lore/0081_pierce_white_thread/0081_00_contents.md`
- R1監査補助: `docs/99_work/20260914_A3_full_recoding/0081_R1_evidence_audit.md`
- baseline: control plane記載の固定blob
- 旧 `0081_10_analysis.md`: **未参照**
- 旧Excel coding値: **未参照**

# 1. 独立判定

## D01 生成年代
- Value: なし
- Status: `U`
- 根拠: 1994年採録は直接確認できるが、それ以前の最初期口承・成立年代は固定できない。刊行年を生成年代へ置換せず、1980年代という背景推測でも埋めない。

## D02 最古確認流通媒体
- Primary: `D02.PRT.BOOK`
- Parent: `D02.PRT`
- Status: `D`
- 根拠: 現在直接確認できる最古の流通媒体は1994年刊『ピアスの白い糸―日本の現代伝説』。それ以前の口承媒体は現Evidenceで固定しない。

## D03 確認流通媒体ポートフォリオ
- Primary: `D03.PRT.BOOK`
- Parent: `D03.PRT`
- Secondary: なし
- Status: `D`
- 根拠: Version Scopeの共有核が書籍へ採録・提示されたことを直接確認できる。学校口承・テレビ等は具体的媒体史を固定していないため追加しない。

## D04 生成・変容パターン
- Primary: `D04.VAR.RETELLING`
- Parent: `D04.VAR`
- Secondary: なし
- Status: `I`
- 根拠: 「引いて実際に失明する物語型」と「引くと失明すると警告される命題型」が共存し、白い糸と失明の因果核を保ちながら再話される。

## D05 提示形式
- Primary: `D05.PRP.PREDICTIVE_CLAIM`
- Parent: `D05.PRP`
- Secondary: `D05.RUL.WARNING`
- Status: `I`
- 根拠: Scope全体を貫く伝達可能な核は「白い糸を引けば失明する」という条件付き予測命題であり、実践上は「引くな」という警告として機能する。個々の物語型・FOAF型をScope全体のPrimaryに固定しない。

## D06 真実性提示
- Value: `D06.T5` — 条件付き信念
- Status: `I`
- 根拠: 「白い糸を引けば失明する」という因果規則を事実らしい危険知識として提示するが、特定一件の直接体験事実へScopeを限定しない。

## D07 意味形成対象
- Primary: `D07.BHF.BODY_ANOMALY` — 身体異常
- Parent: `D07.BHF`
- Secondary: `D07.BHF.MEDICAL_RISK` — 医療リスク
- Status: `D`
- 根拠: 中心問題は耳から見える白い糸状物の正体と、身体改変後にそれを操作した場合の重大な身体リスクである。

## D08 意味形成契機
- Primary: `D08.TRC.PHYSICAL_TRACE` — 物的痕跡
- Parent: `D08.TRC`
- Secondary: `D08.DEX.BODILY_SENSATION` — 身体感覚・身体徴候
- Status: `I`
- 根拠: 目に見える白い糸状物という物的・身体的徴候が、「これは何か」という意味形成を開始させる。

## D09 意味付与操作
- Primary: `D09.CAT.TYPE_ASSIGNMENT` — 類型化
- Parent: `D09.CAT`
- Secondary: `D09.CAU.DIRECT_CAUSE` — 直接原因化; `D09.NOR.TABOOIZATION` — 禁忌化
- Status: `I`
- 根拠: 白い糸を「視神経」という既知の身体構造へ同定し、糸を引くことを失明の直接原因とし、その操作を禁止すべきものへ変換する。

## D10 因果源存在論
- Primary: `D10.NAT.BIOPHYSIO_PROCESS` — 生理・生物過程
- Parent: `D10.NAT`
- Secondary: なし
- Status: `I`
- 根拠: 伝承内では、身体内部の神経構造とそれを引くことで生じる生理学的損傷が因果源として置かれる。これは医学的真偽ではなく伝承内存在論のコード。

## D11 発動・接触条件
- Primary: `D11.MAN.CREATE_ALTER_MANIPULATE` — 作る・加工・操作する
- Parent: `D11.MAN`
- Secondary: `D11.SEN.VISUAL_EXPOSURE` — 見る
- Status: `D`
- 根拠: 危害を発動させる中心条件は白い糸を引く・操作すること。糸を発見するには視覚曝露が必要だが、見るだけで失明するわけではないためSecondary。

## D12 作用対象
- Primary: `D12.FOC.PRACTITIONER` — 実践者
- Parent: `D12.FOC`
- Secondary: なし
- Status: `I`
- 根拠: D13の生理変化を直接受けるのは、白い糸を操作した本人である。

## D13 作用機構
- Primary: `D13.PHY.PHYSIOLOGICAL_CHANGE` — 生理変化
- Parent: `D13.PHY`
- Secondary: なし
- Status: `D`
- 根拠: 伝承は糸の操作によって視覚機能が失われるという直接的な身体機能変化を主作用としている。

## D14 帰結極性
- Value: `D14.NEG` — 負
- Status: `D`
- 根拠: 失明・視力喪失という重大な危害が中心帰結。

## D15 帰結領域
- Primary: `D15.BOD.SEVERE_INJURY` — 重傷・障害
- Parent: `D15.BOD`
- Secondary: `D15.BEH.AVOIDANCE_ROUTE_CHANGE` — 回避・経路変更
- Status: `I`
- 根拠: 主帰結は不可逆的な視覚障害。話が警告として受容される場合には、白い糸を触らないという行動回避も生じるが、中心は身体障害。

## D16 因果時間構造
- Primary: `D16.EVT.IMMEDIATE` — 即時
- Parent: `D16.EVT`
- Secondary: なし
- Status: `I`
- 根拠: 代表形では白い糸を引く操作と視覚喪失が直結し、長い潜伏・遅延を必要としない。

## D17 回避・制御方式
- Primary: `D17.RUL.OBEY_TABOO` — 禁忌遵守
- Parent: `D17.RUL`
- Secondary: `D17.AVO.DO_NOT_ENGAGE` — 接触回避
- Status: `I`
- 根拠: 実践的な制御規則は「白い糸を引いてはいけない」。対象への操作を避けることで危害を回避する。

## D18 作用レイヤー
- `D18.L1=1`
- `D18.L2=0`
- `D18.L3=0`
- Status: `D`
- 根拠: 伝承内部では操作→失明という因果がある。話を読む・聞くこと自体が危害を発動する自己適用規則はなく、現実社会での独立した集団・制度効果を示すX Evidenceも未確認。

## D19 流通範囲
- Primary / Parent / Secondary: なし
- Status: `U`
- 根拠: 1994年の日本の現代伝説集への採録は確認できるが、この個別話が当時どの社会範囲まで流通していたかを直接固定するEvidenceが不足する。

## D20 特権情報保持者
- Primary / Parent / Secondary: なし
- Status: `U`
- 根拠: 友人・医師・周囲の人物が「視神経」と説明する異伝はあるが、Scope全体で一貫した特権情報保持者を固定できない。

## D21 現実アンカー
- Value: `D21.A1` — 一般的現実背景
- Status: `D`
- 根拠: ピアス、耳、眼、神経という一般的現実対象を使うが、特定の病院・人物・商品・事件へ依存しない。

# 2. causal precheck

```text
D11: 白い糸状物を引く／操作する
→ D12: 操作した本人が直接対象になる
→ D13: 視覚機能を失う生理変化が起きる
→ D15: 重大な視覚障害として帰結する
```

接続は成立する。

# 3. D18 L3 precheck

現実の若者がピアスを避けた等の社会効果は推測可能でも、独立X Evidenceを確認していないため `L3=0`。

# 4. U / NA / C precheck

- D01=`U`: 採録年以前の成立年代を安全に年代帯へ写像できない。
- D19=`U`: 個別話の流通社会範囲が未固定。
- D20=`U`: 情報保持者が異伝で変動する。
- NA: なし。
- C: なし。

# 5. taxonomy gap precheck

現行Scopeは既存taxonomyで表現可能。新Child要求なし。

# 6. R3 freeze

本ファイルは旧 `0081_10_analysis.md` および旧Excel coding値を参照せずに作成した独立判定である。以後のR4で初めて旧10を参照する。