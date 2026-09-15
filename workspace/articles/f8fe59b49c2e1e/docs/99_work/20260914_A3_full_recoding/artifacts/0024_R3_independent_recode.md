# 0024 幸福の手紙 — R3 Independent Recode

- Entry_ID: `0024`
- Version Scope正本: `docs/99_work/20260914_A3_full_recoding/0024_R2_version_scope.md`
- Evidence正本: `docs/10_each_lore/0024_kofuku_no_tegami/0024_00_contents.md`
- baseline: control plane記載の固定blob
- 旧 `0024_10_analysis.md`: **未参照**
- 旧Excel coding値: **未参照**

## 1. 独立判定

### D01 生成年代
- Value: `D01.G1` — 1868–1944
- Status: `I`
- 根拠: 1922年の日本流行層を専門研究が同時代新聞資料から固定する。ただし本作業では1922年新聞原紙を直接閲覧していないため、成立層への写像はIとする。

### D02 最古確認流通媒体
- Primary: `D02.PRT.CHAIN_LETTER`
- Parent: `D02.PRT`
- Status: `I`
- 根拠: 丸山2012が1922年の葉書チェーンを資料化する。原葉書／新聞原紙を今回直接固定していないためI。

### D03 確認流通媒体ポートフォリオ
- Primary: `D03.PRT.CHAIN_LETTER`
- Parent: `D03.PRT`
- Status: `I`
- 根拠: Version Scopeの成立に、複写・転送される葉書／手紙というチェーン媒体が構造的に不可欠。新聞は現象を報じる記録媒体であり、Scoped伝承そのものの伝播媒体Primaryとはしない。

### D04 生成・変容パターン
- Status: `U`
- 根拠: 1922年の狭いScope内で、時間に沿った安定した変容を確定できるHistory Evidenceが不足する。後代の不幸の手紙化・電子化をScopeへ持ち込まない。

### D05 提示形式
- Primary: `D05.RUL.CHAIN_INSTRUCTION`
- Parent: `D05.RUL`
- Secondary: `D05.RUL.JINX_RULE`
- Status: `I`
- 根拠: 同文を規定人数・枚数へ転送する指示が中心形式であり、その行為を幸運／悪運へ結ぶジンクス規則が独立して不可欠。

### D06 真実性提示
- Value: `D06.T5` — 条件付き信念
- Status: `I`
- 根拠: 「規定どおり送れば幸運、断てば悪運」という条件付き因果を信じるよう要求する形式であり、一般科学事実や真偽未確定の謎として提示するものではない。

### D07 意味形成対象
- Primary: `D07.DCF.FATE_OMEN` — 運命・予兆
- Parent: `D07.DCF`
- Status: `I`
- 根拠: この伝承がなければ、「将来の幸運／悪運をどのように予測し、制御できるか」が説明されないまま残る。

### D08 意味形成契機
- Primary: `D08.TRC.TEXT_DOCUMENT` — 文書・記載
- Parent: `D08.TRC`
- Secondary: `D08.CLM.UNSUPPORTED_ASSERTION`
- Status: `I`
- 根拠: 受信者が実際に接する手掛かりは転送規則と吉凶条件を記した文書であり、その文書内の因果主張には独立した根拠が提示されない。

### D09 意味付与操作
- Primary: `D09.CTL.TRANSMISSION_RULE` — 伝達規則
- Parent: `D09.CTL`
- Secondary: `D09.PPR.CORRELATION_RULE`
- Status: `I`
- 根拠: 不確実な将来の吉凶を「同文を他者へ送る」という再現可能な伝達規則へ変換し、転送／断絶と幸運／悪運を対応規則として結ぶ。

### D10 因果源存在論
- Primary: `D10.OBJ.INFORMATION_CONTENT` — 情報内容
- Parent: `D10.OBJ`
- Status: `I`
- 根拠: 人格的主体は必要とされず、複写される文面・その規則内容が因果条件を保持して次の受信者へ移る。明示されない神格・呪詛主体を補わない。

### D11 発動・接触条件
- Primary: `D11.INF.RECEIVE_MESSAGE` — 通知・メッセージを受ける
- Parent: `D11.INF`
- Status: `I`
- 根拠: 受信者がチェーン文面を受け取ることで期限・転送条件と吉凶規則の対象になる。転送実行は発動入口というより結果を制御する行為としてD17で扱う。

### D12 作用対象
- Primary: `D12.FOC.PRACTITIONER` — 実践者
- Parent: `D12.FOC`
- Status: `I`
- 根拠: D13の主作用である幸運／悪運付与が直接向けられるのは、転送規則を実行または不実行する受信者本人。

### D13 作用機構
- Primary: `D13.FAT.LUCK_BENEFIT` — 幸運・利益付与
- Parent: `D13.FAT`
- Secondary: `D13.FAT.CURSE_MISFORTUNE`
- Status: `I`
- 根拠: Scopeは報酬前景型であり、転送実行に幸運を付与する作用をPrimaryとする。同じ初期文面に連鎖断絶時の悪運付与が独立して含まれるためSecondary。

### D14 帰結極性
- Value: `D14.MIX`
- Status: `I`
- 根拠: 転送時の正の報酬と、断絶時の負の制裁が同じScoped規則に重要な分岐として併存する。

### D15 帰結領域
- Primary: `D15.OPP.LUCK_MISFORTUNE` — 幸運・不運
- Parent: `D15.OPP`
- Status: `I`
- 根拠: 最終的に変わると主張されるのは受信者の一般的な運勢・吉凶である。

### D16 因果時間構造
- Primary: `D16.DLY.DEADLINE` — 期限付き
- Parent: `D16.DLY`
- Secondary: `D16.TRN.CHAIN_SPREAD`
- Status: `I`
- 根拠: 「24時間以内」「3日以内」等の期限遵守が意味上重要であり、同時にA→B→Cと受信者間を移る連鎖拡散が独立した時間構造を形成する。

### D17 回避・制御方式
- Primary: `D17.USE.STRATEGIC_RULE_USE` — 規則の戦略利用
- Parent: `D17.USE`
- Secondary: `D17.RUL.TIMING_ORDER`
- Status: `I`
- 根拠: 伝承内部では転送規則を利用して幸運を得る／悪運を避けることが制御法となる。期限遵守も独立した条件として必要。

### D18 作用レイヤー
- `D18.L1=1`
- `D18.L2=1`
- `D18.L3=1`
- Status: `D`
- 根拠:
  - L1: 文面内部で転送／断絶と幸運／悪運の因果が明示される。
  - L2: 現実側の受信者が、手紙を受け取ることでその文面上の因果対象になる自己適用構造を持つ。
  - L3: 1951年国会会議録が、この種の手紙の実際の転送行動と郵便制度上の検討を直接記録するX Evidenceである。

### D19 流通範囲
- Primary: `D19.LOC.REGIONAL` — 地域・地方
- Parent: `D19.LOC`
- Status: `I`
- 根拠: Scopeの最古層は1922年東京の新聞資料を中心に追跡される。後代の全国的・世代横断的知名度を1922年Scopeへ遡及しない。

### D20 特権情報保持者
- Primary: `D20.NON.NO_HIDDEN_TRUTH` — 隠れた真相なし
- Parent: `D20.NON`
- Status: `I`
- 根拠: 実行条件・報酬・制裁はチェーン文面自体へ提示され、専門家・内部者だけが持つ追加の真相／解除情報を必要としない。

### D21 現実アンカー
- Value: `D21.A1` — 一般的現実背景
- Status: `I`
- 根拠: 郵便・葉書という現実的通信環境に埋め込まれるが、特定制度規則・特定人物・特定事件を伝承内因果の成立条件として組み込むわけではない。1951年の国会対応はX Evidenceであり、Scope内の因果アンカーへ遡及しない。

## 2. causal precheck

```text
D11: チェーン文面を受信する
→ D12: 転送規則を実行／不実行する受信者本人
→ D13: 転送時は幸運、断絶時は悪運を付与する
→ D15: 本人の幸運／不運が変わる
```

接続は成立する。

## 3. taxonomy gap precheck

現行Scopeは既存taxonomyで表現可能。確定的な新Child要求なし。

## 4. R3 freeze

本ファイルは旧 `0024_10_analysis.md` および旧Excel coding値を参照せずに作成した独立判定である。以後のR4で初めて旧10を参照する。