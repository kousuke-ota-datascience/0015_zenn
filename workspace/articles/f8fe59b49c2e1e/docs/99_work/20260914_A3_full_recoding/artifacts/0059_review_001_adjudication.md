# 0059 Review_001 Coder adjudication

- Entry_ID: `0059`
- Entry: 深泥池の幽霊タクシー
- Review source:
  - `docs/99_work/review_10_each_lore/0059/Review_0059_00_001.md`
  - `docs/99_work/review_10_each_lore/0059/Review_0059_10_001.md`

# 1. 00_contents adjudication

## F00-001 / F00-002 — tutorial正規構造

**受理。**

典拠章を `0.2.1–0.2.3`、Content章を `1.1–1.10` へ再配置した。主体・場所・物・条件・行動・帰結・未解決点を独立させた。

## F00-003 — 分析判断の混入

**受理。**

旧summary・中心命題に含まれていた「伝承構造上の中心」「後代再話は幽霊だったと解釈する」といったD07/D09/D10相当の分析判断をEvidence正本から除去した。00では、乗客消失が報告されたことと、後代資料に幽霊・死者解釈があることを分離して記録する。

## F00-004 — 1969年記事と深泥池の結合directness

**受理。**

以下を明示的に分離した。

1. 1969年記事の存在・日付・見出し「車から乗客消えた」は京都府立図書館の確認を介して固定できる。
2. 深泥池に関係するタクシー幽霊譚の照会に対し、図書館レファレンスが当該記事を初期資料として関連付けている。
3. 本作業では1969年記事本文を直接閲覧しておらず、記事本文自体が深泥池を明記することは未確認である。

## F00-005 — 後代要素の遡及防止

**維持。**

若い女性、雨、濡れた座席、死亡者同定等を1969年へ遡及しない。警察対応も幽霊実在または伝承流通によるX Evidenceとして扱わない。

# 2. 10_analysis adjudication

## F10-001 / F10-003 — D13 `MANIFEST_ONLY`

**受理し、ゼロベース再判定。**

`D13.MAN.MANIFEST_ONLY`を撤回した。現行D13 taxonomyには、対象を「消失させる／不在化する」作用を直接表すChildがない。一方、乗客が車内から不在になるという変化自体はScopeの中心Evidenceである。

したがってD13はコード空欄の **taxonomy gap: 対象の消失／不在化** とし、Status `I` で保持する。原因主体・作用機構をEvidence以上に仮定しないため、`REALITY_ALTERATION`等への近似も行わない。

## F10-002 — D12

**受理。**

`D12.FOC.PROTAGONIST_EXPERIENCER`を撤回し、`D12.OTH.SPECIFIC_OTHER` — 特定他者へ変更した。D13の直接変化を受けるのは観測者である運転手ではなく、消失する乗客である。被害・危害まで確定しないため `VICTIM_TARGET` は採らない。

## D15 causal re-audit

Reviewer指摘を受け、D12→D13→D15をコード体系へ戻って再確認した結果、D15には既存Child `D15.LIF.DISAPPEARANCE` — 失踪・消失が存在した。

したがってD15を以下へ修正した。

- Primary: `D15.LIF.DISAPPEARANCE`
- Secondary: `D15.KNW.UNCERTAINTY_PRESERVED`

これにより因果列は、

```text
D12: 乗客（特定他者）
→ D13: 消失／不在化（taxonomy gap）
→ D15: 乗客の失踪・消失
＋ 消失理由・正体は不確実なまま残る
```

となり、D07の中心異常「乗客消失」を因果モデル内へ直接通す。

## F10-004 — D17

**受理。**

`D17.NON.OBSERVATIONAL_ONLY`を撤回し、`U`へ変更した。安定した制御法が確認できないことは、「制御不要」を意味しない。警察への届出は伝承内制御規則ではない。

## F10-005 — D21

**受理。**

`D21.A2`自体は維持する。深泥池は実在地点であり、図書館レファレンスが1969年記事を深泥池伝承へ関連付けているためである。ただし記事本文での深泥池明記は未確認なのでStatusを `D` から `I` へ変更した。

## F10-006 — D07〜D10 / D18

**維持。**

- D07 `UNEXPLAINED_EVENT`
- D08 `DIRECT_EVENT + WITNESS_REPORT`
- D09 `MYSTERY_LABEL`
- D10 `U`
- D18 `L1=1;L2=0;L3=0 / I`

を維持する。

## F10-007 — 書式

**受理。**

Entry_IDを`0059`へ正規化し、0180基準のD01〜D21正規書式へ統一した。R4 QA・旧10比較はCoding正本から除去した。

# 3. 修正後QA

- 00/10責務分離: PASS
- 1969記事と深泥池関連付けdirectness: 分離済み
- D07→D10: PASS / 維持
- D12: `SPECIFIC_OTHER`
- D13: taxonomy gap「対象の消失／不在化」
- D15: `DISAPPEARANCE` Primary + `UNCERTAINTY_PRESERVED` Secondary
- D12→D13→D15 causal QA: PASS
- D17: `U`
- D18 L3: `0` 維持
- D21: `A2 / I`
- canonical format: PASS

# 4. 修正版checkpoint

- 00 commit: `f88b0f81b8fc737640841640dd4830bd25144757`
- 00 blob: `383cce593f6afd4af92286d548d06c30d6804b94`
- 10 commit: `2b08c72dd5222970b080e21096882ad8f3a7060b`
- 10 blob: `03ff047a67d1cf6947baf1edec71a442d75bfdbc`

Coder再作業完了。次状態は `再レビュー待`。