# 0005 Review_001 Coder Adjudication

## 1. 対象

- Entry: `0005` 紫の鏡
- Reviewer成果物:
  - `docs/99_work/review_10_each_lore/0005/Review_0005_00_001.md`
  - `docs/99_work/review_10_each_lore/0005/Review_0005_10_001.md`
- 修正前00 blob: `06eb6792dcce4c34fd9a6fb7b582a0884ea39ed9`
- 修正前10 blob: `ded7dae87a09a7e54ff12280a2ff012b2a6f1f2c`
- 修正後00 blob: `8d2c360343d74a5fbb77fbc806ed406ba0942199`
- 修正後10 blob: `2e061436bb859e0db426ef950299325ab5defaff`

## 2. 00 Review裁定

| Finding | 裁定 | 対応 |
|---|---|---|
| F00-001 summaryの「自己言及構造」評価 | 受理 | 心理一般化・機能評価を削除し、「伝承を聞く→語を知る→期限まで記憶していると危害」というContent記述へ限定 |
| F00-002 「記憶内容そのものが運命へ作用」 | 受理 | D10/D13を先取りする存在論・作用解釈を除去し、「期限まで覚えていると死亡／不幸」とされる規則だけを記録 |
| F00-003 異伝分離・留保 | 維持 | 期限差、死亡／不幸、複合語、対抗語、人格怪異化の分離を維持 |
| F00-004 tutorial構造 | 維持 | `0.2.1`〜`0.2.3` 等の現行構造を維持 |

## 3. 10 Review裁定

| Finding | 裁定 | 対応 |
|---|---|---|
| F10-001 D17 taxonomy gap | 受理 | `D17.RUL.OBEY_TABOO` を撤回。Primaryを空欄とし、Evidence上直接確認できる「忘却／記憶消去による制御」をtaxonomy gapとして明示 |
| F10-002 D20 `DANGEROUS_TO_KNOW` | 受理 | 「知るだけ」では危害条件が完成しないため撤回。共同体で共有され特権保持者を要しない構造として `D20.NON.COMMON_KNOWLEDGE` へ変更 |
| F10-003 D07〜D16 | 維持 | sense-making / causal modelの骨格を維持 |
| F10-004 D18 L2 | 維持＋Status修正 | `L1=1; L2=1; L3=0` を維持。L3=0がX Evidence不在判定を含むためvector Statusを `D → I` |
| F10-005 Version Scope | 維持 | 1998–1999年の死亡型共有核を維持 |

## 4. D17 taxonomy gapの扱い

Evidence上の回避構造は明確である。

```text
期限前に「紫の鏡」を忘れる
→ 期限時に危険な記憶保持条件が成立しない
→ 死亡を回避する
```

しかし現行D17には「一度知った情報を忘れる／記憶から失う」を直接表すChildがない。

- `DO_NOT_ENGAGE`: 接触前回避であり不適合
- `OBEY_TABOO`: 禁止事項遵守であり、忘却そのものを表現しない

workflow 9.6に従い、最も近いChildへ押し込まずコード空欄とする。Status `D` はEvidence上の構造が直接確認できることを示すもので、既存Child適合を示さない。本件はR5のtaxonomy gap横断確認へ送る。

## 5. Sense-making再確認

```text
D07 将来の死・運命の不確実性
→ D08 根拠未提示の危険命題＋共同体反復
→ D09 記憶保持と死亡を相関規則化し禁忌化
→ D10 危険の因果条件を情報内容へ置く
→ D11 危険語の記憶保持＋期限年齢
→ D12 受容者
→ D13 認知状態を危害条件化
→ D15 死亡
→ D16 期限付き
```

## 6. 書式・正本QA

- 00はEvidence / Analysis責務分離を再確認した。
- 10は0180 Golden Reference形式へ正規化し、基本情報5項目、D01〜D21、判定根拠、現れ方のみをCoding正本へ残した。
- R4 QA・旧10比較等の作業節はCoding正本から除去した。

## 7. 次状態

修正版00/10を `Review_002` の対象とし、Entry Statusを `再レビュー待` とする。Reviewerが00/10双方を承認するまでは `完了` にしない。
