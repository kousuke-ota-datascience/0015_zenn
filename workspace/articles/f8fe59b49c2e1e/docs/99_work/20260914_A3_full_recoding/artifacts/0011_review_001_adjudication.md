# 0011 Review_001 Coder Adjudication

## 1. 対象

- Entry: `0011` こっくりさん
- Reviewer成果物:
  - `docs/99_work/review_10_each_lore/0011/Review_0011_00_001.md`
  - `docs/99_work/review_10_each_lore/0011/Review_0011_10_001.md`
- 修正前00 blob: `75cfe144e341544fbe0552a5b65c5789940c7881`
- 修正前10 blob: `c6c9e4775448b71452b3bc4370032fd3f5dc41fb`
- 修正後00 blob: `f0ac57c2be12f4e2ab0cb5f638f033dc0b841b3b`
- 修正後10 blob: `e826774e81dc51230340c542653e48e649b7401f`

## 2. 00 Review裁定

| Finding | 裁定 | 対応 |
|---|---|---|
| F00-001 tutorial章構造未準拠 | 受理 | `0.2.1`〜`0.2.3`、`1.1`〜`1.10`、`2.1`〜`2.4`、`3.1`〜`3.4`へ正規化 |
| F00-002 Version Scope・意味形成判断混入 | 受理 | `Scope上の扱い`、分析的な「意味形成核」を00から除去し、観察可能な器具・質問・回答・異伝へ分解 |
| F00-003 1887原典directness強化 | 部分受理 | 今回は新規外部調査を追加せず、原典本文頁未固定という制約を明記。Directnessが必要なD11はStatusをIへ下げた |

## 3. 10 Review裁定

| Finding | 裁定 | 対応 |
|---|---|---|
| F10-001 D07 Primary | 受理 | `UNEXPLAINED_EVENT`を撤回し、`D07.ISU.INFORMATION_VOID`をPrimaryへ変更 |
| F10-002 D08循環 | 受理 | D13器具運動をD08から除外し、実践が答えを与えるという先行命題を `D08.CLM.UNSUPPORTED_ASSERTION / I` とした |
| F10-003 D09 | 受理・維持 | `AGENCY_ATTRIBUTION`を維持し、情報空白を回答能力を持つ外部主体へ接続する操作として再記述 |
| F10-004 D04/D20 taxonomy gap | 受理 | `U`を撤回。コード空欄＋taxonomy gapを明示しStatusはEvidence強度に応じ`I` |
| F10-005 D11 Directness | 受理 | 原典本文位置未固定のため `D → I` |
| F10-006 Entry_ID | 受理 | `0011`へ正規化 |

## 4. Sense-making再確認

```text
D07 参加者が自力では知り得ない未来・秘密・判断対象等の情報空白
→ D08 「この実践なら答えを得られる」という先行命題
→ D09 回答能力を持つ外部主体へ情報空白を接続する
→ D10 神格／狐狗狸等の外部主体を原因世界へ置く（C）
→ D11 所定の器具実践を開始
→ D13 外部主体が器具を動かす
→ D15 器具運動を回答として読み、知識を得る
```

D13の器具運動をD07/D08へ逆流させず、意味形成入力とモデル内部の作用を分離した。

## 5. Taxonomy gap

- D04: 外来実践の文化的／物質的適応
- D20: 超自然的回答主体による情報保持

いずれもEvidence不足ではないため `U` を使用しない。単一Entryで新Childは追加せず、R5横断確認へ送る。

## 6. 次状態

修正版00/10を `Review_002` の対象とし、Entry Statusを `再レビュー待` とする。Reviewerが00/10双方を承認するまでは `完了` にしない。