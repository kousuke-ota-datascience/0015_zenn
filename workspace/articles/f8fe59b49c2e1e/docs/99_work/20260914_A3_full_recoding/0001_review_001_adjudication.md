# 0001 Review_001 Coder Adjudication

## 1. 対象

- Entry: `0001` 口裂け女
- Reviewer成果物:
  - `docs/99_work/review_10_each_lore/0001/Review_0001_00_001.md`
  - `docs/99_work/review_10_each_lore/0001/Review_0001_10_001.md`
- 修正前00 blob: `8a6839f6c9615fb092e34124f67b420cb5c89b9e`
- 修正前10 blob: `b0ac350bb70af8b20ccab1d36ea6ee2e6a9f4d2b`
- 修正後00 blob: `b6a3c80f69e6c4b5b2d09a59615f477984ab3e00`
- 修正後10 blob: `d183407a90cdfd88d6b58b17963bb55ca4f908d3`

## 2. 00 Review裁定

| Finding | 裁定 | 対応 |
|---|---|---|
| F00-001 summaryへのsense-making結論混入 | 受理 | 「未知の脅威を予測・制御可能化する共同体知」等の機能評価を削除し、確認できる異伝・回避規則の記録へ戻した |
| F00-002 異なる時期・異伝の因果列混成 | 受理 | `1.6` を早期核／流行過程の代表化型／回答異伝／回避異伝に分離した |
| F00-003 流通媒体・社会対応主体の主体表混入 | 受理 | `1.3` を伝承内容内部の主体に限定し、流通主体・社会対応主体・媒体を別節へ移した |
| F00-004 一次資料本文未実見の管理 | 維持 | 現行の保守的Evidence運用を維持した |

追加して、`1.7` の「制御可能な存在へ再構成」という分析表現を除去し、`2.4` と `3.3` もEvidence観察とAnalysis判断の責務を分離した。

## 3. 10 Review裁定

| Finding | 裁定 | 対応 |
|---|---|---|
| F10-001 D20 `NO_HIDDEN_TRUTH` が強すぎる | 受理 | D20を `U` へ変更。特権保持者が確認できないことと「秘密情報を想定しない」ことを区別した |
| F10-002 D07–D09骨格／D09 tie-break | 一部修正 | D07/D08は維持。D09は中心質問への直接性・保守性から `NAMING` をPrimary、`AGENCY_ATTRIBUTION` をSecondaryへ変更し、`DIRECT_CAUSE` は独立性が弱いため除外した |
| F10-003 D18 bit別Evidence性質 | 受理 | L1/L3の直接EvidenceとL2=0の不在判定を明記し、vector全体のStatusを保守的に `D → I` へ変更した |
| F10-004 D12→D13→D15 | 維持 | `DEMOGRAPHIC_GROUP → PHYSICAL_ATTACK → U` を維持した |
| F10-005 D06 / D17のU | 維持 | ともに `U` を維持した |

## 4. Sense-making再確認

修正版の中心連鎖は以下とする。

```text
D07 見知らぬ他者による子どもへの不確実な脅威
→ D08 子ども共同体で反復される根拠未提示の噂
→ D09 無名の危険を「口裂け女」と命名し、意思ある女性主体へ帰属
→ D10 原因主体を、初期Scopeでは超自然化せず人間個人として置く
```

これにより、D07の「説明を要求する対象」、D08の「手掛かり」、D09の「理解可能化操作」、D10の「原因存在論」を分離する。

## 5. 書式・正本QA

- `0001_00_contents.md`: tutorialのEvidence / Content layerとして再整理。
- `0001_10_analysis.md`: 0180 Golden Referenceに合わせ、基本情報を5項目に限定し、D01〜D21を番号付き見出し・中心質問・固定フィールド・判定根拠・現れ方で記述。
- Coding正本からR4 QA・旧10比較等の作業用節を除去した。過去内容はGit履歴に保持される。

## 6. 次状態

修正版00/10を `Review_002` の対象とし、Entry Statusを `再レビュー待` とする。Reviewerが00/10双方を承認するまでは `完了` にしない。
