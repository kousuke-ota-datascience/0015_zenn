# 0118 Review_001 Coder裁定

- Entry_ID: `0118`
- 対象Review: `Review_0118_00_001.md`, `Review_0118_10_001.md`
- 00判定: 要修正（Major）を受理。
- 10判定: 要修正（Major）を受理。

## 1. 00 Finding裁定

| Finding | 裁定 | 対応 |
|---|---|---|
| F00-001 tutorial構造 | ACCEPT | 0.2.1〜0.2.3、1.3〜1.6の表・項目、2.2/2.3、3.1〜3.3をtutorial正規構造へ整形 |
| F00-002 summaryの意味形成分析 | ACCEPT | 景観の意味形成・痕跡解釈等を除き、怪談上の主張とEvidence上の未確認事項へ限定 |
| F00-003 回避法の論理反転 | ACCEPT | 「切らない」を独立した回避法として扱わず、明示的な回避・解除手順は未確認と記録 |
| F00-004 「安定核」の00側裁定 | ACCEPT | 0.3を一次的に固定できない事実の一覧へ変更し、Version Scope裁定を00から除去 |
| F00-005 流通経路の過剰再構成 | ACCEPT | 2008/2016/2019/2023の確認点のみ時系列記録し、口コミ→媒体の伝播経路は未確定と明記 |

## 2. 10 Finding裁定

| Finding | 裁定 | 対応 |
|---|---|---|
| D10近似コードによるtaxonomy gap隠蔽 | ACCEPT | `SPECIFIC_PLACE` / `IMPERSONAL_CURSE` を撤回し、コード空欄の `taxonomy gap: 特定自然物そのもの／自然物に固有の効力` として `Status=I` を保持 |
| D17 `DO_NOT_ENGAGE/D` | ACCEPT | trigger反転による生成だったため撤回し、コードなし `U` |
| D19 `C` / 全国流通 | ACCEPT | `D19.LOC.LOCAL_TRADITION / I` に限定し全国大衆コードを撤回 |
| D06 `T5` | ACCEPT | 外部ジャンル分類・想定受容行動から真実性を補わず、コードなし `U` |
| D04変容過程 | ACCEPT | 媒体併存を変容機構へ変換せず、コードなし `U` |
| tutorial形式・独自QA節 | ACCEPT | 階層番号・中心質問を復元し、独自QA節をCoding正本から除去 |

## 3. 修正結果

- `0118_00_contents.md`
  - 修正commit: `8a5614b692c94a785af2a3870b2915a64bb5cde1`
  - 修正後blob: `130455f030804f0b4bd653988219b26f93e02616`
- `0118_10_analysis.md`
  - 修正commit: `54932c8a44799d927c442eba7b7342b7f739b034`
  - 修正後blob: `aa5ad0ad4e613f0a859374b05cebb8cd25637e33`

## 4. QA

- 00と10の責務分離: PASS
- Status単一値: PASS
- D10 Evidence不足とtaxonomy不足の分離: PASS
- D17 trigger反転禁止: PASS
- D18 L3独立X Evidence要件: `0` を維持
- tutorial構造: PASS

## 5. 次状態

`再レビュー待 / 001`。Review_002で00/10双方の再レビューを待つ。