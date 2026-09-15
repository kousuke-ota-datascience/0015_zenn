# 個別伝承Review 実行指示

以下の対象Entryについて、個別伝承Reviewを実施せよ。

## 対象リポジトリ

`https://github.com/kousuke-ota-datascience/0015_zenn/tree/main/workspace/articles/f8fe59b49c2e1e/`

## 使用するReview workflow

`docs/10_each_lore/0000_tutorial/0000_workflow_20_review.md`

## 使用するcontrol plane

`docs/99_work/20260914_A3_full_recoding/_control_plane.md`

## 対象Entry_ID

<TARGET_ENTRY_IDS>

## 実行指示

上記Review workflowを正本として、対象Entryの以下の正本成果物をReviewすること。

- `<Entry_ID>_00_contents.md`
- `<Entry_ID>_10_analysis.md`

control planeは、Review対象版および現在状態を確認するための参照情報として使用する。

実行時に指定した上記control planeを使用し、workflow内から別のcontrol planeを推測・探索してはならない。

**レビューエージェントはcontrol planeを更新してはならない。control planeは参照専用とする。**

Reviewerは対象となる正本成果物を修正してはならない。

Review結果は、Review workflowで定める形式・保存規則に従ってReview mdとして記録すること。

既存の過去Review、旧分析値、作業メモ等を正しいものとして前提化せず、現行の正本、workflow、tutorial、理論設計、コード体系、分析コード付与規則に従って独立にReviewすること。

対象Entryが複数ある場合も、Review workflowで定める単位・順序・完了条件に従って処理すること。

## 完了報告

作業完了後、以下を簡潔に報告すること。

- ReviewしたEntry_ID
- 作成したReviewファイル
- 各成果物の対象commit SHA
- Review Seq
- 最終判定
- Major / Moderate / Minor Findingの有無
- 対象版を確定できずReviewを停止した場合はその理由

control planeのStatus、最新レビュー版、pre-SHA、post-SHA等は更新しないこと。
