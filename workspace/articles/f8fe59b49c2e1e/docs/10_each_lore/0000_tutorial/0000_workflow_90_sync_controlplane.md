# 0. ドキュメント情報

## 0.1. 位置付け

本書はcontrol plane同期・整合確認を一元管理するオーケストレーションの正本である。

## 0.2. 設計原則

- 外部入力は `Entry_ID` のみ。
- control planeのcurrent state保持先はNotion `伝承エントリ調査状況`。
- Git上のcanonical artifactとReview結果JSONを事実源としてNotion stateを収束させる。
- 決定論的同期処理の外部入口は `src/status_management/sync_controlplane.py` とする。
- Workflow 90から直接呼び出すPythonはこの1入口に限定する。
- `sync_controlplane.py` の内部で `notion_controlplane.py` / `git_state.py` / `review_state.py` / `reconcile.py` を使用する。
- Workflow 90自身は研究内容やReview意味論を判定しない。

# 1. 目的

指定Entryについて、Notion control planeの状態とGit上のcanonical artifact・Review結果JSONの状態を比較し、同期可能な範囲でcurrent stateを正しい状態へ収束させる。

# 2. インターフェイス

## 2.1. 入力

- `Entry_ID`

## 2.2. 出力

- canonical artifactは生成しない。
- 同期結果を `PASS / UPDATED / BLOCKED / ERROR` のいずれかとして返す。
- 詳細は必要時のみreason / rule ID / remarks / 実行ログへ残す。
- `PASS / UPDATED / BLOCKED / ERROR` はWorkflow 90の実行結果であり、control planeの `Status` 値ではない。

# 3. 管理単位

- 論理単位は `(Entry_ID, 成果物)`。
- `成果物` は `00 / 10 / 20`。
- Notionの `key` はtitle制約用であり、意味論的主キーにしない。

# 4. 対象state

- `Entry_ID`
- `伝承`
- `成果物`
- `Status`
- `最新レビュー版`
- `pre-SHA`
- `post-SHA`
- `remarks`

## 4.1. Status正本

`Status` は以下の7値のみを許容する。

| Status | 定義 |
|---|---|
| `未` | Coder作業未着手 |
| `レビュー待` | 初回Coder成果物commit完了、初回Review待ち |
| `要修正` | Reviewで修正要求が確定し、修正成果物commit前 |
| `再作業中` | CoderがReview指摘への修正作業中 |
| `再レビュー待` | 修正成果物commit完了、再Review待ち |
| `完了` | 対象artifactの最新ReviewがPass |
| `－（対象外）` | 当該成果物を適用対象外と明示した状態 |

- Notion `伝承エントリ調査状況.Status` のSelect optionsを機械的許容値の正本とする。
- `reconcile.py` は上記7値以外を受け入れずBLOCKする。
- `－（対象外）` はGit / Review事実だけから自動付与しない。

## 4.2. Status遷移

Legacy標準Workflowで確定していた状態遷移を継承する。

| イベント | Status | 最新レビュー版 | pre-SHA | post-SHA |
|---|---|---|---|---|
| 初回Coder成果物commit完了 | `レビュー待` | 変更なし | 編集入力checkpoint | 新成果物commit |
| Reviewで修正要求確定 | `要修正` | 実施済みReview Seq | 変更なし | 変更なし |
| Review Pass確定 | `完了` | 実施済みReview Seq | 変更なし | 変更なし |
| 修正開始 | `再作業中` | 変更なし | `old post-SHA` | commit確定まで`old post-SHA`保持 |
| 修正成果物commit完了 | `再レビュー待` | 変更なし | 修正開始時の値を保持 | 新成果物commit |
| 再Reviewで修正要求 | `要修正` | 新Review Seq | 変更なし | 変更なし |
| 再Review Pass | `完了` | 新Review Seq | 変更なし | 変更なし |

追加規則:

- `再レビュー待` では、最新Review targetが現artifactのancestorであることは正常状態であり、stale ReviewとしてBLOCKしない。
- `再作業中` では、直前の修正要求Reviewが最新Reviewとして残っていても `要修正` へ巻き戻さない。
- `完了` 後にcanonical artifactが更新された場合、その版は未Reviewなので `再レビュー待` へ戻す。
- `要修正 -> 再作業中` は「Coderが修正を開始した」という運用イベントを必要とする。Entry_IDだけの同期事実から開始意思を推測してはならない。

# 5. 実行手順

## 5.1. Step 0: Python入口実行

```text
python -m src.status_management.sync_controlplane <Entry_ID>
```

- repository path、Notion識別子、Review保存先等を通常のWorkflow引数として要求しない。

## 5.2. Step 1: 同期結果受領

- `PASS`: 実状態とcontrol planeが整合し、更新不要。
- `UPDATED`: 事実から一意に導出できるcontrol plane更新を実施し、更新後検証まで完了。
- `BLOCKED`: divergence、重複、未知Status、解釈不能なstale state等により安全な自動収束ができない。
- `ERROR`: 設定・I/O・実行不能等で同期処理自体を完了できない。

## 5.3. Step 2: Workflow分岐

- `PASS / UPDATED` の場合のみ呼出元Workflowは後続処理へ進める。
- `BLOCKED / ERROR` の場合はreason / rule IDを保持して停止する。
- Workflow 90自身でSHA比較、Notion更新、Review JSON走査を再実装しない。

## 5.4. Python内部の責務境界

- `notion_controlplane.py`: Notion state取得・更新・更新後確認。Status enumも検査する。
- `git_state.py`: artifact commit / blob / commit graph事実取得。
- `review_state.py`: Review結果JSONを読み、artifactに応じて `review_00_sources.schema.json / review_10_contents.schema.json / review_20_analysis.schema.json` で構造確認したReview事実を取得する。`review_common.schema.json` は共通定義としてのみ利用する。
- `reconcile.py`: I/Oなしで同期可否・Status収束・mutation planを決定。
- `sync_controlplane.py`: 上記を組み立て、mutation適用と最終結果返却をオーケストレーションする。

# 6. SHA関係の解釈

- `exact`: 対象版一致。
- control plane `post-SHA` が現artifact SHAのancestor: control planeが古い。事実から一意に更新可能ならpre/postを前進させる。
- Review対象SHAが現artifact SHAのancestor:
  - `再レビュー待` または修正commit直後であれば正常。
  - その他の状態ではstale ReviewとしてBLOCKする。
- 現artifact SHAがReview対象SHAのancestor: Review target aheadとしてBLOCKする。
- `diverged`: 自動修復せず調査対象。
- Review成果物自身のcommitをcanonical artifactの `post-SHA` として扱わない。

# 7. Review結果の保存先

Review結果JSONは以下を事実源とする。

```text
reviews/10_each_lore/<Entry_ID>/
└─ Review_<Entry_ID>_<Artifact>_<Review_Seq>.json
```

Schema対応:

```text
Artifact 00 -> review_00_sources.schema.json
Artifact 10 -> review_10_contents.schema.json
Artifact 20 -> review_20_analysis.schema.json
```

- Markdown derived viewは同期事実源として使用しない。
- 「最新Review」はmtimeではなくReview Seqを基準とする。
- artifactとSchemaが不一致のReview JSONはmalformedとして扱い、自動同期の事実源にしない。

# 8. Workflowからの呼出し

- Workflow 10は作業開始前、canonical artifact commit後、Review修正後、完了時に必要に応じWorkflow 90を呼ぶ。
- Workflow 20はcontrol planeを直接更新せず、必要な整合確認をWorkflow 90へ委譲する。

# 9. 不変条件

- current stateはNotionに保持する。
- Notion stateはGit / Review事実より優先しない。
- Status enumはNotion Select、Workflow 90、`reconcile.py` で一致させる。
- 同期はidempotentであること。
- 不明状態を推測で正常化しないこと。
- duplicate、malformed Review、divergence、concurrent update等の曖昧状態では自動更新しないこと。
- control planeからEvidence内容やD01〜D21妥当性を推定しないこと。

# 10. 未確定事項

- `要修正 -> 再作業中` の開始イベントを、Entry_IDだけの公開CLIを維持したままどの内部事実で表現するか。
- Review Seq採番・Review JSON保存・Verdict集約を担うPython実装ファイルとの境界。
