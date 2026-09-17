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

# 5. 実行手順

## 5.1. Step 0: Python入口実行

```text
python -m src.status_management.sync_controlplane <Entry_ID>
```

- repository path、Notion識別子、Review保存先等を通常のWorkflow引数として要求しない。

## 5.2. Step 1: 同期結果受領

- `PASS`: 実状態とcontrol planeが整合し、更新不要。
- `UPDATED`: 事実から一意に導出できるcontrol plane更新を実施し、更新後検証まで完了。
- `BLOCKED`: divergence、重複、stale state等により安全な自動収束ができない。
- `ERROR`: 設定・I/O・実行不能等で同期処理自体を完了できない。

## 5.3. Step 2: Workflow分岐

- `PASS / UPDATED` の場合のみ呼出元Workflowは後続処理へ進める。
- `BLOCKED / ERROR` の場合はreason / rule IDを保持して停止する。
- Workflow 90自身でSHA比較、Notion更新、Review JSON走査を再実装しない。

## 5.4. Python内部の責務境界

- `notion_controlplane.py`: Notion state取得・更新・更新後確認。
- `git_state.py`: artifact commit / blob / commit graph事実取得。
- `review_state.py`: `review.schema.json` 準拠のReview結果JSONからReview事実を取得。
- `reconcile.py`: I/Oなしで同期可否とmutation planを決定。
- `sync_controlplane.py`: 上記を組み立て、mutation適用と最終結果返却をオーケストレーションする。

# 6. SHA関係の解釈

- `exact`: 対象版一致。
- control plane SHAがReview対象SHAのancestor: control plane更新漏れ等を疑う。
- Review対象SHAがcontrol plane SHAのancestor: stale Reviewを疑う。
- `diverged`: 自動修復せず調査対象。
- Review成果物自身のcommitをcanonical artifactの `post-SHA` として扱わない。

# 7. Review結果の保存先

Review結果JSONは以下を事実源とする。

```text
reviews/10_each_lore/<Entry_ID>/
└─ Review_<Entry_ID>_<Artifact>_<Review_Seq>.json
```

- Markdown derived viewは同期事実源として使用しない。
- 「最新Review」はmtimeではなくReview Seqを基準とする。

# 8. Workflowからの呼出し

- Workflow 10は作業開始前、canonical artifact commit後、Review修正後、完了時に必要に応じWorkflow 90を呼ぶ。
- Workflow 20はcontrol planeを直接更新せず、必要な整合確認をWorkflow 90へ委譲する。

# 9. 不変条件

- current stateはNotionに保持する。
- Notion stateはGit / Review事実より優先しない。
- 同期はidempotentであること。
- 不明状態を推測で正常化しないこと。
- duplicate、malformed Review、divergence、concurrent update等の曖昧状態では自動更新しないこと。
- control planeからEvidence内容やD01〜D21妥当性を推定しないこと。

# 10. 未確定事項

- `Status` の最終遷移表。
- Review Seq採番・Review JSON保存・Verdict集約を担うPython実装ファイルとの境界。
