# 0. ドキュメント情報

## 0.1. 位置付け

本書はcontrol plane同期・整合確認を一元管理するオーケストレーションの正本である。

## 0.2. 設計原則

- ユーザーからの業務入力は `Entry_ID` のみ。
- control planeのcurrent state保持先はNotion `伝承エントリ調査状況`。
- Git上のcanonical artifactとReview結果JSONを事実源としてNotion stateを収束させる。
- **本番チャット経路のI/OはGitHub / Notion connectorを使用する。**
- **Status収束・fail-stop・mutation planの唯一の決定論的正本は `src/status_management/reconcile.py` とする。**
- connector経路でも `reconcile.py::reconcile_payload()` を実行し、LLMが同じstate transition rulesを文章から再実装しない。
- `sync_controlplane.py` はNotion Public API tokenを持つ外部batch環境向けoptional adapterとし、本番チャット経路の必須入口にはしない。
- Workflow 90自身は研究内容、Review意味論、D01〜D21を判定しない。

# 1. 目的

指定Entryについて、Notion control planeの状態とGit上のcanonical artifact・Review結果JSONの状態を比較し、同期可能な範囲でcurrent stateを正しい状態へ収束させる。

# 2. インターフェイス

## 2.1. 入力

公開入力:

- `Entry_ID`

内部イベント:

- `research_started: [Artifact...]`
  - Workflow 00 / 10が、対象artifactについて初回生成またはReview後修正の**調査タスクへ実遷移**した場合だけ発行する。
  - 初回生成では `未 -> 調査中`、Review後修正では `要修正 -> 調査中` を表す。
  - `Entry_ID`、artifactの存在、Review Findingの存在だけから推測しない。
- `review_started: [Artifact...]`
  - Workflow 20が `review_writer prepare` によりReview targetを固定し、semantic Reviewへ**実遷移**する直前に発行する。
  - 初回Review / re-reviewを共通に `レビュー中` として表す。
  - 同一Review cycleで変更のない既Pass artifactも00 / 10 / 20の3点セットとして再Reviewする場合は対象に含めてよい。

両eventとも `reconcile.py` が適用可能性を決定論的に検査する。同一artifactについてresearchとReviewの開始eventを同時に発行してはならない。

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

`Status` は以下の8値のみを許容する。

| Status | 定義 |
|---|---|
| `未` | 調査タスク未着手 |
| `調査中` | 調査タスク実施中。初回作成とReview後修正を区別せずこの値を使用する |
| `レビュー待` | 初回canonical artifact commit完了、初回Review待ち |
| `レビュー中` | Reviewタスク実施中。初回Review / re-reviewを区別せずこの値を使用する |
| `要修正` | Reviewで修正要求が確定し、次の調査タスク未着手 |
| `再レビュー待` | Review後の調査・修正成果物commit完了、次Review待ち |
| `完了` | 対象artifactのlatest applicable Reviewがcurrent版に対してPass |
| `－（対象外）` | 当該成果物を適用対象外と明示した状態 |

- `再作業中` は廃止し、責務を `調査中` へ統合する。
- Notion `伝承エントリ調査状況.Status` のSelect optionsを機械的許容値の正本とする。
- `reconcile.py` は上記8値以外を受け入れずBLOCKする。
- `－（対象外）` はGit / Review事実だけから自動付与しない。

## 4.2. Status遷移

| イベント | Status | 最新レビュー版 | pre-SHA | post-SHA |
|---|---|---|---|---|
| 初回調査開始 | `調査中` | 変更なし | 変更なし | commit確定まで空欄 |
| 初回Coder成果物commit完了 | `レビュー待` | 変更なし | 編集入力checkpoint | 新成果物commit |
| 初回Review開始 | `レビュー中` | 変更なし | 変更なし | 変更なし |
| Reviewで修正要求確定 | `要修正` | 実施済みReview Seq | 変更なし | 変更なし |
| Review Pass確定 | `完了` | 実施済みReview Seq | 変更なし | 変更なし |
| Review後の調査開始 | `調査中` | 変更なし | `old post-SHA` | commit確定まで`old post-SHA`保持 |
| 修正成果物commit完了 | `再レビュー待` | 変更なし | 修正開始時の値を保持 | 新成果物commit |
| re-review開始 | `レビュー中` | 変更なし | 変更なし | 変更なし |
| re-reviewで修正要求 | `要修正` | 新Review Seq | 変更なし | 変更なし |
| re-review Pass | `完了` | 新Review Seq | 変更なし | 変更なし |

追加規則:

- Workflow 00は開始時のstate classification/finalization後、Workflow 10の実作業へ入る直前に `research_started` を発行する。これにより初回作成でも `未 -> 調査中` を明示する。
- `調査中` では、Review後修正の場合に直前のnon-Pass Reviewがlatest Reviewとして残っていても `要修正` へ巻き戻さない。
- `再レビュー待` では、latest Review targetがcurrent artifactのancestorであることは正常状態であり、stale ReviewとしてBLOCKしない。
- Workflow 20はtarget freeze後、semantic Review開始前に `review_started` を発行する。`レビュー中` では新Review JSONが未保存の間、直前ReviewまたはReview未存在の状態を理由に待機状態へ巻き戻さない。
- 同一Review cycleは00 / 10 / 20の3点セットなので、re-review時に変更されなかった `完了` artifactも、current版と直前Pass Reviewがexact一致する場合は `レビュー中` へ遷移できる。
- `レビュー中` にcanonical artifactが更新された場合はReview target freeze違反としてBLOCKする。
- `完了` 後にReview外でcanonical artifactが更新された場合、そのcurrent版は未Reviewなので `再レビュー待` へ戻す。
- task-start eventの重複実行はidempotentとする。
- static Git / Review factsだけから `調査中` / `レビュー中` を推測して自動付与しない。

# 5. 実行手順

## 5.1. Step 0: control plane snapshot取得

Notion connectorで `Entry_ID` に一致するlogical rowsを取得する。

logical key:

```text
(Entry_ID, 成果物)
```

各 `00 / 10 / 20` についてexact 1 rowを要求する。

取得項目:

- row/page ID
- `Status`
- `最新レビュー版`
- `pre-SHA`
- `post-SHA`
- `remarks`
- 更新競合検知に使えるlast-edited fact

duplicate / missing row / unknown Status / parse不能値はsnapshot issueとして保持し、自動補正しない。

## 5.2. Step 1: Git / Review facts取得

GitHub connectorでcurrent canonical artifactについて次を取得する。

- exists
- latest artifact commit SHA
- artifact blob SHA

Reviewについて:

```text
reviews/10_each_lore/<Entry_ID>/
```

のcanonical Review JSONを読み、

- filename / payload整合
- Review Seq
- target commit SHA
- target blob SHA
- verdict
- artifact別Review Schema

を検査してlatest Review factを作る。

malformed / duplicate / Schema invalid Reviewはsnapshot issueとして扱う。

Markdown Reviewやrendered viewは事実源にしない。

## 5.3. Step 2: SHA relation取得

GitHubのcommit graph事実から各artifactについて必要なrelationを作る。

```text
cp_post:       exact / left_ancestor / right_ancestor / diverged / missing
review_target: exact / left_ancestor / right_ancestor / diverged / missing
```

relationの意味はSection 6に従う。

LLMがcommit時刻や見た目の順序からancestryを推測してはならない。

## 5.4. Step 3: deterministic reconcile実行

connectorで得たfactsをJSON payloadへ正規化し、**current mainの `src/status_management/reconcile.py::reconcile_payload()` をPython execution environmentで実行する。**

payload概念形:

```json
{
  "controlplane": {
    "issues": [],
    "artifacts": {}
  },
  "git": {
    "artifacts": {}
  },
  "review": {
    "issues": [],
    "latest": {}
  },
  "relations": {
    "cp_post:00": "exact",
    "review_target:00": "exact"
  },
  "events": {
    "research_started": [],
    "review_started": []
  }
}
```

重要:

- state transition rulesをWorkflow Markdown側で再実装しない。
- Python環境からrepository moduleを直接importできない場合は、GitHub connectorでcurrent `reconcile.py` を取得して同一sourceを実行してよい。
- current `reconcile.py` を実行できない場合、LLM判断で代替せず `BLOCKED` とする。

reconcile結果:

- `NOOP`
- `UPDATE`
- `BLOCKED`

`BLOCKED` の場合はmutationを行わない。

## 5.5. Step 4: mutation直前の競合確認

`UPDATE` の場合、Notionを書き換える直前にmutation対象rowを再取得する。

Step 0 snapshotと以下が一致することを確認する。

- row/page ID
- Status
- 最新レビュー版
- pre-SHA
- post-SHA
- remarks
- last-edited fact

差分があれば `concurrent_update:<Artifact>` としてBLOCKし、古いmutation planを適用しない。

## 5.6. Step 5: Notion mutation適用

`reconcile_payload()` が返したmutationだけをNotion connectorで適用する。

Workflow 90が独自にStatusやSHAを追加変更してはならない。

## 5.7. Step 6: 更新後verify

mutation対象rowを再取得し、mutation planの全key/valueがexact一致することを確認する。

不一致:

```text
post_update_verification_error
```

として `ERROR`。

## 5.8. Step 7: Workflow 90結果

- reconcile=`NOOP` → `PASS`
- reconcile=`UPDATE` かつmutation + verify成功 → `UPDATED`
- reconcile=`BLOCKED` または競合検知 → `BLOCKED`
- connector / parse / write / verify実行不能 → `ERROR`

`PASS / UPDATED` の場合のみ呼出元Workflowは後続処理へ進める。

## 5.9. optional external adapter

```text
python -m src.status_management.sync_controlplane <Entry_ID>
python -m src.status_management.sync_controlplane <Entry_ID> --research-started <Artifact>
python -m src.status_management.sync_controlplane <Entry_ID> --review-started <Artifact>
```

- `--research-started` は明示的な調査タスク開始eventを渡す内部運用option。
- `--review-started` は明示的なReviewタスク開始eventを渡す内部運用option。
- いずれも `00 / 10 / 20` のみを許容し、複数artifactはoptionを反復する。
- 通常syncでは指定しない。
- 同一artifactへ両optionを同時指定してはならない。

このadapterはNotion Public API tokenを持つbatch / CI / external runtime向けoptional adapterであり、本番チャットWorkflow 00/90の必須経路ではない。adapterも内部では同じ `reconcile.py` を使用し、event適用可否を独自判定しない。

# 6. SHA関係の解釈

- `exact`: 対象版一致。
- control plane `post-SHA` が現artifact SHAのancestor: control planeが古い。事実から一意に更新可能ならpre/postを前進させる。
- Review対象SHAが現artifact SHAのancestor:
  - `再レビュー待`、修正commit直後の `調査中`、またはre-review実施中の `レビュー中` であれば正常。
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

- Workflow 10は通常の事前同期後、対象artifactの調査・作成・修正へ実際に入る直前に `research_started` を付けてWorkflow 90を実行する。
- 初回生成でもReview後修正でも同じeventを使用し、作業中Statusは `調査中` に統一する。
- Workflow 00経由で同じ `research_started` eventが先に反映済みの場合、Workflow 10側の重複eventはidempotentなNOOPとして扱う。
- Workflow 20はReview開始条件を検査し `review_writer prepare` で00 / 10 / 20のtargetをfreezeした後、semantic Reviewを開始する直前に `review_started: [00, 10, 20]` をWorkflow 90へ渡す。
- Workflow 20はcontrol planeを直接更新せず、開始・完了時の同期をWorkflow 90へ委譲する。
- Review JSON write / push後の通常同期で `レビュー中` は新Review結果に応じて `完了` または `要修正` へ収束する。

# 9. 不変条件

- current stateはNotionに保持する。
- Notion stateはGit / Review事実より優先しない。
- Status enumはNotion、Workflow 90、`reconcile.py` で一致させる。
- **state transition / fail-stop / mutation planは `reconcile.py` を唯一の正本とする。**
- connectorはI/O adapterであり、state transition ruleを持たない。
- 同期はidempotentであること。
- 不明状態を推測で正常化しないこと。
- duplicate、malformed Review、divergence、concurrent update等の曖昧状態では自動更新しないこと。
- mutation直前の再読と更新後verifyを省略しないこと。
- control planeからEvidence内容やD01〜D21妥当性を推定しないこと。

# 10. task start events

## 10.1. research start event

`research_started` はWorkflow 10の調査タスクへの**実遷移**を表す。

初回生成:

1. Workflow 00 / 10が対象Entryを確定し、通常Workflow 90同期を完了する。
2. 対象artifactが `未` で、まだcurrent canonical / Review factが存在しないことを確認する。
3. 実際の典拠調査・artifact作成へ入る直前に `research_started` を発行する。
4. Workflow 90成功後、Statusは `調査中` でなければならない。

Review後修正:

1. applicable non-Pass Reviewと修正対象artifactを確定する。
2. current artifact / control plane post-SHA / latest non-Pass Review targetがexact一致する。
3. 実際の再調査・修正へ入る直前に `research_started` を発行する。
4. Workflow 90成功後、Statusは `調査中` でなければならない。

上流変更によるdownstream invalidation等で、直前ReviewがPassの `完了` artifactを実際に再調査・変更する場合も、current artifact / control plane post-SHA / Pass Review targetがexact一致することを条件に、明示 `research_started` で `完了 -> 調査中` として再オープンできる。

stale/diverged Review、artifact更新済み、対象外、未知artifactでは開始eventをBLOCKする。重複eventはidempotentとする。

## 10.2. review start event

`review_started` はWorkflow 20のReviewタスクへの**実遷移**を表す。

1. Workflow 20開始条件とdeterministic validationを満たす。
2. `review_writer prepare <Entry_ID>` により00 / 10 / 20のtarget commit/blobと次Review Seqをfreezeする。
3. semantic Reviewへ入る直前に `review_started: [00, 10, 20]` を発行する。
4. Workflow 90成功後、Review対象3artifactのStatusは `レビュー中` でなければならない。
5. Review write / push後の通常Workflow 90同期で、新Review factに従い `完了` または `要修正` へ収束する。

初回Reviewでは `レビュー待 -> レビュー中`、re-reviewでは `再レビュー待 -> レビュー中` を基本とする。同一3点Review cycleで変更されなかったartifactは、直前Reviewがcurrent版にexact一致するPassであれば `完了 -> レビュー中` を許容する。

## 10.3. 共通不変条件

- task開始は `Entry_ID` や静的なGit / Review状態だけから推測しない。
- 同一artifactに `research_started` と `review_started` を同時に発行しない。
- event適用可否は `reconcile.py` を唯一の決定論的正本とする。
- eventの重複実行はidempotentでなければならない。

