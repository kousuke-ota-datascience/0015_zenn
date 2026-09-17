# 0. ドキュメント情報

## 0.1. 位置付け

本書は、canonical artifactに対する独立意味論Reviewの正本である。

## 0.2. 設計原則

- Reviewerはcanonical artifactを直接修正しない。
- ReviewerはNotion control planeを直接更新しない。
- Schema / deterministic invariantの検査はPythonへ委譲し、Reviewは意味論に集中する。
- Review対象版は開始時にcommit SHA / blob SHAで固定する。
- 過去Review・旧Excel・既存コードを正解としてReviewしない。
- Review結果の正本はGit上のJSONとする。

# 1. 目的と責務

## 1.1. 目的

`00_sources.json → 10_contents.json → 20_analysis.json` の連鎖について、EvidenceからContentへの再構成とContentからAnalysisへの判断が意味論的に妥当かを独立に評価する。

## 1.2. 担う責務

- EvidenceがContentを支持しているか確認する。
- summaryの情報保存性・再構成可能性を確認する。
- 初期形、後代異伝、最古確認、起源の区別を確認する。
- Version Scopeの妥当性を確認する。
- D01〜D21の意味論的妥当性を確認する。
- sense-making / 因果構造の整合を確認する。
- semantic traceabilityを確認する。
- Findingを根拠・影響・修正方向とともに記録する。

## 1.3. 担わない責務

- JSON Schema validation。
- required field / ID重複 / cross-reference / Parent-Child等の機械的検査。
- control planeの同期・修復。
- Review対象artifactの修正。

# 2. インターフェイス

## 2.1. 入力

- 原則 `Entry_ID` のみ。
- Review開始時に対象Entryのcanonical artifactと対象SHA / blobを内部で固定する。

## 2.2. 出力

- Review結果の正本はGit上のJSONとする。
- 保存先は `reviews/10_each_lore/<Entry_ID>/` とする。
- 1成果物につき1 Review結果ファイルを生成する。
- 命名は `Review_<Entry_ID>_<Artifact>_<Review_Seq>.json` とする。
- `Artifact` は `00 / 10 / 20` のいずれか。
- 同一Review cycleでは対象artifact間で同じ `Review_Seq` を共有する。
- Review JSONは `schemas/review.schema.json` に準拠する。
- Review JSONからMarkdown viewを生成する処理はReview cycleの必須工程に含めない。
- 人から明示的な変換指示があった場合のみ `src/rendering/render_review.py` を実行する。
- 生成Markdownはderived viewであり、Review完了条件・Verdict・control plane状態には影響しない。
- 旧Review JSONを上書きせずappend-onlyで追加する。

# 3. Review開始条件

- 対象canonical artifactをGit上で一意に固定できること。
- `python -m src.validation.validate_entry <Entry_ID>` がPASSしていること。
- Workflow 90により、Review対象版を阻害するcontrol plane不整合がないこと。
- 対象SHA / blobを固定できない場合はfail-stopとする。

# 4. Summary Reconstruction Review

## 4.1. Blind Decode

- summaryからReference Storyを見ずに伝承構造を再構成する。

## 4.2. Structural Probe

少なくとも以下を確認する。

- P01 primary actors / roles
- P02 affected targets
- P03 trigger / entry
- P04 cause / agent
- P05 mechanism / action
- P06 state transition
- P07 order / repetition / delay / continuation
- P08 conditions / rules / taboos
- P09 avoidance / control / use
- P10 terminal state
- P11 unresolved / unknown
- P12 initial / later / major variant boundaries

## 4.3. 比較

- Blind DecodeとEvidenceから構成したReference Storyを比較する。
- 差分は `NONE / LOSS / CONFLICT` として扱う。
- LOSSのみを自動Failにせず、後続分析可能性への影響で判断する。

# 5. Evidence → Content Review

- SourceのEvidence roleが妥当か。
- Content主張が参照Evidenceによって実際に支持されるか。
- 後代設定を初期形へ遡及していないか。
- 「最古確認」と「実際の起源」を混同していないか。
- 不明・資料間競合・確認不能を不当に閉じていないか。
- Content層へAnalysis判断が混入していないか。

# 6. Content → Analysis Review

- Version ScopeがEvidence / Contentに照らして妥当か。
- D01〜D21の判定がcoding rulesとContentに整合するか。
- rationaleが参照Contentを実際に支持しているか。
- manifestationがコード定義の言い換えではなくEntry固有の記述か。
- `D / I / U / NA / C` のstatus理由がEvidenceに即しているか。
- taxonomy gapを不適切に既存コードへ押し込めていないか。

# 7. Analysis Invariance

- original Evidenceを読む場合とsummaryを読む場合で、主要分析可能性が実質的に変化していないか確認する。
- 少なくともD07〜D10、D11、D12、D13、D14、D15、D16、D17、D18、D19〜D20、D21、Version Scope / variant boundaryへの影響を確認する。

# 8. FindingとVerdict

## 8.1. Finding

- Findingは `review.schema.json` の構造に従う。
- 各Findingは対象箇所、根拠、影響、修正方向を持つ。
- Finding severityは `Minor / Moderate / Major` とする。

## 8.2. Verdict

VerdictはFinding集合から決定論的に再計算可能でなければならない。

- Findingなし: `Pass`
- Minorのみ: `Minor`
- Moderateが1件以上かつMajorなし: `Moderate`
- Majorが1件以上: `Major`

ReviewerはFindingの意味論的内容を記述し、独自ルールでVerdictを手計算しない。

# 9. 再Review

- 修正後は新しい対象SHA / blobを固定し、新Review SeqでReviewする。
- Review JSON作成・再Review時にMarkdownを自動生成しない。
- Markdown viewが必要な場合は、人からの明示指示を受けて `render_review.py` を個別実行する。
- 旧Reviewを上書きしない。
- stale Review判定・control plane収束はWorkflow 90 / `sync_controlplane.py` へ委譲する。
- Workflow 20から `notion_controlplane.py` / `git_state.py` / `review_state.py` / `reconcile.py` を直接呼び出さない。

# 10. Review結果の監査不変条件

- Review JSONは特定のcanonical artifact commit SHA / blob SHAを対象とする。
- Review結果ファイル自身を追加したcommit SHAを対象artifactのSHAとして扱わない。
- 同一 `(Entry_ID, Artifact, Review_Seq)` を上書きしない。
- Review本文・Finding・Verdictの正本はJSONであり、derived Markdownではない。

# 11. 未確定事項

- Review Seq採番・Review JSON保存・Verdict集約を担うPython実装ファイルの配置。
- `review.schema.json` のFinding `category` 最終enum集合。
- Blind Decodeが `summary.narrative` のみを読むか、`summary.narrative + structure` を読むか。
