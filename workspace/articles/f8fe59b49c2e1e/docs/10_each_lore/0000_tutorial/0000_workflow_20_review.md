# 0. ドキュメント情報

## 0.1. 位置付け

本書は、canonical artifactに対する独立意味論Reviewの正本である。Workflow 10とは独立して起動され、Review cycleの実行と保存を担う。

## 0.2. 設計原則

- Reviewerはcanonical artifactを直接修正しない。
- ReviewerはNotion control planeを直接更新しない。
- Schema / deterministic invariantの検査はPythonへ委譲し、Reviewは意味論に集中する。
- Review対象版は開始時にcommit SHA / blob SHAで固定する。
- 過去Review・旧Excel・既存コードを正解としてReviewしない。
- Review結果の正本はGit上のJSONとする。
- canonical JSONの構造は**Schema上のフィールド監査母集団**を定める。一方、semantic completenessの監査母集団は既存JSON要素に限定せず、Source / Evidence / Content間で保持されるべきsalient meaningを含む。
- Review JSONのartifact固有Schemaに定義された必須check・監査証跡を省略しない。
- Workflow 20の実行契機は、ユーザーによる明示実行またはWorkflow 00からの呼出しに限定する。Workflow 10から暗黙に起動しない。
- Reviewタスクの実行中Statusは初回Review / re-reviewとも `レビュー中` に統一し、その遷移はWorkflow 90の明示的 `review_started` eventでのみ行う。

# 1. 目的と責務

## 1.1. 目的

`00_sources.json → 10_contents.json → 20_analysis.json` の連鎖について、EvidenceからContentへの再構成とContentからAnalysisへの判断が意味論的に妥当かを独立に評価する。

## 1.2. 担う責務

- Source / Evidenceの品質・関係・最古確認・不確実性を確認する。
- EvidenceがContentを支持しているか確認する。
- summaryの情報保存性・再構成可能性を確認する。
- 初期形、後代異伝、最古確認、起源の区別を確認する。
- Version Scopeの妥当性を確認する。
- `macro_category / entry_type` がEntry内容と整合するか確認する。
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
- Review JSONはartifactに応じて次のSchemaに準拠する。

```text
Review_<Entry_ID>_00_<Review_Seq>.json -> schemas/review_00_sources.schema.json
Review_<Entry_ID>_10_<Review_Seq>.json -> schemas/review_10_contents.schema.json
Review_<Entry_ID>_20_<Review_Seq>.json -> schemas/review_20_analysis.schema.json
```

- `schemas/review_common.schema.json` は共通 `$defs` のみを提供し、単独のReview validation targetにはしない。
- Review JSONからMarkdown viewを生成する処理はReview cycleの必須工程に含めない。
- 人から明示的な変換指示があった場合のみ `src/rendering/render_review.py` を実行する。
- 生成Markdownはderived viewであり、Review完了条件・Verdict・control plane状態には影響しない。
- 旧Review JSONを上書きせずappend-onlyで追加する。
- Review Seq採番、対象SHA / blob固定、Verdict集約、save-time Schema validation、append-only書込は `src/reviewing/review_writer.py` に委譲する。
- Reviewerが作成するのはartifact固有のsemantic bodyのみとし、`schema_version / entry_id / artifact / review_seq / target / verdict` はPythonが付与する。
- 新規Review cycleは `schema_version=1.1` とする。Schemaは既存 `1.0` Reviewのread compatibilityを維持し、`1.1` では semantic completeness gateを追加必須とする。

## 2.3. 実行契機

Workflow 20を開始してよいのは次のいずれかの場合だけとする。

1. ユーザーがWorkflow 20の単独実行を明示的に指示した場合。
2. Workflow 00がE2E状態分類に基づきWorkflow 20の実行を決定し、呼び出した場合。

- Workflow 10単独実行がReview-readyへ到達したこと自体は、Workflow 20の実行契機ではない。
- Workflow 20はFinding確定後にWorkflow 10を自動起動しない。Workflow 00配下のE2E実行では制御をWorkflow 00へ戻し、correction開始と再Review順序はWorkflow 00が決定する。

# 3. Review開始条件

- 対象canonical artifactをGit上で一意に固定できること。
- `python -m src.validation.validate_entry <Entry_ID>` がPASSしていること。
- 最初にWorkflow 90を**通常同期**として実行し、Review対象版を阻害するcontrol plane不整合がないことを確認する。この同期では `review_started` を渡さない。
- Review開始時に内部処理として `python -m src.reviewing.review_writer prepare <Entry_ID>` を実行し、00 / 10 / 20の対象commit SHA / blob SHAと次の共有 `Review_Seq` をcycle snapshotとして固定する。
- prepare時点でcanonical artifactに未commit差分がある、既存Review履歴がSchema不正・不完全cycleである、対象SHA / blobを固定できない場合はfail-stopとする。
- **prepare成功後、semantic Reviewへ入る直前**にWorkflow 90へ `review_started: [00, 10, 20]` を渡す。
  - optional external adapterでは各artifactについて `--review-started <Artifact>` を反復指定する。
  - Workflow 90が `BLOCKED / ERROR` の場合、semantic Reviewへ進まない。
  - Workflow 90が `PASS / UPDATED` の場合、00 / 10 / 20のReview対象行がすべて `Status=レビュー中` であることを確認する。
- 初回Reviewでは通常 `レビュー待 -> レビュー中`、修正後re-reviewでは `再レビュー待 -> レビュー中` となる。
- 同一Review cycleは00 / 10 / 20の完全な3点セットであるため、修正されなかったartifactが直前Review `Pass` により `完了` のままでも、current版と直前Review targetがexact一致する場合は `完了 -> レビュー中` を許容する。
- semantic Review中はprepareで得たcycle snapshotを保持し、対象版を後から読み替えない。
- Review中にcanonical artifactが更新された場合はtarget freeze違反であり、Workflow 90の `artifact_changed_during_review` 等によりfail-stopする。

## 3.1. Review実行順序

Blind Decodeへの情報漏洩とlayer bypassを避けるため、同一cycleは原則として次の順で実施する。

1. `prepare` 後、`10_contents.json` の `summary.narrative + summary.structure` **だけ**でReview 10 Blind Decodeを作成し、その記述を固定する。
2. Review 00でSource → Evidenceの妥当性とsalient information completenessを監査する。
3. Review 10でEvidence → Contentのcoverage population completenessを監査し、その後に `coverage_refs` 集合内部のLOSS監査、Reference Story、P01〜P12、Analysis Invarianceを行う。
4. Review 20で**Scoped Contentだけを意味論的根拠**としてContent → Analysisを監査する。

後段でSource / Evidenceから新しい事実を発見しても、Blind Decodeを遡及修正しない。上流layerのomissionとしてFinding化する。


# 4. Review 10: Summary Reconstruction / Coverage Population Review

## 4.1. Blind Decode

- `10_contents.json` の `summary.narrative + summary.structure` だけを入力として、**伝承そのものの具体的展開**を再構成する。
- Blind Decode中は `summary.coverage_refs` のID列、`content_units / variants / uncertainties`、Evidence、Sourceを参照しない。ID列や既知の伝承知識からReference Storyを補ってはならない。
- 再構成はAnalysis用抽象概念に縮約せず、「誰／何に、具体的に何が起き、どう変化し、どう終わるか」を記述する。
- transformation、identity change、terminal scene、motif echoがsummaryに存在する場合は、具体的意味として再構成する。
- Blind Decodeは後続監査を読む前に固定し、後から修正しない。

## 4.2. Coverage Population / Salient Content Completeness

Blind Decode固定後、`00_sources.json` のEvidenceと `10_contents.json` のContent層を比較し、**coverage_refsに入った集合の正しさではなく、Content母集団そのものの完全性**を監査する。

- `summary.coverage_refs` や既存 `content_units` の個数を監査母集団の上限としてはならない。
- 母集団は、Version Scope内のEvidenceに存在し、伝承の再構成・variant識別・不確実性保持・後続Analysisの意味論に必要なsalient informationとする。
- 少なくとも次をsalient候補として確認する。
  - 人物・主体の役割、対象
  - 具体的イベント列・順序・反復・遅延・継続
  - 識別に必要な名称・形態・固有の物
  - 発動・作用条件、規則、禁忌
  - 具体的作用・帰結・terminal state
  - 回避・対処・封印・管理・利用
  - 未解決終端・所在不明・不確実性
  - 誰が何を知る／秘匿する等の情報保持構造
  - 地理アンカー
  - variantを分けるdetail、成立時点と後続付加の境界
- Evidence上salientなのに対応するContent Unit、variant、uncertainty等が存在しない場合、**summary LOSSではなくEvidence → Contentの抽出不足**として `checks.salient_content_completeness=FINDING` とし、`salient_content_completeness` Findingを作成する。
- `coverage_audit` の全件が `NONE` でも、Content母集団に欠落があればReview 10全体はPassにしない。
- genericな上位概念（例: 「危害」「変容」「管理」「情報統制」）は具体情報の併記には使えるが、上記salient detailを置換してはならない。
- Source上のsalient information自体がEvidence化されていない場合はReview 10でSourceから直接補完せず、Review 00の `salient_evidence_completeness` Findingとして扱う。
- 同一年・成立直後の資料でも、原投稿、追加報告・追加聴取、参加者推論、後続再話・検証は、意味・時系列・確度が異なる場合は別の情報層として区別する。

## 4.3. Salient Coverage Audit

- Population completeness監査後に `summary.coverage_refs` を読む。
- 各 `coverage_ref` が指すContent unitを1件ずつ確認し、そのsalient meaningがBlind Decodeから復元できたかを監査する。
- 各refについて `content_ref / salient_meaning / blind_reconstruction / difference` を記録する。
- `difference` は `NONE / LOSS / CONFLICT` とする。
- **Analysisを再構成できるかどうかとは独立に判定する。** 分析コードが同じでも、具体的終端、transformation、identity、motif echo、variantを決める出来事がsummaryから失われていれば `LOSS` とする。
- Evidence上の留保がsummaryで強められた場合は `CONFLICT` とする。
- `coverage_refs` の全件が監査対象であり、人手で別のアンカー集合を後から選んで代替してはならない。
- 本監査は**集合内部のsummary LOSS**を扱う。4.2の**集合母集団そのものの欠落**とは別判定とする。

## 4.4. Reference Story

- Coverage Audit後に `content_units / variants / uncertainties` を参照し、基準となるReference Storyを再構成する。
- Reference Storyは現行Content layerで表現されるsalient meaningを含み、必要な不確実性・variant境界も保持する。
- Evidence / Sourceにしか存在しないdetailをReference Storyへ直接再導入してContent欠落を隠してはならない。必要なら4.2のFindingとして上流修正へ戻す。

## 4.5. Structural Probe

少なくとも以下をP01〜P12として全件確認する。

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

## 4.6. 二層判定

Review 10では次を独立判定する。

1. **Narrative Reconstruction** — summaryだけから現行Content layerの具体的展開を再構成できるか。
2. **Analysis Reconstruction / Invariance** — summaryから後続Analysisに必要な構造を再構成できるか。

- Narrative Reconstructionでsalient unitに `LOSS / CONFLICT` がある場合、Analysis ReconstructionがPASSでもsummary reconstructionはPASSにしない。
- 「重大な異変」「危害」「変容」等の抽象語だけでAnalysis codeを再構成できても、Contentで重要な具体的終端・identity変化・怪異との類似が消えている場合はFindingとする。
- 4.2のpopulation completenessはこの二層判定より上流のgateであり、ここがFINDINGならcoverage集合内部が完全でもReview 10全体はPassにならない。

## 4.7. 比較と保存

- Blind DecodeとReference Storyを比較する。
- `Review_*_10_*.json` には `blind_decode / reference_story / probes[P01-P12] / analysis_invariance / coverage_audit / verdict` を保存する。
- schema v1.1では `checks.salient_content_completeness` を必須とし、notesに監査したEvidence母集団と欠落有無の根拠を残す。
- `checks.narrative_reconstruction` は具体的伝承再構成、`checks.summary_reconstruction` はsummary全体の再構成性、`checks.structural_probe` はP01〜P12の状態を表す。
- `summary.coverage_refs` を持つ対象では、`coverage_audit` はcoverage refsと完全一致しなければならない。
- `coverage_audit` に `LOSS / CONFLICT` が1件でもある場合、`reconstruction.verdict` は `FINDING` とし、対応Findingを残す。
- `reconstruction.verdict` はsummary reconstruction内部の判定である。population completeness Findingがある場合、`coverage_audit` が全件 `NONE` で `reconstruction.verdict=PASS` でも、Finding集合から算出されるReview 10全体のVerdictはPassにならない。


# 5. Review 00 / 10: Evidence・Content Review

## 5.1. Review 00: Source → Evidence

- Source種別・資料同定・参照位置が妥当か。
- Sourceと既存Evidenceの対応、quote / paraphraseの扱いが妥当か。
- 一次資料との関係を誤認していないか。
- 「最古確認」と「実際の起源」を混同していないか。
- 一次未発見、二次のみ、競合、後代付加可能性、確認不能等の不確実性を保持しているか。
- **既存Evidenceの正しさだけでなく、Source上のsalient informationがEvidence層から脱落していないかを監査する。**
- Source上salientなのにEvidence化されていない情報がある場合、schema v1.1の `checks.salient_evidence_completeness=FINDING` とし、`salient_evidence_completeness` Findingを作成する。
- salientnessは4.2の候補群を用い、伝承の具体的再構成、variant境界、不確実性、後続Content / Analysisの意味を変えうるかで判断する。
- 原投稿、追加報告・追加聴取、参加者推論、後続再話・検証は、同時期資料であっても出所と認識論的地位が異なる場合に一括化しない。

## 5.2. Review 10: Evidence → Content

- Content主張が参照Evidenceによって実際に支持されるか。
- 4.2に従い、Evidence上salientな情報のContent化漏れを独立監査する。
- 後代設定を初期形へ遡及していないか。
- 不明・資料間競合・確認不能を不当に閉じていないか。
- Content層へAnalysis判断が混入していないか。
- variant境界がEvidenceと整合するか。
- Review 00で見つかったSource → Evidence omissionを、Review 10がSource直読で補って「Contentは十分」と判定してはならない。上流修正後に再評価する。


# 6. Review 20: Content → Analysis Review

## 6.1. 基本Review

- `macro_category / entry_type` がEntryのContentと意味論的に整合するか。
- Version ScopeがScoped Contentに照らして妥当か。
- D01〜D21の判定がcoding rulesとScoped Contentに整合するか。
- rationaleが参照Contentを実際に支持しているか。
- manifestationがコード定義の言い換えではなくEntry固有のContent記述か。
- `D / I / U / NA / C` のstatus理由がScoped Contentの支持強度に即しているか。
- taxonomy gapを不適切に既存コードへ押し込めていないか。

## 6.2. Content Layer Bypass禁止

- Review 20のAnalysis正当化に利用できる意味論的事実は、**Version Scopeに属するContent layerで表現されていること**を必須とする。
- Source / Evidenceに存在してもScoped Contentに存在しない具体情報を、rationale、manifestation、status、Version Scope、sense-making model、Dimension判定へ直接再導入してはならない。
- Evidence参照は、Contentの支持強度・不確実性・上流omissionを監査する補助証跡としては利用できるが、Contentにない意味を新規に供給する根拠としては利用しない。
- Analysisを正当化するためにSource / Evidence固有detailが必要だと判明した場合、Review 10のomission / semantic traceability Findingとして上流へ戻し、Content修正後にReview 20を再評価する。
- schema v1.1では `checks.content_layer_bypass` を必須とする。bypassを検出した場合は `content_layer_bypass` Findingを作成し、Review 20をPassにしない。

## 6.3. 必須Critical checks

以下は一般的なDimension Reviewへ埋没させず、`Review_*_20_*.json` の個別必須checkとして全件記録する。

- `content_layer_bypass`: Source / Evidence-only情報をScoped Contentを飛ばしてAnalysis正当化へ使用していないか。
- `d12_d13_d15_causal_chain`: D12作用対象 → D13作用・機序 → D15帰結の因果列がContentと整合するか。
- `d18_scope_independent_x_evidence`: D18でL3相当を選ぶ場合、Version Scope内に独立したX Evidenceがあり、かつ必要な意味がContentへ反映されているか。
- `d19_publicability_vs_national_distribution`: 公開可能性・公開媒体の存在を全国流通と誤同一視していないか。
- `d20_missing_info_vs_no_one_knows`: 情報が確認できないことを `NO_ONE_KNOWS` と誤同一視していないか。
- `d21_reality_anchor`: Evidence資料の実在性とScoped Content内の現実アンカーを混同していないか。
- `evidence_confidence_ceiling`: Analysisの確信度がEvidence / Contentの支持強度を超えていないか。
- `entry_classification`: `macro_category / entry_type` がEntry内容と整合するか。

## 6.4. Sense-making再構成

- 20_analysisの判定を読むだけで済ませず、Reviewer自身が**Scoped Contentから**意味形成・因果モデルを再構成する。
- `Review_*_20_*.json` の必須 `sensemaking_reconstruction` に、`model / content_refs / dimension_refs / status` を保存する。
- `evidence_refs` を補助証跡として持たせる場合も、そのEvidenceにしか存在しないdetailをmodelへ追加してはならない。Content不足なら `content_layer_bypass` / Review 10 omissionとしてFinding化する。
- 再構成結果は新たなcanonical Analysisではなく、20_analysisの意味論を検証した監査証跡である。


# 7. Analysis Invarianceと判定順序

- Analysis Invarianceは、**完全性監査済みのScoped Content**を基準に、Contentを読む場合とsummaryを読む場合で主要分析可能性が実質的に変化していないかを確認する。
- Source / Evidenceにのみ存在する情報をsummary側・Analysis側へ補ってInvarianceを成立させてはならない。そこに必要情報がある場合は、先にReview 00 / 10のomissionとして扱う。
- 少なくともD07〜D10、D11、D12、D13、D14、D15、D16、D17、D18、D19〜D20、D21、Version Scope / variant boundaryへの影響を確認する。
- Review 10の `reconstruction.analysis_invariance` とReview 20の `checks.analysis_invariance` を監査証跡として残す。

判定順序と責務は次の通りとする。

1. **Review 00 semantic completeness**: Source → Evidenceのsalient omission。
2. **Review 10 population completeness**: Evidence → Contentのsalient omission。
3. **Review 10 coverage audit / Blind Decode**: 既存coverage_refs集合内部のsummary LOSS / CONFLICT。
4. **P01〜P12 / Narrative Reconstruction**: Contentに存在する具体意味がsummaryで再構成可能か。
5. **Analysis Invariance**: summaryがContent由来の分析可能性を保持するか。
6. **Review 20 Content → Analysis**: Scoped ContentだけでAnalysisが正当化され、layer bypassがないか。

上流gateがFINDINGでも後続監査を実施して追加Findingを得ることはできるが、後続層が上流欠落を直接補完してPassへ戻してはならない。

# 8. FindingとVerdict

## 8.1. Finding

- Findingは対象artifactのReview Schemaに従う。
- Review 00は `review_00_sources.schema.json`、Review 10は `review_10_contents.schema.json`、Review 20は `review_20_analysis.schema.json` を正とする。
- 各Findingは対象箇所、根拠、影響、修正方向を持つ。
- Finding severityは `Minor / Moderate / Major` とする。
- checks / reconstructionの全内容をFindingへ複製せず、修正が必要な単位だけをFinding化する。

## 8.2. Verdict

VerdictはFinding集合から決定論的に再計算可能でなければならない。

- Findingなし: `Pass`
- Minorのみ: `Minor`
- Moderateが1件以上かつMajorなし: `Moderate`
- Majorが1件以上: `Major`

ReviewerはFindingの意味論的内容を記述し、独自ルールでVerdictを手計算しない。

# 9. Review保存と再Review

- Review開始時はSection 3に従って `review_writer prepare` 後に `review_started: [00, 10, 20]` を発行し、Review対象3artifactを `レビュー中` へ遷移させる。
- 00 / 10 / 20のsemantic Reviewが完了したら、prepareで得た `cycle` と3 artifactのsemantic bodyを1つのJSON bundleとして `python -m src.reviewing.review_writer write <Entry_ID>` のstdinへ渡す。
- writerは保存直前に、Entry validation PASS、次Seq、対象commit/blob、working tree一致を再確認する。prepare後にartifact更新または別Review cycleの追加があれば保存せずERRORとする。
- writerは3 artifactを同一 `Review_Seq` で一括validationし、3ファイルすべてが有効な場合だけappend-onlyで保存する。
- 保存後、当該3 Review JSONだけを1つのReview cycle commitとしてcommit / pushする。
- push後にWorkflow 90を**通常同期**として実行する。`review_started` は再送しない。
- 新しいReview JSONがcurrent targetとexact一致した時点で、`レビュー中` は各artifactのVerdictに従い `完了` または `要修正` へ収束する。
- 修正後の再Reviewでは再度prepareを行い、新しい対象SHA / blobと新Review Seqを固定し、同じ `review_started` 手順を反復する。
- Review JSON作成・再Review時にMarkdownを自動生成しない。
- Markdown viewが必要な場合は、人からの明示指示を受けて `render_review.py` を個別実行する。
- 旧Reviewを上書きしない。
- stale Review判定・control plane収束はWorkflow 90 / `sync_controlplane.py` へ委譲する。
- Workflow 20から `notion_controlplane.py` / `git_state.py` / `review_state.py` / `reconcile.py` を直接呼び出さず、Review write側のdeterministic処理は `review_writer.py` の公開入口だけを使用する。

# 10. Review結果の監査不変条件

- Review JSONは特定のcanonical artifact commit SHA / blob SHAを対象とする。
- Review結果ファイル自身を追加したcommit SHAを対象artifactのSHAとして扱わない。
- 同一 `(Entry_ID, Artifact, Review_Seq)` を上書きしない。
- Review本文・Finding・Verdictの正本はJSONであり、derived Markdownではない。
- artifactとReview Schemaの対応を取り違えない。
- Review Schemaが必須化したcheck / reconstructionを省略してPass扱いにしない。
- schema v1.1の `salient_evidence_completeness / salient_content_completeness / content_layer_bypass` は独立gateであり、対応checkが `FINDING` の場合は対応Findingを必須とし、Review全体をPassにしない。
- `coverage_audit` 全件 `NONE` はcoverage集合内部の保存性しか証明せず、coverage population completenessの代替証拠にはならない。

# 11. deterministic Review write実装

- 実装: `src/reviewing/review_writer.py`
- prepare: `python -m src.reviewing.review_writer prepare <Entry_ID>`
- write: `python -m src.reviewing.review_writer write <Entry_ID>`。入力bundleはstdinから受ける。
- Review Seqは既存canonical Review JSONの全artifact共通最大Seq + 1とする。
- 既存cycleが00 / 10 / 20のexact setでない場合、新規Seqを採番せずfail-stopする。
- writeは00 / 10 / 20のexact setを1単位とし、一部artifactだけを新cycleとして保存しない。
- filenameは `Review_<Entry_ID>_<Artifact>_<Review_Seq:03d>.json` とする。
- 既存destinationが存在する場合は上書きせずERRORとする。
- save-time Schema validationには `src.validation.schema_validator.validate_data` を使用する。
- `src/status_management/review_state.py` は引き続きread-onlyとし、write責務を追加しない。
