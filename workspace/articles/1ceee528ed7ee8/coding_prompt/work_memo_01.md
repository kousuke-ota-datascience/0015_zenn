コーディングエージェント向け実装プロンプト

# 0. INTRODUCTION

あなたは Python / CLI / 因果推論パイプラインの実装改善を担当するコーディングエージェントである。

対象パス:

```text
workspace/articles/1ceee528ed7ee8/
```

今回の改修では、既存構造の後方互換性は維持しない。  
より明確で保守可能なアーキテクチャへ移行する。

採用する package 名は以下とする。

```text
causal_core               # 共通基盤・共通型・軽量 utility
causal_discovery          # 因果探索分析のクラス群
causal_inference          # 因果推論分析のクラス群
causal_pipeline_runtime   # discovery / inference を接続する上位制御層
```

既存の以下 package は廃止対象とする。

```text
common_in_causal_inference
causal_discovery_pipeline
causal_inference_pipeline
```

旧 package 名の thin wrapper は作らない。  
production code / tests / docs は新 package 名へ全面的に移行する。

---

# 1. 目的

以下の各観点について、レビュー評点を最低でも **8 / 10 以上** に引き上げる。

対象観点:

1. CLI パイプライン処理
2. コード構造・保守性
3. YAML 設定管理
4. 特徴量作成処理
5. 統計的処理
6. 因果推論分析の実態
7. 診断出力
8. レポート・解釈
9. テスト・再現性
10. Sphinx / ドキュメント

満たすべき状態:

```text
- CLI パイプラインとして再現可能に動作する
- 因果探索と因果推論を混同しない
- estimand / adjustment set / estimator / diagnostics の対応が明示されている
- YAML config の意味論が明示されている
- discovery と inference の feature semantics 不一致を検出できる
- --validate-only / --dry-run が pipeline-level execution strategy として実装されている
- 共通 utility は causal_core に集約され、runtime / discovery / inference と混在しない
- Unit test / Smoke test / Regression test / Sphinx build で検証されている
```

---

# 2. 重要な設計方針

## 2.A. package 名と責務を明確に分ける

採用する物理構成は以下である。

```text
experiment/
  causal_core/
  causal_pipeline_runtime/
  causal_discovery/
  causal_inference/
```

各 package の責務は以下である。

```text
causal_core
  - 共通基盤
  - 共通例外
  - YAML loader
  - path utility
  - config hashing
  - validation primitive
  - feature semantics schema
  - causal design schema
  - stage 非依存の軽量 utility

causal_pipeline_runtime
  - 上位制御層
  - ExecutionPlan
  - StagePlan
  - pipeline execution strategy
  - stage runner orchestration
  - cross-stage validation
  - artifact registry
  - run manifest
  - dry-run
  - validate-only

causal_discovery
  - 因果探索分析のクラス群
  - PC / GES / LiNGAM / NOTEARS
  - discovery feature construction
  - alpha sensitivity
  - bootstrap edge stability
  - discovery diagnostics
  - discovery report

causal_inference
  - 因果推論分析のクラス群
  - causal design
  - adjustment set validation
  - OLS / g-computation / IPW / AIPW
  - balance / overlap / outcome diagnostics
  - treatment effect report
  - edge weight exploratory report
```

依存方向は以下に限定する。

```text
05_causal_discovery_inference_completejourney.py
  -> causal_pipeline_runtime
      -> causal_discovery.runner
      -> causal_inference.runner

causal_pipeline_runtime -> causal_core
causal_discovery        -> causal_core
causal_inference        -> causal_core
```

禁止する依存方向:

```text
causal_core -> causal_pipeline_runtime
causal_core -> causal_discovery
causal_core -> causal_inference

causal_discovery -> causal_inference
causal_inference -> causal_discovery internals

causal_discovery -> causal_pipeline_runtime internals
causal_inference -> causal_pipeline_runtime internals

common_in_causal_inference -> production runtime
```

`causal_pipeline_runtime` は「共通 utility 置き場」ではない。  
`causal_core` が共通基盤を担い、`causal_pipeline_runtime` は stage 横断の runtime 責務だけを担う。

---

## 2.B. `causal_core` は共通基盤に限定する

`causal_core` に置くもの:

```text
causal_core/
  exceptions.py
    - CausalPipelineError
    - ConfigValidationError
    - FeatureSemanticsError
    - ArtifactSchemaError

  logging.py
    - get_logger
    - configure_logging

  paths.py
    - resolve_project_path
    - ensure_directory
    - relative_to_project

  config/
    __init__.py
    yaml_loader.py
      - load_yaml
      - dump_yaml
      - load_yaml_mapping
    hashing.py
      - hash_file
      - hash_mapping
      - hash_resolved_config

  validation/
    __init__.py
    issues.py
      - ValidationSeverity
      - ValidationIssue
    result.py
      - ValidationResult

  features/
    __init__.py
    semantics.py
      - FeatureRole
      - FeatureSemanticSpec
      - FeatureSemanticsCatalog
      - compare_feature_semantics

  causal_design/
    __init__.py
    schema.py
      - Estimand
      - TreatmentSpec
      - OutcomeSpec
      - CausalDesign
```

`causal_core` に置かないもの:

```text
- ExecutionPlan
- StagePlan
- RunStrategy
- DryRunStrategy
- ValidateOnlyStrategy
- PipelineExecutor
- RunManifest
- PC / GES / LiNGAM / NOTEARS
- OLS / IPW / AIPW
- covariate balance
- overlap diagnostics
- treatment effect report
- discovery report
```

判断基準:

```text
stage 非依存かつ runtime 非依存であれば causal_core
pipeline 実行制御に属するなら causal_pipeline_runtime
discovery 分析に属するなら causal_discovery
inference 分析に属するなら causal_inference
```

---

## 2.C. `--validate-only` / `--dry-run` は causal discovery に入れない

### 2.C.1. 背景

もともと discovery stage には、通常の因果探索アルゴリズムに加えて、以下のような stage-local な診断処理が存在する。

```text
- alpha sensitivity
- bootstrap edge stability
- variable diagnostics
- background knowledge validation
```

これらは discovery algorithm 本体そのものではないが、**causal discovery stage の内部診断**である。  
したがって、discovery stage 内で strategy / mode 的に整理すること自体は妥当である。

一方で、以下は discovery stage 内部の診断ではない。

```text
--validate-only
--dry-run
```

これらは discovery を実行するかどうかではなく、pipeline 全体をどう実行するかを決める **pipeline-level execution mode** である。

### 2.C.2. 正しい分類

```text
discovery stage-local mode / diagnostics
  - alpha sensitivity
  - bootstrap stability
  - variable diagnostics

pipeline execution strategy
  - run
  - validate-only
  - dry-run

inference mode strategy
  - treatment_effect
  - edge_weight
```

### 2.C.3. 禁止事項

以下は禁止する。

```text
causal_discovery 側に統合 pipeline の validate-only / dry-run を実装する
causal_discovery が inference config や causal design を直接検証する
causal_discovery が discovery -> inference の接続責務を持つ
causal_inference/modes に validate_only.py や dry_run.py を作る
```

---

## 2.D. pipeline execution strategy と inference mode strategy を分離する

strategy は 2 種類存在する。

### 2.D.1. Pipeline execution strategy

CLI 実行制御の strategy。

```text
run
validate-only
dry-run
```

配置先:

```text
experiment/causal_pipeline_runtime/strategies.py
```

### 2.D.2. Inference mode strategy

因果推論分析モードの strategy。

```text
treatment_effect
edge_weight
```

配置先:

```text
experiment/causal_inference/modes/
```

以下の混在は禁止する。

```text
causal_inference/modes/
  validate_only.py
  dry_run.py
  treatment_effect.py
  edge_weight.py
```

---

## 2.E. 後方互換性は維持しない

今回の目的は、既存構造を温存することではなく、責務分離を明確にした新アーキテクチャへ移行することである。

方針:

```text
- common_in_causal_inference は廃止対象
- causal_discovery_pipeline は causal_discovery へ rename / 移行
- causal_inference_pipeline は causal_inference へ rename / 移行
- causal_core を新設
- causal_pipeline_runtime を新設
- 旧 package の wrapper は作らない
- 旧 API と新 API の二重保守はしない
- production code / tests / docs は新 package 名に全面移行する
```

禁止事項:

```text
- common_in_causal_inference の backward-compatible wrapper を作る
- causal_discovery_pipeline / causal_inference_pipeline の alias package を残す
- 新旧 import を併存させる
- 旧 API を前提とした tests を残す
```

---

# 3. 新アーキテクチャ

## 3.1. 採用する package 構成

```text
experiment/
  causal_core/
    __init__.py
    exceptions.py
    logging.py
    paths.py
    config/
      __init__.py
      yaml_loader.py
      hashing.py
    validation/
      __init__.py
      issues.py
      result.py
    features/
      __init__.py
      semantics.py
    causal_design/
      __init__.py
      schema.py

  causal_pipeline_runtime/
    __init__.py
    orchestration.py
    planning.py
    strategies.py
    execution.py
    validation.py
    artifacts.py

  causal_discovery/
    __init__.py
    cli.py
    runner.py
    config.py
    features/
    algorithms/
    diagnostics/
    reporting/

  causal_inference/
    __init__.py
    cli.py
    runner.py
    config.py
    context.py
    modes/
    estimation/
    diagnostics/
    features/
    reporting/
```

---

## 3.2. `causal_core` の責務

```text
causal_core/
  exceptions.py
    - 共通例外

  logging.py
    - logging utility

  paths.py
    - path resolution

  config/yaml_loader.py
    - YAML load / dump

  config/hashing.py
    - config hash / file hash

  validation/issues.py
    - ValidationIssue
    - ValidationSeverity

  validation/result.py
    - ValidationResult

  features/semantics.py
    - FeatureSemanticSpec
    - FeatureSemanticsCatalog
    - FeatureRole
    - compare_feature_semantics

  causal_design/schema.py
    - CausalDesign
    - TreatmentSpec
    - OutcomeSpec
    - Estimand
```

`causal_core` は他の project package を import しない。  
標準ライブラリ、pandas / numpy などの基盤ライブラリ、typing / dataclasses に依存することは許容する。

---

## 3.3. `causal_pipeline_runtime` の責務

```text
causal_pipeline_runtime/
  planning.py
    - ExecutionPlan
    - StagePlan
    - PipelinePlanner

  strategies.py
    - PipelineCommandStrategy
    - RunStrategy
    - ValidateOnlyStrategy
    - DryRunStrategy

  execution.py
    - StageRunner Protocol
    - DiscoveryStageRunner adapter
    - InferenceStageRunner adapter
    - PipelineExecutor
    - StageResult

  validation.py
    - CrossStageValidator
    - feature semantics consistency check
    - discovery manifest validation
    - causal design presence check
    - adjustment set validation の pipeline-level 呼び出し

  artifacts.py
    - RunManifest
    - ArtifactRegistry
    - ArtifactSpec

  orchestration.py
    - thin facade
    - planner + strategy + executor の接続のみ
```

`orchestration.py` に validate-only / dry-run / run の詳細処理を直書きしない。

---

## 3.4. `causal_discovery` の責務

```text
causal_discovery/
  cli.py
    - discovery 単体 CLI

  runner.py
    - discovery stage-local execution
    - discovery config loading
    - discovery artifacts output

  config.py
    - discovery config schema

  features/
    - discovery feature construction

  algorithms/
    - PC
    - GES
    - LiNGAM
    - NOTEARS

  diagnostics/
    - alpha sensitivity
    - bootstrap edge stability
    - variable diagnostics

  reporting/
    - discovery report
```

`causal_discovery` は `causal_inference` の config / causal design / adjustment set を直接検証しない。

---

## 3.5. `causal_inference` の責務

```text
causal_inference/
  cli.py
    - inference 単体 CLI

  runner.py
    - inference stage-local execution

  config.py
    - inference config schema
    - causal design config schema

  context.py
    - InferenceContext

  modes/
    - TreatmentEffectModeStrategy
    - EdgeWeightModeStrategy

  estimation/
    - OLS coefficient
    - g-computation
    - IPW
    - AIPW
    - multiplicity correction

  diagnostics/
    - design diagnostics
    - covariate balance
    - overlap
    - outcome diagnostics
    - model diagnostics

  features/
    - feature config
    - aggregation registry
    - transform registry
    - encoding registry

  reporting/
    - inference report
```

---

# 4. レビュー評価の採点基準

## 4.1. 共通採点ルール

各観点は 10 点満点で評価する。  
最低目標は **8 / 10 以上** とする。

点数の意味:

| 点数 | 状態 |
|---:|---|
| 0-3 | 動作・設計・検証のいずれかが根本的に不足している |
| 4-5 | 最低限動くが、責務分離・再現性・検証が弱い |
| 6-7 | 実用的な部分はあるが、誤解や拡張困難性が残る |
| 8 | 明確な責務分離、主要 edge case 対応、test/docs が揃う |
| 9 | 8 の条件に加え、拡張性・診断・失敗時説明が強い |
| 10 | production-grade。API 安定性、網羅的検証、運用性まで高い |

今回の完了条件は **各観点 8 点以上** であり、10 点満点を要求するものではない。

---

## 4.2. 観点別 8 / 10 達成条件

| 観点 | 8 / 10 の最低条件 |
|---|---|
| 1. CLI パイプライン処理 | `ExecutionPlan`, `PipelineCommandStrategy`, `StageRunner`, manifest, `--validate-only`, `--dry-run` が実装され、smoke test が通る |
| 2. コード構造・保守性 | `causal_core`, `causal_pipeline_runtime`, `causal_discovery`, `causal_inference` の責務が明確で、旧 package import が production path から消えている |
| 3. YAML 設定管理 | resolved config 保存、config hash、pipeline config、feature semantics、causal design が schema 化される |
| 4. 特徴量作成処理 | aggregation / transform / encoding registry があり、discovery / inference の feature semantics consistency を検証できる |
| 5. 統計的処理 | OLS coefficient と ATE/ATT estimator が分離され、IPW/AIPW の診断・SE・edge case test がある |
| 6. 因果推論分析の実態 | estimand, treatment, outcome, adjustment set, estimator support が明示され、unsupported combination が fail-fast する |
| 7. 診断出力 | design / balance / overlap / outcome / model diagnostics が CSV/Markdown に出る |
| 8. レポート・解釈 | report が estimand, design, diagnostics, warnings, limitations, reproducibility を含み、ATE/ATT/edge weight を混同しない |
| 9. テスト・再現性 | unit / smoke / regression test があり、fixed seed と golden file または tolerance 比較がある |
| 10. Sphinx / ドキュメント | public API docs、methodology docs、`sphinx-build -W` が通る |

---

## 4.3. 8 / 10 未満と判定する条件

以下のいずれかが残る場合、該当観点は 8 / 10 未満と判定する。

```text
- --validate-only / --dry-run が argparse 分岐に直書きされている
- orchestration.py が巨大化している
- causal_pipeline_runtime が共通 utility 置き場になっている
- causal_core が runtime / discovery / inference を import している
- causal_discovery が causal_inference config を直接知っている
- OLS coefficient を ATE / ATT として表示している
- discovery edge weight を識別済み causal effect として表示している
- feature semantics mismatch が warning のみで実行継続する
- adjustment set に post-treatment variable が入っても error にならない
- CLI override が silently ignored される
- manifest なしに discovery artifact path を暗黙参照する
- Sphinx build warning を放置する
- tests が新 architecture の責務分離を検証していない
```

---

# 5. 主要クラス設計

## 5.1. causal_core 共通型

修正対象:

```text
experiment/causal_core/exceptions.py
experiment/causal_core/config/yaml_loader.py
experiment/causal_core/config/hashing.py
experiment/causal_core/validation/issues.py
experiment/causal_core/validation/result.py
experiment/causal_core/features/semantics.py
experiment/causal_core/causal_design/schema.py
experiment/tests/test_core.py
```

実装例:

```python
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str
    location: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ValidationResult:
    issues: list[ValidationIssue]

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)


class FeatureRole(str, Enum):
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    COVARIATE = "covariate"
    MEDIATOR = "mediator"
    COLLIDER = "collider"
    POST_TREATMENT = "post_treatment"


@dataclass(frozen=True)
class FeatureSemanticSpec:
    name: str
    role: FeatureRole
    source_table: str
    source_column: str | None
    unit_id: str
    aggregation: str | None = None
    transform: str | None = None
    dtype: str | None = None
    allowed_for_adjustment: bool = False
    post_treatment: bool = False
```

要件:

```text
- causal_core は runtime / discovery / inference を import しない
- ValidationIssue / ValidationResult は runtime / discovery / inference で共有する
- FeatureSemanticSpec は discovery / inference 間の意味論比較に使う
- CausalDesign は inference と runtime validation の両方で使う
```

---

## 5.2. ExecutionPlan

修正対象:

```text
experiment/causal_pipeline_runtime/planning.py
experiment/causal_pipeline_runtime/artifacts.py
experiment/causal_pipeline_runtime/orchestration.py
experiment/tests/test_planning.py
experiment/tests/test_orchestration.py
experiment/tests/test_manifest.py
```

実装例:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class StagePlan:
    name: Literal["discovery", "inference"]
    enabled: bool
    input_paths: dict[str, Path]
    output_paths: dict[str, Path]
    config_paths: dict[str, Path]
    resolved_args: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExecutionPlan:
    run_id: str
    stages: list[StagePlan]
    resolved_configs: dict[str, Path]
    artifact_registry: "ArtifactRegistry"
    validation_checks: list[str]
    metadata: dict[str, Any]
```

要件:

```text
- CLI args と YAML config から ExecutionPlan を構築する
- dry-run は ExecutionPlan を表示するだけにする
- validate-only は ExecutionPlan に対して validation を走らせるだけにする
- 通常実行は ExecutionPlan に従って stage runner を実行する
- ExecutionPlan は discovery / inference の artifact contract を明示する
- ExecutionPlan に resolved config path と config hash を含める
```

---

## 5.3. PipelineCommandStrategy

修正対象:

```text
experiment/causal_pipeline_runtime/strategies.py
experiment/causal_pipeline_runtime/planning.py
experiment/causal_pipeline_runtime/execution.py
experiment/causal_pipeline_runtime/orchestration.py
experiment/tests/test_pipeline_strategies.py
experiment/tests/test_orchestration.py
```

実装例:

```python
from typing import Protocol


class PipelineCommandStrategy(Protocol):
    name: str

    def execute(self, plan: ExecutionPlan) -> "PipelineCommandResult":
        ...


class ValidateOnlyStrategy:
    name = "validate_only"

    def execute(self, plan: ExecutionPlan) -> "PipelineCommandResult":
        ...


class DryRunStrategy:
    name = "dry_run"

    def execute(self, plan: ExecutionPlan) -> "PipelineCommandResult":
        ...


class RunStrategy:
    name = "run"

    def execute(self, plan: ExecutionPlan) -> "PipelineCommandResult":
        ...
```

要件:

```text
- --validate-only は ValidateOnlyStrategy を選ぶ
- --dry-run は DryRunStrategy を選ぶ
- 通常実行は RunStrategy を選ぶ
- argparse は strategy を選ぶだけにする
- orchestration.py は strategy に処理を委譲する thin facade にする
```

---

## 5.4. StageRunner

修正対象:

```text
experiment/causal_pipeline_runtime/execution.py
experiment/causal_discovery/runner.py
experiment/causal_inference/runner.py
experiment/causal_discovery/cli.py
experiment/causal_inference/cli.py
experiment/tests/test_stage_runners.py
experiment/tests/test_orchestration.py
experiment/tests/test_smoke_pipeline.py
```

実装例:

```python
from typing import Protocol


class StageRunner(Protocol):
    name: str

    def validate_plan(self, stage_plan: StagePlan) -> list["ValidationIssue"]:
        ...

    def run(self, stage_plan: StagePlan) -> "StageResult":
        ...


class DiscoveryStageRunner:
    name = "discovery"

    def validate_plan(self, stage_plan: StagePlan) -> list["ValidationIssue"]:
        ...

    def run(self, stage_plan: StagePlan) -> "StageResult":
        ...


class InferenceStageRunner:
    name = "inference"

    def validate_plan(self, stage_plan: StagePlan) -> list["ValidationIssue"]:
        ...

    def run(self, stage_plan: StagePlan) -> "StageResult":
        ...
```

要件:

```text
- DiscoveryStageRunner は discovery の責務のみを持つ
- InferenceStageRunner は inference の責務のみを持つ
- cross-stage validation は causal_pipeline_runtime/validation.py に置く
- discovery stage が inference config や causal design を直接検証しない
```

---

# 6. グローバル修正対象ファイル一覧

```text
workspace/articles/1ceee528ed7ee8/experiment/05_causal_discovery_inference_completejourney.py

workspace/articles/1ceee528ed7ee8/experiment/causal_core/
  __init__.py
  exceptions.py
  logging.py
  paths.py
  config/
    __init__.py
    yaml_loader.py
    hashing.py
  validation/
    __init__.py
    issues.py
    result.py
  features/
    __init__.py
    semantics.py
  causal_design/
    __init__.py
    schema.py

workspace/articles/1ceee528ed7ee8/experiment/causal_pipeline_runtime/
  __init__.py
  orchestration.py
  planning.py
  strategies.py
  execution.py
  artifacts.py
  validation.py

workspace/articles/1ceee528ed7ee8/experiment/causal_discovery/
  __init__.py
  cli.py
  runner.py
  config.py
  features/
  algorithms/
  diagnostics/
  reporting/

workspace/articles/1ceee528ed7ee8/experiment/causal_inference/
  __init__.py
  cli.py
  runner.py
  config.py
  context.py
  modes/
    __init__.py
    base.py
    treatment_effect.py
    edge_weight.py
  estimation/
    __init__.py
    base.py
    diff_in_means.py
    ols.py
    g_computation.py
    ipw.py
    aipw.py
    inference.py
    multiplicity.py
    treatment_effect.py
  diagnostics/
    __init__.py
    design.py
    balance.py
    overlap.py
    outcome.py
    model.py
  features/
    __init__.py
    config.py
    builder.py
    aggregation.py
    encoding.py
    transforms.py
  reporting/
    __init__.py
    markdown.py
    tables.py
    warnings.py

workspace/articles/1ceee528ed7ee8/experiment/common_in_causal_inference/
  # 廃止対象
  # production import を全て削除する
  # 可能なら削除する

workspace/articles/1ceee528ed7ee8/experiment/causal_discovery_pipeline/
  # 廃止対象
  # causal_discovery へ移行する
  # production import を全て削除する
  # 可能なら削除する

workspace/articles/1ceee528ed7ee8/experiment/causal_inference_pipeline/
  # 廃止対象
  # causal_inference へ移行する
  # production import を全て削除する
  # 可能なら削除する

workspace/articles/1ceee528ed7ee8/conf/
  causal_discovery/
    features.yaml
    analysis.yaml

  causal_inference/
    causal_inference_default.yaml
    completejourney_household.yaml
    pipeline.yaml
    feature_semantics.yaml
    causal_design.yaml

workspace/articles/1ceee528ed7ee8/experiment/tests/
  test_core.py
  test_causal_inference_pipeline.py
  test_orchestration.py
  test_manifest.py
  test_planning.py
  test_pipeline_strategies.py
  test_stage_runners.py
  test_feature_semantics.py
  test_estimators.py
  test_diagnostics.py
  test_cli_overrides.py
  test_smoke_pipeline.py
  test_regression_estimators.py
  conftest.py
  fixtures/

workspace/articles/1ceee528ed7ee8/experiment/docs/sphinx/
  conf.py
  index.rst
  api.rst
  methodology.rst
  methodology/
    estimands.rst
    identification.rst
    adjustment_sets.rst
    diagnostics.rst
    feature_semantics.rst
    limitations.rst
  reproducibility.rst
```

---

# 7. 実装要求

## 7.1. CLI パイプライン処理

### 7.1.1. 修正対象ファイル / 箇所

```text
experiment/05_causal_discovery_inference_completejourney.py
experiment/causal_pipeline_runtime/planning.py
experiment/causal_pipeline_runtime/strategies.py
experiment/causal_pipeline_runtime/execution.py
experiment/causal_pipeline_runtime/orchestration.py
experiment/causal_pipeline_runtime/artifacts.py
experiment/causal_pipeline_runtime/validation.py
experiment/causal_discovery/runner.py
experiment/causal_inference/runner.py
experiment/causal_discovery/cli.py
experiment/causal_inference/cli.py
experiment/causal_core/config/hashing.py
experiment/tests/test_planning.py
experiment/tests/test_pipeline_strategies.py
experiment/tests/test_stage_runners.py
experiment/tests/test_orchestration.py
experiment/tests/test_manifest.py
experiment/tests/test_smoke_pipeline.py
```

### 7.1.2. 実装要件

```text
- --validate-only / --dry-run は PipelineCommandStrategy として実装する
- orchestration.py に validate-only / dry-run の実処理を直書きしない
- ExecutionPlan を導入する
- DryRunStrategy は ExecutionPlan を表示するだけにする
- ValidateOnlyStrategy は ExecutionPlan に対して validation を実行するだけにする
- RunStrategy は ExecutionPlan に従って stage runner を実行する
```

discovery stage 実行後、少なくとも以下を含む manifest を出力する。

```yaml
run_id: ...
stage: discovery
resolved_output_dir: ...
resolved_analysis_config_path: ...
resolved_feature_config_path: ...
resolved_analysis_config_hash: ...
resolved_feature_config_hash: ...
created_at: ...
random_seed: ...
artifacts:
  adjacency_matrix: ...
  edges: ...
  bootstrap_summary: ...
  resolved_config: ...
  resolved_feature_semantics: ...
```

`causal_inference/cli.py` に `--discovery-manifest` を追加する。  
`--discovery-dir` は廃止する。移行用 alias も不要。

`--validate-only` では以下を検証する。

```text
- config file の存在
- output dir の解決
- discovery output の存在
- discovery manifest の schema
- feature semantics consistency
- causal design の存在
- treatment / outcome / covariates の存在
- adjustment set の妥当性
```

`--dry-run` では以下を表示する。

```text
- run_id
- selected pipeline strategy
- stages
- resolved config paths
- input paths
- output paths
- manifest paths
- child args
- validation checks
- artifact plan
```

### 7.1.3. Acceptance Criteria

```text
- python 05_causal_discovery_inference_completejourney.py --validate-only が成功する
- python 05_causal_discovery_inference_completejourney.py --dry-run が成功する
- discovery output dir を YAML 側で変更しても inference が正しい discovery artifact を参照する
- manifest が壊れている場合は明確な error message で fail-fast する
- orchestration.py は thin facade である
- test_pipeline_strategies.py で RunStrategy / ValidateOnlyStrategy / DryRunStrategy が個別にテストされる
- production code が旧 package を import していないことを test で確認する
- full pipeline smoke test が pass する
```

---

## 7.2. CLI override の整合性

### 7.2.1. 修正対象ファイル / 箇所

```text
experiment/05_causal_discovery_inference_completejourney.py
experiment/causal_pipeline_runtime/planning.py
experiment/causal_pipeline_runtime/orchestration.py
experiment/causal_discovery/cli.py
experiment/causal_inference/cli.py
experiment/tests/test_cli_overrides.py
```

### 7.2.2. 実装要件

統合 CLI は prefix 付きで child CLI override を公開する。

```bash
--discovery-alpha
--discovery-alpha-grid
--discovery-pc-indep-test
--discovery-bootstrap-samples
--discovery-bootstrap-sample-fraction
--discovery-random-seed
--discovery-no-background-knowledge
--discovery-notears-threshold

--inference-mode
--inference-treatment
--inference-outcome
--inference-estimand
--inference-effect-methods
--inference-adjustment-strategy
--inference-covariates
--inference-robust-se
--inference-min-samples
--inference-edge-robust-se
```

B案や「config のみに寄せる」選択肢は採用しない。  
方針は **prefix 付き override を明示実装** とする。

### 7.2.3. Acceptance Criteria

```text
- 統合 CLI の --help を見れば override 可能な項目が明確に分かる
- 統合 CLI から渡した override が ExecutionPlan に反映される
- ExecutionPlan から child stage runner に override が確実に渡る
- override parity test を追加する
- 未対応 option が silently ignored されない
```

---

## 7.3. コード構造・保守性

### 7.3.1. 修正対象ファイル / 箇所

```text
experiment/causal_core/
experiment/causal_pipeline_runtime/
experiment/causal_discovery/
experiment/causal_inference/
experiment/docs/sphinx/api.rst
```

### 7.3.2. 実装要件

```text
- causal_core / causal_pipeline_runtime / causal_discovery / causal_inference へ package を再編する
- 旧 package からの import を削除する
- 各 package の __init__.py に public API を明示する
- __all__ を定義する
- private helper と public API を分ける
- Sphinx api.rst は public API 中心にする
```

### 7.3.3. Acceptance Criteria

```text
- CLI file が巨大な orchestration logic を持たない
- orchestration.py が巨大化しない
- planning / strategy / execution / validation が分離されている
- causal_core が runtime / discovery / inference に依存しない
- estimation logic と diagnostics logic が混在しない
- feature construction と causal estimation が分離されている
- 旧 package は production path から完全に外れている
```

---

## 7.4. YAML 設定管理

### 7.4.1. 修正対象ファイル / 箇所

```text
conf/causal_inference/pipeline.yaml
conf/causal_inference/causal_inference_default.yaml
conf/causal_inference/completejourney_household.yaml
conf/causal_inference/feature_semantics.yaml
conf/causal_inference/causal_design.yaml
conf/causal_discovery/features.yaml
conf/causal_discovery/analysis.yaml
experiment/causal_core/config/yaml_loader.py
experiment/causal_core/config/hashing.py
experiment/causal_core/features/semantics.py
experiment/causal_core/causal_design/schema.py
experiment/causal_pipeline_runtime/planning.py
experiment/causal_pipeline_runtime/artifacts.py
experiment/causal_pipeline_runtime/validation.py
experiment/causal_inference/features/config.py
```

### 7.4.2. 実装要件

pipeline config を追加する。

```yaml
pipeline:
  run_id: null
  random_seed: 42
  fail_fast: true
  validate_feature_semantics: true
  artifact_manifest: true

stages:
  discovery:
    enabled: true
    analysis_config: ...
    feature_config: ...
    output_dir: ...
  inference:
    enabled: true
    config: ...
    output_dir: ...
```

feature semantics schema を導入する。

```yaml
features:
  - name: pre_total_spend
    role: covariate
    source_table: transactions
    source_column: sales_value
    unit_id: household_key
    time_window:
      start: pre_period_start
      end: pre_period_end
    aggregation: sum
    transform: log1p
    fill_value: 0
    dtype: float
    allowed_for_adjustment: true
    post_treatment: false
```

causal design config を導入する。

```yaml
causal_design:
  estimand: ATE
  treatment:
    name: campaign_exposure
    time: campaign_start
    levels: [0, 1]
  outcome:
    name: outcome_spend
    window:
      start: campaign_start
      end: campaign_end
  unit: household
  time_zero: campaign_start
  assumptions:
    - consistency
    - conditional_exchangeability
    - positivity
    - no_interference
```

### 7.4.3. Acceptance Criteria

```text
- resolved config が必ず保存される
- config hash が manifest に保存される
- dry-run で resolved config path が表示される
- feature semantics mismatch test が存在する
- invalid adjustment set test が存在する
- resolved feature semantics が artifact として保存される
- cross-stage feature semantics validation は causal_pipeline_runtime/validation.py にある
- FeatureSemanticSpec は causal_core/features/semantics.py に定義されている
- CausalDesign は causal_core/causal_design/schema.py に定義されている
```

---

## 7.5. 特徴量作成処理

### 7.5.1. 修正対象ファイル / 箇所

```text
experiment/causal_core/features/semantics.py
experiment/causal_inference/features/aggregation.py
experiment/causal_inference/features/transforms.py
experiment/causal_inference/features/encoding.py
experiment/causal_inference/features/builder.py
experiment/causal_inference/features/config.py
conf/causal_inference/completejourney_household.yaml
conf/causal_discovery/features.yaml
experiment/tests/test_feature_semantics.py
```

### 7.5.2. 実装要件

aggregation registry:

```python
AGGREGATION_REGISTRY = {
    "sum": ...,
    "mean": ...,
    "count": ...,
    "nunique": ...,
    "max": ...,
    "min": ...,
}
```

transform registry:

```python
TRANSFORM_REGISTRY = {
    "identity": ...,
    "log1p": ...,
    "signed_log1p": ...,
    "zscore": ...,
}
```

encoding registry:

```python
ENCODING_REGISTRY = {
    "one_hot": ...,
    "ordinal": ...,
    "binary": ...,
}
```

role-based table config を使う。

```yaml
tables:
  unit_table:
    name: demographics
    unit_id: household_key

  transaction_table:
    name: transactions
    unit_id: household_key
    time_column: transaction_date

  treatment_assignment_table:
    name: campaigns
    unit_id: household_key
    treatment_column: campaign_id
```

### 7.5.3. Acceptance Criteria

```text
- builder は transactions / campaigns / demographics という具体名に依存しない
- invalid aggregation / transform / encoding は明確な error になる
- discovery と inference で同一 feature spec から同一 feature が生成されることを test する
- one-hot の列順が deterministic
- encoding artifact が保存される
- feature semantics schema は causal_core にあり、builder 実装は causal_inference にある
```

---

## 7.6. 統計的処理

### 7.6.1. 修正対象ファイル / 箇所

```text
experiment/causal_core/causal_design/schema.py
experiment/causal_inference/estimation/base.py
experiment/causal_inference/estimation/ols.py
experiment/causal_inference/estimation/g_computation.py
experiment/causal_inference/estimation/ipw.py
experiment/causal_inference/estimation/aipw.py
experiment/causal_inference/estimation/inference.py
experiment/causal_inference/estimation/multiplicity.py
experiment/causal_inference/diagnostics/model.py
experiment/causal_inference/reporting/markdown.py
conf/causal_inference/causal_inference_default.yaml
experiment/tests/test_estimators.py
experiment/tests/test_regression_estimators.py
```

### 7.6.2. 実装要件

estimator 名を明確に分ける。

```text
ols_coefficient
g_computation_ate
g_computation_att
ipw_ate
ipw_att
aipw_ate
aipw_att
```

`EstimatorSpec` を導入する。

```python
class EstimatorSpec:
    name: str
    supported_estimands: set[str]
    output_scale: str
    requires_propensity: bool
    requires_outcome_model: bool
    interpretation_level: str
```

要件:

```text
- OLS coefficient は regression coefficient として出力する
- OLS coefficient を ATE / ATT として表示しない
- g-computation ATE / ATT を実装する
- IPW ATE / ATT の weight formula を分ける
- stabilized weights を実装する
- ESS を出す
- weighted balance diagnostics を出す
- AIPW cross-fitting を実装する
- cross_fitting_folds > 0 を受け取って未実装にしない
- AIPW score から empirical influence curve を保存する
- 多重比較補正として Bonferroni / BH FDR を実装する
```

### 7.6.3. Acceptance Criteria

```text
- ATT 指定時に単純な OLS coefficient を ATT と表示しない
- unsupported estimator / estimand combination は fail-fast する
- IPW ATE / ATT で異なる weight が使われる
- weighted balance before/after が report に出る
- AIPW ATE / ATT の synthetic DGP test がある
- multiple edge result に adjusted p-value が含まれる
```

---

## 7.7. 因果推論分析の実態

### 7.7.1. 修正対象ファイル / 箇所

```text
conf/causal_inference/causal_design.yaml
conf/causal_inference/completejourney_household.yaml
conf/causal_inference/causal_inference_default.yaml
experiment/causal_core/causal_design/schema.py
experiment/causal_core/features/semantics.py
experiment/causal_inference/config.py
experiment/causal_inference/context.py
experiment/causal_inference/modes/edge_weight.py
experiment/causal_inference/modes/treatment_effect.py
experiment/causal_inference/reporting/markdown.py
experiment/causal_pipeline_runtime/validation.py
experiment/docs/sphinx/methodology/estimands.rst
experiment/docs/sphinx/methodology/identification.rst
experiment/docs/sphinx/methodology/adjustment_sets.rst
experiment/tests/test_feature_semantics.py
```

### 7.7.2. 実装要件

adjustment set に metadata を持たせる。

```yaml
adjustment_sets:
  pre_treatment_covariates:
    description: Pre-treatment household covariates only.
    variables:
      - pre_total_spend
      - pre_num_trips
      - household_size
    forbidden_roles:
      - treatment
      - outcome
      - mediator
      - collider
      - post_treatment
    reviewer_status: manually_reviewed
```

要件:

```text
- causal_design.estimand は必須
- treatment.name は feature semantics に存在し、role が treatment であること
- outcome.name は feature semantics に存在し、role が outcome であること
- adjustment set に post-treatment variable があれば error
- outcome / treatment が covariates に混入すると error
- discovery edge weight と treatment effect を schema 上で分ける
- discovery-only result を ATE / ATT として出力しない
```

edge weight mode の report に以下を明記する。

```text
Discovery graph edges are hypotheses or structural summaries.
Edge-level estimates are exploratory associations/effects under additional assumptions.
They are not automatically identified ATE or ATT estimates.
```

### 7.7.3. Acceptance Criteria

```text
- report 冒頭に estimand summary が出る
- estimator output に estimand metadata が必ず含まれる
- estimand 未指定の場合は error
- edge weight table に interpretation_level を追加する
- report で discovery と inference が明確に別 section
```

---

## 7.8. 診断出力

### 7.8.1. 修正対象ファイル / 箇所

```text
experiment/causal_inference/diagnostics/design.py
experiment/causal_inference/diagnostics/balance.py
experiment/causal_inference/diagnostics/overlap.py
experiment/causal_inference/diagnostics/outcome.py
experiment/causal_inference/diagnostics/model.py
experiment/causal_inference/reporting/markdown.py
experiment/causal_inference/reporting/tables.py
experiment/tests/test_diagnostics.py
```

### 7.8.2. 実装要件

出力する diagnostics:

```text
design diagnostics:
  - sample size
  - treated / control count
  - treatment prevalence
  - missingness by variable
  - excluded samples count and reason
  - covariate count
  - adjustment set name
  - estimand
  - positivity risk flag

covariate balance:
  - unweighted balance
  - IPW weighted balance
  - ATT weighted balance
  - SMD
  - variance ratio
  - missingness difference
  - max absolute SMD
  - number of covariates with abs(SMD) > 0.1

overlap diagnostics:
  - treated/control 別 propensity score quantile
  - histogram / ECDF 用 CSV
  - common support
  - ESS
  - max weight
  - p95/p99 weight
  - clipping fraction
  - overlap violation flag

outcome diagnostics:
  - outcome summary overall
  - outcome summary by treatment
  - zero rate
  - skewness
  - outlier fraction
  - residual diagnostics
  - prediction error summary
  - heavy-tail warning
```

### 7.8.3. Acceptance Criteria

```text
- diagnostics が JSON / CSV / Markdown に出る
- weighted balance test がある
- overlap が悪い synthetic data で warning が出る
- heavy-tailed outcome synthetic data で warning が出る
```

---

## 7.9. レポート・解釈

### 7.9.1. 修正対象ファイル / 箇所

```text
experiment/causal_inference/reporting/markdown.py
experiment/causal_inference/reporting/tables.py
experiment/causal_inference/reporting/warnings.py
experiment/causal_inference/modes/treatment_effect.py
experiment/causal_inference/modes/edge_weight.py
experiment/tests/test_causal_inference_pipeline.py
experiment/tests/test_smoke_pipeline.py
```

### 7.9.2. 実装要件

Markdown report を以下の構成にする。

```text
# Causal Inference Report

## 1. Run Summary
## 2. Causal Design
## 3. Feature Semantics
## 4. Adjustment Set
## 5. Estimation Results
## 6. Design Diagnostics
## 7. Covariate Balance
## 8. Overlap Diagnostics
## 9. Outcome Diagnostics
## 10. Multiplicity / Exploratory Analysis
## 11. Warnings and Limitations
## 12. Reproducibility
```

result table に以下を含める。

```text
method
estimand
estimate
std_error
conf_low
conf_high
p_value
p_value_adjusted
adjustment_method
n_units
n_treated
n_control
effective_sample_size
diagnostic_status
interpretation_level
warnings
```

### 7.9.3. Acceptance Criteria

```text
- report だけ読んでも estimand / estimator / adjustment set / diagnostics が分かる
- warnings が report 冒頭にも summary として出る
- artifact paths が report 末尾に出る
- ATE / ATT / regression coefficient / exploratory edge estimate が混同されない
```

---

## 7.10. テスト・再現性

### 7.10.1. 修正対象ファイル / 箇所

```text
experiment/tests/test_core.py
experiment/tests/test_planning.py
experiment/tests/test_pipeline_strategies.py
experiment/tests/test_stage_runners.py
experiment/tests/test_orchestration.py
experiment/tests/test_manifest.py
experiment/tests/test_cli_overrides.py
experiment/tests/test_feature_semantics.py
experiment/tests/test_estimators.py
experiment/tests/test_diagnostics.py
experiment/tests/test_smoke_pipeline.py
experiment/tests/test_regression_estimators.py
experiment/tests/conftest.py
experiment/tests/fixtures/
```

### 7.10.2. 実装要件

以下の unit test を追加する。

```text
- causal_core が他 project package を import していないこと
- ValidationIssue / ValidationResult
- YAML loader
- config hashing
- FeatureSemanticSpec
- CausalDesign
- ExecutionPlan construction
- PipelineCommandStrategy selection
- ValidateOnlyStrategy behavior
- DryRunStrategy behavior
- RunStrategy behavior
- StageRunner validation / execution
- config loading
- manifest creation / reading
- CLI override
- feature semantics validation
- aggregation registry
- transform registry
- encoding registry
- adjustment set validation
- OLS coefficient
- g-computation ATE / ATT
- IPW ATE / ATT weights
- AIPW ATE / ATT score
- weighted balance
- overlap diagnostics
- multiplicity correction
- production code が旧 package を import していないこと
```

edge cases:

```text
- no treated
- no control
- perfect separation
- extreme propensity
- missing covariates
- rank deficient design matrix
```

### 7.10.3. Acceptance Criteria

```text
- pytest が pass する
- smoke test は CI で 1 分以内を目標にする
- expected artifacts が生成される
- manifest が生成される
- validate-only smoke test がある
- dry-run smoke test がある
- fixed seed DGP の regression test がある
```

---

## 7.11. Sphinx / ドキュメント

### 7.11.1. 修正対象ファイル / 箇所

```text
experiment/docs/sphinx/conf.py
experiment/docs/sphinx/index.rst
experiment/docs/sphinx/api.rst
experiment/docs/sphinx/methodology.rst
experiment/docs/sphinx/methodology/estimands.rst
experiment/docs/sphinx/methodology/identification.rst
experiment/docs/sphinx/methodology/adjustment_sets.rst
experiment/docs/sphinx/methodology/diagnostics.rst
experiment/docs/sphinx/methodology/feature_semantics.rst
experiment/docs/sphinx/methodology/limitations.rst
experiment/docs/sphinx/reproducibility.rst
```

### 7.11.2. 実装要件

主要 public API に Google style docstring を追加する。

```text
Args:
Returns:
Raises:
Notes:
```

因果推論 estimator の `Notes:` には以下を書く。

```text
- estimand
- assumptions
- limitations
```

Sphinx docs に以下を追加する。

```text
methodology/estimands.rst
methodology/identification.rst
methodology/adjustment_sets.rst
methodology/diagnostics.rst
methodology/feature_semantics.rst
methodology/limitations.rst
reproducibility.rst
api.rst
```

### 7.11.3. Acceptance Criteria

```text
- sphinx-build -W が warning-free
- ATE / ATT / IPW / AIPW / OLS coefficient の違いが明記されている
- causal discovery と causal inference の違いが明記されている
- causal_core と causal_pipeline_runtime の違いが明記されている
- pipeline execution strategy と inference mode strategy の違いが明記されている
- causal_core / causal_pipeline_runtime / causal_discovery / causal_inference が public API として文書化されている
```

---

# 8. 優先順位

## 8.P0. 実行可能性と package architecture

最優先で実装する。

```text
1. causal_core package 新設
2. causal_pipeline_runtime package 新設
3. causal_discovery package へ移行
4. causal_inference package へ移行
5. 旧 package production import の撤去
6. ExecutionPlan
7. StagePlan
8. PipelineCommandStrategy
9. RunStrategy
10. ValidateOnlyStrategy
11. DryRunStrategy
12. StageRunner
13. DiscoveryStageRunner
14. InferenceStageRunner
15. run manifest
16. resolved config 保存
17. discovery output dir の manifest 経由連携
18. full pipeline smoke test
19. CLI override 方針の統一
```

---

## 8.P1. 因果推論としての誤解防止

```text
1. feature semantics schema
2. discovery / inference feature consistency check
3. causal design config
4. adjustment set validation
5. ATE / ATT / OLS coefficient の分離
6. edge weight と treatment effect の分離
7. AIPW cross-fitting の実装
```

---

## 8.P2. 診断の充実

```text
1. weighted balance
2. overlap diagnostics
3. ESS / extreme weight diagnostics
4. outcome diagnostics
5. diagnostic warning system
```

---

## 8.P3. 汎用化

```text
1. role-based table config
2. aggregation / transform / encoding registry
3. CompleteJourney 依存の除去
4. config schema validation
```

---

## 8.P4. ドキュメント・保守性

```text
1. public API 整理
2. Google style docstring
3. Sphinx methodology docs
4. Sphinx build test
```

---

# 9. 実装方針

## 9.1. 原則

```text
- 後方互換性は維持しない
- 旧構成ではなく新構成へ移行する
- 旧 API wrapper は作らない
- 既存挙動の温存より、責務分離と誤解防止を優先する
- 因果推論上危険な silent behavior は fail-fast に変更する
- 新 config schema を追加する場合は example config も更新する
- すべての新規 artifact は deterministic な path に保存する
- random seed は manifest に保存する
- causal_core は stage 非依存の共通基盤に限定する
- causal_pipeline_runtime は pipeline execution runtime に限定する
- pipeline execution strategy と inference mode strategy を混同しない
- orchestration.py は thin facade として保つ
- causal_discovery は inference config / causal design / adjustment set の責務を持たない
```

---

## 9.2. 禁止事項

```text
- common_in_causal_inference の後方互換 wrapper を作らない
- causal_discovery_pipeline / causal_inference_pipeline の alias を作らない
- production code から旧 package を import しない
- causal_core から causal_pipeline_runtime / causal_discovery / causal_inference を import しない
- causal_pipeline_runtime を共通 utility 置き場にしない
- --validate-only / --dry-run を causal_discovery に実装しない
- --validate-only / --dry-run を causal_inference mode strategy として実装しない
- orchestration.py に validate-only / dry-run の詳細ロジックを直書きしない
- OLS coefficient を ATE / ATT として表示しない
- discovery graph の edge weight を識別済み causal effect として表示しない
- feature semantics mismatch を warning のみにしない
- cross_fitting_folds を受け取りながら未実装にしない
- CLI override を silently ignored しない
- post-treatment variable を adjustment set に許容しない
- 修正対象ファイルが曖昧なまま実装しない
```

---

# 10. 最終成果物

以下を提出すること。

```text
1. 実装済みコード
2. 更新済み YAML config
3. 更新済み tests
4. 更新済み Sphinx docs
5. 変更点 summary
6. 評点改善 summary
7. 修正対象ファイル summary
8. architecture summary
9. removed compatibility summary
10. dependency summary
```

修正対象ファイル summary は以下の形式にする。

```markdown
# 修正対象ファイル Summary

| ファイル | 変更区分 | 目的 | 関連観点 |
|---|---|---|---|
| experiment/causal_core/features/semantics.py | 新規 | Feature semantics schema | 2, 3, 4, 6 |
| experiment/causal_core/causal_design/schema.py | 新規 | Causal design schema | 3, 6 |
| experiment/causal_pipeline_runtime/planning.py | 新規 | ExecutionPlan / StagePlan | 1, 2, 9 |
| experiment/causal_pipeline_runtime/strategies.py | 新規 | Run / ValidateOnly / DryRun strategy | 1, 2, 9 |
| experiment/causal_pipeline_runtime/execution.py | 新規 | StageRunner / stage execution | 1, 2, 9 |
| experiment/causal_pipeline_runtime/artifacts.py | 新規 | RunManifest / ArtifactRegistry | 1, 3, 9 |
| experiment/causal_inference/estimation/g_computation.py | 新規 | ATE / ATT 対応 estimator | 5, 6 |
```

architecture summary は以下の形式にする。

```markdown
# Architecture Summary

## Packages

- causal_core
- causal_pipeline_runtime
- causal_discovery
- causal_inference

## Core layer

- ValidationIssue
- ValidationResult
- FeatureSemanticSpec
- FeatureSemanticsCatalog
- CausalDesign
- YAML loader
- config hashing
- common exceptions

## Pipeline runtime

- ExecutionPlan
- StagePlan
- RunStrategy
- ValidateOnlyStrategy
- DryRunStrategy
- DiscoveryStageRunner
- InferenceStageRunner
- RunManifest
- ArtifactRegistry

## Analysis packages

- causal_discovery
- causal_inference

## Dependency direction

Correct:

entrypoint
  -> causal_pipeline_runtime
      -> causal_discovery.runner
      -> causal_inference.runner

causal_pipeline_runtime -> causal_core
causal_discovery        -> causal_core
causal_inference        -> causal_core

Forbidden:

causal_core
  -> causal_pipeline_runtime / causal_discovery / causal_inference

causal_discovery
  -> causal_inference config / causal design

Deprecated and removed from production imports:

- common_in_causal_inference
- causal_discovery_pipeline
- causal_inference_pipeline
```

removed compatibility summary は以下の形式にする。

```markdown
# Removed Compatibility Summary

| Removed / Replaced | Replacement | Reason |
|---|---|---|
| common_in_causal_inference | causal_core + causal_pipeline_runtime | common utility と runtime が混在して責務が曖昧だったため |
| causal_discovery_pipeline | causal_discovery | 因果探索分析のクラス群であることを明示するため |
| causal_inference_pipeline | causal_inference | 因果推論分析のクラス群であることを明示するため |
| --discovery-dir | --discovery-manifest | stage 間 artifact contract を manifest に一本化するため |
| implicit args forwarding | ExecutionPlan | dry-run / validate-only / reproducibility を明確化するため |
```

dependency summary は以下の形式にする。

```markdown
# Dependency Summary

| From | Allowed Imports |
|---|---|
| causal_core | standard library, numpy, pandas, yaml |
| causal_pipeline_runtime | causal_core, causal_discovery.runner, causal_inference.runner |
| causal_discovery | causal_core |
| causal_inference | causal_core |

## Forbidden Imports

- causal_core -> causal_pipeline_runtime
- causal_core -> causal_discovery
- causal_core -> causal_inference
- causal_discovery -> causal_inference
- causal_inference -> causal_discovery internals
```

評点改善 summary は以下の形式にする。

```markdown
# 評点改善 Summary

| 観点 | Before | After | 主な改善 |
|---|---:|---:|---|
| 1. CLI パイプライン処理 | 7.0 | 8.5+ | causal_pipeline_runtime, ExecutionPlan, pipeline strategies, stage runners, manifest |
| 2. コード構造・保守性 | 7.0 | 8.5+ | causal_core, package rename, planning/strategy/execution 分離, public API |
| 3. YAML 設定管理 | 7.0 | 8.5+ | resolved config, feature semantics, causal design, schema validation |
| 4. 特徴量作成処理 | 6.0 | 8.0+ | aggregation/transform/encoding registry |
| 5. 統計的処理 | 5.0 | 8.0+ | g-computation, IPW diagnostics, AIPW cross-fitting |
| 6. 因果推論分析の実態 | 4.0 | 8.0+ | causal design, adjustment validation, estimand-estimator mapping |
| 7. 診断出力 | 5.5 | 8.0+ | weighted balance, overlap, ESS, outcome diagnostics |
| 8. レポート・解釈 | 7.0 | 8.5+ | structured report, warnings, interpretation levels |
| 9. テスト・再現性 | 5.0 | 8.0+ | unit/smoke/regression tests |
| 10. Sphinx / ドキュメント | 6.5 | 8.0+ | Google docstring, methodology docs, sphinx-build -W |
```

---

# 11. 完了条件

以下の command が成功すること。  
実行は **uv** を前提とする。

```bash
uv sync
```

```bash
uv run pytest workspace/articles/1ceee528ed7ee8/experiment/tests
```

```bash
uv run sphinx-build -W -b html \
  workspace/articles/1ceee528ed7ee8/experiment/docs/sphinx \
  workspace/articles/1ceee528ed7ee8/experiment/docs/sphinx/_build/html
```

さらに、以下の統合 smoke 実行が成功すること。

```bash
uv run python workspace/articles/1ceee528ed7ee8/experiment/05_causal_discovery_inference_completejourney.py \
  --validate-only
```

```bash
uv run python workspace/articles/1ceee528ed7ee8/experiment/05_causal_discovery_inference_completejourney.py \
  --dry-run
```

```bash
uv run python workspace/articles/1ceee528ed7ee8/experiment/05_causal_discovery_inference_completejourney.py
```

生成 report に以下が含まれること。

```text
- estimand summary
- causal design
- adjustment set
- estimator metadata
- design diagnostics
- covariate balance
- weighted balance
- overlap diagnostics
- outcome diagnostics
- multiplicity correction
- warnings and limitations
- reproducibility metadata
- artifact manifest path
```

dry-run 出力に以下が含まれること。

```text
- selected pipeline strategy
- ExecutionPlan
- StagePlan
- resolved config paths
- input paths
- output paths
- manifest paths
- child stage args
- validation checks
- artifact plan
```

validate-only 出力に以下が含まれること。

```text
- validation status
- checked configs
- checked artifact paths
- checked feature semantics
- checked causal design
- checked adjustment set
- errors / warnings
```

さらに、以下が成立すること。

```text
production code が以下を import していない:

- common_in_causal_inference
- causal_discovery_pipeline
- causal_inference_pipeline
```

さらに、以下が成立すること。

```text
causal_core が以下を import していない:

- causal_pipeline_runtime
- causal_discovery
- causal_inference
```