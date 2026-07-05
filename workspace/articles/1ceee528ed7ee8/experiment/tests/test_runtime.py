from __future__ import annotations

import subprocess
from pathlib import Path

from causal_core.config import load_yaml_mapping
from causal_pipeline_runtime.artifacts import ArtifactRegistry
from causal_pipeline_runtime.orchestration import parse_args
from causal_pipeline_runtime.planning import ExecutionPlan, PipelinePlanner, StagePlan
from causal_pipeline_runtime.strategies import DryRunStrategy, ValidateOnlyStrategy
from causal_pipeline_runtime.validation import CrossStageValidator


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
ENTRYPOINT = EXPERIMENT_DIR / "05_causal_discovery_inference_completejourney.py"


def test_plan_uses_prefixed_overrides_and_manifest_contract() -> None:
    args = parse_args(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--discovery-alpha",
            "0.05",
            "--inference-mode",
            "treatment_effect",
            "--inference-outcome",
            "outcome_quantity",
        ]
    )
    plan = PipelinePlanner(PROJECT_ROOT).build_plan(args, strategy_name="run")
    discovery = plan.stages[0]
    inference = plan.stages[1]

    assert arg_value(discovery.resolved_args, "--alpha") == "0.05"
    assert "--discovery-manifest" in inference.resolved_args
    assert "--discovery-dir" not in inference.resolved_args
    assert arg_value(inference.resolved_args, "--mode") == "treatment_effect"
    assert arg_value(inference.resolved_args, "--outcome") == "outcome_quantity"


def test_pipeline_strategies_validate_and_dry_run() -> None:
    args = parse_args(["--project-root", str(PROJECT_ROOT)])
    validate_plan = PipelinePlanner(PROJECT_ROOT).build_plan(args, strategy_name="validate_only")
    dry_plan = PipelinePlanner(PROJECT_ROOT).build_plan(args, strategy_name="dry_run")

    validation = ValidateOnlyStrategy().execute(validate_plan)
    dry_run = DryRunStrategy().execute(dry_plan)

    assert validation.status == "ok"
    assert dry_run.payload is not None
    assert dry_run.payload["selected_pipeline_strategy"] == "dry_run"


def test_cli_validate_only_and_dry_run_smoke() -> None:
    validate = subprocess.run(
        ["uv", "run", "python", str(ENTRYPOINT), "--validate-only"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    dry = subprocess.run(
        ["uv", "run", "python", str(ENTRYPOINT), "--dry-run"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "validation status: ok" in validate.stdout
    assert "selected_pipeline_strategy" in dry.stdout


def test_cli_full_run_smoke_writes_manifests_and_report() -> None:
    subprocess.run(
        ["uv", "run", "python", str(ENTRYPOINT)],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    article_dir = EXPERIMENT_DIR.parent
    discovery_manifest = article_dir / "artifacts/causal_discovery/manifest.yaml"
    inference_manifest = article_dir / "artifacts/causal_inference/manifest.yaml"
    inference_report = article_dir / "artifacts/causal_inference/edge_weight/edge_effects.md"

    assert discovery_manifest.exists()
    assert inference_manifest.exists()
    assert inference_report.exists()
    assert load_yaml_mapping(discovery_manifest)["stage"] == "discovery"
    assert load_yaml_mapping(inference_manifest)["stage"] == "inference"

    report_text = inference_report.read_text(encoding="utf-8")
    assert "## Causal Design" in report_text
    assert "## Estimand Summary" in report_text
    assert "artifact_manifest_path" in report_text


def test_production_code_has_no_old_package_imports() -> None:
    forbidden = (
        "common_in_causal_inference",
        "causal_discovery_pipeline",
        "causal_inference_pipeline",
    )
    roots = [
        EXPERIMENT_DIR / "05_causal_discovery_inference_completejourney.py",
        EXPERIMENT_DIR / "causal_core",
        EXPERIMENT_DIR / "causal_pipeline_runtime",
        EXPERIMENT_DIR / "causal_discovery",
        EXPERIMENT_DIR / "causal_inference",
    ]
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in ([root] if root.is_file() else root.rglob("*.py"))
    )
    assert not any(name in text for name in forbidden)
    assert not (EXPERIMENT_DIR / "causal_inference/discovery").exists()
    assert (EXPERIMENT_DIR / "causal_inference/discovery_artifacts").exists()


def test_adjustment_set_validation_uses_feature_semantics(tmp_path: Path) -> None:
    feature_config = tmp_path / "features.yaml"
    feature_semantics = tmp_path / "semantics.yaml"
    feature_config.write_text(
        """
adjustment_sets:
  bad_set:
    variables:
      - missing_covariate
      - treated
      - outcome
      - blocked_covariate
      - post_covariate
""",
        encoding="utf-8",
    )
    feature_semantics.write_text(
        """
features:
  - name: treated
    role: treatment
    source_table: campaigns
    source_column: campaign_id
    unit_id: household_id
    allowed_for_adjustment: false
    post_treatment: false
  - name: outcome
    role: outcome
    source_table: transactions
    source_column: sales_value
    unit_id: household_id
    allowed_for_adjustment: false
    post_treatment: false
  - name: blocked_covariate
    role: covariate
    source_table: demographics
    source_column: age
    unit_id: household_id
    allowed_for_adjustment: false
    post_treatment: false
  - name: post_covariate
    role: covariate
    source_table: transactions
    source_column: sales_value
    unit_id: household_id
    allowed_for_adjustment: true
    post_treatment: true
""",
        encoding="utf-8",
    )
    stage = StagePlan(
        name="inference",
        enabled=True,
        input_paths={"discovery_manifest": tmp_path / "manifest.yaml"},
        output_paths={"output_dir": tmp_path, "manifest": tmp_path / "manifest.yaml"},
        config_paths={
            "feature_config": feature_config,
            "feature_semantics": feature_semantics,
        },
        resolved_args=[],
        metadata={},
    )
    plan = ExecutionPlan(
        run_id="test",
        strategy="validate_only",
        stages=[stage],
        resolved_configs={},
        artifact_registry=ArtifactRegistry(),
        validation_checks=[],
        metadata={},
    )

    issues = CrossStageValidator().validate_adjustment_sets(plan)
    codes = {issue.code for issue in issues}

    assert "adjustment_semantics_missing" in codes
    assert "adjustment_role_invalid" in codes
    assert "adjustment_not_allowed" in codes
    assert "adjustment_post_treatment" in codes
    assert "adjustment_forbidden_role" in codes


def arg_value(args: list[str], option: str) -> str | None:
    """Return one CLI option value without assuming argument order."""

    if option not in args:
        return None
    index = args.index(option)
    if index + 1 >= len(args):
        return None
    return args[index + 1]
