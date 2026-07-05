from __future__ import annotations

import subprocess
from pathlib import Path

from causal_pipeline_runtime.orchestration import parse_args
from causal_pipeline_runtime.planning import PipelinePlanner
from causal_pipeline_runtime.strategies import DryRunStrategy, ValidateOnlyStrategy


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

    assert ["--alpha", "0.05"] == discovery.resolved_args[-4:-2]
    assert "--discovery-manifest" in inference.resolved_args
    assert "--discovery-dir" not in inference.resolved_args
    assert ["--mode", "treatment_effect"] == inference.resolved_args[-4:-2]


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
