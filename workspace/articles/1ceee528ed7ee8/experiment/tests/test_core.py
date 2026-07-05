from __future__ import annotations

import ast
from pathlib import Path

from causal_core.causal_design import CausalDesign
from causal_core.config import hash_mapping, load_yaml_mapping
from causal_core.features import FeatureRole, FeatureSemanticSpec, FeatureSemanticsCatalog
from causal_core.validation import ValidationIssue, ValidationResult, ValidationSeverity


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
ARTICLE_DIR = EXPERIMENT_DIR.parent


def test_validation_result_detects_errors() -> None:
    result = ValidationResult(
        [ValidationIssue(ValidationSeverity.ERROR, "bad", "message")]
    )

    assert result.has_errors


def test_yaml_hash_and_causal_design_schema() -> None:
    config = load_yaml_mapping(ARTICLE_DIR / "conf/causal_inference/causal_design.yaml")
    design = CausalDesign.from_mapping(config)

    assert design.estimand.value == "ATE"
    assert design.treatment.name == "treated"
    assert len(hash_mapping(design.to_dict())) == 64


def test_feature_semantics_catalog_loads_roles() -> None:
    catalog = FeatureSemanticsCatalog.from_mapping(
        load_yaml_mapping(ARTICLE_DIR / "conf/causal_inference/feature_semantics.yaml")
    ).by_name()

    assert catalog["treated"].role == FeatureRole.TREATMENT
    assert catalog["outcome_sales_value"].role == FeatureRole.OUTCOME
    assert FeatureSemanticSpec(
        name="x",
        role=FeatureRole.COVARIATE,
        source_table="t",
        source_column="c",
        unit_id="u",
    ).allowed_for_adjustment is False


def test_causal_core_does_not_import_runtime_or_analysis_packages() -> None:
    forbidden = {"causal_pipeline_runtime", "causal_discovery", "causal_inference"}
    for path in (EXPERIMENT_DIR / "causal_core").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = {node.module.split(".")[0]}
            else:
                continue
            assert not imported.intersection(forbidden), path
