"""Cross-stage validation for the integrated pipeline."""

from __future__ import annotations

from pathlib import Path

from causal_core.causal_design import CausalDesign
from causal_core.config import load_yaml_mapping
from causal_core.features import FeatureSemanticsCatalog, compare_feature_semantics
from causal_core.validation import ValidationIssue, ValidationResult, ValidationSeverity

from .planning import ExecutionPlan, StagePlan


class CrossStageValidator:
    """Validate plan-level and cross-stage contracts."""

    def validate(self, plan: ExecutionPlan) -> ValidationResult:
        """Validate an execution plan."""

        issues: list[ValidationIssue] = []
        for stage in plan.enabled_stages():
            issues.extend(self.validate_stage(stage, plan))
        issues.extend(self.validate_feature_semantics(plan))
        issues.extend(self.validate_causal_design(plan))
        issues.extend(self.validate_adjustment_sets(plan))
        return ValidationResult(issues)

    def validate_stage(self, stage: StagePlan, plan: ExecutionPlan) -> list[ValidationIssue]:
        """Validate one stage plan."""

        issues: list[ValidationIssue] = []
        for name, path in stage.config_paths.items():
            if not path.exists():
                issues.append(_error("config_missing", f"missing config file: {path}", f"{stage.name}.{name}"))
        for name, path in stage.output_paths.items():
            if name.endswith("dir") or name == "output_dir":
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    issues.append(_error("output_dir_invalid", str(exc), f"{stage.name}.{name}"))
        if stage.name == "inference":
            manifest = stage.input_paths["discovery_manifest"]
            discovery_enabled = any(candidate.name == "discovery" and candidate.enabled for candidate in plan.stages)
            if manifest.exists():
                issues.extend(self.validate_discovery_manifest_schema(manifest))
            elif not discovery_enabled:
                issues.append(_error("discovery_manifest_missing", f"missing discovery manifest: {manifest}", "inference.discovery_manifest"))
            else:
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.INFO,
                        "discovery_manifest_planned",
                        f"discovery manifest will be produced by this run: {manifest}",
                        "inference.discovery_manifest",
                    )
                )
        return issues

    def validate_discovery_manifest_schema(self, manifest_path: Path) -> list[ValidationIssue]:
        """Validate minimum discovery manifest schema."""

        try:
            manifest = load_yaml_mapping(manifest_path)
        except Exception as exc:
            return [_error("discovery_manifest_unreadable", str(exc), "inference.discovery_manifest")]
        required = {"run_id", "stage", "resolved_output_dir", "artifacts"}
        missing = sorted(required.difference(manifest))
        if missing:
            return [
                _error(
                    "discovery_manifest_invalid",
                    f"manifest missing keys: {missing}",
                    "inference.discovery_manifest",
                )
            ]
        if manifest.get("stage") != "discovery":
            return [
                _error(
                    "discovery_manifest_invalid_stage",
                    f"expected discovery manifest, got {manifest.get('stage')!r}",
                    "inference.discovery_manifest",
                )
            ]
        return []

    def validate_feature_semantics(self, plan: ExecutionPlan) -> list[ValidationIssue]:
        """Validate discovery/inference feature semantics when both configs exist."""

        discovery = _stage_by_name(plan, "discovery")
        inference = _stage_by_name(plan, "inference")
        if discovery is None or inference is None:
            return []
        discovery_path = discovery.config_paths.get("feature_config")
        inference_path = inference.config_paths.get("feature_semantics")
        if discovery_path is None or inference_path is None or not discovery_path.exists() or not inference_path.exists():
            return []
        try:
            discovery_catalog = FeatureSemanticsCatalog.from_feature_config_mapping(
                load_yaml_mapping(discovery_path)
            )
            inference_catalog = FeatureSemanticsCatalog.from_mapping(load_yaml_mapping(inference_path))
            mismatches = compare_feature_semantics(discovery_catalog, inference_catalog)
        except Exception as exc:
            return [_error("feature_semantics_invalid", str(exc), "feature_semantics")]
        return [
            _error("feature_semantics_mismatch", mismatch, "feature_semantics")
            for mismatch in mismatches
        ]

    def validate_causal_design(self, plan: ExecutionPlan) -> list[ValidationIssue]:
        """Validate causal design presence and minimum treatment/outcome semantics."""

        inference = _stage_by_name(plan, "inference")
        if inference is None or not inference.enabled:
            return []
        design_path = inference.config_paths.get("causal_design")
        semantics_path = inference.config_paths.get("feature_semantics")
        if design_path is None or not design_path.exists():
            return [_error("causal_design_missing", f"missing causal design: {design_path}", "causal_design")]
        try:
            design = CausalDesign.from_mapping(load_yaml_mapping(design_path))
        except Exception as exc:
            return [_error("causal_design_invalid", str(exc), "causal_design")]
        if semantics_path is None or not semantics_path.exists():
            return []
        try:
            catalog = FeatureSemanticsCatalog.from_mapping(load_yaml_mapping(semantics_path)).by_name()
        except Exception as exc:
            return [_error("feature_semantics_invalid", str(exc), "feature_semantics")]
        issues: list[ValidationIssue] = []
        treatment = catalog.get(design.treatment.name)
        outcome = catalog.get(design.outcome.name)
        if treatment is None or treatment.role.value != "treatment":
            issues.append(_error("treatment_semantics_invalid", f"treatment is not role=treatment: {design.treatment.name}", "causal_design.treatment"))
        if outcome is None or outcome.role.value != "outcome":
            issues.append(_error("outcome_semantics_invalid", f"outcome is not role=outcome: {design.outcome.name}", "causal_design.outcome"))
        return issues

    def validate_adjustment_sets(self, plan: ExecutionPlan) -> list[ValidationIssue]:
        """Validate configured adjustment sets for obvious bad controls."""

        inference = _stage_by_name(plan, "inference")
        if inference is None or not inference.enabled:
            return []
        feature_config_path = inference.config_paths.get("feature_config")
        if feature_config_path is None or not feature_config_path.exists():
            return []
        try:
            config = load_yaml_mapping(feature_config_path)
        except Exception as exc:
            return [_error("feature_config_invalid", str(exc), "inference.feature_config")]
        adjustment_sets = dict(config.get("adjustment_sets", {}))
        exclude_patterns = adjustment_sets.get("exclude_patterns", [])
        import re

        issues: list[ValidationIssue] = []
        treatment_name = str(dict(config.get("treatment", {})).get("name", "treated"))
        for set_name, set_config in adjustment_sets.items():
            if set_name == "exclude_patterns":
                continue
            variables = list(set_config.get("variables", set_config.get("include", [])))
            for variable in variables:
                if variable == treatment_name:
                    issues.append(_error("adjustment_contains_treatment", f"adjustment set contains treatment: {variable}", f"adjustment_sets.{set_name}"))
                if any(re.search(pattern, str(variable)) for pattern in exclude_patterns):
                    issues.append(_error("adjustment_contains_post_treatment", f"adjustment set contains excluded/post-treatment variable: {variable}", f"adjustment_sets.{set_name}"))
        return issues


def _stage_by_name(plan: ExecutionPlan, name: str) -> StagePlan | None:
    for stage in plan.stages:
        if stage.name == name:
            return stage
    return None


def _error(code: str, message: str, location: str | None = None) -> ValidationIssue:
    return ValidationIssue(ValidationSeverity.ERROR, code, message, location)


__all__ = ["CrossStageValidator"]
