"""Deterministic structural validation of D01-D21 taxonomy references."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .schema_validator import ValidationIssue

DIMENSIONS = tuple(f"D{i:02d}" for i in range(1, 22))


def load_taxonomy_catalog(article_root: Path) -> Mapping[str, Any]:
    configured = os.environ.get("URBAN_LEGEND_TAXONOMY_CATALOG")
    candidates = [Path(configured)] if configured else [
        article_root / "docs/00_research_overview/taxonomy_catalog.json",
        article_root / "docs/00_research_overview/20_urban_legend_parent_child_code_system.json",
    ]
    for path in candidates:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("codes"), dict):
                raise ValueError(f"taxonomy catalog must contain object 'codes': {path}")
            return data
    raise FileNotFoundError("machine-readable taxonomy catalog not found; set URBAN_LEGEND_TAXONOMY_CATALOG")


def _issue(rule: str, path: str, message: str, *, expected=None, actual=None) -> ValidationIssue:
    return ValidationIssue(rule, "20", path, message, "taxonomy_validator", expected, actual)


def validate_taxonomy(analysis: Mapping[str, Any], taxonomy_catalog: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
    errors: list[ValidationIssue] = []
    dimensions = analysis.get("dimensions", [])
    ids = [d.get("dimension_id") for d in dimensions if isinstance(d, dict)]
    if sorted(ids) != list(DIMENSIONS):
        errors.append(_issue("V-DIM-001", "$.dimensions", "dimension set must be exactly D01-D21", expected=list(DIMENSIONS), actual=ids))

    codes: Mapping[str, Any] = taxonomy_catalog.get("codes", {})
    for idx, dim in enumerate(dimensions):
        if not isinstance(dim, dict):
            continue
        status = dim.get("status")
        primary = dim.get("primary")
        secondary = dim.get("secondary", [])
        refs = ([primary] if isinstance(primary, dict) else []) + [x for x in secondary if isinstance(x, dict)]
        if status in {"D", "I"} and not isinstance(primary, dict):
            errors.append(_issue("V-STATUS-001", f"$.dimensions[{idx}].primary", f"status {status} requires primary code"))
        if status in {"U", "NA", "C"} and (primary is not None or secondary):
            errors.append(_issue("V-STATUS-001", f"$.dimensions[{idx}]", f"status {status} forbids primary/secondary codes"))
        for ref_idx, ref in enumerate(refs):
            code_id = ref.get("code_id")
            cat = codes.get(code_id)
            ref_path = f"$.dimensions[{idx}].{'primary' if ref_idx == 0 and isinstance(primary, dict) else 'secondary'}"
            if cat is None:
                errors.append(_issue("V-CODE-001", ref_path, f"unknown code_id: {code_id}"))
                continue
            expected_parent = cat.get("parent_id") if isinstance(cat, dict) else None
            actual_parent = ref.get("parent_id")
            if expected_parent is not None and actual_parent != expected_parent:
                errors.append(_issue("V-PARENT-001", ref_path + ".parent_id", f"parent mismatch for {code_id}", expected=expected_parent, actual=actual_parent))
            dim_id = dim.get("dimension_id")
            if isinstance(code_id, str) and isinstance(dim_id, str) and not code_id.startswith(dim_id + "."):
                errors.append(_issue("V-CODE-002", ref_path + ".code_id", f"code {code_id} does not belong to {dim_id}"))

    return tuple(sorted(errors, key=ValidationIssue.sort_key))
