from __future__ import annotations

import copy
from pathlib import Path

from src.validation.reference_validator import ParsedArtifact, validate_references
from src.validation.schema_validator import validate_data
from src.validation.taxonomy_validator import validate_taxonomy


def test_schema_valid_and_invalid(valid_sources):
    root = Path(__file__).resolve().parents[1]
    schema = root / "docs/10_each_lore/0000_tutorial/schemas/00_sources.schema.json"

    valid = validate_data(valid_sources, schema, artifact="00")
    assert valid.ok
    assert valid.errors == ()

    invalid_data = copy.deepcopy(valid_sources)
    invalid_data.pop("sources")
    invalid = validate_data(invalid_data, schema, artifact="00")
    assert not invalid.ok
    assert any(issue.rule_id == "V-SCHEMA-001" for issue in invalid.errors)


def test_reference_duplicate_and_dangling(valid_sources, valid_contents):
    sources = copy.deepcopy(valid_sources)
    sources["sources"].append(copy.deepcopy(sources["sources"][0]))

    contents = copy.deepcopy(valid_contents)
    contents["content_units"][0]["evidence_refs"] = ["EVD-999"]

    artifacts = {
        "00": ParsedArtifact("00", Path("0001_00_sources.json"), sources),
        "10": ParsedArtifact("10", Path("0001_10_contents.json"), contents),
    }
    issues = validate_references("0001", artifacts)
    messages = {issue.message for issue in issues}

    assert "duplicate source_id: SRC-001" in messages
    assert "dangling evidence reference: EVD-999" in messages


def _dimension(analysis, dimension_id):
    return next(item for item in analysis["dimensions"] if item["dimension_id"] == dimension_id)


def test_taxonomy_valid_fixture(valid_analysis, taxonomy_catalog):
    assert validate_taxonomy(valid_analysis, taxonomy_catalog) == ()


def test_taxonomy_dimension_exact_set(valid_analysis, taxonomy_catalog):
    analysis = copy.deepcopy(valid_analysis)
    analysis["dimensions"] = analysis["dimensions"][:-1]
    issues = validate_taxonomy(analysis, taxonomy_catalog)
    assert {issue.rule_id for issue in issues} == {"V-DIM-001"}


def test_taxonomy_unknown_code(valid_analysis, taxonomy_catalog):
    analysis = copy.deepcopy(valid_analysis)
    dim = _dimension(analysis, "D02")
    dim["primary"] = {"code_id": "D02.BAD.UNKNOWN", "parent_id": "D02.BAD"}
    issues = validate_taxonomy(analysis, taxonomy_catalog)
    assert "V-CODE-001" in {issue.rule_id for issue in issues}


def test_taxonomy_wrong_parent(valid_analysis, taxonomy_catalog):
    analysis = copy.deepcopy(valid_analysis)
    dim = _dimension(analysis, "D02")
    dim["primary"]["parent_id"] = "D02.WRONG"
    issues = validate_taxonomy(analysis, taxonomy_catalog)
    assert "V-PARENT-001" in {issue.rule_id for issue in issues}


def test_taxonomy_invalid_status_combination(valid_analysis, taxonomy_catalog):
    analysis = copy.deepcopy(valid_analysis)
    dim = _dimension(analysis, "D07")
    dim["status"] = "U"
    issues = validate_taxonomy(analysis, taxonomy_catalog)
    assert "V-STATUS-001" in {issue.rule_id for issue in issues}


def test_taxonomy_wrong_dimension(valid_analysis, taxonomy_catalog):
    analysis = copy.deepcopy(valid_analysis)
    d02 = _dimension(analysis, "D02")
    d03 = _dimension(valid_analysis, "D03")
    d02["primary"] = copy.deepcopy(d03["primary"])
    issues = validate_taxonomy(analysis, taxonomy_catalog)
    assert "V-CODE-002" in {issue.rule_id for issue in issues}


def test_validate_entry_through_00_does_not_require_downstream(tmp_path, monkeypatch, valid_sources):
    import json
    from src.validation import validate_entry as ve

    root = tmp_path / "article"
    canonical = root / "docs/10_each_lore/0001"
    canonical.mkdir(parents=True)
    (canonical / "0001_00_sources.json").write_text(
        json.dumps(valid_sources, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(ve, "ARTICLE_ROOT", root)
    monkeypatch.setattr(ve, "CANONICAL_ROOT", root / "docs/10_each_lore")
    monkeypatch.setattr(
        ve,
        "SCHEMA_ROOT",
        Path(__file__).resolve().parents[1] / "docs/10_each_lore/0000_tutorial/schemas",
    )

    result = ve.validate_entry("0001", through="00")
    assert result["result"] == "PASS"
    assert result["through"] == "00"


def test_validate_entry_default_requires_full_chain(tmp_path, monkeypatch, valid_sources):
    import json
    from src.validation import validate_entry as ve

    root = tmp_path / "article"
    canonical = root / "docs/10_each_lore/0001"
    canonical.mkdir(parents=True)
    (canonical / "0001_00_sources.json").write_text(
        json.dumps(valid_sources, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(ve, "ARTICLE_ROOT", root)
    monkeypatch.setattr(ve, "CANONICAL_ROOT", root / "docs/10_each_lore")
    monkeypatch.setattr(
        ve,
        "SCHEMA_ROOT",
        Path(__file__).resolve().parents[1] / "docs/10_each_lore/0000_tutorial/schemas",
    )

    result = ve.validate_entry("0001")
    assert result["result"] == "FAIL"
    assert any("file not found" in issue["message"] for issue in result["errors"])


def test_taxonomy_gap_allows_inferred_dimension_without_primary(
    valid_analysis, taxonomy_catalog
):
    data = copy.deepcopy(valid_analysis)
    target = next(item for item in data["dimensions"] if item["dimension_id"] == "D13")
    target["status"] = "I"
    target["primary"] = None
    target["secondary"] = []
    target["taxonomy_gap"] = {
        "present": True,
        "description": "Evidence supports an inferred mechanism, but no current taxonomy code represents it.",
    }

    issues = validate_taxonomy(data, taxonomy_catalog)
    assert not issues


def test_inferred_dimension_without_primary_requires_taxonomy_gap(
    valid_analysis, taxonomy_catalog
):
    data = copy.deepcopy(valid_analysis)
    target = next(item for item in data["dimensions"] if item["dimension_id"] == "D13")
    target["status"] = "I"
    target["primary"] = None
    target["secondary"] = []
    target["taxonomy_gap"] = {"present": False}

    issues = validate_taxonomy(data, taxonomy_catalog)
    assert any(issue.rule_id == "V-STATUS-001" for issue in issues)
