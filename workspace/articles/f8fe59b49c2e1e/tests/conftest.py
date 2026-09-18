from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ARTICLE_ROOT = Path(__file__).resolve().parents[1]
if str(ARTICLE_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTICLE_ROOT))

SCHEMA_ROOT = ARTICLE_ROOT / "docs/10_each_lore/0000_tutorial/schemas"
CATALOG_PATH = ARTICLE_ROOT / "docs/00_research_overview/taxonomy_catalog.json"

STRUCTURE_KEYS = (
    "primary_actors",
    "affected_targets",
    "entry_trigger",
    "causal_agent",
    "mechanism",
    "state_transition",
    "temporal_structure",
    "rules_and_taboos",
    "avoidance_control",
    "terminal_state",
    "uncertainties",
    "variant_boundaries",
)


@pytest.fixture
def valid_sources():
    return {
        "schema_version": "1.0",
        "entry_id": "0001",
        "sources": [
            {
                "source_id": "SRC-001",
                "source_type": "primary_or_contemporary",
                "title": "Fixture Source",
                "locator": "p.1",
                "relation_to_primary": "primary",
                "evidence_role": ["C"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "EVD-001",
                "source_id": "SRC-001",
                "locator": "p.1",
                "content": "Fixture evidence.",
                "representation": "paraphrase",
            }
        ],
    }


@pytest.fixture
def valid_contents():
    return {
        "schema_version": "1.0",
        "entry_id": "0001",
        "lore_name": "Fixture Lore",
        "summary": {
            "narrative": "A compact but sufficient fixture narrative.",
            "coverage_refs": ["CNT-001"],
            "structure": {
                key: {"status": "known", "value": f"{key} value"}
                for key in STRUCTURE_KEYS
            },
        },
        "content_units": [
            {
                "content_id": "CNT-001",
                "type": "claim",
                "text": "Fixture content claim.",
                "evidence_refs": ["EVD-001"],
            }
        ],
        "variants": [
            {
                "variant_id": "VAR-001",
                "kind": "initial",
                "description": "Fixture initial variant.",
                "content_refs": ["CNT-001"],
                "evidence_refs": ["EVD-001"],
            }
        ],
        "uncertainties": [
            {
                "text": "Fixture uncertainty.",
                "evidence_refs": ["EVD-001"],
                "content_refs": ["CNT-001"],
            }
        ],
    }


@pytest.fixture
def taxonomy_catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def valid_analysis(taxonomy_catalog):
    dimensions = []
    for dimension_id in sorted(taxonomy_catalog["dimensions"]):
        candidates = sorted(
            code_id
            for code_id, record in taxonomy_catalog["codes"].items()
            if record["dimension"] == dimension_id
        )
        code_id = candidates[0]
        record = taxonomy_catalog["codes"][code_id]
        primary = {"code_id": code_id}
        if record["parent_id"] is not None:
            primary["parent_id"] = record["parent_id"]
        dimensions.append(
            {
                "dimension_id": dimension_id,
                "status": "D",
                "primary": primary,
                "secondary": [],
                "rationale": f"{dimension_id} fixture rationale.",
                "manifestation": f"{dimension_id} fixture manifestation.",
                "content_refs": ["CNT-001"],
                "evidence_refs": ["EVD-001"],
            }
        )
    return {
        "schema_version": "1.0",
        "entry_id": "0001",
        "macro_category": "Fixture Category",
        "entry_type": "Fixture Type",
        "version_scope": {
            "description": "Fixture scope.",
            "included_variants": ["VAR-001"],
            "excluded_variants": [],
            "content_refs": ["CNT-001"],
        },
        "dimensions": dimensions,
    }


def _check(status="PASS", notes=None):
    value = {"status": status}
    if notes is not None:
        value["notes"] = notes
    return value


@pytest.fixture
def review_bodies():
    review_00 = {
        "checks": {
            key: _check()
            for key in (
                "source_assessment",
                "evidence_integrity",
                "source_evidence_support",
                "primary_source_relation",
                "earliest_attestation",
                "uncertainty_preservation",
            )
        },
        "findings": [],
    }

    probes = [
        {
            "probe_id": f"P{i:02d}",
            "blind": f"blind {i}",
            "reference": f"reference {i}",
            "difference": "NONE",
        }
        for i in range(1, 13)
    ]
    review_10 = {
        "checks": {
            key: _check()
            for key in (
                "evidence_content_support",
                "narrative_reconstruction",
                "summary_reconstruction",
                "structural_probe",
                "variant_boundary",
                "uncertainty_preservation",
            )
        },
        "reconstruction": {
            "blind_decode": "Fixture blind decode.",
            "reference_story": "Fixture reference story.",
            "coverage_audit": [
                {
                    "content_ref": "CNT-001",
                    "salient_meaning": "Fixture content claim.",
                    "blind_reconstruction": "Fixture content claim.",
                    "difference": "NONE",
                }
            ],
            "probes": probes,
            "analysis_invariance": {"status": "PASS", "notes": "stable"},
            "verdict": "PASS",
        },
        "findings": [],
    }

    review_checks = {
        key: _check()
        for key in (
            "version_scope",
            "dimension_semantics",
            "taxonomy_gap",
            "causality",
            "semantic_traceability",
            "analysis_invariance",
            "status_reasoning",
        )
    }
    review_checks.update(
        {
            key: _check(notes="audited")
            for key in (
                "d12_d13_d15_causal_chain",
                "d18_scope_independent_x_evidence",
                "d19_publicability_vs_national_distribution",
                "d20_missing_info_vs_no_one_knows",
                "d21_reality_anchor",
                "evidence_confidence_ceiling",
                "entry_classification",
            )
        }
    )
    review_20 = {
        "checks": review_checks,
        "sensemaking_reconstruction": {
            "model": "Fixture sense-making model.",
            "content_refs": ["CNT-001"],
            "dimension_refs": ["D07"],
            "status": "PASS",
        },
        "findings": [],
    }
    return {"00": review_00, "10": review_10, "20": review_20}


@pytest.fixture
def review_payloads(review_bodies):
    payloads = {}
    for artifact, body in review_bodies.items():
        payloads[artifact] = {
            "schema_version": "1.0",
            "entry_id": "0001",
            "artifact": artifact,
            "review_seq": 1,
            "target": {
                "artifact_path": f"docs/10_each_lore/0001/0001_{artifact}.json",
                "commit_sha": "a" * 40,
                "blob_sha": "b" * 40,
                "reviewed_at": "2026-09-18T00:00:00Z",
            },
            **copy.deepcopy(body),
            "verdict": "Pass",
        }
    return payloads
