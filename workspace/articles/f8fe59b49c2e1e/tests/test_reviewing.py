from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.reviewing import review_writer
from src.status_management import review_state


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_review_state_filename_payload_mismatch(tmp_path, monkeypatch, review_payloads):
    root = tmp_path / "reviews"
    monkeypatch.setattr(review_state, "REVIEW_ROOT", root)
    payload = copy.deepcopy(review_payloads["00"])
    payload["review_seq"] = 2
    _write_json(root / "0001/Review_0001_00_001.json", payload)

    snapshot = review_state.load_entry_review_state("0001")
    assert "review_filename_payload_mismatch:Review_0001_00_001.json" in snapshot.issues


def test_review_state_duplicate_logical_key(tmp_path, monkeypatch, review_payloads):
    root = tmp_path / "reviews"
    monkeypatch.setattr(review_state, "REVIEW_ROOT", root)
    payload = review_payloads["00"]
    _write_json(root / "0001/Review_0001_00_001.json", payload)
    _write_json(root / "0001/Review_0001_00_1.json", payload)

    snapshot = review_state.load_entry_review_state("0001")
    assert "duplicate_review_key:00:1" in snapshot.issues


def test_review_state_schema_invalid(tmp_path, monkeypatch):
    root = tmp_path / "reviews"
    monkeypatch.setattr(review_state, "REVIEW_ROOT", root)
    _write_json(root / "0001/Review_0001_00_001.json", {"entry_id": "0001"})

    snapshot = review_state.load_entry_review_state("0001")
    assert "review_schema_invalid:Review_0001_00_001.json" in snapshot.issues


def test_review_state_missing_target_sha_defensive_branch(tmp_path, monkeypatch):
    root = tmp_path / "reviews"
    monkeypatch.setattr(review_state, "REVIEW_ROOT", root)
    path = root / "0001/Review_0001_00_001.json"
    _write_json(path, {})

    fake = SimpleNamespace(
        ok=True,
        data={
            "entry_id": "0001",
            "artifact": "00",
            "review_seq": 1,
            "target": {"commit_sha": "a" * 40},
            "verdict": "Pass",
        },
    )
    monkeypatch.setattr(review_state, "validate_artifact", lambda *args, **kwargs: fake)
    snapshot = review_state.load_entry_review_state("0001")
    assert "review_missing_target_sha:Review_0001_00_001.json" in snapshot.issues


@pytest.mark.parametrize(
    ("severities", "expected"),
    [
        ([], "Pass"),
        (["Minor"], "Minor"),
        (["Minor", "Moderate"], "Moderate"),
        (["Major", "Minor"], "Major"),
    ],
)
def test_verdict_aggregation(severities, expected):
    findings = [
        {"finding_id": f"F{i:03d}", "severity": severity}
        for i, severity in enumerate(severities, start=1)
    ]
    assert review_writer.aggregate_verdict(findings) == expected


def test_verdict_rejects_duplicate_finding_id():
    findings = [
        {"finding_id": "F001", "severity": "Minor"},
        {"finding_id": "F001", "severity": "Major"},
    ]
    with pytest.raises(ValueError, match="duplicate finding_id"):
        review_writer.aggregate_verdict(findings)


def test_next_seq_requires_complete_prior_cycle(monkeypatch):
    facts = (
        review_state.ReviewFact("00", 1, "a" * 40, "b" * 40, "Pass", Path("a")),
        review_state.ReviewFact("10", 1, "a" * 40, "b" * 40, "Pass", Path("b")),
    )
    snapshot = review_state.ReviewSnapshot("0001", {}, facts, ())
    monkeypatch.setattr(review_writer, "load_entry_review_state", lambda _: snapshot)

    with pytest.raises(RuntimeError, match="incomplete existing Review cycle"):
        review_writer._next_review_seq("0001")


def test_next_seq_shared_across_complete_cycles(monkeypatch):
    facts = tuple(
        review_state.ReviewFact(artifact, seq, "a" * 40, "b" * 40, "Pass", Path(f"{artifact}-{seq}"))
        for seq in (1, 2)
        for artifact in ("00", "10", "20")
    )
    snapshot = review_state.ReviewSnapshot("0001", {}, facts, ())
    monkeypatch.setattr(review_writer, "load_entry_review_state", lambda _: snapshot)
    assert review_writer._next_review_seq("0001") == 3


def test_write_cycle_schema_validates_and_is_append_only(
    tmp_path, monkeypatch, review_bodies
):
    root = tmp_path / "article"
    review_root = root / "reviews/10_each_lore"
    monkeypatch.setattr(review_writer, "ARTICLE_ROOT", root)
    monkeypatch.setattr(review_writer, "REVIEW_ROOT", review_root)
    monkeypatch.setattr(review_writer, "_validate_cycle_is_current", lambda cycle: None)
    monkeypatch.setattr(
        review_writer, "_validate_review10_coverage_contract", lambda entry_id, body: None
    )

    targets = {
        artifact: review_writer.TargetSnapshot(
            artifact_path=f"docs/{artifact}.json",
            commit_sha="a" * 40,
            blob_sha="b" * 40,
        )
        for artifact in ("00", "10", "20")
    }
    cycle = review_writer.ReviewCycleSnapshot(
        "0001", 1, "2026-09-18T00:00:00Z", targets
    )

    result = review_writer.write_review_cycle(
        "0001", cycle=cycle, reviews=review_bodies
    )
    assert result["result"] == "WRITTEN"
    assert result["verdicts"] == {"00": "Pass", "10": "Pass", "20": "Pass"}
    for artifact in ("00", "10", "20"):
        path = review_root / f"0001/Review_0001_{artifact}_001.json"
        assert path.is_file()

    with pytest.raises(FileExistsError, match="overwrite is forbidden"):
        review_writer.write_review_cycle(
            "0001", cycle=cycle, reviews=review_bodies
        )


def test_writer_rejects_managed_fields(review_bodies, monkeypatch):
    monkeypatch.setattr(
        review_writer, "_validate_review10_coverage_contract", lambda entry_id, body: None
    )
    bodies = copy.deepcopy(review_bodies)
    bodies["00"]["verdict"] = "Major"
    targets = {
        artifact: review_writer.TargetSnapshot(
            artifact_path=f"docs/{artifact}.json",
            commit_sha="a" * 40,
            blob_sha="b" * 40,
        )
        for artifact in ("00", "10", "20")
    }
    cycle = review_writer.ReviewCycleSnapshot(
        "0001", 1, "2026-09-18T00:00:00Z", targets
    )
    with pytest.raises(ValueError, match="writer-managed fields"):
        review_writer._build_payloads(cycle, bodies)


def test_review10_coverage_contract_requires_exact_refs(
    tmp_path, monkeypatch, review_bodies
):
    path = tmp_path / "0001_10_contents.json"
    _write_json(
        path,
        {
            "summary": {
                "coverage_refs": ["CNT-001", "CNT-002"],
            }
        },
    )
    monkeypatch.setattr(
        review_writer,
        "load_entry_git_state",
        lambda entry_id: SimpleNamespace(
            artifacts={"10": SimpleNamespace(path=path)}
        ),
    )

    body = copy.deepcopy(review_bodies["10"])
    with pytest.raises(ValueError, match="exactly match summary.coverage_refs"):
        review_writer._validate_review10_coverage_contract("0001", body)


def test_review10_coverage_loss_requires_finding(
    tmp_path, monkeypatch, review_bodies
):
    path = tmp_path / "0001_10_contents.json"
    _write_json(path, {"summary": {"coverage_refs": ["CNT-001"]}})
    monkeypatch.setattr(
        review_writer,
        "load_entry_git_state",
        lambda entry_id: SimpleNamespace(
            artifacts={"10": SimpleNamespace(path=path)}
        ),
    )

    body = copy.deepcopy(review_bodies["10"])
    body["reconstruction"]["coverage_audit"][0]["difference"] = "LOSS"
    with pytest.raises(ValueError, match="requires reconstruction.verdict=FINDING"):
        review_writer._validate_review10_coverage_contract("0001", body)


def test_review10_coverage_contract_accepts_complete_pass(
    tmp_path, monkeypatch, review_bodies
):
    path = tmp_path / "0001_10_contents.json"
    _write_json(path, {"summary": {"coverage_refs": ["CNT-001"]}})
    monkeypatch.setattr(
        review_writer,
        "load_entry_git_state",
        lambda entry_id: SimpleNamespace(
            artifacts={"10": SimpleNamespace(path=path)}
        ),
    )
    review_writer._validate_review10_coverage_contract(
        "0001", copy.deepcopy(review_bodies["10"])
    )
