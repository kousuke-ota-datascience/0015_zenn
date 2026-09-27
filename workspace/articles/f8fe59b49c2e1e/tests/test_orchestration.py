from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.orchestration import agents_api
from src.orchestration.agents_api import AgentRunResult, AgentSession
from src.orchestration import entry_pipeline as pipeline


def _git_snapshot(*, commit: str = "c1", blob: str = "b1"):
    return SimpleNamespace(
        repository_root=Path("."),
        artifacts={
            artifact: SimpleNamespace(
                exists=True,
                commit_sha=commit,
                blob_sha=blob,
            )
            for artifact in pipeline.ARTIFACTS
        },
    )


def _reviews(verdicts, *, seq=1, target_commit="c1", target_blob="b1"):
    facts = {}
    all_reviews = []
    for artifact in pipeline.ARTIFACTS:
        fact = SimpleNamespace(
            artifact=artifact,
            review_seq=seq,
            target_commit_sha=target_commit,
            target_blob_sha=target_blob,
            verdict=verdicts[artifact],
            review_path=Path(f"Review_{artifact}.json"),
        )
        facts[artifact] = fact
        all_reviews.append(fact)
    return SimpleNamespace(
        issues=(),
        all_reviews=tuple(all_reviews),
        latest=facts,
    )


def test_classify_complete_requires_exact_current_targets(monkeypatch):
    monkeypatch.setattr(pipeline, "load_entry_git_state", lambda _entry: _git_snapshot())
    monkeypatch.setattr(
        pipeline,
        "load_entry_review_state",
        lambda _entry: _reviews({a: "Pass" for a in pipeline.ARTIFACTS}),
    )
    monkeypatch.setattr(
        pipeline,
        "validate_entry",
        lambda *_args, **_kwargs: {"result": "PASS"},
    )
    monkeypatch.setattr(
        pipeline,
        "compare_commits",
        lambda left, right, **_kwargs: "exact" if left == right else "left_ancestor",
    )

    result = pipeline.classify_entry_state("0001")

    assert result.state == pipeline.COMPLETE
    assert result.validation_result == "PASS"
    assert set(result.review_target_relation.values()) == {"exact"}


@pytest.mark.parametrize(
    ("target_commit", "verdicts", "expected"),
    [
        ("c1", {"00": "Minor", "10": "Pass", "20": "Pass"}, pipeline.CORRECTION_REQUIRED),
        ("old", {"00": "Pass", "10": "Pass", "20": "Pass"}, pipeline.STALE_CURRENT),
        ("old", {"00": "Minor", "10": "Pass", "20": "Pass"}, pipeline.REVIEW_READY),
    ],
)
def test_classify_review_relation_states(
    monkeypatch,
    target_commit,
    verdicts,
    expected,
):
    monkeypatch.setattr(pipeline, "load_entry_git_state", lambda _entry: _git_snapshot())
    monkeypatch.setattr(
        pipeline,
        "load_entry_review_state",
        lambda _entry: _reviews(verdicts, target_commit=target_commit),
    )
    monkeypatch.setattr(
        pipeline,
        "validate_entry",
        lambda *_args, **_kwargs: {"result": "PASS"},
    )
    monkeypatch.setattr(
        pipeline,
        "compare_commits",
        lambda left, right, **_kwargs: "exact" if left == right else "left_ancestor",
    )

    result = pipeline.classify_entry_state("0001")
    assert result.state == expected


def test_reviewer_context_is_frozen_and_creator_context_is_absent(
    monkeypatch,
    tmp_path,
):
    workflow20 = tmp_path / "workflow20.md"
    workflow20.write_text("workflow-20", encoding="utf-8")
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text('{"taxonomy":"current"}', encoding="utf-8")
    coding = tmp_path / "coding.md"
    coding.write_text("coding-rules", encoding="utf-8")
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    (schemas / "review.schema.json").write_text(
        '{"type":"object"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(agents_api, "WORKFLOW_20", workflow20)
    monkeypatch.setattr(agents_api, "TAXONOMY", taxonomy)
    monkeypatch.setattr(agents_api, "CODING_RULES", coding)
    monkeypatch.setattr(agents_api, "SCHEMA_ROOT", schemas)
    monkeypatch.setattr(agents_api, "repository_root", lambda: tmp_path)

    calls = []

    def frozen(_repo, commit, blob, path):
        calls.append((commit, blob, path))
        return json.dumps({"artifact_path": path})

    monkeypatch.setattr(agents_api, "_frozen_file", frozen)

    cycle = {
        "entry_id": "0001",
        "review_seq": 7,
        "targets": {
            artifact: {
                "artifact_path": f"canonical/{artifact}.json",
                "commit_sha": f"commit-{artifact}",
                "blob_sha": f"blob-{artifact}",
            }
            for artifact in ("00", "10", "20")
        },
    }

    context = agents_api.build_reviewer_context("0001", cycle)

    assert len(calls) == 3
    assert set(context["frozen_canonical"]) == {"00", "10", "20"}
    assert context["context_contract"]["creator_conversation_history"] == "FORBIDDEN"
    assert context["context_contract"]["creator_rationale"] == "FORBIDDEN"
    assert context["context_contract"]["creator_intermediate_notes"] == "FORBIDDEN"
    serialized = json.dumps(context, ensure_ascii=False)
    assert "creator_session" not in serialized
    assert "creator_history" not in serialized


class _FakeRuntime:
    def __init__(self):
        self.reviewer_ids = []

    def create_creator_session(self, entry_id):
        return AgentSession("creator", f"creator:{entry_id}", object(), object())

    def create_reviewer_session(self, entry_id, review_seq):
        session_id = f"reviewer:{entry_id}:{review_seq}:{len(self.reviewer_ids)}"
        self.reviewer_ids.append(session_id)
        return AgentSession("reviewer", session_id, object(), object())

    def run_creator(self, session, prompt):
        return AgentRunResult("creator done")

    def run_reviewer(self, session, context, prompt):
        return AgentRunResult(
            json.dumps(
                {"reviews": {"00": {}, "10": {}, "20": {}}},
                ensure_ascii=False,
            )
        )


def test_each_review_cycle_gets_a_fresh_reviewer_session(monkeypatch):
    counter = iter((1, 2))

    def prepare(_entry):
        seq = next(counter)
        return SimpleNamespace(
            review_seq=seq,
            to_dict=lambda seq=seq: {
                "entry_id": "0001",
                "review_seq": seq,
                "targets": {
                    a: {
                        "artifact_path": f"{a}.json",
                        "commit_sha": "a" * 40,
                        "blob_sha": "b" * 40,
                    }
                    for a in pipeline.ARTIFACTS
                },
            },
        )

    monkeypatch.setattr(pipeline, "prepare_review_cycle", prepare)
    monkeypatch.setattr(
        pipeline,
        "sync_controlplane",
        lambda *_args, **_kwargs: {"result": "PASS", "verified": True},
    )
    monkeypatch.setattr(
        pipeline,
        "build_reviewer_context",
        lambda entry, cycle: {"entry_id": entry, "review_seq": cycle["review_seq"]},
    )
    monkeypatch.setattr(
        pipeline,
        "write_review_cycle",
        lambda entry, cycle, reviews: {
            "entry_id": entry,
            "review_seq": cycle["review_seq"],
            "paths": {a: f"reviews/{a}.json" for a in pipeline.ARTIFACTS},
            "verdicts": {a: "Pass" for a in pipeline.ARTIFACTS},
        },
    )
    monkeypatch.setattr(pipeline, "_commit_review_cycle", lambda _written: "commit")

    runtime = _FakeRuntime()
    first = pipeline._run_review_cycle("0001", runtime)
    second = pipeline._run_review_cycle("0001", runtime)

    assert first["review_session_id"] != second["review_session_id"]
    assert runtime.reviewer_ids == [
        "reviewer:0001:1:0",
        "reviewer:0001:2:1",
    ]


def test_review_cycle_limit_blocks_after_five(monkeypatch):
    classification = pipeline.EntryClassification(
        "0001",
        pipeline.REVIEW_READY,
        validation_result="PASS",
    )
    monkeypatch.setattr(
        pipeline,
        "sync_controlplane",
        lambda *_args, **_kwargs: {"result": "PASS", "verified": True},
    )
    monkeypatch.setattr(
        pipeline,
        "classify_entry_state",
        lambda _entry: classification,
    )
    calls = []
    monkeypatch.setattr(
        pipeline,
        "_run_review_cycle",
        lambda entry, runtime: calls.append(entry),
    )

    result = pipeline.run_entry_pipeline("0001", runtime=_FakeRuntime())

    assert result.result == "BLOCKED"
    assert result.reason_code == "MAX_REVIEW_CYCLES"
    assert result.review_cycles_created == pipeline.MAX_REVIEW_CYCLES_PER_RUN
    assert len(calls) == pipeline.MAX_REVIEW_CYCLES_PER_RUN


def test_complete_entry_is_idempotent_and_does_not_create_sessions(monkeypatch):
    classification = pipeline.EntryClassification(
        "0001",
        pipeline.COMPLETE,
        latest_review_seq=3,
        validation_result="PASS",
        review_verdicts={a: "Pass" for a in pipeline.ARTIFACTS},
        review_target_relation={a: "exact" for a in pipeline.ARTIFACTS},
    )
    monkeypatch.setattr(
        pipeline,
        "sync_controlplane",
        lambda *_args, **_kwargs: {"result": "PASS", "verified": True},
    )
    monkeypatch.setattr(
        pipeline,
        "classify_entry_state",
        lambda _entry: classification,
    )
    monkeypatch.setattr(
        pipeline,
        "validate_entry",
        lambda *_args, **_kwargs: {"result": "PASS"},
    )

    class NoSessionRuntime(_FakeRuntime):
        def create_creator_session(self, entry_id):
            raise AssertionError("COMPLETE entry must not create Creator session")

        def create_reviewer_session(self, entry_id, review_seq):
            raise AssertionError("COMPLETE entry must not create Reviewer session")

    result = pipeline.run_entry_pipeline("0001", runtime=NoSessionRuntime())

    assert result.result == "PASS"
    assert result.canonical_review_exact is True
    assert result.validation_pass is True
    assert result.control_plane_synchronized is True
