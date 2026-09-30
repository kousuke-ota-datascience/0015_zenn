from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.orchestration import codex_cli
from src.orchestration.codex_cli import RuntimeResult, RuntimeSession
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

    monkeypatch.setattr(codex_cli, "WORKFLOW_20", workflow20)
    monkeypatch.setattr(codex_cli, "TAXONOMY", taxonomy)
    monkeypatch.setattr(codex_cli, "CODING_RULES", coding)
    monkeypatch.setattr(codex_cli, "SCHEMA_ROOT", schemas)
    monkeypatch.setattr(codex_cli, "repository_root", lambda: tmp_path)

    calls = []

    def frozen(_repo, commit, blob, path):
        calls.append((commit, blob, path))
        return json.dumps({"artifact_path": path})

    monkeypatch.setattr(codex_cli, "_frozen_file", frozen)

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

    context = codex_cli.build_reviewer_context("0001", cycle)

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
        return RuntimeSession("creator", f"creator:{entry_id}")

    def create_reviewer_session(self, entry_id, review_seq):
        session_id = f"reviewer:{entry_id}:{review_seq}:{len(self.reviewer_ids)}"
        self.reviewer_ids.append(session_id)
        return RuntimeSession("reviewer", session_id)

    def run_creator(self, session, prompt):
        return RuntimeResult("creator done")

    def run_reviewer(self, session, context, prompt):
        return RuntimeResult(
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


class _RetryRuntime(_FakeRuntime):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = iter(outputs)
        self.reviewer_threads = []

    def run_reviewer(self, session, context, prompt):
        assert session.thread_id is None
        session.thread_id = f"thread:{session.session_id}"
        self.reviewer_threads.append(session.thread_id)
        return RuntimeResult(next(self.outputs))


def _stub_review_cycle_dependencies(monkeypatch, *, write_review_cycle):
    monkeypatch.setattr(
        pipeline,
        "prepare_review_cycle",
        lambda _entry: SimpleNamespace(
            review_seq=7,
            to_dict=lambda: {
                "entry_id": "0001",
                "review_seq": 7,
                "targets": {
                    a: {
                        "artifact_path": f"{a}.json",
                        "commit_sha": "a" * 40,
                        "blob_sha": "b" * 40,
                    }
                    for a in pipeline.ARTIFACTS
                },
            },
        ),
    )
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
    monkeypatch.setattr(pipeline, "write_review_cycle", write_review_cycle)
    monkeypatch.setattr(pipeline, "_commit_review_cycle", lambda _written: "commit")


def _successful_write(entry, cycle, reviews):
    return {
        "entry_id": entry,
        "review_seq": cycle["review_seq"],
        "paths": {a: f"reviews/{a}.json" for a in pipeline.ARTIFACTS},
        "verdicts": {a: "Pass" for a in pipeline.ARTIFACTS},
    }


def test_review_retry_after_json_parse_failure_uses_fresh_reviewer(monkeypatch):
    _stub_review_cycle_dependencies(
        monkeypatch,
        write_review_cycle=_successful_write,
    )
    valid = json.dumps(
        {"reviews": {"00": {}, "10": {}, "20": {}}},
        ensure_ascii=False,
    )
    runtime = _RetryRuntime(["not-json", valid])

    result = pipeline._run_review_cycle("0001", runtime)

    assert result["review_session_id"] == runtime.reviewer_ids[1]
    assert runtime.reviewer_ids == [
        "reviewer:0001:7:0",
        "reviewer:0001:7:1",
    ]
    assert len(set(runtime.reviewer_threads)) == 2


def test_review_retry_after_writer_value_error_uses_fresh_reviewer(monkeypatch):
    calls = 0

    def flaky_write(entry, cycle, reviews):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("schema validation failed")
        return _successful_write(entry, cycle, reviews)

    _stub_review_cycle_dependencies(
        monkeypatch,
        write_review_cycle=flaky_write,
    )
    valid = json.dumps(
        {"reviews": {"00": {}, "10": {}, "20": {}}},
        ensure_ascii=False,
    )
    runtime = _RetryRuntime([valid, valid])

    result = pipeline._run_review_cycle("0001", runtime)

    assert calls == 2
    assert result["review_session_id"] == runtime.reviewer_ids[1]
    assert runtime.reviewer_ids == [
        "reviewer:0001:7:0",
        "reviewer:0001:7:1",
    ]
    assert len(set(runtime.reviewer_threads)) == 2


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



def test_parse_codex_jsonl_extracts_thread_and_last_agent_message():
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "first"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "final"},
                }
            ),
        ]
    )

    thread_id, output = codex_cli._parse_codex_jsonl(stdout)

    assert thread_id == "thread-1"
    assert output == "final"


def test_codex_creator_resumes_the_same_thread(monkeypatch):
    monkeypatch.setattr(codex_cli.shutil, "which", lambda _binary: "/usr/bin/codex")
    runtime = codex_cli.CodexCLIRuntime()
    monkeypatch.setattr(runtime, "_ensure_chatgpt_login", lambda: None)
    monkeypatch.setattr(codex_cli, "_read_text", lambda _path: "workflow-10")

    calls = []

    def invoke(args, prompt, *, cwd):
        calls.append((list(args), prompt, cwd))
        return "thread-creator", "done"

    monkeypatch.setattr(runtime, "_invoke", invoke)

    session = runtime.create_creator_session("0001")
    runtime.run_creator(session, "first task")
    runtime.run_creator(session, "correction task")

    assert session.thread_id == "thread-creator"
    assert 'sandbox_mode="danger-full-access"' in calls[0][0]
    assert 'web_search="live"' in calls[0][0]
    assert 'approval_policy="never"' in calls[0][0]
    assert "--search" not in calls[0][0]
    assert "--ask-for-approval" not in calls[0][0]
    assert "resume" not in calls[0][0]
    resume_index = calls[1][0].index("resume")
    assert calls[1][0][resume_index + 1] == "thread-creator"


def test_codex_reviewer_is_fresh_read_only_and_isolated(monkeypatch):
    monkeypatch.setattr(codex_cli.shutil, "which", lambda _binary: "/usr/bin/codex")
    runtime = codex_cli.CodexCLIRuntime()
    monkeypatch.setattr(runtime, "_ensure_chatgpt_login", lambda: None)

    calls = []

    def invoke(args, prompt, *, cwd):
        calls.append((list(args), prompt, cwd))
        assert (cwd / ".git").exists()
        return "thread-review", '{"reviews":{"00":{},"10":{},"20":{}}}'

    monkeypatch.setattr(runtime, "_invoke", invoke)

    session = runtime.create_reviewer_session("0001", 4)
    result = runtime.run_reviewer(
        session,
        {"entry_id": "0001", "review_seq": 4},
        "review this frozen cycle",
    )

    args, prompt, cwd = calls[0]
    assert result.output.startswith('{"reviews"')
    assert session.thread_id == "thread-review"
    assert 'sandbox_mode="read-only"' in args
    assert 'web_search="live"' in args
    assert 'approval_policy="never"' in args
    assert "--search" not in args
    assert "--ephemeral" not in args
    assert "--skip-git-repo-check" not in args
    assert "--ignore-user-config" not in args
    assert "--ignore-rules" not in args
    assert "--ask-for-approval" not in args
    assert "resume" not in args
    assert cwd != codex_cli.ARTICLE_ROOT
    assert "Creator conversation" in prompt


def test_codex_runtime_rejects_non_chatgpt_login(monkeypatch):
    monkeypatch.setattr(codex_cli.shutil, "which", lambda _binary: "/usr/bin/codex")

    class Proc:
        returncode = 0
        stdout = ""
        stderr = "Logged in using an API key - sk-...redacted"

    monkeypatch.setattr(codex_cli.subprocess, "run", lambda *args, **kwargs: Proc())

    runtime = codex_cli.CodexCLIRuntime()

    with pytest.raises(codex_cli.CodexCLIAuthError, match="authenticated with ChatGPT"):
        runtime.create_creator_session("0001")



def test_resolve_push_remote_uses_single_non_origin_remote(monkeypatch, tmp_path):
    monkeypatch.delenv("WORKFLOW_GIT_REMOTE", raising=False)
    monkeypatch.setattr(
        pipeline,
        "_git",
        lambda _repo, *args: "0015_zenn" if args == ("remote",) else "",
    )
    monkeypatch.setattr(pipeline, "_git_optional", lambda _repo, *args: None)

    assert pipeline._resolve_push_remote(tmp_path) == "0015_zenn"


def test_resolve_push_remote_prefers_explicit_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKFLOW_GIT_REMOTE", "work")
    monkeypatch.setattr(
        pipeline,
        "_git",
        lambda _repo, *args: "origin\nwork" if args == ("remote",) else "",
    )

    assert pipeline._resolve_push_remote(tmp_path) == "work"


def test_resolve_push_remote_uses_tracking_remote(monkeypatch, tmp_path):
    monkeypatch.delenv("WORKFLOW_GIT_REMOTE", raising=False)
    monkeypatch.setattr(
        pipeline,
        "_git",
        lambda _repo, *args: "origin\n0015_zenn" if args == ("remote",) else "",
    )

    def optional(_repo, *args):
        if args == ("branch", "--show-current"):
            return "workflow/0025"
        if args == ("config", "--get", "branch.workflow/0025.remote"):
            return "0015_zenn"
        return None

    monkeypatch.setattr(pipeline, "_git_optional", optional)

    assert pipeline._resolve_push_remote(tmp_path) == "0015_zenn"


def test_resolve_push_remote_requires_override_when_ambiguous(monkeypatch, tmp_path):
    monkeypatch.delenv("WORKFLOW_GIT_REMOTE", raising=False)
    monkeypatch.setattr(
        pipeline,
        "_git",
        lambda _repo, *args: "alpha\nbeta" if args == ("remote",) else "",
    )
    monkeypatch.setattr(pipeline, "_git_optional", lambda _repo, *args: None)

    with pytest.raises(RuntimeError, match="WORKFLOW_GIT_REMOTE"):
        pipeline._resolve_push_remote(tmp_path)
