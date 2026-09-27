"""Executable Workflow 00 entry pipeline.

Public interface: one Entry_ID.  Semantic work is delegated to isolated
Workflow 10 / Workflow 20 Agents SDK sessions; deterministic enforcement is
delegated to the existing validation, review_writer and control-plane modules.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

from src.orchestration.agents_api import (
    AgentsRuntime,
    OpenAIAgentsAPI,
    build_reviewer_context,
)
from src.reviewing.review_writer import prepare_review_cycle, write_review_cycle
from src.status_management.git_state import compare_commits, load_entry_git_state
from src.status_management.review_state import ReviewFact, load_entry_review_state
from src.status_management.sync_controlplane import sync_controlplane
from src.validation.validate_entry import validate_entry

ARTICLE_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ("00", "10", "20")
PASS_VERDICT = "Pass"

MAX_REVIEW_CYCLES_PER_RUN = 5
MAX_PIPELINE_STEPS_PER_RUN = 24
MAX_REVIEW_RESPONSE_ATTEMPTS = 3

NEW = "NEW"
PARTIAL = "PARTIAL"
REVIEW_READY = "REVIEW_READY"
CORRECTION_REQUIRED = "CORRECTION_REQUIRED"
STALE_CURRENT = "STALE_CURRENT"
COMPLETE = "COMPLETE"
BLOCKED_STATE = "BLOCKED_STATE"
ERROR_STATE = "ERROR"


@dataclass(frozen=True)
class EntryClassification:
    entry_id: str
    state: str
    reason: str | None = None
    latest_review_seq: int | None = None
    validation_result: str | None = None
    review_verdicts: dict[str, str] | None = None
    review_target_relation: dict[str, str] | None = None


@dataclass(frozen=True)
class PipelineResult:
    result: str
    entry_id: str
    state: str
    latest_review_seq: int | None
    review_cycles_created: int
    reason_code: str | None = None
    messages: tuple[str, ...] = ()
    canonical_review_exact: bool = False
    validation_pass: bool = False
    control_plane_synchronized: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["messages"] = list(self.messages)
        return data


def _valid_entry_id(entry_id: str) -> bool:
    return len(entry_id) == 4 and entry_id.isdigit()


def _latest_cycle(reviews: Any) -> tuple[int | None, dict[str, ReviewFact] | None, str | None]:
    if not reviews.all_reviews:
        return None, None, None

    seqs = sorted({fact.review_seq for fact in reviews.all_reviews})
    latest_seq = seqs[-1]
    facts = {
        fact.artifact: fact
        for fact in reviews.all_reviews
        if fact.review_seq == latest_seq
    }
    if set(facts) != set(ARTIFACTS):
        return latest_seq, None, f"incomplete_review_cycle:{latest_seq}"

    if any(reviews.latest.get(a) != facts[a] for a in ARTIFACTS):
        return latest_seq, None, f"latest_review_cycle_not_shared:{latest_seq}"
    return latest_seq, facts, None


def _target_relations(git: Any, facts: Mapping[str, ReviewFact]) -> dict[str, str]:
    relations: dict[str, str] = {}
    for artifact in ARTIFACTS:
        current = git.artifacts[artifact]
        fact = facts[artifact]
        if not current.commit_sha or not current.blob_sha:
            relations[artifact] = "missing"
            continue
        commit_relation = compare_commits(
            fact.target_commit_sha,
            current.commit_sha,
            repo=git.repository_root,
        )
        if (
            commit_relation == "exact"
            and fact.target_blob_sha == current.blob_sha
        ):
            relations[artifact] = "exact"
        elif commit_relation == "left_ancestor":
            relations[artifact] = "stale"
        elif commit_relation == "exact":
            relations[artifact] = "diverged_blob"
        else:
            relations[artifact] = commit_relation
    return relations


def classify_entry_state(entry_id: str) -> EntryClassification:
    """Classify current canonical/Review facts without mutating them."""
    if not _valid_entry_id(entry_id):
        return EntryClassification(entry_id, ERROR_STATE, "INVALID_ENTRY_ID")

    try:
        git = load_entry_git_state(entry_id)
        reviews = load_entry_review_state(entry_id)
    except Exception as exc:
        return EntryClassification(
            entry_id,
            ERROR_STATE,
            f"STATE_READ_ERROR:{type(exc).__name__}:{exc}",
        )

    if reviews.issues:
        return EntryClassification(
            entry_id,
            BLOCKED_STATE,
            "REVIEW_HISTORY_INVALID:" + ",".join(reviews.issues),
        )

    present = [a for a in ARTIFACTS if git.artifacts[a].exists]
    if not present:
        return EntryClassification(entry_id, NEW)

    committed = [
        a
        for a in ARTIFACTS
        if git.artifacts[a].exists
        and git.artifacts[a].commit_sha
        and git.artifacts[a].blob_sha
    ]
    if len(present) < 3 or len(committed) < 3:
        return EntryClassification(entry_id, PARTIAL, "CANONICAL_CHAIN_INCOMPLETE")

    validation = validate_entry(entry_id, through="20")
    validation_result = str(validation.get("result"))
    if validation_result == "ERROR":
        return EntryClassification(
            entry_id,
            ERROR_STATE,
            "VALIDATION_ERROR",
            validation_result=validation_result,
        )
    if validation_result != "PASS":
        return EntryClassification(
            entry_id,
            PARTIAL,
            "VALIDATION_FAIL",
            validation_result=validation_result,
        )

    seq, facts, cycle_issue = _latest_cycle(reviews)
    if cycle_issue:
        return EntryClassification(
            entry_id,
            BLOCKED_STATE,
            cycle_issue,
            latest_review_seq=seq,
            validation_result=validation_result,
        )
    if facts is None:
        return EntryClassification(
            entry_id,
            REVIEW_READY,
            latest_review_seq=None,
            validation_result=validation_result,
        )

    verdicts = {artifact: facts[artifact].verdict for artifact in ARTIFACTS}
    relations = _target_relations(git, facts)
    unsafe = {
        artifact: relation
        for artifact, relation in relations.items()
        if relation not in {"exact", "stale"}
    }
    if unsafe:
        return EntryClassification(
            entry_id,
            BLOCKED_STATE,
            "UNSAFE_REVIEW_TARGET_RELATION:" + json.dumps(unsafe, sort_keys=True),
            latest_review_seq=seq,
            validation_result=validation_result,
            review_verdicts=verdicts,
            review_target_relation=relations,
        )

    all_pass = all(verdicts[a] == PASS_VERDICT for a in ARTIFACTS)
    all_exact = all(relations[a] == "exact" for a in ARTIFACTS)

    if all_exact and all_pass:
        return EntryClassification(
            entry_id,
            COMPLETE,
            latest_review_seq=seq,
            validation_result=validation_result,
            review_verdicts=verdicts,
            review_target_relation=relations,
        )
    if all_exact:
        return EntryClassification(
            entry_id,
            CORRECTION_REQUIRED,
            latest_review_seq=seq,
            validation_result=validation_result,
            review_verdicts=verdicts,
            review_target_relation=relations,
        )
    if all_pass:
        return EntryClassification(
            entry_id,
            STALE_CURRENT,
            latest_review_seq=seq,
            validation_result=validation_result,
            review_verdicts=verdicts,
            review_target_relation=relations,
        )

    # A non-Pass Review followed by a canonical correction is not applicable
    # to the new target anymore; the corrected current target requires re-review.
    return EntryClassification(
        entry_id,
        REVIEW_READY,
        "CORRECTED_TARGET_REQUIRES_FRESH_REVIEW",
        latest_review_seq=seq,
        validation_result=validation_result,
        review_verdicts=verdicts,
        review_target_relation=relations,
    )


def _read_latest_nonpass_reviews(entry_id: str) -> dict[str, Any]:
    snapshot = load_entry_review_state(entry_id)
    seq, facts, issue = _latest_cycle(snapshot)
    if issue or facts is None or seq is None:
        raise RuntimeError(issue or "no applicable Review cycle")
    result: dict[str, Any] = {}
    for artifact, fact in facts.items():
        if fact.verdict == PASS_VERDICT:
            continue
        result[artifact] = json.loads(fact.review_path.read_text(encoding="utf-8"))
    if not result:
        raise RuntimeError("CORRECTION_REQUIRED without a non-Pass Review")
    return result


def _creator_prompt(entry_id: str, classification: EntryClassification) -> str:
    if classification.state == CORRECTION_REQUIRED:
        findings = _read_latest_nonpass_reviews(entry_id)
        return f"""Continue Workflow 10 for Entry_ID {entry_id} as a correction task.

Applicable current Review Finding bundle:
{json.dumps(findings, ensure_ascii=False, sort_keys=True)}

Determine the highest-upstream semantic root cause yourself. Do not use fixed
artifact mapping from Review severity/artifact. Re-evaluate all invalidated
downstream artifacts. Immediately before actual work on each artifact, emit the
existing Workflow 90 research_started event for that artifact. Perform required
deterministic validation, commit and push canonical changes, run normal Workflow
90 synchronization after canonical commits, then return control to Workflow 00.
Do not run Workflow 20 and do not write Review JSON.
"""

    return f"""Execute Workflow 10 for Entry_ID {entry_id}.
Current Workflow 00 classification is {classification.state}
({classification.reason or "no additional reason"}).

Resume from the earliest missing/invalid canonical artifact. Immediately before
actual work on each artifact, emit the existing Workflow 90 research_started
event for that artifact. Use current external evidence and current rules.
Complete the canonical 00 -> 10 -> 20 chain as required, run deterministic
validation, commit and push canonical changes, synchronize Workflow 90 after
canonical commits, and return control to Workflow 00 when review-ready.
Do not run Workflow 20 and do not write Review JSON.
"""


def _artifact_signature(entry_id: str) -> tuple[tuple[str, str | None, str | None], ...]:
    git = load_entry_git_state(entry_id)
    return tuple(
        (a, git.artifacts[a].commit_sha, git.artifacts[a].blob_sha)
        for a in ARTIFACTS
    )


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _parse_reviewer_output(text: str) -> Mapping[str, Any]:
    match = _JSON_FENCE.match(text)
    if match:
        text = match.group(1)
    data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError("Reviewer output must be one JSON object")
    reviews = data.get("reviews")
    if not isinstance(reviews, Mapping) or set(reviews) != set(ARTIFACTS):
        raise ValueError("Reviewer output reviews must contain exactly 00, 10, 20")
    return reviews


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "git " + " ".join(args) + " failed: " + proc.stderr.strip()
        )
    return proc.stdout.strip()


def _commit_review_cycle(write_result: Mapping[str, Any]) -> str:
    """Commit only the append-only Review files created by review_writer."""
    repo = Path(_git(ARTICLE_ROOT, "rev-parse", "--show-toplevel")).resolve()
    raw_paths = write_result.get("paths")
    review_seq = write_result.get("review_seq")
    entry_id = write_result.get("entry_id")
    if not isinstance(raw_paths, Mapping) or set(raw_paths) != set(ARTIFACTS):
        raise RuntimeError("review_writer returned invalid paths")

    rel_paths: list[str] = []
    for artifact in ARTIFACTS:
        path = (ARTICLE_ROOT / str(raw_paths[artifact])).resolve()
        rel_paths.append(path.relative_to(repo).as_posix())

    _git(repo, "add", "--", *rel_paths)
    _git(
        repo,
        "commit",
        "-m",
        f"review({entry_id}): add Workflow 20 cycle {int(review_seq):03d}",
        "--",
        *rel_paths,
    )
    commit_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "push", "origin", "HEAD")
    return commit_sha


def _run_review_cycle(
    entry_id: str,
    runtime: AgentsRuntime,
) -> dict[str, Any]:
    prepared = prepare_review_cycle(entry_id)
    cycle = prepared.to_dict()

    started = sync_controlplane(
        entry_id,
        review_started=ARTIFACTS,
    )
    if started.get("result") not in {"PASS", "UPDATED"} or not started.get("verified"):
        raise RuntimeError(
            "review_started Workflow 90 failed: "
            + json.dumps(started, ensure_ascii=False, sort_keys=True)
        )

    context = build_reviewer_context(entry_id, cycle)
    reviewer = runtime.create_reviewer_session(entry_id, prepared.review_seq)
    prompt = (
        "Execute Workflow 20 semantic Review independently for this frozen cycle. "
        "Return only the required JSON bundle. Reconstruct claims from the frozen "
        "artifacts and verify them against current external evidence where required."
    )

    last_error: Exception | None = None
    for attempt in range(1, MAX_REVIEW_RESPONSE_ATTEMPTS + 1):
        output = runtime.run_reviewer(reviewer, context, prompt)
        try:
            reviews = _parse_reviewer_output(output.output)
            written = write_review_cycle(entry_id, cycle=cycle, reviews=reviews)
            commit_sha = _commit_review_cycle(written)
            after = sync_controlplane(entry_id)
            if after.get("result") not in {"PASS", "UPDATED"} or not after.get("verified"):
                raise RuntimeError(
                    "post-Review Workflow 90 failed: "
                    + json.dumps(after, ensure_ascii=False, sort_keys=True)
                )
            return {
                "review_seq": prepared.review_seq,
                "review_session_id": reviewer.session_id,
                "review_commit_sha": commit_sha,
                "verdicts": written["verdicts"],
            }
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            prompt = (
                "Your previous response failed deterministic Review save validation. "
                f"Attempt {attempt}: {exc}. Correct the JSON against the supplied "
                "Workflow 20 and schemas. Return the full 00/10/20 JSON bundle only."
            )

    raise RuntimeError(f"Reviewer response did not validate: {last_error}")


def _result(
    entry_id: str,
    classification: EntryClassification,
    *,
    result: str,
    review_cycles: int,
    reason_code: str | None = None,
    messages: tuple[str, ...] = (),
    control_plane_synchronized: bool = False,
) -> PipelineResult:
    exact = bool(
        classification.review_target_relation
        and all(
            classification.review_target_relation.get(a) == "exact"
            for a in ARTIFACTS
        )
    )
    return PipelineResult(
        result=result,
        entry_id=entry_id,
        state=classification.state,
        latest_review_seq=classification.latest_review_seq,
        review_cycles_created=review_cycles,
        reason_code=reason_code,
        messages=messages,
        canonical_review_exact=exact,
        validation_pass=classification.validation_result == "PASS",
        control_plane_synchronized=control_plane_synchronized,
    )


def run_entry_pipeline(
    entry_id: str,
    *,
    runtime: AgentsRuntime | None = None,
) -> PipelineResult:
    """Run Workflow 00 end-to-end for one Entry_ID."""
    if not _valid_entry_id(entry_id):
        classification = EntryClassification(entry_id, ERROR_STATE, "INVALID_ENTRY_ID")
        return _result(
            entry_id,
            classification,
            result="ERROR",
            review_cycles=0,
            reason_code="INVALID_ENTRY_ID",
        )

    initial_sync = sync_controlplane(entry_id)
    if initial_sync.get("result") not in {"PASS", "UPDATED"} or not initial_sync.get("verified"):
        classification = EntryClassification(
            entry_id,
            BLOCKED_STATE
            if initial_sync.get("result") == "BLOCKED"
            else ERROR_STATE,
            str(initial_sync.get("reason_code") or "WORKFLOW90_START_FAILED"),
        )
        return _result(
            entry_id,
            classification,
            result="BLOCKED" if classification.state == BLOCKED_STATE else "ERROR",
            review_cycles=0,
            reason_code=classification.reason,
            messages=tuple(str(x) for x in initial_sync.get("messages", [])),
        )

    runtime = runtime or OpenAIAgentsAPI()
    creator = None
    review_cycles = 0

    for _step in range(MAX_PIPELINE_STEPS_PER_RUN):
        classification = classify_entry_state(entry_id)

        if classification.state == COMPLETE:
            final_sync = sync_controlplane(entry_id)
            synchronized = (
                final_sync.get("result") in {"PASS", "UPDATED"}
                and bool(final_sync.get("verified"))
            )
            final_validation = validate_entry(entry_id, through="20")
            final_classification = classify_entry_state(entry_id)
            if (
                synchronized
                and final_validation.get("result") == "PASS"
                and final_classification.state == COMPLETE
            ):
                return _result(
                    entry_id,
                    final_classification,
                    result="PASS",
                    review_cycles=review_cycles,
                    control_plane_synchronized=True,
                )
            return _result(
                entry_id,
                final_classification,
                result="ERROR",
                review_cycles=review_cycles,
                reason_code="FINAL_CONVERGENCE_FAILED",
                messages=tuple(str(x) for x in final_sync.get("messages", [])),
                control_plane_synchronized=synchronized,
            )

        if classification.state == BLOCKED_STATE:
            return _result(
                entry_id,
                classification,
                result="BLOCKED",
                review_cycles=review_cycles,
                reason_code=classification.reason or "BLOCKED_STATE",
            )

        if classification.state == ERROR_STATE:
            return _result(
                entry_id,
                classification,
                result="ERROR",
                review_cycles=review_cycles,
                reason_code=classification.reason or "ERROR_STATE",
            )

        if classification.state in {NEW, PARTIAL, CORRECTION_REQUIRED}:
            if creator is None:
                try:
                    creator = runtime.create_creator_session(entry_id)
                except Exception as exc:
                    return _result(
                        entry_id,
                        classification,
                        result="ERROR",
                        review_cycles=review_cycles,
                        reason_code=f"CREATOR_SESSION_ERROR:{type(exc).__name__}",
                        messages=(str(exc),),
                    )
            before = _artifact_signature(entry_id)
            try:
                runtime.run_creator(
                    creator,
                    _creator_prompt(entry_id, classification),
                )
            except Exception as exc:
                return _result(
                    entry_id,
                    classification,
                    result="ERROR",
                    review_cycles=review_cycles,
                    reason_code=f"CREATOR_RUN_ERROR:{type(exc).__name__}",
                    messages=(str(exc),),
                )
            after = _artifact_signature(entry_id)
            post_sync = sync_controlplane(entry_id)
            if post_sync.get("result") not in {"PASS", "UPDATED"} or not post_sync.get("verified"):
                return _result(
                    entry_id,
                    classify_entry_state(entry_id),
                    result="BLOCKED" if post_sync.get("result") == "BLOCKED" else "ERROR",
                    review_cycles=review_cycles,
                    reason_code=str(post_sync.get("reason_code") or "WORKFLOW90_CREATOR_FAILED"),
                    messages=tuple(str(x) for x in post_sync.get("messages", [])),
                )
            if before == after:
                return _result(
                    entry_id,
                    classify_entry_state(entry_id),
                    result="BLOCKED",
                    review_cycles=review_cycles,
                    reason_code="CREATOR_NO_PROGRESS",
                )
            continue

        if classification.state in {REVIEW_READY, STALE_CURRENT}:
            if review_cycles >= MAX_REVIEW_CYCLES_PER_RUN:
                return _result(
                    entry_id,
                    classification,
                    result="BLOCKED",
                    review_cycles=review_cycles,
                    reason_code="MAX_REVIEW_CYCLES",
                )
            try:
                _run_review_cycle(entry_id, runtime)
            except Exception as exc:
                return _result(
                    entry_id,
                    classify_entry_state(entry_id),
                    result="ERROR",
                    review_cycles=review_cycles,
                    reason_code=f"REVIEW_CYCLE_ERROR:{type(exc).__name__}",
                    messages=(str(exc),),
                )
            review_cycles += 1
            after_review = classify_entry_state(entry_id)
            if (
                review_cycles >= MAX_REVIEW_CYCLES_PER_RUN
                and after_review.state != COMPLETE
            ):
                return _result(
                    entry_id,
                    after_review,
                    result="BLOCKED",
                    review_cycles=review_cycles,
                    reason_code="MAX_REVIEW_CYCLES",
                )
            continue

        return _result(
            entry_id,
            classification,
            result="ERROR",
            review_cycles=review_cycles,
            reason_code=f"UNHANDLED_STATE:{classification.state}",
        )

    classification = classify_entry_state(entry_id)
    return _result(
        entry_id,
        classification,
        result="BLOCKED",
        review_cycles=review_cycles,
        reason_code="MAX_PIPELINE_STEPS",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry_id", help="four-digit Entry_ID")
    args = parser.parse_args(argv)
    result = run_entry_pipeline(args.entry_id)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return {"PASS": 0, "BLOCKED": 1, "ERROR": 2}[result.result]


if __name__ == "__main__":
    sys.exit(main())
