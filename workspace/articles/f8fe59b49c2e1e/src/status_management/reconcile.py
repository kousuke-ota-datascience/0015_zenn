"""Pure reconciliation of Notion, Git, and Review snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping


ARTIFACTS = ("00", "10", "20")
ALLOWED_STATUSES = frozenset(
    {"未", "調査中", "レビュー待", "レビュー中", "要修正", "再レビュー待", "完了", "－（対象外）"}
)


@dataclass(frozen=True)
class Mutation:
    artifact: str
    changes: dict[str, Any]


@dataclass(frozen=True)
class ReconcileResult:
    outcome: str
    mutations: tuple[Mutation, ...]
    issues: tuple[str, ...]
    summary: str


def _status_from_review(verdict: str) -> str:
    return "完了" if verdict == "Pass" else "要修正"


def _normalize_changes(cp, changes: dict[str, Any]) -> dict[str, Any]:
    current_by_name = {
        "Status": cp.status,
        "最新レビュー版": cp.latest_review_seq,
        "pre-SHA": cp.pre_sha,
        "post-SHA": cp.post_sha,
        "remarks": cp.remarks,
    }
    return {name: value for name, value in changes.items() if current_by_name.get(name) != value}


def _validate_research_start(
    artifact: str,
    cp,
    git,
    review,
    *,
    cp_relation: str,
    review_relation: str,
) -> str | None:
    """Return an issue code when a research-start event is not applicable."""
    if cp.status not in {"未", "要修正", "調査中", "完了"}:
        return f"research_start_invalid_status:{artifact}:{cp.status}"

    # Initial generation: no committed artifact/checkpoint and no Review fact yet.
    if cp.post_sha is None and review is None:
        if cp.status not in {"未", "調査中"}:
            return f"research_start_invalid_initial_status:{artifact}:{cp.status}"
        if git is not None and git.exists:
            return f"research_start_initial_artifact_already_committed:{artifact}"
        return None

    # Any re-opened committed artifact must still match its control-plane
    # checkpoint. The Review relation then determines which explicit task is
    # being resumed: correction after non-Pass, or downstream re-research of a
    # previously passed artifact.
    if git is None or not git.exists or not git.commit_sha or not git.blob_sha:
        return f"research_start_missing_artifact:{artifact}"
    if not cp.post_sha:
        return f"research_start_missing_controlplane_post:{artifact}"
    if cp_relation != "exact":
        return f"research_start_requires_exact_controlplane:{artifact}:{cp_relation}"
    if review is None:
        return f"research_start_requires_review:{artifact}"
    if review_relation != "exact":
        return f"research_start_requires_exact_review:{artifact}:{review_relation}"
    if review.target_blob_sha != git.blob_sha:
        return f"review_target_blob_mismatch:{artifact}"

    if cp.status == "要修正":
        if review.verdict == "Pass":
            return f"research_start_on_passed_review:{artifact}"
        return None

    if cp.status == "完了":
        if review.verdict != "Pass":
            return f"research_start_completed_without_pass:{artifact}"
        return None

    if cp.status == "調査中":
        # Idempotent repeat after either an explicit correction start or a
        # deliberate re-open of a previously passed artifact.
        return None

    return f"research_start_invalid_status:{artifact}:{cp.status}"


def _validate_review_start(
    artifact: str,
    cp,
    git,
    review,
    *,
    cp_relation: str,
    review_relation: str,
) -> str | None:
    """Return an issue code when a review-start event is not applicable."""
    if cp.status not in {"レビュー待", "再レビュー待", "完了", "レビュー中"}:
        return f"review_start_invalid_status:{artifact}:{cp.status}"
    if git is None or not git.exists or not git.commit_sha or not git.blob_sha:
        return f"review_start_missing_artifact:{artifact}"
    if not cp.post_sha:
        return f"review_start_missing_controlplane_post:{artifact}"
    if cp_relation != "exact":
        return f"review_start_requires_exact_controlplane:{artifact}:{cp_relation}"

    if cp.status == "レビュー待":
        if review is not None or cp.latest_review_seq is not None:
            return f"review_start_initial_review_fact_present:{artifact}"
        return None

    if cp.status == "再レビュー待":
        if review is None:
            return f"review_start_requires_previous_review:{artifact}"
        if cp.latest_review_seq != review.review_seq:
            return f"review_start_review_seq_mismatch:{artifact}"
        if review_relation != "left_ancestor":
            return f"review_start_requires_stale_previous_review:{artifact}:{review_relation}"
        return None

    if cp.status == "完了":
        if review is None:
            return f"review_start_requires_previous_review:{artifact}"
        if cp.latest_review_seq != review.review_seq:
            return f"review_start_review_seq_mismatch:{artifact}"
        if review_relation != "exact":
            return f"review_start_completed_requires_exact_review:{artifact}:{review_relation}"
        if review.target_blob_sha != git.blob_sha:
            return f"review_target_blob_mismatch:{artifact}"
        if review.verdict != "Pass":
            return f"review_start_completed_without_pass:{artifact}"
        return None

    # Idempotent repeated event while Review is already active.
    if review is None:
        if cp.latest_review_seq is not None:
            return f"review_start_review_seq_without_review:{artifact}"
        return None
    if cp.latest_review_seq != review.review_seq:
        return f"review_start_active_review_seq_mismatch:{artifact}"
    if review_relation not in {"exact", "left_ancestor"}:
        return f"review_start_active_unsafe_review_relation:{artifact}:{review_relation}"
    if review_relation == "exact" and review.target_blob_sha != git.blob_sha:
        return f"review_target_blob_mismatch:{artifact}"
    return None


def reconcile(
    controlplane_snapshot,
    git_snapshot,
    review_snapshot,
    relations: Mapping[tuple[str, str], str],
    *,
    research_started: Iterable[str] = (),
    review_started: Iterable[str] = (),
) -> ReconcileResult:
    """Derive a mutation plan from already-read facts.

    research_started is emitted only when Workflow 00/10 actually enters an
    initial-generation or post-Review research task. review_started is emitted
    only when Workflow 20 has frozen a valid Review target and actually begins
    the semantic Review task. Neither event is inferred from Entry_ID or static
    Git/Review facts.
    """
    issues: list[str] = []
    mutations: list[Mutation] = []
    research_started = frozenset(research_started)
    review_started = frozenset(review_started)

    unknown_research = sorted(research_started - set(ARTIFACTS))
    unknown_review = sorted(review_started - set(ARTIFACTS))
    if unknown_research or unknown_review:
        event_issues = [
            *(f"invalid_research_start_artifact:{x}" for x in unknown_research),
            *(f"invalid_review_start_artifact:{x}" for x in unknown_review),
        ]
        return ReconcileResult(
            "BLOCKED",
            (),
            tuple(event_issues),
            "task-start event contains an unknown artifact",
        )

    overlap = sorted(research_started & review_started)
    if overlap:
        return ReconcileResult(
            "BLOCKED",
            (),
            tuple(f"conflicting_task_start_events:{x}" for x in overlap),
            "an artifact cannot start research and Review simultaneously",
        )

    if getattr(controlplane_snapshot, "issues", ()):
        return ReconcileResult(
            "BLOCKED",
            (),
            tuple(controlplane_snapshot.issues),
            "control plane snapshot is not uniquely usable",
        )
    if getattr(review_snapshot, "issues", ()):
        return ReconcileResult(
            "BLOCKED",
            (),
            tuple(review_snapshot.issues),
            "review snapshot contains malformed or duplicate facts",
        )

    for artifact in ARTIFACTS:
        cp = controlplane_snapshot.artifacts.get(artifact)
        git = git_snapshot.artifacts.get(artifact)
        review = review_snapshot.latest.get(artifact)

        if cp is None:
            issues.append(f"missing_controlplane:{artifact}")
            continue
        if cp.status not in ALLOWED_STATUSES:
            issues.append(f"invalid_status_value:{artifact}:{cp.status!r}")
            continue

        cp_relation = relations.get(("cp_post", artifact), "missing") if cp.post_sha else "missing"
        review_relation = relations.get(("review_target", artifact), "missing") if review else "missing"

        if artifact in research_started:
            event_issue = _validate_research_start(
                artifact,
                cp,
                git,
                review,
                cp_relation=cp_relation,
                review_relation=review_relation,
            )
            if event_issue:
                issues.append(event_issue)
                continue
            normalized = _normalize_changes(cp, {"Status": "調査中"})
            if normalized:
                mutations.append(Mutation(artifact, normalized))
            continue

        if artifact in review_started:
            event_issue = _validate_review_start(
                artifact,
                cp,
                git,
                review,
                cp_relation=cp_relation,
                review_relation=review_relation,
            )
            if event_issue:
                issues.append(event_issue)
                continue
            normalized = _normalize_changes(cp, {"Status": "レビュー中"})
            if normalized:
                mutations.append(Mutation(artifact, normalized))
            continue

        # Target-outside is explicit operational state; never infer or clear it
        # from Git/Review facts alone.
        if cp.status == "－（対象外）":
            continue

        if git is None or not git.exists:
            if cp.status not in {"未", "調査中"} or cp.post_sha or review:
                issues.append(f"artifact_missing_but_state_present:{artifact}")
            continue
        if not git.commit_sha or not git.blob_sha:
            issues.append(f"artifact_not_committed:{artifact}")
            continue

        changes: dict[str, Any] = {}

        if cp.post_sha is None:
            if review is not None:
                issues.append(f"review_exists_before_controlplane_post:{artifact}")
                continue
            if cp.status not in {"未", "調査中"}:
                issues.append(f"missing_post_sha_for_status:{artifact}:{cp.status}")
                continue
            changes["post-SHA"] = git.commit_sha
            changes["Status"] = "レビュー待"

        elif cp_relation == "exact":
            if review is None:
                if cp.latest_review_seq is not None:
                    issues.append(f"review_seq_without_review_file:{artifact}:{cp.latest_review_seq}")
                    continue
                if cp.status == "未":
                    changes["Status"] = "レビュー待"
                elif cp.status in {"レビュー待", "レビュー中"}:
                    pass
                elif cp.status == "調査中":
                    # A committed artifact with a current checkpoint ends the
                    # initial research task and becomes initial-Review-ready.
                    changes["Status"] = "レビュー待"
                else:
                    issues.append(f"status_requires_review_fact:{artifact}:{cp.status}")
                    continue

            elif review_relation == "exact":
                if review.target_blob_sha != git.blob_sha:
                    issues.append(f"review_target_blob_mismatch:{artifact}")
                    continue

                if cp.status == "レビュー中" and cp.latest_review_seq == review.review_seq:
                    # The currently visible Review fact is the pre-existing one;
                    # keep the explicit operational Review-in-progress state.
                    pass
                else:
                    changes["最新レビュー版"] = review.review_seq
                    changes["post-SHA"] = git.commit_sha
                    if cp.status == "調査中" and review.verdict != "Pass":
                        # During post-Review research, the triggering non-Pass
                        # Review remains the latest fact until a new Review cycle.
                        pass
                    else:
                        changes["Status"] = _status_from_review(review.verdict)

            elif review_relation == "left_ancestor":
                # The latest Review targets an older artifact version. This is
                # normal after a research/correction commit and before re-review.
                if cp.status == "レビュー中":
                    # The old Review remains the latest persisted fact while a
                    # new Review cycle is in progress.
                    pass
                elif cp.status in {"要修正", "調査中", "再レビュー待", "完了"}:
                    changes["最新レビュー版"] = review.review_seq
                    changes["Status"] = "再レビュー待"
                else:
                    issues.append(f"unexpected_stale_review:{artifact}:{cp.status}")
                    continue

            elif review_relation == "right_ancestor":
                issues.append(f"review_target_ahead_of_artifact:{artifact}")
                continue
            else:
                issues.append(f"unsafe_review_sha_relation:{artifact}:{review_relation}")
                continue

        elif cp_relation == "left_ancestor":
            # Git contains a newer canonical artifact than the control-plane
            # checkpoint. Advance pre/post and derive the waiting state.
            if cp.status == "レビュー中":
                issues.append(f"artifact_changed_during_review:{artifact}")
                continue

            changes["pre-SHA"] = cp.post_sha
            changes["post-SHA"] = git.commit_sha

            if review is None:
                changes["Status"] = "再レビュー待" if cp.latest_review_seq else "レビュー待"

            elif review_relation == "exact":
                if review.target_blob_sha != git.blob_sha:
                    issues.append(f"review_target_blob_mismatch:{artifact}")
                    continue
                changes["最新レビュー版"] = review.review_seq
                changes["Status"] = _status_from_review(review.verdict)

            elif review_relation == "left_ancestor":
                changes["最新レビュー版"] = review.review_seq
                changes["Status"] = "再レビュー待"

            elif review_relation == "right_ancestor":
                issues.append(f"review_target_ahead_of_artifact:{artifact}")
                continue
            else:
                issues.append(f"unsafe_review_sha_relation:{artifact}:{review_relation}")
                continue

        else:
            issues.append(f"unsafe_controlplane_sha_relation:{artifact}:{cp_relation}")
            continue

        normalized = _normalize_changes(cp, changes)
        if normalized:
            mutations.append(Mutation(artifact, normalized))

    if issues:
        return ReconcileResult(
            "BLOCKED",
            (),
            tuple(sorted(set(issues))),
            "unsafe or ambiguous state; no mutation generated",
        )
    if mutations:
        return ReconcileResult(
            "UPDATE",
            tuple(mutations),
            (),
            f"{len(mutations)} artifact state(s) require synchronization",
        )
    return ReconcileResult("NOOP", (), (), "control plane already matches Git and Review facts")


def _namespace_map(items: Mapping[str, Mapping[str, Any]]) -> dict[str, SimpleNamespace]:
    return {key: SimpleNamespace(**dict(value)) for key, value in items.items()}


def reconcile_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-friendly, I/O-free adapter for connector-driven production sync."""
    cp_data = dict(payload.get("controlplane") or {})
    git_data = dict(payload.get("git") or {})
    review_data = dict(payload.get("review") or {})

    cp = SimpleNamespace(
        artifacts=_namespace_map(cp_data.get("artifacts") or {}),
        issues=tuple(cp_data.get("issues") or ()),
    )
    git = SimpleNamespace(
        artifacts=_namespace_map(git_data.get("artifacts") or {}),
    )
    reviews = SimpleNamespace(
        latest=_namespace_map(review_data.get("latest") or {}),
        issues=tuple(review_data.get("issues") or ()),
    )

    relations: dict[tuple[str, str], str] = {}
    for key, value in dict(payload.get("relations") or {}).items():
        if isinstance(key, str) and ":" in key:
            relation_kind, artifact = key.split(":", 1)
            relations[(relation_kind, artifact)] = str(value)

    events = dict(payload.get("events") or {})
    result = reconcile(
        cp,
        git,
        reviews,
        relations,
        research_started=events.get("research_started") or (),
        review_started=events.get("review_started") or (),
    )
    return {
        "outcome": result.outcome,
        "summary": result.summary,
        "issues": list(result.issues),
        "mutations": [
            {"artifact": mutation.artifact, "changes": dict(mutation.changes)}
            for mutation in result.mutations
        ],
    }
