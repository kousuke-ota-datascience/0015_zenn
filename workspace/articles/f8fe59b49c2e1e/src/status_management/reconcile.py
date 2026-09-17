"""Pure reconciliation of Notion, Git, and Review snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ALLOWED_STATUSES = frozenset(
    {"未", "レビュー待", "要修正", "再作業中", "再レビュー待", "完了", "－（対象外）"}
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


def reconcile(controlplane_snapshot, git_snapshot, review_snapshot, relations: Mapping[tuple[str, str], str]) -> ReconcileResult:
    issues: list[str] = []
    mutations: list[Mutation] = []

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

    for artifact in ("00", "10", "20"):
        cp = controlplane_snapshot.artifacts.get(artifact)
        git = git_snapshot.artifacts.get(artifact)
        review = review_snapshot.latest.get(artifact)

        if cp is None:
            issues.append(f"missing_controlplane:{artifact}")
            continue
        if cp.status not in ALLOWED_STATUSES:
            issues.append(f"invalid_status_value:{artifact}:{cp.status!r}")
            continue

        # Target-outside is explicit operational state; never infer or clear it
        # from Git/Review facts alone.
        if cp.status == "－（対象外）":
            continue

        if git is None or not git.exists:
            if cp.status != "未" or cp.post_sha or review:
                issues.append(f"artifact_missing_but_state_present:{artifact}")
            continue
        if not git.commit_sha or not git.blob_sha:
            issues.append(f"artifact_not_committed:{artifact}")
            continue

        changes: dict[str, Any] = {}
        cp_relation = relations.get(("cp_post", artifact), "missing") if cp.post_sha else "missing"
        review_relation = relations.get(("review_target", artifact), "missing") if review else "missing"

        if cp.post_sha is None:
            if review is not None:
                issues.append(f"review_exists_before_controlplane_post:{artifact}")
                continue
            if cp.status != "未":
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
                elif cp.status not in {"レビュー待"}:
                    issues.append(f"status_requires_review_fact:{artifact}:{cp.status}")
                    continue

            elif review_relation == "exact":
                if review.target_blob_sha != git.blob_sha:
                    issues.append(f"review_target_blob_mismatch:{artifact}")
                    continue
                changes["最新レビュー版"] = review.review_seq
                changes["post-SHA"] = git.commit_sha
                if review.verdict == "Pass":
                    changes["Status"] = "完了"
                elif cp.status == "再作業中":
                    # A sync during active correction must not roll operational
                    # state back to 要修正 merely because the triggering Review
                    # remains the latest Review fact.
                    pass
                else:
                    changes["Status"] = "要修正"

            elif review_relation == "left_ancestor":
                # The latest Review targets an older artifact version. This is
                # normal after a correction commit and before re-review.
                if cp.status in {"要修正", "再作業中", "再レビュー待", "完了"}:
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
