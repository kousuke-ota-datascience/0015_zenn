"""Pure reconciliation of Notion, Git, and Review snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


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


def reconcile(controlplane_snapshot, git_snapshot, review_snapshot, relations: Mapping[tuple[str, str], str]) -> ReconcileResult:
    issues: list[str] = []
    mutations: list[Mutation] = []

    if getattr(controlplane_snapshot, "issues", ()):
        return ReconcileResult("BLOCKED", (), tuple(controlplane_snapshot.issues), "control plane snapshot is not uniquely usable")
    if getattr(review_snapshot, "issues", ()):
        return ReconcileResult("BLOCKED", (), tuple(review_snapshot.issues), "review snapshot contains malformed or duplicate facts")

    for artifact in ("00", "10", "20"):
        cp = controlplane_snapshot.artifacts.get(artifact)
        git = git_snapshot.artifacts.get(artifact)
        review = review_snapshot.latest.get(artifact)
        if cp is None:
            issues.append(f"missing_controlplane:{artifact}")
            continue
        if git is None or not git.exists:
            if cp.status not in {None, "未", "－（対象外）"} or cp.post_sha or review:
                issues.append(f"artifact_missing_but_state_present:{artifact}")
            continue
        if not git.commit_sha or not git.blob_sha:
            issues.append(f"artifact_not_committed:{artifact}")
            continue

        changes: dict[str, Any] = {}
        cp_relation = relations.get(("cp_post", artifact), "missing") if cp.post_sha else "missing"
        if cp.post_sha is None:
            changes["post-SHA"] = git.commit_sha
            if cp.status in {None, "未"}:
                changes["Status"] = "レビュー待"
        elif cp_relation == "exact":
            pass
        elif cp_relation == "left_ancestor":
            if review is not None:
                issues.append(f"stale_review_after_artifact_change:{artifact}")
                continue
            changes["pre-SHA"] = cp.post_sha
            changes["post-SHA"] = git.commit_sha
            changes["Status"] = "再レビュー待" if cp.latest_review_seq else "レビュー待"
        else:
            issues.append(f"unsafe_controlplane_sha_relation:{artifact}:{cp_relation}")
            continue

        if review is not None:
            review_relation = relations.get(("review_target", artifact), "missing")
            if review_relation == "exact":
                if review.target_blob_sha != git.blob_sha:
                    issues.append(f"review_target_blob_mismatch:{artifact}")
                    continue
                changes["最新レビュー版"] = review.review_seq
                changes["post-SHA"] = git.commit_sha
                changes["Status"] = "完了" if review.verdict == "Pass" else "要修正"
            elif review_relation == "left_ancestor":
                issues.append(f"stale_review:{artifact}")
                continue
            elif review_relation == "right_ancestor":
                issues.append(f"review_target_ahead_of_artifact:{artifact}")
                continue
            else:
                issues.append(f"unsafe_review_sha_relation:{artifact}:{review_relation}")
                continue

        normalized: dict[str, Any] = {}
        current_by_name = {
            "Status": cp.status,
            "最新レビュー版": cp.latest_review_seq,
            "pre-SHA": cp.pre_sha,
            "post-SHA": cp.post_sha,
            "remarks": cp.remarks,
        }
        for name, value in changes.items():
            if current_by_name.get(name) != value:
                normalized[name] = value
        if normalized:
            mutations.append(Mutation(artifact, normalized))

    if issues:
        return ReconcileResult("BLOCKED", (), tuple(sorted(set(issues))), "unsafe or ambiguous state; no mutation generated")
    if mutations:
        return ReconcileResult("UPDATE", tuple(mutations), (), f"{len(mutations)} artifact state(s) require synchronization")
    return ReconcileResult("NOOP", (), (), "control plane already matches Git and Review facts")
