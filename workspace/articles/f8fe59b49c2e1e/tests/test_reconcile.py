from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.status_management.reconcile import reconcile


def _cp(status, *, latest=None, pre=None, post=None):
    return SimpleNamespace(
        status=status,
        latest_review_seq=latest,
        pre_sha=pre,
        post_sha=post,
        remarks=None,
    )


def _git(*, exists=True, commit="g" * 40, blob="b" * 40):
    return SimpleNamespace(exists=exists, commit_sha=commit, blob_sha=blob)


def _review(verdict, *, seq=1, commit="g" * 40, blob="b" * 40):
    return SimpleNamespace(
        verdict=verdict,
        review_seq=seq,
        target_commit_sha=commit,
        target_blob_sha=blob,
    )


def _snapshots(cp00, git00=None, review00=None):
    outside = _cp("－（対象外）")
    cp = SimpleNamespace(
        artifacts={"00": cp00, "10": outside, "20": outside},
        issues=(),
    )
    missing_git = _git(exists=False, commit=None, blob=None)
    git = SimpleNamespace(
        artifacts={"00": git00 or _git(), "10": missing_git, "20": missing_git}
    )
    latest = {} if review00 is None else {"00": review00}
    reviews = SimpleNamespace(latest=latest, issues=())
    return cp, git, reviews


def _changes(result):
    assert len(result.mutations) == 1
    return result.mutations[0].changes


def test_status_un_to_review_wait():
    cp, git, reviews = _snapshots(_cp("未", post=None))
    result = reconcile(cp, git, reviews, {})
    assert result.outcome == "UPDATE"
    assert _changes(result) == {"post-SHA": "g" * 40, "Status": "レビュー待"}


def test_reconcile_idempotent_review_wait_no_review():
    cp, git, reviews = _snapshots(_cp("レビュー待", post="g" * 40))
    result = reconcile(cp, git, reviews, {("cp_post", "00"): "exact"})
    assert result.outcome == "NOOP"


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [("Pass", "完了"), ("Minor", "要修正"), ("Moderate", "要修正"), ("Major", "要修正")],
)
def test_review_exact_sets_status(verdict, expected):
    cp, git, reviews = _snapshots(
        _cp("レビュー待", post="g" * 40),
        review00=_review(verdict),
    )
    result = reconcile(
        cp,
        git,
        reviews,
        {("cp_post", "00"): "exact", ("review_target", "00"): "exact"},
    )
    assert result.outcome == "UPDATE"
    changes = _changes(result)
    assert changes["Status"] == expected
    assert changes["最新レビュー版"] == 1


def test_active_rework_does_not_roll_back_to_needs_fix():
    cp, git, reviews = _snapshots(
        _cp("再作業中", latest=None, post="g" * 40),
        review00=_review("Major"),
    )
    result = reconcile(
        cp,
        git,
        reviews,
        {("cp_post", "00"): "exact", ("review_target", "00"): "exact"},
    )
    assert result.outcome == "UPDATE"
    changes = _changes(result)
    assert changes == {"最新レビュー版": 1}


@pytest.mark.parametrize("status", ["要修正", "再作業中", "再レビュー待", "完了"])
def test_stale_review_moves_correction_states_to_rereview(status):
    cp, git, reviews = _snapshots(
        _cp(status, latest=1, post="g" * 40),
        review00=_review("Major", seq=1, commit="o" * 40),
    )
    result = reconcile(
        cp,
        git,
        reviews,
        {("cp_post", "00"): "exact", ("review_target", "00"): "left_ancestor"},
    )
    if status == "再レビュー待":
        assert result.outcome == "NOOP"
    else:
        assert result.outcome == "UPDATE"
        assert _changes(result)["Status"] == "再レビュー待"


def test_completed_artifact_update_becomes_rereview_wait():
    old = "o" * 40
    new = "n" * 40
    cp, git, reviews = _snapshots(
        _cp("完了", latest=1, post=old),
        git00=_git(commit=new),
        review00=_review("Pass", seq=1, commit=old),
    )
    result = reconcile(
        cp,
        git,
        reviews,
        {("cp_post", "00"): "left_ancestor", ("review_target", "00"): "left_ancestor"},
    )
    assert result.outcome == "UPDATE"
    changes = _changes(result)
    assert changes["pre-SHA"] == old
    assert changes["post-SHA"] == new
    assert changes["Status"] == "再レビュー待"


def test_target_outside_is_never_inferred_or_cleared():
    cp, git, reviews = _snapshots(_cp("－（対象外）"))
    result = reconcile(cp, git, reviews, {})
    assert result.outcome == "NOOP"


def test_diverged_controlplane_blocks_without_mutation():
    cp, git, reviews = _snapshots(_cp("レビュー待", post="x" * 40))
    result = reconcile(cp, git, reviews, {("cp_post", "00"): "diverged"})
    assert result.outcome == "BLOCKED"
    assert result.mutations == ()
    assert "unsafe_controlplane_sha_relation:00:diverged" in result.issues
