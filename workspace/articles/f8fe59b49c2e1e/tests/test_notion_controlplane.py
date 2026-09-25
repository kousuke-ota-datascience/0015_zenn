from __future__ import annotations

from types import SimpleNamespace

from src.status_management import notion_controlplane as nc
from src.status_management import sync_controlplane as sc
from src.status_management.reconcile import Mutation, ReconcileResult


def _rich(value):
    if value is None:
        return {"type": "rich_text", "rich_text": []}
    return {
        "type": "rich_text",
        "rich_text": [{"plain_text": str(value), "type": "text", "text": {"content": str(value)}}],
    }


def _select(value):
    return {"type": "select", "select": None if value is None else {"name": value}}


def _row(entry_id, artifact, *, row_id=None, status="未", edited="2026-09-18T00:00:00Z"):
    props = {
        "Entry_ID": _rich(entry_id),
        "成果物": _select(artifact),
        nc.PROPERTY_NAMES["Status"]: _select(status),
        nc.PROPERTY_NAMES["最新レビュー版"]: _rich(None),
        nc.PROPERTY_NAMES["pre-SHA"]: _rich(None),
        nc.PROPERTY_NAMES["post-SHA"]: _rich(None),
        nc.PROPERTY_NAMES["remarks"]: _rich(None),
    }
    return {
        "id": row_id or f"row-{artifact}",
        "last_edited_time": edited,
        "properties": props,
    }


def _artifact(artifact, *, fingerprint="fp", status="未"):
    return nc.ControlPlaneArtifact(
        row_id=f"row-{artifact}",
        entry_id="0001",
        artifact=artifact,
        status=status,
        latest_review_seq=None,
        pre_sha=None,
        post_sha=None,
        remarks=None,
        fingerprint=fingerprint,
    )


def test_load_entry_state_detects_duplicate(monkeypatch):
    rows = [
        _row("0001", "00", row_id="a"),
        _row("0001", "00", row_id="b"),
        _row("0001", "10"),
        _row("0001", "20"),
    ]
    monkeypatch.setattr(nc, "_list_rows", lambda: rows)
    snapshot = nc.load_entry_state("0001")
    assert "duplicate_logical_key:0001:00" in snapshot.issues


def test_load_entry_state_detects_missing_row(monkeypatch):
    rows = [_row("0001", "00"), _row("0001", "10")]
    monkeypatch.setattr(nc, "_list_rows", lambda: rows)
    snapshot = nc.load_entry_state("0001")
    assert "missing_row:0001:20" in snapshot.issues


def test_apply_mutations_rejects_concurrent_update(monkeypatch):
    expected = nc.ControlPlaneSnapshot(
        "0001",
        {
            "00": _artifact("00", fingerprint="old"),
            "10": _artifact("10", fingerprint="10"),
            "20": _artifact("20", fingerprint="20"),
        },
        (),
    )
    current = nc.ControlPlaneSnapshot(
        "0001",
        {
            "00": _artifact("00", fingerprint="new"),
            "10": _artifact("10", fingerprint="10"),
            "20": _artifact("20", fingerprint="20"),
        },
        (),
    )
    monkeypatch.setattr(nc, "load_entry_state", lambda _: current)
    result = nc.apply_mutations(
        [Mutation("00", {"Status": "レビュー待"})],
        expected,
    )
    assert not result.success
    assert result.error == "concurrent_update:00"


def test_sync_forwards_explicit_correction_start(monkeypatch):
    cp = SimpleNamespace(artifacts={}, issues=())
    git = SimpleNamespace(artifacts={})
    reviews = SimpleNamespace(latest={}, issues=())
    captured = {}

    monkeypatch.setattr(sc, "load_entry_state", lambda _: cp)
    monkeypatch.setattr(sc, "load_entry_git_state", lambda _: git)
    monkeypatch.setattr(sc, "load_entry_review_state", lambda _: reviews)
    monkeypatch.setattr(sc, "_relations", lambda *args: {})

    def fake_reconcile(*args, **kwargs):
        captured["correction_started"] = tuple(kwargs.get("correction_started", ()))
        return ReconcileResult("NOOP", (), (), "fixture noop")

    monkeypatch.setattr(sc, "reconcile", fake_reconcile)

    result = sc.sync_controlplane("0001", correction_started=("00", "20"))

    assert result["result"] == "PASS"
    assert result["verified"]
    assert captured["correction_started"] == ("00", "20")


def test_sync_detects_post_update_verification_error(monkeypatch):
    before = nc.ControlPlaneSnapshot(
        "0001",
        {
            "00": _artifact("00", fingerprint="00", status="未"),
            "10": _artifact("10", fingerprint="10"),
            "20": _artifact("20", fingerprint="20"),
        },
        (),
    )
    after = nc.ControlPlaneSnapshot(
        "0001",
        {
            "00": _artifact("00", fingerprint="00-new", status="未"),
            "10": _artifact("10", fingerprint="10"),
            "20": _artifact("20", fingerprint="20"),
        },
        (),
    )
    states = iter([before, after])
    monkeypatch.setattr(sc, "load_entry_state", lambda _: next(states))
    monkeypatch.setattr(sc, "load_entry_git_state", lambda _: SimpleNamespace(artifacts={}))
    monkeypatch.setattr(sc, "load_entry_review_state", lambda _: SimpleNamespace(latest={}))
    monkeypatch.setattr(sc, "_relations", lambda *args: {})
    monkeypatch.setattr(
        sc,
        "reconcile",
        lambda *args: ReconcileResult(
            "UPDATE",
            (Mutation("00", {"Status": "レビュー待"}),),
            (),
            "fixture update",
        ),
    )
    monkeypatch.setattr(
        sc,
        "apply_mutations",
        lambda *args: nc.ApplyResult(True, ("00",)),
    )

    result = sc.sync_controlplane("0001")
    assert result["result"] == "ERROR"
    assert result["reason_code"] == "post_update_verification_error"
    assert not result["verified"]
