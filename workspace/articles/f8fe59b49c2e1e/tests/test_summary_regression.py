import json
from pathlib import Path

ARTICLE_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = ARTICLE_ROOT / "docs" / "10_each_lore"


def _contents(entry_id: str, slug: str) -> dict:
    path = CANONICAL_ROOT / f"{entry_id}_{slug}" / f"{entry_id}_10_contents.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_canonical_summaries_cover_existing_content_units():
    paths = sorted(CANONICAL_ROOT.glob("[0-9][0-9][0-9][0-9]_*/*_10_contents.json"))
    assert paths
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        content_ids = {item["content_id"] for item in data["content_units"]}
        coverage_refs = data["summary"]["coverage_refs"]
        assert coverage_refs, f"{path}: summary.coverage_refs must not be empty"
        assert len(coverage_refs) == len(set(coverage_refs)), f"{path}: duplicate coverage ref"
        missing = sorted(set(coverage_refs) - content_ids)
        assert not missing, f"{path}: dangling summary coverage refs: {missing}"


def test_0178_saruyume_salient_story_units_are_selected_before_summary():
    data = _contents("0178", "saruyume")
    coverage = set(data["summary"]["coverage_refs"])
    expected = {f"CNT-{i:03d}" for i in range(1, 10)}
    assert expected <= coverage


def test_0179_kunekune_terminal_identity_and_final_scene_are_salient():
    data = _contents("0179", "kunekune")
    coverage = set(data["summary"]["coverage_refs"])

    # Do not bind semantic meaning to a historical CNT number. Blind regeneration
    # may legitimately reorder/re-slice content units; what must remain invariant
    # is that the salient meaning itself exists and is covered.
    terminal = next(
        x
        for x in data["content_units"]
        if "白い" in x["text"]
        and "くねくね動く" in x["text"]
        and x["type"] == "outcome"
    )
    assert terminal["content_id"] in coverage

    identity = next(
        (
            x
            for x in data["content_units"]
            if "元の兄ではなくなったかのよう" in x["text"]
            or "以前の兄" in x["text"] and "兄ではなく" in x["text"]
        ),
        terminal,
    )
    assert identity["content_id"] in coverage

    final_scene = next(
        x
        for x in data["content_units"]
        if "見てはならない" in x["text"] and "間近に見" in x["text"]
    )
    assert final_scene["content_id"] in coverage

    narrative = data["summary"]["narrative"]
    assert "くねくね動く" in narrative
    assert "兄ではなくなったかのよう" in narrative
    assert "見てはならない" in narrative and "間近に見" in narrative
