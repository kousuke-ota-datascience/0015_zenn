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

    # The known regression: abstracting this to only "mental/behavioral change"
    # loses the brother's motif echo / identity terminal state.
    assert "CNT-009" in coverage
    terminal = next(x for x in data["content_units"] if x["content_id"] == "CNT-009")
    assert "白い物体と同じようにくねくね" in terminal["text"]
    assert "元の兄ではなくなったかのよう" in terminal["text"]

    # The actual terminal scene is also salient: the narrator himself finally
    # sees the forbidden object at close range.
    assert "CNT-011" in coverage
    final_scene = next(x for x in data["content_units"] if x["content_id"] == "CNT-011")
    assert "間近に見てしまう" in final_scene["text"]

    narrative = data["summary"]["narrative"]
    assert "白い物体と同じようにくねくね動く状態" in narrative
    assert "以前の兄ではなくなったかのよう" in narrative
    assert "見てはならない白い物体を間近に見てしまう" in narrative
