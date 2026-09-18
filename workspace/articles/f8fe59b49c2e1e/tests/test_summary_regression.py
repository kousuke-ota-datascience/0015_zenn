import json
from pathlib import Path

ARTICLE_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = ARTICLE_ROOT / "docs" / "10_each_lore"


def _contents(entry_id: str, slug: str) -> dict:
    path = CANONICAL_ROOT / f"{entry_id}_{slug}" / f"{entry_id}_10_contents.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _covered_units(data: dict) -> list[dict]:
    coverage = set(data["summary"]["coverage_refs"])
    return [unit for unit in data["content_units"] if unit["content_id"] in coverage]


def _find_covered(data: dict, predicate, label: str) -> dict:
    matches = [unit for unit in _covered_units(data) if predicate(unit)]
    assert matches, f"missing salient covered meaning: {label}"
    return matches[0]


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


def test_0178_saruyume_salient_story_meanings_are_covered():
    data = _contents("0178", "saruyume")

    _find_covered(data, lambda x: "無人駅" in x["text"], "unmanned station")
    _find_covered(data, lambda x: "お猿さん電車" in x["text"], "monkey train")
    _find_covered(
        data,
        lambda x: "順番" in x["text"] and "投稿者" in x["text"],
        "passengers processed in sequence and narrator turn approaching",
    )
    _find_covered(
        data,
        lambda x: "夢だと認識" in x["text"] and "覚醒" in x["text"],
        "awakening as escape",
    )
    _find_covered(
        data,
        lambda x: "4年後" in x["text"] and "続き" in x["text"],
        "same dream resumes four years later",
    )
    _find_covered(
        data,
        lambda x: "再び覚醒" in x["text"],
        "second awakening",
    )
    _find_covered(
        data,
        lambda x: "次回は逃がさない" in x["text"],
        "terminal warning",
    )
    _find_covered(
        data,
        lambda x: "2003年" in x["text"] and "読んだ" in x["text"] and "類似" in x["text"],
        "later reader-side derivative",
    )


def test_0179_kunekune_terminal_identity_and_final_scene_are_salient():
    data = _contents("0179", "kunekune")

    terminal = _find_covered(
        data,
        lambda x: "白い" in x["text"]
        and "くねくね動く" in x["text"]
        and x["type"] == "outcome",
        "brother motif echo",
    )
    assert (
        "元の兄ではなくなったかのよう" in terminal["text"]
        or ("以前の兄" in terminal["text"] and "兄ではなく" in terminal["text"])
    )

    _find_covered(
        data,
        lambda x: "見てはならない" in x["text"] and "間近に見" in x["text"],
        "narrator final forbidden sighting",
    )

    narrative = data["summary"]["narrative"]
    assert "くねくね動く" in narrative
    assert "兄ではなくなったかのよう" in narrative
    assert "見てはならない" in narrative and "間近に見" in narrative
