import json
from pathlib import Path

ARTICLE_ROOT = Path(__file__).resolve().parents[1]


def _summary(entry_id: str, slug: str) -> str:
    path = ARTICLE_ROOT / "docs" / "10_each_lore" / f"{entry_id}_{slug}" / f"{entry_id}_10_contents.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["summary"]["narrative"]


def test_0178_saruyume_summary_preserves_legacy_semantics():
    summary = _summary("0178", "saruyume")
    required = [
        "薄暗い無人駅",
        "お猿さん電車",
        "順番",
        "覚醒",
        "4年後",
        "前回の続き",
        "再び覚醒",
        "次回は逃がさない",
        "猿夢＋",
        "後代派生",
        "絶対的初出",
    ]
    missing = [term for term in required if term not in summary]
    assert not missing, f"0178 summary lost Legacy semantics: {missing}"


def test_0179_kunekune_summary_preserves_legacy_semantics():
    summary = _summary("0179", "kunekune")
    required = [
        "白い人影",
        "詳しく見",
        "分からないほうがいい",
        "重大な精神・行動上の異変",
        "混ぜて詳しく書いた",
        "秋田",
        "田園",
        "双眼鏡",
        "祖父",
        "見てはならない",
        "2003年増補",
        "2001年型へ遡及しない",
        "読者が記事を理解しただけで作用する型",
    ]
    missing = [term for term in required if term not in summary]
    assert not missing, f"0179 summary lost Legacy semantics: {missing}"
