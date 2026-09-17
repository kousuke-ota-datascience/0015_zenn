from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.rendering import render_00_sources, render_10_contents, render_20_analysis, render_review


@pytest.mark.parametrize(
    ("module", "artifact", "filename", "fixture_name"),
    [
        (render_00_sources, "00", "0001_00_sources.json", "valid_sources"),
        (render_10_contents, "10", "0001_10_contents.json", "valid_contents"),
        (render_20_analysis, "20", "0001_20_analysis.json", "valid_analysis"),
    ],
)
def test_canonical_renderers_do_not_modify_source(
    request, tmp_path, monkeypatch, module, artifact, filename, fixture_name
):
    root = tmp_path / "article"
    source = root / "docs/10_each_lore/0001" / filename
    source.parent.mkdir(parents=True, exist_ok=True)
    data = copy.deepcopy(request.getfixturevalue(fixture_name))
    source.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    before = source.read_bytes()

    monkeypatch.setattr(module, "ARTICLE_ROOT", root)
    monkeypatch.setattr(module, "resolve_artifact_path", lambda entry_id, kind: source)

    output = module.render("0001")
    assert output.is_file()
    assert source.read_bytes() == before


def test_review_renderer_does_not_modify_review_json(
    tmp_path, monkeypatch, review_payloads
):
    root = tmp_path / "article"
    source = root / "reviews/10_each_lore/0001/Review_0001_00_001.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps(review_payloads["00"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    before = source.read_bytes()

    monkeypatch.setattr(render_review, "ARTICLE_ROOT", root)
    output = render_review.render(str(source))

    assert output.is_file()
    assert source.read_bytes() == before
