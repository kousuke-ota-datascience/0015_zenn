from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd


def load_script_module(project_root: Path) -> ModuleType:
    module_path = (
        project_root
        / "articles/3771bdc6a25760/experiment/01_load_completejourney.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_load_completejourney_script",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_uses_packaged_project_root_resolver(project_root: Path) -> None:
    module = load_script_module(project_root)
    start = project_root / "articles/3771bdc6a25760/notebooks"

    assert module.find_project_root(start) == project_root


def test_load_completejourney_loads_selected_parquet(
    project_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_script_module(project_root)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    expected = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    expected.to_parquet(data_dir / "sample.parquet")

    dataset_yaml = tmp_path / "completejourney.yaml"
    dataset_yaml.write_text(
        f"""
default:
  file:
    path: "{data_dir}"
    name: ""
    type: "parquet"
sample:
  file:
    name: "sample.parquet"
""",
        encoding="utf-8",
    )

    class FakeCustomLogger:
        def __init__(self, *args, **kwargs):
            self._logger = logging.getLogger("test_load_completejourney")

        def get_logger(self):
            return self._logger

    monkeypatch.setattr(module, "CustomLogger", FakeCustomLogger)

    loaded = module.load_completejourney(
        project_root=project_root,
        dataset_yaml=dataset_yaml,
        entries=("sample",),
        use_dask=False,
    )

    pd.testing.assert_frame_equal(loaded["sample"], expected)
