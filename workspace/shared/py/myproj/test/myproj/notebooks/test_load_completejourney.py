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
        / "articles/3771bdc6a25760/notebooks/01_load_completejourney.py"
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


def test_resolve_placeholders_preserves_unknown_keys(project_root: Path) -> None:
    module = load_script_module(project_root)

    resolved = module.resolve_placeholders(
        {
            "path": "{path_sys_base}/shared/data",
            "typo_path": "{path_sysy_base}/shared/data",
            "unknown": "{not_defined}/x",
            "nested": ["{path_sys_base}/a"],
        },
        {
            "path_sys_base": str(project_root),
            "path_sysy_base": str(project_root),
        },
    )

    assert resolved["path"] == f"{project_root}/shared/data"
    assert resolved["typo_path"] == f"{project_root}/shared/data"
    assert resolved["unknown"] == "{not_defined}/x"
    assert resolved["nested"] == [f"{project_root}/a"]


def test_load_dataset_definition_resolves_project_root_placeholder(
    project_root: Path,
    tmp_path: Path,
) -> None:
    module = load_script_module(project_root)
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        """
default:
  file:
    path: "{path_sys_base}/shared/data"
    name: ""
    type: "parquet"
sample:
  file:
    name: "sample.parquet"
""",
        encoding="utf-8",
    )

    data = module.load_dataset_definition(dataset_yaml, project_root)

    assert data["default"]["file"]["path"] == f"{project_root}/shared/data"


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

    import myproj.logger.custom_logger as custom_logger

    class FakeCustomLogger:
        def __init__(self, *args, **kwargs):
            self._logger = logging.getLogger("test_load_completejourney")

        def get_logger(self):
            return self._logger

    monkeypatch.setattr(custom_logger, "CustomLogger", FakeCustomLogger)

    loaded = module.load_completejourney(
        project_root=project_root,
        dataset_yaml=dataset_yaml,
        entries=("sample",),
        use_dask=False,
    )

    pd.testing.assert_frame_equal(loaded["sample"], expected)
