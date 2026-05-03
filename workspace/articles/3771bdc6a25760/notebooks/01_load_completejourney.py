from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


LOGGER_NAME = "load_completejourney"
DATASET_YAML = Path("shared/py/myproj/conf/dataset/completejourney.yaml")
SRC_ROOT = Path("shared/py/myproj/src")


class PlaceholderDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    search_from = current if current.is_dir() else current.parent

    for path in (search_from, *search_from.parents):
        if (path / "pyproject.toml").exists():
            return path

    raise RuntimeError(f"pyproject.toml was not found from {search_from}")


def add_project_src_to_path(project_root: Path) -> None:
    src_root = project_root / SRC_ROOT
    if not src_root.exists():
        raise FileNotFoundError(f"source root does not exist: {src_root}")

    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)


def resolve_placeholders(value: Any, placeholders: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return value.format_map(PlaceholderDict(placeholders))

    if isinstance(value, Mapping):
        return {
            key: resolve_placeholders(child, placeholders)
            for key, child in value.items()
        }

    if isinstance(value, list):
        return [resolve_placeholders(child, placeholders) for child in value]

    return value


def load_dataset_definition(path: Path, project_root: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as file:
        dataset_definition = yaml.safe_load(file) or {}

    placeholders = {
        "path_sys_base": str(project_root),
        "path_sysy_base": str(project_root),
    }
    return resolve_placeholders(dataset_definition, placeholders)


def summarize_frame(name: str, frame: Any) -> str:
    columns = getattr(frame, "columns", None)
    column_count = len(columns) if columns is not None else "-"

    try:
        row_count = f"{len(frame):,}"
    except TypeError:
        row_count = "-"

    return f"{name}: rows={row_count}, columns={column_count}"


def load_completejourney(
    *, ## 変数を入力する際の、位置引数での呼び出しを禁止するために　"*" を挟んでいる。これ以降の引数はキーワード引数でしか渡せないことを強制する
    project_root: Path,
    dataset_yaml: Path,
    entries: Sequence[str] | None,
    use_dask: bool,
) -> dict[str, Any]:
    add_project_src_to_path(project_root)

    from myproj.io.file_io import FileConfigRegistry, FileIOUtils
    from myproj.logger.custom_logger import CustomLogger

    logger = CustomLogger(
        LOGGER_NAME,
        Path(__file__).with_suffix(".log"),
    ).get_logger()
    dataset_definition = load_dataset_definition(dataset_yaml, project_root)
    registry = FileConfigRegistry.from_mapping(dataset_definition)
    selected_entries = tuple(entries) if entries else tuple(registry.entries.keys())
    file_io = FileIOUtils(logger)

    data: dict[str, Any] = {}
    for entry in selected_entries:
        config = registry.read_config(entry)
        data[entry] = file_io.read_file(config, use_dask=use_dask)

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load completejourney parquet datasets via FileIOUtils.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to the nearest parent containing pyproject.toml.",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=None,
        help="Dataset definition YAML. Defaults to shared/py/myproj/conf/dataset/completejourney.yaml.",
    )
    parser.add_argument(
        "--entries",
        nargs="*",
        default=None,
        help="Logical dataset names to load. Defaults to all entries in the YAML.",
    )
    parser.add_argument(
        "--use-dask",
        action="store_true",
        help="Load parquet files as dask dataframes instead of pandas dataframes.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else find_project_root(Path(__file__))
    )
    dataset_yaml = (
        args.dataset_yaml.resolve()
        if args.dataset_yaml is not None
        else project_root / DATASET_YAML
    )

    data = load_completejourney(
        project_root=project_root,
        dataset_yaml=dataset_yaml,
        entries=args.entries,
        use_dask=args.use_dask,
    )

    for name, frame in data.items():
        print(summarize_frame(name, frame))


if __name__ == "__main__":
    main()
