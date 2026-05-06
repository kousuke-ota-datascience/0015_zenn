from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from myproj.io.config_resolver import find_project_root, load_dataset_definition
from myproj.io.file_io import FileConfigRegistry, FileIOUtils
from myproj.logger.custom_logger import CustomLogger


LOGGER_NAME = "load_completejourney"
DATASET_YAML = Path("shared/py/myproj/conf/dataset/completejourney/10_interim.yaml")


def summarize_frame(name: str, frame: Any) -> str:
    columns = getattr(frame, "columns", None)
    column_count = len(columns) if columns is not None else "-"

    try:
        row_count = f"{len(frame):,}"
    except TypeError:
        row_count = "-"

    return f"{name}: rows={row_count}, columns={column_count}"


def load_completejourney(
    *,  # 以降はキーワード専用引数
    project_root: Path,
    dataset_yaml: Path,
    entries: Sequence[str] | None,
    use_dask: bool,
) -> dict[str, Any]:
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
        help=(
            "Dataset definition YAML. Defaults to "
            "shared/py/myproj/conf/dataset/completejourney/10_interim.yaml."
        ),
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
