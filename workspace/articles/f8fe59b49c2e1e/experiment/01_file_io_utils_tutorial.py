from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from myproj.io.config_resolver import (
    find_project_root,
    load_dataset_definition,
    resolve_placeholders,
)
from myproj.io.file_io import (
    CsvReadOptions,
    FileConfigRegistry,
    FileIOUtils,
    FileReadConfig,
    FileSpec,
    FileWriteConfig,
    ReadOptions,
    WriteOptions,
)


DATASET_YAML = Path("shared/py/myproj/conf/dataset/completejourney/10_interim.yaml")
OUTPUT_CONFIG_YAML = Path("articles/_template/conf/01_file_io_utils_tutorial/99_output.yaml")
LOGGER_NAME = "file_io_utils_tutorial"


def build_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "ordered_at": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "sales_value": [1200.5, 900.0, 1500.25],
            "segment": ["A", "B", "A"],
        }
    )


def print_frame_summary(name: str, frame: pd.DataFrame) -> None:
    print(f"{name}: shape={frame.shape}")
    print(frame)
    print()


def tutorial_direct_dataclass_configs(file_io: FileIOUtils, work_dir: Path) -> None:
    print("1. dataclass config を直接組み立てて読み書きする")

    sample = build_sample_frame()
    csv_write_config = FileWriteConfig(
        file=FileSpec(path=work_dir, name="orders.csv", type="csv"),
        options=WriteOptions(),
    )
    csv_read_config = FileReadConfig(
        file=csv_write_config.file,
        options=ReadOptions(),
    )

    file_io.write_file(sample, csv_write_config)
    loaded = file_io.read_file(csv_read_config)
    print_frame_summary("orders.csv", loaded)

    parquet_write_config = FileWriteConfig(
        file=FileSpec(path=work_dir, name="orders.parquet", type="parquet"),
        options=WriteOptions(),
    )
    parquet_read_config = FileReadConfig(
        file=parquet_write_config.file,
        options=ReadOptions(),
    )

    file_io.write_file(sample, parquet_write_config)
    loaded = file_io.read_file(parquet_read_config)
    print_frame_summary("orders.parquet", loaded)


def tutorial_yaml_and_pickle(file_io: FileIOUtils, work_dir: Path) -> None:
    print("2. YAML / pickle を読み書きする")

    metadata: dict[str, Any] = {
        "dataset": "orders",
        "columns": ["customer_id", "ordered_at", "sales_value", "segment"],
    }
    yaml_config = FileWriteConfig(
        file=FileSpec(path=work_dir, name="metadata.yaml", type="yaml"),
        options=WriteOptions(),
    )
    file_io.write_file(metadata, yaml_config)
    loaded_yaml = file_io.read_file(
        FileReadConfig(file=yaml_config.file, options=ReadOptions())
    )
    print(f"metadata.yaml: {loaded_yaml}")

    pickle_config = FileWriteConfig(
        file=FileSpec(path=work_dir, name="metadata.pkl", type="pickle"),
        options=WriteOptions(),
    )
    file_io.write_file(metadata, pickle_config)
    loaded_pickle = file_io.read_file(
        FileReadConfig(file=pickle_config.file, options=ReadOptions())
    )
    print(f"metadata.pkl: {loaded_pickle}")
    print()


def tutorial_registry_configs(file_io: FileIOUtils, work_dir: Path) -> None:
    print("3. FileConfigRegistry で default + 個別設定をマージして読む")

    sample = build_sample_frame()
    sample.to_parquet(work_dir / "orders_from_registry.parquet")

    config_definition = {
        "default": {
            "file": {
                "path": str(work_dir),
                "name": "",
                "type": "parquet",
            }
        },
        "orders": {
            "file": {
                "name": "orders_from_registry.parquet",
            }
        },
    }
    registry = FileConfigRegistry.from_mapping(config_definition)
    loaded = file_io.read_file(registry.read_config("orders"))
    print_frame_summary("orders_from_registry.parquet", loaded)


def tutorial_file_output_with_registry(file_io: FileIOUtils, work_dir: Path) -> None:
    print("4. FileConfigRegistry.write_config() で出力設定を管理する")

    sample = build_sample_frame()
    metadata: dict[str, Any] = {
        "dataset": "orders",
        "rows": len(sample),
        "columns": sample.columns.tolist(),
    }
    project_root = find_project_root(Path(__file__))
    output_definition = load_dataset_definition(
        project_root / OUTPUT_CONFIG_YAML,
        project_root,
        extra_placeholders={"tutorial_output_dir": str(work_dir)},
    )
    registry = FileConfigRegistry.from_mapping(output_definition)
    output_dir = registry.write_config("orders_tsv").file.path

    file_io.write_file(sample, registry.write_config("orders_tsv"))
    file_io.write_file(sample, registry.write_config("orders_parquet"))
    file_io.write_file(metadata, registry.write_config("orders_metadata"))

    tsv_read_config = FileReadConfig(
        file=registry.write_config("orders_tsv").file,
        options=ReadOptions(csv=CsvReadOptions(delimiter="\t")),
    )
    loaded_tsv = file_io.read_file(tsv_read_config)
    loaded_metadata = file_io.read_file(
        FileReadConfig(
            file=registry.write_config("orders_metadata").file,
            options=ReadOptions(),
        )
    )

    print(f"output files: {sorted(path.name for path in output_dir.iterdir())}")
    print_frame_summary("orders.tsv", loaded_tsv)
    print(f"orders_metadata.yaml: {loaded_metadata}")
    print()


def tutorial_placeholder_resolution(project_root: Path) -> None:
    print("5. config_resolver で placeholder を解決する")

    resolved = resolve_placeholders(
        {
            "path": "{path_sys_base}/shared/data",
            "unknown": "{not_defined}/path",
        },
        {"path_sys_base": str(project_root)},
    )
    print(f"resolved path: {resolved['path']}")
    print(f"unknown placeholder remains: {resolved['unknown']}")

    if (project_root / DATASET_YAML).exists():
        completejourney_definition = load_dataset_definition(
            project_root / DATASET_YAML,
            project_root,
        )
        print(
            "completejourney default path: "
            f"{completejourney_definition['default']['file']['path']}"
        )
    print()


def tutorial_wildcard_read(file_io: FileIOUtils, work_dir: Path) -> None:
    print("6. wildcard を展開して複数ファイルをまとめて読む")

    for index in range(1, 4):
        frame = pd.DataFrame(
            {
                "batch": [index, index],
                "value": [index * 10, index * 10 + 1],
            }
        )
        frame.to_csv(work_dir / f"batch_{index}.csv", index=False)

    wildcard_config = FileReadConfig(
        file=FileSpec(path=work_dir, name="batch_*.csv", type="csv"),
        options=ReadOptions(),
    )
    expanded_config = file_io.expand_wildcards(wildcard_config)
    loaded = file_io.read_files(expanded_config, concat=True)

    print(f"expanded names: {expanded_config.file.names}")
    print_frame_summary("batch_*.csv", loaded)


def tutorial_dtype_conversion(file_io: FileIOUtils) -> None:
    print("7. convert_dtype で読み込み後の型を整える")

    frame = build_sample_frame()
    file_io.convert_dtype(
        frame,
        {
            "datetime64": {"ordered_at": "%Y-%m-%d"},
            "int64": ["customer_id"],
            "float64": ["sales_value"],
        },
    )

    print(frame.dtypes)
    print()


def run_tutorial(project_root: Path, work_dir: Path) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    file_io = FileIOUtils(logger)

    print(f"project_root: {project_root}")
    print(f"work_dir: {work_dir}")
    print()

    tutorial_direct_dataclass_configs(file_io, work_dir)
    tutorial_yaml_and_pickle(file_io, work_dir)
    tutorial_registry_configs(file_io, work_dir)
    tutorial_file_output_with_registry(file_io, work_dir)
    tutorial_placeholder_resolution(project_root)
    tutorial_wildcard_read(file_io, work_dir)
    tutorial_dtype_conversion(file_io)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a self-contained FileIOUtils tutorial.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to the nearest parent containing pyproject.toml.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for tutorial output files. Defaults to a temporary directory.",
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

    if args.work_dir is not None:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        run_tutorial(project_root, work_dir)
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        run_tutorial(project_root, Path(temp_dir))


if __name__ == "__main__":
    main()
